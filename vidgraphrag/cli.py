"""
VidGraphRAG Headless CLI
========================

无界面（"黑屏"）的命令行入口，封装核心 RAG 引擎，不依赖 Electron / 前端。

子命令：
    vidgraphrag ingest  <video...>   索引视频，构建双层存储与知识图谱
    vidgraphrag query   "问题"        文本问答；--image 支持以图搜视频
    vidgraphrag serve                 启动 HTTP 服务（需安装 [server] extra）

LLM 后端通过 --llm 选择，API Key 一律从环境变量读取（见 --help）。
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
import warnings

warnings.filterwarnings("ignore")


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")
    # 第三方库降噪
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _build_llm(name: str):
    """把 --llm 名称映射到 _llm 中的 LLMConfig 实例。"""
    from ._llm import (
        openai_config,
        openai_4o_mini_config,
        azure_openai_config,
        ollama_config,
        dashscope_config,
        deepseek_bge_config,
    )

    mapping = {
        "openai": openai_config,
        "openai-mini": openai_4o_mini_config,
        "azure": azure_openai_config,
        "ollama": ollama_config,
        "dashscope": dashscope_config,
        "deepseek": deepseek_bge_config,
    }
    if name not in mapping:
        raise SystemExit(f"Unknown --llm '{name}'. Choose one of: {', '.join(mapping)}")
    return mapping[name]


def _require_keys(llm_name: str) -> None:
    """启动前校验所选后端所需的环境变量，缺失则给出清晰提示。"""
    required = {
        "openai": ["OPENAI_API_KEY"],
        "openai-mini": ["OPENAI_API_KEY"],
        "azure": ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"],
        "ollama": [],
        "dashscope": ["DASHSCOPE_API_KEY"],
        "deepseek": ["DEEPSEEK_API_KEY", "SILICONFLOW_API_KEY"],
    }.get(llm_name, [])
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise SystemExit(
            f"error: --llm {llm_name} requires environment variable(s): {', '.join(missing)}\n"
            f"       export them before running, e.g.: export {missing[0]}=..."
        )


def _build_engine(args):
    """按公共参数构建 VideoRAG 引擎实例。"""
    from . import VideoRAG

    llm = _build_llm(args.llm)
    engine = VideoRAG(llm=llm, working_dir=args.workdir)

    # 双层 Shot 处理可通过 --legacy 关闭，退回固定时长分段
    if getattr(args, "legacy", False):
        engine.enable_shot_level = False

    # 为 DashScope 后端注入 caption / ASR 所需的 key（fusion 模式使用）
    if args.llm == "dashscope":
        engine.ali_dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")

    return engine


def cmd_ingest(args) -> int:
    _require_keys(args.llm)
    engine = _build_engine(args)
    print(f"Indexing {len(args.videos)} video(s) into '{args.workdir}' "
          f"(mode={'fusion' if engine.enable_shot_level else 'legacy'}) ...")
    engine.insert_video(video_path_list=list(args.videos))
    print("Ingest complete.")
    return 0


def cmd_query(args) -> int:
    _require_keys(args.llm)
    engine = _build_engine(args)

    # 以媒体（图/音/视频）搜视频：需 ImageBind 视觉/声学编码器
    media = next((m for m in ("image", "audio", "video") if getattr(args, m)), None)
    if media:
        _ensure_imagebind(engine)
        path = getattr(args, media)
        if media == "image":
            results = engine.query_by_image(image_path=path, top_k=args.topk)
        elif media == "audio":
            results = engine.query_by_audio(audio_path=path, top_k=args.topk)
        else:
            results = engine.query_by_video_clip(video_path=path, top_k=args.topk)
        print(f"# 以{ {'image':'图','audio':'音','video':'视频片段'}[media] }搜视频 · top {args.topk}")
        for i, shot in enumerate(results, 1):
            score = shot.get("retrieval_score", 0.0)
            content = shot.get("content", {})
            caption = content.get("vlm_shot_summary", {}).get("caption", "")
            print(f"[{i}] score={score:.4f}  {caption}")
        return 0

    from . import QueryParam

    param = QueryParam(mode="videorag")
    param.wo_reference = args.no_reference
    if args.with_imagebind:
        _ensure_imagebind(engine)

    response = engine.query(query=args.question, param=param)
    print(response)
    return 0


def _ensure_imagebind(engine) -> None:
    """惰性加载 ImageBind 并挂到引擎上，供查询侧视觉/声学通路使用。"""
    if getattr(engine, "_imagebind_model", None) is not None or getattr(engine, "visual_embedder", None):
        return
    try:
        from ._videoutil import load_imagebind_model
    except Exception as e:  # pragma: no cover - 依赖缺失时降级
        raise SystemExit(
            f"error: image/audio/video query & visual channel need the [imagebind] extra "
            f"(pip install 'vidgraphrag[imagebind]'): {e}"
        )
    print("Loading ImageBind model ...")
    engine._imagebind_model = load_imagebind_model()
    print("ImageBind loaded.")


def cmd_serve(args) -> int:
    _require_keys(args.llm)
    try:
        from ._server import run_server
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            f"error: 'serve' requires the [server] extra "
            f"(pip install 'vidgraphrag[server]'): {e}"
        )
    run_server(engine_factory=lambda: _build_engine(args), host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vidgraphrag",
        description="VidGraphRAG headless CLI — omni-modal video RAG engine without any GUI.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="-v info, -vv debug")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--workdir", default="./videorag-workdir", help="索引/存储工作目录 (default: ./videorag-workdir)")
        p.add_argument("--llm", default="openai",
                       help="LLM 后端: openai|openai-mini|azure|ollama|dashscope|deepseek (default: openai)")
        p.add_argument("--legacy", action="store_true", help="关闭 Shot 双层架构，退回固定时长分段")

    p_ing = sub.add_parser("ingest", help="索引视频")
    add_common(p_ing)
    p_ing.add_argument("videos", nargs="+", help="视频文件路径（可多个）")
    p_ing.set_defaults(func=cmd_ingest)

    p_q = sub.add_parser("query", help="文本问答 / 以图·音·视频片段搜视频")
    add_common(p_q)
    p_q.add_argument("question", nargs="?", default="", help="查询问题；使用 --image/--audio/--video 时可为空")
    p_q.add_argument("--image", help="以图搜视频：图片路径")
    p_q.add_argument("--audio", help="以音搜视频：音频文件路径（wav/mp3 等）")
    p_q.add_argument("--video", help="以视频片段搜视频：短视频文件路径")
    p_q.add_argument("--topk", type=int, default=5, help="返回条数 (default: 5)")
    p_q.add_argument("--no-reference", action="store_true", help="答案不附带视频片段引用")
    p_q.add_argument("--with-imagebind", action="store_true", help="文本查询也启用视觉通路（需加载 ImageBind）")
    p_q.set_defaults(func=cmd_query)

    p_s = sub.add_parser("serve", help="启动 HTTP 服务（需 [server] extra）")
    add_common(p_s)
    p_s.add_argument("--host", default="127.0.0.1")
    p_s.add_argument("--port", type=int, default=8000)
    p_s.set_defaults(func=cmd_serve)

    return parser


def main(argv=None) -> int:
    # 视频处理依赖 spawn 启动方式（子进程安全）
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass  # 已设置

    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
