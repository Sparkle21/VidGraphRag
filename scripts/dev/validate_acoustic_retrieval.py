#!/usr/bin/env python3
"""
P1 + P2 真机端到端验证：声学/视频片段检索
==========================================

目的：验证「audio_vec 已换成 ImageBind 真实声学向量（P1）」与
「查询侧 query_by_audio / query_by_video_clip（P2）」在真实视频上可用，
并给出一个直观的正确性信号——用视频某时间段的音轨/画面去检索，
命中的 Top-K 是否覆盖该时间段（说明声学与视觉跨模态对齐生效）。

依赖：
    - DASHSCOPE_API_KEY 环境变量（ASR + VLM caption + 文本 embedding）
    - [imagebind] extra（torch + imagebind，用于声学/视觉编码）
    - moviepy（从视频切出查询用的音频/视频片段）

用法：
    python3.11 scripts/dev/validate_acoustic_retrieval.py
    python3.11 scripts/dev/validate_acoustic_retrieval.py --video tests/test.mp4 --clip-start 3 --clip-dur 3
    # 已索引过、只想复测查询：加 --skip-ingest --workdir <之前的目录>
"""

import argparse
import multiprocessing
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def _cut_clip(video_path: str, start: float, dur: float, out_dir: str):
    """从视频切出 [start, start+dur] 的 wav 与 mp4 各一份，返回 (wav_path, mp4_path)。"""
    try:
        try:
            from moviepy.editor import VideoFileClip
        except ImportError:
            from moviepy import VideoFileClip
    except Exception as e:
        print(f" moviepy 不可用，无法切片：{e}")
        return None, None

    wav = os.path.join(out_dir, "query_clip.wav")
    mp4 = os.path.join(out_dir, "query_clip.mp4")
    with VideoFileClip(video_path) as clip:
        end = min(start + dur, clip.duration)
        sub = clip.subclip(start, end)
        if sub.audio is not None:
            sub.audio.write_audiofile(wav, verbose=False, logger=None)
        sub.write_videofile(mp4, codec="libx264", audio_codec="aac", verbose=False, logger=None)
    return (wav if os.path.exists(wav) else None), (mp4 if os.path.exists(mp4) else None)


def _shot_time_range(shot):
    """从 shot 取 (start, end) 秒；取不到返回 None。
    实测 shot 结构用顶层 time_start/time_end（见 videorag.py _process_single_shot）。"""
    st = shot.get("time_start")
    en = shot.get("time_end")
    if st is None or en is None:
        # 兼容性兜底：回退到若干常见键
        content = shot.get("content", shot)
        st = st if st is not None else content.get("start")
        en = en if en is not None else content.get("end")
    if st is not None and en is not None:
        try:
            return float(st), float(en)
        except Exception:
            return None
    return None


def _overlap(start, end, rng):
    if rng is None:
        return None
    s, e = rng
    return max(0.0, min(end, e) - max(start, s))


def _report(title, results, clip_start, clip_dur):
    print(f"\n=== {title} ===")
    if not results:
        print("  （无命中）")
        return
    hit_covered = False
    for i, shot in enumerate(results, 1):
        score = shot.get("retrieval_score", 0.0)
        sid = shot.get("shot_id", f"#{i}")
        rng = _shot_time_range(shot)
        cov = _overlap(clip_start, clip_start + clip_dur, rng)
        cov_str = f"overlap={cov:.2f}s" if cov is not None else "time=n/a"
        if cov:
            hit_covered = True
        print(f"  [{i}] {sid}  score={score:.4f}  {cov_str}")
    verdict = " 命中覆盖查询片段（跨模态对齐生效）" if hit_covered \
        else " 未命中对应时间段（可能 shot 过少/片段过短，或纯文本无音轨）"
    print(f"  → {verdict}")


def main():
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    ap = argparse.ArgumentParser(description="VidGraphRAG 声学/视频片段检索端到端验证")
    ap.add_argument("--video", default="tests/test.mp4", help="待索引视频")
    ap.add_argument("--workdir", default=None, help="复用已有索引目录；缺省用临时目录并执行索引")
    ap.add_argument("--llm", default="dashscope", help="LLM 后端（默认 dashscope）")
    ap.add_argument("--clip-start", type=float, default=2.0, help="查询片段起始秒")
    ap.add_argument("--clip-dur", type=float, default=3.0, help="查询片段时长秒")
    ap.add_argument("--topk", type=int, default=3, help="返回 shot 数")
    ap.add_argument("--skip-ingest", action="store_true", help="跳过索引（需配合 --workdir）")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        print(f" 找不到视频：{args.video}")
        return 1

    from vidgraphrag._llm import dashscope_config, openai_config
    llm = {"dashscope": dashscope_config, "openai": openai_config}.get(args.llm)
    if llm is None:
        print(f" 脚本内置映射不含 --llm={args.llm}（可选 dashscope/openai）")
        return 1
    if args.llm == "dashscope" and not os.getenv("DASHSCOPE_API_KEY"):
        print(" 请先 export DASHSCOPE_API_KEY=...（ASR/VLM/embedding 依赖）")
        return 1

    from vidgraphrag import VideoRAG
    from vidgraphrag.cli import _ensure_imagebind

    workdir = args.workdir or tempfile.mkdtemp(prefix="acoustic_val_")
    engine = VideoRAG(llm=llm, working_dir=workdir)
    if args.llm == "dashscope":
        engine.ali_dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")

    if not args.skip_ingest:
        print(f" 索引 {args.video} → {workdir}（fusion 模式，首次会加载 ImageBind 权重）...")
        engine.insert_video(video_path_list=[args.video])
        print(" 索引完成")

    # 媒体查询必须有 ImageBind
    _ensure_imagebind(engine)

    tmp = tempfile.mkdtemp(prefix="clip_")
    wav, mp4 = _cut_clip(args.video, args.clip_start, args.clip_dur, tmp)
    print(f"\n查询片段：start={args.clip_start}s dur={args.clip_dur}s  wav={wav}  mp4={mp4}")

    if wav:
        _report("以音搜视频 query_by_audio", engine.query_by_audio(audio_path=wav, top_k=args.topk),
                args.clip_start, args.clip_dur)
    else:
        print("\n 视频无音轨或抽音失败，跳过以音搜视频")

    if mp4:
        _report("以视频片段搜视频 query_by_video_clip",
                engine.query_by_video_clip(video_path=mp4, top_k=args.topk),
                args.clip_start, args.clip_dur)

    print(f"\n（索引目录：{workdir}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
