"""
VidGraphRAG 极简 HTTP 服务（headless）
=====================================

替代原 Electron/Flask 桌面后端的轻量封装，供外部程序通过 HTTP 调用引擎。
仅在 `vidgraphrag serve` 时按需导入，fastapi/uvicorn 为可选依赖（[server] extra）。

端点：
    GET  /health                 健康检查
    POST /ingest   {videos[]}    索引视频
    POST /query    {question, topk?, image?}   问答 / 以图搜视频
"""

from __future__ import annotations

from typing import Callable, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class IngestRequest(BaseModel):
    videos: List[str]


class QueryRequest(BaseModel):
    question: str = ""
    topk: int = 5
    image: Optional[str] = None
    audio: Optional[str] = None
    video: Optional[str] = None
    no_reference: bool = False


def create_app(engine_factory: Callable[[], object]) -> FastAPI:
    """engine_factory 为无参可调用对象，返回一个已配置好的 VideoRAG 实例（惰性构建、进程内单例）。"""
    app = FastAPI(title="VidGraphRAG", version="0.1.0")
    _engine = {}

    def engine():
        if "e" not in _engine:
            _engine["e"] = engine_factory()
        return _engine["e"]

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/ingest")
    def ingest(req: IngestRequest):
        if not req.videos:
            raise HTTPException(400, "videos list is empty")
        try:
            engine().insert_video(video_path_list=req.videos)
        except Exception as e:  # pragma: no cover
            raise HTTPException(500, f"ingest failed: {e}")
        return {"status": "ok", "indexed": len(req.videos)}

    @app.post("/query")
    def query(req: QueryRequest):
        eng = engine()
        try:
            media = next((m for m in ("image", "audio", "video") if getattr(req, m)), None)
            if media:
                path = getattr(req, media)
                if media == "image":
                    results = eng.query_by_image(image_path=path, top_k=req.topk)
                elif media == "audio":
                    results = eng.query_by_audio(audio_path=path, top_k=req.topk)
                else:
                    results = eng.query_by_video_clip(video_path=path, top_k=req.topk)
                shots = [
                    {
                        "shot_id": s.get("shot_id"),
                        "score": s.get("retrieval_score", 0.0),
                        "caption": s.get("content", {}).get("vlm_shot_summary", {}).get("caption", ""),
                    }
                    for s in results
                ]
                return {"type": media, "shots": shots}

            from . import QueryParam

            param = QueryParam(mode="videorag")
            param.wo_reference = req.no_reference
            answer = eng.query(query=req.question, param=param)
            return {"type": "text", "answer": answer}
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:  # pragma: no cover
            raise HTTPException(500, f"query failed: {e}")

    return app


def run_server(engine_factory: Callable[[], object], host: str, port: int) -> None:
    import uvicorn

    app = create_app(engine_factory)
    print(f"VidGraphRAG server listening on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
