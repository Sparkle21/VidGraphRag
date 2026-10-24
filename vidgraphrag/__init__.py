"""
=============================================================================
  VidGraphRAG (Fusion - Active Development)
=============================================================================

  Shot-Segment 双层架构的多模态视频 RAG 系统，基于 VideoRAG (HKUDS) 改造。
  Multi-modal video RAG with Shot-Segment dual-layer fusion architecture,
  derived from VideoRAG (HKUDS) and extended with:
    - Shot-level scene-change detection
    - Three-channel feature encoding (visual / audio / VLM)
    - Dual-layer KV + multi-modal vector DB
    - Hybrid query routing (Shot recall + Segment graph reasoning)

  Baseline reference: see `Vimo-desktop/python_backend/videorag/`
                      (frozen as `videorag` package, baseline-frozen-v1.0).

  Status: Active development. APIs may change.
=============================================================================
"""

__version__ = "fusion-dev-0.1.0"
__status__ = "active-development"
__doc_role__ = "VidGraphRAG fusion (Shot-Segment dual-layer) - active dev."

from .videorag import VideoRAG, QueryParam
from ._wiki import WikiConfig, WikiScale

# 保留原类名 VideoRAG 用于过渡期兼容，后续会重命名为 VidGraphRAG
VidGraphRAG = VideoRAG

__all__ = [
    "VideoRAG",
    "VidGraphRAG",
    "QueryParam",
    "WikiConfig",
    "WikiScale",
    "__version__",
    "__status__",
]