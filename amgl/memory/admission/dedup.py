"""首版只做规范化精确去重与显式事实键比较，不冒充语义向量仲裁。"""
import hashlib
import unicodedata


def content_key(content: str, memory_type: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", content).split())
    return hashlib.sha256((memory_type + ":" + normalized).encode()).hexdigest()
