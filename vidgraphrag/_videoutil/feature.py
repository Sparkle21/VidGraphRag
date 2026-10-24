import os
import torch
import pickle
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ImageBindModel, ModalityType


# ==================== ImageBind 权重定位与加载 ====================

def resolve_imagebind_checkpoint() -> Optional[str]:
    """定位本地已存在的 ImageBind huge 权重，避免 ImageBind 默认按当前工作目录
    (`.checkpoints/imagebind_huge.pth`) 找不到而联网重下 4.8GB。

    优先级：环境变量 IMAGEBIND_CHECKPOINT > 当前目录 .checkpoints > 仓库内桌面端目录。
    找不到返回 None（调用方可退回 imagebind 默认行为）。
    """
    candidates = []
    env = os.getenv("IMAGEBIND_CHECKPOINT")
    if env:
        candidates.append(env)
    candidates += [
        os.path.join(".checkpoints", "imagebind_huge.pth"),
        os.path.join("Vimo-desktop", "python_backend", ".checkpoints", "imagebind_huge.pth"),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


def load_imagebind_model(device: Optional[str] = None) -> ImageBindModel:
    """构造并加载 ImageBind huge 模型，优先复用本地权重，设备自适应 CPU/GPU。

    先以 pretrained=False 建图（不触发下载），再从 resolve_imagebind_checkpoint()
    命中的路径 load_state_dict；若本地无权重则退回 imagebind 默认 pretrained=True。
    """
    model = imagebind_model.imagebind_huge(pretrained=False)
    ckpt = resolve_imagebind_checkpoint()
    if ckpt:
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    else:
        # 本地未找到：交给 imagebind 默认逻辑（.checkpoints/ 或联网下载）
        model = imagebind_model.imagebind_huge(pretrained=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model


def encode_video_segments(video_paths, embedder: ImageBindModel):
    """
        使用 ImageBind 模型将视频片段编码为统一语义空间的向量表示

        Args:
            video_paths (list[str]): 视频文件路径列表，每个路径指向一个视频片段文件
            embedder (ImageBindModel): 预训练的 ImageBind 多模态嵌入模型实例

        Returns:
            torch.Tensor: 视频特征向量数组，形状为 (num_videos, embedding_dim)
                          其中 embedding_dim 取决于 ImageBind 模型配置（通常为 1024）

        Raises:
            FileNotFoundError: 当视频文件不存在时由 data.load_and_transform_video_data 抛出
            RuntimeError: 当 GPU 显存不足或模型推理失败时抛出

        Note:
            - 使用 ImageBind 的视觉模态编码器提取视频全局特征
            - 自动检测模型所在设备（CPU/GPU）并在同一设备上执行推理
            - 使用 torch.no_grad() 禁用梯度计算以节省显存和加速推理
            - 返回前将 embeddings 转移到 CPU 避免占用 GPU 内存
            - 支持批量处理多个视频路径

        Example:
            >>> from imagebind.models import imagebind_model
            >>> embedder = imagebind_model.imagebind_huge(pretrained=True).cuda().eval()
            >>> video_paths = ["seg1.mp4", "seg2.mp4", "seg3.mp4"]
            >>> features = encode_video_segments(video_paths, embedder)
            >>> print(features.shape)  # torch.Size([3, 1024])
    """
    device = next(embedder.parameters()).device
    inputs = {
        ModalityType.VISION: data.load_and_transform_video_data(video_paths, device),
    } # 加载并转换视频数据
    with torch.no_grad():
        embeddings = embedder(inputs)[ModalityType.VISION] # 获取视频特征
    embeddings = embeddings.cpu() # 转移到 CPU，避免占用 GPU 内存
    return embeddings


# ==================== Shot 级三路特征编码 ====================

def _extract_shot_audio_wav(video_path: str, wav_path: str):
    """从镜头视频抽取音频轨为 wav；无音轨或失败时返回 None。

    优先使用 moviepy（项目已有依赖）；兼容 moviepy 1.x/2.x 的导入路径。
    """
    try:
        try:
            from moviepy.editor import VideoFileClip
        except ImportError:
            from moviepy import VideoFileClip
    except Exception as e:
        print(f"Warning: moviepy unavailable, skip acoustic for {video_path}: {e}")
        return None
    try:
        with VideoFileClip(video_path) as clip:
            if clip.audio is None:
                return None
            clip.audio.write_audiofile(wav_path, verbose=False, logger=None)
        return wav_path if os.path.exists(wav_path) else None
    except Exception as e:
        print(f"Warning: audio extraction failed for {video_path}: {e}")
        return None


def _encode_shot_acoustic(shot_paths: List[str], embedder: ImageBindModel, dim: int = 1024) -> np.ndarray:
    """用 ImageBind audio 编码器对每个镜头的真实音频波形编码，返回 (N, dim) 声学向量。

    无法抽取音频 / 编码失败的镜头以零向量占位，保证与 shot_paths 逐行对齐。
    语音"内容"不在此通路（由 BM25 基于 ASR 文本覆盖），此处只表征"怎么发声"。
    """
    import tempfile
    import shutil

    device = next(embedder.parameters()).device
    result = np.zeros((len(shot_paths), dim), dtype=np.float32)

    tmp_dir = tempfile.mkdtemp(prefix="shot_audio_")
    try:
        wav_paths = [
            _extract_shot_audio_wav(sp, os.path.join(tmp_dir, f"a_{i}.wav"))
            for i, sp in enumerate(shot_paths)
        ]

        batch_size = 4
        for start in tqdm(range(0, len(wav_paths), batch_size), desc="Acoustic encoding"):
            idxs = [j for j in range(start, min(start + batch_size, len(wav_paths))) if wav_paths[j]]
            if not idxs:
                continue
            paths = [wav_paths[j] for j in idxs]
            try:
                inputs = {
                    ModalityType.AUDIO: data.load_and_transform_audio_data(paths, device),
                }
                with torch.no_grad():
                    emb = embedder(inputs)[ModalityType.AUDIO].cpu().numpy()
                for k, j in enumerate(idxs):
                    result[j] = emb[k]
            except Exception as e:
                print(f"Warning: acoustic batch {start // batch_size} failed: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return result

def encode_shot_features(
    shot_paths: List[str],
    embedder: ImageBindModel,
    vlm_captions: List[str],
    asr_texts: List[str],
    api_key: Optional[str] = None,
    enable_acoustic: bool = True
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    编码每个 Shot 的三路特征
    
    三路特征定义：
    1. visual_vec: ImageBind vision 编码器输出的视觉向量 (1024 维)
    2. audio_vec:  ImageBind audio 编码器输出的声学向量 (1024 维)——表征"怎么发声"
                   （语气/背景音乐/环境音）；语音"内容"仍由 BM25(ASR 文本) 通道保留
    3. vlm_vec:    对 VLM caption 做 text embedding (1536 维，阿里云百炼)
    
    Args:
        shot_paths: Shot 视频文件路径列表（含音轨）
        embedder: ImageBind 模型实例
        vlm_captions: 每个 Shot 的 VLM 分析 caption 列表
        asr_texts: 每个 Shot 的 ASR 语音识别文本列表（保留入参；声学通路不再使用它）
        api_key: 阿里云百炼 API Key（可选，从环境变量读取）
        enable_acoustic: 是否启用真·声学编码；False 时 audio_vec 退化为零向量
    
    Returns:
        {
            "s_001": {
                "visual_vec": np.array([...]),  # 1024 维
                "audio_vec": np.array([...]),   # 1024 维（ImageBind 声学）
                "vlm_vec": np.array([...])      # 1536 维
            },
            ...
        }
    
    Example:
        >>> shot_paths = ["shot1.mp4", "shot2.mp4"]
        >>> vlm_captions = ["A woman opens a box", "She smiles"]
        >>> asr_texts = ["hello world", "thank you"]
        >>> features = encode_shot_features(shot_paths, embedder, vlm_captions, asr_texts)
    """
    if len(shot_paths) != len(vlm_captions) or len(shot_paths) != len(asr_texts):
        raise ValueError("shot_paths, vlm_captions, and asr_texts must have the same length")
    
    results = {}
    
    # Step 1: 批量编码视觉特征（ImageBind）
    print(f"Encoding {len(shot_paths)} shots with ImageBind...")
    device = next(embedder.parameters()).device # 获取设备
    
    # 分批处理，避免显存溢出
    batch_size = 4
    visual_embeddings = []
    for i in tqdm(range(0, len(shot_paths), batch_size), desc="Visual encoding"):
        batch_paths = shot_paths[i:i+batch_size]
        try:
            inputs = {
                # 加载视频并预处理成模型能处理的格式
                ModalityType.VISION: data.load_and_transform_video_data(batch_paths, device)
            }
            with torch.no_grad(): # 禁用梯度计算以节省显存和加速推理
                batch_embeddings = embedder(inputs)[ModalityType.VISION] # 获取视觉特征
            visual_embeddings.append(batch_embeddings.cpu()) # 转移到 CPU，避免占用 GPU 内存
        except Exception as e:
            print(f"Warning: Failed to encode batch {i//batch_size}: {e}")
            # 填充零向量
            zero_vec = torch.zeros((len(batch_paths), 1024)) # 创建零向量
            visual_embeddings.append(zero_vec) # 填充零向量
    
    visual_embeddings = torch.cat(visual_embeddings, dim=0).numpy() # 转换为 numpy 数组
    # torch.cat 张量拼接操作，把多个 tensor 沿指定维度连起来

    # Step 2: 编码文本语义特征（VLM caption → 百炼 text embedding, 1536 维）
    print("Encoding VLM caption text features with Bailian API...")
    vlm_embeddings = _batch_encode_text_bailian(vlm_captions, api_key) # 编码 VLM 语义特征

    # Step 2b: 编码真·声学特征（镜头音频 → ImageBind audio, 1024 维）
    #          语音内容仍由 BM25(ASR 文本) 通道保留，本通路只表征语气/音乐/环境音
    if enable_acoustic:
        print("Encoding acoustic features with ImageBind audio modality...")
        audio_embeddings = _encode_shot_acoustic(shot_paths, embedder)
    else:
        audio_embeddings = np.zeros((len(shot_paths), 1024))
    
    # Step 3: 组装结果
    for i, shot_path in enumerate(shot_paths):
        shot_id = os.path.basename(shot_path).replace('.mp4', '')
        if not shot_id.startswith('s_'):
            shot_id = f"s_{i+1:03d}"
        
        results[shot_id] = {
            'visual_vec': visual_embeddings[i],  # 1024 维
            'audio_vec': audio_embeddings[i],    # 1024 维（ImageBind 声学）
            'vlm_vec': vlm_embeddings[i]         # 1536 维
        }
    
    print(f"Encoded {len(results)} shots with 3-way features")
    return results


def _batch_encode_text_bailian(
    texts: List[str],
    api_key: Optional[str] = None,
    model: str = "text-embedding-v2"
) -> np.ndarray:
    """
    使用阿里云百炼批量编码文本为向量
    
    Args:
        texts: 文本列表
        api_key: API Key
        model: 模型名称，默认 text-embedding-v2
    
    Returns:
        numpy 数组，形状 (num_texts, embedding_dim)
    """
    import httpx
    
    api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Warning: DASHSCOPE_API_KEY not set, using zero vectors for text embeddings")
        return np.zeros((len(texts), 1536))
    
    embeddings = []
    
    # 批量调用 API（每次最多 25 条）
    batch_size = 25
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        try:
            response = httpx.post(
                "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "input": {
                        "texts": batch_texts
                    }
                },
                timeout=30.0
            )
            response.raise_for_status()
            result = response.json()
            
            # 提取 embeddings
            for item in result["output"]["embeddings"]:
                embeddings.append(np.array(item["embedding"]))
        
        except Exception as e:
            print(f"Warning: Failed to encode text batch {i//batch_size}: {e}")
            # 填充零向量
            for _ in batch_texts:
                embeddings.append(np.zeros(1536))
    
    return np.array(embeddings)


def aggregate_segment_features(
    segment_shots: Dict[str, List[Dict]],
    embedder: ImageBindModel,
    api_key: Optional[str] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    从 Shot 特征聚合为 Segment 特征
    
    Args:
        segment_shots: {
            "seg_001": [
                {"shot_id": "s_001", "vlm_caption": "...", "asr_text": "...", "shot_path": "..."},
                ...
            ],
            ...
        }
        embedder: ImageBind 模型
        api_key: 阿里云 API Key
    
    Returns:
        {
            "seg_001": {
                "visual_vec": np.array([...]),  # 平均池化
                "audio_vec": np.array([...]),
                "vlm_vec": np.array([...])
            },
            ...
        }
    """
    segment_features = {}
    
    for seg_id, shots in segment_shots.items():
        if not shots:
            continue
        
        # 提取各字段
        shot_paths = [shot['shot_path'] for shot in shots]
        vlm_captions = [shot.get('vlm_caption', '') for shot in shots]
        asr_texts = [shot.get('asr_text', '') for shot in shots]
        
        # 编码 Shot 级特征
        shot_features = encode_shot_features(
            shot_paths=shot_paths,
            embedder=embedder,
            vlm_captions=vlm_captions,
            asr_texts=asr_texts,
            api_key=api_key
        )
        
        # 聚合为 Segment 级特征（平均池化）
        visual_vecs = np.stack([shot_features[sid]['visual_vec'] for sid in shot_features])
        audio_vecs = np.stack([shot_features[sid]['audio_vec'] for sid in shot_features])
        vlm_vecs = np.stack([shot_features[sid]['vlm_vec'] for sid in shot_features])
        
        segment_features[seg_id] = {
            'visual_vec': np.mean(visual_vecs, axis=0),
            'audio_vec': np.mean(audio_vecs, axis=0),
            'vlm_vec': np.mean(vlm_vecs, axis=0)
        }
    
    print(f"Aggregated {len(segment_features)} segments from shots")
    return segment_features

def encode_image_query(image_path: str, embedder: ImageBindModel) -> np.ndarray:
    """
    用 ImageBind 将图片编码为 visual 向量（1024 维）。
    与 visual_vec 在同一空间，可直接用于 search_visual。
    """
    device = next(embedder.parameters()).device
    inputs = {
        ModalityType.VISION: data.load_and_transform_vision_data([image_path], device),
    }
    with torch.no_grad():
        embeddings = embedder(inputs)[ModalityType.VISION]
    return embeddings.cpu().numpy()[0]


def encode_audio_query(audio_path: str, embedder: ImageBindModel) -> np.ndarray:
    """用 ImageBind 将音频文件编码为声学向量（1024 维）。
    与入库侧 audio_vec 同空间，可直接用于 search_audio（以音搜视频）。"""
    device = next(embedder.parameters()).device
    inputs = {
        ModalityType.AUDIO: data.load_and_transform_audio_data([audio_path], device),
    }
    with torch.no_grad():
        embeddings = embedder(inputs)[ModalityType.AUDIO]
    return embeddings.cpu().numpy()[0]


def encode_video_clip_query(video_path: str, embedder: ImageBindModel) -> np.ndarray:
    """用 ImageBind 将视频片段编码为 visual 向量（1024 维）。
    与入库侧 visual_vec 同空间，可直接用于 search_visual（以视频片段搜视频）。"""
    device = next(embedder.parameters()).device
    inputs = {
        ModalityType.VISION: data.load_and_transform_video_data([video_path], device),
    }
    with torch.no_grad():
        embeddings = embedder(inputs)[ModalityType.VISION]
    return embeddings.cpu().numpy()[0]


def encode_string_query(query:str, embedder: ImageBindModel):
    """
        使用 ImageBind 模型将文本查询编码为统一语义空间的向量表示

        Args:
            query (str): 文本查询字符串，用于检索匹配的视频片段
            embedder (ImageBindModel): 预训练的 ImageBind 多模态嵌入模型实例

        Returns:
            torch.Tensor: 文本特征向量，形状为 (1, embedding_dim)
                          与视频特征在同一语义空间，可直接计算相似度

        Raises:
            RuntimeError: 当模型推理失败或输入文本过长时抛出

        Note:
            - 使用 ImageBind 的文本模态编码器提取查询语义特征
            - 输出特征与视频特征维度相同，支持跨模态相似度计算
            - 自动处理设备转移和梯度禁用
            - 适用于 RAG 系统中的查询 - 文档匹配场景

        Example:
            >>> embedder = imagebind_model.imagebind_huge(pretrained=True).cuda().eval()
            >>> query = "machine learning tutorial"
            >>> query_embedding = encode_string_query(query, embedder)
            >>> # 与视频特征计算余弦相似度进行检索
            >>> similarities = torch.nn.functional.cosine_similarity(
            ...     query_embedding, video_features, dim=1
            ... )
    """
    device = next(embedder.parameters()).device # 获取模型所在设备
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text([query], device), # 加载并转换文本数据
    }
    with torch.no_grad():
        embeddings = embedder(inputs)[ModalityType.TEXT] # 获取文本特征
    embeddings = embeddings.cpu() # 转移到 CPU，避免占用 GPU 内存
    return embeddings