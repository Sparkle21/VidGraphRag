import os
import time
import shutil
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from moviepy.video import fx as vfx
from moviepy.video.io.VideoFileClip import VideoFileClip
from .._utils import logger

try:
    from scenedetect import detect, ContentDetector
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False
    logger.warning("PySceneDetect not installed. Shot detection will be disabled.")

# 长视频的智能分段处理
def detect_shots_with_scene_change(
    video_path: str,
    threshold: float = 30.0,
    min_shot_duration: float = 0.5
) -> List[Dict]:
    """
    使用内容变化检测算法识别镜头边界
    
    Args:
        video_path (str): 视频文件路径
        threshold (float): 场景检测阈值，默认 30.0
        min_shot_duration (float): 最小镜头时长（秒），过滤过短的镜头
    
    Returns:
        List[Dict]: 按时间排序的 Shot 列表
        [
            {"shot_id": "s_001", "start": 0.0, "end": 3.24, "duration": 3.24},
            {"shot_id": "s_002", "start": 3.24, "end": 8.12, "duration": 4.88},
            ...
        ]
    
    Raises:
        ImportError: 如果 PySceneDetect 未安装
        FileNotFoundError: 如果视频文件不存在
    
    Example:
        >>> shots = detect_shots_with_scene_change("video.mp4")
        >>> print(f"Detected {len(shots)} shots")
    """
    if not SCENEDETECT_AVAILABLE:
        raise ImportError(
            "PySceneDetect is required for shot detection. "
            "Install it with: pip install scenedetect[opencv]"
        )
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    try:
        # 使用 ContentDetector 检测场景变化
        # PySceneDetect 的 ContentDetector 基于帧间差异检测：
        scene_list = detect(
            video_path,
            ContentDetector(threshold=threshold)
        )
        
        if not scene_list:
            logger.warning(f"No scenes detected in {video_path}, treating as single shot")
            with VideoFileClip(video_path) as video:
                duration = video.duration
            return [{
                "shot_id": "s_001",
                "start": 0.0,
                "end": duration,
                "duration": duration
            }]
        
        # 转换为标准格式
        shots = []
        for i, scene in enumerate(scene_list):
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            duration = end_time - start_time
            
            # 过滤过短的镜头
            if duration < min_shot_duration:
                continue
            
            shots.append({
                "shot_id": f"s_{i+1:03d}",
                "start": round(start_time, 2),
                "end": round(end_time, 2),
                "duration": round(duration, 2)
            })
        
        logger.info(f"Detected {len(shots)} shots from {video_path}")
        return shots
        
    except Exception as e:
        logger.error(f"Error detecting shots in {video_path}: {e}")
        raise


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _find_semantic_breakpoints(
    shots: List[Dict],
    shot_embeddings: Dict[str, np.ndarray],
    similarity_threshold: float = 0.65,
) -> Tuple[List[int], List[float]]:
    """
    计算相邻 Shot 相似度曲线，找局部最小值作为候选断点。

    Returns:
        breakpoints: 候选断点索引列表（在 shots[i] 之后切开）
        similarities: 长度 len(shots)-1 的相邻相似度序列
    """
    n = len(shots)
    if n < 2:
        return [], []

    similarities = []
    for i in range(n - 1):
        id_a = shots[i]["shot_id"]
        id_b = shots[i + 1]["shot_id"]
        vec_a = shot_embeddings.get(id_a)
        vec_b = shot_embeddings.get(id_b)
        if vec_a is not None and vec_b is not None:
            similarities.append(_cosine_similarity(vec_a, vec_b))
        else:
            similarities.append(1.0)  # 缺向量时不切

    breakpoints = []
    for i in range(len(similarities)):
        if similarities[i] >= similarity_threshold:
            continue
        is_local_min = True
        if i > 0 and similarities[i] >= similarities[i - 1]:
            is_local_min = False
        if i < len(similarities) - 1 and similarities[i] >= similarities[i + 1]:
            is_local_min = False
        if is_local_min:
            breakpoints.append(i)

    return breakpoints, similarities


def _build_segments_from_breaks(
    shots: List[Dict],
    break_indices: List[int],
) -> List[Dict]:
    """根据断点索引列表把 shots 切成 segments。break_indices 中的值表示在 shots[i] 之后切开。"""
    sorted_breaks = sorted(set(break_indices))
    segments = []
    prev = 0
    for brk in sorted_breaks:
        seg_shots = shots[prev:brk + 1]
        if seg_shots:
            segments.append(seg_shots)
        prev = brk + 1
    # 最后一段
    if prev < len(shots):
        segments.append(shots[prev:])

    result = []
    for idx, seg_shots in enumerate(segments):
        result.append({
            "segment_id": f"seg_{idx + 1:03d}",
            "shot_ids": [s["shot_id"] for s in seg_shots],
            "start": round(seg_shots[0]["start"], 2),
            "end": round(seg_shots[-1]["end"], 2),
            "duration": round(seg_shots[-1]["end"] - seg_shots[0]["start"], 2),
        })
    return result


def merge_shots_to_segments(
    shots: List[Dict],
    max_duration: int = 30,
    min_duration: int = 10,
    shot_embeddings: Dict[str, np.ndarray] = None,
    similarity_threshold: float = 0.65,
) -> List[Dict]:
    """
    将连续的 Shots 聚合成 Segments。

    两种模式：
    - shot_embeddings=None：纯时间贪心（向后兼容）
    - shot_embeddings 有值：全局最优切分（语义断点 + 时间约束）

    Args:
        shots: Shots 列表，按时间排序
        max_duration: 每个 Segment 的最大时长（秒）
        min_duration: 每个 Segment 的最小时长（秒），仅语义模式生效
        shot_embeddings: {shot_id: visual_vec} 字典，传入后启用语义切分
        similarity_threshold: 相似度低于此值才可能成为断点

    Returns:
        Segments 列表
    """
    if not shots:
        return []

    # ── 语义模式：全局最优切分 ──
    if shot_embeddings is not None:
        breakpoints, similarities = _find_semantic_breakpoints(
            shots, shot_embeddings, similarity_threshold
        )

        # Step 1: 用 min_duration 过滤过密的断点
        filtered = []
        prev_end_time = shots[0]["start"]
        for bp in breakpoints:
            bp_end_time = shots[bp]["end"]
            if bp_end_time - prev_end_time >= min_duration:
                filtered.append(bp)
                prev_end_time = bp_end_time

        # Step 2: 对 > max_duration 的 segment 强制切分
        final_breaks = list(filtered)
        segments_tmp = _build_segments_from_breaks(shots, final_breaks)
        extra_breaks = []
        for seg in segments_tmp:
            if seg["duration"] <= max_duration:
                continue
            # 在该 segment 内部找相似度最低的位置切一刀
            seg_shot_ids = seg["shot_ids"]
            first_idx = next(i for i, s in enumerate(shots) if s["shot_id"] == seg_shot_ids[0])
            last_idx = next(i for i, s in enumerate(shots) if s["shot_id"] == seg_shot_ids[-1])
            # 在 [first_idx, last_idx) 范围内找 similarities 的最低点
            best_bp = None
            best_sim = float("inf")
            for i in range(first_idx, last_idx):
                if i < len(similarities) and similarities[i] < best_sim:
                    # 切开后两边都不能小于 min_duration
                    left_dur = shots[i]["end"] - shots[first_idx]["start"]
                    right_dur = shots[last_idx]["end"] - shots[i + 1]["start"]
                    if left_dur >= min_duration and right_dur >= min_duration:
                        best_sim = similarities[i]
                        best_bp = i
            if best_bp is not None:
                extra_breaks.append(best_bp)

        if extra_breaks:
            final_breaks = sorted(set(final_breaks + extra_breaks))
            segments_tmp = _build_segments_from_breaks(shots, final_breaks)

        logger.info(
            f"Semantic merge: {len(shots)} shots → {len(segments_tmp)} segments "
            f"(breakpoints: {len(final_breaks)}, threshold: {similarity_threshold})"
        )
        return segments_tmp

    # ── 纯时间贪心模式（向后兼容） ──
    segments = []
    current_segment_shots = []
    current_segment_start = None
    current_segment_end = None
    segment_index = 0

    for shot in shots:
        shot_start = shot["start"]
        shot_end = shot["end"]

        if current_segment_start is None:
            current_segment_shots.append(shot["shot_id"])
            current_segment_start = shot_start
            current_segment_end = shot_end
            continue

        potential_duration = shot_end - current_segment_start

        if potential_duration <= max_duration:
            current_segment_shots.append(shot["shot_id"])
            current_segment_end = shot_end
        else:
            segment_index += 1
            segments.append({
                "segment_id": f"seg_{segment_index:03d}",
                "shot_ids": current_segment_shots.copy(),
                "start": round(current_segment_start, 2),
                "end": round(current_segment_end, 2),
                "duration": round(current_segment_end - current_segment_start, 2)
            })
            current_segment_shots = [shot["shot_id"]]
            current_segment_start = shot_start
            current_segment_end = shot_end

    if current_segment_shots:
        segment_index += 1
        segments.append({
            "segment_id": f"seg_{segment_index:03d}",
            "shot_ids": current_segment_shots,
            "start": round(current_segment_start, 2),
            "end": round(current_segment_end, 2),
            "duration": round(current_segment_end - current_segment_start, 2)
        })

    logger.info(f"Merged {len(shots)} shots into {len(segments)} segments")
    return segments


def split_video(
    video_path,
    working_dir,
    segment_length,
    num_frames_per_segment,
    audio_output_format='mp3',
    enable_shot_level: bool = False,
    shot_detection_threshold: float = 30.0,
):
    """
       将长视频切分为多个片段，并提取音频和关键帧时间信息
       
       支持两种模式：
       1. 传统模式（enable_shot_level=False）：按固定时长切分
       2. 双层模式（enable_shot_level=True）：先检测镜头，再聚合成片段

       Args:
           video_path (str): 输入视频的完整文件路径
           working_dir (str): 工作目录路径，用于存储缓存文件
           segment_length (int): 每个片段的时长（秒）
           num_frames_per_segment (int): 每个片段需要提取的关键帧数量
           audio_output_format (str, optional): 音频输出格式，默认为 'mp3'
           enable_shot_level (bool, optional): 是否启用镜头级处理，默认为 False
           shot_detection_threshold (float, optional): 镜头检测阈值，默认为 30.0

       Returns:
           Union[tuple, dict]: 
               - 传统模式：tuple(segment_index2name, segment_times_info)
               - 双层模式：dict {
                   "shots": [...],
                   "segments": [...],
                   "segment_index2name": {...},
                   "segment_times_info": {...}
                 }

       Raises:
           FileNotFoundError: 当视频文件不存在时抛出
           Exception: 当音频提取失败时记录警告但继续处理

       Note:
           - 使用唯一时间戳创建缓存目录，避免并发冲突
           - 自动清理已存在的旧缓存目录
           - 最后一段如果短于 5 秒则合并到前一段
           - 音频提取失败时不中断流程（处理无声视频）
           - 关键帧在片段内均匀采样，时间戳转换为全局时间

       Example:
           >>> # 传统模式
           >>> index2name, times_info = split_video(
           ...     video_path="lecture.mp4",
           ...     working_dir="./cache",
           ...     segment_length=60,
           ...     num_frames_per_segment=4
           ... )
           
           >>> # 双层模式
           >>> result = split_video(
           ...     video_path="lecture.mp4",
           ...     working_dir="./cache",
           ...     segment_length=30,
           ...     num_frames_per_segment=4,
           ...     enable_shot_level=True
           ... )
           >>> shots = result["shots"]
           >>> segments = result["segments"]
    """
    unique_timestamp = str(int(time.time() * 1000)) # 时间戳
    video_name = os.path.basename(video_path).split('.')[0] # 视频名称
    video_segment_cache_path = os.path.join(working_dir, '_cache', video_name)
    if os.path.exists(video_segment_cache_path): # 如果缓存目录存在
        shutil.rmtree(video_segment_cache_path) # 删除缓存目录
    os.makedirs(video_segment_cache_path, exist_ok=False) # 创建缓存目录
    
    # 双层模式：先检测镜头，再聚合成片段
    if enable_shot_level:
        logger.info("Using shot-level processing mode")
        
        # Step 1: 检测镜头
        try:
            shots = detect_shots_with_scene_change(
                video_path,
                threshold=shot_detection_threshold
            )
        except Exception as e:
            logger.warning(f"Shot detection failed: {e}. Falling back to fixed segmentation.")
            enable_shot_level = False
        
        if enable_shot_level and shots:
            # Step 2: 聚合为片段
            segments = merge_shots_to_segments(shots, max_duration=segment_length)
            
            # Step 3: 生成与传统模式兼容的数据结构
            segment_index2name = {}
            segment_times_info = {}
            
            with VideoFileClip(video_path) as video:
                for seg_idx, segment in enumerate(segments):
                    start = segment["start"]
                    end = segment["end"]
                    
                    subvideo = video.subclip(start, end)
                    subvideo_length = subvideo.duration
                    frame_times = np.linspace(
                        0, subvideo_length, num_frames_per_segment, endpoint=False
                    )
                    frame_times += start  # 转换为全局时间戳
                    
                    segment_index2name[f"{seg_idx}"] = (
                        f"{unique_timestamp}-{seg_idx}-{start}-{end}"
                    )
                    segment_times_info[f"{seg_idx}"] = {
                        "frame_times": frame_times,
                        "timestamp": (start, end)
                    }
                    
                    # save audio
                    audio_file_base_name = segment_index2name[f"{seg_idx}"]
                    audio_file = f'{audio_file_base_name}.{audio_output_format}'
                    try:
                        subaudio = subvideo.audio
                        subaudio.write_audiofile(
                            os.path.join(video_segment_cache_path, audio_file),
                            codec='mp3',
                            verbose=False,
                            logger=None
                        )
                    except Exception as e:
                        logger.warning(
                            f"Warning: Failed to extract audio for segment {seg_idx} "
                            f"({start}-{end}). Probably due to lack of audio track."
                        )
            
            # 返回双层结构
            return {
                "shots": shots,
                "segments": segments,
                "segment_index2name": segment_index2name,
                "segment_times_info": segment_times_info
            }
    
    # 传统模式：按固定时长切分
    logger.info("Using traditional fixed-length segmentation mode")
    segment_index = 0
    segment_index2name, segment_times_info = {}, {}
    with VideoFileClip(video_path) as video:
        total_video_length = int(video.duration) # 视频总时长
        start_times = list(range(0, total_video_length, segment_length)) # 切分点
        # 如果最后一段小于 5 秒，则合并到最后一段
        if len(start_times) > 1 and (total_video_length - start_times[-1]) < 5:
            start_times = start_times[:-1] # 删除最后一个切分点
        
        for start in tqdm(start_times, desc=f"Spliting Video {video_name}"): # 遍历切分点
            if start != start_times[-1]: # 如果不是最后一个切分点
                end = min(start + segment_length, total_video_length)
            else:
                end = total_video_length
            
            subvideo = video.subclip(start, end) # 截取子视频
            subvideo_length = subvideo.duration
            frame_times = np.linspace(0, subvideo_length, num_frames_per_segment, endpoint=False) # 生成关键帧时间戳
            frame_times += start # 转换为全局时间戳
            
            segment_index2name[f"{segment_index}"] = f"{unique_timestamp}-{segment_index}-{start}-{end}" # 片段索引到名称映射
            segment_times_info[f"{segment_index}"] = {"frame_times": frame_times, "timestamp": (start, end)} # 片段时间信息
            
            # save audio
            audio_file_base_name = segment_index2name[f"{segment_index}"] # 音频文件基础名称
            audio_file = f'{audio_file_base_name}.{audio_output_format}' # 音频文件名
            try:
                subaudio = subvideo.audio # 子视频的音频
                subaudio.write_audiofile(os.path.join(video_segment_cache_path, audio_file), codec='mp3', verbose=False, logger=None) # 保存音频文件
            except Exception as e:
                logger.warning(f"Warning: Failed to extract audio for video {video_name} ({start}-{end}). Probably due to lack of audio track.")

            segment_index += 1 # 片段索引加1

    return segment_index2name, segment_times_info

def saving_video_segments(
    video_name,
    video_path,
    working_dir,
    segment_index2name,
    segment_times_info,
    error_queue,
    video_output_format='mp4',
):
    """
        将切分后的视频片段保存为独立的视频文件（支持多进程并行处理）

        Args:
            video_name (str): 视频名称标识
            video_path (str): 原始视频文件的完整路径
            working_dir (str): 工作目录路径，包含缓存子目录
            segment_index2name (dict): 片段索引到名称的映射字典
            segment_times_info (dict): 片段的时间信息字典，包含 timestamp 字段
            error_queue (multiprocessing.Queue): 进程间错误通信队列
            video_output_format (str, optional): 视频输出格式，默认为 'mp4'

        Raises:
            RuntimeError: 当视频处理过程中出现异常时抛出
            Exception: 通过 error_queue 传递详细错误信息给主进程

        Note:
            - 设计用于多进程并行执行，加速视频片段保存
            - 使用 libx264 编码器保证兼容性
            - 通过 error_queue 实现跨进程错误传递
            - 与 split_video 分离以支持流水线并行

        Example:
            >>> from multiprocessing import Process, Queue
            >>> error_queue = Queue()
            >>> p = Process(target=saving_video_segments, args=(
            ...     "lecture", "./video.mp4", "./cache", index2name, times_info, error_queue
            ... ))
            >>> p.start()
            >>> p.join()
    """
    try:
        with VideoFileClip(video_path) as video: # 加载原始视频
            video_segment_cache_path = os.path.join(working_dir, '_cache', video_name)
            for index in tqdm(segment_index2name, desc=f"Saving Video Segments {video_name}"): # 遍历片段索引
                start, end = segment_times_info[index]["timestamp"][0], segment_times_info[index]["timestamp"][1] # 获取片段时间戳
                video_file = f'{segment_index2name[index]}.{video_output_format}' # 生成视频文件名
                subvideo = video.subclip(start, end) # 截取子视频
                subvideo.write_videofile(os.path.join(video_segment_cache_path, video_file), codec='libx264', verbose=False, logger=None) # 保存视频文件
    except Exception as e:
        error_queue.put(f"Error in saving_video_segments:\n {str(e)}")
        raise RuntimeError