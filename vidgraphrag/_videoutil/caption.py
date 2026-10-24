import os
import re
import json
import torch
import base64
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Optional
from io import BytesIO
from transformers import AutoModel, AutoTokenizer
from moviepy.video.io.VideoFileClip import VideoFileClip
# 视频片段的多模态字幕生成
def encode_video(video, frame_times):
    """
        从视频中提取指定时间点的帧并进行标准化处理

        Args:
            video (VideoFileClip): MoviePy 视频文件对象
            frame_times (np.ndarray): 需要提取帧的时间点数组（秒）

        Returns:
            list[PIL.Image]: 处理后的图像帧列表，所有帧统一调整为 1280x720 分辨率

        Note:
            - 按顺序提取每一帧
            - 将 numpy 数组转换为 PIL Image 格式
            - 使用双线性插值 resize 到固定尺寸以适配模型输入
    """
    frames = []
    for t in frame_times:
        frames.append(video.get_frame(t))
    frames = np.stack(frames, axis=0)
    # 将 numpy 数组转换为 PIL Image 格式
    frames = [Image.fromarray(v.astype('uint8')).resize((1280, 720)) for v in frames]
    return frames


# ==================== Shot 级 VLM 分析（阿里云百炼）====================

async def shot_vlm_analysis_bailian(
    keyframes: List[np.ndarray],
    asr_context: str = "",
    api_key: str = None,
    model: str = "qwen3-vl-8b-instruct"
) -> Dict:
    """
    使用阿里云百炼 Qwen-VL 对单个镜头进行结构化 VLM 分析
    
    Args:
        keyframes: 关键帧列表（numpy 数组）
        asr_context: ASR 语音识别文本上下文
        api_key: 阿里云 API Key（从环境变量 DASHSCOPE_API_KEY 读取）
        model: 模型名称，默认 qwen3-vl-8b-instruct
    
    Returns:
        结构化字典，包含 subject/action/environment/ocr/caption
    """
    import httpx
    
    api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量或传入 api_key 参数")
    
    try:
        # 转换关键帧为 base64
        image_contents = []
        for i, frame in enumerate(keyframes[:5]):  # 最多 5 帧，避免 token 超限
            if isinstance(frame, np.ndarray):
                img = Image.fromarray(frame.astype('uint8'))
            else:
                img = frame
            
            # 转为 base64
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_contents.append(img_base64)
        
        print(f"[DEBUG] Processed {len(image_contents)} frames")
        
        # 构建 prompt
        asr_info = f"\nASR Transcript:\n{asr_context}" if asr_context else ""
        text_prompt = f"""Please analyze this video shot in a structured way.

{asr_info}

Provide your analysis in JSON format with the following fields:
- subject: Who or what are the main subjects? (people, objects)
- action: What actions are happening?
- environment: What is the scene/environment?
- ocr: What visible text appears in the frame? (if any)
- caption: A comprehensive natural language description combining all information

Output ONLY valid JSON, no other text."""
        
        # 构建消息内容（百炼 API 格式）
        content = []
        # 先添加文本
        content.append({"text": text_prompt})
        # 再添加图片
        for img_b64 in image_contents:
            content.append({"image": f"data:image/jpeg;base64,{img_b64}"})
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        # 调用阿里云百炼 API
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "input": {
                        "messages": messages
                    },
                    "parameters": {
                        "result_format": "message"
                    }
                }
            )
            
            # 如果出错，打印详细信息
            if response.status_code != 200:
                print(f"\n[ERROR] HTTP {response.status_code}")
                print(f"[ERROR] Response:\n{response.text}\n")
            
            response.raise_for_status()
            result = response.json()
            
            # 提取回复内容
            output_text = result["output"]["choices"][0]["message"]["content"][0]["text"]
        
        # 解析 JSON 响应
        parsed_result = _parse_structured_response(output_text)
        return parsed_result
        
    except Exception as e:
        print(f"Warning: shot_vlm_analysis_bailian failed: {e}")
        return {
            'subject': [],
            'action': [],
            'environment': [],
            'ocr': [],
            'caption': f"Error analyzing shot: {str(e)}"
        }


# 保持原有接口兼容（同步版本，内部调用异步）
def shot_vlm_analysis(
    model_or_key,  # 兼容旧接口：可以是 model 对象或 api_key
    tokenizer=None,  # 兼容旧接口
    shot_path: str = "",
    keyframes: List[np.ndarray] = None,
    asr_context: str = ""
) -> Dict:
    """
    兼容接口的 Shot 级 VLM 分析
    - 如果传入的是字符串（API Key），使用阿里云百炼
    - 如果传入的是模型对象，使用本地 MiniCPM-V（原有逻辑）
    """
    import asyncio
    
    # 判断是 API Key 还是模型对象
    if isinstance(model_or_key, str):
        # 阿里云百炼模式
        api_key = model_or_key
        
        # 检查是否已有运行中的事件循环
        try:
            loop = asyncio.get_running_loop()
            # 已有循环，使用 create_task + run_until_complete 会冲突
            # 改用 nest_asyncio 或直接 await（如果在 async 上下文中）
            import nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # 没有运行中的循环，创建新的
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                shot_vlm_analysis_bailian(
                    keyframes=keyframes or [],
                    asr_context=asr_context,
                    api_key=api_key
                )
            )
            return result
        finally:
            # 只在是自己创建的循环时才关闭
            if not hasattr(asyncio, '_nest_patched'):
                loop.close()
    else:
        # 本地模型模式（原有逻辑，需要实现）
        print("Warning: 本地模型模式暂未实现，请使用阿里云百炼 API")
        return {
            'subject': [],
            'action': [],
            'environment': [],
            'ocr': [],
            'caption': "Local model mode not implemented yet."
        }


def _parse_structured_response(response: str) -> Dict:
    """
    解析 VLM 的结构化响应，提取 JSON 格式
    
    Args:
        response: 模型原始输出
    
    Returns:
        解析后的字典
    """
    try:
        # 尝试直接解析 JSON
        cleaned = response.strip()
        
        # 如果响应包含 ```json ... ``` 标记，提取内容
        json_match = re.search(r'```json\s*(.*?)\s*```', cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(1)
        
        # 尝试找到第一个 { 和最后一个 }
        start_idx = cleaned.find('{')
        end_idx = cleaned.rfind('}')
        if start_idx != -1 and end_idx != -1:
            cleaned = cleaned[start_idx:end_idx+1]
        
        result = json.loads(cleaned)
        
        # 验证必需字段
        required_fields = ['subject', 'action', 'environment', 'ocr', 'caption']
        for field in required_fields:
            if field not in result:
                result[field] = [] if field != 'caption' else "No description available"
        
        # 确保列表字段是列表类型
        for field in ['subject', 'action', 'environment', 'ocr']:
            if not isinstance(result[field], list):
                result[field] = [str(result[field])] if result[field] else []
        
        # 确保 caption 是字符串
        if not isinstance(result['caption'], str):
            result['caption'] = str(result['caption'])
        
        return result
        
    except Exception as e:
        print(f"Warning: Failed to parse structured response: {e}")
        # 降级：将整个响应作为 caption
        return {
            'subject': [],
            'action': [],
            'environment': [],
            'ocr': [],
            'caption': response.strip()[:500]  # 限制长度
        }


async def segment_narrative_understanding_bailian(
    shot_summaries_list: List[Dict],
    aggregated_asr: str = "",
    api_key: str = None,
    model: str = "qwen3-8b"
) -> str:
    """
    使用阿里云百炼基于多个镜头的信息生成整体叙事理解
    
    Args:
        shot_summaries_list: 多个 Shot 的 summary 列表
        aggregated_asr: 聚合的 ASR 文本
        api_key: 阿里云 API Key
        model: 模型名称，默认 qwen3-8b
    
    Returns:
        连贯的叙事段落（200-300 字）
    """
    import httpx
    
    api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量或传入 api_key 参数")
    
    try:
        # 构建镜头描述文本
        shot_descriptions = []
        for i, summary in enumerate(shot_summaries_list, 1):
            caption = summary.get('caption', '')
            subjects = ', '.join(summary.get('subject', []))
            actions = ', '.join(summary.get('action', []))
            desc = f"Shot {i}: {caption}"
            if subjects:
                desc += f" Main subjects: {subjects}."
            if actions:
                desc += f" Actions: {actions}."
            shot_descriptions.append(desc)
        
        shots_text = "\n".join(shot_descriptions)
        asr_info = f"\nAggregated ASR Transcript:\n{aggregated_asr}" if aggregated_asr else ""
        
        prompt = f"""You are given descriptions of multiple consecutive video shots from the same segment:

{shots_text}
{asr_info}

Please understand and summarize the overall narrative by answering:
1. What story do these shots tell together?
2. What are the relationships between characters/entities?
3. What is the cause-and-effect flow of events?
4. What is the overall theme or emotional tone?

Provide a coherent narrative paragraph (200-300 words) that captures the essence of this segment."""
        
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # 调用阿里云百炼 OpenAI 兼容接口
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": messages,
                    "enable_thinking": False
                }
            )
            response.raise_for_status()
            result = response.json()

            narrative = result["choices"][0]["message"]["content"]
        
        # 清理输出
        narrative = narrative.replace("\n", " ").replace("$", "").strip()
        
        # 限制长度
        if len(narrative) > 3000:
            narrative = narrative[:3000] + "..."

        return narrative

    except Exception as e:
        print(f"Warning: segment_narrative_understanding_bailian failed: {e}")
        fallback = " ".join([s.get('caption', '') for s in shot_summaries_list])
        return fallback[:1500] if fallback else "Unable to generate narrative understanding."


def segment_narrative_understanding(
    model_or_key,
    tokenizer=None,
    shot_summaries_list: List[Dict] = None,
    aggregated_asr: str = ""
) -> str:
    """
    兼容接口的 Segment 级叙事理解
    """
    import asyncio
    
    if isinstance(model_or_key, str):
        api_key = model_or_key
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                segment_narrative_understanding_bailian(
                    shot_summaries_list=shot_summaries_list or [],
                    aggregated_asr=aggregated_asr,
                    api_key=api_key
                )
            )
            return result
        finally:
            loop.close()
    else:
        print("Warning: 本地模型模式暂未实现，请使用阿里云百炼 API")
        return "Local model mode not implemented yet."


def shot_segment_caption(
    model,
    tokenizer,
    video_name: str,
    video_path: str,
    segment_index2name: dict,
    transcripts: dict,
    segment_times_info: dict,
    shots_data: dict,
    caption_result: dict,
    error_queue
):
    """
    改进的 Segment 级 caption 生成：先分析每个 Shot，再生成整体叙事
    
    Args:
        model: MiniCPM-V 模型实例
        tokenizer: 对应的分词器
        video_name: 视频名称标识
        video_path: 视频文件的完整路径
        segment_index2name: 片段索引到名称的映射字典
        transcripts: 各片段的语音识别文本结果 {index: transcript}
        segment_times_info: 各片段的时间信息
        shots_data: Shot 级数据 {segment_index: [{shot_id, start, end, keyframes}, ...]}
        caption_result: 用于存储生成的 caption 结果的字典（传引用修改）
        error_queue: 进程间错误通信队列
    
    Note:
        - 对每个 Segment 内的 Shots 进行结构化分析
        - 基于 Shot summaries 生成整体叙事理解
        - 保留向后兼容：如果 shots_data 为空，回退到原有逻辑
    """
    try:
        with VideoFileClip(video_path) as video:
            for index in tqdm(segment_index2name, desc=f"Captioning Video {video_name}"):
                # 检查是否有 Shot 级数据
                if shots_data and index in shots_data and len(shots_data[index]) > 0:
                    # 新流程：Shot 级分析 + Segment 叙事
                    shot_summaries = []
                    
                    for shot_info in shots_data[index]:
                        # 对每个 Shot 进行结构化分析
                        shot_summary = shot_vlm_analysis(
                            model=model,
                            tokenizer=tokenizer,
                            shot_path=shot_info.get('shot_path', ''),
                            keyframes=shot_info['keyframes'],
                            asr_context=transcripts.get(index, '')
                        )
                        shot_summaries.append(shot_summary)
                    
                    # 生成 Segment 级叙事理解
                    aggregated_asr = transcripts.get(index, '')
                    narrative = segment_narrative_understanding(
                        model=model,
                        tokenizer=tokenizer,
                        shot_summaries_list=shot_summaries,
                        aggregated_asr=aggregated_asr
                    )
                    
                    # 存储结果
                    caption_result[index] = {
                        'type': 'narrative',
                        'content': narrative,
                        'shot_summaries': shot_summaries,
                        'transcript': aggregated_asr
                    }
                else:
                    # 旧流程：直接对 Segment 生成 caption（向后兼容）
                    frame_times = segment_times_info[index]["frame_times"]
                    video_frames = encode_video(video, frame_times)
                    segment_transcript = transcripts[index]
                    
                    query = f"The transcript of the current video:\n{segment_transcript}.\nNow provide a description (caption) of the video in English."
                    msgs = [{'role': 'user', 'content': video_frames + [query]}]
                    params = {
                        "use_image_id": False,
                        "max_slice_nums": 2
                    }
                    
                    segment_caption_text = model.chat(
                        image=None,
                        msgs=msgs,
                        tokenizer=tokenizer,
                        **params
                    )
                    
                    caption_result[index] = {
                        'type': 'legacy',
                        'content': segment_caption_text.replace("\n", "").replace("$", ""),
                        'transcript': segment_transcript
                    }
                
                torch.cuda.empty_cache() # 清理显存
                
    except Exception as e:
        error_queue.put(f"Error in shot_segment_caption:\n {str(e)}")
        raise RuntimeError

def segment_caption(video_name, video_path, segment_index2name, transcripts, segment_times_info, caption_result, error_queue):
    """
        使用 MiniCPM-V 多模态大模型为视频片段生成英文描述（caption）

        Args:
            video_name (str): 视频名称标识
            video_path (str): 视频文件的完整路径
            segment_index2name (dict): 片段索引到名称的映射字典
            transcripts (dict): 各片段的语音识别文本结果 {index: transcript}
            segment_times_info (dict): 各片段的时间信息，包含 frame_times 字段
            caption_result (dict): 用于存储生成的 caption 结果的字典（传引用修改）
            error_queue (multiprocessing.Queue): 进程间错误通信队列

        Raises:
            RuntimeError: 当模型推理过程中出现异常时抛出

        Note:
            - 使用量化版本的 MiniCPM-V-2_6-int4 模型降低显存占用
        - 结合 ASR 转录文本和视觉帧生成综合描述
        - 每个片段独立处理，支持并行执行
        - 每处理完一个片段自动清理 GPU 缓存
        - 通过 error_queue 实现多进程错误传递
    """
    try:
        model = AutoModel.from_pretrained('./MiniCPM-V-2_6-int4', trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained('./MiniCPM-V-2_6-int4', trust_remote_code=True)
        model.eval() # 进入评估模式

        with VideoFileClip(video_path) as video: # 加载视频文件
            for index in tqdm(segment_index2name, desc=f"Captioning Video {video_name}"):
                frame_times = segment_times_info[index]["frame_times"] # 获取当前片段的时间点数组
                video_frames = encode_video(video, frame_times)
                segment_transcript = transcripts[index] # 获取当前片段的转录文本
                query = f"The transcript of the current video:\n{segment_transcript}.\nNow provide a description (caption) of the video in English."
                msgs = [{'role': 'user', 'content': video_frames + [query]}] # 构建输入消息
                params = {}
                params["use_image_id"] = False # 是否使用图像 ID
                params["max_slice_nums"] = 2 # 最大分片数量
                segment_caption = model.chat( # 调用模型进行推理
                    image=None,
                    msgs=msgs,
                    tokenizer=tokenizer,
                    **params # 传入参数
                )
                caption_result[index] = segment_caption.replace("\n", "").replace("<|endoftext|>", "")
                torch.cuda.empty_cache() # 清理 GPU 缓存
    except Exception as e:
        error_queue.put(f"Error in segment_caption:\n {str(e)}")
        raise RuntimeError

def merge_segment_information(segment_index2name, segment_times_info, transcripts, captions):
    inserting_segments = {}
    for index in segment_index2name:
        inserting_segments[index] = {"content": None, "time": None}
        segment_name = segment_index2name[index]
        inserting_segments[index]["time"] = '-'.join(segment_name.split('-')[-2:])
        inserting_segments[index]["content"] = f"Caption:\n{captions[index]}\nTranscript:\n{transcripts[index]}\n\n"
        inserting_segments[index]["transcript"] = transcripts[index]
        inserting_segments[index]["frame_times"] = segment_times_info[index]["frame_times"].tolist()
    return inserting_segments


def retrieved_segment_caption(caption_model, caption_tokenizer, refine_knowledge, retrieved_segments, video_path_db, video_segments, num_sampled_frames):
    """
        针对检索到的视频片段，根据特定查询生成详细的视觉描述和信息抽取

        Args:
            caption_model (AutoModel): 已加载的 MiniCPM-V 多模态模型实例
            caption_tokenizer (AutoTokenizer): 对应的分词器
            refine_knowledge (str): 需要抽取的特定信息或查询主题
            retrieved_segments (list): 检索到的片段标识列表，格式为 "video_name_index"
            video_path_db (BaseKVStorage): 视频路径数据库，_data 字段存储 {video_name: path}
            video_segments (BaseKVStorage): 视频片段信息数据库，_data 字段存储片段元数据
            num_sampled_frames (int): 每个片段需要采样的帧数
        Returns:
            dict: 生成的 caption 结果字典，键为片段标识，值为格式化的文本内容
                  格式：{segment_id: "Caption:\\n{caption}\\nTranscript:\\n{transcript}"}

        Note:
            - 支持动态查询驱动的 caption 生成（与 segment_caption 的区别）
            - 使用 eval() 解析时间戳字符串（需注意安全性）
            - 均匀采样帧：np.linspace(start, end, num, endpoint=False)
            - 每处理一个片段后清理 GPU 缓存以避免显存溢出
            - 适用于检索增强生成（RAG）场景中的按需 caption 生成

        Example:
            >>> retrieved_segments = ["lecture_python_0-120", "doc_nature_5-300"]
            >>> refine_knowledge = "machine learning algorithms"
            >>> results = retrieved_segment_caption(..., refine_knowledge, ...)
            >>> # 生成针对机器学习算法的详细视频描述
        """
    # model = AutoModel.from_pretrained('./MiniCPM-V-2_6-int4', trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained('./MiniCPM-V-2_6-int4', trust_remote_code=True)
    # model.eval()

    caption_result = {}
    for this_segment in tqdm(retrieved_segments, desc='Captioning Segments for Given Query'):
        video_name = '_'.join(this_segment.split('_')[:-1])
        index = this_segment.split('_')[-1] # 片段索引
        video_path = video_path_db._data[video_name] # 视频路径
        timestamp = video_segments._data[video_name][index]["time"].split('-') # 时间戳
        start, end = eval(timestamp[0]), eval(timestamp[1])
        video = VideoFileClip(video_path)
        frame_times = np.linspace(start, end, num_sampled_frames, endpoint=False)
        video_frames = encode_video(video, frame_times)
        segment_transcript = video_segments._data[video_name][index]["transcript"] # 片段转录文本
        # query = f"The transcript of the current video:\n{segment_transcript}.\nGiven a question: {query}, you have to extract relevant information from the video and transcript for answering the question."
        query = f"The transcript of the current video:\n{segment_transcript}.\nNow provide a very detailed description (caption) of the video in English and extract relevant information about: {refine_knowledge}'"
        msgs = [{'role': 'user', 'content': video_frames + [query]}]
        params = {}
        params["use_image_id"] = False
        params["max_slice_nums"] = 2
        segment_caption = caption_model.chat(
            image=None,
            msgs=msgs,
            tokenizer=caption_tokenizer,
            **params
        )
        this_caption = segment_caption.replace("\n", "").replace("<|endoftext|>", "")
        caption_result[this_segment] = f"Caption:\n{this_caption}\nTranscript:\n{segment_transcript}\n\n"
        torch.cuda.empty_cache() # 清理 GPU 缓存

    return caption_result