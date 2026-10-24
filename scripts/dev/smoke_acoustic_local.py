#!/usr/bin/env python3
"""
P1 真声学通道 · 纯本地端到端 smoke（零 API / 零下载）
=====================================================

目的：用真实 ImageBind 权重（本地 .checkpoints，复用桌面端那份，不联网下载）
证明 P1 的声学通路端到端可用——把一段视频切成若干镜头片，逐片抽音轨做
ImageBind AUDIO 编码成 1024 维声学向量，写入一个全新的 MultiModalVectorDB，
再用第 j 片自身的音频去检索，期望：
  - 命中 Top-1 就是第 j 片（自匹配），且
  - 返回的距离是「有区分度」的（对照旧 1536 索引被 1024 错乱解读时
    「三条全同 0.0164」的垃圾结果）。

同时顺带验证 bug #2 修复：load_imagebind_model() 从本地解析权重、CPU 自适应，
不需要在仓库根放 .checkpoints 也不会触发下载。

用法：
    /opt/homebrew/bin/python3.11 scripts/dev/smoke_acoustic_local.py
    ... --video tests/test.mp4 --clips 8 --clip-dur 4
"""

import argparse
import multiprocessing
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def _cut_clips(video_path: str, n: int, dur: float, out_dir: str):
    """从视频里均匀切 n 段（每段约 dur 秒），返回 mp4 路径列表。"""
    try:
        try:
            from moviepy.editor import VideoFileClip
        except ImportError:
            from moviepy import VideoFileClip
    except Exception as e:
        raise SystemExit(f" moviepy 不可用：{e}")

    paths = []
    with VideoFileClip(video_path) as clip:
        total = clip.duration
        if total < dur:
            n = 1
        # 均匀取样起点，保证彼此不重叠（尽量）
        step = max(dur, total / max(n, 1))
        for i in range(n):
            start = i * step
            end = min(start + dur, total)
            if start >= total - 0.1:
                break
            sub = clip.subclip(start, end)
            out = os.path.join(out_dir, f"clip_{i:02d}.mp4")
            sub.write_videofile(out, codec="libx264", audio_codec="aac",
                                verbose=False, logger=None)
            if os.path.exists(out):
                paths.append(out)
    return paths


def main():
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    ap = argparse.ArgumentParser(description="VidGraphRAG P1 声学通路本地 smoke")
    ap.add_argument("--video", default="tests/test.mp4")
    ap.add_argument("--clips", type=int, default=8, help="切多少段做互检")
    ap.add_argument("--clip-dur", type=float, default=4.0, help="每段时长秒")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        print(f" 找不到视频：{args.video}")
        return 1

    import asyncio
    import numpy as np
    from vidgraphrag._videoutil.feature import (
        load_imagebind_model, resolve_imagebind_checkpoint,
        _encode_shot_acoustic, encode_audio_query, _extract_shot_audio_wav,
    )
    from vidgraphrag._storage.vdb_nanovectordb import MultiModalVectorDB

    ckpt = resolve_imagebind_checkpoint()
    print(f" ImageBind 权重来源: {ckpt or '(未找到，将走默认逻辑)'}")
    print(" 加载 ImageBind（本地、CPU 自适应，不下载）...")
    embedder = load_imagebind_model()
    print(f" ImageBind 就绪，device={next(embedder.parameters()).device}")

    tmp = tempfile.mkdtemp(prefix="acoustic_smoke_")
    print(f" 切分 {args.clips} 段（每段≈{args.clip_dur}s）...")
    clip_paths = _cut_clips(args.video, args.clips, args.clip_dur, tmp)
    if len(clip_paths) < 2:
        print(" 可切的片段太少（视频过短或无音轨），无法做互检")
        return 1
    print(f" 得到 {len(clip_paths)} 段")

    print(" 逐段抽音轨 → ImageBind AUDIO 编码（1024 维声学向量）...")
    acoustic = _encode_shot_acoustic(clip_paths, embedder)  # (N,1024)
    print(f" 声学矩阵 shape={acoustic.shape}")
    # 零向量行数 = 抽音失败/无音轨的段
    zero_rows = int(np.sum(np.all(np.isclose(acoustic, 0.0), axis=1)))
    if zero_rows:
        print(f"   注：{zero_rows}/{len(clip_paths)} 段声学为零向量（无音轨/抽音失败）")

    shot_ids = [f"clip_{i:02d}" for i in range(len(clip_paths))]
    data = {sid: {"audio_vec": acoustic[i].astype(np.float32)}
            for i, sid in enumerate(shot_ids)}

    print(" 写入全新 MultiModalVectorDB（audio_vec_dim=1024）并逐段自检索...")
    loop = asyncio.new_event_loop()
    vdb = MultiModalVectorDB(
        namespace="shot_vectors",
        global_config={"working_dir": tmp, "audio_vec_dim": 1024},
        embedding_func=None,
    )
    loop.run_until_complete(vdb.upsert(data))

    self_hit = 0
    valid = 0
    gaps = []
    for i, path in enumerate(clip_paths):
        if np.all(np.isclose(acoustic[i], 0.0)):
            continue  # 跳过零向量段（无音轨）
        qwav = _extract_shot_audio_wav(path, os.path.join(tmp, f"q_{i:02d}.wav"))
        if not qwav:
            continue
        valid += 1
        q = encode_audio_query(qwav, embedder).astype(np.float32)
        res = loop.run_until_complete(vdb.search_audio(q, top_k=3))
        top_id, _ = res[0]
        is_self = (top_id == shot_ids[i])
        self_hit += int(is_self)
        gaps.append(1.0 - res[1][1] if len(res) > 1 else 0.0)  # 第2名相似度，越低区分越好
        dists = [round(1.0 - s, 4) for _, s in res]
        print(f"   [{shot_ids[i]}] top3={[(sid, round(s,3)) for sid,s in res]}  "
              f"self={is_self}")

    if valid == 0:
        print("\n 所有片段声学均为零向量——该视频可能无有效音轨，声学互检无法进行。")
        return 2

    uniq = len(set(gaps))
    print("\n=== 结论 ===")
    print(f"  有效自检索: {valid} 段")
    print(f"  自匹配 Top-1 命中率: {self_hit}/{valid}")
    print(f"  距离区分度(非全同): 第2名相似度集合大小={uniq}"
          f"{'（结果有区分，非乱码）' if uniq >= max(2, valid - 1) else '（ 偏均匀，疑似乱码/音轨过相似）'}")
    if self_hit == valid and uniq >= max(2, valid - 1):
        print("   P1 声学通路端到端可用（真实 ImageBind 权重 + 1024 索引 + search_audio）")
        return 0
    print("   结果未达预期，请检查音轨内容是否过短/过相似")
    return 2


if __name__ == "__main__":
    sys.exit(main())
