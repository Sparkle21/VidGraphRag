import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger("vidgraphrag")


def assess_shot_quality(
    keyframes: List[np.ndarray],
    duration: float,
    brightness_lo: float = 10.0,
    brightness_hi: float = 245.0,
    static_threshold: float = 3.0,
    min_duration: float = 0.3,
) -> Dict:
    """
    评估 Shot 的信息量，用于在 VLM 调用前过滤低价值片段。

    Returns:
        {
            'is_low_info': bool,
            'reason': str,  # 'black_frame' / 'white_frame' / 'static' / 'too_short' / 'ok'
            'brightness_mean': float,
            'frame_diff_mean': float,
        }
    """
    result = {
        "is_low_info": False,
        "reason": "ok",
        "brightness_mean": 128.0,
        "frame_diff_mean": 50.0,
    }

    if duration < min_duration:
        result["is_low_info"] = True
        result["reason"] = "too_short"
        return result

    if not keyframes:
        result["is_low_info"] = True
        result["reason"] = "no_frames"
        return result

    brightness_values = [float(np.mean(f)) for f in keyframes]
    result["brightness_mean"] = float(np.mean(brightness_values))

    if result["brightness_mean"] < brightness_lo:
        result["is_low_info"] = True
        result["reason"] = "black_frame"
        return result

    if result["brightness_mean"] > brightness_hi:
        result["is_low_info"] = True
        result["reason"] = "white_frame"
        return result

    if len(keyframes) >= 2:
        diffs = []
        for i in range(len(keyframes) - 1):
            a = keyframes[i].astype(np.float32)
            b = keyframes[i + 1].astype(np.float32)
            diffs.append(float(np.mean(np.abs(a - b))))
        result["frame_diff_mean"] = float(np.mean(diffs))
        if result["frame_diff_mean"] < static_threshold:
            result["is_low_info"] = True
            result["reason"] = "static"
            return result

    return result
