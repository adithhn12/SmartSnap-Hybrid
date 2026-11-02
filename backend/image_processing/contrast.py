"""
Contrast Estimation Module for SmartSnap.
Uses Root Mean Square (RMS) luminance contrast and interdecile dynamic range.
Calibrated to prevent premature 100% saturation on ordinary photographs.
"""

from typing import Dict, Any
import cv2
import numpy as np


def estimate_contrast(image_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Computes contrast using:
    1. RMS contrast: sigma_I / 127.5 * 100 (normalized against 8-bit theoretical maximum standard deviation).
    2. Interdecile dynamic range spread: (P90 - P10) / 255 * 100.
    """
    if len(image_bgr.shape) == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr.copy()

    std_dev = float(np.std(gray))
    p10 = float(np.percentile(gray, 10))
    p90 = float(np.percentile(gray, 90))
    p5 = float(np.percentile(gray, 5))
    p95 = float(np.percentile(gray, 95))
    interdecile_spread = p90 - p10  # 0 to 255

    # 1. Normalized RMS contrast (max theoretical std_dev is 127.5 for binary {0, 255})
    rms_contrast = (std_dev / 127.5) * 100.0

    # 2. Normalized dynamic range spread
    range_contrast = (interdecile_spread / 255.0) * 100.0

    # 3. Combined photographic contrast metric
    combined = 0.60 * rms_contrast + 0.40 * range_contrast
    contrast_score = float(np.round(np.clip(combined, 0.0, 100.0), 2))

    if contrast_score < 30.0:
        desc = "Low Contrast (flat tonal distribution, washed out or hazy appearance)"
    elif contrast_score <= 68.0:
        desc = "Medium / Balanced Contrast (natural dynamic spread, pleasing gradations)"
    else:
        desc = "High Contrast (punchy dynamic range, deep shadows and crisp highlights)"

    return {
        "contrast": contrast_score,
        "rms_contrast": float(np.round(rms_contrast, 2)),
        "std_dev": float(np.round(std_dev, 2)),
        "dynamic_range_p10_p90": float(np.round(interdecile_spread, 2)),
        "dynamic_range_p5_p95": float(np.round(p95 - p5, 2)),
        "method": "Normalized RMS contrast combined with interdecile dynamic range spread",
        "description": desc,
        "explanation": f"Luminance standard deviation is {std_dev:.1f} (RMS contrast: {rms_contrast:.1f}%), 10th-90th spread: {interdecile_spread:.1f}. Combined contrast score: {contrast_score:.1f}%.",
    }
