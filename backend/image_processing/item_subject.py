"""Lightweight OpenCV proposal for the primary object in item photography."""

from typing import Any, Dict

import cv2
import numpy as np


def estimate_item_subject(
    image_bgr: np.ndarray,
    max_side: int = 640,
) -> Dict[str, Any]:
    """Estimate a central foreground object without a trained detector.

    The proposed box is only used as category-aware CV evidence. If foreground
    segmentation is ambiguous, the result stays neutral and has no subject box.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Item subject estimation requires a three-channel BGR image.")

    height, width = image_bgr.shape[:2]
    if min(height, width) < 64:
        return _unavailable("Image is too small for stable foreground segmentation.")

    scale = min(1.0, max_side / float(max(height, width)))
    if scale < 1.0:
        resized = cv2.resize(
            image_bgr,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = image_bgr

    small_height, small_width = resized.shape[:2]
    x_margin = max(1, round(small_width * 0.04))
    y_margin = max(1, round(small_height * 0.04))
    rect = (
        x_margin,
        y_margin,
        small_width - 2 * x_margin,
        small_height - 2 * y_margin,
    )
    mask = np.zeros((small_height, small_width), dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        resized,
        mask,
        rect,
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_RECT,
    )
    foreground = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
    )
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(foreground)
    if component_count <= 1:
        return _unavailable("No stable foreground object boundary was found.")

    candidates = [
        (int(stats[index, cv2.CC_STAT_AREA]), index)
        for index in range(1, component_count)
        if stats[index, cv2.CC_STAT_AREA] >= small_height * small_width * 0.02
    ]
    if not candidates:
        return _unavailable("No foreground region large enough to assess was found.")

    object_area, index = max(candidates)
    x = int(stats[index, cv2.CC_STAT_LEFT])
    y = int(stats[index, cv2.CC_STAT_TOP])
    box_width = int(stats[index, cv2.CC_STAT_WIDTH])
    box_height = int(stats[index, cv2.CC_STAT_HEIGHT])
    frame_area = float(small_width * small_height)
    area_fraction = object_area / frame_area
    box_fraction = (box_width * box_height) / frame_area
    foreground_area = int(np.count_nonzero(foreground))
    component_share = object_area / max(foreground_area, 1)

    if (
        area_fraction > 0.85
        or box_fraction > 0.92
        or box_width < small_width * 0.08
        or box_height < small_height * 0.08
        or component_share < 0.30
    ):
        return _unavailable(
            "Foreground boundaries were ambiguous; object visibility was not scored."
        )

    scale_back = 1.0 / scale
    original_box = {
        "x": max(0, round(x * scale_back)),
        "y": max(0, round(y * scale_back)),
        "w": min(width, round(box_width * scale_back)),
        "h": min(height, round(box_height * scale_back)),
    }

    size_score = float(
        np.clip(100.0 - abs(area_fraction - 0.28) * 100.0, 55.0, 95.0)
    )
    border_margin = min(
        x,
        y,
        small_width - (x + box_width),
        small_height - (y + box_height),
    ) / float(min(small_width, small_height))
    framing_score = float(np.clip(border_margin / 0.08, 0.35, 1.0) * 100.0)
    visibility = float(np.round(0.65 * size_score + 0.35 * framing_score, 2))
    confidence = float(np.round(np.clip(component_share, 0.0, 1.0), 3))

    return {
        "object_visibility": visibility,
        "object_count": 1,
        "object_detected": True,
        "object_box": original_box,
        "object_area_pct": float(np.round(area_fraction * 100.0, 2)),
        "confidence": confidence,
        "method": "OpenCV GrabCut foreground proposal with central framing prior",
        "description": "Primary object region estimated",
        "explanation": (
            f"A foreground region covering {area_fraction * 100.0:.1f}% of the frame "
            f"was proposed with {confidence * 100.0:.0f}% segmentation consistency. "
            "This is a framing heuristic, not semantic object recognition."
        ),
    }


def _unavailable(explanation: str) -> Dict[str, Any]:
    return {
        "object_visibility": 50.0,
        "object_count": 0,
        "object_detected": False,
        "object_box": None,
        "object_area_pct": None,
        "confidence": 0.0,
        "method": "OpenCV foreground proposal; neutral fallback",
        "description": "Object visibility not assessed",
        "explanation": explanation,
    }
