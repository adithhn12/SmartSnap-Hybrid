"""
Brightness and Exposure Estimation Module for SmartSnap.
Performs:
1. Global perceptual luminance calculation.
2. Highlight and shadow clipping detection.
3. Subject-region luminance extraction (prevents dark backdrops from penalizing well-lit subjects).
4. Background luminance comparison.
"""

from typing import Dict, Any, List, Optional
import cv2
import numpy as np

if "." in __package__:
    from ..categories import category_key
else:
    from categories import category_key


def estimate_brightness(
    image_bgr: np.ndarray,
    faces: Optional[List[Dict[str, int]]] = None,
    category: str = "General Photography",
    subject_regions: Optional[List[Dict[str, int]]] = None,
) -> Dict[str, Any]:
    """
    Computes exposure metrics with subject-versus-background discrimination.
    Prevents low-key/night portraits with dark backgrounds from being falsely classified as underexposed.
    """
    if len(image_bgr.shape) == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr.copy()

    total_pixels = float(gray.size)
    global_luminance = float(np.mean(gray))

    # Clipping statistics
    shadow_clipped_pct = float(np.sum(gray < 10) / total_pixels * 100.0)
    highlight_clipped_pct = float(np.sum(gray > 245) / total_pixels * 100.0)

    # Subject and Background Luminance
    subject_luminance: Optional[float] = None
    background_luminance: Optional[float] = None

    category_type = category_key(category)
    if category_type == "portrait":
        regions = faces or []
    elif category_type == "item":
        regions = subject_regions or []
    else:
        regions = []

    if regions:
        h, w = gray.shape[:2]
        subject_mask = np.zeros((h, w), dtype=np.uint8)
        for region in regions:
            x, y = region.get("x", 0), region.get("y", 0)
            region_width, region_height = region.get("w", 0), region.get("h", 0)
            subject_mask[
                max(0, y):min(h, y + region_height),
                max(0, x):min(w, x + region_width),
            ] = 255

        if np.sum(subject_mask) > 100:
            subject_luminance = float(np.mean(gray[subject_mask == 255]))
            bg_pixels = gray[subject_mask == 0]
            if bg_pixels.size > 0:
                background_luminance = float(np.mean(bg_pixels))

    # Calculate effective photographic brightness
    if subject_luminance is not None:
        # In subject photography, subject exposure is the primary driver
        effective_luminance = 0.65 * subject_luminance + 0.35 * global_luminance
    else:
        effective_luminance = global_luminance

    # Normalize to 0 - 100
    brightness_score = float(np.round((effective_luminance / 255.0) * 100.0, 2))

    if brightness_score < 34.0:
        desc = "Dark / Underexposed (dominant shadow tones, loss of detail in dark regions)"
    elif brightness_score <= 68.0:
        desc = "Normal / Balanced Exposure (even tonal balance across midtones)"
    else:
        desc = "Overexposed / Blown (excessive luminance, highlight clipping)"

    return {
        "brightness": brightness_score,
        "global_luminance": float(np.round((global_luminance / 255.0) * 100.0, 2)),
        "subject_luminance": float(np.round((subject_luminance / 255.0) * 100.0, 2)) if subject_luminance is not None else None,
        "background_luminance": float(np.round((background_luminance / 255.0) * 100.0, 2)) if background_luminance is not None else None,
        "shadow_clipped_pct": float(np.round(shadow_clipped_pct, 2)),
        "highlight_clipped_pct": float(np.round(highlight_clipped_pct, 2)),
        "method": "Perceptual luminance with subject-background segmentation and clipping detection",
        "description": desc,
        "explanation": (
            f"Effective exposure: {brightness_score:.1f}% "
            + (f"(Subject: {(subject_luminance/255.0)*100:.1f}%, Global: {(global_luminance/255.0)*100:.1f}%). " if subject_luminance is not None else f"(Global: {brightness_score:.1f}%). ")
            + f"Clipped highlights: {highlight_clipped_pct:.1f}%, Clipped shadows: {shadow_clipped_pct:.1f}%."
        ),
    }
