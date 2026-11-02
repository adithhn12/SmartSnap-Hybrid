"""Deterministic, image-only features shared by training and inference."""

from typing import Dict

import cv2
import numpy as np

from ..image_processing import (
    estimate_blur,
    estimate_brightness,
    estimate_contrast,
    estimate_noise,
)

FEATURE_NAMES = (
    "brightness",
    "contrast",
    "colorfulness",
    "sharpness",
    "noise",
)

PREPROCESSING_CONFIG = {
    "color_order": "BGR uint8, decoded by OpenCV",
    "brightness": "Whole-image grayscale luminance score in [0, 100]",
    "contrast": "Existing SmartSnap RMS and percentile contrast score in [0, 100]",
    "colorfulness": "Hasler-Susstrunk opponent-colorfulness statistic on 8-bit channels",
    "sharpness": "Whole-image General Photography sharpness score in [0, 100]",
    "noise": "Existing SmartSnap texture-suppressed Immerkaer noise score in [0, 100]",
}


def _colorfulness(image_bgr: np.ndarray) -> float:
    image = image_bgr.astype(np.float32)
    blue, green, red = cv2.split(image)
    rg = np.abs(red - green)
    yb = np.abs(0.5 * (red + green) - blue)
    sigma = np.sqrt(np.var(rg) + np.var(yb))
    mean = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    return float(0.3 * sigma + 0.4 * mean)


def extract_quality_features(image_bgr: np.ndarray) -> Dict[str, float]:
    """Extract the exact fixed-order feature set used to fit and query the model."""
    if not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
        raise ValueError("Expected a non-empty decoded image array.")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError("Expected a three-channel uint8 BGR image.")

    features = {
        "brightness": float(estimate_brightness(image_bgr)["brightness"]),
        "contrast": float(estimate_contrast(image_bgr)["contrast"]),
        "colorfulness": _colorfulness(image_bgr),
        "sharpness": float(
            estimate_blur(
                image_bgr,
                category="General Photography",
                noise_correction=False,
            )["sharpness"]
        ),
        "noise": float(estimate_noise(image_bgr)["noise"]),
    }
    if not all(np.isfinite(value) for value in features.values()):
        raise ValueError("Image feature extraction produced a non-finite value.")
    return features
