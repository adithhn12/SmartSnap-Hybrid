"""
Noise Estimation Module for SmartSnap.
Uses edge-suppressed Immerkaer discrete Laplacian residual estimation
to distinguish genuine sensor noise from natural high-frequency textures (foliage, hair, fabric).
"""

from typing import Dict, Any
import cv2
import numpy as np


def estimate_noise(image_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Estimates additive sensor noise / grain using texture-suppressed Immerkaer kernel:
    N = [[ 1, -2,  1],
         [-2,  4, -2],
         [ 1, -2,  1]]
    Excludes strong edges and fine textures so that sharp in-focus areas are not misclassified as noise.
    """
    if len(image_bgr.shape) == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr.copy()

    h, w = gray.shape[:2]

    # Handle tiny images
    if h < 8 or w < 8:
        return {
            "noise": 0.0,
            "sigma": 0.0,
            "method": "Immerkaer kernel",
            "description": "Image too small for noise estimation",
            "explanation": "Noise estimation skipped for tiny image.",
        }

    # 1. Compute Sobel gradient magnitude to find smooth / low-texture patches
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)

    # Low-texture mask: bottom 30% gradient magnitude (avoids edges, text, foliage, hair)
    grad_interior = grad_mag[1:-1, 1:-1]
    cutoff = float(np.percentile(grad_interior, 30))
    smooth_mask = (grad_interior <= cutoff)

    # 2. Convolve with Immerkaer discrete Laplacian noise mask
    kernel = np.array([[1, -2, 1],
                       [-2, 4, -2],
                       [1, -2, 1]], dtype=np.float64)

    filtered = cv2.filter2D(gray.astype(np.float64), -1, kernel, borderType=cv2.BORDER_REFLECT)
    interior_filtered = np.abs(filtered[1:-1, 1:-1])

    # 3. Compute noise sigma exclusively on the smooth patches
    n_smooth = int(np.sum(smooth_mask))
    if n_smooth > 100:
        sigma = float(np.sum(interior_filtered[smooth_mask]) * np.sqrt(0.5 * np.pi) / (6.0 * n_smooth))
    else:
        # Fallback to entire interior if image is completely textured
        sigma = float(np.sum(interior_filtered) * np.sqrt(0.5 * np.pi) / (6.0 * (w - 2) * (h - 2)))

    # 4. Realistic photographic noise scaling:
    # sigma < 1.0: exceptionally clean sensor (ISO 50-100)
    # sigma 1.0 - 3.5: clean/minimal grain (ISO 200-800)
    # sigma 3.5 - 7.0: moderate grain (ISO 1600-3200)
    # sigma > 9.0: heavy sensor noise / chroma noise (ISO 6400+)
    # Map sigma [0, 9.0] to [0, 100]
    norm_noise = float(np.clip((sigma / 8.5) * 100.0, 0.0, 100.0))
    noise_score = float(np.round(norm_noise, 2))

    if noise_score < 28.0:
        desc = "Low Noise (clean sensor signal, minimal grain)"
    elif noise_score <= 65.0:
        desc = "Medium Noise (moderate grain or high-ISO noise)"
    else:
        desc = "High Noise (heavy sensor grain or digital artifacting)"

    return {
        "noise": noise_score,
        "sigma": float(np.round(sigma, 3)),
        "smooth_pixel_fraction": float(np.round(n_smooth / float(interior_filtered.size) * 100.0, 1)),
        "method": "Texture-suppressed Immerkaer discrete Laplacian noise estimation",
        "description": desc,
        "explanation": f"Estimated noise sigma is {sigma:.2f} on homogeneous image patches ({n_smooth / float(interior_filtered.size) * 100.0:.0f}% of frame sampled). Normalized noise score: {noise_score:.1f}%.",
    }
