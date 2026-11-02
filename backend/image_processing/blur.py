"""
Blur and Sharpness Estimation Module for SmartSnap V3.
Performs:
1. Standardized resolution normalization (preventing megapixel bias).
2. Standardized face ROI scaling (preventing face size/crop bias).
3. Primary subject and face-region sharpness evaluation.
4. Directional motion blur and gradient coherence assessment.
5. Background sharpness and focus discrepancy (missed focus) detection.
6. Multi-face Portrait Photography assessment.
7. Multi-region 3x3 distributed grid evaluation for landscapes.
8. Foreground-region sharpness assessment for Item Photography.
9. Consistent sharpness scale (0 = extremely blurry, 100 = razor sharp).
"""

from typing import Dict, Any, List, Optional
import cv2
import numpy as np

from .noise import estimate_noise

if "." in __package__:
    from ..categories import category_key
else:
    from categories import category_key


LAPLACIAN_WHITE_NOISE_GAIN = 20.0


def _noise_corrected_laplacian_variance(
    region: np.ndarray, noise_sigma: float
) -> float:
    observed_variance = float(cv2.Laplacian(region, cv2.CV_64F).var())
    return max(0.0, observed_variance - LAPLACIAN_WHITE_NOISE_GAIN * noise_sigma**2)


def estimate_blur(
    image_bgr: np.ndarray,
    faces: Optional[List[Dict[str, int]]] = None,
    category: str = "General Photography",
    target_dim: int = 1000,
    noise_correction: bool = True,
    subject_regions: Optional[List[Dict[str, int]]] = None,
) -> Dict[str, Any]:
    """
    Computes photographic sharpness and blur index with subject-first intelligence.
    Guarantees that a portrait with a blurred subject will NEVER be classified as sharp
    merely because the background contains high-frequency edges or high variance.
    """
    category_type = category_key(category)
    h, w = image_bgr.shape[:2]

    # 1. Standardize image scale for resolution invariance
    max_side = max(h, w)
    if max_side > target_dim:
        scale = target_dim / float(max_side)
        gray = cv2.resize(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr.copy()
        scale = 1.0

    gh, gw = gray.shape

    # 2. Correct Laplacian energy for the estimated white-noise contribution.
    # A 4-neighbor Laplacian has squared-kernel gain 20 for independent noise.
    noise_sigma = (
        float(estimate_noise(gray)["sigma"]) if noise_correction else 0.0
    )
    lap_global = cv2.Laplacian(gray, cv2.CV_64F)
    var_global_raw = float(lap_global.var())
    var_global = max(
        0.0, var_global_raw - LAPLACIAN_WHITE_NOISE_GAIN * noise_sigma**2
    )
    s_global = float(np.round(100.0 * (1.0 - np.exp(-var_global / 160.0)), 2))

    # 3. Block-level focal sharpness (top 20% sharpest blocks)
    bh, bw = max(16, gh // 16), max(16, gw // 16)
    block_vars = []
    for y in range(0, gh - bh + 1, bh):
        for x in range(0, gw - bw + 1, bw):
            b = gray[y:y + bh, x:x + bw]
            block_vars.append(
                _noise_corrected_laplacian_variance(b, noise_sigma)
            )

    block_vars = np.array(block_vars)
    if len(block_vars) > 0:
        top_k = max(1, int(len(block_vars) * 0.20))
        var_focal = float(np.mean(np.sort(block_vars)[-top_k:]))
    else:
        var_focal = var_global

    # 4. Verified Face / Subject Sharpness Evaluation
    # Sort faces by area descending so primary subject face is first
    if category_type == "portrait":
        regions = faces or []
    elif category_type == "item":
        regions = subject_regions or []
    else:
        regions = []
    sorted_regions = sorted(
        regions,
        key=lambda item: item.get("w", 0) * item.get("h", 0),
        reverse=True,
    )

    var_subject: Optional[float] = None
    s_face: Optional[float] = None
    s_object: Optional[float] = None
    motion_coherence: float = 0.0
    subject_sharpnesses: List[Dict[str, float]] = []

    for region in sorted_regions:
        fx = int(region.get("x", 0) * scale)
        fy = int(region.get("y", 0) * scale)
        fw = int(region.get("w", 0) * scale)
        fh = int(region.get("h", 0) * scale)

        subject_roi = gray[max(0, fy):min(gh, fy + fh), max(0, fx):min(gw, fx + fw)]
        if subject_roi.size > 100:
            # Standardize subject crops to compare detail independently of scale.
            standardized_subject = cv2.resize(
                subject_roi,
                (160, 160),
                interpolation=cv2.INTER_AREA if fw > 160 else cv2.INTER_LINEAR
            )

            subject_noise_sigma = float(estimate_noise(standardized_subject)["sigma"])
            vf_raw = _noise_corrected_laplacian_variance(
                standardized_subject, subject_noise_sigma
            )

            # Directional motion blur detection: structure tensor coherence
            gx = cv2.Sobel(standardized_subject, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(standardized_subject, cv2.CV_64F, 0, 1, ksize=3)
            jxx = np.mean(gx**2)
            jyy = np.mean(gy**2)
            jxy = np.mean(gx * gy)
            trace = jxx + jyy
            det = jxx * jyy - jxy**2
            l1 = 0.5 * (trace + np.sqrt(max(0.0, trace**2 - 4 * det)))
            l2 = 0.5 * (trace - np.sqrt(max(0.0, trace**2 - 4 * det)))
            coherence = float((l1 - l2) / (l1 + l2 + 1e-6))
            p_motion = max(0.30, 1.0 - 1.5 * max(0.0, coherence - 0.35))

            # High-frequency detail density (reblur difference)
            b_sm = cv2.GaussianBlur(standardized_subject, (3, 3), 0.8)
            d_hf = float(
                np.abs(standardized_subject.astype(float) - b_sm.astype(float)).mean()
            )
            f_detail = float(np.clip(d_hf / 1.6, 0.40, 1.30))

            v_eff = vf_raw * p_motion * f_detail
            sf = float(100.0 * (1.0 - np.exp(-v_eff / 90.0)))
            subject_sharpnesses.append({
                "sharpness": sf,
                "var_std": vf_raw,
                "coherence": coherence,
                "p_motion": p_motion,
            })

    if subject_sharpnesses:
        primary = subject_sharpnesses[0]
        var_subject = float(np.round(primary["var_std"], 2))
        primary_sharpness = float(np.round(primary["sharpness"], 2))
        motion_coherence = float(np.round(primary["coherence"], 3))
        if category_type == "portrait":
            face_scores = [item["sharpness"] for item in subject_sharpnesses]
            s_face = float(
                np.round(0.65 * min(face_scores) + 0.35 * np.mean(face_scores), 2)
            )
        elif category_type == "item":
            s_object = primary_sharpness

    # 5. Background sharpness outside the category-relevant subject regions.
    bg_mask = np.ones((gh, gw), dtype=bool)
    for region in sorted_regions:
        fx = int(region.get("x", 0) * scale)
        fy = int(region.get("y", 0) * scale)
        fw = int(region.get("w", 0) * scale)
        fh = int(region.get("h", 0) * scale)
        bg_mask[max(0, fy):min(gh, fy + fh), max(0, fx):min(gw, fx + fw)] = False

    var_bg = (
        max(
            0.0,
            float(lap_global[bg_mask].var())
            - LAPLACIAN_WHITE_NOISE_GAIN * noise_sigma**2,
        )
        if np.any(bg_mask)
        else var_global
    )
    s_bg = float(np.round(100.0 * (1.0 - np.exp(-var_bg / 160.0)), 2))

    # 6. Category-Aware Effective Sharpness Computation
    focus_discrepancy = False

    if category_type == "portrait":
        if s_face is not None:
            # Core Portrait Rule:
            # Use every detected face while preserving a stronger penalty for the softest face.
            # A sharper background signals missed focus; softer background blur is not penalized.
            if s_bg > s_face + 12.0:
                focus_discrepancy = True
                disc_ratio = s_face / (s_bg + 1e-5)
                effective_sharpness = s_face * (disc_ratio ** 0.50)
            else:
                effective_sharpness = s_face
        else:
            # Portrait selected but no face found
            effective_sharpness = 0.60 * (100.0 * (1.0 - np.exp(-var_focal / 160.0))) + 0.40 * s_global

    elif category_type == "item":
        if s_object is not None:
            if s_bg > s_object + 12.0:
                focus_discrepancy = True
                disc_ratio = s_object / (s_bg + 1e-5)
                effective_sharpness = s_object * (disc_ratio ** 0.50)
            else:
                effective_sharpness = s_object
        else:
            effective_sharpness = (
                0.60 * (100.0 * (1.0 - np.exp(-var_focal / 160.0)))
                + 0.40 * s_global
            )

    elif category_type == "landscape":
        # 3x3 distributed grid blocks across the entire scenery
        grid_vars = []
        for r in range(3):
            for c in range(3):
                y1, y2 = r * gh // 3, (r + 1) * gh // 3
                x1, x2 = c * gw // 3, (c + 1) * gw // 3
                grid_vars.append(
                    _noise_corrected_laplacian_variance(
                        gray[y1:y2, x1:x2], noise_sigma
                    )
                )
        grid_vars = np.array(grid_vars)
        med_var = float(np.median(grid_vars))
        top3_var = float(np.mean(np.sort(grid_vars)[-3:]))
        landscape_eff_var = 0.50 * med_var + 0.35 * top3_var + 0.15 * var_global
        effective_sharpness = float(100.0 * (1.0 - np.exp(-landscape_eff_var / 150.0)))

    else:
        # General Photography
        if s_face is not None:
            if s_bg > s_face + 15.0:
                focus_discrepancy = True
                disc_ratio = s_face / (s_bg + 1e-5)
                effective_sharpness = s_face * (disc_ratio ** 0.45)
            else:
                effective_sharpness = 0.75 * s_face + 0.25 * s_global
        else:
            effective_sharpness = 0.60 * (100.0 * (1.0 - np.exp(-var_focal / 160.0))) + 0.40 * s_global

    sharpness = float(np.round(np.clip(effective_sharpness, 0.0, 100.0), 2))
    blur = float(np.round(100.0 - sharpness, 2))

    if sharpness >= 70.0:
        desc = "High sharpness (crisp in-focus subject edges with well-resolved detail)"
    elif sharpness >= 45.0:
        desc = "Moderate sharpness (acceptable clarity, minor focus softening or mild motion)"
    else:
        desc = "Low sharpness / Blurry (severe optical defocus, motion blur, or camera shake)"

    # Build detailed explanation
    exp_parts = [f"Effective sharpness: {sharpness:.1f}% (Blur index: {blur:.1f}%)."]
    if category_type == "portrait" and s_face is not None:
        exp_parts.append(
            f"Face-region sharpness: {s_face:.1f}% across {len(subject_sharpnesses)} detected face(s)."
        )
    elif category_type == "item" and s_object is not None:
        exp_parts.append(f"Primary object sharpness: {s_object:.1f}%.")
    exp_parts.append(f"Global sharpness: {s_global:.1f}%, Background sharpness: {s_bg:.1f}%.")
    if motion_coherence > 0.40:
        exp_parts.append(f"Directional motion smear detected (coherence: {motion_coherence:.2f}).")
    if focus_discrepancy:
        exp_parts.append("Focus discrepancy: background is significantly sharper than the primary subject (missed focus).")

    return {
        "sharpness": sharpness,
        "blur": blur,
        "global_sharpness": s_global,
        "face_sharpness": s_face,
        "face_sharpnesses": (
            [float(np.round(item["sharpness"], 2)) for item in subject_sharpnesses]
            if category_type == "portrait"
            else []
        ),
        "object_sharpness": s_object,
        "background_sharpness": s_bg,
        "var_global": float(np.round(var_global, 2)),
        "var_global_raw": float(np.round(var_global_raw, 2)),
        "var_focal": float(np.round(var_focal, 2)),
        "var_subject": var_subject,
        "var_bg": float(np.round(var_bg, 2)),
        "noise_sigma": float(np.round(noise_sigma, 3)),
        "motion_coherence": motion_coherence,
        "focus_discrepancy": focus_discrepancy,
        "method": (
            "Noise-corrected subject-aware multi-region sharpness with motion & focus discrepancy analysis"
            if noise_correction
            else "Subject-aware multi-region sharpness with motion & focus discrepancy analysis"
        ),
        "description": desc,
        "explanation": " ".join(exp_parts),
    }
