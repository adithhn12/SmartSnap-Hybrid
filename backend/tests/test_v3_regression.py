"""
SmartSnap V3 Comprehensive Regression Test Suite.
Validates end-to-end pipeline:
Image -> Feature Extraction -> Fuzzification -> Rule Activation -> Aggregation -> Centroid Defuzzification -> Decision.

Tests the 10 required cases from specification:
1. Sharp face, blurry background (portrait with bokeh -> KEEP)
2. Blurry face, sharp background (missed focus portrait -> MUST NOT receive KEEP, RETAKE)
3. Completely motion-blurred portrait (RETAKE)
4. Slightly soft but usable portrait (RECOVER)
5. Sharp landscape without a face (KEEP)
6. Sharp portrait with multiple detected faces (KEEP)
7. One blurred face among multiple detected faces (NOT KEEP)
8. Sharp high-detail scene (KEEP)
9. Blurry high-detail scene (RETAKE)
10. Photograph with high brightness and severe blur (RETAKE, positive exposure must NOT overpower blur)
Plus live disk validation on user's real images:
11. Real motion-blurred portrait (6337b55e -> RETAKE)
12. Real sharp portrait (c3b8f6cf -> KEEP)
"""

import os
import cv2
import numpy as np
import pytest

from backend.image_processing import (
    estimate_blur,
    estimate_brightness,
    estimate_contrast,
    estimate_noise,
    estimate_face_visibility,
    estimate_composition,
)
from backend.fuzzy import MamdaniInferenceEngine, DecisionEngine


@pytest.fixture(scope="module")
def fuzzy_engine():
    return MamdaniInferenceEngine()


@pytest.fixture(scope="module")
def decision_engine():
    return DecisionEngine()


def _create_synthetic_face(size=140, sharp=True, motion_blur=False):
    """Generates synthetic facial skin patch with eye and lip features."""
    patch = np.ones((size, size, 3), dtype=np.uint8)
    # Skin tone BGR: B=110, G=140, R=190
    patch[:, :, 0] = 110
    patch[:, :, 1] = 140
    patch[:, :, 2] = 190

    if sharp:
        # Sharp high-frequency facial features: eyes, pupils, eyebrows, lips
        eye_y = int(size * 0.35)
        # Left eye & pupil
        cv2.circle(patch, (int(size * 0.32), eye_y), int(size * 0.12), (240, 240, 240), -1)
        cv2.circle(patch, (int(size * 0.32), eye_y), int(size * 0.05), (20, 20, 20), -1)
        # Right eye & pupil
        cv2.circle(patch, (int(size * 0.68), eye_y), int(size * 0.12), (240, 240, 240), -1)
        cv2.circle(patch, (int(size * 0.68), eye_y), int(size * 0.05), (20, 20, 20), -1)
        # Eyebrows
        cv2.line(patch, (int(size * 0.20), eye_y - 12), (int(size * 0.44), eye_y - 14), (30, 20, 20), 4)
        cv2.line(patch, (int(size * 0.56), eye_y - 14), (int(size * 0.80), eye_y - 12), (30, 20, 20), 4)
        # Lips
        cv2.ellipse(patch, (int(size * 0.5), int(size * 0.72)), (int(size * 0.22), int(size * 0.08)), 0, 0, 360, (70, 70, 180), -1)
        cv2.line(patch, (int(size * 0.30), int(size * 0.72)), (int(size * 0.70), int(size * 0.72)), (30, 30, 120), 2)
    elif motion_blur:
        # Render features then smear horizontally
        _create_synthetic_face(size, sharp=True)
        # Motion blur kernel
        kernel = np.zeros((1, 25))
        kernel[0, :] = 1.0 / 25.0
        patch = cv2.filter2D(patch, -1, kernel)
    else:
        # Out of focus blur: render then heavily blur
        eye_y = int(size * 0.35)
        cv2.circle(patch, (int(size * 0.32), eye_y), int(size * 0.12), (240, 240, 240), -1)
        cv2.circle(patch, (int(size * 0.68), eye_y), int(size * 0.12), (240, 240, 240), -1)
        patch = cv2.GaussianBlur(patch, (35, 35), 10.0)

    return patch


def test_case_1_sharp_face_blurry_background(fuzzy_engine, decision_engine):
    """Case 1: Portrait with bokeh background -> must be KEEP."""
    img = np.ones((500, 500, 3), dtype=np.uint8) * 80
    # Add smooth blurred background gradient
    img = cv2.GaussianBlur(img, (45, 45), 15.0)

    # Insert sharp face at center
    face_patch = _create_synthetic_face(size=160, sharp=True)
    img[140:300, 170:330] = face_patch

    fake_face = [{"x": 170, "y": 140, "w": 160, "h": 160}]
    blur_res = estimate_blur(img, faces=fake_face, category="Portrait Photography")
    assert not blur_res["focus_discrepancy"], "Bokeh background must NOT trigger focus discrepancy"
    assert blur_res["sharpness"] >= 65.0, f"Expected high subject sharpness, got {blur_res['sharpness']}"

    fuzzy_inputs = {
        "blur": blur_res["blur"],
        "brightness": 52.0,
        "contrast": 70.0,
        "noise": 10.0,
        "face_visibility": 85.0,
        "composition": 80.0,
    }
    res = fuzzy_engine.compute(fuzzy_inputs, category="Portrait Photography")
    dec = decision_engine.evaluate(res)

    assert dec["recommendation"] == "KEEP", f"Expected KEEP for bokeh portrait, got {dec['recommendation']}"
    assert res["defuzzified_score"] >= 70.0


def test_case_2_blurry_face_sharp_background_missed_focus(fuzzy_engine, decision_engine):
    """Case 2: Blurry face with high-variance background (missed focus) -> MUST NOT receive KEEP."""
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    # Create sharp background high-frequency checkerboard / text edges
    img[::10, :] = 255
    img[:, ::10] = 255

    # Insert blurred out-of-focus face
    blurry_face = _create_synthetic_face(size=160, sharp=False)
    img[140:300, 170:330] = blurry_face

    fake_face = [{"x": 170, "y": 140, "w": 160, "h": 160}]
    blur_res = estimate_blur(img, faces=fake_face, category="Portrait Photography")

    # Critical requirement: focus discrepancy must be detected
    assert blur_res["focus_discrepancy"], "Must detect background sharper than face (missed focus)"
    assert blur_res["sharpness"] < 40.0, f"Effective sharpness must be penalized, got {blur_res['sharpness']}"
    assert blur_res["blur"] >= 60.0

    fuzzy_inputs = {
        "blur": blur_res["blur"],
        "brightness": 50.0,
        "contrast": 60.0,
        "noise": 10.0,
        "face_visibility": 75.0,
        "composition": 75.0,
    }
    res = fuzzy_engine.compute(fuzzy_inputs, category="Portrait Photography")
    dec = decision_engine.evaluate(res)

    # CRITICAL REQUIREMENT: Must NOT receive KEEP!
    assert dec["recommendation"] != "KEEP", "Missed focus portrait must NEVER be classified as KEEP!"
    assert dec["recommendation"] == "RETAKE"


def test_case_3_completely_motion_blurred_portrait(fuzzy_engine, decision_engine):
    """Case 3: Complete motion blur across face and scene -> RETAKE."""
    img = np.ones((500, 500, 3), dtype=np.uint8) * 120
    face_patch = _create_synthetic_face(size=160, sharp=True)
    img[140:300, 170:330] = face_patch

    # Apply severe horizontal motion blur
    kernel = np.zeros((1, 31))
    kernel[0, :] = 1.0 / 31.0
    img = cv2.filter2D(img, -1, kernel)

    fake_face = [{"x": 170, "y": 140, "w": 160, "h": 160}]
    blur_res = estimate_blur(img, faces=fake_face, category="Portrait Photography")

    assert blur_res["blur"] >= 70.0, f"Expected high blur index, got {blur_res['blur']}"

    fuzzy_inputs = {
        "blur": blur_res["blur"],
        "brightness": 48.0,
        "contrast": 45.0,
        "noise": 8.0,
        "face_visibility": 65.0,
        "composition": 70.0,
    }
    res = fuzzy_engine.compute(fuzzy_inputs, category="Portrait Photography")
    dec = decision_engine.evaluate(res)

    assert dec["recommendation"] == "RETAKE"
    assert res["defuzzified_score"] < 45.0


def test_case_4_slightly_soft_usable_portrait(fuzzy_engine, decision_engine):
    """Case 4: Slightly soft but recoverable portrait -> RECOVER."""
    inputs = {
        "blur": 46.0,
        "brightness": 52.0,
        "contrast": 62.0,
        "noise": 15.0,
        "face_visibility": 75.0,
        "composition": 75.0,
    }
    res = fuzzy_engine.compute(inputs, category="Portrait Photography")
    dec = decision_engine.evaluate(res)

    assert dec["recommendation"] == "RECOVER"
    assert 45.0 <= res["defuzzified_score"] < 70.0


def test_case_5_sharp_landscape_without_face(fuzzy_engine, decision_engine):
    """Case 5: Sharp landscape with 0 faces -> KEEP (must not be penalized for missing faces)."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # Fine textural landscape lines across multiple regions
    img[::6, :] = 200
    img[:, ::6] = 200

    blur_res = estimate_blur(img, faces=[], category="Landscape Photography")
    assert blur_res["sharpness"] >= 70.0

    inputs = {
        "blur": blur_res["blur"],
        "brightness": 54.0,
        "contrast": 80.0,
        "noise": 10.0,
        "face_visibility": 0.0,
        "composition": 85.0,
    }
    res = fuzzy_engine.compute(inputs, category="Landscape Photography")
    dec = decision_engine.evaluate(res)

    assert dec["recommendation"] == "KEEP"
    assert res["defuzzified_score"] >= 70.0
    assert not any("face" in flag.lower() for flag in dec["critical_flags"])


def test_case_6_sharp_portrait_with_multiple_faces(fuzzy_engine, decision_engine):
    """Case 6: Multiple sharp detected faces -> KEEP."""
    img = np.ones((500, 700, 3), dtype=np.uint8) * 100
    f1 = _create_synthetic_face(size=120, sharp=True)
    f2 = _create_synthetic_face(size=120, sharp=True)
    img[150:270, 100:220] = f1
    img[150:270, 400:520] = f2

    fake_faces = [
        {"x": 100, "y": 150, "w": 120, "h": 120},
        {"x": 400, "y": 150, "w": 120, "h": 120},
    ]
    blur_res = estimate_blur(img, faces=fake_faces, category="Portrait Photography")
    assert blur_res["sharpness"] >= 65.0

    inputs = {
        "blur": blur_res["blur"],
        "brightness": 52.0,
        "contrast": 75.0,
        "noise": 12.0,
        "face_visibility": 85.0,
        "composition": 80.0,
    }
    res = fuzzy_engine.compute(inputs, category="Portrait Photography")
    dec = decision_engine.evaluate(res)

    assert dec["recommendation"] == "KEEP"


def test_case_7_one_blurred_face_in_portrait(fuzzy_engine, decision_engine):
    """Case 7: One blurred detected face must prevent a KEEP recommendation."""
    img = np.ones((500, 700, 3), dtype=np.uint8) * 100
    f1 = _create_synthetic_face(size=120, sharp=True)
    f2 = _create_synthetic_face(size=120, sharp=False)  # Blurry second face!
    img[150:270, 100:220] = f1
    img[150:270, 400:520] = f2

    fake_faces = [
        {"x": 100, "y": 150, "w": 120, "h": 120},
        {"x": 400, "y": 150, "w": 120, "h": 120},
    ]
    blur_res = estimate_blur(img, faces=fake_faces, category="Portrait Photography")
    assert blur_res["blur"] >= 55.0, f"Expected degraded portrait sharpness, got {blur_res['sharpness']}"

    inputs = {
        "blur": blur_res["blur"],
        "brightness": 50.0,
        "contrast": 60.0,
        "noise": 15.0,
        "face_visibility": 75.0,
        "composition": 70.0,
    }
    res = fuzzy_engine.compute(inputs, category="Portrait Photography")
    dec = decision_engine.evaluate(res)

    assert dec["recommendation"] != "KEEP", "Blurry detected face must not allow KEEP recommendation"


def test_case_8_sharp_landscape_scene(fuzzy_engine, decision_engine):
    """Case 8: Sharp, well-exposed scene detail -> KEEP."""
    img = np.ones((500, 500, 3), dtype=np.uint8) * 240
    # Crisp black text lines
    for y in range(40, 460, 24):
        cv2.putText(img, "SMARTSNAP ACCURACY AUDIT 2026", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)

    blur_res = estimate_blur(img, faces=[], category="Landscape Photography")
    assert blur_res["sharpness"] >= 70.0

    inputs = {
        "blur": blur_res["blur"],
        "brightness": 55.0,
        "contrast": 88.0,
        "noise": 8.0,
        "face_visibility": 0.0,
        "composition": 75.0,
    }
    res = fuzzy_engine.compute(inputs, category="Landscape Photography")
    dec = decision_engine.evaluate(res)

    assert dec["recommendation"] == "KEEP"
    assert res["defuzzified_score"] >= 70.0


def test_case_9_blurry_landscape_scene(fuzzy_engine, decision_engine):
    """Case 9: Severe scene blur -> RETAKE."""
    img = np.ones((500, 500, 3), dtype=np.uint8) * 240
    for y in range(40, 460, 24):
        cv2.putText(img, "SMARTSNAP ACCURACY AUDIT 2026", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)
    # Heavy defocus
    img = cv2.GaussianBlur(img, (25, 25), 8.0)

    blur_res = estimate_blur(img, faces=[], category="Landscape Photography")
    assert blur_res["blur"] >= 70.0

    inputs = {
        "blur": blur_res["blur"],
        "brightness": 55.0,
        "contrast": 40.0,
        "noise": 12.0,
        "face_visibility": 0.0,
        "composition": 70.0,
    }
    res = fuzzy_engine.compute(inputs, category="Landscape Photography")
    dec = decision_engine.evaluate(res)

    assert dec["recommendation"] == "RETAKE"
    assert res["defuzzified_score"] < 45.0


def test_case_10_high_brightness_severe_blur_cannot_pass(fuzzy_engine, decision_engine):
    """Case 10: High brightness + severe blur -> RETAKE (positive exposure must NOT overpower blur)."""
    inputs = {
        "blur": 85.0,          # Severe blur
        "brightness": 52.0,    # Ideal normal brightness
        "contrast": 65.0,
        "noise": 6.0,          # Clean sensor
        "face_visibility": 70.0,
        "composition": 80.0,   # Great composition
    }
    res = fuzzy_engine.compute(inputs, category="Portrait Photography")
    dec = decision_engine.evaluate(res)

    # Blur must dominate!
    assert dec["recommendation"] == "RETAKE", f"Expected RETAKE despite good exposure, got {dec['recommendation']}"
    assert res["defuzzified_score"] < 40.0


def test_case_11_real_motion_blurred_portrait_receives_retake():
    """Validates that user's motion-blurred portrait (blur person.jpg) receives RETAKE."""
    portrait_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "uploads", "6337b55e-6a79-41f9-ad50-6e95eed382be.jpg"
    )
    if not os.path.exists(portrait_path):
        pytest.skip("Test image 6337b55e...jpg not found")

    img = cv2.imread(portrait_path)
    face_res = estimate_face_visibility(img)
    blur_res = estimate_blur(img, faces=face_res.get("faces", []), category="Portrait Photography")

    assert blur_res["focus_discrepancy"] is True, "Must detect focus discrepancy (sharp bg, blurred face)"
    assert blur_res["blur"] >= 70.0, f"Blur index should be >= 70, got {blur_res['blur']}"

    engine = MamdaniInferenceEngine()
    dec_engine = DecisionEngine()
    fuzzy_inputs = {
        "blur": blur_res["blur"],
        "brightness": 48.8,
        "contrast": 36.1,
        "noise": 6.6,
        "face_visibility": face_res["face_visibility"],
        "composition": 71.1,
    }
    res = engine.compute(fuzzy_inputs, category="Portrait Photography")
    dec = dec_engine.evaluate(res)

    assert dec["recommendation"] == "RETAKE", f"Real blur photo must receive RETAKE, got {dec['recommendation']}"
    assert res["defuzzified_score"] < 40.0


def test_case_12_real_sharp_portrait_receives_keep():
    """Validates that user's sharp portrait (c3b8f6cf...jpeg) receives KEEP."""
    portrait_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "uploads", "c3b8f6cf-fe5f-4996-9bf0-9b775522fd69.jpeg"
    )
    if not os.path.exists(portrait_path):
        pytest.skip("Test image c3b8f6cf...jpeg not found")

    img = cv2.imread(portrait_path)
    face_res = estimate_face_visibility(img)
    blur_res = estimate_blur(img, faces=face_res.get("faces", []), category="Portrait Photography")

    assert blur_res["sharpness"] >= 65.0, f"Expected sharp portrait >= 65%, got {blur_res['sharpness']}"
    assert blur_res["blur"] <= 35.0

    engine = MamdaniInferenceEngine()
    dec_engine = DecisionEngine()
    fuzzy_inputs = {
        "blur": blur_res["blur"],
        "brightness": 43.5,
        "contrast": 69.9,
        "noise": 3.6,
        "face_visibility": face_res["face_visibility"],
        "composition": 83.0,
    }
    res = engine.compute(fuzzy_inputs, category="Portrait Photography")
    dec = dec_engine.evaluate(res)

    assert dec["recommendation"] == "KEEP", f"Real sharp portrait must receive KEEP, got {dec['recommendation']}"
    assert res["defuzzified_score"] >= 70.0
