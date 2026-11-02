"""
Tests for OpenCV image processing feature extraction.
"""

import numpy as np
import cv2
import pytest
import os
from pathlib import Path
from backend.image_processing.face_detection import FaceDetector
from backend.image_processing import (
    estimate_blur,
    estimate_brightness,
    estimate_contrast,
    estimate_noise,
    estimate_face_visibility,
    estimate_composition,
    estimate_item_subject,
)


def _make_rock_tree_landscape(seed=2026):
    rng = np.random.default_rng(seed)
    height, width = 512, 768
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        image[y, :, 0] = int(120 - 40 * y / height)
        image[y, :, 1] = int(185 - 70 * y / height)
        image[y, :, 2] = int(220 - 75 * y / height)

    mountain = np.array(
        [[0, 230], [110, 90], [240, 218], [365, 65], [520, 220], [690, 115], [767, 230]],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [mountain], (110, 125, 135))
    for _ in range(500):
        x = int(rng.integers(0, width))
        y = int(rng.integers(75, 245))
        color = int(rng.integers(75, 160))
        cv2.line(
            image,
            (x, y),
            (
                min(width - 1, x + int(rng.integers(2, 15))),
                min(250, y + int(rng.integers(-8, 9))),
            ),
            (color, color, color),
            1,
        )
    for x in range(0, width, 18):
        if rng.random() < 0.8:
            tree_height = int(rng.integers(35, 110))
            tree = np.array(
                [[x, 245], [x + 8, 245 - tree_height], [x + 17, 245]],
                dtype=np.int32,
            )
            cv2.fillPoly(image, [tree], (25, 65, 42))
    cv2.rectangle(image, (0, 265), (width - 1, height - 1), (130, 145, 145), -1)
    reflection = cv2.flip(image[:250], 0)
    image[265:512] = reflection[:247]
    for _ in range(35):
        x = int(rng.integers(0, width))
        y = int(rng.integers(10, 75))
        cv2.ellipse(
            image,
            (x, y),
            (int(rng.integers(8, 38)), int(rng.integers(3, 10))),
            0,
            0,
            360,
            (195, 210, 220),
            -1,
        )
    for y in range(280, 510, 8):
        shade = int(rng.integers(110, 155))
        cv2.line(image, (0, y), (width - 1, y), (shade, shade, shade), 1)
    return image


def _local_upload(filename):
    path = Path(__file__).resolve().parents[1] / "uploads" / filename
    if not path.is_file():
        pytest.skip("Optional local diagnostic image is not available.")
    image = cv2.imread(str(path))
    if image is None:
        pytest.skip("Optional local diagnostic image could not be decoded.")
    return image


def test_blur_sharp_vs_blurry():
    # Create sharp synthetic image with high-frequency checkerboard / lines
    sharp_img = np.zeros((400, 400, 3), dtype=np.uint8)
    sharp_img[::8, :] = 255
    sharp_img[:, ::8] = 255

    # Blurry version
    blurry_img = cv2.GaussianBlur(sharp_img, (31, 31), 10)

    sharp_res = estimate_blur(sharp_img)
    blurry_res = estimate_blur(blurry_img)

    assert sharp_res["sharpness"] > blurry_res["sharpness"]
    assert sharp_res["blur"] < blurry_res["blur"]


def test_brightness_dark_vs_bright():
    dark_img = np.ones((200, 200, 3), dtype=np.uint8) * 30
    bright_img = np.ones((200, 200, 3), dtype=np.uint8) * 220

    dark_res = estimate_brightness(dark_img)
    bright_res = estimate_brightness(bright_img)

    assert dark_res["brightness"] < 25.0
    assert bright_res["brightness"] > 80.0


def test_contrast_flat_vs_high():
    flat_img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    high_contrast = np.zeros((200, 200, 3), dtype=np.uint8)
    high_contrast[:100, :] = 255

    flat_res = estimate_contrast(flat_img)
    high_res = estimate_contrast(high_contrast)

    assert high_res["contrast"] > flat_res["contrast"]


def test_noise_clean_vs_noisy():
    clean_img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    noise = np.random.normal(0, 25, (200, 200, 3)).astype(np.float64)
    noisy_img = np.clip(clean_img + noise, 0, 255).astype(np.uint8)

    clean_res = estimate_noise(clean_img)
    noisy_res = estimate_noise(noisy_img)

    assert noisy_res["noise"] > clean_res["noise"]


def test_composition():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # Circle placed near rule of thirds (100, 100)
    cv2.circle(img, (100, 100), 25, (255, 255, 255), -1)

    comp_res = estimate_composition(img)
    assert 0 <= comp_res["composition"] <= 100
    assert "subject_type" in comp_res


def test_portrait_composition_uses_primary_face():
    img = np.zeros((300, 600, 3), dtype=np.uint8)
    faces = [
        {"x": 80, "y": 80, "w": 80, "h": 80},
        {"x": 440, "y": 80, "w": 80, "h": 80},
    ]

    result = estimate_composition(
        img, faces=faces, category="Portrait Photography"
    )

    assert result["subject_type"] == "Detected Face"
    assert result["subject_center"]["x"] == pytest.approx(0.2)


def test_item_composition_uses_primary_object_region():
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    object_region = {"x": 70, "y": 30, "w": 260, "h": 240}

    result = estimate_composition(
        img,
        category="Item Photography",
        subject_regions=[object_region],
    )

    assert result["subject_type"] == "Detected primary object region"
    assert result["subject_center"] == {"x": 0.5, "y": 0.5}
    assert result["composition"] > 80.0


def test_item_subject_proposal_returns_box_or_neutral_fallback():
    image = np.full((240, 320, 3), 30, dtype=np.uint8)
    cv2.rectangle(image, (95, 45), (225, 195), (220, 220, 220), -1)

    result = estimate_item_subject(image)

    assert 0.0 <= result["object_visibility"] <= 100.0
    if result["object_detected"]:
        assert result["object_box"]["w"] > 0
        assert result["object_box"]["h"] > 0
        assert "not semantic object recognition" in result["explanation"]
    else:
        assert result["object_visibility"] == 50.0


@pytest.mark.parametrize("shape", ["rectangle", "circle"])
def test_item_subject_proposal_handles_object_shapes_and_clutter(shape):
    rng = np.random.default_rng(77)
    image = np.full((300, 400, 3), 55, dtype=np.uint8)
    image[::12, :, :] = 64
    image[:, ::12, :] = 64
    if shape == "rectangle":
        cv2.rectangle(image, (135, 55), (275, 250), (210, 210, 210), -1)
        cv2.rectangle(image, (155, 80), (255, 220), (25, 25, 25), 3)
    else:
        cv2.circle(image, (205, 150), 92, (210, 210, 210), -1)
        cv2.circle(image, (205, 150), 58, (30, 30, 30), 4)
    image[120:155, 182:220] = np.clip(
        image[120:155, 182:220].astype(np.int16)
        + rng.integers(-20, 21, size=(35, 38, 1)),
        0,
        255,
    ).astype(np.uint8)
    cv2.rectangle(image, (10, 25), (70, 100), (180, 90, 40), -1)

    result = estimate_item_subject(image)

    assert 0.0 <= result["object_visibility"] <= 100.0
    if result["object_detected"]:
        box = result["object_box"]
        assert 0 <= box["x"] < image.shape[1]
        assert 0 <= box["y"] < image.shape[0]
        assert box["w"] > 0 and box["h"] > 0
    else:
        assert result["object_visibility"] == 50.0


def test_partially_obscured_portrait_face_keeps_bounded_region_metrics():
    clear_face = _make_rock_tree_landscape()[80:200, 120:240].copy()
    obscured_face = clear_face.copy()
    obscured_face[42:82, 25:95] = 90
    image = np.full((220, 220, 3), 70, dtype=np.uint8)
    image[50:170, 50:170] = obscured_face
    faces = [{"x": 50, "y": 50, "w": 120, "h": 120}]

    result = estimate_blur(
        image,
        faces=faces,
        category="Portrait Photography",
    )

    clear_result = estimate_blur(
        cv2.copyMakeBorder(
            clear_face, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(70, 70, 70)
        ),
        faces=faces,
        category="Portrait Photography",
    )
    assert result["face_sharpnesses"]
    assert all(0.0 <= value <= 100.0 for value in result["face_sharpnesses"])
    assert 0.0 <= result["sharpness"] <= 100.0
    assert 0.0 <= clear_result["sharpness"] <= 100.0


def test_face_visibility_does_not_reward_a_sharp_occlusion_edge():
    image = _local_upload("c3b8f6cf-fe5f-4996-9bf0-9b775522fd69.jpeg")
    baseline = estimate_face_visibility(image)
    if not baseline["faces"]:
        pytest.skip("Reference portrait face was not detected.")

    face = baseline["faces"][0]
    occluded = image.copy()
    covered_height = max(1, round(face["h"] * 0.25))
    occluded[
        face["y"] : face["y"] + covered_height,
        face["x"] : face["x"] + face["w"],
    ] = 0
    result = estimate_face_visibility(occluded)

    assert 0.0 <= result["face_visibility"] <= 100.0
    assert result["face_visibility"] < baseline["face_visibility"]


def test_item_exposure_uses_object_region_without_face_metrics():
    image = np.full((220, 320, 3), 25, dtype=np.uint8)
    image[45:175, 95:225] = 190
    object_region = [{"x": 95, "y": 45, "w": 130, "h": 130}]
    fake_face = [{"x": 0, "y": 0, "w": 80, "h": 80}]

    subject_weighted = estimate_brightness(
        image,
        faces=fake_face,
        category="Item Photography",
        subject_regions=object_region,
    )
    face_only = estimate_brightness(
        image,
        faces=fake_face,
        category="Item Photography",
    )

    assert subject_weighted["subject_luminance"] > subject_weighted["global_luminance"]
    assert subject_weighted["brightness"] > subject_weighted["global_luminance"]
    assert face_only["subject_luminance"] is None


def test_blur_receives_faces_kwarg():
    """Verify that passing faces= kwarg to estimate_blur does not raise errors."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    fake_faces = [{"x": 50, "y": 50, "w": 80, "h": 80}]
    res_with_faces = estimate_blur(img, faces=fake_faces, category="Portrait Photography")
    res_without_faces = estimate_blur(img, faces=None)
    assert "sharpness" in res_with_faces
    assert "blur" in res_with_faces
    assert "var_subject" in res_with_faces
    assert res_with_faces["var_subject"] is not None
    assert res_without_faces["var_subject"] is None


def test_non_human_categories_ignore_face_boxes_for_blur_and_brightness():
    image = _make_rock_tree_landscape()
    face = [{"x": 180, "y": 120, "w": 80, "h": 80}]

    landscape_with_face = estimate_blur(image, faces=face, category="Landscape Photography")
    landscape_without_face = estimate_blur(image, category="Landscape Photography")
    general_with_face = estimate_blur(image, faces=face, category="General Photography")
    general_without_face = estimate_blur(image, category="General Photography")
    brightness_with_face = estimate_brightness(
        image, faces=face, category="Landscape Photography"
    )
    brightness_without_face = estimate_brightness(image)

    assert landscape_with_face["sharpness"] == landscape_without_face["sharpness"]
    assert landscape_with_face["face_sharpness"] is None
    assert general_with_face["sharpness"] == general_without_face["sharpness"]
    assert general_with_face["face_sharpness"] is None
    assert brightness_with_face["brightness"] == brightness_without_face["brightness"]


def test_portrait_sharpness_accounts_for_every_detected_face():
    sharp = _make_rock_tree_landscape()[80:200, 120:240]
    soft = cv2.GaussianBlur(sharp, (31, 31), 10.0)
    image = np.full((300, 500, 3), 80, dtype=np.uint8)
    image[80:200, 80:200] = sharp
    image[80:200, 320:440] = soft
    faces = [
        {"x": 80, "y": 80, "w": 120, "h": 120},
        {"x": 320, "y": 80, "w": 120, "h": 120},
    ]

    result = estimate_blur(image, faces=faces, category="Portrait Photography")

    assert len(result["face_sharpnesses"]) == 2
    assert result["face_sharpnesses"][0] > result["face_sharpnesses"][1]
    assert result["face_sharpness"] < result["face_sharpnesses"][0]


def test_item_blur_distinguishes_object_detail_from_background_texture():
    region = {"x": 150, "y": 90, "w": 180, "h": 180}
    checker = (np.indices((180, 180)).sum(axis=0) // 6 % 2 * 180 + 35).astype(
        np.uint8
    )
    sharp_object = cv2.cvtColor(checker, cv2.COLOR_GRAY2BGR)
    smooth_background = np.full((360, 480, 3), 110, dtype=np.uint8)
    sharp_object_scene = smooth_background.copy()
    sharp_object_scene[90:270, 150:330] = sharp_object

    textured_background = np.indices((360, 480)).sum(axis=0) // 4 % 2
    textured_background = (textured_background * 180 + 35).astype(np.uint8)
    blurred_object_scene = cv2.cvtColor(textured_background, cv2.COLOR_GRAY2BGR)
    blurred_object_scene[90:270, 150:330] = cv2.GaussianBlur(
        sharp_object, (31, 31), 10.0
    )

    sharp_result = estimate_blur(
        sharp_object_scene,
        category="Item Photography",
        subject_regions=[region],
    )
    blurred_result = estimate_blur(
        blurred_object_scene,
        category="Item Photography",
        subject_regions=[region],
    )

    assert sharp_result["object_sharpness"] > blurred_result["object_sharpness"]
    assert blurred_result["focus_discrepancy"] is True
    assert blurred_result["blur"] > sharp_result["blur"]


def test_landscape_rock_tree_cloud_and_reflection_textures_are_not_noise_or_blur():
    image = _make_rock_tree_landscape()

    faces = estimate_face_visibility(image)
    blur = estimate_blur(image, faces=faces["faces"], category="Landscape Photography")
    brightness = estimate_brightness(image)
    contrast = estimate_contrast(image)
    noise = estimate_noise(image)

    assert faces["face_count"] == 0
    assert blur["sharpness"] > 70.0
    assert blur["blur"] < 30.0
    assert 40.0 <= brightness["brightness"] <= 70.0
    assert 20.0 <= contrast["contrast"] <= 70.0
    assert noise["noise"] < 20.0


def test_face_detector_requires_independent_cascade_and_preprocessing_consensus():
    detector = FaceDetector()
    box = (100, 80, 60, 60)
    false_single_cascade = [
        (box, "default", "gray"),
        (box, "default", "equalized"),
    ]
    unstable_secondary_vote = [
        (box, "default", "gray"),
        (box, "alt", "gray"),
        (box, "default", "equalized"),
    ]
    stable_secondary_vote = [
        (box, cascade, variant)
        for cascade in ("default", "alt")
        for variant in ("gray", "equalized")
    ]

    assert detector._verified_candidates(false_single_cascade) == []
    assert detector._verified_candidates(unstable_secondary_vote) == []
    verified = detector._verified_candidates(stable_secondary_vote)
    assert len(verified) == 1
    assert verified[0][1] == pytest.approx(2.0 / 3.0)


def test_known_false_landscape_haar_proposal_is_not_reported_as_face():
    image = _local_upload("d6a68dca-9a8a-4b40-8778-6ddd61d79fe2.jpg")

    result = estimate_face_visibility(image)

    assert result["face_count"] == 0
    assert result["face_visibility"] == 0.0
    assert result["confidence"] == 0.0
    assert "unverified" in result["explanation"].lower()


def test_clear_portrait_uses_consensus_face_and_face_focus():
    image = _local_upload("09f89e70-8828-415b-a625-d6c8d8ed099b.jpeg")

    faces = estimate_face_visibility(image)
    blur = estimate_blur(image, faces=faces["faces"], category="Portrait Photography")

    assert faces["face_count"] >= 1
    assert faces["confidence"] >= 2.0 / 3.0
    assert blur["face_sharpness"] is not None
    assert blur["sharpness"] > 50.0


def test_blurred_portrait_face_is_verified_but_not_reported_sharp():
    image = _local_upload("4ae4bd8c-697e-4821-81fc-567b251803e9.jpg")

    faces = estimate_face_visibility(image)
    blur = estimate_blur(image, faces=faces["faces"], category="Portrait Photography")

    assert faces["face_count"] >= 1
    assert faces["confidence"] >= 2.0 / 3.0
    assert blur["face_sharpness"] is not None
    assert blur["sharpness"] < 25.0
    assert blur["blur"] > 75.0


def test_sharp_balanced_landscape_and_severe_motion_blur_are_distinguished():
    sharp_image = _make_rock_tree_landscape()
    sharp = estimate_blur(sharp_image, category="Landscape Photography")
    exposure = estimate_brightness(sharp_image)

    motion_source = np.full((512, 768, 3), 125, dtype=np.uint8)
    for x in range(0, 768, 8):
        shade = 235 if (x // 8) % 2 else 20
        cv2.rectangle(motion_source, (x, 0), (x + 3, 511), (shade, shade, shade), -1)
    motion_kernel = np.ones((1, 41), dtype=np.float32) / 41.0
    motion_blurred = cv2.filter2D(motion_source, -1, motion_kernel)
    motion = estimate_blur(motion_blurred, category="Landscape Photography")

    assert sharp["blur"] < 30.0
    assert 40.0 <= exposure["brightness"] <= 70.0
    assert motion["blur"] > 80.0


def test_high_sensor_noise_is_not_mistaken_for_sharpness():
    rng = np.random.default_rng(42)
    smooth = np.full((256, 384, 3), 125, dtype=np.uint8)
    noisy = np.clip(
        smooth.astype(np.int16) + rng.normal(0, 15, smooth.shape),
        0,
        255,
    ).astype(np.uint8)

    clean_blur = estimate_blur(smooth, category="Landscape Photography")
    noisy_blur = estimate_blur(noisy, category="Landscape Photography")
    noisy_estimate = estimate_noise(noisy)

    assert noisy_estimate["noise"] > 80.0
    assert noisy_blur["sharpness"] < clean_blur["sharpness"] + 30.0


def test_brightness_subject_background_segmentation():
    """Verify that subject_luminance is populated when faces are passed."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 30  # dark background
    # Bright subject region
    img[50:150, 50:150] = 200
    fake_faces = [{"x": 50, "y": 50, "w": 100, "h": 100}]
    res = estimate_brightness(img, faces=fake_faces, category="Portrait Photography")
    assert res["subject_luminance"] is not None
    assert res["background_luminance"] is not None
    # Subject should be brighter than background
    assert res["subject_luminance"] > res["background_luminance"]
    # Effective brightness should be influenced toward subject
    assert res["brightness"] > res["background_luminance"]


def test_contrast_does_not_saturate_on_normal_image():
    """Verify that the updated contrast formula does not return 100 for normal images."""
    # A typical grayscale gradient — should have moderate contrast, not 100%
    gradient = np.tile(np.linspace(40, 200, 200, dtype=np.uint8), (200, 1))
    gradient_bgr = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
    res = estimate_contrast(gradient_bgr)
    assert res["contrast"] < 95.0, f"Contrast saturated at {res['contrast']} for a gradient image"
    assert res["contrast"] > 10.0


def test_noise_returns_new_fields():
    """Verify new noise fields are present."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    res = estimate_noise(img)
    assert "sigma" in res
    assert "smooth_pixel_fraction" in res
    assert res["sigma"] >= 0


def test_reference_portrait_realistic():
    """
    Validates that the known reference portrait (previously misclassified) now
    receives a realistic sharpness score (>= 55%) from the improved estimator.
    This test is skipped if the portrait file is not available.
    """
    import os
    portrait_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "backend", "uploads", "c3b8f6cf-fe5f-4996-9bf0-9b775522fd69.jpeg"
    )
    if not os.path.exists(portrait_path):
        pytest.skip("Reference portrait not found in uploads directory — skipping validation test.")

    img = cv2.imread(portrait_path)
    assert img is not None, "Failed to decode reference portrait"

    face_res = estimate_face_visibility(img)
    faces = face_res.get("faces", [])
    blur_res = estimate_blur(img, faces=faces)

    assert blur_res["sharpness"] >= 55.0, (
        f"Reference portrait sharpness expected >= 55%, got {blur_res['sharpness']:.1f}%. "
        f"Var focal={blur_res.get('var_focal')}, var_subject={blur_res.get('var_subject')}, "
        f"var_global={blur_res.get('var_global')}, effective_var={blur_res.get('effective_var')}"
    )
    assert blur_res["blur"] <= 45.0, (
        f"Reference portrait blur index expected <= 45%, got {blur_res['blur']:.1f}%"
    )
    contrast_res = estimate_contrast(img)
    assert contrast_res["contrast"] < 95.0, (
        f"Contrast saturated at {contrast_res['contrast']:.1f}% for reference portrait"
    )
