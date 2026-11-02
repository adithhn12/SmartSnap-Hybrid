import csv

import cv2
import numpy as np
import pytest

from backend.fuzzy import DecisionEngine, MamdaniInferenceEngine
from backend.image_processing import (
    estimate_blur,
    estimate_composition,
    estimate_face_visibility,
)
from backend.ml.dataset_loader import load_koniq_records
from backend.ml.feature_extractor import FEATURE_NAMES, extract_quality_features
from backend.ml.predictor import RandomForestPredictor
from backend.ml.train_model import train_model
from backend.services.analysis_service import AnalysisService, ModelUnavailableError


def _write_dataset(root, count=40):
    image_dir = root / "512x384"
    annotation_dir = root / "annotations"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir()
    rows = []
    for index in range(count):
        image = np.full((48, 64, 3), 30 + index * 4, dtype=np.uint8)
        cv2.rectangle(image, (index % 20, 10), (40, 35), (255 - index * 3, 80, 120), -1)
        success, encoded = cv2.imencode(".jpg", image)
        assert success
        image_name = f"sample_{index:03d}.jpg"
        (image_dir / image_name).write_bytes(encoded.tobytes())
        rows.append({"image_name": image_name, "MOS": f"{1.0 + (index % 40) / 10:.1f}"})

    with (annotation_dir / "koniq10k_scores_and_distributions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_name", "MOS"])
        writer.writeheader()
        writer.writerows(rows)

    with (annotation_dir / "koniq10k_indicators.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_id", "brightness", "deep_feature"])
        writer.writeheader()
        writer.writerow({"image_id": "unrelated-id", "brightness": 0.5, "deep_feature": 99})


def test_dataset_matching_reports_missing_and_unannotated_images(tmp_path):
    _write_dataset(tmp_path)
    (tmp_path / "512x384" / "unannotated.jpg").write_bytes(b"not used")
    (tmp_path / "512x384" / "badmos.jpg").write_bytes(b"not decoded during annotation matching")
    scores_path = tmp_path / "annotations" / "koniq10k_scores_and_distributions.csv"
    with scores_path.open("a", encoding="utf-8") as file:
        file.write("not_present.jpg,3.0\nbadmos.jpg,\n,4.0\n")

    records, report = load_koniq_records(str(tmp_path))

    assert len(records) == 40
    assert report["matched_records"] == 40
    assert report["unmatched_annotations"] == ["not_present.jpg"]
    assert report["unannotated_images"] == ["unannotated.jpg"]
    assert report["invalid_mos"] == ["badmos.jpg: missing or non-numeric MOS"]
    assert report["missing_image_names"] == ["row 44"]
    assert all(record["image_name"] != "not_present.jpg" for record in records)


def test_feature_extractor_returns_finite_fixed_order_features():
    image = np.zeros((96, 128, 3), dtype=np.uint8)
    image[:, 32:96] = (30, 140, 230)

    features = extract_quality_features(image)

    assert tuple(features) == FEATURE_NAMES
    assert all(np.isfinite(value) for value in features.values())
    assert 0.0 <= features["brightness"] <= 100.0
    assert 0.0 <= features["sharpness"] <= 100.0
    with pytest.raises(ValueError):
        extract_quality_features(np.array([], dtype=np.uint8))


def test_rf_sharpness_feature_preserves_the_loaded_model_preprocessing():
    rng = np.random.default_rng(42)
    noisy_image = np.clip(
        np.full((256, 384, 3), 125, dtype=np.int16)
        + rng.normal(0, 15, (256, 384, 3)),
        0,
        255,
    ).astype(np.uint8)

    extracted = extract_quality_features(noisy_image)
    trained_model_sharpness = estimate_blur(
        noisy_image,
        category="General Photography",
        noise_correction=False,
    )["sharpness"]
    corrected_cv_sharpness = estimate_blur(
        noisy_image,
        category="Landscape Photography",
    )["sharpness"]

    assert extracted["sharpness"] == trained_model_sharpness
    assert corrected_cv_sharpness < trained_model_sharpness


def test_inference_uses_the_same_whole_image_feature_extractor():
    image = np.full((128, 128, 3), 110, dtype=np.uint8)
    cv2.rectangle(image, (35, 25), (95, 105), (230, 190, 160), -1)

    class ConstantModel:
        def predict(self, values):
            return np.array([3.0])

    predictor = object.__new__(RandomForestPredictor)
    predictor.model = ConstantModel()
    predictor.feature_names = FEATURE_NAMES
    predictor.model_version = "test"
    prediction = predictor.predict_image(image)

    assert prediction["features"] == extract_quality_features(image)


def test_training_saves_loadable_model_and_predicts_mos(tmp_path):
    _write_dataset(tmp_path)
    model_path = tmp_path / "model.joblib"
    results_path = tmp_path / "results.json"

    results = train_model(
        dataset_root=str(tmp_path),
        model_path=str(model_path),
        results_path=str(results_path),
        n_estimators=4,
        random_state=17,
    )
    predictor = RandomForestPredictor(model_path)
    image = cv2.imread(str(tmp_path / "512x384" / "sample_000.jpg"))

    prediction = predictor.predict_image(image)

    assert model_path.is_file()
    assert results_path.is_file()
    assert sum(results["split_counts"].values()) == 40
    assert set(results["test_metrics"]) == {"mae", "mse", "rmse", "r2", "srcc"}
    assert 1.0 <= prediction["predicted_mos"] <= 5.0
    assert prediction["model_version"] == "rf-koniq10k-1.2.0"


def test_predictor_rejects_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train it with"):
        RandomForestPredictor(tmp_path / "missing.joblib")


def test_hybrid_fuzzy_inference_uses_mos_and_independent_cv_context():
    engine = MamdaniInferenceEngine()
    low = engine.compute(
        {"predicted_mos": 1.4, "blur": 10.0, "face_visibility": 75.0, "composition": 75.0},
        only_provided_inputs=True,
        pipeline="hybrid",
    )
    high = engine.compute(
        {"predicted_mos": 4.7, "blur": 10.0, "face_visibility": 75.0, "composition": 75.0},
        only_provided_inputs=True,
        pipeline="hybrid",
    )

    assert "predicted_mos" in engine.get_system_metadata()["variables"]
    assert "predicted_mos" in high["memberships"]
    assert any(rule["rule_id"].startswith("H") for rule in high["activated_rules"])
    assert high["defuzzified_score"] > low["defuzzified_score"]
    assert "blur" in high["memberships"]


def test_hybrid_mos_ordering_from_mamdani_output():
    engine = MamdaniInferenceEngine()
    scores = []
    for mos in (1.2, 3.0, 4.8):
        fuzzy = engine.compute(
            {"predicted_mos": mos, "blur": 10.0, "face_visibility": 80.0, "composition": 80.0},
            only_provided_inputs=True,
            pipeline="hybrid",
        )
        scores.append(fuzzy["defuzzified_score"])
    assert scores[0] < scores[1] < scores[2]


@pytest.mark.parametrize(
    ("name", "mos", "blur", "face_visibility", "composition", "category", "prior_score"),
    [
        ("landscape", 3.4176, 0.14, 72.0, 65.38, "Landscape Photography", 54.03),
        ("blurred street", 2.6383, 99.37, 43.08, 86.68, "General Photography", 53.25),
        ("sports car", 3.3407, 0.62, 0.0, 59.76, "General Photography", 53.25),
    ],
)
def test_recorded_diagnostic_profiles_use_corroborating_fuzzy_evidence(
    name, mos, blur, face_visibility, composition, category, prior_score
):
    engine = MamdaniInferenceEngine()
    result = engine.compute(
        {
            "predicted_mos": mos,
            "blur": blur,
            "face_visibility": face_visibility,
            "composition": composition,
        },
        category=category,
        only_provided_inputs=True,
        pipeline="hybrid",
    )
    fired_ids = {rule["rule_id"] for rule in result["activated_rules"]}

    if name == "blurred street":
        assert "H1" in fired_ids
        assert result["defuzzified_score"] < prior_score
    else:
        assert "H15" in fired_ids
        assert result["defuzzified_score"] > prior_score


@pytest.mark.parametrize(
    ("name", "inputs", "expected_recommendation"),
    [
        (
            "reported portrait",
            {
                "predicted_mos": 2.6448,
                "blur": 16.73,
                "brightness": 41.51,
                "contrast": 69.94,
                "noise": 3.56,
                "face_visibility": 78.98,
                "composition": 83.15,
            },
            "KEEP",
        ),
        (
            "good portrait",
            {
                "predicted_mos": 3.4,
                "blur": 5,
                "brightness": 52,
                "contrast": 75,
                "noise": 5,
                "face_visibility": 85,
                "composition": 85,
            },
            "KEEP",
        ),
        (
            "moderate portrait",
            {
                "predicted_mos": 3.0,
                "blur": 46,
                "brightness": 45,
                "contrast": 50,
                "noise": 20,
                "face_visibility": 60,
                "composition": 55,
            },
            "RECOVER",
        ),
        (
            "poor portrait",
            {
                "predicted_mos": 2.64,
                "blur": 99.4,
                "brightness": 30,
                "contrast": 20,
                "noise": 80,
                "face_visibility": 20,
                "composition": 20,
            },
            "RETAKE",
        ),
    ],
)
def test_hybrid_portrait_rules_balance_model_and_cv_evidence(
    name, inputs, expected_recommendation
):
    engine = MamdaniInferenceEngine()
    result = engine.compute(
        inputs,
        category="Portrait Photography",
        only_provided_inputs=True,
        pipeline="hybrid",
    )
    decision = DecisionEngine().evaluate(result)
    activated_ids = {rule["rule_id"] for rule in result["activated_rules"]}

    assert decision["recommendation"] == expected_recommendation, name
    assert decision["thresholds"] == {"keep": 70.0, "recover": 45.0}
    if name == "reported portrait":
        assert {"H2", "H3", "H19", "H20"} <= activated_ids
        assert result["defuzzified_score"] >= decision["thresholds"]["keep"]
        assert "Primary caution rule: [H3:" in decision["explanation"]
        assert "Primary supporting rule: [H19:" in decision["explanation"]
        assert "Evidence is mixed:" in decision["explanation"]
    if name == "moderate portrait":
        assert not {"H19", "H20"} & activated_ids
    if name == "poor portrait":
        assert "H1" in activated_ids
        assert not {"H19", "H20"} & activated_ids


@pytest.mark.parametrize(
    ("category", "subject_key"),
    [
        ("Portrait Photography", "face_visibility"),
        ("Item Photography", "object_visibility"),
        ("Landscape Photography", None),
        ("General Photography", None),
    ],
)
@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ("good", "KEEP"),
        ("moderate", "RECOVER"),
        ("poor", "RETAKE"),
    ],
)
def test_category_profiles_cover_good_moderate_and_poor_cases(
    category, subject_key, profile, expected
):
    profile_inputs = {
        "good": {
            "predicted_mos": 3.4,
            "blur": 5,
            "brightness": 52,
            "contrast": 75,
            "noise": 5,
            "face_visibility": 85,
            "object_visibility": 85,
            "composition": 85,
        },
        "moderate": {
            "predicted_mos": 3.0,
            "blur": 46,
            "brightness": 45,
            "contrast": 50,
            "noise": 20,
            "face_visibility": 60,
            "object_visibility": 60,
            "composition": 55,
        },
        "poor": {
            "predicted_mos": 2.4,
            "blur": 95,
            "brightness": 30,
            "contrast": 20,
            "noise": 80,
            "face_visibility": 20,
            "object_visibility": 20,
            "composition": 20,
        },
    }
    inputs = dict(profile_inputs[profile])
    if category != "Portrait Photography":
        inputs.pop("face_visibility")
    if category != "Item Photography":
        inputs.pop("object_visibility")

    result = MamdaniInferenceEngine().compute(
        inputs,
        category=category,
        pipeline="hybrid",
        only_provided_inputs=True,
    )
    decision = DecisionEngine().evaluate(result)

    assert decision["recommendation"] == expected
    assert (subject_key in result["input_values"]) == (subject_key is not None)
    if expected == "KEEP":
        assert result["defuzzified_score"] >= decision["thresholds"]["keep"]
    elif expected == "RETAKE":
        assert result["defuzzified_score"] < decision["thresholds"]["recover"]


def test_unsupported_category_is_rejected():
    with pytest.raises(ValueError, match="Unsupported photography category"):
        MamdaniInferenceEngine().compute(
            {"blur": 10, "composition": 80},
            category="Other Category",
        )


@pytest.mark.parametrize(
    ("category", "subject_key"),
    [
        ("Portrait Photography", "face_visibility"),
        ("Item Photography", "object_visibility"),
    ],
)
def test_exposure_defects_reduce_good_subject_scores(category, subject_key):
    engine = MamdaniInferenceEngine()
    common = {
        "predicted_mos": 3.3,
        "blur": 5,
        "contrast": 15,
        "noise": 5,
        "composition": 85,
        subject_key: 85,
    }
    results = {
        brightness: engine.compute(
            {**common, "brightness": brightness},
            category=category,
            pipeline="hybrid",
            only_provided_inputs=True,
        )
        for brightness in (52, 18, 92)
    }

    assert results[18]["defuzzified_score"] < results[52]["defuzzified_score"]
    assert results[92]["defuzzified_score"] < results[52]["defuzzified_score"]
    assert "H22" in {r["rule_id"] for r in results[18]["activated_rules"]}
    assert "H23" in {r["rule_id"] for r in results[92]["activated_rules"]}


def test_very_low_mos_with_strong_portrait_cv_remains_recover_not_keep():
    result = MamdaniInferenceEngine().compute(
        {
            "predicted_mos": 1.4,
            "blur": 5,
            "brightness": 52,
            "contrast": 75,
            "noise": 5,
            "face_visibility": 85,
            "composition": 85,
        },
        category="Portrait Photography",
        only_provided_inputs=True,
        pipeline="hybrid",
    )
    decision = DecisionEngine().evaluate(result)

    assert any(rule["rule_id"] == "H19" for rule in result["activated_rules"])
    assert any(rule["rule_id"] == "H20" for rule in result["activated_rules"])
    assert result["defuzzified_score"] < decision["thresholds"]["keep"]
    assert decision["recommendation"] == "RECOVER"


def test_severe_blur_uses_mamdani_rules_to_reach_retake_score_band():
    engine = MamdaniInferenceEngine()
    decision_engine = DecisionEngine()
    result = engine.compute(
        {"predicted_mos": 2.64, "blur": 99.4, "face_visibility": 50.0, "composition": 60.0},
        pipeline="hybrid",
        only_provided_inputs=True,
    )
    decision = decision_engine.evaluate(result)

    assert any(rule["rule_id"] == "H1" for rule in result["activated_rules"])
    assert result["defuzzified_score"] < decision_engine.recover_threshold
    assert decision["recommendation"] == "RETAKE"


@pytest.mark.parametrize("category", ["Landscape Photography", "Item Photography"])
def test_non_portrait_hybrid_score_is_independent_of_face_detection(category):
    engine = MamdaniInferenceEngine()
    common = {"predicted_mos": 3.3, "blur": 5.0, "composition": 70.0}
    if category == "Item Photography":
        common["object_visibility"] = 80.0
    results = [
        engine.compute(
            {**common, "face_visibility": face},
            category=category,
            pipeline="hybrid",
            only_provided_inputs=True,
        )
        for face in (0.0, 100.0)
    ]

    assert results[0]["defuzzified_score"] == results[1]["defuzzified_score"]
    assert "face_visibility" not in results[0]["memberships"]
    assert "face_visibility" not in results[0]["input_values"]
    assert not any(rule["rule_id"] in {"H7", "H11", "H16"} for rule in results[0]["activated_rules"])


@pytest.mark.parametrize(
    "category",
    ["Landscape Photography", "Item Photography", "General Photography"],
)
def test_non_human_category_discards_face_input(category):
    engine = MamdaniInferenceEngine()
    common = {"predicted_mos": 3.3, "blur": 12.0, "noise": 8.0, "composition": 70.0}
    if category == "Item Photography":
        common["object_visibility"] = 80.0
    without_face = engine.compute(
        common, category=category, pipeline="hybrid", only_provided_inputs=True
    )
    with_false_face = engine.compute(
        {**common, "face_visibility": 100.0},
        category=category,
        pipeline="hybrid",
        only_provided_inputs=True,
    )

    assert with_false_face["defuzzified_score"] == without_face["defuzzified_score"]
    assert "face_visibility" not in with_false_face["memberships"]
    assert "face_visibility" not in with_false_face["input_values"]


def test_medium_noise_has_an_explicit_mamdani_penalty_without_changing_cutoffs():
    inputs = {
        "blur": 18.0,
        "brightness": 36.0,
        "contrast": 98.0,
        "noise": 57.0,
        "face_visibility": 75.0,
        "composition": 70.0,
    }
    result = MamdaniInferenceEngine().compute(
        inputs, category="General Photography", only_provided_inputs=True
    )
    decision = DecisionEngine().evaluate(result)
    rule = next(rule for rule in result["activated_rules"] if rule["rule_id"] == "R39")

    assert result["memberships"]["noise"]["Medium"] == pytest.approx(0.7083)
    assert rule["firing_strength"] == pytest.approx(0.6021)
    assert result["defuzzified_score"] == pytest.approx(63.29, abs=0.02)
    assert rule["firing_strength"] > next(
        item["firing_strength"]
        for item in result["activated_rules"]
        if item["rule_id"] == "U2"
    )
    assert decision["recommendation"] == "RECOVER"
    assert decision["thresholds"]["keep"] == 70.0


def test_generic_high_contrast_reward_requires_normal_exposure():
    engine = MamdaniInferenceEngine()
    common = {"blur": 18.0, "contrast": 98.0, "noise": 8.0, "composition": 70.0}
    dark = engine.compute(
        {**common, "brightness": 36.0}, only_provided_inputs=True
    )
    balanced = engine.compute(
        {**common, "brightness": 52.0}, only_provided_inputs=True
    )
    dark_u2 = next(
        rule for rule in dark["activated_rules"] if rule["rule_id"] == "U2"
    )
    balanced_u2 = next(
        rule for rule in balanced["activated_rules"] if rule["rule_id"] == "U2"
    )

    assert dark_u2["firing_strength"] < balanced_u2["firing_strength"]
    assert balanced["defuzzified_score"] > dark["defuzzified_score"]


def test_hybrid_high_noise_reduces_score_independently_of_predicted_mos():
    engine = MamdaniInferenceEngine()
    common = {
        "predicted_mos": 3.3,
        "blur": 10.0,
        "composition": 70.0,
        "face_visibility": 0.0,
    }
    clean = engine.compute(
        {**common, "noise": 8.0},
        category="Landscape Photography",
        pipeline="hybrid",
        only_provided_inputs=True,
    )
    noisy = engine.compute(
        {**common, "noise": 95.0},
        category="Landscape Photography",
        pipeline="hybrid",
        only_provided_inputs=True,
    )

    assert any(rule["rule_id"] == "H17" for rule in noisy["activated_rules"])
    assert noisy["defuzzified_score"] < clean["defuzzified_score"]


def test_hybrid_pipeline_requires_all_core_evidence():
    with pytest.raises(ValueError, match="requires inputs"):
        MamdaniInferenceEngine().compute(
            {"predicted_mos": 3.0},
            pipeline="hybrid",
        )


def test_non_portrait_composition_ignores_face_boxes():
    image = np.zeros((160, 240, 3), dtype=np.uint8)
    cv2.rectangle(image, (75, 35), (165, 130), (255, 255, 255), -1)
    false_face = [{"x": 0, "y": 0, "w": 20, "h": 20}]

    landscape = estimate_composition(image, faces=false_face, category="Landscape Photography")
    general = estimate_composition(image, faces=false_face, category="General Photography")
    portrait = estimate_composition(image, faces=false_face, category="Portrait Photography")

    assert landscape["subject_type"] == "Visual Saliency Centroid"
    assert general["subject_type"] == "Visual Saliency Centroid"
    assert landscape["composition"] == general["composition"]
    assert portrait["subject_type"] == "Detected Face"


@pytest.mark.parametrize(
    "category",
    [
        "Item Photography",
        "Landscape Photography",
        "General Photography",
    ],
)
def test_non_human_categories_skip_face_detection_and_report_not_assessed(
    category, monkeypatch
):
    service = AnalysisService()
    if service.predictor is None:
        pytest.skip("Local Random Forest model has not been trained.")

    def unexpected_face_detection(_image):
        pytest.fail("Face detection should be skipped for non-human categories.")

    monkeypatch.setattr(
        "backend.services.analysis_service.estimate_face_visibility",
        unexpected_face_detection,
    )
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.circle(image, (64, 64), 24, (230, 230, 230), -1)
    success, encoded = cv2.imencode(".jpg", image)
    assert success

    result = service.analyze_image_file(
        encoded.tobytes(), "hybrid-test.jpg", category=category
    )

    assert 1.0 <= result["random_forest"]["predicted_mos"] <= 5.0
    assert result["random_forest"]["model_version"] == service.predictor.model_version
    assert "predicted_mos" in result["fuzzy"]["input_values"]
    assert result["fuzzy"]["input_values"]["brightness"] == result["features"]["brightness"]["value"]
    assert result["fuzzy"]["input_values"]["contrast"] == result["features"]["contrast"]["value"]
    assert "brightness" in result["fuzzy"]["memberships"]
    assert "contrast" in result["fuzzy"]["memberships"]
    assert "noise" in result["fuzzy"]["input_values"]
    assert "face_visibility" not in result["fuzzy"]["input_values"]
    assert result["features"]["face_visibility"]["applicable_to_quality"] is False
    if category == "Item Photography":
        assert result["features"]["object_visibility"]["applicable_to_quality"] is True
        assert "object_visibility" in result["fuzzy"]["input_values"]
        if result["features"]["object_visibility"]["object_detected"]:
            assert result["features"]["blur"]["object_sharpness"] is not None
            assert result["features"]["blur"]["raw_stat"].startswith("Object:")
    else:
        assert result["features"]["object_visibility"]["applicable_to_quality"] is False
        assert "object_visibility" not in result["fuzzy"]["input_values"]
    assert result["features"]["face_visibility"]["face_count"] == 0
    assert result["features"]["face_visibility"]["raw_stat"] == f"Not assessed ({category})"
    assert result["features"]["composition"]["subject_type"] != "Detected Face"
    assert "defuzzified_score" in result["fuzzy"]
    assert result["decision"]["recommendation"] in {"KEEP", "RECOVER", "RETAKE"}


def test_portrait_category_runs_face_detection(monkeypatch):
    service = AnalysisService()
    if service.predictor is None:
        pytest.skip("Local Random Forest model has not been trained.")
    detected = {
        "face_visibility": 85.0,
        "face_count": 1,
        "faces": [{"x": 40, "y": 30, "w": 48, "h": 48}],
        "area_coverage_pct": 14.0,
        "confidence": 1.0,
        "method": "test consensus",
        "description": "Clear",
        "explanation": "Test face detection result",
    }
    calls = []

    def verified_face_detection(_image):
        calls.append(True)
        return detected

    monkeypatch.setattr(
        "backend.services.analysis_service.estimate_face_visibility",
        verified_face_detection,
    )
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success

    result = service.analyze_image_file(
        encoded.tobytes(), "portrait.jpg", category="Portrait Photography"
    )

    assert calls == [True]
    assert result["features"]["face_visibility"]["face_count"] == 1
    assert result["features"]["face_visibility"]["raw_stat"] == "Faces detected: 1"
    assert result["features"]["face_visibility"]["applicable_to_quality"] is True
    assert len(result["features"]["blur"]["face_sharpnesses"]) == 1
    assert result["fuzzy"]["input_values"]["brightness"] == result["features"]["brightness"]["value"]
    assert result["fuzzy"]["input_values"]["contrast"] == result["features"]["contrast"]["value"]
    assert "face_visibility" in result["fuzzy"]["input_values"]


def test_api_response_shape_and_invalid_image_handling(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from backend import app as app_module

    class StubService:
        predictor = None
        model_error = "not loaded"

        def analyze_image_file(self, content, filename, category):
            raise ValueError("Failed to decode uploaded image.")

    monkeypatch.setattr(app_module, "service", StubService())
    client = TestClient(app_module.app)

    response = client.post(
        "/api/analyze",
        files={"file": ("invalid.jpg", b"not an image", "image/jpeg")},
        data={"category": "General Photography"},
    )
    assert response.status_code == 400

    validation_service = AnalysisService(model_path=tmp_path / "missing.joblib")
    monkeypatch.setattr(app_module, "service", validation_service)
    response = client.post(
        "/api/simulate",
        json={"inputs": {}, "category": "Unsupported"},
    )
    assert response.status_code == 400
    assert "Unsupported photography category" in response.json()["detail"]

    unavailable_service = AnalysisService(model_path=str(tmp_path / "missing.joblib"))
    monkeypatch.setattr(app_module, "service", unavailable_service)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    response = client.post(
        "/api/analyze",
        files={"file": ("valid.jpg", encoded.tobytes(), "image/jpeg")},
    )
    assert response.status_code == 503

    with pytest.raises(ModelUnavailableError):
        unavailable_service.analyze_image_file(encoded.tobytes(), "valid.jpg")

    class ResponseService:
        predictor = None
        model_error = None

        def analyze_image_file(self, content, filename, category):
            return {
                "random_forest": {"predicted_mos": 3.2, "normalized_quality_score": 55.0},
                "fuzzy": {"defuzzified_score": 61.0},
                "decision": {"recommendation": "RECOVER"},
                "features": {},
            }

    monkeypatch.setattr(app_module, "service", ResponseService())
    response = client.post(
        "/api/analyze",
        files={"file": ("valid.jpg", b"encoded image", "image/jpeg")},
    )
    assert response.status_code == 200
    assert set(response.json()) >= {"random_forest", "fuzzy", "decision", "features"}
