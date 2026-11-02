"""
Automated validation of SmartSnap Mamdani Fuzzy Inference Engine.
Tests all 15-20 required scenarios from specification.
"""

import pytest
from backend.fuzzy.inference import MamdaniInferenceEngine
from backend.fuzzy.decision import DecisionEngine


@pytest.fixture(scope="module")
def fuzzy_engine():
    return MamdaniInferenceEngine()


@pytest.fixture(scope="module")
def decision_engine():
    return DecisionEngine()


def test_scenario_1_excellent_photograph(fuzzy_engine, decision_engine):
    # Sharp (low blur), balanced lighting, good contrast, low noise, good composition
    inputs = {
        "blur": 10.0,
        "brightness": 52.0,
        "contrast": 85.0,
        "noise": 8.0,
        "face_visibility": 85.0,
        "composition": 85.0,
    }
    result = fuzzy_engine.compute(inputs, category="General Photography")
    decision = decision_engine.evaluate(result)

    assert result["defuzzified_score"] >= 70.0, f"Expected KEEP score >= 70, got {result['defuzzified_score']}"
    assert decision["recommendation"] == "KEEP"
    assert len(result["activated_rules"]) > 0


def test_scenario_2_extremely_blurry_photograph(fuzzy_engine, decision_engine):
    inputs = {
        "blur": 95.0,
        "brightness": 50.0,
        "contrast": 60.0,
        "noise": 15.0,
        "face_visibility": 50.0,
        "composition": 50.0,
    }
    result = fuzzy_engine.compute(inputs, category="General Photography")
    decision = decision_engine.evaluate(result)

    assert result["defuzzified_score"] < 45.0
    assert decision["recommendation"] == "RETAKE"


def test_scenario_3_very_dark_photograph(fuzzy_engine, decision_engine):
    inputs = {
        "blur": 25.0,
        "brightness": 8.0,
        "contrast": 15.0,
        "noise": 40.0,
        "face_visibility": 10.0,
        "composition": 40.0,
    }
    result = fuzzy_engine.compute(inputs, category="General Photography")
    decision = decision_engine.evaluate(result)

    assert result["defuzzified_score"] < 50.0
    assert decision["recommendation"] in ["RETAKE", "RECOVER"]


def test_scenario_4_overexposed_photograph(fuzzy_engine, decision_engine):
    inputs = {
        "blur": 20.0,
        "brightness": 94.0,
        "contrast": 20.0,
        "noise": 10.0,
        "face_visibility": 20.0,
        "composition": 45.0,
    }
    result = fuzzy_engine.compute(inputs, category="General Photography")
    decision = decision_engine.evaluate(result)

    assert result["defuzzified_score"] < 60.0
    assert decision["recommendation"] in ["RETAKE", "RECOVER"]


def test_scenario_5_high_noise_photograph(fuzzy_engine, decision_engine):
    inputs = {
        "blur": 25.0,
        "brightness": 50.0,
        "contrast": 60.0,
        "noise": 90.0,
        "face_visibility": 50.0,
        "composition": 50.0,
    }
    result = fuzzy_engine.compute(inputs, category="General Photography")
    decision = decision_engine.evaluate(result)

    assert result["defuzzified_score"] < 65.0
    assert any("noise" in p.lower() for p in decision["problems"])


def test_scenario_6_poor_composition(fuzzy_engine, decision_engine):
    inputs = {
        "blur": 15.0,
        "brightness": 52.0,
        "contrast": 70.0,
        "noise": 10.0,
        "face_visibility": 50.0,
        "composition": 10.0,
    }
    result = fuzzy_engine.compute(inputs, category="General Photography")
    decision = decision_engine.evaluate(result)

    # Score should be reduced from excellent
    assert result["defuzzified_score"] < 80.0
    assert any("composition" in p.lower() or "framing" in p.lower() for p in decision["problems"])


def test_scenario_7_clear_portrait(fuzzy_engine, decision_engine):
    inputs = {
        "blur": 12.0,
        "brightness": 54.0,
        "contrast": 75.0,
        "noise": 12.0,
        "face_visibility": 92.0,
        "composition": 85.0,
    }
    result = fuzzy_engine.compute(inputs, category="Portrait Photography")
    decision = decision_engine.evaluate(result)

    assert result["defuzzified_score"] >= 75.0
    assert decision["recommendation"] == "KEEP"


def test_scenario_8_blurry_portrait(fuzzy_engine, decision_engine):
    inputs = {
        "blur": 85.0,
        "brightness": 50.0,
        "contrast": 60.0,
        "noise": 20.0,
        "face_visibility": 20.0,
        "composition": 60.0,
    }
    result = fuzzy_engine.compute(inputs, category="Portrait Photography")
    decision = decision_engine.evaluate(result)

    assert result["defuzzified_score"] < 40.0
    assert decision["recommendation"] == "RETAKE"


def test_scenario_9_good_landscape_without_face(fuzzy_engine, decision_engine):
    # Face visibility is 0, but in Landscape this must NOT penalize the quality
    inputs = {
        "blur": 10.0,
        "brightness": 52.0,
        "contrast": 88.0,
        "noise": 10.0,
        "face_visibility": 0.0,
        "composition": 88.0,
    }
    result = fuzzy_engine.compute(inputs, category="Landscape Photography")
    decision = decision_engine.evaluate(result)

    assert result["defuzzified_score"] >= 75.0
    assert decision["recommendation"] == "KEEP"
    # Ensure no false critical flag about missing faces
    assert not any("face" in flag.lower() for flag in decision["critical_flags"])


def test_scenario_10_good_item_photo(fuzzy_engine, decision_engine):
    inputs = {
        "blur": 8.0,
        "brightness": 55.0,
        "contrast": 90.0,
        "noise": 10.0,
        "object_visibility": 90.0,
        "composition": 70.0,
    }
    result = fuzzy_engine.compute(inputs, category="Item Photography")
    decision = decision_engine.evaluate(result)

    assert result["defuzzified_score"] >= 75.0
    assert decision["recommendation"] == "KEEP"


def test_scenario_11_moderately_blurry_well_exposed(fuzzy_engine, decision_engine):
    inputs = {
        "blur": 48.0,
        "brightness": 53.0,
        "contrast": 65.0,
        "noise": 15.0,
        "face_visibility": 60.0,
        "composition": 60.0,
    }
    result = fuzzy_engine.compute(inputs, category="General Photography")
    decision = decision_engine.evaluate(result)

    # Moderate blur with good exposure should land squarely in RECOVER zone
    assert 45.0 <= result["defuzzified_score"] <= 74.0
    assert decision["recommendation"] == "RECOVER"


def test_scenario_12_sharp_but_poorly_exposed(fuzzy_engine, decision_engine):
    inputs = {
        "blur": 10.0,
        "brightness": 20.0,
        "contrast": 40.0,
        "noise": 15.0,
        "face_visibility": 50.0,
        "composition": 65.0,
    }
    result = fuzzy_engine.compute(inputs, category="General Photography")
    decision = decision_engine.evaluate(result)

    # Sharp but underexposed is prime RECOVER candidate
    assert 45.0 <= result["defuzzified_score"] <= 74.0
    assert decision["recommendation"] == "RECOVER"


def test_scenario_13_good_composition_with_high_noise(fuzzy_engine, decision_engine):
    inputs = {
        "blur": 15.0,
        "brightness": 50.0,
        "contrast": 70.0,
        "noise": 78.0,
        "face_visibility": 60.0,
        "composition": 85.0,
    }
    result = fuzzy_engine.compute(inputs, category="General Photography")
    decision = decision_engine.evaluate(result)

    # High noise reduces quality to moderate/recover
    assert result["defuzzified_score"] < 75.0


def test_scenario_14_multiple_partially_degraded(fuzzy_engine, decision_engine):
    inputs = {
        "blur": 45.0,
        "brightness": 35.0,
        "contrast": 40.0,
        "noise": 45.0,
        "face_visibility": 40.0,
        "composition": 40.0,
    }
    result = fuzzy_engine.compute(inputs, category="General Photography")
    decision = decision_engine.evaluate(result)

    # All metrics are mediocre -> Moderate / RECOVER
    assert 40.0 <= result["defuzzified_score"] <= 65.0


def test_scenario_15_borderline_quality(fuzzy_engine, decision_engine):
    inputs = {
        "blur": 35.0,
        "brightness": 45.0,
        "contrast": 55.0,
        "noise": 30.0,
        "face_visibility": 55.0,
        "composition": 55.0,
    }
    result = fuzzy_engine.compute(inputs, category="General Photography")
    decision = decision_engine.evaluate(result)

    assert 45.0 <= result["defuzzified_score"] <= 76.0


def test_simultaneous_rule_activations_and_determinism(fuzzy_engine):
    inputs = {
        "blur": 28.0,
        "brightness": 48.0,
        "contrast": 60.0,
        "noise": 25.0,
        "face_visibility": 70.0,
        "composition": 65.0,
    }
    # Multiple rules should activate simultaneously in overlapping regions
    res1 = fuzzy_engine.compute(inputs, category="General Photography")
    res2 = fuzzy_engine.compute(inputs, category="General Photography")

    assert len(res1["activated_rules"]) >= 2, "Multiple rules should fire in overlapping zones"
    assert res1["defuzzified_score"] == res2["defuzzified_score"], "Deterministic output required for identical inputs"


def test_no_dead_zones_in_rule_base(fuzzy_engine):
    """
    Verify that no random input combination produces zero activated rules
    (which would cause silent fallback to score=50 without any rule firing).
    """
    import numpy as np
    np.random.seed(42)
    sample_points = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    categories = [
        "Portrait Photography",
        "Item Photography",
        "Landscape Photography",
        "General Photography",
    ]
    dead_zones = []

    for cat in categories:
        for _ in range(150):
            inp = {
                "blur": float(np.random.choice(sample_points)),
                "brightness": float(np.random.choice(sample_points)),
                "contrast": float(np.random.choice(sample_points)),
                "noise": float(np.random.choice(sample_points)),
                "face_visibility": float(np.random.choice(sample_points)),
                "composition": float(np.random.choice(sample_points)),
            }
            res = fuzzy_engine.compute(inp, category=cat)
            if res["activated_count"] == 0:
                dead_zones.append({"category": cat, "inputs": inp})

    assert len(dead_zones) == 0, (
        f"Found {len(dead_zones)} dead-zone combinations with 0 activated rules. "
        f"First 3 examples: {dead_zones[:3]}"
    )


def test_scenario_1_excellent_photograph_keep_threshold_70(fuzzy_engine, decision_engine):
    """Test that KEEP threshold is 70 (updated from 75)."""
    inputs = {
        "blur": 10.0,
        "brightness": 52.0,
        "contrast": 85.0,
        "noise": 8.0,
        "face_visibility": 85.0,
        "composition": 85.0,
    }
    result = fuzzy_engine.compute(inputs, category="General Photography")
    decision = decision_engine.evaluate(result)

    assert result["defuzzified_score"] >= 70.0, f"Expected KEEP score >= 70, got {result['defuzzified_score']}"
    assert decision["recommendation"] == "KEEP"
    assert len(result["activated_rules"]) > 0
    # Verify KEEP threshold is 70
    assert decision["thresholds"]["keep"] == 70.0
