"""
Integration tests for SmartSnap Service and Processing Pipeline.
"""

import cv2
import numpy as np
import pytest
from backend.services.analysis_service import AnalysisService


@pytest.fixture(scope="module")
def service():
    return AnalysisService()


def test_service_metadata(service):
    meta = service.fuzzy_engine.get_system_metadata()
    assert "variables" in meta
    assert "rules" in meta
    assert "blur" in meta["variables"]
    assert "quality" in meta["variables"]


def test_service_simulate(service):
    inputs = {
        "blur": 15.0,
        "brightness": 52.0,
        "contrast": 75.0,
        "noise": 10.0,
        "face_visibility": 90.0,
        "composition": 85.0,
    }
    result = service.simulate_fuzzy(inputs, category="Portrait")
    assert "fuzzy" in result
    assert "decision" in result
    assert result["decision"]["recommendation"] == "KEEP"
    assert result["fuzzy"]["defuzzified_score"] >= 75.0


def test_service_analyze_image(service):
    # Generate a realistic test image in memory
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # Background gradient
    for i in range(400):
        img[i, :, :] = int(50 + 100 * (i / 400))
    # Draw circle subject at rule-of-thirds
    cv2.circle(img, (133, 133), 45, (220, 220, 220), -1)

    is_success, buffer = cv2.imencode(".jpg", img)
    assert is_success

    record = service.analyze_image_file(buffer.tobytes(), "test_scene.jpg", category="General Photography")

    assert "id" in record
    assert "features" in record
    assert "fuzzy" in record
    assert "decision" in record
    assert "aggregated_curve" in record["fuzzy"]
    assert "defuzzified_score" in record["fuzzy"]
    assert len(record["features"]) == 6


def test_service_dashboard_and_history(service):
    stats = service.get_dashboard_stats()
    assert "total_analyzed" in stats
    assert stats["total_analyzed"] >= 1

    hist = service.get_history()
    assert len(hist) >= 1
    assert hist[0]["id"] is not None
