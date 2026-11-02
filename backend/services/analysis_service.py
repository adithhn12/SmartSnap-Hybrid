"""
Analysis Service for SmartSnap.
Bridges Image Processing Feature Extraction with the Mamdani Fuzzy Inference Engine.
Maintains analysis history and image storage.
"""

import os
import uuid
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

if "." in __package__:
    from ..image_processing import (
        estimate_blur,
        estimate_brightness,
        estimate_contrast,
        estimate_noise,
        estimate_face_visibility,
        estimate_composition,
    )
    from ..fuzzy import MamdaniInferenceEngine, DecisionEngine
    from ..ml.predictor import DEFAULT_MODEL_PATH, RandomForestPredictor
else:
    from image_processing import (
        estimate_blur,
        estimate_brightness,
        estimate_contrast,
        estimate_noise,
        estimate_face_visibility,
        estimate_composition,
    )
    from fuzzy import MamdaniInferenceEngine, DecisionEngine
    from ml.predictor import DEFAULT_MODEL_PATH, RandomForestPredictor

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "history.json")
LOGGER = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """Raised when image analysis is requested without a trained model artifact."""


class AnalysisService:
    def __init__(self, model_path: Optional[str] = None):
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        self.fuzzy_engine = MamdaniInferenceEngine()
        self.decision_engine = DecisionEngine()
        self.predictor: Optional[RandomForestPredictor] = None
        self.model_error: Optional[str] = None
        try:
            self.predictor = RandomForestPredictor(model_path or DEFAULT_MODEL_PATH)
        except FileNotFoundError as error:
            self.model_error = str(error)
            LOGGER.warning("Random Forest model unavailable: %s", error)
        self.history: List[Dict[str, Any]] = self._load_history()

    def _load_history(self) -> List[Dict[str, Any]]:
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_history(self):
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
        except Exception:
            LOGGER.exception("Error saving history")

    def analyze_image_file(
        self,
        file_bytes: bytes,
        filename: str,
        category: str = "General Photography"
    ) -> Dict[str, Any]:
        """
        Processes uploaded image file through OpenCV feature extraction and Mamdani fuzzy inference.
        """
        # Decode image from buffer
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise ValueError("Failed to decode uploaded image. Supported formats are JPG, JPEG, and PNG.")

        if self.predictor is None:
            raise ModelUnavailableError(
                self.model_error or "Random Forest model is unavailable; train the model before analysis."
            )

        # RF features remain whole-image statistics, matching MOS training.
        # Run face detection only when the selected category needs face evidence.
        face_applicable = category.casefold() in {"portrait", "group photograph"}
        if face_applicable:
            face_res = estimate_face_visibility(img_bgr)
        else:
            face_res = {
                "face_visibility": 0.0,
                "face_count": 0,
                "faces": [],
                "area_coverage_pct": 0.0,
                "confidence": 0.0,
                "method": "Not evaluated for this category",
                "description": "Not assessed for non-human-focused categories",
                "explanation": (
                    f"Face detection is disabled for {category}; choose Portrait or "
                    "Group Photograph to assess face visibility."
                ),
            }
        detected_faces = face_res.get("faces", [])
        quality_faces = detected_faces if face_applicable else []
        prediction = self.predictor.predict_image(img_bgr)
        predicted_mos = float(prediction["predicted_mos"])

        # Save image to upload dir with UUID
        ext = os.path.splitext(filename)[1].lower()
        if not ext or ext not in [".jpg", ".jpeg", ".png"]:
            ext = ".jpg"
        file_id = str(uuid.uuid4())
        saved_filename = f"{file_id}{ext}"
        saved_path = os.path.join(UPLOAD_DIR, saved_filename)
        with open(saved_path, "wb") as f:
            f.write(file_bytes)

        # 1. Category-aware Computer Vision diagnostics
        h, w = img_bgr.shape[:2]

        blur_res = estimate_blur(img_bgr, faces=quality_faces, category=category)
        bright_res = estimate_brightness(
            img_bgr, faces=quality_faces, category=category
        )
        contrast_res = estimate_contrast(img_bgr)
        noise_res = estimate_noise(img_bgr)
        comp_res = estimate_composition(
            img_bgr,
            faces=quality_faces,
            category=category,
        )

        # Keep MOS as the learned perceptual prior. Blur is supplied separately as
        # explicit, interpretable catastrophic-defect evidence so severe blur cannot
        # disappear behind an otherwise moderate MOS estimate.
        fuzzy_inputs = {
            "predicted_mos": predicted_mos,
            "blur": blur_res["blur"],
            "brightness": bright_res["brightness"],
            "contrast": contrast_res["contrast"],
            "noise": noise_res["noise"],
            "composition": comp_res["composition"],
        }
        if face_applicable:
            fuzzy_inputs["face_visibility"] = face_res["face_visibility"]

        # Existing CV-only rules remain available to the simulation/playground.
        fuzzy_result = self.fuzzy_engine.compute(
            fuzzy_inputs,
            category=category,
            only_provided_inputs=True,
            pipeline="hybrid",
        )

        # 3. Decision Evaluation (KEEP / RECOVER / RETAKE)
        decision_result = self.decision_engine.evaluate(fuzzy_result)

        # Diagnostic panel details
        diagnostics = {
            "global_sharpness": blur_res.get("global_sharpness"),
            "face_sharpness": blur_res.get("face_sharpness"),
            "background_sharpness": blur_res.get("background_sharpness"),
            "focus_discrepancy": blur_res.get("focus_discrepancy"),
            "motion_coherence": blur_res.get("motion_coherence"),
            "blur_memberships": fuzzy_result.get("memberships", {}).get("blur", {}),
            "activated_blur_rules": [
                {
                    "rule_id": r["rule_id"],
                    "name": r["name"],
                    "firing_strength": r["firing_strength"],
                    "consequent": r["consequent"][1],
                }
                for r in fuzzy_result.get("activated_rules", [])
                if any(ant[0] == "blur" for ant in r.get("antecedents", []))
            ],
        }

        # Feature details dictionary with explanation of how they were obtained
        feature_breakdown = {
            "blur": {
                "fuzzy_input_name": "Blur",
                "value": blur_res["blur"],
                "sharpness": blur_res["sharpness"],
                "global_sharpness": blur_res.get("global_sharpness"),
                "face_sharpness": blur_res.get("face_sharpness"),
                "background_sharpness": blur_res.get("background_sharpness"),
                "motion_coherence": blur_res.get("motion_coherence"),
                "focus_discrepancy": blur_res.get("focus_discrepancy"),
                "var_global": blur_res.get("var_global"),
                "var_focal": blur_res.get("var_focal"),
                "var_subject": blur_res.get("var_subject"),
                "var_bg": blur_res.get("var_bg"),
                "raw_stat": (
                    (f"Subject: {blur_res['face_sharpness']:.1f}%, " if blur_res.get("face_sharpness") is not None else "")
                    + f"Global: {blur_res.get('global_sharpness', 0):.1f}%, BG: {blur_res.get('background_sharpness', 0):.1f}%"
                    + (" [Missed Focus!]" if blur_res.get("focus_discrepancy") else "")
                ),
                "description": blur_res["description"],
                "explanation": blur_res["explanation"],
            },
            "brightness": {
                "fuzzy_input_name": "Brightness",
                "value": bright_res["brightness"],
                "global_luminance": bright_res.get("global_luminance"),
                "subject_luminance": bright_res.get("subject_luminance"),
                "background_luminance": bright_res.get("background_luminance"),
                "shadow_clipped_pct": bright_res.get("shadow_clipped_pct"),
                "highlight_clipped_pct": bright_res.get("highlight_clipped_pct"),
                "raw_stat": (
                    f"Effective Lum: {bright_res['brightness']:.1f}%"
                    + (f" (Subject: {bright_res['subject_luminance']:.1f}%" if bright_res.get("subject_luminance") is not None else "")
                    + f", Global: {bright_res.get('global_luminance', 0):.1f}%)"
                ),
                "description": bright_res["description"],
                "explanation": bright_res["explanation"],
            },
            "contrast": {
                "fuzzy_input_name": "Contrast",
                "value": contrast_res["contrast"],
                "rms_contrast": contrast_res.get("rms_contrast"),
                "std_dev": contrast_res.get("std_dev"),
                "dynamic_range_p10_p90": contrast_res.get("dynamic_range_p10_p90"),
                "raw_stat": f"StdDev: {contrast_res.get('std_dev', 0):.1f}, P10-P90 range: {contrast_res.get('dynamic_range_p10_p90', 0):.1f}",
                "description": contrast_res["description"],
                "explanation": contrast_res["explanation"],
            },
            "noise": {
                "fuzzy_input_name": "Noise",
                "value": noise_res["noise"],
                "sigma": noise_res.get("sigma"),
                "smooth_pixel_fraction": noise_res.get("smooth_pixel_fraction"),
                "raw_stat": f"Immerkaer Sigma: {noise_res.get('sigma', 0):.3f} (on {noise_res.get('smooth_pixel_fraction', 0):.0f}% smooth patches)",
                "description": noise_res["description"],
                "explanation": noise_res["explanation"],
            },
            "face_visibility": {
                "fuzzy_input_name": "Face Visibility",
                "value": face_res["face_visibility"],
                "face_count": face_res["face_count"],
                "faces": face_res["faces"],
                "confidence": face_res.get("confidence", 0.0),
                "applicable_to_quality": face_applicable,
                "raw_stat": (
                    f"Faces detected: {face_res['face_count']}"
                    if face_applicable
                    else f"Not assessed ({category})"
                ),
                "description": face_res["description"],
                "explanation": face_res["explanation"],
            },
            "composition": {
                "fuzzy_input_name": "Composition",
                "value": comp_res["composition"],
                "subject_type": comp_res.get("subject_type"),
                "subject_center": comp_res.get("subject_center"),
                "min_anchor_distance": comp_res.get("min_anchor_distance"),
                "min_border_distance": comp_res.get("min_border_distance"),
                "raw_stat": f"Subject: {comp_res.get('subject_type', 'N/A')} @ ({comp_res.get('subject_center', {}).get('x', 0):.2f}, {comp_res.get('subject_center', {}).get('y', 0):.2f})",
                "description": comp_res["description"],
                "explanation": comp_res["explanation"],
            },
        }

        # Compile complete analysis record
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "id": file_id,
            "filename": filename,
            "saved_filename": saved_filename,
            "image_url": f"/api/uploads/{saved_filename}",
            "dimensions": {"width": w, "height": h},
            "category": category,
            "timestamp": timestamp,
            "features": feature_breakdown,
            "random_forest": {
                "predicted_mos": predicted_mos,
                "normalized_quality_score": float(np.round((predicted_mos - 1.0) * 25.0, 2)),
                "model_version": prediction["model_version"],
                "feature_values": prediction["features"],
            },
            "diagnostics": diagnostics,
            "fuzzy": fuzzy_result,
            "decision": decision_result,
        }

        # Prepend to history (keep latest 50)
        self.history.insert(0, record)
        if len(self.history) > 50:
            self.history = self.history[:50]
        self._save_history()

        return record

    def simulate_fuzzy(self, inputs: Dict[str, float], category: str = "General Photography") -> Dict[str, Any]:
        """
        Runs the exact same Mamdani Fuzzy Inference Engine for the simulation playground.
        """
        fuzzy_result = self.fuzzy_engine.compute(inputs, category=category)
        decision_result = self.decision_engine.evaluate(fuzzy_result)
        return {
            "inputs": inputs,
            "category": category,
            "fuzzy": fuzzy_result,
            "decision": decision_result,
        }

    def get_dashboard_stats(self) -> Dict[str, Any]:
        total = len(self.history)
        keep = sum(1 for r in self.history if r.get("decision", {}).get("recommendation") == "KEEP")
        recover = sum(1 for r in self.history if r.get("decision", {}).get("recommendation") == "RECOVER")
        retake = sum(1 for r in self.history if r.get("decision", {}).get("recommendation") == "RETAKE")

        recent = [
            {
                "id": r["id"],
                "filename": r["filename"],
                "saved_filename": r["saved_filename"],
                "image_url": r["image_url"],
                "category": r["category"],
                "score": r["fuzzy"]["defuzzified_score"],
                "recommendation": r["decision"]["recommendation"],
                "timestamp": r["timestamp"],
            }
            for r in self.history[:10]
        ]

        return {
            "total_analyzed": total,
            "keep_count": keep,
            "recover_count": recover,
            "retake_count": retake,
            "recent": recent,
        }

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history

    def get_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        for r in self.history:
            if r["id"] == record_id:
                return r
        return None

    def clear_history(self):
        self.history = []
        self._save_history()
