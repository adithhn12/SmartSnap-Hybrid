"""Validated joblib model loading and Random Forest MOS inference."""

from pathlib import Path
from typing import Dict, Mapping, Union

import joblib
import numpy as np

from .feature_extractor import FEATURE_NAMES, extract_quality_features

DEFAULT_MODEL_PATH = Path(__file__).with_name("random_forest_model.joblib")


class RandomForestPredictor:
    def __init__(self, model_path: Union[str, Path] = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Trained Random Forest model not found at {self.model_path}. "
                "Train it with: python -m backend.ml.train_model"
            )
        artifact = joblib.load(self.model_path)
        required = {"model", "feature_names", "preprocessing_config", "model_version"}
        missing = required.difference(artifact)
        if missing:
            raise ValueError(f"Model artifact is missing required fields: {sorted(missing)}")
        if tuple(artifact["feature_names"]) != FEATURE_NAMES:
            raise ValueError(
                "Model feature order does not match the inference extractor: "
                f"{artifact['feature_names']} != {FEATURE_NAMES}"
            )
        self.model = artifact["model"]
        self.feature_names = tuple(artifact["feature_names"])
        self.preprocessing_config = artifact["preprocessing_config"]
        self.model_version = str(artifact["model_version"])
        self.evaluation = artifact.get("evaluation")

    @staticmethod
    def load(model_path: Union[str, Path] = DEFAULT_MODEL_PATH) -> "RandomForestPredictor":
        return RandomForestPredictor(model_path)

    def predict(self, features: Mapping[str, float]) -> float:
        missing = [name for name in self.feature_names if name not in features]
        if missing:
            raise ValueError(f"Prediction features are missing: {missing}")
        values = np.asarray([[float(features[name]) for name in self.feature_names]], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Prediction features must all be finite.")
        return float(np.clip(self.model.predict(values)[0], 1.0, 5.0))

    def predict_image(self, image_bgr: np.ndarray) -> Dict[str, object]:
        features: Dict[str, float] = extract_quality_features(image_bgr)
        return {
            "predicted_mos": self.predict(features),
            "features": features,
            "model_version": self.model_version,
        }
