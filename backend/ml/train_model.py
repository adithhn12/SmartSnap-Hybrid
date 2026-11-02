"""Reproducibly train and evaluate the KonIQ-10k Random Forest regressor."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from .dataset_loader import load_koniq_records
from .evaluate_model import calculate_metrics
from .feature_extractor import FEATURE_NAMES, PREPROCESSING_CONFIG, extract_quality_features

LOGGER = logging.getLogger("backend.ml.train_model")
MODEL_VERSION = "rf-koniq10k-1.2.0"
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[2] / "datasets"
DEFAULT_MODEL_PATH = Path(__file__).with_name("random_forest_model.joblib")
DEFAULT_RESULTS_PATH = Path(__file__).with_name("training_results.json")


def _read_feature_matrix(
    records: List[Dict[str, object]],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    features: List[List[float]] = []
    targets: List[float] = []
    included_names: List[str] = []
    invalid_images: List[str] = []

    for index, record in enumerate(records, start=1):
        image_path = Path(str(record["image_path"]))
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            invalid_images.append(image_path.name)
            continue
        try:
            extracted = extract_quality_features(image)
        except (ValueError, cv2.error) as error:
            invalid_images.append(f"{image_path.name}: {error}")
            continue
        features.append([extracted[name] for name in FEATURE_NAMES])
        targets.append(float(record["mos"]))
        included_names.append(image_path.name)
        if index % 1000 == 0:
            LOGGER.info("Extracted features for %d/%d matched images", index, len(records))

    if not features:
        raise ValueError("No valid image features could be extracted from the matched dataset.")
    return np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64), included_names, invalid_images


def _feature_distribution_summary(X: np.ndarray) -> Dict[str, Dict[str, float]]:
    quantiles = np.quantile(X, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99], axis=0)
    labels = ("p01", "p05", "p25", "p50", "p75", "p95", "p99")
    return {
        name: {
            label: float(quantiles[index, feature_index])
            for index, label in enumerate(labels)
        }
        for feature_index, name in enumerate(FEATURE_NAMES)
    }


def train_model(
    dataset_root: str = str(DEFAULT_DATASET_ROOT),
    model_path: str = str(DEFAULT_MODEL_PATH),
    results_path: str = str(DEFAULT_RESULTS_PATH),
    n_estimators: int = 180,
    random_state: int = 42,
) -> Dict[str, object]:
    if n_estimators < 2:
        raise ValueError("n_estimators must be at least 2.")
    records, dataset_report = load_koniq_records(dataset_root)
    X, y, image_names, invalid_images = _read_feature_matrix(records)
    if len(y) < 10:
        raise ValueError(f"At least 10 valid annotated images are required; found {len(y)}.")

    train_validation_indices, test_indices = train_test_split(
        np.arange(len(y)), test_size=0.15, random_state=random_state
    )
    train_indices, validation_indices = train_test_split(
        train_validation_indices,
        test_size=0.15 / 0.85,
        random_state=random_state,
    )

    candidates = []
    for trees in sorted({max(40, n_estimators // 2), n_estimators}):
        for min_samples_leaf in (1, 2):
            candidate = RandomForestRegressor(
                n_estimators=trees,
                max_features=1.0,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=-1,
            )
            candidate.fit(X[train_indices], y[train_indices])
            validation_prediction = candidate.predict(X[validation_indices])
            validation_metrics = calculate_metrics(y[validation_indices], validation_prediction)
            candidates.append(
                {
                    "model": candidate,
                    "n_estimators": trees,
                    "min_samples_leaf": min_samples_leaf,
                    "validation_mae": validation_metrics["mae"],
                    "validation_metrics": validation_metrics,
                }
            )

    selected = min(candidates, key=lambda candidate: candidate["validation_mae"])
    train_and_validation_indices = np.concatenate((train_indices, validation_indices))
    final_model = RandomForestRegressor(
        n_estimators=selected["n_estimators"],
        max_features=1.0,
        min_samples_leaf=selected["min_samples_leaf"],
        random_state=random_state,
        n_jobs=-1,
    )
    final_model.fit(X[train_and_validation_indices], y[train_and_validation_indices])

    test_prediction = final_model.predict(X[test_indices])
    test_metrics = calculate_metrics(y[test_indices], test_prediction)
    split_counts = {
        "train": int(len(train_indices)),
        "validation": int(len(validation_indices)),
        "test": int(len(test_indices)),
    }
    results: Dict[str, object] = {
        "model_version": MODEL_VERSION,
        "target": "MOS, continuous human opinion score in [1, 5]",
        "feature_names": list(FEATURE_NAMES),
        "preprocessing_config": PREPROCESSING_CONFIG,
        "random_state": random_state,
        "split_counts": split_counts,
        "training_feature_distributions": _feature_distribution_summary(X[train_indices]),
        "selected_hyperparameters": {
            "n_estimators": selected["n_estimators"],
            "max_features": 1.0,
            "min_samples_leaf": selected["min_samples_leaf"],
        },
        "validation_candidates": [
            {
                "n_estimators": candidate["n_estimators"],
                "min_samples_leaf": candidate["min_samples_leaf"],
                "validation_metrics": candidate["validation_metrics"],
            }
            for candidate in candidates
        ],
        "test_metrics": test_metrics,
        "dataset_validation": dataset_report,
        "matched_images_skipped_as_invalid": invalid_images,
        "valid_images_used": len(y),
        "test_image_names": [image_names[int(index)] for index in test_indices],
    }

    model_file = Path(model_path)
    results_file = Path(results_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": final_model,
        "feature_names": list(FEATURE_NAMES),
        "preprocessing_config": PREPROCESSING_CONFIG,
        "model_version": MODEL_VERSION,
        "target": results["target"],
        "random_state": random_state,
        "selected_hyperparameters": results["selected_hyperparameters"],
        "evaluation": test_metrics,
    }
    joblib.dump(artifact, model_file, compress=3)
    with results_file.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)

    LOGGER.info("Saved model: %s", model_file)
    LOGGER.info("Saved training results: %s", results_file)
    LOGGER.info("Untouched test metrics: %s", json.dumps(test_metrics, sort_keys=True))
    LOGGER.info("Dataset matching report: %s", json.dumps(dataset_report, sort_keys=True))
    LOGGER.info("Skipped invalid matched images: %d", len(invalid_images))
    return results


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--results-path", default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--n-estimators", type=int, default=180)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = train_model(
        dataset_root=args.dataset_root,
        model_path=args.model_path,
        results_path=args.results_path,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )
    print(json.dumps(results["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
