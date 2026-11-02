"""Evaluation metrics for continuous KonIQ MOS predictions."""

from typing import Dict

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or predicted.shape != actual.shape or actual.size == 0:
        raise ValueError("Actual and predicted values must be non-empty one-dimensional arrays of equal size.")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(predicted)):
        raise ValueError("Evaluation values must all be finite.")
    correlation = spearmanr(actual, predicted).statistic
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": float(mean_squared_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "srcc": float(correlation) if np.isfinite(correlation) else 0.0,
    }
