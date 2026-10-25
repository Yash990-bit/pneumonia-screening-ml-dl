"""Evaluation and clinical screening package.

Educational research prototype — NOT intended for clinical diagnosis or medical decision-making.
"""

from src.evaluation.metrics import (
    DISCLAIMER,
    ClinicalMetrics,
    calibrate_classifier,
    compute_calibration_curve,
    compute_clinical_metrics,
    predict_calibrated_probabilities,
)

__all__ = [
    "DISCLAIMER",
    "ClinicalMetrics",
    "calibrate_classifier",
    "compute_calibration_curve",
    "compute_clinical_metrics",
    "predict_calibrated_probabilities",
]
