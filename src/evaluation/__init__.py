"""Evaluation and clinical screening visualization package.

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
from src.evaluation.plots import (
    plot_calibration_curves,
    plot_confusion_matrix,
    plot_hog_overlay_samples,
    plot_roc_curves_comparison,
)

__all__ = [
    "DISCLAIMER",
    "ClinicalMetrics",
    "calibrate_classifier",
    "compute_calibration_curve",
    "compute_clinical_metrics",
    "predict_calibrated_probabilities",
    "plot_calibration_curves",
    "plot_confusion_matrix",
    "plot_hog_overlay_samples",
    "plot_roc_curves_comparison",
]
