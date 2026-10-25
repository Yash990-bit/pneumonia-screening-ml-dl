"""Clinical Evaluation Engine and Probability Calibration module.

Educational research prototype — NOT intended for clinical diagnosis or medical decision-making.

This module provides medical screening evaluation metrics and Platt scaling
probability calibration for classical machine learning models (e.g., SVMs)
and deep learning models. In medical screening, sensitivity (recall on the
positive pneumonia class) is paramount to avoid missed diagnoses (false negatives),
while specificity is critical to prevent unnecessary downstream procedures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import warnings
from typing import Any, Mapping

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

DISCLAIMER = (
    "Educational research prototype — NOT intended for clinical diagnosis or medical decision-making."
)


@dataclass(frozen=True)
class ClinicalMetrics(Mapping[str, Any]):
    """Standardized clinical evaluation metrics for pneumonia screening.

    Follows clinical research reporting standards:
    - Sensitivity (Recall): TP / (TP + FN) - detects pneumonia cases.
    - Specificity: TN / (TN + FP) - identifies healthy patients.
    - Precision: TP / (TP + FP) - positive predictive value.
    - F1-Score: harmonic mean of Precision and Sensitivity.
    - Balanced Accuracy: (Sensitivity + Specificity) / 2.
    - Accuracy: (TP + TN) / (TP + TN + FP + FN).
    - ROC-AUC: Area under the Receiver Operating Characteristic curve.
    - Brier Score: Mean squared calibration error for probabilities.
    - Confusion Matrix Counts: Raw integers (TP, FP, TN, FN).
    """

    sensitivity: float
    specificity: float
    precision: float
    f1_score: float
    balanced_accuracy: float
    accuracy: float
    roc_auc: float | None
    brier_score: float | None
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    total_samples: int
    positive_class: str = "PNEUMONIA"
    negative_class: str = "NORMAL"
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a standard serializable dictionary."""
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def __iter__(self):
        return iter(asdict(self))

    def __len__(self) -> int:
        return len(asdict(self))

    def summary(self) -> str:
        """Produce a formatted clinical screening performance summary string."""
        auc_str = f"{self.roc_auc:.4f}" if self.roc_auc is not None else "N/A"
        brier_str = f"{self.brier_score:.4f}" if self.brier_score is not None else "N/A"
        return (
            f"--- Clinical Evaluation Summary ---\n"
            f"Sensitivity (Recall): {self.sensitivity * 100:.2f}%\n"
            f"Specificity:          {self.specificity * 100:.2f}%\n"
            f"Precision:            {self.precision * 100:.2f}%\n"
            f"F1-Score:             {self.f1_score:.4f}\n"
            f"Balanced Accuracy:    {self.balanced_accuracy * 100:.2f}%\n"
            f"Accuracy:             {self.accuracy * 100:.2f}%\n"
            f"ROC-AUC:              {auc_str}\n"
            f"Brier Score:          {brier_str}\n"
            f"Confusion Matrix:     TP={self.true_positives}, FP={self.false_positives}, "
            f"TN={self.true_negatives}, FN={self.false_negatives} (N={self.total_samples})\n"
            f"Note: {self.disclaimer}"
        )


def compute_clinical_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    y_prob: np.ndarray | list[float] | None = None,
    *,
    pos_label: int = 1,
) -> ClinicalMetrics:
    """Compute comprehensive clinical metrics and calibration statistics.

    Parameters
    ----------
    y_true : array-like of shape (N,)
        Ground truth binary labels (0 = NORMAL, 1 = PNEUMONIA).
    y_pred : array-like of shape (N,)
        Predicted binary class labels (0 or 1).
    y_prob : array-like of shape (N,) or (N, 2), optional
        Predicted probabilities for the positive class (or 2-column proba matrix).
        If provided, ROC-AUC and Brier score are computed.
    pos_label : int, default=1
        Label identifying the positive class (PNEUMONIA).

    Returns
    -------
    ClinicalMetrics
        Frozen dataclass containing clinical screening metrics and raw counts.
    """
    y_true_arr = np.asarray(y_true).astype(int).reshape(-1)
    y_pred_arr = np.asarray(y_pred).astype(int).reshape(-1)

    if y_true_arr.size != y_pred_arr.size:
        raise ValueError(
            f"Shape mismatch: y_true has {y_true_arr.size} items, y_pred has {y_pred_arr.size} items."
        )

    if y_true_arr.size == 0:
        raise ValueError("Cannot compute clinical metrics on empty arrays.")

    # Compute 2x2 confusion matrix with fixed labels [0, 1]
    # cm layout: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    total = int(y_true_arr.size)

    # Clinical screening rates
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    if (precision + sensitivity) > 0:
        f1_score = float(2.0 * (precision * sensitivity) / (precision + sensitivity))
    else:
        f1_score = 0.0

    balanced_accuracy = float((sensitivity + specificity) / 2.0)
    accuracy = float((tp + tn) / total) if total > 0 else 0.0

    # Probability-based metrics (ROC-AUC and Brier Score)
    roc_auc: float | None = None
    brier_score: float | None = None

    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob)
        if y_prob_arr.ndim == 2:
            if y_prob_arr.shape[1] == 2:
                # Standard scikit-learn binary predict_proba: col 0 = class 0, col 1 = class 1
                prob_pos = y_prob_arr[:, pos_label].astype(float)
            elif y_prob_arr.shape[1] == 1:
                prob_pos = y_prob_arr[:, 0].astype(float)
            else:
                raise ValueError(f"Unexpected shape for y_prob: {y_prob_arr.shape}")
        elif y_prob_arr.ndim == 1:
            prob_pos = y_prob_arr.astype(float)
        else:
            raise ValueError(f"y_prob must be 1-D or 2-D array; got shape {y_prob_arr.shape}")

        if prob_pos.size != y_true_arr.size:
            raise ValueError(
                f"Shape mismatch: y_true has {y_true_arr.size} items, y_prob has {prob_pos.size} items."
            )

        # ROC-AUC requires at least one sample of each class
        unique_classes = np.unique(y_true_arr)
        if len(unique_classes) > 1:
            try:
                roc_auc = float(roc_auc_score(y_true_arr, prob_pos))
            except Exception:
                roc_auc = None

        try:
            brier_score = float(brier_score_loss(y_true_arr, prob_pos, pos_label=pos_label))
        except Exception:
            brier_score = None

    return ClinicalMetrics(
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        f1_score=f1_score,
        balanced_accuracy=balanced_accuracy,
        accuracy=accuracy,
        roc_auc=roc_auc,
        brier_score=brier_score,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        total_samples=total,
    )


def calibrate_classifier(
    estimator: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    """Calibrate an already-fitted classifier using validation data (Platt scaling).

    Zero-Leakage Constraint:
    The base estimator must have already been trained on the training partition.
    This calibrator is fit strictly on the held-out validation split (``X_val``, ``y_val``).
    Test data is never seen during calibration.

    Parameters
    ----------
    estimator : estimator instance
        A pre-fitted scikit-learn classifier (e.g. LinearSVC or SVC).
    X_val : np.ndarray of shape (n_val_samples, n_features)
        Validation feature matrix.
    y_val : np.ndarray of shape (n_val_samples,)
        Validation ground-truth labels (0 = NORMAL, 1 = PNEUMONIA).
    method : {"sigmoid", "isotonic"}, default="sigmoid"
        Probability calibration method. Platt scaling uses "sigmoid".

    Returns
    -------
    CalibratedClassifierCV
        Calibrated model ready to predict probabilities on test data.
    """
    X_val_arr = np.asarray(X_val)
    y_val_arr = np.asarray(y_val).reshape(-1)

    if X_val_arr.ndim != 2:
        raise ValueError(f"X_val must be a 2-D matrix; got shape {X_val_arr.shape}")
    if X_val_arr.shape[0] != y_val_arr.size:
        raise ValueError(
            f"Mismatched shapes: X_val has {X_val_arr.shape[0]} rows, y_val has {y_val_arr.size} labels."
        )

    unique_classes = set(np.unique(y_val_arr).tolist())
    if unique_classes != {0, 1}:
        raise ValueError(
            f"Validation labels must contain both classes {0, 1} for calibration; got {unique_classes}."
        )

    # Support both scikit-learn >= 1.6/1.8 (FrozenEstimator) and legacy (<1.6 cv="prefit")
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(
            estimator=FrozenEstimator(estimator),
            method=method,
        )
    except ImportError:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            calibrated = CalibratedClassifierCV(
                estimator=estimator,
                method=method,
                cv="prefit",
            )
    calibrated.fit(X_val_arr, y_val_arr)

    return calibrated


def predict_calibrated_probabilities(
    calibrated_clf: CalibratedClassifierCV,
    X: np.ndarray,
) -> np.ndarray:
    """Compute calibrated positive class probabilities for feature matrix X.

    Parameters
    ----------
    calibrated_clf : CalibratedClassifierCV
        Calibrated model produced by ``calibrate_classifier``.
    X : np.ndarray of shape (N, n_features)
        Feature matrix.

    Returns
    -------
    np.ndarray of shape (N,)
        Calibrated probabilities for the positive class (PNEUMONIA, class 1).
    """
    X_arr = np.asarray(X)
    probs_2d = calibrated_clf.predict_proba(X_arr)
    # Return probability of class 1 (index 1)
    return probs_2d[:, 1].astype(float)


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute true positive fraction and mean predicted probability for calibration diagram.

    Parameters
    ----------
    y_true : np.ndarray of shape (N,)
        True binary labels (0 or 1).
    y_prob : np.ndarray of shape (N,)
        Predicted positive class probabilities in range [0, 1].
    n_bins : int, default=10
        Number of bins to partition probabilities.
    strategy : {"uniform", "quantile"}, default="uniform"
        Binning strategy.

    Returns
    -------
    prob_true : np.ndarray
        Empirical proportion of positive samples in each bin.
    prob_pred : np.ndarray
        Mean predicted probability in each bin.
    """
    y_true_arr = np.asarray(y_true).reshape(-1)
    y_prob_arr = np.asarray(y_prob).reshape(-1)
    return calibration_curve(y_true_arr, y_prob_arr, n_bins=n_bins, strategy=strategy)
