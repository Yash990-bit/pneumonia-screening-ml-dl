"""Classical SVM model selection and training for the Phase 1 experiments.

The final test split is deliberately outside this module's API. Hyperparameters
are selected with stratified cross-validation on training data, checked once on
validation data, and the selected estimator is then refit on train + validation
for later final evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Literal

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import balanced_accuracy_score

ModelName = Literal["hog_linear_svm", "hog_rbf_svm", "hog_lbp_rbf_svm"]

C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
GAMMA_VALUES: list[str | float] = ["scale", "auto", 0.001, 0.01, 0.1]


@dataclass
class ClassicalTrainingResult:
    """Selected estimator and measurements from one training run."""

    name: ModelName
    estimator: SVC | LinearSVC
    best_params: dict[str, Any]
    best_cv_score: float
    validation_score: float
    search_seconds: float
    validation_fit_seconds: float
    final_fit_seconds: float
    cv_folds: int


def make_estimator(name: ModelName) -> SVC | LinearSVC:
    """Create one requested class-balanced SVM architecture."""
    if name == "hog_linear_svm":
        return LinearSVC(class_weight="balanced", max_iter=20_000, random_state=42)
    if name in {"hog_rbf_svm", "hog_lbp_rbf_svm"}:
        return SVC(kernel="rbf", class_weight="balanced")
    raise ValueError(f"Unsupported classical model: {name}")


def parameter_grid(name: ModelName) -> dict[str, list[Any]]:
    """Return the task-specified search space for a model."""
    if name == "hog_linear_svm":
        return {"C": C_VALUES}
    if name in {"hog_rbf_svm", "hog_lbp_rbf_svm"}:
        return {"C": C_VALUES, "gamma": GAMMA_VALUES}
    raise ValueError(f"Unsupported classical model: {name}")


def fit_and_select(
    name: ModelName,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    random_seed: int = 42,
    cv_folds: int = 5,
    n_jobs: int = 1,
) -> ClassicalTrainingResult:
    """Tune on train folds, score on validation, then refit on train + val.

    ``balanced_accuracy`` gives equal weight to Normal and Pneumonia recall,
    which is appropriate for the imbalanced classes in this project. The
    validation arrays are not passed to GridSearchCV.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).reshape(-1)
    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val).reshape(-1)
    if X_train.ndim != 2 or X_val.ndim != 2:
        raise ValueError("Feature arrays must be 2-D matrices.")
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("Train and validation feature dimensions do not match.")
    if X_train.shape[0] != y_train.size or X_val.shape[0] != y_val.size:
        raise ValueError("Each feature row must have exactly one label.")
    if set(np.unique(y_train)) != {0, 1}:
        raise ValueError("Training labels must contain both classes encoded as 0 and 1.")
    if set(np.unique(y_val)) != {0, 1}:
        raise ValueError("Validation labels must contain both classes encoded as 0 and 1.")
    if cv_folds < 2 or np.min(np.unique(y_train, return_counts=True)[1]) < cv_folds:
        raise ValueError("Each training class must have at least cv_folds examples.")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    base_estimator = make_estimator(name)
    if name == "hog_linear_svm":
        base_estimator.set_params(random_state=random_seed)
    search = GridSearchCV(
        estimator=base_estimator,
        param_grid=parameter_grid(name),
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=n_jobs,
        refit=False,
        return_train_score=False,
        error_score="raise",
    )

    search_started = time.perf_counter()
    search.fit(X_train, y_train)
    search_seconds = time.perf_counter() - search_started

    best_params = dict(search.best_params_)
    selected = make_estimator(name).set_params(**best_params)
    if name == "hog_linear_svm":
        selected.set_params(random_state=random_seed)
    validation_fit_started = time.perf_counter()
    selected.fit(X_train, y_train)
    validation_fit_seconds = time.perf_counter() - validation_fit_started
    validation_score = float(balanced_accuracy_score(y_val, selected.predict(X_val)))

    X_final = np.concatenate((X_train, X_val), axis=0)
    y_final = np.concatenate((y_train, y_val), axis=0)
    final_estimator = make_estimator(name).set_params(**best_params)
    if name == "hog_linear_svm":
        final_estimator.set_params(random_state=random_seed)
    fit_started = time.perf_counter()
    final_estimator.fit(X_final, y_final)
    final_fit_seconds = time.perf_counter() - fit_started

    return ClassicalTrainingResult(
        name=name,
        estimator=final_estimator,
        best_params=best_params,
        best_cv_score=float(search.best_score_),
        validation_score=validation_score,
        search_seconds=search_seconds,
        validation_fit_seconds=validation_fit_seconds,
        final_fit_seconds=final_fit_seconds,
        cv_folds=cv_folds,
    )
