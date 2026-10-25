"""Unit and integration tests for classical SVM model selection and training."""

from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.svm import LinearSVC, SVC

from src.models.classical import (
    C_VALUES,
    GAMMA_VALUES,
    fit_and_select,
    make_estimator,
    parameter_grid,
)


class TestClassicalModels:
    """Tests for model definition, parameter grids, and fitting pipeline."""

    def test_make_estimator_configurations(self) -> None:
        """Verify estimator instances and class_weight settings."""
        linear_est = make_estimator("hog_linear_svm")
        assert isinstance(linear_est, LinearSVC)
        assert linear_est.class_weight == "balanced"

        rbf_est = make_estimator("hog_rbf_svm")
        assert isinstance(rbf_est, SVC)
        assert rbf_est.kernel == "rbf"
        assert rbf_est.class_weight == "balanced"

        combined_est = make_estimator("hog_lbp_rbf_svm")
        assert isinstance(combined_est, SVC)
        assert combined_est.kernel == "rbf"
        assert combined_est.class_weight == "balanced"

    def test_make_estimator_unsupported_name(self) -> None:
        """Verify invalid model name raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported classical model"):
            make_estimator("unsupported_random_forest")  # type: ignore[arg-type]

    def test_parameter_grid_definitions(self) -> None:
        """Verify grid search hyperparameters match assignment specifications."""
        grid_linear = parameter_grid("hog_linear_svm")
        assert grid_linear == {"C": C_VALUES}

        grid_rbf = parameter_grid("hog_rbf_svm")
        assert grid_rbf == {"C": C_VALUES, "gamma": GAMMA_VALUES}

        grid_combined = parameter_grid("hog_lbp_rbf_svm")
        assert grid_combined == {"C": C_VALUES, "gamma": GAMMA_VALUES}

    def test_fit_and_select_runs_and_refits(self) -> None:
        """Verify cross-validation parameter selection and refit on train+val."""
        np.random.seed(42)
        X_train = np.random.randn(60, 15).astype(np.float32)
        y_train = np.array([0] * 16 + [1] * 44)  # ~73% class 1 imbalanced
        X_val = np.random.randn(20, 15).astype(np.float32)
        y_val = np.array([0] * 5 + [1] * 15)

        result = fit_and_select(
            "hog_linear_svm",
            X_train,
            y_train,
            X_val,
            y_val,
            random_seed=42,
            cv_folds=3,
            n_jobs=1,
        )

        assert result.name == "hog_linear_svm"
        assert "C" in result.best_params
        assert 0.0 <= result.best_cv_score <= 1.0
        assert 0.0 <= result.validation_score <= 1.0
        assert result.cv_folds == 3
        assert result.search_seconds > 0.0

        # Verify final estimator was refit on 80 samples (60 train + 20 val)
        assert hasattr(result.estimator, "coef_")

    def test_dimension_and_label_validation(self) -> None:
        """Verify input validation for mismatched shapes and invalid labels."""
        X_train = np.zeros((20, 10))
        y_train = np.array([0] * 10 + [1] * 10)
        X_val_mismatch = np.zeros((10, 5))
        y_val = np.array([0] * 5 + [1] * 5)

        with pytest.raises(ValueError, match="dimensions do not match"):
            fit_and_select("hog_linear_svm", X_train, y_train, X_val_mismatch, y_val, cv_folds=2)

        # Single class label should fail
        y_train_single = np.ones(20)
        X_val_10 = np.zeros((10, 10))
        with pytest.raises(ValueError, match="must contain both classes"):
            fit_and_select("hog_linear_svm", X_train, y_train_single, X_val_10, y_val, cv_folds=2)

    def test_estimator_serialization(self) -> None:
        """Verify fitted model can be saved and loaded with joblib."""
        np.random.seed(42)
        X_train = np.random.randn(40, 10).astype(np.float32)
        y_train = np.array([0] * 10 + [1] * 30)
        X_val = np.random.randn(10, 10).astype(np.float32)
        y_val = np.array([0] * 3 + [1] * 7)

        result = fit_and_select(
            "hog_rbf_svm",
            X_train,
            y_train,
            X_val,
            y_val,
            random_seed=42,
            cv_folds=2,
            n_jobs=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "hog_rbf_svm.joblib"
            joblib.dump(result.estimator, model_path)
            assert model_path.is_file()

            loaded_estimator = joblib.load(model_path)
            preds = loaded_estimator.predict(X_val)
            assert preds.shape == (10,)
