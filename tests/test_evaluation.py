"""Unit and integration test suite for clinical evaluation and probability calibration.

Educational research prototype — NOT intended for clinical diagnosis or medical decision-making.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.svm import LinearSVC, SVC

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
from scripts.evaluate import evaluate_all_models


class TestClinicalMetrics:
    """Tests for compute_clinical_metrics calculation, edge cases, and data structure."""

    def test_perfect_predictions(self) -> None:
        """Verify metric outputs when predictions are 100% correct."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([0.05, 0.1, 0.95, 0.98, 0.92])

        metrics = compute_clinical_metrics(y_true, y_pred, y_prob=y_prob)

        assert isinstance(metrics, ClinicalMetrics)
        assert metrics.sensitivity == 1.0
        assert metrics.specificity == 1.0
        assert metrics.precision == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.balanced_accuracy == 1.0
        assert metrics.accuracy == 1.0
        assert metrics.true_positives == 3
        assert metrics.true_negatives == 2
        assert metrics.false_positives == 0
        assert metrics.false_negatives == 0
        assert metrics.total_samples == 5
        assert metrics.roc_auc == 1.0
        assert metrics.brier_score is not None
        assert metrics.brier_score < 0.05
        assert DISCLAIMER in metrics.disclaimer

    def test_inverted_predictions(self) -> None:
        """Verify metric outputs when predictions are completely inverted."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2])

        metrics = compute_clinical_metrics(y_true, y_pred, y_prob=y_prob)

        assert metrics.sensitivity == 0.0
        assert metrics.specificity == 0.0
        assert metrics.precision == 0.0
        assert metrics.f1_score == 0.0
        assert metrics.balanced_accuracy == 0.0
        assert metrics.accuracy == 0.0
        assert metrics.true_positives == 0
        assert metrics.true_negatives == 0
        assert metrics.false_positives == 2
        assert metrics.false_negatives == 2
        assert metrics.total_samples == 4
        assert metrics.roc_auc == 0.0
        assert metrics.brier_score is not None
        assert metrics.brier_score > 0.6

    def test_known_confusion_matrix_calculation(self) -> None:
        """Verify explicit calculations against known theoretical values."""
        # 4 Normal (0), 6 Pneumonia (1)
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        # TN=2, FP=2, FN=1, TP=5
        y_pred = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 0])

        metrics = compute_clinical_metrics(y_true, y_pred)

        assert metrics.true_positives == 5
        assert metrics.false_positives == 2
        assert metrics.true_negatives == 2
        assert metrics.false_negatives == 1
        assert metrics.total_samples == 10

        assert pytest.approx(metrics.sensitivity, abs=1e-5) == 5.0 / 6.0
        assert pytest.approx(metrics.specificity, abs=1e-5) == 2.0 / 4.0
        assert pytest.approx(metrics.precision, abs=1e-5) == 5.0 / 7.0
        expected_f1 = 2.0 * ((5 / 7) * (5 / 6)) / ((5 / 7) + (5 / 6))
        assert pytest.approx(metrics.f1_score, abs=1e-5) == expected_f1
        assert pytest.approx(metrics.balanced_accuracy, abs=1e-5) == (5 / 6 + 2 / 4) / 2.0
        assert pytest.approx(metrics.accuracy, abs=1e-5) == 7.0 / 10.0
        assert metrics.roc_auc is None
        assert metrics.brier_score is None

    def test_probability_input_formats(self) -> None:
        """Verify handling of 1-D probabilities and 2-column probability matrices."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        # 1-D probability format
        prob_1d = np.array([0.1, 0.9, 0.2, 0.8])
        res_1d = compute_clinical_metrics(y_true, y_pred, y_prob=prob_1d)
        assert res_1d.roc_auc == 1.0

        # 2-D probability format (sklearn style: [prob_0, prob_1])
        prob_2d = np.array([
            [0.9, 0.1],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.2, 0.8],
        ])
        res_2d = compute_clinical_metrics(y_true, y_pred, y_prob=prob_2d)
        assert res_2d.roc_auc == 1.0
        assert res_1d.brier_score == res_2d.brier_score

    def test_mapping_and_dictionary_access(self) -> None:
        """Verify dictionary-like and object attribute access."""
        metrics = compute_clinical_metrics([0, 1], [0, 1])
        d = metrics.to_dict()
        assert isinstance(d, dict)
        assert d["sensitivity"] == 1.0
        assert metrics["sensitivity"] == 1.0
        assert "true_positives" in dict(metrics)

    def test_input_validation_errors(self) -> None:
        """Verify proper exceptions on mismatched lengths or empty arrays."""
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_clinical_metrics([0, 1], [0])

        with pytest.raises(ValueError, match="empty arrays"):
            compute_clinical_metrics([], [])

        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_clinical_metrics([0, 1], [0, 1], y_prob=[0.5])

    def test_zero_division_safety(self) -> None:
        """Verify no ZeroDivisionError when predictions have zero positives or negatives."""
        # Model predicts all zeros: TP=0, FP=0
        y_true = np.array([0, 1, 1])
        y_pred = np.array([0, 0, 0])
        metrics = compute_clinical_metrics(y_true, y_pred)
        assert metrics.precision == 0.0
        assert metrics.f1_score == 0.0
        assert metrics.sensitivity == 0.0
        assert metrics.specificity == 1.0


class TestProbabilityCalibration:
    """Tests for zero-leakage Platt scaling and reliability curve computation."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic feature matrices and binary labels."""
        rng = np.random.default_rng(42)
        n_train, n_val, n_test, n_feat = 50, 20, 25, 10

        X_train = rng.normal(0, 1, (n_train, n_feat))
        y_train = np.array([0] * 15 + [1] * 35)

        X_val = rng.normal(0, 1, (n_val, n_feat))
        y_val = np.array([0] * 6 + [1] * 14)

        X_test = rng.normal(0, 1, (n_test, n_feat))
        y_test = np.array([0] * 8 + [1] * 17)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def test_calibration_zero_leakage_pipeline(self, synthetic_data) -> None:
        """Verify CalibratedClassifierCV fits strictly on val split and predicts on test."""
        X_train, y_train, X_val, y_val, X_test, y_test = synthetic_data

        # Train base SVM estimator strictly on train split
        base_svm = LinearSVC(class_weight="balanced", random_state=42, max_iter=2000)
        base_svm.fit(X_train, y_train)

        # Calibrate strictly on validation split (zero leakage)
        calibrated_clf = calibrate_classifier(base_svm, X_val, y_val, method="sigmoid")

        # Predict calibrated probabilities on test split
        probs = predict_calibrated_probabilities(calibrated_clf, X_test)

        assert probs.ndim == 1
        assert len(probs) == len(X_test)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

        # Evaluate clinical metrics with calibrated probabilities
        preds = calibrated_clf.predict(X_test)
        metrics = compute_clinical_metrics(y_test, preds, y_prob=probs)

        assert 0.0 <= metrics.sensitivity <= 1.0
        assert 0.0 <= metrics.specificity <= 1.0
        assert metrics.brier_score is not None
        assert 0.0 <= metrics.brier_score <= 1.0
        assert metrics.roc_auc is not None
        assert 0.0 <= metrics.roc_auc <= 1.0

    def test_calibration_validation_checks(self, synthetic_data) -> None:
        """Verify input validation for single-class validation data."""
        X_train, y_train, X_val, _, _, _ = synthetic_data
        base_svm = LinearSVC(random_state=42).fit(X_train, y_train)

        # Single class validation array should raise ValueError
        y_val_single_class = np.ones(len(X_val))
        with pytest.raises(ValueError, match="must contain both classes"):
            calibrate_classifier(base_svm, X_val, y_val_single_class)

    def test_compute_calibration_curve(self) -> None:
        """Verify calibration curve binning and probability bounds."""
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.85, 0.9, 0.95])

        prob_true, prob_pred = compute_calibration_curve(y_true, y_prob, n_bins=3)

        assert len(prob_true) == len(prob_pred)
        assert np.all(prob_true >= 0.0) and np.all(prob_true <= 1.0)
        assert np.all(prob_pred >= 0.0) and np.all(prob_pred <= 1.0)


class TestEvaluationPlots:
    """Tests for 300 DPI figure rendering and file creation."""

    def test_plot_confusion_matrix(self) -> None:
        """Verify confusion matrix heatmap rendering."""
        cm = np.array([[33, 5], [3, 79]])
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_cm.png"
            fig = plot_confusion_matrix(cm, model_name="hog_linear_svm", output_path=out_file)
            assert out_file.is_file()
            assert out_file.stat().st_size > 5000
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_plot_roc_curves_comparison(self) -> None:
        """Verify combined ROC curve rendering."""
        models_data = {
            "hog_linear_svm": {
                "y_true": np.array([0, 0, 1, 1]),
                "y_prob": np.array([0.1, 0.2, 0.8, 0.9]),
                "auc": 1.0,
            },
            "hog_rbf_svm": {
                "y_true": np.array([0, 0, 1, 1]),
                "y_prob": np.array([0.2, 0.3, 0.7, 0.85]),
                "auc": 0.95,
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_roc.png"
            fig = plot_roc_curves_comparison(models_data, output_path=out_file)
            assert out_file.is_file()
            assert out_file.stat().st_size > 5000
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_plot_calibration_curves(self) -> None:
        """Verify calibration reliability diagram rendering."""
        models_data = {
            "hog_linear_svm": {
                "y_true": np.array([0, 0, 1, 1]),
                "y_prob": np.array([0.1, 0.2, 0.8, 0.9]),
                "brier_score": 0.02,
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_cal.png"
            fig = plot_calibration_curves(models_data, output_path=out_file, n_bins=3)
            assert out_file.is_file()
            assert out_file.stat().st_size > 5000
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_plot_hog_overlay_samples(self) -> None:
        """Verify HOG gradient overlay rendering with synthetic sample generation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_hog_overlay.png"
            fig = plot_hog_overlay_samples(output_path=out_file)
            assert out_file.is_file()
            assert out_file.stat().st_size > 10000
            import matplotlib.pyplot as plt
            plt.close(fig)


class TestEvaluateScript:
    """Tests for the evaluate.py CLI workflow."""

    def test_evaluate_all_models_workflow(self) -> None:
        """Verify end-to-end evaluation runner execution and artifact export."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "results"
            dummy_models = Path(tmp_dir) / "models"
            dummy_features = Path(tmp_dir) / "features"

            df_comp, json_res = evaluate_all_models(
                models_dir=dummy_models,
                feature_dir=dummy_features,
                output_dir=out_dir,
                generate_plots=True,
                dpi=100,  # fast test DPI
            )

            assert len(df_comp) == 3
            assert (out_dir / "metrics" / "model_comparison.csv").is_file()
            assert (out_dir / "metrics" / "model_comparison.json").is_file()
            assert (out_dir / "figures" / "roc_curves_comparison.png").is_file()
            assert (out_dir / "figures" / "calibration_curves.png").is_file()
            assert (out_dir / "figures" / "hog_overlay_samples.png").is_file()
            assert (out_dir / "figures" / "confusion_matrix_hog_linear_svm.png").is_file()

            # Verify mandatory disclaimer in outputs
            csv_text = (out_dir / "metrics" / "model_comparison.csv").read_text(encoding="utf-8")
            assert DISCLAIMER in csv_text
            assert json_res["metadata"]["disclaimer"] == DISCLAIMER
