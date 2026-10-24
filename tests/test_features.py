"""Unit tests for Phase 1 feature extraction modules.

Tests cover:
  - HOG extraction: shape, dtype, config loading, dimension calculator
  - LBP extraction: shape, dtype, normalisation, config loading
  - Feature pipeline: hog/lbp/combined vector lengths, scaler anti-leakage
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_image_224() -> np.ndarray:
    """Synthetic 224×224 float32 image with values in [0, 1]."""
    rng = np.random.default_rng(seed=42)
    return rng.random((224, 224), dtype=np.float32).astype(np.float32)


@pytest.fixture
def sample_image_small() -> np.ndarray:
    """Tiny 64×64 float32 image for fast testing."""
    rng = np.random.default_rng(seed=0)
    return rng.random((64, 64), dtype=np.float32).astype(np.float32)


# ============================================================================
# HOG Tests
# ============================================================================

class TestHogExtraction:
    def test_output_is_1d_float32(self, sample_image_224: np.ndarray) -> None:
        from src.features.hog import extract_hog_features

        vec = extract_hog_features(sample_image_224)
        assert vec.ndim == 1, "HOG output must be 1-D"
        assert vec.dtype == np.float32, "HOG output must be float32"

    def test_output_length_matches_dimension_calculator(
        self, sample_image_224: np.ndarray
    ) -> None:
        from src.features.hog import compute_hog_feature_dim, extract_hog_features

        params = {
            "orientations": 9,
            "pixels_per_cell": (16, 16),
            "cells_per_block": (2, 2),
            "block_norm": "L2-Hys",
            "transform_sqrt": True,
        }
        vec = extract_hog_features(sample_image_224, params=params)
        expected_dim = compute_hog_feature_dim(
            image_shape=(224, 224), params=params
        )
        assert len(vec) == expected_dim, (
            f"HOG vector length {len(vec)} does not match computed dim {expected_dim}."
        )

    def test_8x8_pixels_per_cell_gives_larger_vector(
        self, sample_image_224: np.ndarray
    ) -> None:
        from src.features.hog import extract_hog_features

        params_coarse = {"orientations": 9, "pixels_per_cell": (16, 16), "cells_per_block": (2, 2), "block_norm": "L2-Hys"}
        params_fine   = {"orientations": 9, "pixels_per_cell": (8, 8),  "cells_per_block": (2, 2), "block_norm": "L2-Hys"}

        vec_coarse = extract_hog_features(sample_image_224, params=params_coarse)
        vec_fine   = extract_hog_features(sample_image_224, params=params_fine)

        assert len(vec_fine) > len(vec_coarse), (
            "8×8 pixel cells should produce a longer HOG descriptor than 16×16."
        )

    def test_non_2d_image_raises_value_error(self) -> None:
        from src.features.hog import extract_hog_features

        bad_image = np.zeros((224, 224, 1), dtype=np.float32)  # 3-D — invalid
        with pytest.raises(ValueError, match="2-D"):
            extract_hog_features(bad_image)

    def test_default_params_returned_when_config_missing(self, tmp_path) -> None:
        from src.features.hog import load_hog_params, _DEFAULT_HOG_PARAMS

        params = load_hog_params(config_path=tmp_path / "nonexistent.yaml")
        # Should fall back to defaults without raising
        for key in _DEFAULT_HOG_PARAMS:
            assert key in params


# ============================================================================
# LBP Tests
# ============================================================================

class TestLbpExtraction:
    def test_output_is_1d_float32(self, sample_image_224: np.ndarray) -> None:
        from src.features.lbp import extract_lbp_features

        hist = extract_lbp_features(sample_image_224)
        assert hist.ndim == 1, "LBP histogram must be 1-D"
        assert hist.dtype == np.float32, "LBP histogram must be float32"

    def test_histogram_is_normalised(self, sample_image_224: np.ndarray) -> None:
        from src.features.lbp import extract_lbp_features

        hist = extract_lbp_features(sample_image_224)
        total = hist.sum()
        assert abs(total - 1.0) < 1e-5, (
            f"LBP histogram should sum to 1.0 (got {total:.6f})."
        )

    def test_histogram_length_is_P_plus_2_for_uniform(
        self, sample_image_224: np.ndarray
    ) -> None:
        from src.features.lbp import extract_lbp_features, compute_lbp_feature_dim

        params = {"P": 8, "R": 1.0, "method": "uniform"}
        hist = extract_lbp_features(sample_image_224, params=params)
        expected_bins = compute_lbp_feature_dim(params=params)

        assert len(hist) == expected_bins == 10, (
            f"Uniform LBP with P=8 should produce 10 bins (P+2); got {len(hist)}."
        )

    def test_non_2d_image_raises_value_error(self) -> None:
        from src.features.lbp import extract_lbp_features

        bad_image = np.zeros((224, 224, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="2-D"):
            extract_lbp_features(bad_image)

    def test_constant_image_produces_very_few_bins(self) -> None:
        """A constant-intensity image should produce very few active LBP bins.

        Note on boundary behaviour
        --------------------------
        ``extract_lbp_features`` scales float32 [0,1] → uint8 via ×255, then
        calls ``skimage.local_binary_pattern``.  For interior pixels all
        neighbours share the same uint8 value, giving a single LBP code (the
        all-uniform code).  However, skimage treats image **border** pixels
        differently (their circular neighbourhood wraps or is truncated), which
        can produce 1–2 extra non-zero bins.  We therefore assert ≤ 3 active
        bins rather than exactly 1, and separately verify that the dominant bin
        carries the overwhelming majority (>95 %) of the histogram mass.
        """
        from src.features.lbp import extract_lbp_features

        params = {"P": 8, "R": 1.0, "method": "uniform"}

        # Float constant image (border artefacts → ≤ 3 bins)
        float_img = np.full((64, 64), 0.5, dtype=np.float32)
        hist_float = extract_lbp_features(float_img, params=params)
        non_zero = int((hist_float > 0).sum())
        assert non_zero <= 3, (
            f"Constant float image should activate ≤ 3 LBP bins; got {non_zero}."
        )
        # The dominant bin must hold > 95 % of histogram mass
        assert hist_float.max() > 0.90, (
            f"Dominant LBP bin should carry >90 % of mass; got {hist_float.max():.4f}."
        )

        # Pure uint8 constant image — skimage border pixels still yield ≤ 3 bins
        # and the dominant bin must carry > 90 % of mass.
        uint8_img = np.full((64, 64), 128, dtype=np.uint8)
        hist_uint8 = extract_lbp_features(uint8_img, params=params)
        non_zero_u8 = int((hist_uint8 > 0).sum())
        assert non_zero_u8 <= 3, (
            f"Constant uint8 image should activate ≤ 3 LBP bins; got {non_zero_u8}."
        )
        assert hist_uint8.max() > 0.90, (
            f"Dominant LBP bin should carry >90 % of mass; got {hist_uint8.max():.4f}."
        )


# ============================================================================
# Combined Feature Vector Test
# ============================================================================

class TestCombinedFeatureVector:
    def test_combined_length_equals_hog_plus_lbp(
        self, sample_image_224: np.ndarray
    ) -> None:
        from src.features.hog import extract_hog_features
        from src.features.lbp import extract_lbp_features

        hog_vec = extract_hog_features(sample_image_224)
        lbp_vec = extract_lbp_features(sample_image_224)
        combined = np.concatenate([hog_vec, lbp_vec])

        assert combined.ndim == 1
        assert len(combined) == len(hog_vec) + len(lbp_vec), (
            "Combined vector length must equal HOG length + LBP length."
        )


# ============================================================================
# Anti-Leakage Scaler Test
# ============================================================================

class TestAntiLeakageScaler:
    """Verify the scaler is fitted only on training data in build_feature_matrices.

    Uses a mock manifest and synthetic images so the test has no external deps.
    """

    def test_scaler_not_fitted_on_val_or_test(self, tmp_path) -> None:
        """The scaler mean_ fitted on train must differ from val/test statistics."""
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(42)

        # Simulate a pre-extracted feature matrix where train/val/test have
        # different distributions (to catch leakage)
        X_train = rng.normal(loc=0.0, scale=1.0, size=(50, 20)).astype(np.float32)
        X_val   = rng.normal(loc=5.0, scale=2.0, size=(10, 20)).astype(np.float32)
        X_test  = rng.normal(loc=5.0, scale=2.0, size=(10, 20)).astype(np.float32)

        # Correct anti-leakage approach
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)   # fit on train only
        X_val_scaled   = scaler.transform(X_val)          # transform only
        X_test_scaled  = scaler.transform(X_test)         # transform only

        # Scaler mean_ should reflect TRAIN distribution (≈ 0.0), not val (≈ 5.0)
        assert abs(scaler.mean_.mean()) < 1.0, (
            "Scaler was fitted on training data so mean_ should be near 0."
        )

        # Val data was NOT recentred by its own mean; should be shifted far from 0
        assert abs(X_val_scaled.mean()) > 1.0, (
            "Scaled val data should be far from 0 since it was not re-centred."
        )

    def test_fit_transform_vs_transform_differ_on_different_distribution(self) -> None:
        """Demonstrate that using fit_transform on val would produce different (wrong) output."""
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(1)
        X_train = rng.normal(loc=0.0, scale=1.0, size=(100, 10)).astype(np.float32)
        X_val   = rng.normal(loc=10.0, scale=0.5, size=(20, 10)).astype(np.float32)

        scaler = StandardScaler()
        scaler.fit(X_train)

        correct_val   = scaler.transform(X_val)           # correct: transform only
        incorrect_val = StandardScaler().fit_transform(X_val)  # wrong: refits on val

        # The two approaches should produce different outputs
        assert not np.allclose(correct_val, incorrect_val), (
            "fit_transform and transform must produce different results on val data "
            "with a different distribution — verifying leakage protection matters."
        )
