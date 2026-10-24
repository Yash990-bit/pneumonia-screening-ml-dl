"""Feature concatenation, scaling, and extraction pipeline.

Orchestrates the full classical ML feature engineering pipeline:

  1. Iterates over the train / val / test splits via ``iter_split_images``.
  2. Extracts HOG and/or LBP vectors per image.
  3. Supports three feature configurations:
       - ``hog``      : HOG descriptor only
       - ``lbp``      : LBP histogram only
       - ``combined`` : [HOG vector || LBP vector] concatenated
  4. Fits a ``StandardScaler`` **strictly on training data only**.
  5. Transforms val and test matrices with the fitted scaler.
  6. Caches feature matrices to ``data/processed/`` as compressed ``.npz`` files.

Anti-Leakage Contract
---------------------
``fit`` / ``fit_transform`` is called **only** on train split data.
Val and test data are transformed with the already-fitted scaler.
This is enforced by the public ``build_feature_matrices`` function and
explicitly documented at each call-site.

Educational research prototype — NOT a clinical diagnostic tool.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler

from src.data.preprocess import iter_split_images
from src.features.hog import extract_hog_features, load_hog_params
from src.features.lbp import extract_lbp_features, load_lbp_params

logger = logging.getLogger(__name__)

# Allowed feature configuration names
FeatureConfig = Literal["hog", "lbp", "combined"]

# File name template for cached split matrices
_CACHE_NAME_TEMPLATE = "features_{split}_{config}.npz"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: Path | str | None) -> dict:
    """Load YAML config or return empty dict if path is None / missing."""
    if config_path is None:
        return {}
    p = Path(config_path)
    if not p.is_file():
        logger.warning("Config not found at %s — using defaults for all params.", p)
        return {}
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _extract_single(
    img: np.ndarray,
    feature_config: FeatureConfig,
    hog_params: dict,
    lbp_params: dict,
) -> np.ndarray:
    """Extract a feature vector from one image according to the chosen config."""
    if feature_config == "hog":
        return extract_hog_features(img, params=hog_params)
    elif feature_config == "lbp":
        return extract_lbp_features(img, params=lbp_params)
    elif feature_config == "combined":
        hog_vec = extract_hog_features(img, params=hog_params)
        lbp_vec = extract_lbp_features(img, params=lbp_params)
        return np.concatenate([hog_vec, lbp_vec], axis=0)
    else:
        raise ValueError(
            f"Unknown feature_config '{feature_config}'. "
            "Choose from: 'hog', 'lbp', 'combined'."
        )


def _extract_split(
    manifest_path: Path,
    split: str,
    feature_config: FeatureConfig,
    hog_params: dict,
    lbp_params: dict,
    project_root: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix and label array for one split.

    Parameters
    ----------
    manifest_path : Path
        Absolute path to ``split_manifest.csv``.
    split : str
        One of ``'train'``, ``'val'``, ``'test'``.
    feature_config : FeatureConfig
        Which features to compute.
    hog_params, lbp_params : dict
        Pre-loaded parameter dictionaries.
    project_root : Path | None
        Passed to ``iter_split_images`` for path resolution.

    Returns
    -------
    X : np.ndarray of shape (N, D)
        Feature matrix (float32).
    y : np.ndarray of shape (N,)
        Integer label array — 1 for PNEUMONIA, 0 for NORMAL.
    """
    features_list: list[np.ndarray] = []
    labels_list: list[int] = []

    t0 = time.perf_counter()
    count = 0

    for img_arr, label_str, _path in iter_split_images(
        manifest_path=manifest_path,
        split=split,
        project_root=project_root,
    ):
        vec = _extract_single(img_arr, feature_config, hog_params, lbp_params)
        features_list.append(vec)
        labels_list.append(1 if label_str.upper() == "PNEUMONIA" else 0)
        count += 1

        if count % 200 == 0:
            elapsed = time.perf_counter() - t0
            logger.info(
                "  [%s | %s] %d images processed (%.1fs elapsed)",
                split.upper(), feature_config, count, elapsed,
            )

    if not features_list:
        raise RuntimeError(
            f"No images found for split='{split}' in manifest at {manifest_path}. "
            "Check that split_manifest.csv exists and has the expected rows."
        )

    X = np.stack(features_list, axis=0).astype(np.float32)
    y = np.array(labels_list, dtype=np.int32)

    elapsed = time.perf_counter() - t0
    logger.info(
        "[%s | %s] Extracted %d images → X shape %s in %.1fs",
        split.upper(), feature_config, count, X.shape, elapsed,
    )
    return X, y


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_feature_matrices(
    manifest_path: Path | str,
    feature_config: FeatureConfig = "combined",
    config_path: Path | str | None = None,
    project_root: Path | None = None,
) -> tuple[
    np.ndarray, np.ndarray,   # X_train, y_train
    np.ndarray, np.ndarray,   # X_val,   y_val
    np.ndarray, np.ndarray,   # X_test,  y_test
    StandardScaler,           # fitted scaler
]:
    """Extract, scale, and return feature matrices for all three splits.

    **Anti-leakage guarantee**: ``StandardScaler.fit`` is called **only** on
    ``X_train``. Val and test sets are transformed (not fitted) with the same
    scaler object.

    Parameters
    ----------
    manifest_path : Path | str
        Path to ``data/processed/split_manifest.csv``.
    feature_config : {'hog', 'lbp', 'combined'}
        Which feature descriptor(s) to compute.
    config_path : Path | str | None
        Path to ``configs/config.yaml`` for HOG/LBP hyper-parameters.
        If *None*, module defaults are used.
    project_root : Path | None
        Project root for resolving relative image paths in the manifest.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test, scaler
        Scaled feature matrices (float32), integer label arrays, and the
        fitted scaler instance.
    """
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Split manifest not found at: {manifest_path}\n"
            "Run `python scripts/prepare_data.py` first to generate it."
        )

    hog_params = load_hog_params(config_path)
    lbp_params = load_lbp_params(config_path)

    logger.info("Feature config : %s", feature_config)
    logger.info("HOG params     : %s", hog_params)
    logger.info("LBP params     : %s", lbp_params)

    # -----------------------------------------------------------------------
    # Step 1: Extract raw (unscaled) features for all splits
    # -----------------------------------------------------------------------
    logger.info("Extracting TRAIN features …")
    X_train, y_train = _extract_split(
        manifest_path, "train", feature_config, hog_params, lbp_params, project_root
    )

    logger.info("Extracting VAL features …")
    X_val, y_val = _extract_split(
        manifest_path, "val", feature_config, hog_params, lbp_params, project_root
    )

    logger.info("Extracting TEST features …")
    X_test, y_test = _extract_split(
        manifest_path, "test", feature_config, hog_params, lbp_params, project_root
    )

    # -----------------------------------------------------------------------
    # Step 2: Fit scaler ONLY on training data — then transform all splits
    #
    # ANTI-LEAKAGE RULE:
    #   scaler.fit()           → TRAIN ONLY
    #   scaler.transform()     → val, test (never fit_transform)
    # -----------------------------------------------------------------------
    logger.info(
        "Fitting StandardScaler on TRAIN split only (N=%d, D=%d) …",
        X_train.shape[0], X_train.shape[1],
    )
    scaler = StandardScaler()
    X_train_scaled: np.ndarray = scaler.fit_transform(X_train)   # fit + transform train
    X_val_scaled: np.ndarray   = scaler.transform(X_val)          # transform only — NO fit
    X_test_scaled: np.ndarray  = scaler.transform(X_test)         # transform only — NO fit

    logger.info(
        "Scaling complete. Train %s | Val %s | Test %s",
        X_train_scaled.shape, X_val_scaled.shape, X_test_scaled.shape,
    )

    return (
        X_train_scaled, y_train,
        X_val_scaled,   y_val,
        X_test_scaled,  y_test,
        scaler,
    )


def save_features(
    output_dir: Path | str,
    feature_config: FeatureConfig,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    scaler: StandardScaler,
) -> dict[str, Path]:
    """Save scaled feature matrices and the fitted scaler to disk.

    Matrices are saved as compressed ``.npz`` files under ``output_dir``.
    The scaler is serialised with ``joblib``.

    Parameters
    ----------
    output_dir : Path | str
        Destination directory (``data/processed/`` by convention).
    feature_config : FeatureConfig
        Used to build output file names (e.g., ``features_train_combined.npz``).
    X_train, y_train, X_val, y_val, X_test, y_test : np.ndarray
        Scaled matrices and label arrays.
    scaler : StandardScaler
        The scaler fitted on training data.

    Returns
    -------
    dict[str, Path]
        Mapping of ``{split: path}`` for each saved ``.npz`` file, plus
        ``'scaler'`` for the serialised scaler.
    """
    import joblib

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}

    for split_name, X, y in [
        ("train", X_train, y_train),
        ("val",   X_val,   y_val),
        ("test",  X_test,  y_test),
    ]:
        fname = _CACHE_NAME_TEMPLATE.format(split=split_name, config=feature_config)
        fpath = out_dir / fname
        np.savez_compressed(fpath, X=X, y=y)
        saved[split_name] = fpath
        logger.info("Saved %s → %s  (X=%s, y=%s)", split_name, fpath, X.shape, y.shape)

    # Save the fitted scaler so downstream consumers can inverse-transform or inspect it
    scaler_path = out_dir / f"scaler_{feature_config}.joblib"
    joblib.dump(scaler, scaler_path)
    saved["scaler"] = scaler_path
    logger.info("Saved scaler → %s", scaler_path)

    return saved


def load_features(
    output_dir: Path | str,
    feature_config: FeatureConfig,
) -> tuple[
    np.ndarray, np.ndarray,   # X_train, y_train
    np.ndarray, np.ndarray,   # X_val,   y_val
    np.ndarray, np.ndarray,   # X_test,  y_test
    StandardScaler,           # fitted scaler
]:
    """Load cached feature matrices from disk.

    Parameters
    ----------
    output_dir : Path | str
        Directory where features were previously saved (``data/processed/``).
    feature_config : FeatureConfig
        Must match the ``feature_config`` used during ``save_features``.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test, scaler

    Raises
    ------
    FileNotFoundError
        If any expected cache file is missing.
    """
    import joblib

    out_dir = Path(output_dir)

    def _load_npz(split: str) -> tuple[np.ndarray, np.ndarray]:
        fname = _CACHE_NAME_TEMPLATE.format(split=split, config=feature_config)
        fpath = out_dir / fname
        if not fpath.is_file():
            raise FileNotFoundError(
                f"Feature cache not found: {fpath}. "
                "Run `python scripts/extract_features.py` first."
            )
        data = np.load(fpath)
        return data["X"], data["y"]

    X_train, y_train = _load_npz("train")
    X_val,   y_val   = _load_npz("val")
    X_test,  y_test  = _load_npz("test")

    scaler_path = out_dir / f"scaler_{feature_config}.joblib"
    if not scaler_path.is_file():
        raise FileNotFoundError(
            f"Scaler not found: {scaler_path}. "
            "Run `python scripts/extract_features.py` first."
        )
    scaler: StandardScaler = joblib.load(scaler_path)

    logger.info(
        "Loaded cached features [%s]: train=%s val=%s test=%s",
        feature_config, X_train.shape, X_val.shape, X_test.shape,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler
