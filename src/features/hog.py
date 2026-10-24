"""HOG (Histogram of Oriented Gradients) feature extraction module.

Extracts gradient and edge-based descriptors from chest X-ray images using
skimage.feature.hog. Parameters are configurable via configs/config.yaml.

HOG captures local edge and gradient structure in cell-level histograms and
is well-suited for detecting structural lung boundaries in X-rays, though it
cannot capture diffuse opacity textures (see lbp.py for that).

Educational research prototype — NOT a clinical diagnostic tool.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from skimage.feature import hog

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default HOG hyper-parameters (overridden by config.yaml when available)
# ---------------------------------------------------------------------------
_DEFAULT_HOG_PARAMS: dict[str, Any] = {
    "orientations": 9,
    "pixels_per_cell": (16, 16),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "transform_sqrt": True,   # Gamma-compression for illumination robustness
    "feature_vector": True,   # Always return 1-D vector
}


def load_hog_params(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load HOG parameters from a YAML config file.

    Falls back to ``_DEFAULT_HOG_PARAMS`` if the config file is missing or
    if it does not contain a ``hog`` section.

    Parameters
    ----------
    config_path : Path | str | None
        Path to ``configs/config.yaml``. If *None*, defaults are returned.

    Returns
    -------
    dict[str, Any]
        Merged HOG parameter dictionary.
    """
    params = dict(_DEFAULT_HOG_PARAMS)

    if config_path is None:
        return params

    cfg_path = Path(config_path)
    if not cfg_path.is_file():
        logger.warning("Config file not found at %s — using default HOG params.", cfg_path)
        return params

    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg: dict = yaml.safe_load(fh) or {}

    hog_cfg: dict = cfg.get("hog", {})

    if "orientations" in hog_cfg:
        params["orientations"] = int(hog_cfg["orientations"])

    if "pixels_per_cell" in hog_cfg:
        ppc = hog_cfg["pixels_per_cell"]
        # Accept both list [16, 16] and already-tuple (16, 16)
        params["pixels_per_cell"] = tuple(int(v) for v in ppc)

    if "cells_per_block" in hog_cfg:
        cpb = hog_cfg["cells_per_block"]
        params["cells_per_block"] = tuple(int(v) for v in cpb)

    if "block_norm" in hog_cfg:
        params["block_norm"] = str(hog_cfg["block_norm"])

    if "transform_sqrt" in hog_cfg:
        params["transform_sqrt"] = bool(hog_cfg["transform_sqrt"])

    return params


def extract_hog_features(
    image: np.ndarray,
    params: dict[str, Any] | None = None,
    config_path: Path | str | None = None,
) -> np.ndarray:
    """Extract HOG feature vector from a single grayscale image.

    Parameters
    ----------
    image : np.ndarray
        2-D float32 array of shape ``(H, W)`` with values in ``[0, 1]``.
    params : dict[str, Any] | None
        HOG parameter dictionary. If *None*, parameters are loaded from
        ``config_path`` or fall back to module defaults.
    config_path : Path | str | None
        Path to ``configs/config.yaml``. Used only when *params* is *None*.

    Returns
    -------
    np.ndarray
        1-D float32 HOG feature vector.

    Raises
    ------
    ValueError
        If *image* is not a 2-D array.
    """
    if image.ndim != 2:
        raise ValueError(
            f"HOG extraction expects a 2-D grayscale array; got shape {image.shape}."
        )

    if params is None:
        params = load_hog_params(config_path)

    feature_vec: np.ndarray = hog(
        image,
        orientations=params["orientations"],
        pixels_per_cell=params["pixels_per_cell"],
        cells_per_block=params["cells_per_block"],
        block_norm=params["block_norm"],
        transform_sqrt=params.get("transform_sqrt", True),
        feature_vector=True,
        channel_axis=None,   # single-channel greyscale
    )
    return feature_vec.astype(np.float32)


def compute_hog_feature_dim(
    image_shape: tuple[int, int] = (224, 224),
    params: dict[str, Any] | None = None,
    config_path: Path | str | None = None,
) -> int:
    """Compute the expected HOG feature vector length without running on real data.

    Useful for pre-allocating feature matrices.

    Parameters
    ----------
    image_shape : tuple[int, int]
        ``(height, width)`` of the input images.
    params : dict[str, Any] | None
        HOG parameter dictionary. Falls back to config/defaults if *None*.
    config_path : Path | str | None
        Path to config YAML (used only when *params* is *None*).

    Returns
    -------
    int
        Number of elements in the HOG feature vector.
    """
    if params is None:
        params = load_hog_params(config_path)

    dummy = np.zeros(image_shape, dtype=np.float32)
    return extract_hog_features(dummy, params=params).shape[0]
