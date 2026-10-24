"""LBP (Local Binary Pattern) texture feature extraction module.

Extracts micro-texture descriptors from chest X-ray images using
skimage.feature.local_binary_pattern. The uniform LBP histogram captures
diffuse lung opacities (cloudiness) that HOG edge features miss.

Chest X-rays with bacterial pneumonia exhibit focal consolidation and ground-
glass opacity — textural patterns that LBP encodes well as local intensity
orderings, providing complementary information to HOG gradients.

Educational research prototype — NOT a clinical diagnostic tool.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from skimage.feature import local_binary_pattern

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default LBP hyper-parameters (overridden by config.yaml when available)
# ---------------------------------------------------------------------------
_DEFAULT_LBP_PARAMS: dict[str, Any] = {
    "P": 8,           # Number of circularly symmetric neighbour set points
    "R": 1.0,         # Radius of circle (pixels)
    "method": "uniform",  # 'uniform' LBP codes reduce bins from 2^P to P+2
}

# Number of bins in the uniform LBP histogram: P + 2 (P uniform patterns + 1 non-uniform)
_UNIFORM_LBP_BINS = _DEFAULT_LBP_PARAMS["P"] + 2   # = 10 for P=8


def load_lbp_params(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load LBP parameters from a YAML config file.

    Falls back to ``_DEFAULT_LBP_PARAMS`` when the config is absent or lacks
    an ``lbp`` section.

    Parameters
    ----------
    config_path : Path | str | None
        Path to ``configs/config.yaml``. Pass *None* to use defaults.

    Returns
    -------
    dict[str, Any]
        LBP parameter dictionary with keys ``P``, ``R``, ``method``.
    """
    params = dict(_DEFAULT_LBP_PARAMS)

    if config_path is None:
        return params

    cfg_path = Path(config_path)
    if not cfg_path.is_file():
        logger.warning("Config file not found at %s — using default LBP params.", cfg_path)
        return params

    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg: dict = yaml.safe_load(fh) or {}

    lbp_cfg: dict = cfg.get("lbp", {})

    if "P" in lbp_cfg:
        params["P"] = int(lbp_cfg["P"])
    if "R" in lbp_cfg:
        params["R"] = float(lbp_cfg["R"])
    if "method" in lbp_cfg:
        params["method"] = str(lbp_cfg["method"])

    return params


def _lbp_histogram_bins(P: int, method: str) -> int:
    """Return the number of histogram bins for a given LBP configuration.

    For the ``uniform`` method: P + 2 bins (P uniform patterns + 1 catch-all
    for non-uniform patterns).
    For ``default`` / ``ror`` / ``nri_uniform``: 2^P bins.
    """
    if method in ("uniform", "nri_uniform"):
        return P + 2
    return 2 ** P


def extract_lbp_features(
    image: np.ndarray,
    params: dict[str, Any] | None = None,
    config_path: Path | str | None = None,
) -> np.ndarray:
    """Extract a normalised LBP histogram from a single grayscale image.

    The image is converted to uint8 before LBP computation (required by
    skimage when values are floats in [0, 1]).

    Parameters
    ----------
    image : np.ndarray
        2-D float32 array of shape ``(H, W)`` with values in ``[0, 1]``.
    params : dict[str, Any] | None
        LBP parameter dictionary. Falls back to config/defaults if *None*.
    config_path : Path | str | None
        Path to ``configs/config.yaml``. Used only when *params* is *None*.

    Returns
    -------
    np.ndarray
        1-D float32 normalised LBP histogram of length ``P + 2`` (for
        ``method='uniform'``).

    Raises
    ------
    ValueError
        If *image* is not a 2-D array.
    """
    if image.ndim != 2:
        raise ValueError(
            f"LBP extraction expects a 2-D grayscale array; got shape {image.shape}."
        )

    if params is None:
        params = load_lbp_params(config_path)

    P: int = params["P"]
    R: float = float(params["R"])
    method: str = params["method"]
    n_bins: int = _lbp_histogram_bins(P, method)

    # skimage LBP works on uint8 images; scale float [0,1] → [0,255]
    if image.dtype in (np.float32, np.float64):
        img_uint8 = (image * 255.0).astype(np.uint8)
    else:
        img_uint8 = image.astype(np.uint8)

    lbp_image: np.ndarray = local_binary_pattern(img_uint8, P=P, R=R, method=method)

    # Build histogram over [0, n_bins) and normalise to unit sum
    hist, _ = np.histogram(
        lbp_image.ravel(),
        bins=n_bins,
        range=(0, n_bins),
        density=False,
    )
    hist = hist.astype(np.float32)

    total = hist.sum()
    if total > 0:
        hist /= total   # L1 normalisation → probability distribution

    return hist


def compute_lbp_feature_dim(
    params: dict[str, Any] | None = None,
    config_path: Path | str | None = None,
) -> int:
    """Return the length of the LBP feature histogram.

    Parameters
    ----------
    params : dict[str, Any] | None
        LBP parameter dict. Loaded from config/defaults if *None*.
    config_path : Path | str | None
        Path to config YAML.

    Returns
    -------
    int
        Number of histogram bins (``P + 2`` for ``method='uniform'``).
    """
    if params is None:
        params = load_lbp_params(config_path)
    return _lbp_histogram_bins(params["P"], params["method"])
