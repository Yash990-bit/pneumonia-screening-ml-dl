"""Feature extraction package.

Provides HOG, LBP, and combined feature extraction pipelines for Phase 1
classical ML experiments. All scalers are fitted on training data only.
"""

from .hog import extract_hog_features, load_hog_params, compute_hog_feature_dim
from .lbp import extract_lbp_features, load_lbp_params, compute_lbp_feature_dim
from .build_features import build_feature_matrices, save_features, load_features

__all__ = [
    # HOG
    "extract_hog_features",
    "load_hog_params",
    "compute_hog_feature_dim",
    # LBP
    "extract_lbp_features",
    "load_lbp_params",
    "compute_lbp_feature_dim",
    # Pipeline
    "build_feature_matrices",
    "save_features",
    "load_features",
]
