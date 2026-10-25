#!/usr/bin/env python3
"""Tune and train the Phase 1 class-balanced SVM models.

The runner reads only the train and validation feature caches. It never opens
``features_test_*.npz``; the test split remains reserved for final evaluation.

Examples
--------
python scripts/train_classical.py
python scripts/train_classical.py --models hog_rbf_svm hog_lbp_rbf_svm
python scripts/train_classical.py --feature-dir data/processed --cv-folds 5

Educational research prototype — NOT a clinical diagnostic tool.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from src.models.classical import ModelName

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_classical")

MODEL_FEATURES: dict[str, str] = {
    "hog_linear_svm": "hog",
    "hog_rbf_svm": "hog",
    "hog_lbp_rbf_svm": "combined",
}
ALL_MODELS = list(MODEL_FEATURES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run stratified CV hyperparameter selection for class-balanced SVMs. "
            "Only train and validation caches are read; test data is untouched."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--feature-dir", type=Path, default=PROJECT_ROOT / "data" / "processed",
        help="Directory containing features_{train,val}_{hog,combined}.npz.",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=PROJECT_ROOT / "results" / "models",
        help="Directory for serialized .joblib estimators.",
    )
    parser.add_argument(
        "--metrics-file", type=Path,
        default=PROJECT_ROOT / "results" / "metrics" / "classical_training.json",
        help="JSON summary containing CV and validation results.",
    )
    parser.add_argument(
        "--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS,
        help="Models to run (defaults to all three).",
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="Stratified CV folds.")
    parser.add_argument("--seed", type=int, default=None, help="Override configs/config.yaml seed.")
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Parallel CV workers; 1 limits memory use for RBF SVM searches.",
    )
    return parser


def _random_seed(explicit_seed: int | None) -> int:
    if explicit_seed is not None:
        return explicit_seed
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    if config_path.is_file():
        import yaml

        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        return int(config.get("random_seed", 42))
    return 42


def _load_split(feature_dir: Path, split: str, feature_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load only the requested train or validation cache."""
    import numpy as np

    if split not in {"train", "val"}:
        raise ValueError("The training runner may only load train and val splits.")
    path = feature_dir / f"features_{split}_{feature_name}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature cache not found: {path}. Run scripts/extract_features.py first."
        )
    with np.load(path, allow_pickle=False) as cache:
        if not {"X", "y"}.issubset(cache.files):
            raise ValueError(f"Expected X and y arrays in {path}.")
        X = np.asarray(cache["X"])
        y = np.asarray(cache["y"]).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError(f"Invalid feature/label dimensions in {path}: X={X.shape}, y={y.shape}.")
    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite feature values found in {path}.")
    labels = set(np.unique(y).tolist())
    if labels != {0, 1}:
        raise ValueError(
            f"Labels in {path} must be integer encoded 0=NORMAL and 1=PNEUMONIA; got {labels}."
        )
    return X, y


def _json_value(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import joblib

    from src.models.classical import ModelName, fit_and_select

    seed = _random_seed(args.seed)
    feature_dir = args.feature_dir.resolve()
    model_dir = args.model_dir.resolve()
    metrics_file = args.metrics_file.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Phase 1 SVM tuning: CV folds=%d, seed=%d, scoring=balanced_accuracy", args.cv_folds, seed)
    logger.info("Feature cache directory: %s", feature_dir)
    logger.info("The test split is not loaded or used by this script.")

    records: list[dict[str, Any]] = []
    for model_name in args.models:
        model_name = cast(ModelName, model_name)
        feature_name = MODEL_FEATURES[model_name]
        X_train, y_train = _load_split(feature_dir, "train", feature_name)
        X_val, y_val = _load_split(feature_dir, "val", feature_name)
        logger.info(
            "Running %s with %s features (train=%s, val=%s)",
            model_name, feature_name, X_train.shape, X_val.shape,
        )

        result = fit_and_select(
            model_name,
            X_train,
            y_train,
            X_val,
            y_val,
            random_seed=seed,
            cv_folds=args.cv_folds,
            n_jobs=args.n_jobs,
        )
        artifact_path = model_dir / f"{model_name}.joblib"
        joblib.dump(result.estimator, artifact_path)
        record = {
            "model": model_name,
            "feature_cache": feature_name,
            "class_weight": "balanced",
            "positive_class": "PNEUMONIA",
            "label_encoding": {"0": "NORMAL", "1": "PNEUMONIA"},
            "cv_folds": result.cv_folds,
            "cv_scoring": "balanced_accuracy",
            "best_cv_balanced_accuracy": result.best_cv_score,
            "validation_balanced_accuracy": result.validation_score,
            "best_params": _json_value(result.best_params),
            "search_seconds": result.search_seconds,
            "validation_fit_seconds": result.validation_fit_seconds,
            "final_fit_seconds": result.final_fit_seconds,
            "total_training_seconds": (
                result.search_seconds + result.validation_fit_seconds + result.final_fit_seconds
            ),
            "final_fit_splits": ["train", "val"],
            "artifact": str(artifact_path),
            "test_split_used": False,
        }
        records.append(record)
        logger.info(
            "%s: best=%s, CV balanced accuracy=%.4f, validation balanced accuracy=%.4f, "
            "training=%.1fs, saved=%s",
            model_name,
            result.best_params,
            result.best_cv_score,
            result.validation_score,
            record["total_training_seconds"],
            artifact_path,
        )

    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "cv_folds": args.cv_folds,
        "scoring": "balanced_accuracy",
        "test_split_used": False,
        "medical_research_note": (
            "Educational research prototype; not intended for clinical diagnosis or medical decision-making."
        ),
        "models": records,
    }
    metrics_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("Training summary saved to %s", metrics_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
