#!/usr/bin/env python3
"""CLI entry point for Phase 1 feature extraction.

Extracts HOG and/or LBP features for all three splits (train / val / test)
and caches the results as compressed .npz files under data/processed/.

Usage examples
--------------
# Extract all three configurations with default 16×16 pixels-per-cell:
python scripts/extract_features.py

# Extract only combined features:
python scripts/extract_features.py --config combined

# Use 8×8 pixels-per-cell (finer HOG):
python scripts/extract_features.py --pixels-per-cell 8 8

# Override config YAML explicitly:
python scripts/extract_features.py --project-config configs/config.yaml

Anti-Leakage Guarantee
-----------------------
StandardScaler is fitted ONLY on the training split and then applied to val
and test without re-fitting. This is enforced in build_features.build_feature_matrices.

Educational research prototype — NOT a clinical diagnostic tool.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Project root bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import (   # noqa: E402
    FeatureConfig,
    build_feature_matrices,
    save_features,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extract_features")

_ALL_CONFIGS: list[FeatureConfig] = ["hog", "lbp", "combined"]


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract HOG / LBP feature matrices for all splits and cache as .npz.\n"
            "Scaler is fit ONLY on the training split (anti-leakage)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--project-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "config.yaml",
        help="Path to configs/config.yaml (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Path to split_manifest.csv. "
            "Defaults to <processed_data>/split_manifest.csv from config.yaml."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to save .npz feature files. "
            "Defaults to <processed_data> from config.yaml."
        ),
    )
    parser.add_argument(
        "--config",
        choices=["hog", "lbp", "combined", "all"],
        default="all",
        help=(
            "Feature configuration to extract. "
            "'all' runs hog, lbp, and combined sequentially (default: all)."
        ),
    )

    # HOG override flags (optional — they override config.yaml values)
    hog_group = parser.add_argument_group("HOG overrides")
    hog_group.add_argument(
        "--orientations",
        type=int,
        default=None,
        help="HOG orientations (default: from config or 9).",
    )
    hog_group.add_argument(
        "--pixels-per-cell",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="HOG pixels_per_cell e.g. --pixels-per-cell 8 8 or 16 16.",
    )
    hog_group.add_argument(
        "--cells-per-block",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="HOG cells_per_block e.g. --cells-per-block 2 2.",
    )
    hog_group.add_argument(
        "--block-norm",
        type=str,
        default=None,
        help="HOG block normalisation method (default: L2-Hys).",
    )

    return parser


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_yaml_config(path: Path) -> dict:
    if not path.is_file():
        logger.warning("Config not found at %s — using defaults.", path)
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_paths(args: argparse.Namespace, cfg: dict) -> tuple[Path, Path]:
    """Resolve manifest and output-dir from args / config / defaults."""
    processed_rel = cfg.get("paths", {}).get("processed_data", "data/processed")
    processed_dir = (PROJECT_ROOT / processed_rel).resolve()

    manifest = args.manifest or (processed_dir / "split_manifest.csv")
    output_dir = args.output_dir or processed_dir

    return Path(manifest), Path(output_dir)


def _apply_cli_hog_overrides(args: argparse.Namespace, config_path: Path) -> Path:
    """If CLI HOG overrides are provided, write a temporary patched config and return its path.

    This avoids mutating the original config.yaml while still letting the
    feature modules load params from a file (so callers stay consistent).
    """
    has_overrides = any([
        args.orientations is not None,
        args.pixels_per_cell is not None,
        args.cells_per_block is not None,
        args.block_norm is not None,
    ])

    if not has_overrides:
        return config_path   # use the original config as-is

    cfg = _load_yaml_config(config_path)
    hog_section: dict = cfg.setdefault("hog", {})

    if args.orientations is not None:
        hog_section["orientations"] = args.orientations
        logger.info("HOG override: orientations=%d", args.orientations)

    if args.pixels_per_cell is not None:
        hog_section["pixels_per_cell"] = list(args.pixels_per_cell)
        logger.info("HOG override: pixels_per_cell=%s", args.pixels_per_cell)

    if args.cells_per_block is not None:
        hog_section["cells_per_block"] = list(args.cells_per_block)
        logger.info("HOG override: cells_per_block=%s", args.cells_per_block)

    if args.block_norm is not None:
        hog_section["block_norm"] = args.block_norm
        logger.info("HOG override: block_norm=%s", args.block_norm)

    # Write patched config to a temp location alongside the original
    patched_path = config_path.parent / "_patched_config.yaml"
    with patched_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, default_flow_style=False)
    logger.info("Patched config written to %s (for this run only).", patched_path)

    return patched_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = _load_yaml_config(args.project_config)
    manifest_path, output_dir = _resolve_paths(args, cfg)
    effective_config_path = _apply_cli_hog_overrides(args, args.project_config)

    configs_to_run: list[FeatureConfig] = (
        _ALL_CONFIGS if args.config == "all" else [args.config]   # type: ignore[list-item]
    )

    logger.info("=" * 60)
    logger.info("Phase 1 — Feature Extraction")
    logger.info("=" * 60)
    logger.info("Manifest      : %s", manifest_path)
    logger.info("Output dir    : %s", output_dir)
    logger.info("Config YAML   : %s", effective_config_path)
    logger.info("Configurations: %s", configs_to_run)
    logger.info("=" * 60)

    if not manifest_path.is_file():
        logger.error(
            "split_manifest.csv not found at %s.\n"
            "Please run: python scripts/prepare_data.py",
            manifest_path,
        )
        return 1

    summary: dict[str, dict] = {}

    for feat_cfg in configs_to_run:
        logger.info("")
        logger.info("──── Running feature config: %s ────", feat_cfg.upper())

        (
            X_train, y_train,
            X_val,   y_val,
            X_test,  y_test,
            scaler,
        ) = build_feature_matrices(
            manifest_path=manifest_path,
            feature_config=feat_cfg,
            config_path=effective_config_path,
            project_root=PROJECT_ROOT,
        )

        saved_paths = save_features(
            output_dir=output_dir,
            feature_config=feat_cfg,
            X_train=X_train, y_train=y_train,
            X_val=X_val,     y_val=y_val,
            X_test=X_test,   y_test=y_test,
            scaler=scaler,
        )

        summary[feat_cfg] = {
            "train_shape": X_train.shape,
            "val_shape":   X_val.shape,
            "test_shape":  X_test.shape,
            "files":       {k: str(v) for k, v in saved_paths.items()},
        }

    # ---------------------------------------------------------------------------
    # Summary report
    # ---------------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("EXTRACTION COMPLETE — SUMMARY")
    logger.info("=" * 60)

    for feat_cfg, info in summary.items():
        logger.info("")
        logger.info("  Config       : %s", feat_cfg.upper())
        logger.info("  Train shape  : %s  (N_train × D)", info["train_shape"])
        logger.info("  Val shape    : %s  (N_val   × D)", info["val_shape"])
        logger.info("  Test shape   : %s  (N_test  × D)", info["test_shape"])
        logger.info("  Saved files  :")
        for k, v in info["files"].items():
            logger.info("    %-8s → %s", k, v)

    logger.info("")
    logger.info(
        "Anti-leakage status: StandardScaler fitted on TRAIN only. "
        "Val and Test were transformed — NEVER fitted."
    )
    logger.info("Features ready for SVM training (Keshav's pipeline).")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
