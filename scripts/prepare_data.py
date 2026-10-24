#!/usr/bin/env python3
"""Command-line script to generate reproducible, leakage-free stratified splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import build_stratified_manifest, save_manifest


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build reproducible stratified splits and generate split_manifest.csv."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "config.yaml",
        help="Path to project config YAML (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of training set to allocate to validation (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified splitting (default: 42)",
    )
    return parser


def print_split_summary(manifest_df) -> None:
    print("\n" + "=" * 55)
    print("STRATIFIED SPLIT GENERATION SUMMARY (LEAKAGE-FREE)")
    print("=" * 55)
    total = len(manifest_df)
    print(f"Total images cataloged: {total}")

    for split_name in ("train", "val", "test"):
        split_subset = manifest_df[manifest_df["split"] == split_name]
        class_counts = split_subset["label"].value_counts().to_dict()
        norm_count = class_counts.get("NORMAL", 0)
        pneu_count = class_counts.get("PNEUMONIA", 0)
        split_total = len(split_subset)
        pct_pneu = (pneu_count / split_total * 100) if split_total > 0 else 0

        tag = " [ORIGINAL TEST SET - UNTOUCHED]" if split_name == "test" else ""
        print(f"\n[{split_name.upper()} SET]{tag}")
        print(f"  Total Images : {split_total:,} ({split_total / total * 100:.1f}%)")
        print(f"  NORMAL       : {norm_count:,} ({norm_count / split_total * 100:.1f}%)")
        print(f"  PNEUMONIA    : {pneu_count:,} ({pct_pneu:.1f}%)")

    # Verify zero leakage across splits
    train_hashes = set(manifest_df[manifest_df["split"] == "train"]["sha256"])
    val_hashes = set(manifest_df[manifest_df["split"] == "val"]["sha256"])
    test_hashes = set(manifest_df[manifest_df["split"] == "test"]["sha256"])

    leak_train_val = len(train_hashes.intersection(val_hashes))
    leak_train_test = len(train_hashes.intersection(test_hashes))
    leak_val_test = len(val_hashes.intersection(test_hashes))

    print("\n" + "-" * 55)
    print("DATA LEAKAGE AUDIT VERIFICATION:")
    print(f"  Train ∩ Val exact duplicate hashes  : {leak_train_val}")
    print(f"  Train ∩ Test exact duplicate hashes : {leak_train_test}")
    print(f"  Val ∩ Test exact duplicate hashes   : {leak_val_test}")

    if leak_train_val == 0 and leak_train_test == 0 and leak_val_test == 0:
        print("  Status: ZERO DATA LEAKAGE CONFIRMED! ALL SPLITS FULLY ISOLATED.")
    else:
        print("  WARNING: Cross-split duplicates detected!")
    print("=" * 55 + "\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    dataset_rel = config.get("paths", {}).get("dataset", "data/raw/chest_xray")
    processed_rel = config.get("paths", {}).get("processed_data", "data/processed")

    dataset_root = (PROJECT_ROOT / dataset_rel).resolve()
    output_dir = (PROJECT_ROOT / processed_rel).resolve()
    manifest_path = output_dir / "split_manifest.csv"

    print(f"Reading dataset from : {dataset_root}")
    print(f"Using random seed    : {args.seed}")
    print(f"Validation ratio     : {args.val_ratio * 100:.1f}% of train set")

    manifest_df = build_stratified_manifest(
        dataset_root=dataset_root,
        val_ratio=args.val_ratio,
        random_seed=args.seed,
        project_root=PROJECT_ROOT,
    )

    save_manifest(manifest_df, manifest_path)
    print_split_summary(manifest_df)
    print(f"Split manifest successfully written to: {manifest_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
