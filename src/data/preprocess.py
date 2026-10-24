"""Core data preprocessing, image standardization, and stratified splitting pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Generator, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def standardize_image(
    image_path: Path | str,
    target_size: tuple[int, int] = (224, 224),
    normalize_range: bool = True,
) -> np.ndarray:
    """Load an image, standardize to single-channel grayscale, and resize.

    Parameters
    ----------
    image_path : Path | str
        Path to the image file.
    target_size : tuple[int, int], default=(224, 224)
        Desired output dimensions (width, height).
    normalize_range : bool, default=True
        If True, returns float32 in [0.0, 1.0]. If False, returns uint8 in [0, 255].

    Returns
    -------
    np.ndarray
        Standardized 2D array of shape (height, width).
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    with Image.open(path) as img:
        # Convert all formats (RGB, RGBA, Grayscale) to clean single-channel L
        grayscale_img = img.convert("L")
        resized_img = grayscale_img.resize(target_size, Image.Resampling.BILINEAR)
        array = np.array(resized_img)

    if normalize_range:
        return array.astype(np.float32) / 255.0
    return array.astype(np.uint8)


def iter_split_images(
    manifest_path: Path | str,
    split: str,
    project_root: Path | None = None,
    target_size: tuple[int, int] = (224, 224),
    normalize_range: bool = True,
) -> Generator[tuple[np.ndarray, str, str], None, None]:
    """Iterate over standardized images and labels for a specific split.

    Yields
    ------
    tuple[np.ndarray, str, str]
        (standardized_image_array, label_string, relative_image_path)
    """
    path = Path(manifest_path).resolve()
    base = Path(project_root).resolve() if project_root else path.parents[2]
    df = pd.read_csv(path)

    split_df = df[df["split"].str.lower() == split.lower()]
    for row in split_df.itertuples(index=False):
        raw_p = Path(row.image_path)
        if raw_p.is_absolute() and raw_p.exists():
            img_path = raw_p
        elif (base / raw_p).exists():
            img_path = base / raw_p
        elif raw_p.exists():
            img_path = raw_p.resolve()
        else:
            img_path = base / raw_p
        img_arr = standardize_image(img_path, target_size=target_size, normalize_range=normalize_range)
        yield img_arr, row.label, row.image_path




def scan_split_images(
    split_dir: Path,
    expected_classes: Sequence[str] = ("NORMAL", "PNEUMONIA"),
) -> list[dict[str, str]]:
    """Scan a split directory for class folders and image files."""
    records: list[dict[str, str]] = []
    if not split_dir.exists():
        return records

    for class_folder in split_dir.iterdir():
        if not class_folder.is_dir():
            continue
        class_name = class_folder.name.upper()
        if class_name not in expected_classes:
            continue

        for file_path in sorted(class_folder.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                records.append({
                    "relative_path": str(file_path),
                    "label": class_name,
                })
    return records


def build_stratified_manifest(
    dataset_root: Path | str,
    val_ratio: float = 0.15,
    random_seed: int = 42,
    project_root: Path | None = None,
) -> pd.DataFrame:
    """Build a complete, leakage-free split manifest with a stratified validation set.

    Parameters
    ----------
    dataset_root : Path | str
        Root path to raw chest_xray directory containing train and test folders.
    val_ratio : float, default=0.15
        Proportion of the training set to allocate to the new validation set.
    random_seed : int, default=42
        Random seed for reproducible stratified splitting.
    project_root : Path, optional
        Project root to make relative paths portable.

    Returns
    -------
    pd.DataFrame
        Manifest DataFrame with columns: image_path, label, split, sha256.
    """
    root = Path(dataset_root).resolve()
    base = Path(project_root).resolve() if project_root else root.parents[1]

    # Resolve train and test paths (case-insensitive search)
    train_dir = next((p for p in root.iterdir() if p.name.lower() in ("train", "training")), None)
    test_dir = next((p for p in root.iterdir() if p.name.lower() == "test"), None)

    if not train_dir or not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found under {root}")
    if not test_dir or not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found under {root}")

    train_records = scan_split_images(train_dir)
    test_records = scan_split_images(test_dir)

    if not train_records:
        raise ValueError(f"No valid images found in {train_dir}")
    if not test_records:
        raise ValueError(f"No valid images found in {test_dir}")

    # Compute SHA-256 for all training images first to prevent duplicate leakage
    train_data = []
    for r in train_records:
        path = Path(r["relative_path"])
        h = compute_sha256(path)
        train_data.append({"relative_path": r["relative_path"], "label": r["label"], "sha256": h})
    train_df = pd.DataFrame(train_data)

    # Stratify at the unique hash level so identical images never cross splits
    unique_hashes = train_df.groupby("sha256")["label"].first().reset_index()
    strat_train_hashes, strat_val_hashes = train_test_split(
        unique_hashes["sha256"].values,
        test_size=val_ratio,
        stratify=unique_hashes["label"].values,
        random_state=random_seed,
    )
    val_hash_set = set(strat_val_hashes)

    manifest_rows: list[dict[str, str]] = []

    for row in train_df.itertuples(index=False):
        full_path = Path(row.relative_path)
        rel_path = full_path.relative_to(base) if full_path.is_relative_to(base) else full_path
        split = "val" if row.sha256 in val_hash_set else "train"
        manifest_rows.append({
            "image_path": str(rel_path),
            "label": row.label,
            "split": split,
            "sha256": row.sha256,
        })

    # Test set is kept untouched from the original test split
    for record in test_records:
        full_path = Path(record["relative_path"])
        rel_path = full_path.relative_to(base) if full_path.is_relative_to(base) else full_path
        manifest_rows.append({
            "image_path": str(rel_path),
            "label": record["label"],
            "split": "test",
            "sha256": compute_sha256(full_path),
        })

    manifest_df = pd.DataFrame(manifest_rows)
    return manifest_df


def save_manifest(manifest_df: pd.DataFrame, output_path: Path | str) -> Path:
    """Save manifest DataFrame to a CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(path, index=False)
    return path

