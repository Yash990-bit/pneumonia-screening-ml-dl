"""Unit tests for data preprocessing and stratified split pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.preprocess import (
    build_stratified_manifest,
    compute_sha256,
    standardize_image,
)


class PreprocessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "chest_xray"
        for split in ("train", "test"):
            for cls in ("NORMAL", "PNEUMONIA"):
                (self.root / split / cls).mkdir(parents=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_dummy_image(self, rel_path: str, color: int = 128, mode: str = "L", size=(100, 100)):
        path = self.root / rel_path
        Image.new(mode, size, color=color).save(path)
        return path

    def test_standardize_image_rgb_to_grayscale_and_resize(self) -> None:
        img_path = self._create_dummy_image("train/NORMAL/sample_rgb.jpg", color=200, mode="RGB", size=(400, 300))
        standardized = standardize_image(img_path, target_size=(224, 224), normalize_range=True)

        self.assertEqual(standardized.shape, (224, 224))
        self.assertEqual(standardized.dtype, np.float32)
        self.assertTrue(0.0 <= standardized.min() <= standardized.max() <= 1.0)

    def test_stratified_split_preserves_class_ratios_and_no_leakage(self) -> None:
        # Create 20 train images (10 Normal, 10 Pneumonia)
        for i in range(10):
            self._create_dummy_image(f"train/NORMAL/n_{i}.jpg", color=i)
            self._create_dummy_image(f"train/PNEUMONIA/p_{i}.jpg", color=100 + i)

        # Create 4 test images
        for i in range(2):
            self._create_dummy_image(f"test/NORMAL/tn_{i}.jpg", color=200 + i)
            self._create_dummy_image(f"test/PNEUMONIA/tp_{i}.jpg", color=220 + i)

        manifest = build_stratified_manifest(self.root, val_ratio=0.2, random_seed=42)

        train_count = len(manifest[manifest["split"] == "train"])
        val_count = len(manifest[manifest["split"] == "val"])
        test_count = len(manifest[manifest["split"] == "test"])

        self.assertEqual(test_count, 4)
        self.assertEqual(train_count, 16)
        self.assertEqual(val_count, 4)

        # Check hash separation (Zero leakage)
        train_hashes = set(manifest[manifest["split"] == "train"]["sha256"])
        val_hashes = set(manifest[manifest["split"] == "val"]["sha256"])
        test_hashes = set(manifest[manifest["split"] == "test"]["sha256"])

        self.assertEqual(len(train_hashes.intersection(val_hashes)), 0)
        self.assertEqual(len(train_hashes.intersection(test_hashes)), 0)
        self.assertEqual(len(val_hashes.intersection(test_hashes)), 0)


if __name__ == "__main__":
    unittest.main()
