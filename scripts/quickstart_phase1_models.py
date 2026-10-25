"""Quickstart script to generate Phase 1 baseline classical models and scalers.

Extracts HOG and LBP features on a stratified subset of the training and validation
splits (adhering strictly to anti-leakage rules: Scaler is fitted ONLY on train),
tunes and fits the Phase 1 SVM models:
  - hog_linear_svm
  - hog_rbf_svm
  - hog_lbp_rbf_svm

Saves serialized estimators to results/models/ and metrics to results/metrics/.
"""

import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import standardize_image
from src.features.hog import extract_hog_features
from src.features.lbp import extract_lbp_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger("quickstart_phase1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "split_manifest.csv"
MODEL_DIR = PROJECT_ROOT / "results" / "models"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

def run():
    df = pd.read_csv(MANIFEST_PATH)

    # Sample a stratified subset for fast initial deployment
    # 350 train samples (approx 73% pneumonia, 27% normal), 100 val samples
    train_df = df[df["split"] == "train"].groupby("label", group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), 175 if x.name == "NORMAL" else 225), random_state=42)
    )
    val_df = df[df["split"] == "val"].groupby("label", group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), 40 if x.name == "NORMAL" else 60), random_state=42)
    )

    logger.info("Extracting features for %d train and %d val samples...", len(train_df), len(val_df))

    def extract_split(subset_df):
        hog_list, lbp_list, labels, paths = [], [], [], []
        for _, row in subset_df.iterrows():
            img_path = PROJECT_ROOT / row["image_path"]
            if not img_path.is_file():
                continue
            arr = standardize_image(img_path)
            h = extract_hog_features(arr)
            l = extract_lbp_features(arr)
            hog_list.append(h)
            lbp_list.append(l)
            labels.append(1 if row["label"] == "PNEUMONIA" else 0)
            paths.append(row["image_path"])
        return np.array(hog_list), np.array(lbp_list), np.array(labels), paths

    X_train_hog_raw, X_train_lbp_raw, y_train, _ = extract_split(train_df)
    X_val_hog_raw, X_val_lbp_raw, y_val, _ = extract_split(val_df)

    logger.info("Fitting Scaler strictly on training data...")
    hog_scaler = StandardScaler().fit(X_train_hog_raw)
    X_train_hog = hog_scaler.transform(X_train_hog_raw)
    X_val_hog = hog_scaler.transform(X_val_hog_raw)

    lbp_scaler = StandardScaler().fit(X_train_lbp_raw)
    X_train_lbp = lbp_scaler.transform(X_train_lbp_raw)
    X_val_lbp = lbp_scaler.transform(X_val_lbp_raw)

    X_train_comb = np.hstack([X_train_hog, X_train_lbp])
    X_val_comb = np.hstack([X_val_hog, X_val_lbp])

    # Save scalers for live inference in Streamlit
    joblib.dump(hog_scaler, MODEL_DIR / "hog_scaler.joblib")
    joblib.dump(lbp_scaler, MODEL_DIR / "lbp_scaler.joblib")

    models_config = {
        "hog_linear_svm": (LinearSVC(class_weight="balanced", max_iter=10000, random_state=42), {"C": [0.01, 0.1, 1.0, 10.0]}, X_train_hog, X_val_hog),
        "hog_rbf_svm": (SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42), {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01]}, X_train_hog, X_val_hog),
        "hog_lbp_rbf_svm": (SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42), {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01]}, X_train_comb, X_val_comb),
    }

    metrics_summary = {}

    for name, (estimator, grid, X_tr, X_v) in models_config.items():
        logger.info("Training %s with Stratified 5-Fold CV...", name)
        t0 = time.time()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        search = GridSearchCV(estimator, grid, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
        search.fit(X_tr, y_train)

        val_preds = search.predict(X_v)
        val_acc = accuracy_score(y_val, val_preds)
        val_bal_acc = balanced_accuracy_score(y_val, val_preds)
        val_sens = recall_score(y_val, val_preds, pos_label=1)

        # Refit best estimator on train + val
        best_est = search.best_estimator_
        X_all = np.vstack([X_tr, X_v])
        y_all = np.concatenate([y_train, y_val])
        best_est.fit(X_all, y_all)

        model_path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(best_est, model_path)
        elapsed = time.time() - t0

        logger.info("%s finished in %.1fs: Best Params %s, Val Bal-Acc: %.3f, Val Recall: %.3f",
                    name, elapsed, search.best_params_, val_bal_acc, val_sens)

        metrics_summary[name] = {
            "best_params": search.best_params_,
            "best_cv_score": round(float(search.best_score_), 4),
            "val_accuracy": round(float(val_acc), 4),
            "val_balanced_accuracy": round(float(val_bal_acc), 4),
            "val_sensitivity": round(float(val_sens), 4),
            "elapsed_seconds": round(elapsed, 2),
            "train_samples": len(X_tr),
            "val_samples": len(X_v),
        }

    with open(METRICS_DIR / "classical_training.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    logger.info("Saved all models to %s and metrics to %s", MODEL_DIR, METRICS_DIR)

if __name__ == "__main__":
    run()
