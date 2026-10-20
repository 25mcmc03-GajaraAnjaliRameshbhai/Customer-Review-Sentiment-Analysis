"""
train.py
--------
Train a sentiment classification model for ReviewLens.

Usage
-----
    python train.py

Outputs saved to models/
    pipeline.pkl    — best fitted sklearn Pipeline
    metadata.json   — training metrics and metadata
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.preprocess import preprocess_batch
from src.train_model import (
    build_lr_pipeline,
    build_nb_pipeline,
    evaluate_pipeline,
    get_top_features,
    tune_class_weights,
)
from src.utils import (
    MODEL_DIR,
    METADATA_PATH,
    PIPELINE_PATH,
    load_dataset,
    save_metadata,
)


def _derive_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'label' column from 'rating' if label is absent."""
    def rating_to_label(r) -> str:
        try:
            r = float(r)
        except (TypeError, ValueError):
            return "neutral"
        if r >= 4:
            return "positive"
        elif r <= 2:
            return "negative"
        return "neutral"

    df = df.copy()
    df["label"] = df["rating"].apply(rating_to_label)
    return df


def main() -> None:
    print("=" * 60)
    print("  ReviewLens — Model Training")
    print("=" * 60)

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    df = load_dataset()
    if df is None:
        print("\n❌  No dataset found.")
        print("    Run:  python prepare_dataset.py")
        sys.exit(1)

    print(f"\n✅  Loaded {len(df):,} reviews")

    # ── 2. Validate / derive columns ──────────────────────────────────────────
    if "review_text" not in df.columns:
        print("❌  'review_text' column is missing from the dataset.")
        sys.exit(1)

    if "label" not in df.columns:
        if "rating" in df.columns:
            print("ℹ️   Deriving labels from 'rating' column …")
            df = _derive_labels(df)
        else:
            print("❌  Neither 'label' nor 'rating' column found.")
            sys.exit(1)

    # ── 3. Clean text ─────────────────────────────────────────────────────────
    print("\nPreprocessing text …")
    df["clean_text"] = preprocess_batch(df["review_text"].fillna("").tolist())
    df = df[df["clean_text"].str.strip().astype(bool)].reset_index(drop=True)
    print(f"Reviews remaining after cleaning: {len(df):,}")

    # ── 4. Train / test split ─────────────────────────────────────────────────
    X: list = df["clean_text"].tolist()
    y: list = df["label"].tolist()

    print(f"\nClass distribution (full dataset):")
    for lbl, cnt in pd.Series(y).value_counts().items():
        print(f"  {lbl:<10} {cnt:>5}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── 5. Tune class weights / regularisation for Logistic Regression ────────
    # scikit-learn's built-in class_weight="balanced" over-corrects for the
    # rare "neutral" class on this dataset (high recall, very low precision).
    # This small cross-validated search finds a better-balanced weighting.
    # See docs/PROJECT_REPORT.md for the full comparison of alternatives tried.
    best_C, best_weights, _ = tune_class_weights(X_train, y_train)

    # ── 6. Train both models ──────────────────────────────────────────────────
    print("🔄  Training Logistic Regression (tuned) …")
    lr_pipe = build_lr_pipeline(C=best_C, class_weight=best_weights)
    lr_pipe.fit(X_train, y_train)
    lr_metrics = evaluate_pipeline(lr_pipe, X_test, y_test, label="Logistic Regression")

    print("🔄  Training Naive Bayes (baseline) …")
    nb_pipe = build_nb_pipeline()
    nb_pipe.fit(X_train, y_train)
    nb_metrics = evaluate_pipeline(nb_pipe, X_test, y_test, label="Naive Bayes")

    # ── 7. Select best model by macro F1 ─────────────────────────────────────
    if lr_metrics["macro_f1"] >= nb_metrics["macro_f1"]:
        best_pipe    = lr_pipe
        best_metrics = lr_metrics
        best_name    = "LogisticRegression"
    else:
        best_pipe    = nb_pipe
        best_metrics = nb_metrics
        best_name    = "MultinomialNB"

    print(f"\n🏆  Best model: {best_name}  (Macro F1 = {best_metrics['macro_f1']:.4f})")

    # ── 8. Save pipeline ──────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, PIPELINE_PATH)
    print(f"✅  Pipeline saved → {PIPELINE_PATH}")

    # ── 9. Extract top features (LR only) ────────────────────────────────────
    top_features: dict = {}
    if best_name == "LogisticRegression":
        top_features = get_top_features(best_pipe, n=20)

    # ── 10. Build and save metadata ────────────────────────────────────────────
    class_dist = pd.Series(y_train).value_counts().to_dict()
    accuracy   = float(accuracy_score(y_test, best_pipe.predict(X_test)))

    metadata = {
        "model_name":                best_name,
        "trained_at":                datetime.now().isoformat(),
        "number_of_training_samples": len(X_train),
        "number_of_test_samples":    len(X_test),
        "class_distribution":        {str(k): int(v) for k, v in class_dist.items()},
        "hyperparameters": {
            "C":            best_C,
            "class_weight": best_weights,
        } if best_name == "LogisticRegression" else {},
        "metrics": {
            "accuracy":    round(accuracy, 4),
            "macro_f1":    best_metrics["macro_f1"],
            "weighted_f1": best_metrics["weighted_f1"],
        },
        "confusion_matrix":      best_metrics["confusion_matrix"],
        "confusion_matrix_labels": sorted(set(y_test)),
        "top_positive_features": top_features.get("positive", [])[:20],
        "top_negative_features": top_features.get("negative", [])[:20],
    }

    save_metadata(metadata)

    # ── 11. Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training Complete ✅")
    print(f"  Model      : {best_name}")
    print(f"  Accuracy   : {metadata['metrics']['accuracy']:.4f}")
    print(f"  Macro F1   : {metadata['metrics']['macro_f1']:.4f}")
    print(f"  Weighted F1: {metadata['metrics']['weighted_f1']:.4f}")
    print(f"  Pipeline   : {PIPELINE_PATH}")
    print(f"  Metadata   : {METADATA_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
