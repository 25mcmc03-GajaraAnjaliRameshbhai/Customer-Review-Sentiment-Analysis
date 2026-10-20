"""
src/utils.py
------------
Shared utilities, path constants and helper functions for ReviewLens.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# ── Path constants ────────────────────────────────────────────────────────────

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
MODEL_DIR      = PROJECT_ROOT / "models"
DATA_DIR       = PROJECT_ROOT / "data"

PIPELINE_PATH  = MODEL_DIR / "pipeline.pkl"
METADATA_PATH  = MODEL_DIR / "metadata.json"


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_dataset() -> Optional[pd.DataFrame]:
    """
    Load the unified review dataset from data/raw/reviews.csv.

    This file is built by running `python prepare_dataset.py`, which
    merges the real Kaggle review datasets (Amazon, Flipkart, App Store,
    Play Store, Coursera, Zomato, IMDB, etc.) into ReviewLens's schema:
    review_text, rating, date, source, category, label.

    Returns None if the file does not exist yet.
    """
    path = DATA_DIR / "raw" / "reviews.csv"
    if path.exists():
        print(f"Loading dataset from {path}")
        return pd.read_csv(path)
    return None


# ── Metadata ─────────────────────────────────────────────────────────────────

def save_metadata(meta: Dict[str, Any]) -> None:
    """Persist training metadata to ``models/metadata.json``."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"Metadata saved → {METADATA_PATH}")


def load_metadata() -> Optional[Dict[str, Any]]:
    """Load training metadata, or return None if not found."""
    if not METADATA_PATH.exists():
        return None
    with open(METADATA_PATH) as fh:
        return json.load(fh)


# ── Model status ──────────────────────────────────────────────────────────────

def models_exist() -> bool:
    """Return True if the trained pipeline artifact is present."""
    return PIPELINE_PATH.exists()


# ── Upload limits ──────────────────────────────────────────────────────────
# Enforced in src/pages/explorer.py. Kept here so the same numbers can be
# referenced consistently in both the validation logic and the UI copy
# that tells the user about them.

MAX_UPLOAD_SIZE_MB   = 50        # hard cap — files larger than this are rejected
RECOMMENDED_MAX_ROWS = 20_000     # soft guidance — larger files still work, just slower
MAX_UPLOAD_ROWS      = 150_000    # hard cap — files with more rows are rejected
PREDICT_CHUNK_SIZE   = 2_000      # batch size used for chunked prediction + progress bar


# ── Label-color / display helpers ─────────────────────────────────────────────

LABEL_COLORS: Dict[str, str] = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral":  "#f39c12",
}

LABEL_EMOJI: Dict[str, str] = {
    "positive": "🟢",
    "negative": "🔴",
    "neutral":  "🟡",
}


def get_color_for_label(label: str) -> str:
    """Return the hex colour string for *label*."""
    return LABEL_COLORS.get(str(label).lower(), "#95a5a6")


def get_emoji_for_label(label: str) -> str:
    """Return an emoji indicator for *label*."""
    return LABEL_EMOJI.get(str(label).lower(), "⚪")


def format_label(label: str) -> str:
    """Capitalise a label string for display."""
    return str(label).capitalize()
