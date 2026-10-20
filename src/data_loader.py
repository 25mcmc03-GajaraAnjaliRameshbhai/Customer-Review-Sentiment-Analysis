"""
src/data_loader.py
-------------------
Generic loader that reads every raw Kaggle CSV described in
src/data_sources.py and converts it into ReviewLens's unified schema:

    review_text | rating | date | source | category | label

The same three small helper functions (_rating_to_label,
_binary_to_label, _sentiment_to_label) handle every source file —
no per-file special-casing is needed beyond the config in
src/data_sources.py.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.data_sources import SOURCES, SourceConfig

RAW_KAGGLE_DIR = Path(__file__).resolve().parent.parent / "data" / "kaggle_raw"


# ── Label-mapping helpers ─────────────────────────────────────────────────────

def _rating_to_label(rating) -> Optional[str]:
    """Map a 1-5 star rating to positive / negative / neutral."""
    try:
        r = float(rating)
    except (TypeError, ValueError):
        return None
    if r >= 4:
        return "positive"
    if r <= 2:
        return "negative"
    return "neutral"


def _binary_to_label(value) -> Optional[str]:
    """Map a 0/1 'liked' style column to negative / positive."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return "positive" if v == 1 else "negative"


def _sentiment_to_label(value) -> Optional[str]:
    """Normalise an already-labelled sentiment column to lowercase."""
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v if v in {"positive", "negative", "neutral"} else None


_LABEL_MAPPERS = {
    "rating":    _rating_to_label,
    "binary":    _binary_to_label,
    "sentiment": _sentiment_to_label,
}


# ── Per-file loader ────────────────────────────────────────────────────────────

def load_source(config: SourceConfig) -> pd.DataFrame:
    """
    Load a single raw CSV and standardise it to the common schema.

    Returns an empty DataFrame (with a printed warning) if the file is
    missing or its expected columns are not found, so one bad file never
    crashes the whole pipeline.
    """
    path = RAW_KAGGLE_DIR / config.file_name
    if not path.exists():
        print(f"  ⚠️  Skipped (file not found): {config.file_name}")
        return pd.DataFrame()

    # Try the configured encoding first, then fall back to latin-1, which can
    # decode any byte sequence — some Kaggle exports are not strictly UTF-8.
    try:
        df = pd.read_csv(path, encoding=config.encoding, on_bad_lines="skip", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip", low_memory=False)

    if config.text_col not in df.columns or config.label_col not in df.columns:
        print(f"  ⚠️  Skipped (expected columns missing): {config.file_name}")
        return pd.DataFrame()

    out = pd.DataFrame()
    out["review_text"] = df[config.text_col]
    out["date"] = df[config.date_col] if config.date_col and config.date_col in df.columns else None

    mapper = _LABEL_MAPPERS[config.label_type]
    out["label"] = df[config.label_col].apply(mapper)
    out["rating"] = pd.to_numeric(df[config.label_col], errors="coerce") \
        if config.label_type == "rating" else None

    out["source"]   = config.source_name
    out["category"] = config.category

    # Drop rows with missing text or a label that could not be mapped
    out = out.dropna(subset=["review_text", "label"])
    out = out[out["review_text"].astype(str).str.strip().astype(bool)]

    if config.sample_size and len(out) > config.sample_size:
        out = out.sample(n=config.sample_size, random_state=42)

    print(f"  ✅  {config.file_name:<45} → {len(out):>6,} usable rows")
    return out.reset_index(drop=True)


# ── Full pipeline ──────────────────────────────────────────────────────────────

def build_unified_dataset() -> pd.DataFrame:
    """
    Load every configured Kaggle source, merge them, and return one
    clean DataFrame in ReviewLens's unified schema.
    """
    frames = []
    print(f"Loading {len(SOURCES)} Kaggle source files from {RAW_KAGGLE_DIR} …\n")

    for cfg in SOURCES:
        frame = load_source(cfg)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise RuntimeError(
            "No source files could be loaded. "
            f"Place the raw Kaggle CSVs in: {RAW_KAGGLE_DIR}"
        )

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["review_text"])
    merged = merged[merged["review_text"].astype(str).str.len() > 3]
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle sources together

    return merged[["review_text", "rating", "date", "source", "category", "label"]]
