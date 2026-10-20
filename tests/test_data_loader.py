"""
tests/test_data_loader.py
--------------------------
Tests for src/data_loader.py — the pipeline that merges raw Kaggle CSVs
into ReviewLens's unified schema.

Label-mapping tests always run (they need no files).
Full-pipeline tests are skipped automatically if the raw Kaggle CSVs
are not present in data/kaggle_raw/ (they are not shipped with this
repo due to size — see README.md for how to obtain them).

Run:
    pytest tests/test_data_loader.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from src.data_loader import (
    RAW_KAGGLE_DIR,
    _binary_to_label,
    _rating_to_label,
    _sentiment_to_label,
    build_unified_dataset,
    load_source,
)
from src.data_sources import SOURCES, SourceConfig

KAGGLE_DATA_AVAILABLE = RAW_KAGGLE_DIR.exists() and any(RAW_KAGGLE_DIR.iterdir())
SKIP_MSG = (
    "Raw Kaggle CSVs not found in data/kaggle_raw/ — "
    "these are not shipped with the repo due to size. "
    "See README.md to download them and re-run prepare_dataset.py."
)


# ── Rating → label mapping ────────────────────────────────────────────────────

class TestRatingToLabel:
    def test_rating_5_is_positive(self):
        assert _rating_to_label(5) == "positive"

    def test_rating_4_is_positive(self):
        assert _rating_to_label(4) == "positive"

    def test_rating_3_is_neutral(self):
        assert _rating_to_label(3) == "neutral"

    def test_rating_2_is_negative(self):
        assert _rating_to_label(2) == "negative"

    def test_rating_1_is_negative(self):
        assert _rating_to_label(1) == "negative"

    def test_invalid_rating_returns_none(self):
        assert _rating_to_label("not a number") is None
        assert _rating_to_label(None) is None


# ── Binary → label mapping ────────────────────────────────────────────────────

class TestBinaryToLabel:
    def test_one_is_positive(self):
        assert _binary_to_label(1) == "positive"

    def test_zero_is_negative(self):
        assert _binary_to_label(0) == "negative"

    def test_invalid_returns_none(self):
        assert _binary_to_label("x") is None
        assert _binary_to_label(None) is None


# ── Existing sentiment column normalisation ───────────────────────────────────

class TestSentimentToLabel:
    def test_lowercase_positive(self):
        assert _sentiment_to_label("positive") == "positive"

    def test_uppercase_normalised(self):
        assert _sentiment_to_label("POSITIVE") == "positive"

    def test_mixed_case_with_whitespace(self):
        assert _sentiment_to_label("  Negative  ") == "negative"

    def test_unrecognised_value_returns_none(self):
        assert _sentiment_to_label("mixed") is None

    def test_non_string_returns_none(self):
        assert _sentiment_to_label(5) is None
        assert _sentiment_to_label(None) is None


# ── load_source (uses a small temp CSV, no Kaggle data needed) ───────────────

class TestLoadSource:
    def test_loads_and_standardises_rating_source(self, tmp_path):
        # Build a tiny fake "raw Kaggle" CSV
        raw_dir = tmp_path / "kaggle_raw"
        raw_dir.mkdir()
        df = pd.DataFrame({
            "text_col":   ["Great product", "Bad product", "", None],
            "rating_col": [5, 1, 3, 4],
        })
        df.to_csv(raw_dir / "fake.csv", index=False)

        config = SourceConfig(
            file_name="fake.csv", text_col="text_col",
            label_type="rating", label_col="rating_col",
            category="Test", source_name="TestSource",
        )

        import src.data_loader as dl
        original_dir = dl.RAW_KAGGLE_DIR
        dl.RAW_KAGGLE_DIR = raw_dir
        try:
            result = load_source(config)
        finally:
            dl.RAW_KAGGLE_DIR = original_dir

        # Empty/None text rows should be dropped -> 2 usable rows remain
        assert len(result) == 2
        assert set(result.columns) == {"review_text", "rating", "date", "source", "category", "label"}
        assert set(result["label"]) == {"positive", "negative"}
        assert (result["source"] == "TestSource").all()
        assert (result["category"] == "Test").all()

    def test_missing_file_returns_empty_dataframe(self, tmp_path):
        config = SourceConfig(
            file_name="does_not_exist.csv", text_col="a",
            label_type="rating", label_col="b",
            category="Test", source_name="TestSource",
        )
        import src.data_loader as dl
        original_dir = dl.RAW_KAGGLE_DIR
        dl.RAW_KAGGLE_DIR = tmp_path
        try:
            result = load_source(config)
        finally:
            dl.RAW_KAGGLE_DIR = original_dir
        assert result.empty


# ── Source configuration sanity checks ────────────────────────────────────────

class TestSourceConfig:
    def test_at_least_ten_sources_configured(self):
        assert len(SOURCES) >= 10

    def test_every_source_has_required_fields(self):
        for cfg in SOURCES:
            assert cfg.file_name
            assert cfg.text_col
            assert cfg.label_col
            assert cfg.label_type in {"rating", "sentiment", "binary"}
            assert cfg.category
            assert cfg.source_name

    def test_categories_are_diverse(self):
        categories = {cfg.category for cfg in SOURCES}
        assert len(categories) >= 4  # e.g. Electronics, App, Course, Restaurant, ...


# ── Full pipeline (skipped if raw Kaggle CSVs are not present) ───────────────

@pytest.mark.skipif(not KAGGLE_DATA_AVAILABLE, reason=SKIP_MSG)
def test_build_unified_dataset_schema():
    df = build_unified_dataset()
    assert set(df.columns) == {"review_text", "rating", "date", "source", "category", "label"}


@pytest.mark.skipif(not KAGGLE_DATA_AVAILABLE, reason=SKIP_MSG)
def test_build_unified_dataset_labels_valid():
    df = build_unified_dataset()
    assert set(df["label"].unique()).issubset({"positive", "negative", "neutral"})


@pytest.mark.skipif(not KAGGLE_DATA_AVAILABLE, reason=SKIP_MSG)
def test_build_unified_dataset_no_empty_text():
    df = build_unified_dataset()
    assert (df["review_text"].astype(str).str.strip() != "").all()


@pytest.mark.skipif(not KAGGLE_DATA_AVAILABLE, reason=SKIP_MSG)
def test_build_unified_dataset_no_duplicates():
    df = build_unified_dataset()
    assert df["review_text"].duplicated().sum() == 0


@pytest.mark.skipif(not KAGGLE_DATA_AVAILABLE, reason=SKIP_MSG)
def test_build_unified_dataset_reasonable_size():
    df = build_unified_dataset()
    assert len(df) > 10_000  # sanity check — should be tens of thousands of real reviews
