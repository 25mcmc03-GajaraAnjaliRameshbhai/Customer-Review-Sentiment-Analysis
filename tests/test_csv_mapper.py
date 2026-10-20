"""
tests/test_csv_mapper.py
--------------------------
Tests for src/csv_mapper.py — the logic that lets Review Explorer
accept ANY CSV, regardless of column names, by auto-detecting which
column holds review text / rating / date / source / category.

Run:
    pytest tests/test_csv_mapper.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.csv_mapper import apply_mapping, detect_columns, infer_column_type, validate_mapping


# ── Exact ReviewLens schema (the "easy" case) ─────────────────────────────────

class TestDetectColumnsExactSchema:
    def test_standard_schema_detected(self):
        df = pd.DataFrame({
            "review_text": ["great", "bad"],
            "rating": [5, 1],
            "date": ["2024-01-01", "2024-01-02"],
            "source": ["Amazon", "Flipkart"],
            "category": ["Mobile", "Laptop"],
        })
        mapping = detect_columns(df)
        assert mapping["review_text"] == "review_text"
        assert mapping["rating"] == "rating"
        assert mapping["date"] == "date"
        assert mapping["source"] == "source"
        assert mapping["category"] == "category"


# ── Differently-named columns (the real-world case) ───────────────────────────

class TestDetectColumnsAlternateNames:
    def test_comments_and_stars_detected(self):
        df = pd.DataFrame({
            "Comments": ["nice product", "poor quality"],
            "Stars": [5, 1],
            "Timestamp": ["2024-01-01", "2024-01-02"],
            "Platform": ["Amazon", "Flipkart"],
        })
        mapping = detect_columns(df)
        assert mapping["review_text"] == "Comments"
        assert mapping["rating"] == "Stars"
        assert mapping["date"] == "Timestamp"
        assert mapping["source"] == "Platform"

    def test_feedback_column_detected(self):
        df = pd.DataFrame({"feedback": ["good", "bad"], "score": [4, 2]})
        mapping = detect_columns(df)
        assert mapping["review_text"] == "feedback"
        assert mapping["rating"] == "score"

    def test_case_insensitive_matching(self):
        df = pd.DataFrame({"REVIEW": ["good"], "RATING": [5]})
        mapping = detect_columns(df)
        assert mapping["review_text"] == "REVIEW"
        assert mapping["rating"] == "RATING"

    def test_underscore_and_space_variants(self):
        df = pd.DataFrame({"review text": ["good"], "product_name": ["Phone"]})
        mapping = detect_columns(df)
        assert mapping["review_text"] == "review text"
        assert mapping["category"] == "product_name"


# ── Minimal CSV (text-only, no optional columns) ──────────────────────────────

class TestDetectColumnsMinimal:
    def test_single_column_csv(self):
        df = pd.DataFrame({"review_text": ["good", "bad", "okay"]})
        mapping = detect_columns(df)
        assert mapping["review_text"] == "review_text"
        assert mapping["rating"] is None
        assert mapping["date"] is None
        assert mapping["source"] is None
        assert mapping["category"] is None


# ── No name matches anything — fallback to longest-text heuristic ────────────

class TestDetectColumnsFallback:
    def test_falls_back_to_longest_text_column(self):
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "notes": [
                "This is a fairly long piece of free text about the product experience.",
                "Another long free text field describing what happened with the order.",
                "A third detailed note about the overall product quality and delivery.",
            ],
        })
        mapping = detect_columns(df)
        assert mapping["review_text"] == "notes"

    def test_does_not_pick_short_id_like_column(self):
        df = pd.DataFrame({
            "code": ["A1", "B2", "C3"],
            "long_description": [
                "Excellent build quality and very fast shipping overall experience.",
                "Terrible packaging, item arrived broken and support was unhelpful.",
                "Decent product for the price, would consider buying again someday.",
            ],
        })
        mapping = detect_columns(df)
        assert mapping["review_text"] == "long_description"


# ── apply_mapping ──────────────────────────────────────────────────────────────

class TestApplyMapping:
    def test_full_mapping_produces_standard_schema(self):
        df = pd.DataFrame({
            "Comments": ["nice", "bad"],
            "Stars": [5, 1],
        })
        mapping = {"review_text": "Comments", "rating": "Stars",
                  "date": None, "source": None, "category": None}
        result = apply_mapping(df, mapping)
        assert list(result["review_text"]) == ["nice", "bad"]
        assert list(result["rating"]) == [5, 1]
        assert "date" not in result.columns
        assert "source" not in result.columns

    def test_unmapped_optional_fields_are_omitted(self):
        df = pd.DataFrame({"text": ["good", "bad"]})
        mapping = {"review_text": "text", "rating": None,
                  "date": None, "source": None, "category": None}
        result = apply_mapping(df, mapping)
        assert list(result.columns) == ["review_text"]

    def test_review_text_always_cast_to_string(self):
        df = pd.DataFrame({"text": [123, 456]})  # numeric column, unusual but shouldn't crash
        mapping = {"review_text": "text", "rating": None,
                  "date": None, "source": None, "category": None}
        result = apply_mapping(df, mapping)
        assert list(result["review_text"]) == ["123", "456"]
        assert all(isinstance(v, str) for v in result["review_text"])

    def test_rating_coerced_to_numeric(self):
        df = pd.DataFrame({"text": ["good", "bad"], "stars": ["5", "1"]})
        mapping = {"review_text": "text", "rating": "stars",
                  "date": None, "source": None, "category": None}
        result = apply_mapping(df, mapping)
        assert pd.api.types.is_numeric_dtype(result["rating"])
        assert list(result["rating"]) == [5.0, 1.0]

    def test_invalid_rating_values_become_nan(self):
        df = pd.DataFrame({"text": ["good", "bad", "ok"], "stars": ["5", "not_a_rating", "3"]})
        mapping = {"review_text": "text", "rating": "stars",
                  "date": None, "source": None, "category": None}
        result = apply_mapping(df, mapping)
        assert result["rating"].isna().sum() == 1

    def test_date_normalised_to_iso_format(self):
        df = pd.DataFrame({"text": ["good"], "when": ["03/15/2024"]})
        mapping = {"review_text": "text", "rating": None,
                  "date": "when", "source": None, "category": None}
        result = apply_mapping(df, mapping)
        assert result["date"].iloc[0] == "2024-03-15"


# ── infer_column_type ─────────────────────────────────────────────────────────

class TestInferColumnType:
    def test_numeric_column(self):
        assert infer_column_type(pd.Series([1, 2, 3, 4, 5])) == "numeric"

    def test_numeric_strings_column(self):
        assert infer_column_type(pd.Series(["1", "2", "3", "4"])) == "numeric"

    def test_date_column(self):
        assert infer_column_type(pd.Series(["2024-01-01", "2024-02-15", "2024-03-20"])) == "date"

    def test_text_column(self):
        s = pd.Series(["This is a great product", "Terrible experience overall", "It was okay I guess"])
        assert infer_column_type(s) == "text"

    def test_empty_column(self):
        assert infer_column_type(pd.Series([None, None, ""])) == "empty"

    def test_mixed_mostly_numeric_still_numeric(self):
        # One bad value shouldn't flip a clearly-numeric column to "text"
        s = pd.Series(["1", "2", "3", "4", "5", "6", "7", "8", "9", "oops"])
        assert infer_column_type(s) == "numeric"


# ── validate_mapping ───────────────────────────────────────────────────────────

class TestValidateMapping:
    def test_good_mapping_reports_ok(self):
        df = pd.DataFrame({
            "review_text": ["This is a fairly detailed review of the product experience."] * 5,
            "rating": [5, 4, 3, 2, 1],
        })
        mapping = {"review_text": "review_text", "rating": "rating",
                  "date": None, "source": None, "category": None}
        result = validate_mapping(df, mapping)
        assert result["review_text"]["ok"] is True
        assert result["rating"]["ok"] is True

    def test_wrong_rating_column_flagged(self):
        df = pd.DataFrame({
            "review_text": ["Great product overall", "Terrible experience here"],
            "rating": ["five stars please", "not applicable"],  # not actually numeric
        })
        mapping = {"review_text": "review_text", "rating": "rating",
                  "date": None, "source": None, "category": None}
        result = validate_mapping(df, mapping)
        assert result["rating"]["ok"] is False

    def test_mostly_blank_text_column_flagged(self):
        df = pd.DataFrame({"review_text": ["", "", "", "good product"]})
        mapping = {"review_text": "review_text", "rating": None,
                  "date": None, "source": None, "category": None}
        result = validate_mapping(df, mapping)
        assert result["review_text"]["ok"] is False

    def test_unmapped_fields_omitted_from_results(self):
        df = pd.DataFrame({"review_text": ["good", "bad"]})
        mapping = {"review_text": "review_text", "rating": None,
                  "date": None, "source": None, "category": None}
        result = validate_mapping(df, mapping)
        assert "rating" not in result
        assert "date" not in result
