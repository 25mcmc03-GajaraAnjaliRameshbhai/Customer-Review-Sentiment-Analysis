"""
src/csv_mapper.py
-------------------
Auto-detects which columns in an arbitrary, user-uploaded CSV correspond
to ReviewLens's schema (review_text, rating, date, source, category),
so the dashboard can accept **any** CSV — not just one with an exact,
pre-agreed column layout.

Detection strategy (in order):
  1. Keyword matching — column names are compared (case-insensitively,
     ignoring spaces/underscores) against a list of common names for
     each field (e.g. "review", "comment", "feedback" all mean
     review_text; "stars", "score", "rating" all mean rating).
  2. Fallback for review_text only — if no column name matches any
     keyword, the column with the longest average string length is
     assumed to be the text column (review text is almost always the
     longest free-text field in a reviews CSV).

The result is always just a *suggestion*. src/pages/explorer.py shows
it to the user in dropdowns so they can confirm or override it before
any analysis runs — auto-detection is a convenience, not a silent
guess baked into the pipeline.
"""

from typing import Dict, Optional

import pandas as pd

# Keywords checked against normalised column names (lowercase, no
# spaces/underscores/hyphens) for each schema field. Order matters
# only in that earlier keywords are just as valid as later ones —
# any match counts.
FIELD_KEYWORDS: Dict[str, list] = {
    "review_text": [
        "reviewtext", "review", "text", "comment", "feedback",
        "description", "content", "summary", "message", "body", "opinion",
    ],
    "rating": [
        "rating", "rate", "stars", "star", "score", "overall", "liked",
    ],
    "date": [
        "date", "time", "timestamp", "reviewdate", "createdat", "postedon",
        "purchasedon", "ordered", "purchased",
    ],
    "source": [
        "source", "platform", "site", "channel", "window", "store",
        "seller", "vendor", "retailer", "boughtfrom", "marketplace",
    ],
    "category": [
        "category", "product", "productname", "type", "genre", "department",
    ],
}


def _normalise(col_name: str) -> str:
    """Lowercase and strip spaces/underscores/hyphens for keyword matching."""
    return str(col_name).lower().replace(" ", "").replace("_", "").replace("-", "")


def _best_keyword_match(columns: list, keywords: list) -> Optional[str]:
    """Return the first column whose normalised name contains any keyword."""
    normalised = {col: _normalise(col) for col in columns}
    for keyword in keywords:
        for col, norm in normalised.items():
            if keyword in norm:
                return col
    return None


def _longest_text_column(df: pd.DataFrame, exclude: list) -> Optional[str]:
    """
    Fallback: pick the string-like column with the longest average
    string length, excluding any columns already assigned to another
    field. Used only when no column name matches a review_text keyword.
    """
    candidates = [
        c for c in df.columns
        if c not in exclude and (pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object)
    ]
    if not candidates:
        return None

    avg_lengths = {
        c: df[c].astype(str).str.len().mean() for c in candidates
    }
    return max(avg_lengths, key=avg_lengths.get)


def infer_column_type(series: pd.Series) -> str:
    """
    Infer the practical data type of a column by inspecting its actual
    values — not just its pandas dtype (which is unreliable for data
    read from CSV, where everything can arrive as strings).

    Returns one of: "numeric", "date", "text", "empty".

    This is what lets ReviewLens validate a user's column choice
    against real content (e.g. warn if the column picked as "rating"
    is mostly non-numeric) rather than trusting the column name alone.
    """
    non_null = series.dropna()
    non_null = non_null[non_null.astype(str).str.strip() != ""]

    if len(non_null) == 0:
        return "empty"

    numeric_ok = pd.to_numeric(non_null, errors="coerce").notna().mean()
    if numeric_ok >= 0.9:
        return "numeric"

    date_ok = pd.to_datetime(non_null, errors="coerce", format="mixed").notna().mean()
    if date_ok >= 0.9:
        return "date"

    return "text"


def validate_mapping(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> Dict[str, dict]:
    """
    Validate a column mapping against the *actual data*, not just the
    column names. For each mapped field, reports whether the selected
    column's content actually looks like what that field expects.

    Returns
    -------
    dict keyed by field name, each value a dict with:
        ok               : bool   — whether the column looks usable
        message          : str    — human-readable summary for the UI
        detected_type    : str    — infer_column_type() result
    Unmapped fields are omitted.
    """
    results: Dict[str, dict] = {}

    text_col = mapping.get("review_text")
    if text_col is not None and text_col in df.columns:
        series = df[text_col].fillna("").astype(str)
        blank_pct = (series.str.strip() == "").mean() * 100
        avg_len = series.str.len().mean()
        ok = blank_pct < 50 and avg_len >= 3
        results["review_text"] = {
            "ok": bool(ok),
            "detected_type": "text",
            "message": (
                f"{blank_pct:.0f}% of rows are blank in this column — "
                "this may not be the right column for review text."
                if not ok else
                f"Looks good — average length {avg_len:.0f} characters, {blank_pct:.0f}% blank."
            ),
        }

    rating_col = mapping.get("rating")
    if rating_col is not None and rating_col in df.columns:
        detected = infer_column_type(df[rating_col])
        unparseable_pct = (1 - pd.to_numeric(df[rating_col], errors="coerce").notna().mean()) * 100
        ok = detected == "numeric" and unparseable_pct < 30
        results["rating"] = {
            "ok": bool(ok),
            "detected_type": detected,
            "message": (
                f"{unparseable_pct:.0f}% of values are not numeric — "
                "this may not be a rating column."
                if not ok else
                f"Looks numeric ({100 - unparseable_pct:.0f}% valid values)."
            ),
        }

    date_col = mapping.get("date")
    if date_col is not None and date_col in df.columns:
        detected = infer_column_type(df[date_col])
        unparseable_pct = (1 - pd.to_datetime(df[date_col], errors="coerce", format="mixed").notna().mean()) * 100
        ok = detected == "date" and unparseable_pct < 30
        results["date"] = {
            "ok": bool(ok),
            "detected_type": detected,
            "message": (
                f"{unparseable_pct:.0f}% of values don't parse as dates."
                if not ok else
                f"Looks like valid dates ({100 - unparseable_pct:.0f}% parseable)."
            ),
        }

    return results


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Guess which column in *df* corresponds to each ReviewLens schema
    field.

    Returns
    -------
    dict with keys: review_text, rating, date, source, category
    Each value is either a column name from *df*, or None if no
    confident guess could be made (the field is left unmapped).
    """
    columns = list(df.columns)
    mapping: Dict[str, Optional[str]] = {}

    # review_text is required — try keywords first, then the longest-text fallback
    mapping["review_text"] = _best_keyword_match(columns, FIELD_KEYWORDS["review_text"])
    if mapping["review_text"] is None:
        mapping["review_text"] = _longest_text_column(df, exclude=[])

    used = {mapping["review_text"]}

    # Optional fields — keyword match only, skipping columns already used
    for field in ("rating", "date", "source", "category"):
        remaining_cols = [c for c in columns if c not in used]
        match = _best_keyword_match(remaining_cols, FIELD_KEYWORDS[field])
        mapping[field] = match
        if match is not None:
            used.add(match)

    return mapping


def apply_mapping(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    """
    Build a new DataFrame in ReviewLens's standard schema from *df*
    using the given column *mapping* (as returned by detect_columns,
    possibly edited by the user).

    Only fields with a non-None mapping are included; unmapped
    optional fields (rating/date/source/category) are simply omitted
    — the rest of the dashboard already treats these as optional.

    Rating and date columns are coerced to proper numeric / datetime
    types here (invalid values become NaN/NaT) so every downstream
    page can rely on clean types instead of re-parsing messy raw
    strings repeatedly.
    """
    out = pd.DataFrame()
    out["review_text"] = df[mapping["review_text"]].astype(str)

    rating_col = mapping.get("rating")
    if rating_col is not None and rating_col in df.columns:
        out["rating"] = pd.to_numeric(df[rating_col], errors="coerce")

    date_col = mapping.get("date")
    if date_col is not None and date_col in df.columns:
        parsed = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
        out["date"] = parsed.dt.strftime("%Y-%m-%d")

    for field in ("source", "category"):
        col = mapping.get(field)
        if col is not None and col in df.columns:
            out[field] = df[col]

    return out
