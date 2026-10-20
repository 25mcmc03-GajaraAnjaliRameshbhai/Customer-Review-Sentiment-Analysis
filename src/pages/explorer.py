"""
src/pages/explorer.py
-----------------------
🔍 Review Explorer — the bulk-analysis entry point.

Two things happen here:
  1. The user can upload ANY CSV — column names and row count don't
     need to match a fixed schema. src/csv_mapper.py guesses which
     column holds the review text (and rating/date/source/category,
     if present), validates the guess against the actual data (not
     just the column name), and the user confirms or corrects the
     mapping before analysis runs. This becomes the dashboard's
     "active dataset" for every other page — Trends, Comparison,
     Summary, Export.
  2. The user can browse, search, and filter whichever dataset is
     currently active (uploaded CSV, or the default real-data sample).

Upload limits (file size, row count) are enforced here and are always
shown to the user up front — see src/utils.py for the exact numbers.
"""

import pandas as pd
import streamlit as st

from src.csv_mapper import apply_mapping, detect_columns, validate_mapping
from src.dashboard_utils import predict_batch, reset_active_dataset, set_active_dataset
from src.utils import (
    MAX_UPLOAD_ROWS,
    MAX_UPLOAD_SIZE_MB,
    PREDICT_CHUNK_SIZE,
    RECOMMENDED_MAX_ROWS,
)


def render(pipeline) -> None:
    st.header("🔍 Review Explorer")
    st.markdown(
        "Upload **any** CSV to analyse it — column names don't need to match "
        "anything specific, the app will detect them — or browse/filter/search "
        "the dataset currently loaded in the dashboard."
    )

    _render_upload_info_panel()

    # ── CSV upload ────────────────────────────────────────────────────────────
    uploaded = st.file_uploader("⬆️ Upload reviews CSV", type=["csv"])

    if uploaded is not None:
        _handle_upload(uploaded, pipeline)

    if st.button("↩️ Reset to Sample Dataset"):
        reset_active_dataset(pipeline)
        st.success("Reset to a sample of the real training data.")
        st.rerun()

    st.markdown("---")

    active_df = st.session_state.get("active_df", pd.DataFrame())
    dataset_name = st.session_state.get("active_df_name", "No data loaded")

    if active_df.empty:
        st.info("No data to explore yet.")
        return

    st.markdown(f"**Currently exploring:** {dataset_name}  ·  **{len(active_df):,} reviews**")

    # ── Filters ───────────────────────────────────────────────────────────────
    f1, f2, f3, f4 = st.columns(4)

    with f1:
        sentiment_filter = st.multiselect(
            "Sentiment", options=["positive", "negative", "neutral"],
            default=["positive", "negative", "neutral"],
        )
    with f2:
        category_options = sorted(active_df["category"].dropna().unique()) if "category" in active_df.columns else []
        category_filter = st.multiselect("Category", options=category_options, default=category_options)
    with f3:
        source_options = sorted(active_df["source"].dropna().unique()) if "source" in active_df.columns else []
        source_filter = st.multiselect("Source", options=source_options, default=source_options)
    with f4:
        search_term = st.text_input("Search text", placeholder="e.g. battery")

    filtered = active_df[active_df["predicted_label"].isin(sentiment_filter)]
    if category_filter:
        filtered = filtered[filtered["category"].isin(category_filter)]
    if source_filter:
        filtered = filtered[filtered["source"].isin(source_filter)]
    if search_term.strip():
        filtered = filtered[filtered["review_text"].str.contains(search_term, case=False, na=False)]

    st.caption(f"Showing {len(filtered):,} of {len(active_df):,} reviews")

    display_cols = ["review_text", "predicted_label", "confidence"]
    for col in ["rating", "date", "source", "category"]:
        if col in filtered.columns:
            display_cols.insert(-2, col)

    st.dataframe(
        filtered[display_cols].head(300),
        width='stretch',
        hide_index=True,
        height=450,
    )
    if len(filtered) > 300:
        st.caption(f"Showing first 300 of {len(filtered):,} matching rows.")


def _render_upload_info_panel() -> None:
    """
    Always-visible panel telling the user exactly where to upload,
    what's supported, and the practical limits — per the project's
    upload-transparency requirements.
    """
    st.info(
        f"**📤 Upload location:** use the file picker directly below.\n\n"
        f"**📄 Supported format:** `.csv` only (comma-separated values).\n\n"
        f"**📦 Maximum file size:** {MAX_UPLOAD_SIZE_MB} MB per upload.\n\n"
        f"**📊 Row limits:** works with any size up to **{MAX_UPLOAD_ROWS:,} rows**. "
        f"For the fastest results, we recommend keeping uploads under "
        f"**{RECOMMENDED_MAX_ROWS:,} rows** — larger files are still processed "
        f"automatically in batches with a progress bar, just more slowly.\n\n"
        f"**🧭 Column names:** don't need to match anything specific — any CSV "
        f"with at least one column of review text works. Rating, date, source, "
        f"and category columns are optional and auto-detected if present."
    )


def _handle_upload(uploaded, pipeline) -> None:
    """
    Read the uploaded CSV, validate it (size, structure, content),
    auto-detect its columns, let the user confirm or correct the
    mapping, then run predictions (chunked, with progress) once
    confirmed.
    """
    # ── 1. File size check (before even trying to parse) ──────────────────────
    size_mb = uploaded.size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        st.error(
            f"❌ File is {size_mb:.1f} MB, which exceeds the "
            f"{MAX_UPLOAD_SIZE_MB} MB limit. Please split it into smaller "
            f"files and upload them one at a time."
        )
        return

    # ── 2. Parse the CSV, with a clear error for every failure mode ───────────
    df_raw = _read_csv_safely(uploaded)
    if df_raw is None:
        return  # error already shown by _read_csv_safely

    if len(df_raw.columns) == 0:
        st.error("❌ The uploaded file has no columns. Please check the file and try again.")
        return

    if df_raw.empty:
        st.error("❌ The uploaded file has columns but no data rows.")
        return

    # ── 3. Row count check ─────────────────────────────────────────────────────
    if len(df_raw) > MAX_UPLOAD_ROWS:
        st.error(
            f"❌ File has {len(df_raw):,} rows, which exceeds the "
            f"{MAX_UPLOAD_ROWS:,}-row limit. Please split it into smaller "
            f"files and upload them one at a time."
        )
        return

    if len(df_raw) > RECOMMENDED_MAX_ROWS:
        st.warning(
            f"⚠️ This file has {len(df_raw):,} rows — larger than the "
            f"recommended {RECOMMENDED_MAX_ROWS:,} for fast processing. "
            f"It will still work, just take longer. Processing happens in "
            f"batches with a progress bar below once you click Analyze."
        )

    # ── 4. Detect + let the user confirm column mapping ───────────────────────
    suggested = detect_columns(df_raw)
    all_columns = list(df_raw.columns)

    st.markdown("#### 🧭 Confirm Column Mapping")
    st.caption(
        "Detected automatically from your file's column names and content — "
        "correct any of these if they look wrong, then click Analyze."
    )

    def _selectbox_for(field: str, label: str, required: bool):
        options = (["— none —"] if not required else []) + all_columns
        guess = suggested.get(field)
        default_index = options.index(guess) if guess in options else 0
        choice = st.selectbox(label, options, index=default_index, key=f"map_{field}")
        return None if choice == "— none —" else choice

    c1, c2 = st.columns(2)
    with c1:
        text_col = _selectbox_for("review_text", "Review text column *", required=True)
        rating_col = _selectbox_for("rating", "Rating column (optional)", required=False)
        date_col = _selectbox_for("date", "Date column (optional)", required=False)
    with c2:
        source_col = _selectbox_for("source", "Source column (optional)", required=False)
        category_col = _selectbox_for("category", "Category column (optional)", required=False)

    mapping = {
        "review_text": text_col,
        "rating":      rating_col,
        "date":        date_col,
        "source":      source_col,
        "category":    category_col,
    }

    # ── 5. Validate the mapping against the actual data ────────────────────────
    if text_col is None:
        st.error("❌ No column detected for review text. Please select one manually — it's required.")
        return

    validation = validate_mapping(df_raw, mapping)
    _render_validation_feedback(validation)

    with st.expander("👀 Preview raw file (first 5 rows)"):
        st.dataframe(df_raw.head(5), width='stretch')

    if not st.button("✅ Analyze Uploaded CSV", type="primary"):
        return

    if not validation.get("review_text", {}).get("ok", True):
        st.error(
            "❌ The selected review text column doesn't look usable "
            "(mostly blank). Please pick a different column."
        )
        return

    mapped_df = apply_mapping(df_raw, mapping)
    mapped_df["review_text"] = mapped_df["review_text"].fillna("").astype(str)
    mapped_df = mapped_df[mapped_df["review_text"].str.strip().astype(bool)].reset_index(drop=True)

    if mapped_df.empty:
        st.error("❌ No usable review text found in the selected column.")
        return

    dropped = len(df_raw) - len(mapped_df)
    if dropped > 0:
        st.caption(f"ℹ️ Skipped {dropped:,} row(s) with blank review text.")

    # ── 6. Run predictions — chunked with a real progress bar for large files ─
    total = len(mapped_df)
    if total > PREDICT_CHUNK_SIZE:
        progress_bar = st.progress(0, text=f"Analyzing 0 / {total:,} reviews …")

        def _update_progress(done: int, grand_total: int) -> None:
            progress_bar.progress(
                done / grand_total, text=f"Analyzing {done:,} / {grand_total:,} reviews …"
            )

        annotated = predict_batch(
            mapped_df, pipeline, chunk_size=PREDICT_CHUNK_SIZE, progress_callback=_update_progress
        )
        progress_bar.empty()
    else:
        with st.spinner(f"Analyzing {total:,} reviews …"):
            annotated = predict_batch(mapped_df, pipeline)

    set_active_dataset(annotated, f"Uploaded file: {uploaded.name}")
    st.success(f"✅ Analyzed {len(annotated):,} reviews. Explore them below, or check other pages.")
    st.rerun()


def _read_csv_safely(uploaded):
    """
    Parse the uploaded file into a DataFrame, handling the most common
    real-world failure modes with a specific, actionable message for
    each rather than one generic error.
    """
    try:
        return pd.read_csv(uploaded)
    except UnicodeDecodeError:
        # Common for files exported from Excel on Windows — not strictly UTF-8.
        try:
            uploaded.seek(0)
            return pd.read_csv(uploaded, encoding="latin-1")
        except Exception as exc:
            st.error(f"❌ Could not read the file's text encoding: {exc}")
            return None
    except pd.errors.EmptyDataError:
        st.error("❌ The uploaded file is empty or has no readable content.")
        return None
    except pd.errors.ParserError as exc:
        st.error(
            f"❌ This file doesn't look like a valid CSV — it may be corrupted, "
            f"or in a different format (e.g. an Excel file renamed to .csv). "
            f"Details: {exc}"
        )
        return None
    except Exception as exc:
        st.error(f"❌ Could not read the CSV: {exc}")
        return None


def _render_validation_feedback(validation: dict) -> None:
    """Show a compact ✅/⚠️ line per mapped field based on validate_mapping()."""
    if not validation:
        return
    lines = []
    for field, result in validation.items():
        icon = "✅" if result["ok"] else "⚠️"
        field_label = field.replace("_", " ").title()
        lines.append(f"{icon} **{field_label}** — {result['message']}")
    st.markdown("  \n".join(lines))
