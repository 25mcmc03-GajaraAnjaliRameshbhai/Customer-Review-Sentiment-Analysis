"""
src/pages/summary.py
----------------------
📄 Summary — a plain-language, auto-generated summary of the active
dataset. Answers the question a business owner actually cares about:
"are customers happy, what do they complain about, and is anything
platform- or category-specific standing out?"
"""

import pandas as pd
import streamlit as st

from src.preprocess import preprocess_batch
from src.dashboard_utils import top_words_df


def render(active_df: pd.DataFrame, dataset_name: str) -> None:
    st.header("📄 Summary")
    st.caption(f"Auto-generated summary for: {dataset_name}")

    if active_df.empty:
        st.info("No data loaded. Visit **Review Explorer** to upload a CSV or load the sample dataset.")
        return

    total = len(active_df)
    vc    = active_df["predicted_label"].value_counts()
    pos_pct = vc.get("positive", 0) / total * 100
    neg_pct = vc.get("negative", 0) / total * 100
    neu_pct = vc.get("neutral", 0) / total * 100

    # ── Overall verdict ───────────────────────────────────────────────────────
    if pos_pct >= 60:
        verdict = "Overall sentiment is **strongly positive**."
    elif pos_pct >= 45:
        verdict = "Overall sentiment leans **positive**, with a meaningful negative minority."
    elif neg_pct >= 45:
        verdict = "Overall sentiment leans **negative** — this dataset needs attention."
    else:
        verdict = "Overall sentiment is **mixed**, with no single dominant group."

    st.markdown(f"### {verdict}")
    st.markdown(
        f"Out of **{total:,}** reviews analysed: "
        f"**{pos_pct:.1f}%** positive, **{neg_pct:.1f}%** negative, **{neu_pct:.1f}%** neutral."
    )

    # ── Top complaint / praise words ──────────────────────────────────────────
    neg_texts = preprocess_batch(
        active_df.loc[active_df["predicted_label"] == "negative", "review_text"].tolist()
    )
    pos_texts = preprocess_batch(
        active_df.loc[active_df["predicted_label"] == "positive", "review_text"].tolist()
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🏆 What customers praise most")
        if pos_texts:
            top_pos = top_words_df(pos_texts, n=8)
            st.markdown(", ".join(f"**{w}**" for w in top_pos["word"]))
        else:
            st.markdown("_No positive reviews to summarise._")

    with col2:
        st.markdown("#### ⚠️ What customers complain about most")
        if neg_texts:
            top_neg = top_words_df(neg_texts, n=8)
            st.markdown(", ".join(f"**{w}**" for w in top_neg["word"]))
        else:
            st.markdown("_No negative reviews to summarise._")

    # ── Category / source callouts ────────────────────────────────────────────
    st.markdown("---")

    if "category" in active_df.columns and active_df["category"].nunique() > 1:
        cat_totals = active_df.groupby("category").size()
        cat_pos    = active_df[active_df["predicted_label"] == "positive"].groupby("category").size()
        pos_rate   = (cat_pos / cat_totals * 100).fillna(0)

        if len(pos_rate) > 1:
            best_cat  = pos_rate.idxmax()
            worst_cat = pos_rate.idxmin()
            st.markdown(
                f"- 🏅 **Best-performing category:** {best_cat} ({pos_rate[best_cat]:.1f}% positive)\n"
                f"- 🚩 **Category needing attention:** {worst_cat} ({pos_rate[worst_cat]:.1f}% positive)"
            )

    # ── Trend direction — is sentiment improving or worsening? ────────────────
    if "date" in active_df.columns and active_df["date"].notna().any():
        _render_trend_section(active_df)

    # ── Confidence note ───────────────────────────────────────────────────────
    avg_conf = active_df["confidence"].mean()
    st.markdown("---")
    st.caption(
        f"Average model confidence across all predictions: **{avg_conf:.1%}**. "
        "Low-confidence predictions are usually short or ambiguous reviews."
    )


def _positive_rate_trend(df_time: pd.DataFrame):
    """
    Split *df_time* (must already have a parsed 'date' column, sorted)
    at its date midpoint and compare positive-sentiment share between
    the two halves.

    Returns None if there isn't enough data on both sides to say
    anything meaningful, otherwise (earlier_pct, later_pct, delta).
    """
    if len(df_time) < 20 or df_time["date"].nunique() < 2:
        return None

    midpoint = df_time["date"].min() + (df_time["date"].max() - df_time["date"].min()) / 2
    earlier = df_time[df_time["date"] <= midpoint]
    later   = df_time[df_time["date"] > midpoint]

    if len(earlier) < 5 or len(later) < 5:
        return None

    earlier_pct = (earlier["predicted_label"] == "positive").mean() * 100
    later_pct   = (later["predicted_label"] == "positive").mean() * 100
    return earlier_pct, later_pct, later_pct - earlier_pct


def _trend_sentence(earlier_pct: float, later_pct: float, delta: float, label_prefix: str = "") -> str:
    if abs(delta) < 3:
        return f"📊 {label_prefix}sentiment has stayed roughly steady over time (±{abs(delta):.1f} points)."
    elif delta > 0:
        return (f"📈 {label_prefix}sentiment is **improving** — positive share rose from "
                f"{earlier_pct:.1f}% to {later_pct:.1f}% over the period.")
    else:
        return (f"📉 {label_prefix}sentiment is **declining** — positive share fell from "
                f"{earlier_pct:.1f}% to {later_pct:.1f}% over the period.")


def _render_trend_section(active_df: pd.DataFrame) -> None:
    """
    Renders the "is sentiment improving or worsening" insight.

    Important: when the dataset spans multiple sources/platforms
    (e.g. Amazon + Play Store), an aggregate earlier-vs-later split is
    misleading if the two halves happen to be dominated by different
    sources with different baseline sentiment — that would look like a
    time trend but is really just a source-mix artifact. To avoid
    that, the trend is computed **per source** when more than one
    source is present, and only the aggregate figure is shown when
    there's just one (or no) source column — which is the common case
    for a real single-platform CSV upload.
    """
    df_time = active_df.copy()
    df_time["date"] = pd.to_datetime(df_time["date"], errors="coerce")
    df_time = df_time.dropna(subset=["date"]).sort_values("date")

    if df_time.empty:
        return

    has_multiple_sources = "source" in df_time.columns and df_time["source"].nunique() > 1

    if not has_multiple_sources:
        result = _positive_rate_trend(df_time)
        if result:
            st.markdown(f"**Trend:** {_trend_sentence(*result)}")
        return

    # Multiple sources: compute a trend per source, skip sources without
    # enough dated rows, and be explicit that mixed-source aggregates
    # aren't shown for this reason.
    rows = []
    for source_name, group in df_time.groupby("source"):
        result = _positive_rate_trend(group)
        if result:
            earlier_pct, later_pct, delta = result
            rows.append({
                "Source": source_name,
                "Earlier period": f"{earlier_pct:.1f}%",
                "Later period": f"{later_pct:.1f}%",
                "Change": f"{delta:+.1f} pts",
            })

    if not rows:
        return

    st.markdown("**📊 Trend (by source)**")
    st.caption(
        "Shown per source rather than as one combined trend, since this dataset "
        "mixes multiple platforms — combining them could make a shift in *which* "
        "platform contributed more reviews look like a change in sentiment over time."
    )
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
