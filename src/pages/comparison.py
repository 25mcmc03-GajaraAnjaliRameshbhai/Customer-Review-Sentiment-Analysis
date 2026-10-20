"""
src/pages/comparison.py
-------------------------
⚖ Product Comparison — compares sentiment side-by-side across product
categories and review sources, so a user can quickly see which product
type or platform has the most complaints.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils import LABEL_COLORS


def render(active_df: pd.DataFrame) -> None:
    st.header("⚖️ Product Comparison")
    st.caption("Compare sentiment across product categories and review sources.")

    if active_df.empty:
        st.info("No data loaded. Visit **Review Explorer** to upload a CSV or load the sample dataset.")
        return

    # ── Category comparison ──────────────────────────────────────────────────
    if "category" in active_df.columns and active_df["category"].notna().any():
        st.subheader("📦 By Category")

        cat_agg = (
            active_df.groupby(["category", "predicted_label"])
            .size().reset_index(name="count")
        )
        fig = px.bar(
            cat_agg, x="category", y="count", color="predicted_label",
            barmode="group", color_discrete_map=LABEL_COLORS,
            title="Review Volume by Category and Sentiment",
        )
        st.plotly_chart(fig, width='stretch')

        # % positive per category — a normalised view that's fairer when
        # categories have very different review counts
        cat_totals = active_df.groupby("category").size()
        cat_pos    = active_df[active_df["predicted_label"] == "positive"].groupby("category").size()
        pos_rate   = (cat_pos / cat_totals * 100).fillna(0).round(1).reset_index(name="positive_pct")
        pos_rate.columns = ["category", "positive_pct"]
        pos_rate = pos_rate.sort_values("positive_pct", ascending=True)

        fig2 = px.bar(
            pos_rate, x="positive_pct", y="category", orientation="h",
            color_discrete_sequence=[LABEL_COLORS["positive"]],
            title="% Positive Reviews by Category",
            labels={"positive_pct": "% Positive"},
        )
        st.plotly_chart(fig2, width='stretch')
    else:
        st.info("Active dataset has no 'category' column — category comparison unavailable.")

    st.markdown("---")

    # ── Source comparison ─────────────────────────────────────────────────────
    if "source" in active_df.columns and active_df["source"].notna().any():
        st.subheader("🌐 By Source")

        src_agg = (
            active_df.groupby(["source", "predicted_label"])
            .size().reset_index(name="count")
        )
        fig3 = px.bar(
            src_agg, x="source", y="count", color="predicted_label",
            barmode="stack", color_discrete_map=LABEL_COLORS,
            title="Review Volume by Source and Sentiment",
        )
        st.plotly_chart(fig3, width='stretch')
    else:
        st.info("Active dataset has no 'source' column — source comparison unavailable.")
