"""
src/pages/trends.py
----------------------
📈 Trends & Charts — the visual analytics core of the dashboard:
sentiment distribution, top keywords, word clouds, and sentiment
over time (when a date column is available in the active dataset).
"""

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard_utils import make_wordcloud, top_words_df
from src.preprocess import preprocess_batch
from src.utils import LABEL_COLORS


def render(active_df: pd.DataFrame) -> None:
    st.header("📈 Trends & Charts")

    if active_df.empty:
        st.info("No data loaded. Visit **Review Explorer** to upload a CSV or load the sample dataset.")
        return

    # ── Sentiment distribution ───────────────────────────────────────────────
    st.subheader("Sentiment Distribution")
    vc = active_df["predicted_label"].value_counts().reset_index()
    vc.columns = ["sentiment", "count"]

    pie_col, bar_col = st.columns(2)
    with pie_col:
        fig_pie = px.pie(
            vc, values="count", names="sentiment", color="sentiment",
            color_discrete_map=LABEL_COLORS, hole=0.35, title="Sentiment Share",
        )
        fig_pie.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_pie, width='stretch')
    with bar_col:
        fig_bar = px.bar(
            vc, x="sentiment", y="count", color="sentiment",
            color_discrete_map=LABEL_COLORS, text="count", title="Sentiment Count",
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, width='stretch')

    # ── Top keywords ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔑 Top Keywords")

    pos_mask  = active_df["predicted_label"] == "positive"
    neg_mask  = active_df["predicted_label"] == "negative"
    pos_texts = preprocess_batch(active_df.loc[pos_mask, "review_text"].tolist())
    neg_texts = preprocess_batch(active_df.loc[neg_mask, "review_text"].tolist())

    kw1, kw2 = st.columns(2)
    with kw1:
        st.markdown("##### ✅ Top Positive Words")
        if pos_texts:
            pw_df = top_words_df(pos_texts)
            fig = px.bar(pw_df, x="count", y="word", orientation="h",
                        color_discrete_sequence=[LABEL_COLORS["positive"]], height=420)
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, yaxis_title="")
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No positive reviews found.")

    with kw2:
        st.markdown("##### ❌ Top Negative Words")
        if neg_texts:
            nw_df = top_words_df(neg_texts)
            fig = px.bar(nw_df, x="count", y="word", orientation="h",
                        color_discrete_sequence=[LABEL_COLORS["negative"]], height=420)
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, yaxis_title="")
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No negative reviews found.")

    # ── Word clouds ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("☁️ Word Clouds")
    wc1, wc2 = st.columns(2)
    with wc1:
        st.markdown("**Positive Reviews**")
        if pos_mask.any():
            fig = make_wordcloud(active_df.loc[pos_mask, "review_text"], "Greens")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No positive reviews for word cloud.")
    with wc2:
        st.markdown("**Negative Reviews**")
        if neg_mask.any():
            fig = make_wordcloud(active_df.loc[neg_mask, "review_text"], "Reds")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No negative reviews for word cloud.")

    # ── Sentiment over time (adaptive — only if 'date' column present) ────────
    if "date" in active_df.columns and active_df["date"].notna().any():
        st.markdown("---")
        st.subheader("📅 Sentiment Over Time")
        try:
            df_time = active_df.copy()
            df_time["date"] = pd.to_datetime(df_time["date"], errors="coerce")
            df_time = df_time.dropna(subset=["date"])
            df_time["month"] = df_time["date"].dt.to_period("M").astype(str)
            time_agg = df_time.groupby(["month", "predicted_label"]).size().reset_index(name="count")

            fig = px.line(
                time_agg, x="month", y="count", color="predicted_label",
                color_discrete_map=LABEL_COLORS, markers=True, title="Monthly Sentiment Trend",
            )
            fig.update_xaxes(tickangle=45, title="Month")
            fig.update_yaxes(title="Review Count")
            st.plotly_chart(fig, width='stretch')
        except Exception:
            st.warning("⚠️ Could not parse the 'date' column for a time trend.")

    # ── Model confidence distribution ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("🎯 Prediction Confidence")
    st.caption(
        "How certain the model is across all predictions. A cluster near 100% "
        "means clear-cut reviews; a cluster near 33–50% means many ambiguous ones."
    )
    fig_conf = px.histogram(
        active_df, x="confidence", color="predicted_label", nbins=25,
        color_discrete_map=LABEL_COLORS, opacity=0.75,
        labels={"confidence": "Model Confidence"},
    )
    fig_conf.update_layout(barmode="overlay", yaxis_title="Number of Reviews")
    st.plotly_chart(fig_conf, width='stretch')
    st.caption(f"Average confidence across all predictions: **{active_df['confidence'].mean():.1%}**")

    # ── Rating vs. sentiment mismatch (only if 'rating' column present) ───────
    if "rating" in active_df.columns and active_df["rating"].notna().any():
        st.markdown("---")
        st.subheader("🚩 Rating vs. Sentiment Mismatch")
        st.caption(
            "Reviews where the star rating disagrees with the text's predicted "
            "sentiment — e.g. a 5-star rating with negative-sounding text. These "
            "are worth a manual look: they can indicate sarcasm, mis-clicked "
            "star ratings, or fake/incentivized reviews."
        )

        df_r = active_df.dropna(subset=["rating"]).copy()

        def _rating_bucket(r):
            if r >= 4:
                return "positive"
            if r <= 2:
                return "negative"
            return "neutral"

        df_r["rating_sentiment"] = df_r["rating"].apply(_rating_bucket)
        mismatched = df_r[df_r["rating_sentiment"] != df_r["predicted_label"]]

        m1, m2 = st.columns(2)
        m1.metric("Mismatched Reviews", f"{len(mismatched):,}")
        m2.metric("% of Rated Reviews", f"{len(mismatched) / len(df_r):.1%}" if len(df_r) else "0%")

        if not mismatched.empty:
            heat = (
                df_r.groupby(["rating_sentiment", "predicted_label"])
                .size().reset_index(name="count")
            )
            fig_heat = px.density_heatmap(
                df_r, x="rating_sentiment", y="predicted_label", z=None,
                category_orders={
                    "rating_sentiment": ["negative", "neutral", "positive"],
                    "predicted_label": ["negative", "neutral", "positive"],
                },
                labels={"rating_sentiment": "Rating implies", "predicted_label": "Model predicted"},
                title="Rating-implied sentiment vs. Model-predicted sentiment",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig_heat, width='stretch')

            with st.expander(f"👀 View {min(len(mismatched), 100)} mismatched reviews"):
                show_cols = ["review_text", "rating", "rating_sentiment", "predicted_label", "confidence"]
                st.dataframe(
                    mismatched[show_cols].head(100).rename(columns={
                        "rating_sentiment": "rating_implies", "predicted_label": "model_predicted",
                    }),
                    width='stretch', hide_index=True,
                )
        else:
            st.success("No mismatches found — ratings and predicted sentiment agree throughout.")
