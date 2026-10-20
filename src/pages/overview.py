"""
src/pages/overview.py
----------------------
📊 Overview — landing page. Gives a quick snapshot of the currently
active dataset and the trained model, so the first thing a viewer sees
is "what am I looking at" before diving into any specific analysis.
"""

import pandas as pd
import streamlit as st

from src.utils import get_emoji_for_label


def render(active_df, metadata, dataset_name: str) -> None:
    st.header("📊 Overview")
    st.caption(
        "A quick snapshot of the currently active dataset and the trained model."
    )

    if active_df.empty:
        st.warning("No data loaded yet.")
        return

    st.markdown(f"**Active dataset:** {dataset_name}")

    # ── Dataset snapshot ──────────────────────────────────────────────────────
    total = len(active_df)
    vc     = active_df["predicted_label"].value_counts()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", f"{total:,}")
    c2.metric("🟢 Positive", f"{vc.get('positive', 0) / total:.1%}")
    c3.metric("🔴 Negative", f"{vc.get('negative', 0) / total:.1%}")
    c4.metric("🟡 Neutral", f"{vc.get('neutral', 0) / total:.1%}")

    if "category" in active_df.columns:
        st.markdown(f"**Categories present:** {', '.join(sorted(active_df['category'].dropna().unique()))}")
    if "source" in active_df.columns:
        st.markdown(f"**Sources present:** {', '.join(sorted(active_df['source'].dropna().unique()))}")
    if "date" in active_df.columns and active_df["date"].notna().any():
        parsed_dates = pd.to_datetime(active_df["date"], errors="coerce").dropna()
        if not parsed_dates.empty:
            st.markdown(
                f"**Date range:** {parsed_dates.min().strftime('%Y-%m-%d')} "
                f"to {parsed_dates.max().strftime('%Y-%m-%d')}"
            )
    st.markdown(f"**Average model confidence:** {active_df['confidence'].mean():.1%}")

    st.markdown("---")

    # ── Model snapshot ────────────────────────────────────────────────────────
    st.subheader("🤖 Model Snapshot")
    if metadata:
        m1, m2, m3 = st.columns(3)
        m1.metric("Model", metadata.get("model_name", "—"))
        m2.metric("Accuracy", f"{metadata.get('metrics', {}).get('accuracy', 0):.1%}")
        m3.metric("Macro F1", f"{metadata.get('metrics', {}).get('macro_f1', 0):.3f}")
        st.caption(
            f"Trained on {metadata.get('number_of_training_samples', 0):,} real reviews "
            "collected from Amazon, Flipkart, App Store, Play Store, Coursera, Zomato, and IMDB."
        )
    else:
        st.info("Model metadata not found. Run `python train.py`.")

    st.markdown("---")

    # ── Recent reviews preview ───────────────────────────────────────────────
    st.subheader("🔎 Sample Reviews")
    preview = active_df[["review_text", "predicted_label", "confidence"]].head(5).copy()
    preview["predicted_label"] = preview["predicted_label"].apply(
        lambda x: f"{get_emoji_for_label(x)} {x.capitalize()}"
    )
    st.dataframe(preview, width='stretch', hide_index=True)
