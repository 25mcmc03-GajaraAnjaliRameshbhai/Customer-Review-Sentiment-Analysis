"""
app.py
------
ReviewLens — Sentiment Analysis and Insight Dashboard
Streamlit web application entry point.

This file is intentionally thin: it loads the trained model, sets up
the sidebar navigation, and dispatches to the relevant page module in
src/pages/. All page-specific logic lives in its own file, so each
sidebar item maps to exactly one file — easy to read, easy to explain,
and easy to change without touching the rest of the app.

Run:
    streamlit run app.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src.dashboard_utils import (
    get_active_dataset,
    load_metadata_cached,
    load_pipeline,
    train_pipeline_if_missing,
)
from src.pages import comparison, explorer, export, overview, performance, predict, summary, trends
from src.utils import PIPELINE_PATH

# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ReviewLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stMetricValue { font-size: 1.6rem !important; }
    .tag {
        display: inline-block; padding: 3px 10px; border-radius: 14px;
        margin: 3px 2px; font-size: 0.82rem; font-weight: 600;
    }
    .tag-pos { background: #d5f5e3; color: #1a7a3e; }
    .tag-neg { background: #fadbd8; color: #a93226; }
    .tag-neu { background: #fdebd0; color: #9c640c; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Sidebar navigation ────────────────────────────────────────────────────────

PAGES = {
    "📊 Overview":          "overview",
    "🤖 Predict Sentiment": "predict",
    "🔍 Review Explorer":   "explorer",
    "📈 Trends & Charts":   "trends",
    "⚖️ Product Comparison": "comparison",
    "📄 Summary":           "summary",
    "🧠 Model Performance": "performance",
    "📥 Export Reports":    "export",
}


def main() -> None:
    with st.sidebar:
        st.markdown("## 🔍 ReviewLens")
        st.caption("Sentiment Analysis & Insight Dashboard")
        st.markdown("Trained on **real Kaggle review data** — TF-IDF + Logistic Regression")
        st.markdown("---")

        selection = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

    st.title("🔍 ReviewLens — Sentiment Analysis & Insight Dashboard")

    # ── Model guard ───────────────────────────────────────────────────────────
    pipeline = load_pipeline()
    if pipeline is None:
        st.warning(
            "Trained model not found in this deployment. "
            "Starting one-time training from `data/raw/reviews.csv`."
        )
        with st.spinner("Training model (first run only). This may take 1-3 minutes..."):
            if train_pipeline_if_missing():
                load_pipeline.clear()
                load_metadata_cached.clear()
                pipeline = load_pipeline()

    if pipeline is None:
        st.error(
            "⚠️ **Model not found.**\n\n"
            "Automatic training could not run because required files are missing.\n\n"
            "Run these commands in your project environment, then redeploy:\n\n"
            "```\npython prepare_dataset.py\npython train.py\n```"
        )
        st.stop()

    metadata = load_metadata_cached()

    # Trends/Comparison/Summary/Export all share one "active dataset" so a
    # CSV uploaded in Review Explorer is instantly reflected everywhere else.
    active_df    = get_active_dataset(pipeline)
    dataset_name = st.session_state.get("active_df_name", "Sample dataset")

    # ── Route to the selected page ───────────────────────────────────────────
    page = PAGES[selection]

    # Re-read active_df from session_state each time, in case a CSV was just
    # uploaded on the Review Explorer page (Streamlit reruns top-to-bottom).
    active_df = st.session_state.get("active_df", active_df)

    if page == "overview":
        overview.render(active_df, metadata, dataset_name)
    elif page == "predict":
        predict.render(pipeline)
    elif page == "explorer":
        explorer.render(pipeline)
    elif page == "trends":
        trends.render(active_df)
    elif page == "comparison":
        comparison.render(active_df)
    elif page == "summary":
        summary.render(active_df, dataset_name)
    elif page == "performance":
        performance.render(metadata)
    elif page == "export":
        export.render(active_df, dataset_name)


if __name__ == "__main__":
    main()
