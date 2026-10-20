"""
src/dashboard_utils.py
-----------------------
Shared helper functions used by every page of the Streamlit dashboard:
loading cached model/data, running predictions, and building charts.

Keeping these in one place means each page file (src/pages/*.py) only
has to import what it needs and stays focused on layout — not logic.
"""

from collections import Counter
from typing import Optional

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from wordcloud import WordCloud

from src.explain import explain_prediction, score_terms_for_indices
from src.preprocess import preprocess_batch, preprocess_text
from src.utils import DATA_DIR, PIPELINE_PATH, load_metadata

matplotlib.use("Agg")


# ── Cached resource loaders ───────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load the trained sklearn Pipeline from disk (cached across reruns)."""
    if not PIPELINE_PATH.exists():
        return None
    return joblib.load(PIPELINE_PATH)


def train_pipeline_if_missing() -> bool:
    """
    Train the model once if ``models/pipeline.pkl`` is missing.

    Returns True when the trained pipeline exists after this call.
    """
    if PIPELINE_PATH.exists():
        return True

    dataset_path = DATA_DIR / "raw" / "reviews.csv"
    if not dataset_path.exists():
        return False

    from train import main as train_main

    train_main()
    return PIPELINE_PATH.exists()


@st.cache_data(show_spinner=False)
def load_metadata_cached():
    return load_metadata()


@st.cache_data(show_spinner=False)
def load_master_dataset(sample_size: int = 1500) -> pd.DataFrame:
    """
    Load a random sample of the real, merged Kaggle dataset
    (data/raw/reviews.csv) for use as the dashboard's default demo data.

    Using a sample (rather than the full ~30K rows) keeps the dashboard
    fast to render; the model was still trained on the full dataset.
    """
    path = DATA_DIR / "raw" / "reviews.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=7).reset_index(drop=True)
    return df


# ── Prediction helpers ────────────────────────────────────────────────────────

def predict_single(raw_text: str, pipeline) -> Optional[dict]:
    """Preprocess and explain a single review. Returns None on empty text."""
    clean = preprocess_text(raw_text)
    if not clean.strip():
        return None
    return explain_prediction(clean, pipeline)


def predict_batch(
    df: pd.DataFrame,
    pipeline,
    chunk_size: Optional[int] = None,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Run sentiment prediction on every row of *df*.

    Adds columns: predicted_label, confidence,
                  top_positive_terms, top_negative_terms.

    Parameters
    ----------
    chunk_size : int, optional
        If given, rows are processed in batches of this size instead of
        all at once. This keeps memory bounded and lets the caller show
        real progress on large uploads via *progress_callback*, rather
        than freezing on one big call.
    progress_callback : callable, optional
        Called as `progress_callback(rows_done, rows_total)` after each
        chunk. Ignored if *chunk_size* is not given.
    """
    if chunk_size is None or len(df) <= chunk_size:
        return _predict_chunk(df, pipeline)

    results = []
    total = len(df)
    for start in range(0, total, chunk_size):
        chunk = df.iloc[start:start + chunk_size]
        results.append(_predict_chunk(chunk, pipeline))
        if progress_callback is not None:
            progress_callback(min(start + chunk_size, total), total)

    return pd.concat(results, ignore_index=True)


def _predict_chunk(df: pd.DataFrame, pipeline) -> pd.DataFrame:
    """Run prediction on a single (already-sized) chunk of rows."""
    raw_texts   = df["review_text"].fillna("").astype(str).tolist()
    clean_texts = preprocess_batch(raw_texts)

    predictions = pipeline.predict(clean_texts)
    probas      = pipeline.predict_proba(clean_texts)

    df = df.copy()
    df["predicted_label"] = predictions
    df["confidence"]      = np.round(np.max(probas, axis=1), 3)

    # -- Efficient per-row term extraction via LR coefficients -----------------
    # Uses the same scoring logic as the single-review explainer
    # (src/explain.py::score_terms_for_indices) so bulk CSV results and
    # single-review results are always consistent — no separate,
    # potentially-drifting copy of the coefficient math here.
    top_pos_list: list = []
    top_neg_list: list = []
    top_neu_list: list = []

    vectorizer = pipeline.named_steps.get("tfidf")
    clf        = pipeline.named_steps.get("clf")

    if hasattr(clf, "coef_") and vectorizer is not None:
        classes       = list(clf.classes_)
        feature_names = vectorizer.get_feature_names_out()
        tfidf_matrix  = vectorizer.transform(clean_texts)

        for i in range(len(clean_texts)):
            row_nz = tfidf_matrix[i].nonzero()[1]
            pos_terms, neg_terms, neu_terms = score_terms_for_indices(
                row_nz, feature_names, clf, classes, n_terms=5
            )
            top_pos_list.append(", ".join(pos_terms))
            top_neg_list.append(", ".join(neg_terms))
            top_neu_list.append(", ".join(neu_terms))
    else:
        top_pos_list = [""] * len(clean_texts)
        top_neg_list = [""] * len(clean_texts)
        top_neu_list = [""] * len(clean_texts)

    df["top_positive_terms"] = top_pos_list
    df["top_negative_terms"] = top_neg_list
    df["top_neutral_terms"]  = top_neu_list
    return df


# ── Chart / visual helpers ────────────────────────────────────────────────────

def make_wordcloud(text_series: pd.Series, colormap: str = "Greens") -> plt.Figure:
    """Generate a word-cloud figure from a Series of raw review strings."""
    all_text = " ".join(preprocess_batch(text_series.fillna("").astype(str).tolist()))
    fig, ax  = plt.subplots(figsize=(10, 4))

    if not all_text.strip():
        ax.text(0.5, 0.5, "No text available",
                ha="center", va="center", fontsize=14, color="grey")
        ax.axis("off")
        return fig

    wc = WordCloud(
        width=900, height=400, background_color="white",
        colormap=colormap, max_words=120, collocations=False,
    ).generate(all_text)

    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def top_words_df(clean_texts: list, n: int = 15) -> pd.DataFrame:
    """Return a DataFrame of the *n* most frequent words in *clean_texts*."""
    counter = Counter(w for t in clean_texts for w in t.split() if len(w) > 2)
    return pd.DataFrame(counter.most_common(n), columns=["word", "count"])


# ── Session-state active dataset ──────────────────────────────────────────────
# The "active dataset" is whatever the dashboard is currently analysing:
# either a sample of the real training corpus (default) or a CSV the user
# uploaded in the Review Explorer page. All other pages (Trends, Comparison,
# Summary, Export) read from this same session_state entry so the whole
# dashboard stays in sync.

def get_active_dataset(pipeline) -> pd.DataFrame:
    """Return the current active, annotated dataset — creating a default
    sample from the real training corpus on first load."""
    if "active_df" not in st.session_state:
        sample = load_master_dataset(sample_size=1500)
        if not sample.empty:
            st.session_state["active_df"] = predict_batch(sample, pipeline)
            st.session_state["active_df_name"] = "Sample of training data (real Kaggle reviews)"
        else:
            st.session_state["active_df"] = pd.DataFrame()
            st.session_state["active_df_name"] = "No data loaded"
    return st.session_state["active_df"]


def set_active_dataset(df: pd.DataFrame, name: str) -> None:
    """Replace the active dataset (e.g. after a CSV upload)."""
    st.session_state["active_df"] = df
    st.session_state["active_df_name"] = name


def reset_active_dataset(pipeline) -> None:
    """Reset the active dataset back to a fresh sample of the real corpus."""
    sample = load_master_dataset(sample_size=1500)
    st.session_state["active_df"] = predict_batch(sample, pipeline)
    st.session_state["active_df_name"] = "Sample of training data (real Kaggle reviews)"
