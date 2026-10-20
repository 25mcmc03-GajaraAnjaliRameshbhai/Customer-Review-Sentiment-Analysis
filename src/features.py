"""
src/features.py
---------------
TF-IDF feature-extraction helpers for ReviewLens.
The vectorizer is embedded inside the scikit-learn Pipeline in train_model.py,
so this module mainly provides factory and persistence utilities.
"""

from pathlib import Path
from typing import Optional

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(
    max_features: int = 10_000,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """
    Create a configured TF-IDF vectorizer.

    Parameters
    ----------
    max_features : int
        Vocabulary cap.
    ngram_range : tuple
        (min_n, max_n) for n-gram extraction.
    min_df : int
        Ignore terms that appear in fewer than *min_df* documents.
    sublinear_tf : bool
        Apply log(1 + tf) scaling.

    Returns
    -------
    TfidfVectorizer (unfitted)
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=None,       # stop-words handled upstream in preprocess.py
        sublinear_tf=sublinear_tf,
        min_df=min_df,
    )


def save_vectorizer(vectorizer: TfidfVectorizer, path: Path) -> None:
    """Persist a fitted vectorizer to *path* using joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path)
    print(f"Vectorizer saved → {path}")


def load_vectorizer(path: Path) -> Optional[TfidfVectorizer]:
    """
    Load a previously saved vectorizer.

    Returns None if the file does not exist.
    """
    if not path.exists():
        return None
    return joblib.load(path)
