"""
src/preprocess.py
-----------------
Reusable text-preprocessing pipeline for ReviewLens.

Public API
----------
preprocess_text(text, lemmatize_text=True) -> str
preprocess_batch(texts, lemmatize_text=True) -> list[str]
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Optional, List
import nltk


# ── NLTK resource bootstrap ──────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_NLTK_DATA_DIR = _PROJECT_ROOT / ".nltk_data"
_NLTK_DATA_DIR.mkdir(exist_ok=True)

if str(_NLTK_DATA_DIR) not in nltk.data.path:
    nltk.data.path.insert(0, str(_NLTK_DATA_DIR))


def _ensure_nltk_resources() -> None:
    """Download required NLTK data into a repo-local cache."""
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for data_path, pkg_name in resources:
        try:
            nltk.data.find(data_path)
        except LookupError:
            try:
                nltk.download(pkg_name, quiet=True, download_dir=str(_NLTK_DATA_DIR))
            except Exception:
                pass


# Fallback stopwords keep preprocessing functional even when NLTK corpora are
# unavailable or blocked by the platform's security policy.
_FALLBACK_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "could", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "him", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "with", "you",
    "your", "yours", "yourself", "yourselves"
}

try:
    _ensure_nltk_resources()
    from nltk.corpus import stopwords          # noqa: E402
    from nltk.stem import WordNetLemmatizer    # noqa: E402
    _STOP_WORDS = set(stopwords.words("english"))
    _LEMMATIZER = WordNetLemmatizer()
except Exception:
    _STOP_WORDS = _FALLBACK_STOP_WORDS
    _LEMMATIZER = None

# ── Compile regexes once ─────────────────────────────────────────────────────

_RE_HTML       = re.compile(r"<[^>]+>")
_RE_URL        = re.compile(r"https?://\S+|www\.\S+")
_RE_SPECIAL    = re.compile(r"[^a-z0-9\s]")
_RE_WHITESPACE = re.compile(r"\s+")


# ── Individual transformation helpers ────────────────────────────────────────

def remove_html(text: str) -> str:
    """Strip HTML tags from *text*."""
    return _RE_HTML.sub(" ", text)


def remove_urls(text: str) -> str:
    """Remove http/https and www URLs from *text*."""
    return _RE_URL.sub(" ", text)


def remove_special_chars(text: str) -> str:
    """Keep only lowercase letters, digits and whitespace."""
    return _RE_SPECIAL.sub(" ", text)


def remove_extra_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines to a single space."""
    return _RE_WHITESPACE.sub(" ", text).strip()


def remove_stopwords(text: str) -> str:
    """Drop English stop-words from space-tokenised *text*."""
    return " ".join(w for w in text.split() if w not in _STOP_WORDS)


def lemmatize_text(text: str) -> str:
    """Lemmatize each token in space-tokenised *text*."""
    return " ".join(_lemmatize_word(w) for w in text.split())


@lru_cache(maxsize=50_000)
def _lemmatize_word(word: str) -> str:
    """
    Lemmatize a single word, cached.

    Review text is highly repetitive (the same common words — "great",
    "battery", "product" — appear thousands of times across a dataset),
    so memoising the per-word NLTK call is a simple, safe optimisation
    that meaningfully speeds up preprocessing on large CSV uploads
    without changing the output.
    """
    if _LEMMATIZER is None:
        return word
    return _LEMMATIZER.lemmatize(word)


# ── Main public function ─────────────────────────────────────────────────────

def preprocess_text(
    text: Optional[str],
    lemmatize: bool = True,
) -> str:
    """
    Full preprocessing pipeline.

    Steps applied in order:
      1. Guard against None / non-string input  → return ""
      2. Lowercase
      3. Remove HTML tags
      4. Remove URLs
      5. Remove special characters (keep a-z, 0-9, space)
      6. Collapse extra whitespace
      7. Remove English stop-words
      8. Lemmatize  (optional, default True)

    Parameters
    ----------
    text : str or None
        Raw review string.
    lemmatize : bool
        Whether to apply WordNet lemmatization.

    Returns
    -------
    str
        Cleaned text ready for feature extraction.

    Examples
    --------
    >>> preprocess_text("This product is AMAZING!!! <br> Best purchase ever.")
    'product amazing best purchase'
    """
    if not text or not isinstance(text, str):
        return ""

    text = text.lower()
    text = remove_html(text)
    text = remove_urls(text)
    text = remove_special_chars(text)
    text = remove_extra_whitespace(text)
    text = remove_stopwords(text)

    if lemmatize:
        text = lemmatize_text(text)

    return text


def preprocess_batch(
    texts: List[Optional[str]],
    lemmatize: bool = True,
) -> List[str]:
    """
    Apply :func:`preprocess_text` to every element of *texts*.

    Parameters
    ----------
    texts : list[str | None]
    lemmatize : bool

    Returns
    -------
    list[str]
    """
    return [preprocess_text(t, lemmatize) for t in texts]
