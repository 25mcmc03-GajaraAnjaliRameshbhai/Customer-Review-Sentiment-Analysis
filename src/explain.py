"""
src/explain.py
--------------
Lightweight, coefficient-based explainability for ReviewLens.

Uses TF-IDF term weights × Logistic Regression class coefficients
to surface the words that most pushed the model toward 'positive',
'negative', or 'neutral'. No external explainability library required.

Two separate signals are computed per word:
  - pos_vs_neg score = coef[positive] − coef[negative]
    Where a word sits on the positive/negative spectrum. This is
    reliable whenever the review leans clearly one way or the other.
  - neutral_pull score = coef[neutral] − avg(coef[positive], coef[negative])
    How much a word specifically indicates hedging/ambivalence (e.g.
    "okay", "average", "nothing special") rather than just being a
    weak positive or weak negative word. Without this second signal,
    a "neutral" prediction could only be explained in terms of leftover
    positive/negative tug-of-war, never in terms of what actually reads
    as neutral — see docs/PROJECT_REPORT.md for why this distinction
    matters given the neutral class's comparatively low precision.

Public API
----------
explain_prediction(text, pipeline, n_terms=10) -> dict
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.pipeline import Pipeline


def explain_prediction(
    text: str,
    pipeline: Pipeline,
    n_terms: int = 10,
) -> Dict[str, Any]:
    """
    Explain a single prediction from a TF-IDF + Logistic Regression pipeline.

    Parameters
    ----------
    text : str
        Pre-processed (cleaned) review text.
    pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline with 'tfidf' and 'clf' steps.
    n_terms : int
        Maximum number of keywords to return per category.

    Returns
    -------
    dict
        predicted_label : str
        confidence      : float (max class probability)
        probabilities   : dict[str, float]
        positive_terms  : list[str]  words pushing toward positive (vs. negative)
        negative_terms  : list[str]  words pushing toward negative (vs. positive)
        neutral_terms   : list[str]  words indicating hedging/ambivalence —
                                      the words that specifically explain a
                                      "neutral" call, not just weak pos/neg
        explanation     : str        human-readable summary
    """
    vectorizer = pipeline.named_steps.get("tfidf")
    clf        = pipeline.named_steps.get("clf")

    # ── Predict ───────────────────────────────────────────────────────────────
    predicted_label = pipeline.predict([text])[0]
    proba_array     = pipeline.predict_proba([text])[0]
    classes         = list(clf.classes_)
    confidence      = float(round(float(np.max(proba_array)), 4))
    proba_dict      = {
        str(c): round(float(p), 4)
        for c, p in zip(classes, proba_array)
    }

    positive_terms, negative_terms, neutral_terms = _extract_contributing_terms(
        text, vectorizer, clf, classes, n_terms
    )

    explanation = _build_explanation(predicted_label, positive_terms, negative_terms, neutral_terms)

    return {
        "predicted_label": str(predicted_label),
        "confidence":      confidence,
        "probabilities":   proba_dict,
        "positive_terms":  positive_terms,
        "negative_terms":  negative_terms,
        "neutral_terms":   neutral_terms,
        "explanation":     explanation,
    }


# ── Private helpers ───────────────────────────────────────────────────────────

def _extract_contributing_terms(
    text: str,
    vectorizer,
    clf,
    classes: List[str],
    n_terms: int,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Score every word present in *text* on two axes (positive-vs-negative,
    and neutral-pull) and return the top words on each.
    """
    if not hasattr(clf, "coef_") or vectorizer is None:
        return [], [], []

    tfidf_matrix = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    nonzero_idx = tfidf_matrix.nonzero()[1]

    return score_terms_for_indices(nonzero_idx, feature_names, clf, classes, n_terms)


def score_terms_for_indices(
    nonzero_idx,
    feature_names,
    clf,
    classes: List[str],
    n_terms: int,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Core scoring logic, factored out so both the single-review path
    above and the vectorised bulk-CSV path in
    src/dashboard_utils.py::_predict_chunk can share one implementation
    instead of two copies that could silently drift apart.

    Parameters
    ----------
    nonzero_idx : array-like of int
        Indices into the TF-IDF vocabulary for the words present in
        one review (already computed by the caller, since for bulk
        prediction the caller vectorises the whole batch at once for
        speed rather than one row at a time).
    """
    if len(nonzero_idx) == 0:
        return [], [], []

    pos_idx = classes.index("positive") if "positive" in classes else None
    neg_idx = classes.index("negative") if "negative" in classes else None
    neu_idx = classes.index("neutral") if "neutral" in classes else None

    pos_neg_scores: List[Tuple[str, float]] = []
    neutral_scores: List[Tuple[str, float]] = []

    for idx in nonzero_idx:
        word = feature_names[idx]
        pos_coef = clf.coef_[pos_idx][idx] if pos_idx is not None else 0.0
        neg_coef = clf.coef_[neg_idx][idx] if neg_idx is not None else 0.0
        neu_coef = clf.coef_[neu_idx][idx] if neu_idx is not None else 0.0

        # Where this word sits on the positive/negative spectrum
        pos_neg_scores.append((word, float(pos_coef - neg_coef)))

        # How much this word specifically signals hedging/ambivalence,
        # rather than just being a weak positive or weak negative word
        if neu_idx is not None:
            neutral_scores.append((word, float(neu_coef - (pos_coef + neg_coef) / 2)))

    pos_neg_scores.sort(key=lambda x: x[1], reverse=True)
    positive_terms = [w for w, s in pos_neg_scores if s > 0][:n_terms]
    negative_terms = [w for w, s in pos_neg_scores if s < 0]
    negative_terms = negative_terms[-n_terms:][::-1]  # strongest negatives first

    neutral_scores.sort(key=lambda x: x[1], reverse=True)
    neutral_terms = [w for w, s in neutral_scores if s > 0][:n_terms]

    return positive_terms, negative_terms, neutral_terms


def _build_explanation(
    label: str,
    positive_terms: List[str],
    negative_terms: List[str],
    neutral_terms: Optional[List[str]] = None,
) -> str:
    """Compose a one-sentence human-readable explanation."""
    neutral_terms = neutral_terms or []
    parts: List[str] = []

    if str(label) == "neutral":
        # For a neutral call, lead with the words that actually explain
        # "neutral" (hedging/ambivalent language), not the leftover
        # positive/negative tug-of-war — those are shown as secondary
        # context, since neutral reviews often still contain some of
        # each without being decisively either.
        if neutral_terms:
            parts.append(f"Hedging/ambivalent signals: {', '.join(neutral_terms[:5])}")
        if positive_terms:
            parts.append(f"Positive pull: {', '.join(positive_terms[:3])}")
        if negative_terms:
            parts.append(f"Negative pull: {', '.join(negative_terms[:3])}")

        summary = "Review carries mixed or neutral sentiment."
        if not neutral_terms and (positive_terms or negative_terms):
            summary += (" No strongly hedging language was detected — the neutral call "
                       "comes from roughly balanced positive and negative signals instead.")
    else:
        if positive_terms:
            parts.append(f"Positive signals: {', '.join(positive_terms[:5])}")
        if negative_terms:
            parts.append(f"Negative signals: {', '.join(negative_terms[:5])}")

        summary_map = {
            "positive": "Review is predominantly positive.",
            "negative": "Review is predominantly negative.",
        }
        summary = summary_map.get(str(label), "Sentiment unclear.")

    detail = " | ".join(parts)
    return f"{summary} {detail}".strip()
