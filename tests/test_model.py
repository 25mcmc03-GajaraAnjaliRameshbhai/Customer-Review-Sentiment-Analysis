"""
tests/test_model.py
-------------------
Tests for trained model artifacts and prediction behaviour.
All tests are skipped automatically if the model has not been trained yet.

Run after training:
    python train.py
    pytest tests/test_model.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.utils import PIPELINE_PATH

TRAINED = PIPELINE_PATH.exists()
SKIP_MSG = "Model not trained yet — run: python train.py"

VALID_LABELS = {"positive", "negative", "neutral"}


# ── File existence ────────────────────────────────────────────────────────────

@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_pipeline_file_exists():
    assert PIPELINE_PATH.exists(), "pipeline.pkl must exist after training"


@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_metadata_file_exists():
    from src.utils import METADATA_PATH
    assert METADATA_PATH.exists(), "metadata.json must exist after training"


# ── Prediction correctness ────────────────────────────────────────────────────

@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_prediction_returns_valid_label():
    import joblib
    from src.preprocess import preprocess_text

    pipeline = joblib.load(PIPELINE_PATH)
    clean    = preprocess_text("This product is really amazing and excellent value.")
    pred     = pipeline.predict([clean])[0]
    assert pred in VALID_LABELS, f"Unexpected label: {pred}"


@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_negative_review_predicted_negative_or_neutral():
    import joblib
    from src.preprocess import preprocess_text

    pipeline = joblib.load(PIPELINE_PATH)
    clean    = preprocess_text("Absolutely terrible. Stopped working after two days. Waste of money.")
    pred     = pipeline.predict([clean])[0]
    # Should be negative (or at least not positive)
    assert pred in {"negative", "neutral"}, f"Expected negative/neutral, got: {pred}"


@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_positive_review_predicted_positive_or_neutral():
    import joblib
    from src.preprocess import preprocess_text

    pipeline = joblib.load(PIPELINE_PATH)
    clean    = preprocess_text("Amazing product! Best purchase ever. Highly recommend.")
    pred     = pipeline.predict([clean])[0]
    assert pred in {"positive", "neutral"}, f"Expected positive/neutral, got: {pred}"


# ── Probability output ────────────────────────────────────────────────────────

@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_probabilities_sum_to_one():
    import joblib
    import numpy as np
    from src.preprocess import preprocess_text

    pipeline = joblib.load(PIPELINE_PATH)
    clean    = preprocess_text("Very bad product. Terrible quality. Disappointed.")
    probas   = pipeline.predict_proba([clean])[0]
    assert abs(sum(probas) - 1.0) < 1e-5, f"Probabilities sum to {sum(probas)}, expected 1.0"


@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_probabilities_all_non_negative():
    import joblib
    from src.preprocess import preprocess_text

    pipeline = joblib.load(PIPELINE_PATH)
    clean    = preprocess_text("Good enough product. Nothing special.")
    probas   = pipeline.predict_proba([clean])[0]
    assert all(p >= 0 for p in probas), "All probabilities must be non-negative"


@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_three_classes_returned():
    import joblib

    pipeline = joblib.load(PIPELINE_PATH)
    classes  = list(pipeline.named_steps["clf"].classes_)
    assert len(classes) == 3
    assert set(classes) == VALID_LABELS


# ── Batch prediction ──────────────────────────────────────────────────────────

@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_batch_prediction_correct_length():
    import joblib
    from src.preprocess import preprocess_batch

    texts    = ["Amazing product", "Terrible quality", "Average item"]
    pipeline = joblib.load(PIPELINE_PATH)
    cleaned  = preprocess_batch(texts)
    preds    = pipeline.predict(cleaned)
    assert len(preds) == 3


@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_batch_all_labels_valid():
    import joblib
    from src.preprocess import preprocess_batch

    texts    = ["I love this", "I hate this", "It is okay", "Not bad", "Worst ever"]
    pipeline = joblib.load(PIPELINE_PATH)
    cleaned  = preprocess_batch(texts)
    preds    = pipeline.predict(cleaned)
    assert all(p in VALID_LABELS for p in preds)


# ── Explainability ────────────────────────────────────────────────────────────

@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_explain_returns_required_keys():
    import joblib
    from src.preprocess import preprocess_text
    from src.explain import explain_prediction

    pipeline = joblib.load(PIPELINE_PATH)
    clean    = preprocess_text("Excellent product. Very happy with my purchase.")
    result   = explain_prediction(clean, pipeline)

    for key in ("predicted_label", "confidence", "probabilities",
                "positive_terms", "negative_terms", "neutral_terms", "explanation"):
        assert key in result, f"Missing key: {key}"


@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_explain_confidence_in_range():
    import joblib
    from src.preprocess import preprocess_text
    from src.explain import explain_prediction

    pipeline   = joblib.load(PIPELINE_PATH)
    clean      = preprocess_text("Great camera and smooth performance.")
    result     = explain_prediction(clean, pipeline)
    confidence = result["confidence"]
    assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} not in [0, 1]"

@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_explain_neutral_review_gets_hedging_keywords():
    """
    For a review with clear hedging language ("okay", "nothing
    special", "could be better"), the neutral_terms field should
    surface that hedging language — not just leave neutral predictions
    explained only in terms of leftover positive/negative pull.
    """
    import joblib
    from src.preprocess import preprocess_text
    from src.explain import explain_prediction

    pipeline = joblib.load(PIPELINE_PATH)
    clean = preprocess_text("This product is okay, nothing special, could be better for the price.")
    result = explain_prediction(clean, pipeline)

    assert result["predicted_label"] == "neutral"
    assert len(result["neutral_terms"]) > 0
    # At least one recognisable hedging phrase should be present
    hedging_vocab = {"okay", "nothing special", "could better", "average", "average product"}
    assert hedging_vocab & set(result["neutral_terms"])


@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_explain_neutral_prediction_explanation_mentions_hedging_signals():
    """The human-readable explanation string for a neutral prediction
    should lead with hedging/ambivalent language, not just positive
    and negative signals as if it were a weak positive or weak negative."""
    import joblib
    from src.preprocess import preprocess_text
    from src.explain import explain_prediction

    pipeline = joblib.load(PIPELINE_PATH)
    clean = preprocess_text("It works fine, meets expectations, average product for the price.")
    result = explain_prediction(clean, pipeline)

    assert result["predicted_label"] == "neutral"
    assert "Hedging" in result["explanation"] or "hedging" in result["explanation"]


@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_explain_positive_prediction_explanation_unaffected_by_neutral_logic():
    """Adding neutral-term extraction should not change the explanation
    format for clearly positive or negative predictions — they should
    still lead with positive/negative signals only, not hedging language."""
    import joblib
    from src.preprocess import preprocess_text
    from src.explain import explain_prediction

    pipeline = joblib.load(PIPELINE_PATH)
    clean = preprocess_text("Absolutely amazing, best purchase ever, love it so much!")
    result = explain_prediction(clean, pipeline)

    assert result["predicted_label"] == "positive"
    assert "Positive signals" in result["explanation"]
    assert "Hedging" not in result["explanation"]


@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_explain_neutral_terms_is_always_a_list():
    """neutral_terms should always be a list (possibly empty), even for
    clearly positive/negative predictions, so callers never need a
    None-check."""
    import joblib
    from src.preprocess import preprocess_text
    from src.explain import explain_prediction

    pipeline = joblib.load(PIPELINE_PATH)
    for text in ["Amazing product!", "Terrible, awful, worst ever.", "It's fine I guess."]:
        clean = preprocess_text(text)
        result = explain_prediction(clean, pipeline)
        assert isinstance(result["neutral_terms"], list)
