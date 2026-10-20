"""
tests/test_dashboard.py
-------------------------
Integration tests for the Streamlit dashboard using Streamlit's
official `AppTest` framework, which actually executes app.py and lets
us inspect the rendered output and catch runtime exceptions — the
same kind of bugs (missing columns, bad chart calls, etc.) that would
only otherwise surface by clicking through the app manually.

All tests are skipped automatically if the model has not been trained
yet, since app.py requires models/pipeline.pkl to run.

Run:
    python train.py
    pytest tests/test_dashboard.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from streamlit.testing.v1 import AppTest

from src.utils import PIPELINE_PATH

TRAINED = PIPELINE_PATH.exists()
SKIP_MSG = "Model not trained yet — run: python train.py"

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

PAGE_LABELS = [
    "📊 Overview",
    "🤖 Predict Sentiment",
    "🔍 Review Explorer",
    "📈 Trends & Charts",
    "⚖️ Product Comparison",
    "📄 Summary",
    "🧠 Model Performance",
    "📥 Export Reports",
]


@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_app_loads_without_exception():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=45)
    assert not at.exception


@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
@pytest.mark.parametrize("page_label", PAGE_LABELS)
def test_each_sidebar_page_renders_without_exception(page_label):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=45)
    at.radio[0].set_value(page_label).run(timeout=45)
    assert not at.exception, f"Page '{page_label}' raised an exception"


@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_predict_sentiment_flow_returns_metrics():
    """End-to-end test of the single-review prediction interaction."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=45)
    at.radio[0].set_value("🤖 Predict Sentiment").run(timeout=45)

    at.text_area[0].set_value("Excellent product, very happy with the purchase!")
    at.button[0].click().run(timeout=45)

    assert not at.exception
    metric_labels = [m.label for m in at.metric]
    assert "Predicted Sentiment" in metric_labels
    assert "Confidence" in metric_labels


@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_predict_sentiment_empty_input_shows_warning():
    """Empty input should show a warning, not crash."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=45)
    at.radio[0].set_value("🤖 Predict Sentiment").run(timeout=45)

    at.button[0].click().run(timeout=45)

    assert not at.exception
    assert len(at.warning) > 0


@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_csv_upload_with_unusual_column_names_end_to_end(tmp_path):
    """
    Regression test for flexible CSV upload: a file with completely
    different column names ("Customer Comments", "Stars Given",
    "Purchased On", "Bought From") should still be auto-mapped and
    analysed correctly — proving the dashboard is not tied to a fixed
    review_text/rating/date/source/category schema.
    """
    import pandas as pd

    csv_path = tmp_path / "weird_schema.csv"
    pd.DataFrame({
        "Customer Comments": [
            "Absolutely love this, works perfectly every time!",
            "Terrible experience, broke within a week of use.",
        ],
        "Stars Given": [5, 1],
        "Purchased On": ["2024-03-01", "2024-03-05"],
        "Bought From": ["Amazon", "Flipkart"],
    }).to_csv(csv_path, index=False)

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=45)
    at.radio[0].set_value("🔍 Review Explorer").run(timeout=45)

    content = csv_path.read_bytes()
    at.file_uploader[0].upload("weird_schema.csv", content, "text/csv").run(timeout=45)
    assert not at.exception

    # Auto-detected mapping should be pre-filled correctly
    mapping_by_label = {sb.label: sb.value for sb in at.selectbox}
    assert mapping_by_label["Review text column *"] == "Customer Comments"
    assert mapping_by_label["Rating column (optional)"] == "Stars Given"

    analyze_btn = [b for b in at.button if "Analyze" in b.label][0]
    analyze_btn.click().run(timeout=45)
    assert not at.exception

    active_df = at.session_state["active_df"]
    assert len(active_df) == 2
    assert set(active_df["predicted_label"]).issubset({"positive", "negative", "neutral"})
    assert "rating" in active_df.columns  # optional field was correctly mapped through


@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_csv_upload_rejects_oversized_file():
    """A file larger than MAX_UPLOAD_SIZE_MB should be rejected with a
    clear, actionable error — not silently truncated or crashed on."""
    from src.utils import MAX_UPLOAD_SIZE_MB

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=45)
    at.radio[0].set_value("🔍 Review Explorer").run(timeout=45)

    # Build content comfortably over the limit
    row = b'"This is a filler review to pad out the file size significantly."\n'
    target_bytes = int((MAX_UPLOAD_SIZE_MB + 5) * 1024 * 1024)
    big_content = b"review_text\n" + row * (target_bytes // len(row))

    at.file_uploader[0].upload("huge.csv", big_content, "text/csv").run(timeout=60)

    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("exceeds" in e and "MB" in e for e in errors)


@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_csv_upload_rejects_too_many_rows(tmp_path):
    """A file with more rows than MAX_UPLOAD_ROWS should be rejected
    with a clear message, not silently truncated or hung."""
    import pandas as pd

    from src.utils import MAX_UPLOAD_ROWS

    csv_path = tmp_path / "too_many_rows.csv"
    pd.DataFrame({"review_text": ["ok"] * (MAX_UPLOAD_ROWS + 1)}).to_csv(csv_path, index=False)

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=45)
    at.radio[0].set_value("🔍 Review Explorer").run(timeout=45)

    at.file_uploader[0].upload("too_many_rows.csv", csv_path.read_bytes(), "text/csv").run(timeout=45)

    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("exceeds" in e and "row" in e for e in errors)


@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_csv_upload_shows_recommended_row_warning(tmp_path):
    """A file above RECOMMENDED_MAX_ROWS but below the hard cap should
    proceed, with a warning (not an error) about slower processing."""
    import pandas as pd

    from src.utils import MAX_UPLOAD_ROWS, RECOMMENDED_MAX_ROWS

    n_rows = RECOMMENDED_MAX_ROWS + 500
    assert n_rows < MAX_UPLOAD_ROWS

    csv_path = tmp_path / "medium_large.csv"
    pd.DataFrame({"review_text": ["Decent product, works as expected."] * n_rows}).to_csv(csv_path, index=False)

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=45)
    at.radio[0].set_value("🔍 Review Explorer").run(timeout=45)

    at.file_uploader[0].upload("medium_large.csv", csv_path.read_bytes(), "text/csv").run(timeout=45)

    assert not at.exception
    assert len(at.error) == 0  # should NOT be rejected
    warnings = [w.value for w in at.warning]
    assert any("recommended" in w for w in warnings)


@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_csv_upload_empty_file_shows_clear_error(tmp_path):
    """A CSV with headers but zero data rows should show a specific
    error, not an empty page or a crash."""
    import pandas as pd

    csv_path = tmp_path / "empty.csv"
    pd.DataFrame({"review_text": []}).to_csv(csv_path, index=False)

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=45)
    at.radio[0].set_value("🔍 Review Explorer").run(timeout=45)

    at.file_uploader[0].upload("empty.csv", csv_path.read_bytes(), "text/csv").run(timeout=45)

    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("no data rows" in e for e in errors)


@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_csv_upload_corrupted_file_shows_clear_error():
    """A malformed / non-CSV file should show a specific parse error,
    not crash the app with a raw traceback."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=45)
    at.radio[0].set_value("🔍 Review Explorer").run(timeout=45)

    corrupted = b'this is not\na valid, csv,,,file\n"unterminated quote,,,'
    at.file_uploader[0].upload("corrupted.csv", corrupted, "text/csv").run(timeout=45)

    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("valid CSV" in e or "corrupted" in e for e in errors)


@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_csv_upload_validation_flags_bad_rating_column(tmp_path):
    """When the selected rating column is mostly non-numeric, the
    validation panel should surface a warning rather than silently
    producing garbage output."""
    import pandas as pd

    csv_path = tmp_path / "bad_ratings.csv"
    pd.DataFrame({
        "review_text": ["Great product overall", "Terrible experience here", "It was okay I suppose"],
        "rating": ["five stars please", "not applicable", "3"],
    }).to_csv(csv_path, index=False)

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=45)
    at.radio[0].set_value("🔍 Review Explorer").run(timeout=45)

    at.file_uploader[0].upload("bad_ratings.csv", csv_path.read_bytes(), "text/csv").run(timeout=45)

    assert not at.exception
    all_markdown = " ".join(md.value for md in at.markdown)
    assert "not numeric" in all_markdown


@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_csv_upload_large_file_uses_chunked_progress(tmp_path):
    """
    A file larger than PREDICT_CHUNK_SIZE should complete successfully
    via the chunked prediction path (proving predict_batch's chunking
    integrates correctly with the real app, not just in isolation).
    """
    import pandas as pd

    from src.utils import PREDICT_CHUNK_SIZE

    n_rows = PREDICT_CHUNK_SIZE + 500
    csv_path = tmp_path / "chunked_test.csv"
    pd.DataFrame({
        "review_text": (["Excellent quality, highly recommend!", "Poor quality, very disappointed."] * n_rows)[:n_rows],
    }).to_csv(csv_path, index=False)

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=45)
    at.radio[0].set_value("🔍 Review Explorer").run(timeout=45)

    at.file_uploader[0].upload("chunked_test.csv", csv_path.read_bytes(), "text/csv").run(timeout=45)
    assert not at.exception

    analyze_btn = [b for b in at.button if "Analyze" in b.label][0]
    analyze_btn.click().run(timeout=90)

    assert not at.exception
    active_df = at.session_state["active_df"]
    assert len(active_df) == n_rows


@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_trends_page_shows_confidence_and_mismatch_sections():
    """
    The Trends & Charts page should show the confidence-distribution
    and rating-vs-sentiment-mismatch insights when the active dataset
    has a 'rating' column (true for the default sample dataset, which
    is drawn from the real training corpus).
    """
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=45)
    at.radio[0].set_value("📈 Trends & Charts").run(timeout=45)

    assert not at.exception
    subheaders = [s.value for s in at.subheader]
    assert "🎯 Prediction Confidence" in subheaders
    assert "🚩 Rating vs. Sentiment Mismatch" in subheaders

    metric_labels = [m.label for m in at.metric]
    assert "Mismatched Reviews" in metric_labels
    assert "% of Rated Reviews" in metric_labels


@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_overview_page_shows_confidence_and_date_range():
    """Overview should surface average confidence, and date range when
    the active dataset has a parseable 'date' column."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=45)
    at.radio[0].set_value("📊 Overview").run(timeout=45)

    assert not at.exception
    all_markdown = " ".join(md.value for md in at.markdown)
    assert "Average model confidence" in all_markdown
    assert "Date range" in all_markdown


@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_summary_trend_uses_per_source_breakdown_for_mixed_sources():
    """
    The default sample dataset mixes multiple platforms (Amazon, Play
    Store, etc.) with uneven date coverage. The Summary page's trend
    insight must NOT show one misleading combined-source trend line in
    this case — it should show a per-source breakdown instead, since
    a naive earlier/later split can mistake a shift in *which*
    platform dominates each half for an actual sentiment trend.
    """
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=45)
    at.radio[0].set_value("📄 Summary").run(timeout=45)

    assert not at.exception
    all_markdown = " ".join(md.value for md in at.markdown)
    assert "Trend (by source)" in all_markdown
    assert "**Trend:**" not in all_markdown


@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_summary_trend_shows_simple_aggregate_for_single_source(tmp_path):
    """
    For a single-source dataset (the realistic case for most CSV
    uploads — e.g. one company's own Amazon export), the trend insight
    should show a simple, direct aggregate line rather than the
    per-source table (which would be redundant with only one source).
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", "2024-06-30", periods=100)
    negative_texts = ["Terrible product, very disappointed", "Bad quality, does not work well"]
    positive_texts = ["Amazing product, love it!", "Great quality, highly recommend"]
    texts = [
        rng.choice(negative_texts) if i < 50 else rng.choice(positive_texts)
        for i in range(100)
    ]

    csv_path = tmp_path / "single_source.csv"
    pd.DataFrame({
        "review_text": texts,
        "date": dates.strftime("%Y-%m-%d"),
        "source": "Amazon",
    }).to_csv(csv_path, index=False)

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=45)
    at.radio[0].set_value("🔍 Review Explorer").run(timeout=45)
    at.file_uploader[0].upload("single_source.csv", csv_path.read_bytes(), "text/csv").run(timeout=45)

    analyze_btn = [b for b in at.button if "Analyze" in b.label][0]
    analyze_btn.click().run(timeout=45)
    assert not at.exception

    at.radio[0].set_value("📄 Summary").run(timeout=45)
    assert not at.exception

    all_markdown = " ".join(md.value for md in at.markdown)
    assert "**Trend:**" in all_markdown
    assert "Trend (by source)" not in all_markdown


@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_predict_neutral_review_shows_reliability_caveat():
    """
    Neutral predictions should show an explicit warning about the
    neutral class's lower reliability, and a dedicated hedging-keyword
    panel — not be presented with the same unqualified confidence as a
    positive/negative call.
    """
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=45)
    at.radio[0].set_value("🤖 Predict Sentiment").run(timeout=45)
    at.text_area[0].set_value("This product is okay, nothing special, could be better for the price.")
    at.button[0].click().run(timeout=45)

    assert not at.exception
    warnings = [w.value for w in at.warning]
    assert any("extra caution" in w for w in warnings)

    all_markdown = " ".join(m.value for m in at.markdown)
    assert "Hedging / Neutral Keywords" in all_markdown


@pytest.mark.skipif(not TRAINED, reason=SKIP_MSG)
def test_predict_positive_review_shows_no_neutral_caveat():
    """A clear positive/negative prediction should not show the
    neutral-reliability caveat or the hedging-keyword panel — it
    would be irrelevant clutter for an unambiguous call."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=45)
    at.radio[0].set_value("🤖 Predict Sentiment").run(timeout=45)
    at.text_area[0].set_value("Absolutely amazing, best purchase ever, love it so much!")
    at.button[0].click().run(timeout=45)

    assert not at.exception
    warnings = [w.value for w in at.warning]
    assert not any("extra caution" in w for w in warnings)

    all_markdown = " ".join(m.value for m in at.markdown)
    assert "Hedging / Neutral Keywords" not in all_markdown
