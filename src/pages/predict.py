"""
src/pages/predict.py
----------------------
🤖 Predict Sentiment — the core single-review analysis feature.
User pastes one review, clicks Analyze, and gets:
  - predicted sentiment + confidence
  - probability breakdown across all 3 classes
  - positive / negative / hedging keyword tags (model explainability)

Note on neutral predictions: the underlying model's neutral-class
precision is comparatively low (~28% on the real training data — see
docs/PROJECT_REPORT.md), since neutral sentiment is inherently more
ambiguous and much rarer than clearly positive or negative reviews.
When the prediction itself is neutral, this page shows an explicit
caveat rather than presenting it with the same confidence as a
positive/negative call.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard_utils import predict_single
from src.utils import LABEL_COLORS, get_emoji_for_label

# Keep in sync with the neutral-class precision reported in
# docs/PROJECT_REPORT.md and shown on the Model Performance page.
NEUTRAL_CLASS_PRECISION = 0.28


def render(pipeline) -> None:
    st.header("🤖 Predict Sentiment")
    st.markdown(
        "Paste a customer review below and click **Analyze Review** "
        "to get an instant sentiment prediction with keyword highlights."
    )

    review_input = st.text_area(
        label="Enter review text:",
        placeholder="e.g.  The phone camera is excellent, but the battery backup is very poor.",
        height=140,
    )

    run_analysis = st.button("🔍 Analyze Review", type="primary")

    if not run_analysis:
        return

    if not review_input.strip():
        st.warning("⚠️ Please enter a review before clicking Analyze.")
        return

    with st.spinner("Analyzing …"):
        result = predict_single(review_input, pipeline)

    if result is None:
        st.error("The input text could not be processed. Please try a more detailed review.")
        return

    label      = result["predicted_label"]
    confidence = result["confidence"]
    proba      = result["probabilities"]
    emoji      = get_emoji_for_label(label)

    st.markdown("---")

    # -- Reliability caveat for neutral predictions ─────────────────────────
    # Shown before the results so it can't be missed, not buried after them.
    if label == "neutral":
        st.warning(
            f"⚠️ **This is a neutral prediction — treat it with extra caution.** "
            f"Neutral sentiment is inherently more ambiguous than clearly positive "
            f"or negative text, and is the model's weakest class: on real review "
            f"data, only about **{NEUTRAL_CLASS_PRECISION:.0%}** of reviews predicted "
            f"as neutral actually are. Consider this a starting point, not a "
            f"confident verdict — see the Model Performance page for the full "
            f"per-class accuracy breakdown."
        )

    # -- Metric cards ────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Sentiment", f"{emoji} {label.capitalize()}")
    c2.metric("Confidence", f"{confidence:.1%}")
    c3.metric("Review Length", f"{len(review_input.split())} words")

    # -- Probability bar chart ──────────────────────────────────────────────
    st.markdown("#### 📈 Probability Breakdown")
    prob_df = pd.DataFrame([
        {"Sentiment": k.capitalize(), "Probability": v} for k, v in proba.items()
    ])
    fig = px.bar(
        prob_df, x="Sentiment", y="Probability", color="Sentiment",
        color_discrete_map={
            "Positive": LABEL_COLORS["positive"],
            "Negative": LABEL_COLORS["negative"],
            "Neutral":  LABEL_COLORS["neutral"],
        },
        text="Probability", height=280,
    )
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig.update_layout(showlegend=False, yaxis_range=[0, 1.15])
    st.plotly_chart(fig, width='stretch')

    # -- Keyword tags (explainability) ──────────────────────────────────────
    # Three columns when the prediction is neutral (hedging language is the
    # primary explanation there), two columns otherwise — no point showing
    # an empty "hedging signals" panel for a clearly positive/negative call.
    show_neutral_panel = label == "neutral" and bool(result["neutral_terms"])
    cols = st.columns(3) if show_neutral_panel else st.columns(2)

    with cols[0]:
        st.markdown("#### ✅ Positive Keywords")
        if result["positive_terms"]:
            html = " ".join(f'<span class="tag tag-pos">{w}</span>' for w in result["positive_terms"])
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("No strong positive terms detected.")

    with cols[1]:
        st.markdown("#### ❌ Negative Keywords")
        if result["negative_terms"]:
            html = " ".join(f'<span class="tag tag-neg">{w}</span>' for w in result["negative_terms"])
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("No strong negative terms detected.")

    if show_neutral_panel:
        with cols[2]:
            st.markdown("#### 🟡 Hedging / Neutral Keywords")
            html = " ".join(f'<span class="tag tag-neu">{w}</span>' for w in result["neutral_terms"])
            st.markdown(html, unsafe_allow_html=True)
            st.caption("Words that specifically indicate ambivalence, not just weak positive/negative.")

    # -- Explanation ─────────────────────────────────────────────────────────
    st.markdown("#### 💡 Explanation")
    st.info(result["explanation"])
