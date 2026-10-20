"""
src/pages/performance.py
--------------------------
🧠 Model Performance — reports how the model performed on the held-out
test set during training (from models/metadata.json), plus the
confusion matrix, per-class breakdown, tuned hyperparameters, and the
most influential words per class. This is the model-trust / validation
page of the dashboard.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def render(metadata: dict) -> None:
    st.header("🧠 Model Performance")

    if not metadata:
        st.error("⚠️ No training metadata found. Run `python train.py` first.")
        return

    # ── Model info ────────────────────────────────────────────────────────────
    st.subheader("Model Info")
    c1, c2, c3 = st.columns(3)
    c1.metric("Algorithm", metadata.get("model_name", "—"))
    c2.metric("Training Samples", f"{metadata.get('number_of_training_samples', 0):,}")
    c3.metric("Test Samples", f"{metadata.get('number_of_test_samples', 0):,}")
    st.caption(f"Trained on: {metadata.get('trained_at', '—')[:19].replace('T', ' ')}")

    # ── Tuned hyperparameters ───────────────────────────────────────────────
    hyperparams = metadata.get("hyperparameters", {})
    if hyperparams:
        with st.expander("⚙️ Tuned hyperparameters (found via cross-validated grid search)"):
            st.markdown(
                f"- **Regularisation (C):** {hyperparams.get('C', '—')}\n"
                f"- **Class weights:** {hyperparams.get('class_weight', '—')}"
            )
            st.caption(
                "scikit-learn's built-in class_weight=\"balanced\" over-corrected for "
                "the rare 'neutral' class on this dataset (high recall, very low "
                "precision). These weights were instead found via a small "
                "cross-validated search on the training set only, optimising macro F1 "
                "— see docs/PROJECT_REPORT.md for the full comparison."
            )

    st.markdown("---")

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.subheader("Evaluation Metrics")
    metrics = metadata.get("metrics", {})
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
    m2.metric("Macro F1", f"{metrics.get('macro_f1', 0):.3f}")
    m3.metric("Weighted F1", f"{metrics.get('weighted_f1', 0):.3f}")

    st.caption(
        "Macro F1 weights all three classes equally, so it is the fairer "
        "metric when classes are imbalanced (as they are with real review data, "
        "where 'neutral' reviews are naturally rarer than clearly positive or negative ones)."
    )

    st.markdown("---")

    # ── Training class distribution ──────────────────────────────────────────
    st.subheader("Training Class Distribution")
    class_dist = metadata.get("class_distribution", {})
    if class_dist:
        dist_df = pd.DataFrame(
            [{"label": k, "count": v} for k, v in class_dist.items()]
        )
        fig = px.bar(dist_df, x="label", y="count", color="label", text="count",
                     color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#f39c12"})
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    # ── Confusion matrix + per-class breakdown ─────────────────────────────────
    st.subheader("Confusion Matrix & Per-Class Breakdown (Test Set)")
    cm     = metadata.get("confusion_matrix")
    labels = metadata.get("confusion_matrix_labels")

    if cm and labels:
        cm_arr = np.array(cm)
        cm_df = pd.DataFrame(cm_arr, index=labels, columns=labels)

        col_cm, col_table = st.columns([3, 2])
        with col_cm:
            fig = px.imshow(
                cm_df, text_auto=True, color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual", color="Count"),
            )
            st.plotly_chart(fig, width='stretch')
            st.caption(
                "Rows = actual label, columns = predicted label. "
                "The diagonal shows correct predictions."
            )
        with col_table:
            per_class = _per_class_metrics_from_confusion_matrix(cm_arr, labels)
            st.dataframe(
                pd.DataFrame(per_class).style.format({
                    "Precision": "{:.2f}", "Recall": "{:.2f}", "F1": "{:.2f}",
                }),
                width='stretch', hide_index=True,
            )
            st.caption(
                "Computed directly from the confusion matrix: Precision = "
                "correct / predicted-as-this-class; Recall = correct / "
                "actually-this-class."
            )
    else:
        st.info("Confusion matrix not available — re-run `python train.py` to generate it.")

    st.markdown("---")

    # ── Top features ──────────────────────────────────────────────────────────
    st.subheader("Most Influential Words")
    f1, f2 = st.columns(2)
    with f1:
        st.markdown("##### ✅ Top Positive Signals")
        pos_features = metadata.get("top_positive_features", [])
        st.write(", ".join(pos_features) if pos_features else "—")
    with f2:
        st.markdown("##### ❌ Top Negative Signals")
        neg_features = metadata.get("top_negative_features", [])
        st.write(", ".join(neg_features) if neg_features else "—")


def _per_class_metrics_from_confusion_matrix(cm: np.ndarray, labels: list) -> list:
    """
    Compute precision, recall, and F1 per class directly from the
    confusion matrix, so no extra fields need to be stored in
    metadata.json beyond what's already saved for the heatmap.
    """
    rows = []
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        support = cm[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        rows.append({
            "Class": label,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Support": int(support),
        })
    return rows
