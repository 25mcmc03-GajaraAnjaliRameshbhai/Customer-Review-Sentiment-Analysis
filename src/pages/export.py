"""
src/pages/export.py
----------------------
📥 Export Reports — lets the user download the currently active,
annotated dataset (with predicted_label, confidence, and keyword
columns added) as a CSV.
"""

from datetime import datetime

import pandas as pd
import streamlit as st


def render(active_df: pd.DataFrame, dataset_name: str) -> None:
    st.header("📥 Export Reports")
    st.caption(f"Download the currently active, annotated dataset: {dataset_name}")

    if active_df.empty:
        st.info("No data loaded. Visit **Review Explorer** to upload a CSV or load the sample dataset.")
        return

    st.markdown(f"**{len(active_df):,} annotated reviews** ready for download.")

    with st.expander("👀 Preview (first 20 rows)"):
        preview_cols = ["review_text", "predicted_label", "confidence",
                        "top_positive_terms", "top_negative_terms", "top_neutral_terms"]
        preview_cols = [c for c in preview_cols if c in active_df.columns]
        st.dataframe(active_df[preview_cols].head(20), width='stretch', hide_index=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    st.download_button(
        label="⬇️ Download Annotated CSV",
        data=active_df.to_csv(index=False),
        file_name=f"reviewlens_annotated_{timestamp}.csv",
        mime="text/csv",
        type="primary",
    )

    # ── Quick text summary report ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Text Summary Report")

    vc = active_df["predicted_label"].value_counts()
    total = len(active_df)
    report_lines = [
        "ReviewLens — Sentiment Analysis Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Dataset: {dataset_name}",
        f"Total reviews: {total:,}",
        "",
        "Sentiment breakdown:",
        f"  Positive: {vc.get('positive', 0):,} ({vc.get('positive', 0) / total:.1%})",
        f"  Negative: {vc.get('negative', 0):,} ({vc.get('negative', 0) / total:.1%})",
        f"  Neutral:  {vc.get('neutral', 0):,} ({vc.get('neutral', 0) / total:.1%})",
    ]
    report_text = "\n".join(report_lines)

    st.code(report_text, language=None)
    st.download_button(
        label="⬇️ Download Summary Report (.txt)",
        data=report_text,
        file_name=f"reviewlens_summary_{timestamp}.txt",
        mime="text/plain",
    )
