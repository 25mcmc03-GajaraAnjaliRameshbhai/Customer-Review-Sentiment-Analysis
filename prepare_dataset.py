"""
prepare_dataset.py
-------------------
Builds ReviewLens's training dataset from real Kaggle CSV files
(NOT synthetic data).

Usage
-----
    1. Place the raw Kaggle CSVs inside:  data/kaggle_raw/
    2. Run:  python prepare_dataset.py
    3. Output is written to:  data/raw/reviews.csv
       (train.py automatically picks this file up)

This script only needs to be run once, or again if you add / change
a source in src/data_sources.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import build_unified_dataset
from src.utils import DATA_DIR


def main() -> None:
    print("=" * 60)
    print("  ReviewLens — Building Dataset from Real Kaggle Sources")
    print("=" * 60 + "\n")

    df = build_unified_dataset()

    out_path = DATA_DIR / "raw" / "reviews.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\n✅  Unified dataset saved → {out_path}")
    print(f"    Total reviews : {len(df):,}")
    print(f"\nLabel distribution:\n{df['label'].value_counts().to_string()}")
    print(f"\nCategory distribution:\n{df['category'].value_counts().to_string()}")
    print(f"\nSource distribution:\n{df['source'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
