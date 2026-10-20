"""
src/data_sources.py
--------------------
Configuration that describes how each raw Kaggle CSV file maps onto
ReviewLens's unified review schema:

    review_text | rating | date | source | category | label

Editing this file is the ONLY thing needed to add, remove, or adjust a
data source. The merging logic itself lives in src/data_loader.py and
is completely generic — it just reads this list.

Each raw file has a different structure (different column names,
different ways of expressing sentiment), so every entry tells the
loader exactly which column to use for text, which column carries the
sentiment signal, and how to interpret that signal.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SourceConfig:
    file_name: str            # CSV file name inside data/kaggle_raw/
    text_col: str              # column holding the raw review text
    label_type: str            # "rating" (1-5 stars) | "sentiment" (text label) | "binary" (0/1)
    label_col: str             # column holding the rating / sentiment / binary value
    category: str              # product category shown in the dashboard
    source_name: str           # platform name shown in the dashboard
    date_col: Optional[str] = None       # column holding a date, if available
    sample_size: Optional[int] = None    # cap rows taken from this file (None = take all)
    encoding: str = "utf-8"


# ── Active sources ────────────────────────────────────────────────────────────
# Sample sizes are capped per-source so no single huge file (e.g. the 205K-row
# Flipkart dataset) dominates the merged corpus, and so the whole pipeline
# trains quickly on a normal laptop CPU.

SOURCES = [
    SourceConfig(
        file_name="Amazon  Reviews.csv",
        text_col="reviewText", label_type="rating", label_col="overall",
        date_col="reviewTime", category="Electronics", source_name="Amazon",
        sample_size=4000,
    ),
    SourceConfig(
        file_name="Amazon Products Reviews Dataset.csv",
        text_col="reviews.text", label_type="rating", label_col="reviews.rating",
        date_col="reviews.date", category="Electronics", source_name="Amazon",
        sample_size=3000,
    ),
    SourceConfig(
        file_name="App Store Reviews.csv",
        text_col="review", label_type="rating", label_col="rating",
        date_col="date", category="App", source_name="App Store",
        sample_size=4000,
    ),
    SourceConfig(
        file_name="Google Play Store Reviews.csv",
        text_col="content", label_type="rating", label_col="score",
        date_col="at", category="App", source_name="Play Store",
        sample_size=4000,
    ),
    SourceConfig(
        file_name="Flipkart Products Review Dataset.csv",
        text_col="Review", label_type="sentiment", label_col="Sentiment",
        category="E-commerce", source_name="Flipkart",
        sample_size=6000,
    ),
    SourceConfig(
        file_name="Product ReviewEqual.csv",
        text_col="Review", label_type="sentiment", label_col="Sentiment",
        category="E-commerce", source_name="Flipkart",
        sample_size=4000,
    ),
    SourceConfig(
        file_name="E-commerce Product Reviews.csv",
        text_col="Text", label_type="sentiment", label_col="Label",
        category="E-commerce", source_name="Website",
        sample_size=None,
    ),
    SourceConfig(
        file_name="Product Reviews Dataset.csv",
        text_col="ReviewText", label_type="rating", label_col="Rating",
        date_col="ReviewDate", category="E-commerce", source_name="Website",
        sample_size=None,
    ),
    SourceConfig(
        file_name="Zomato Reviews.csv",
        text_col="review", label_type="rating", label_col="rating",
        category="Restaurant", source_name="Zomato",
        sample_size=4000,
    ),
    SourceConfig(
        file_name="Restaurant_Reviews.csv",
        text_col="Review", label_type="binary", label_col="Liked",
        category="Restaurant", source_name="Survey",
        sample_size=None,
    ),
    SourceConfig(
        file_name="Coursera's Course Reviews Dataset.csv",
        text_col="Review", label_type="rating", label_col="Label",
        category="Course", source_name="Coursera",
        sample_size=5000,
    ),
    SourceConfig(
        file_name="Coursera's Courses Review Dataset.csv",
        text_col="Review", label_type="rating", label_col="Label",
        category="Course", source_name="Coursera",
        sample_size=5000,
    ),
    SourceConfig(
        file_name="IMDB Dataset.csv",
        text_col="review", label_type="sentiment", label_col="sentiment",
        category="Movie", source_name="IMDB",
        sample_size=4000,
    ),
]

# ── Excluded files (documented for transparency) ─────────────────────────────
#
#   Customer Feedback Dataset.csv    -> malformed CSV (all fields merged into
#                                        one column due to bad quoting); only
#                                        99 rows, not worth a custom parser.
#   Flipkart Product Reviews.csv     -> corrupted file encoding; duplicate of
#                                        "Flipkart Products Review Dataset.csv"
#                                        which is already included.
#   Product Review Ratio.csv         -> corrupted file encoding.
#   Product Review sentiment.csv     -> corrupted file encoding.
#
# All four were confirmed unreadable / malformed during inspection.
# The 13 sources above already give 7 diverse categories
# (Electronics, App, E-commerce, Restaurant, Course, Movie) and 8 platforms.
