"""
tests/test_preprocess.py
------------------------
Unit tests for src/preprocess.py
Run: pytest tests/test_preprocess.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.preprocess import (
    preprocess_text,
    preprocess_batch,
    remove_html,
    remove_urls,
    remove_special_chars,
    remove_extra_whitespace,
)


# ── remove_html ───────────────────────────────────────────────────────────────

class TestRemoveHtml:
    def test_removes_br_tag(self):
        assert "<br>" not in remove_html("Hello <br> World")

    def test_removes_anchor_tag(self):
        result = remove_html("<a href='http://x.com'>link</a>")
        assert "<a" not in result and "</a>" not in result

    def test_removes_bold_tag(self):
        assert "<b>" not in remove_html("Some <b>bold</b> text")

    def test_plain_text_unchanged(self):
        text = "plain text without tags"
        assert text in remove_html(text)


# ── remove_urls ───────────────────────────────────────────────────────────────

class TestRemoveUrls:
    def test_removes_https_url(self):
        result = remove_urls("Visit https://example.com for info")
        assert "https://example.com" not in result

    def test_removes_http_url(self):
        result = remove_urls("See http://shop.com/product?id=1")
        assert "http://" not in result

    def test_removes_www_url(self):
        result = remove_urls("Go to www.flipkart.com now")
        assert "www.flipkart.com" not in result

    def test_no_false_removal(self):
        result = remove_urls("Great product overall")
        assert "Great product" in result


# ── remove_special_chars ──────────────────────────────────────────────────────

class TestRemoveSpecialChars:
    def test_removes_exclamation(self):
        assert "!" not in remove_special_chars("Amazing!!!")

    def test_removes_hash(self):
        assert "#" not in remove_special_chars("Good #product")

    def test_keeps_letters_and_digits(self):
        result = remove_special_chars("abc 123")
        assert "abc" in result and "123" in result


# ── preprocess_text ───────────────────────────────────────────────────────────

class TestPreprocessText:

    def test_none_returns_empty(self):
        assert preprocess_text(None) == ""

    def test_empty_string_returns_empty(self):
        assert preprocess_text("") == ""

    def test_whitespace_only_returns_empty(self):
        assert preprocess_text("     ") == ""

    def test_lowercase_conversion(self):
        result = preprocess_text("THIS IS UPPERCASE TEXT")
        assert result == result.lower()

    def test_html_removed(self):
        result = preprocess_text("Great product <br> very good quality")
        assert "<br>" not in result

    def test_url_removed(self):
        result = preprocess_text("Check https://amazon.com for the deal")
        assert "https" not in result

    def test_stopwords_removed(self):
        result = preprocess_text("this is a very good product")
        tokens = result.split()
        for stopword in ["this", "is", "a"]:
            assert stopword not in tokens

    def test_special_chars_removed(self):
        result = preprocess_text("Hello!!! Great #product @mention")
        for char in ["!", "#", "@"]:
            assert char not in result

    def test_combined_pipeline(self):
        """Full pipeline test matching the spec example."""
        raw = "This product is AMAZING!!! <br> Best purchase ever."
        result = preprocess_text(raw)
        assert "amazing" in result
        assert "purchase" in result
        assert "!!!" not in result
        assert "<br>" not in result

    def test_non_string_input_returns_empty(self):
        assert preprocess_text(123) == ""
        assert preprocess_text([]) == ""


# ── preprocess_batch ──────────────────────────────────────────────────────────

class TestPreprocessBatch:

    def test_returns_list(self):
        result = preprocess_batch(["Good product", "Bad quality"])
        assert isinstance(result, list)

    def test_same_length_as_input(self):
        texts  = ["A", "B", "C", None, ""]
        result = preprocess_batch(texts)
        assert len(result) == len(texts)

    def test_handles_none_elements(self):
        result = preprocess_batch([None, "great product"])
        assert result[0] == ""
        assert isinstance(result[1], str)
