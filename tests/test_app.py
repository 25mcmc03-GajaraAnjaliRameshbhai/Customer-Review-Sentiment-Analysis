"""
tests/test_app.py
-----------------
Smoke tests — verify file existence and that every module (including
every dashboard page) imports without error. No live Streamlit
server is needed for these; see the AppTest-based checks in
tests/test_dashboard.py for actual page-render tests.

Run:
    pytest tests/test_app.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


# ── File existence ────────────────────────────────────────────────────────────

def test_app_file_exists():
    assert (Path(__file__).parent.parent / "app.py").exists()

def test_train_file_exists():
    assert (Path(__file__).parent.parent / "train.py").exists()

def test_prepare_dataset_file_exists():
    assert (Path(__file__).parent.parent / "prepare_dataset.py").exists()

def test_requirements_file_exists():
    assert (Path(__file__).parent.parent / "requirements.txt").exists()

def test_notebook_file_exists():
    assert (Path(__file__).parent.parent / "ReviewLens_Notebook.ipynb").exists()

def test_notebook_is_valid_and_error_free():
    """
    The shipped notebook must be valid, parseable .ipynb JSON with no
    error outputs in any cell — it's meant to be read (and re-run) as
    a viva walkthrough, so a broken cell would undermine that purpose
    the moment someone opens it.
    """
    nbformat = pytest.importorskip("nbformat")
    nb_path = Path(__file__).parent.parent / "ReviewLens_Notebook.ipynb"
    nb = nbformat.read(nb_path, as_version=4)

    assert len(nb["cells"]) > 20  # a real walkthrough, not a stub

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        for out in cell.get("outputs", []):
            assert out.get("output_type") != "error", (
                f"Cell {i} has an error output: "
                f"{out.get('ename')}: {out.get('evalue')}"
            )


# ── Core module imports ────────────────────────────────────────────────────────

def test_preprocess_importable():
    from src.preprocess import preprocess_text, preprocess_batch
    assert callable(preprocess_text)
    assert callable(preprocess_batch)

def test_features_importable():
    from src.features import build_vectorizer
    assert callable(build_vectorizer)

def test_train_model_importable():
    from src.train_model import build_lr_pipeline, build_nb_pipeline, evaluate_pipeline
    assert callable(build_lr_pipeline)
    assert callable(build_nb_pipeline)
    assert callable(evaluate_pipeline)

def test_evaluate_importable():
    from src.evaluate import full_evaluation
    assert callable(full_evaluation)

def test_explain_importable():
    from src.explain import explain_prediction
    assert callable(explain_prediction)

def test_utils_importable():
    from src.utils import load_dataset, load_metadata, save_metadata, models_exist, get_color_for_label
    assert callable(load_dataset)
    assert callable(load_metadata)
    assert callable(save_metadata)
    assert callable(models_exist)
    assert callable(get_color_for_label)

def test_data_sources_importable():
    from src.data_sources import SOURCES, SourceConfig
    assert isinstance(SOURCES, list)
    assert len(SOURCES) > 0

def test_data_loader_importable():
    from src.data_loader import build_unified_dataset, load_source
    assert callable(build_unified_dataset)
    assert callable(load_source)

def test_dashboard_utils_importable():
    from src.dashboard_utils import predict_single, predict_batch, make_wordcloud, top_words_df
    assert callable(predict_single)
    assert callable(predict_batch)
    assert callable(make_wordcloud)
    assert callable(top_words_df)

def test_csv_mapper_importable():
    from src.csv_mapper import detect_columns, apply_mapping
    assert callable(detect_columns)
    assert callable(apply_mapping)


# ── Dashboard page modules ────────────────────────────────────────────────────
# Each sidebar item in app.py maps to exactly one file in src/pages/.
# These tests confirm every page module imports cleanly and exposes a
# render() function, before any Streamlit runtime checks run.

@pytest.mark.parametrize("page_name", [
    "overview", "predict", "explorer", "trends",
    "comparison", "summary", "performance", "export",
])
def test_page_module_importable_with_render_function(page_name):
    import importlib
    module = importlib.import_module(f"src.pages.{page_name}")
    assert hasattr(module, "render"), f"src/pages/{page_name}.py must define render()"
    assert callable(module.render)


# ── Directory structure ───────────────────────────────────────────────────────

def test_data_raw_directory_exists():
    assert (Path(__file__).parent.parent / "data" / "raw").is_dir()

def test_data_processed_directory_exists():
    assert (Path(__file__).parent.parent / "data" / "processed").is_dir()

def test_models_directory_exists():
    assert (Path(__file__).parent.parent / "models").is_dir()

def test_docs_directory_exists():
    assert (Path(__file__).parent.parent / "docs").is_dir()

def test_pages_directory_exists():
    assert (Path(__file__).parent.parent / "src" / "pages").is_dir()


# ── Functional smoke tests ────────────────────────────────────────────────────

def test_preprocess_basic_functionality():
    from src.preprocess import preprocess_text
    result = preprocess_text("This is AMAZING!!! <br> Best product ever.")
    assert isinstance(result, str)
    assert "<br>" not in result
    assert "!!!" not in result

def test_lr_pipeline_can_be_built():
    from src.train_model import build_lr_pipeline
    pipeline = build_lr_pipeline()
    assert pipeline is not None
    assert "tfidf" in pipeline.named_steps
    assert "clf" in pipeline.named_steps

def test_nb_pipeline_can_be_built():
    from src.train_model import build_nb_pipeline
    pipeline = build_nb_pipeline()
    assert "tfidf" in pipeline.named_steps
    assert "clf" in pipeline.named_steps

def test_full_evaluation_shape():
    from src.evaluate import full_evaluation
    y_true = ["positive", "negative", "neutral", "positive", "negative"]
    y_pred = ["positive", "negative", "neutral", "negative", "negative"]
    result = full_evaluation(y_true, y_pred)
    assert "accuracy" in result
    assert "macro_f1" in result
    assert "weighted_f1" in result
    assert "report" in result
    assert "confusion_matrix" in result
    assert 0.0 <= result["accuracy"] <= 1.0

def test_utils_color_helper():
    from src.utils import get_color_for_label
    assert get_color_for_label("positive").startswith("#")
    assert get_color_for_label("negative").startswith("#")
    assert get_color_for_label("neutral").startswith("#")
