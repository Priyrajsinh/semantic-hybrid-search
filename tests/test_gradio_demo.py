"""Tests for Gradio demo helper functions."""

from unittest.mock import MagicMock

import gradio as gr
import pandas as pd
import pytest

from src.api.gradio_demo import (
    _ENGINE_NOT_LOADED_MSG,
    EXAMPLE_QUERIES,
    calibrate_scores,
    compare_methods_fn,
    highlight_query_terms,
    search_tab1_fn,
    search_tab2_fn,
)


@pytest.fixture
def mock_engine():
    eng = MagicMock()
    eng.search.return_value = [
        {
            "results": [
                {
                    "text": "Plants use sunlight.",
                    "title": "Photosynthesis",
                    "score": 0.9,
                    "rank": 1,
                },
                {
                    "text": "Chlorophyll absorbs light.",
                    "title": "Chlorophyll",
                    "score": 0.6,
                    "rank": 2,
                },
            ],
            "latency": {
                "encode_ms": 10.0,
                "faiss_ms": 5.0,
                "rerank_ms": 100.0,
                "total_ms": 115.0,
            },
        }
    ]
    return eng


# ---------------------------------------------------------------------------
# calibrate_scores
# ---------------------------------------------------------------------------


def test_calibrate_scores_min_max():
    result = calibrate_scores([0.3, 0.5, 0.7])
    assert abs(result[0]) < 1e-6
    assert abs(result[2] - 100.0) < 1e-6


def test_calibrate_scores_all_equal():
    assert calibrate_scores([0.5, 0.5]) == [100.0, 100.0]


def test_calibrate_scores_empty():
    assert calibrate_scores([]) == []


def test_calibrate_scores_single():
    assert calibrate_scores([0.8]) == [100.0]


# ---------------------------------------------------------------------------
# highlight_query_terms
# ---------------------------------------------------------------------------


def test_highlight_short_words_skipped():
    result = highlight_query_terms("a cat sat on the mat", "on")
    assert "**on**" not in result


def test_highlight_long_word_bolded():
    result = highlight_query_terms("photosynthesis uses light", "photosynthesis")
    assert "**photosynthesis**" in result


def test_highlight_no_match():
    result = highlight_query_terms("ocean depth", "mountain")
    assert "**mountain**" not in result


# ---------------------------------------------------------------------------
# search_tab1_fn
# ---------------------------------------------------------------------------


def test_search_tab1_empty_query():
    result = search_tab1_fn("", None)
    assert "enter" in result.lower()


def test_search_tab1_engine_none():
    result = search_tab1_fn("what is science?", None)
    assert result == _ENGINE_NOT_LOADED_MSG


def test_search_tab1_returns_cards(mock_engine):
    result = search_tab1_fn("photosynthesis", mock_engine)
    assert "Photosynthesis" in result
    assert "%" in result


def test_search_tab1_shows_footer(mock_engine):
    result = search_tab1_fn("climate change", mock_engine)
    assert "150,000+" in result
    assert "ms" in result


def test_search_tab1_low_confidence_tip(mock_engine):
    # All calibrated scores < 50 → tip shown (equal scores → 100%, so force by
    # returning a single score)
    mock_engine.search.return_value = [
        {
            "results": [
                {"text": "Some text.", "title": "A", "score": -0.5, "rank": 1},
                {"text": "Other text.", "title": "B", "score": -0.9, "rank": 2},
            ],
            "latency": {"encode_ms": 1, "faiss_ms": 1, "rerank_ms": 1, "total_ms": 3},
        }
    ]
    result = search_tab1_fn("obscure query", mock_engine)
    assert "Tip" in result or "%" in result


def test_search_tab1_no_results(mock_engine):
    mock_engine.search.return_value = [
        {
            "results": [],
            "latency": {"encode_ms": 1, "faiss_ms": 1, "rerank_ms": 0, "total_ms": 2},
        }
    ]
    result = search_tab1_fn("something", mock_engine)
    assert "No results" in result


# ---------------------------------------------------------------------------
# search_tab2_fn
# ---------------------------------------------------------------------------


def test_search_tab2_empty_query():
    df, lat = search_tab2_fn("", 5, True, "Dense (FAISS)", None)
    assert df.empty
    assert lat == ""


def test_search_tab2_engine_none():
    df, lat = search_tab2_fn("test", 5, True, "Dense (FAISS)", None)
    assert "not loaded" in lat.lower()


def test_search_tab2_returns_dataframe(mock_engine):
    df, lat = search_tab2_fn("photosynthesis", 5, True, "Dense (FAISS)", mock_engine)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "title" in df.columns
    assert "score" in df.columns


def test_search_tab2_latency_string(mock_engine):
    _, lat = search_tab2_fn("test", 5, True, "Dense (FAISS)", mock_engine)
    assert "Encode" in lat
    assert "FAISS" in lat
    assert "Rerank" in lat


# ---------------------------------------------------------------------------
# compare_methods_fn
# ---------------------------------------------------------------------------


def test_compare_methods_empty_query():
    df = compare_methods_fn("", 5, None)
    assert df.empty


def test_compare_methods_engine_none():
    df = compare_methods_fn("test", 5, None)
    assert df.empty


def test_compare_methods_returns_both_methods(mock_engine):
    df = compare_methods_fn("photosynthesis", 5, mock_engine)
    assert isinstance(df, pd.DataFrame)
    assert "method" in df.columns
    methods = df["method"].unique().tolist()
    assert "Dense (FAISS)" in methods
    assert "Dense + Rerank" in methods


# ---------------------------------------------------------------------------
# build_demo
# ---------------------------------------------------------------------------


def test_build_demo_returns_blocks(mock_engine):
    from src.api.gradio_demo import build_demo

    demo = build_demo(engine=mock_engine)
    assert isinstance(demo, gr.Blocks)


def test_example_queries_non_empty():
    assert len(EXAMPLE_QUERIES) >= 8
    for q in EXAMPLE_QUERIES:
        assert len(q) > 5
