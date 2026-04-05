"""Tests for chunker, validation, dataset loading, and data pipeline."""

import json

import pandas as pd
import pytest

import src.data.pipeline as pipeline_mod
from src.data.chunker import chunk_articles
from src.data.dataset import load_wikipedia
from src.data.pipeline import run_pipeline
from src.data.validation import validate_chunks
from src.evaluation.eval_queries import (
    _curated_queries,
    build_eval_queries,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeDataset:
    """Minimal HuggingFace Dataset stand-in for monkeypatching."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def select(self, indices):
        return _FakeDataset([self._rows[i] for i in indices])


def _valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_title": ["Alpha", "Beta"],
            "chunk_index": [0, 0],
            "text": ["hello world foo bar baz", "another passage here ok"],
        }
    )


# ---------------------------------------------------------------------------
# chunk_articles tests
# ---------------------------------------------------------------------------


def test_chunk_single_article():
    """256-token chunks with 50-token overlap produce the expected count."""
    tokens = list(range(300))
    text = " ".join(str(t) for t in tokens)
    articles = [{"title": "T", "text": text}]
    df = chunk_articles(articles, chunk_size=256, chunk_overlap=50)
    # step = 256 - 50 = 206; ceil((300 - 256) / 206) + 1 = 2 chunks
    assert len(df) == 2
    # First chunk: tokens 0-255 → 256 tokens
    assert len(df.iloc[0]["text"].split()) == 256
    # Second chunk starts at token 206 → overlaps with first
    first_end_token = df.iloc[0]["text"].split()[-50:]
    second_start_token = df.iloc[1]["text"].split()[:50]
    assert first_end_token == second_start_token


def test_chunk_empty_text_skipped():
    articles = [
        {"title": "Empty", "text": ""},
        {"title": "Blank", "text": "   "},
        {"title": "Valid", "text": "some real content here"},
    ]
    df = chunk_articles(articles)
    assert all(df["article_title"] == "Valid")
    assert len(df) >= 1


def test_chunk_metadata_columns():
    articles = [{"title": "Doc", "text": "one two three four five six"}]
    df = chunk_articles(articles)
    assert list(df.columns) == ["article_title", "chunk_index", "text"]
    assert df["chunk_index"].dtype == int or df["chunk_index"].dtype == "int64"


# ---------------------------------------------------------------------------
# validation tests
# ---------------------------------------------------------------------------


def test_validation_passes():
    df = _valid_df()
    result = validate_chunks(df)
    assert len(result) == len(df)


def test_validation_rejects_missing_title():
    df = pd.DataFrame(
        {
            "article_title": [None],
            "chunk_index": [0],
            "text": ["long enough text to pass length check"],
        }
    )
    with pytest.raises(Exception):
        validate_chunks(df)


def test_validation_rejects_short_text():
    df = pd.DataFrame(
        {
            "article_title": ["Title"],
            "chunk_index": [0],
            "text": ["short"],
        }
    )
    with pytest.raises(Exception):
        validate_chunks(df)


def test_validation_rejects_negative_chunk_index():
    df = pd.DataFrame(
        {
            "article_title": ["Title"],
            "chunk_index": [-1],
            "text": ["long enough text to pass length check"],
        }
    )
    with pytest.raises(Exception):
        validate_chunks(df)


# ---------------------------------------------------------------------------
# load_wikipedia tests
# ---------------------------------------------------------------------------


def test_load_wikipedia_returns_list(monkeypatch):
    rows = [{"title": f"Art {i}", "text": f"body {i}"} for i in range(5)]
    monkeypatch.setattr(
        "src.data.dataset.load_dataset",
        lambda *a, **kw: _FakeDataset(rows),
    )
    config = {
        "data": {
            "dataset_name": "wikipedia",
            "dataset_config": "20220301.simple",
            "n_articles": 3,
        }
    }
    result = load_wikipedia(config)
    assert isinstance(result, list)
    assert len(result) == 3


def test_load_wikipedia_dict_keys(monkeypatch):
    rows = [{"title": "Alpha", "text": "some body text"}]
    monkeypatch.setattr(
        "src.data.dataset.load_dataset",
        lambda *a, **kw: _FakeDataset(rows),
    )
    config = {
        "data": {
            "dataset_name": "wikipedia",
            "dataset_config": "20220301.simple",
            "n_articles": 10,
        }
    }
    result = load_wikipedia(config)
    assert result[0]["title"] == "Alpha"
    assert result[0]["text"] == "some body text"


# ---------------------------------------------------------------------------
# run_pipeline tests
# ---------------------------------------------------------------------------


def test_run_pipeline_creates_parquet(monkeypatch, tmp_path):
    fake_articles = [{"title": "Wiki", "text": "hello world foo bar baz"}]
    fake_df = _valid_df()

    monkeypatch.setattr("src.data.pipeline.load_wikipedia", lambda cfg: fake_articles)
    monkeypatch.setattr(
        "src.data.pipeline.chunk_articles",
        lambda articles, **kw: fake_df,
    )
    monkeypatch.setattr("src.data.pipeline.validate_chunks", lambda df: df)
    monkeypatch.setattr(pipeline_mod, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(pipeline_mod, "RAW_DIR", tmp_path / "raw")

    run_pipeline()

    assert (tmp_path / "processed" / "chunks.parquet").exists()


def test_run_pipeline_creates_checksums(monkeypatch, tmp_path):
    fake_articles = [{"title": "Wiki", "text": "hello world foo bar baz"}]
    fake_df = _valid_df()

    monkeypatch.setattr("src.data.pipeline.load_wikipedia", lambda cfg: fake_articles)
    monkeypatch.setattr(
        "src.data.pipeline.chunk_articles",
        lambda articles, **kw: fake_df,
    )
    monkeypatch.setattr("src.data.pipeline.validate_chunks", lambda df: df)
    monkeypatch.setattr(pipeline_mod, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(pipeline_mod, "RAW_DIR", tmp_path / "raw")

    run_pipeline()

    checksums_file = tmp_path / "raw" / "checksums.json"
    assert checksums_file.exists()
    with open(str(checksums_file)) as fh:
        data = json.load(fh)
    assert "chunks.parquet" in data
    assert len(data["chunks.parquet"]) == 64  # SHA-256 hex length


# ---------------------------------------------------------------------------
# eval_queries tests
# ---------------------------------------------------------------------------


def test_curated_queries_matches_known_title():
    articles = [{"title": "Paris", "text": "capital of France"}]
    pairs = _curated_queries(articles)
    assert any("Paris" in p["relevant_titles"] for p in pairs)
    assert all("query" in p and "relevant_titles" in p for p in pairs)


def test_curated_queries_no_match_returns_empty():
    articles = [{"title": "Nonexistent Article XYZ 999", "text": "blah"}]
    pairs = _curated_queries(articles)
    assert pairs == []


def test_build_eval_queries_success_path(monkeypatch, tmp_path):
    articles = [{"title": "Paris", "text": "capital"}]
    fake_pairs = [{"query": "q?", "relevant_titles": ["Paris"]}] * 60

    monkeypatch.setattr(
        "src.evaluation.eval_queries._filter_msmarco",
        lambda arts, max_q: fake_pairs[:max_q],
    )
    out = tmp_path / "eval_queries.json"
    result = build_eval_queries(articles, output_path=out, max_queries=200)

    assert out.exists()
    assert len(result) == 60


def test_build_eval_queries_fallback_on_few_matches(monkeypatch, tmp_path):
    articles = [{"title": "Paris", "text": "capital"}]

    monkeypatch.setattr(
        "src.evaluation.eval_queries._filter_msmarco",
        lambda arts, max_q: [{"query": "q?", "relevant_titles": ["Paris"]}],
    )
    out = tmp_path / "eval_queries.json"
    result = build_eval_queries(articles, output_path=out)

    assert out.exists()
    assert len(result) >= 1  # fell back to curated set


def test_build_eval_queries_fallback_on_exception(monkeypatch, tmp_path):
    articles = [{"title": "Paris", "text": "capital"}]

    def _raise(*args, **kwargs):
        raise RuntimeError("network error")

    monkeypatch.setattr("src.evaluation.eval_queries._filter_msmarco", _raise)
    out = tmp_path / "eval_queries.json"
    result = build_eval_queries(articles, output_path=out)

    assert out.exists()
    assert isinstance(result, list)
