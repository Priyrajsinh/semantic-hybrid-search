"""Tests for src/retrieval/build_index.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import src.retrieval.build_index as _build_mod  # noqa: F401,E501 — ensures module is imported
from src.retrieval.build_index import build_index


@pytest.fixture()
def mock_chunks_df():
    return pd.DataFrame(
        {
            "article_title": ["Alpha", "Beta"],
            "chunk_index": [0, 0],
            "text": ["Alpha text content here.", "Beta text content here."],
        }
    )


@pytest.fixture()
def minimal_config():
    return {
        "encoder": {
            "model_name": "all-MiniLM-L6-v2",
            "embedding_dim": 384,
            "batch_size": 32,
        },
        "index": {
            "nlist": 100,
            "fallback_threshold": 1000,
            "nprobe": 10,
        },
        "mlflow": {"experiment_name": "test-experiment"},
    }


def test_build_index_runs(mock_chunks_df, minimal_config):
    """build_index() calls encoder, FAISSIndex.build, and save without error."""
    fake_embeddings = np.random.default_rng(0).random((2, 384)).astype(np.float32)

    mock_idx = MagicMock()
    mock_idx.n_vectors = 2

    mock_enc = MagicMock()
    mock_enc.encode.return_value = fake_embeddings

    with (
        patch("builtins.open", MagicMock()),
        patch("yaml.safe_load", return_value=minimal_config),
        patch("pandas.read_parquet", return_value=mock_chunks_df),
        patch("pandas.DataFrame.to_parquet"),
        patch("pathlib.Path.mkdir"),
        patch("src.retrieval.build_index.SemanticEncoder", return_value=mock_enc),
        patch("src.retrieval.build_index.FAISSIndex", return_value=mock_idx),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run") as mock_run,
    ):
        mock_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_run.return_value.__exit__ = MagicMock(return_value=False)

        build_index()

    mock_enc.encode.assert_called_once()
    mock_idx.build.assert_called_once()
    mock_idx.save.assert_called_once()
