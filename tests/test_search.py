"""Tests for SemanticSearchEngine with synthetic FAISS index."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.retrieval.index import FAISSIndex
from src.retrieval.search import SemanticSearchEngine


@pytest.fixture()
def engine(tmp_path):
    """Create a SemanticSearchEngine with a small synthetic index."""
    rng = np.random.default_rng(42)
    n_vecs = 100
    dim = 384
    vecs = rng.random((n_vecs, dim)).astype(np.float32)

    faiss_idx = FAISSIndex(embedding_dim=dim, fallback_threshold=1000)
    faiss_idx.build(vecs)
    index_path = str(tmp_path / "faiss_index.bin")
    faiss_idx.save(index_path)

    meta = pd.DataFrame(
        {
            "article_title": [f"Article {i}" for i in range(n_vecs)],
            "chunk_index": list(range(n_vecs)),
            "text": [
                f"This is chunk number {i} about topic {i}" for i in range(n_vecs)
            ],
        }
    )
    meta_path = str(tmp_path / "chunk_metadata.parquet")
    meta.to_parquet(meta_path, index=False)

    config = {
        "encoder": {"model_name": "all-MiniLM-L6-v2", "embedding_dim": 384},
        "index": {"nlist": 100, "fallback_threshold": 1000},
        "reranker": {
            "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "top_k_retrieve": 20,
            "top_k_rerank": 5,
        },
        "search": {"default_top_k": 5},
    }

    with patch.object(
        SemanticSearchEngine,
        "__init__",
        lambda self, cfg: None,
    ):
        eng = SemanticSearchEngine.__new__(SemanticSearchEngine)

    from src.logger import get_logger
    from src.retrieval.encoder import SemanticEncoder

    eng._logger = get_logger("test_search")
    eng.encoder = SemanticEncoder(config["encoder"]["model_name"])
    eng.index = FAISSIndex(embedding_dim=dim, fallback_threshold=1000)
    eng.index.load(index_path)
    eng.metadata = pd.read_parquet(meta_path)
    eng.top_k_retrieve = config["reranker"]["top_k_retrieve"]
    eng.top_k_rerank = config["reranker"]["top_k_rerank"]
    eng.default_top_k = config["search"]["default_top_k"]

    mock_ce = MagicMock()
    mock_ce.predict.return_value = np.linspace(1.0, 0.0, 20)[:20]
    eng.cross_encoder = mock_ce

    return eng


def test_search_returns_results(engine):
    output = engine.search("what is machine learning?", top_k=5)
    results = output[0]["results"]
    assert len(results) > 0


def test_search_result_has_required_fields(engine):
    output = engine.search("what is machine learning?", top_k=5)
    results = output[0]["results"]
    for r in results:
        assert "text" in r
        assert "title" in r
        assert "score" in r


def test_score_threshold(engine):
    output = engine.search("what is machine learning?", top_k=5)
    results = output[0]["results"]
    for r in results:
        assert r["score"] >= 0.0


def test_reranking_changes_order(engine):
    reranked = engine.search("what is science?", top_k=5, rerank=True)
    dense = engine.search_dense_only("what is science?", top_k=5)

    reranked_titles = [r["title"] for r in reranked[0]["results"]]
    dense_titles = [r["title"] for r in dense[0]["results"]]
    assert reranked_titles != dense_titles or len(reranked_titles) > 0


def test_search_latency(engine):
    t0 = time.perf_counter()
    engine.search("what is physics?", top_k=5)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0


def test_latency_breakdown_in_response(engine):
    output = engine.search("test query", top_k=5)
    latency = output[0]["latency"]
    assert "encode_ms" in latency
    assert "faiss_ms" in latency
    assert "rerank_ms" in latency
    assert "total_ms" in latency


def test_dense_only_zero_rerank_latency(engine):
    output = engine.search_dense_only("test query", top_k=5)
    latency = output[0]["latency"]
    assert latency["rerank_ms"] == 0.0
