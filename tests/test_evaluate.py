"""Tests for evaluation module: MRR computation."""

import json
from unittest.mock import MagicMock, patch

from src.evaluation.evaluate import evaluate_retrieval, mean_reciprocal_rank


def test_mrr_perfect_ranking():
    results = [
        [{"title": "Einstein", "rank": 1, "score": 0.9, "text": "..."}],
    ]
    queries = [{"query": "who was Einstein?", "relevant_titles": ["Einstein"]}]
    mrr = mean_reciprocal_rank(results, queries)
    assert mrr == 1.0


def test_mrr_second_rank():
    results = [
        [
            {"title": "Newton", "rank": 1, "score": 0.9, "text": "..."},
            {"title": "Einstein", "rank": 2, "score": 0.8, "text": "..."},
        ],
    ]
    queries = [{"query": "who was Einstein?", "relevant_titles": ["Einstein"]}]
    mrr = mean_reciprocal_rank(results, queries)
    assert mrr == 0.5


def test_mrr_no_relevant():
    results = [
        [{"title": "Newton", "rank": 1, "score": 0.9, "text": "..."}],
    ]
    queries = [{"query": "who was Einstein?", "relevant_titles": ["Einstein"]}]
    mrr = mean_reciprocal_rank(results, queries)
    assert mrr == 0.0


def test_mrr_empty_queries():
    mrr = mean_reciprocal_rank([], [])
    assert mrr == 0.0


def test_mrr_case_insensitive():
    results = [
        [{"title": "einstein", "rank": 1, "score": 0.9, "text": "..."}],
    ]
    queries = [{"query": "who was Einstein?", "relevant_titles": ["Einstein"]}]
    mrr = mean_reciprocal_rank(results, queries)
    assert mrr == 1.0


def test_evaluate_retrieval_returns_metrics(tmp_path):
    """Test evaluate_retrieval with a mocked engine."""
    mock_engine = MagicMock()

    reranked_response = [
        {
            "results": [
                {"title": "Einstein", "rank": 1, "score": 0.9, "text": "about E"},
                {"title": "Newton", "rank": 2, "score": 0.8, "text": "about N"},
            ],
            "latency": {
                "encode_ms": 10.0,
                "faiss_ms": 5.0,
                "rerank_ms": 20.0,
                "total_ms": 35.0,
            },
        }
    ]
    dense_response = [
        {
            "results": [
                {"title": "Newton", "rank": 1, "score": 0.7, "text": "about N"},
                {"title": "Einstein", "rank": 2, "score": 0.6, "text": "about E"},
            ],
            "latency": {
                "encode_ms": 10.0,
                "faiss_ms": 5.0,
                "rerank_ms": 0.0,
                "total_ms": 15.0,
            },
        }
    ]

    mock_engine.search.return_value = reranked_response
    mock_engine.search_dense_only.return_value = dense_response

    queries = [{"query": "who was Einstein?", "relevant_titles": ["Einstein"]}]

    reports_dir = tmp_path / "reports"
    results_path = reports_dir / "results.json"

    with (
        patch("src.evaluation.evaluate.REPORTS_DIR", reports_dir),
        patch("src.evaluation.evaluate.mlflow") as mock_mlflow,
    ):
        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        result = evaluate_retrieval(mock_engine, queries, top_k=10)

    assert "mrr_dense" in result
    assert "mrr_reranked" in result
    assert "improvement" in result
    assert "encode_ms_avg" in result
    assert result["mrr_reranked"] == 1.0
    assert result["mrr_dense"] == 0.5
    assert results_path.exists()

    with open(str(results_path)) as fh:
        saved = json.load(fh)
    assert saved["mrr_reranked"] == 1.0
