"""Retrieval evaluation: MRR@K for dense vs reranked search."""

import json
from pathlib import Path

import mlflow
import yaml

from src.logger import get_logger
from src.retrieval.search import SemanticSearchEngine

_logger = get_logger(__name__)

EVAL_QUERIES_PATH = Path("data/processed/eval_queries.json")
REPORTS_DIR = Path("reports")
CONFIG_PATH = Path("config/config.yaml")


def _load_eval_queries() -> list[dict]:
    """Load eval queries from data/processed/eval_queries.json."""
    with open(str(EVAL_QUERIES_PATH)) as fh:
        queries = json.load(fh)
    _logger.info("Loaded %d eval queries from %s", len(queries), EVAL_QUERIES_PATH)
    return queries


def mean_reciprocal_rank(results: list[list[dict]], queries: list[dict]) -> float:
    """Compute MRR: mean of 1/rank for the first relevant result per query."""
    rr_sum = 0.0
    for query_info, result_list in zip(queries, results):
        relevant = {t.lower() for t in query_info["relevant_titles"]}
        for item in result_list:
            if item["title"].lower() in relevant:
                rr_sum += 1.0 / item["rank"]
                break
    return rr_sum / len(queries) if queries else 0.0


def evaluate_retrieval(
    engine: SemanticSearchEngine,
    queries: list[dict],
    top_k: int = 10,
) -> dict:
    """Run MRR evaluation for dense-only and reranked search."""
    reranked_results: list[list[dict]] = []
    dense_results: list[list[dict]] = []

    latency_encode: list[float] = []
    latency_faiss: list[float] = []
    latency_rerank: list[float] = []

    for q in queries:
        reranked = engine.search(q["query"], top_k=top_k, rerank=True)
        reranked_results.append(reranked[0]["results"])
        lat = reranked[0]["latency"]
        latency_encode.append(lat["encode_ms"])
        latency_faiss.append(lat["faiss_ms"])
        latency_rerank.append(lat["rerank_ms"])

        dense = engine.search_dense_only(q["query"], top_k=top_k)
        dense_results.append(dense[0]["results"])

    mrr_reranked = mean_reciprocal_rank(reranked_results, queries)
    mrr_dense = mean_reciprocal_rank(dense_results, queries)
    improvement = mrr_reranked - mrr_dense

    encode_avg = sum(latency_encode) / len(latency_encode) if latency_encode else 0
    faiss_avg = sum(latency_faiss) / len(latency_faiss) if latency_faiss else 0
    rerank_avg = sum(latency_rerank) / len(latency_rerank) if latency_rerank else 0

    result = {
        "mrr_dense": round(mrr_dense, 4),
        "mrr_reranked": round(mrr_reranked, 4),
        "improvement": round(improvement, 4),
        "encode_ms_avg": round(encode_avg, 1),
        "faiss_ms_avg": round(faiss_avg, 1),
        "rerank_ms_avg": round(rerank_avg, 1),
        "n_queries": len(queries),
    }

    _logger.info(
        "MRR dense=%.4f reranked=%.4f improvement=%.4f",
        mrr_dense,
        mrr_reranked,
        improvement,
    )

    with open(str(CONFIG_PATH)) as fh:
        config = yaml.safe_load(fh)

    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="evaluate_retrieval"):
        mlflow.log_metrics(
            {
                "mrr_dense": mrr_dense,
                "mrr_reranked": mrr_reranked,
                "mrr_improvement": improvement,
                "encode_ms_avg": encode_avg,
                "faiss_ms_avg": faiss_avg,
                "rerank_ms_avg": rerank_avg,
            }
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = REPORTS_DIR / "results.json"
    with open(str(results_path), "w") as fh:
        json.dump(result, fh, indent=2)
    _logger.info("Saved evaluation results to %s", results_path)

    return result


if __name__ == "__main__":
    with open(str(CONFIG_PATH)) as fh:
        cfg = yaml.safe_load(fh)
    engine = SemanticSearchEngine(cfg)
    queries = _load_eval_queries()
    evaluate_retrieval(engine, queries)
