"""Retrieval evaluation: MRR@K for dense, BM25, hybrid, and hybrid+reranked."""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import yaml

from src.logger import get_logger
from src.retrieval.search import SemanticSearchEngine

matplotlib.use("Agg")
plt.switch_backend("Agg")

_logger = get_logger(__name__)

EVAL_QUERIES_PATH = Path("data/processed/eval_queries.json")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
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
    """Run MRR evaluation for 4 methods: dense, BM25, hybrid, hybrid+reranked."""
    dense_results: list[list[dict]] = []
    bm25_results: list[list[dict]] = []
    hybrid_results: list[list[dict]] = []
    hybrid_reranked_results: list[list[dict]] = []

    latency_encode: list[float] = []
    latency_faiss: list[float] = []
    latency_rerank: list[float] = []
    latency_bm25: list[float] = []

    for q in queries:
        # Dense only
        dense = engine.search_dense_only(q["query"], top_k=top_k)
        dense_results.append(dense[0]["results"])

        # Reranked (for latency tracking)
        reranked = engine.search(q["query"], top_k=top_k, rerank=True)
        lat = reranked[0]["latency"]
        latency_encode.append(lat["encode_ms"])
        latency_faiss.append(lat["faiss_ms"])
        latency_rerank.append(lat["rerank_ms"])

        # BM25 only
        bm25 = engine.search_bm25(q["query"], top_k=top_k)
        bm25_results.append(bm25[0]["results"])
        latency_bm25.append(bm25[0]["latency"]["bm25_ms"])

        # Hybrid without reranking
        hybrid = engine.search_hybrid(q["query"], top_k=top_k, rerank=False)
        hybrid_results.append(hybrid[0]["results"])

        # Hybrid with reranking
        hybrid_rr = engine.search_hybrid(q["query"], top_k=top_k, rerank=True)
        hybrid_reranked_results.append(hybrid_rr[0]["results"])

    mrr_dense = mean_reciprocal_rank(dense_results, queries)
    mrr_bm25 = mean_reciprocal_rank(bm25_results, queries)
    mrr_hybrid = mean_reciprocal_rank(hybrid_results, queries)
    mrr_hybrid_reranked = mean_reciprocal_rank(hybrid_reranked_results, queries)

    encode_avg = sum(latency_encode) / len(latency_encode) if latency_encode else 0
    faiss_avg = sum(latency_faiss) / len(latency_faiss) if latency_faiss else 0
    rerank_avg = sum(latency_rerank) / len(latency_rerank) if latency_rerank else 0
    bm25_avg = sum(latency_bm25) / len(latency_bm25) if latency_bm25 else 0

    result = {
        "dense_only": {"mrr@10": round(mrr_dense, 4)},
        "bm25_only": {"mrr@10": round(mrr_bm25, 4)},
        "hybrid_rrf": {"mrr@10": round(mrr_hybrid, 4)},
        "hybrid_reranked": {"mrr@10": round(mrr_hybrid_reranked, 4)},
        "latency": {
            "encode_ms_avg": round(encode_avg, 1),
            "faiss_ms_avg": round(faiss_avg, 1),
            "rerank_ms_avg": round(rerank_avg, 1),
            "bm25_ms_avg": round(bm25_avg, 1),
        },
        "n_queries": len(queries),
    }

    _logger.info(
        "MRR dense=%.4f bm25=%.4f hybrid=%.4f hybrid_reranked=%.4f",
        mrr_dense,
        mrr_bm25,
        mrr_hybrid,
        mrr_hybrid_reranked,
    )

    # MLflow logging
    with open(str(CONFIG_PATH)) as fh:
        config = yaml.safe_load(fh)

    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="evaluate_retrieval"):
        mlflow.log_metrics(
            {
                "mrr_dense": mrr_dense,
                "mrr_bm25": mrr_bm25,
                "mrr_hybrid_rrf": mrr_hybrid,
                "mrr_hybrid_reranked": mrr_hybrid_reranked,
                "encode_ms_avg": encode_avg,
                "faiss_ms_avg": faiss_avg,
                "rerank_ms_avg": rerank_avg,
                "bm25_ms_avg": bm25_avg,
            }
        )

    # Save results
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = REPORTS_DIR / "results.json"
    with open(str(results_path), "w") as fh:
        json.dump(result, fh, indent=2)
    _logger.info("Saved evaluation results to %s", results_path)

    # Generate comparison bar chart
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    methods = ["Dense Only", "BM25 Only", "Hybrid RRF", "Hybrid + Rerank"]
    mrr_values = [mrr_dense, mrr_bm25, mrr_hybrid, mrr_hybrid_reranked]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        methods, mrr_values, color=["#4285F4", "#EA4335", "#FBBC04", "#34A853"]
    )
    ax.set_ylabel("MRR@10")
    ax.set_title("Retrieval Method Comparison — MRR@10")
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, mrr_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.4f}",
            ha="center",
            fontsize=10,
        )
    fig.tight_layout()
    chart_path = FIGURES_DIR / "mrr_comparison.png"
    fig.savefig(str(chart_path), dpi=150)
    plt.close(fig)
    _logger.info("Saved MRR comparison chart to %s", chart_path)

    return result


if __name__ == "__main__":
    with open(str(CONFIG_PATH)) as fh:
        cfg = yaml.safe_load(fh)
    engine = SemanticSearchEngine(cfg)
    queries = _load_eval_queries()
    evaluate_retrieval(engine, queries)
