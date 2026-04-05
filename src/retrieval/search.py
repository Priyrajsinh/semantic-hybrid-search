"""Semantic search engine: encode + retrieve + rerank."""

import functools
import time

import pandas as pd
from sentence_transformers import CrossEncoder

from src.exceptions import IndexNotFoundError
from src.logger import get_logger
from src.retrieval.encoder import SemanticEncoder
from src.retrieval.index import FAISSIndex


class SemanticSearchEngine:
    """Orchestrates dense retrieval and cross-encoder reranking."""

    def __init__(self, config: dict) -> None:
        self._logger = get_logger(__name__)

        self.encoder = SemanticEncoder(config["encoder"]["model_name"])

        self.index = FAISSIndex(
            embedding_dim=config["encoder"]["embedding_dim"],
            nlist=config["index"]["nlist"],
            fallback_threshold=config["index"]["fallback_threshold"],
        )
        index_path = "models/faiss_index.bin"
        try:
            self.index.load(index_path)
        except Exception as exc:
            raise IndexNotFoundError(f"FAISS index not found at {index_path}") from exc

        meta_path = "models/chunk_metadata.parquet"
        try:
            self.metadata = pd.read_parquet(meta_path)
        except Exception as exc:
            raise IndexNotFoundError(
                f"Chunk metadata not found at {meta_path}"
            ) from exc

        self.cross_encoder = CrossEncoder(config["reranker"]["model_name"])
        self.top_k_retrieve = config["reranker"]["top_k_retrieve"]
        self.top_k_rerank = config["reranker"]["top_k_rerank"]
        self.default_top_k = config["search"]["default_top_k"]

        self._logger.info(
            "SemanticSearchEngine ready: %d vectors, reranker=%s",
            self.index.n_vectors,
            config["reranker"]["model_name"],
        )

    def search(self, query: str, top_k: int = 5, rerank: bool = True) -> list[dict]:
        """Two-stage retrieval: FAISS dense -> cross-encoder rerank."""
        t0 = time.perf_counter()

        query_vec = self.encoder.encode_query(query)
        encode_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        retrieve_k = self.top_k_retrieve if rerank else top_k
        scores, indices = self.index.search(query_vec, top_k=retrieve_k)
        faiss_ms = (time.perf_counter() - t1) * 1000

        candidates = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            row = self.metadata.iloc[idx]
            candidates.append(
                {
                    "text": row["text"],
                    "title": row["article_title"],
                    "score": float(score),
                }
            )

        rerank_ms = 0.0
        if rerank and candidates:
            t2 = time.perf_counter()
            texts_tuple = tuple(c["text"] for c in candidates)
            ce_scores = self._cached_rerank(query, texts_tuple)
            for i, c in enumerate(candidates):
                c["score"] = ce_scores[i]
            candidates.sort(key=lambda x: x["score"], reverse=True)
            candidates = candidates[:top_k]
            rerank_ms = (time.perf_counter() - t2) * 1000

        for rank, c in enumerate(candidates, 1):
            c["rank"] = rank

        total_ms = (time.perf_counter() - t0) * 1000
        latency = {
            "total_ms": round(total_ms, 1),
            "encode_ms": round(encode_ms, 1),
            "faiss_ms": round(faiss_ms, 1),
            "rerank_ms": round(rerank_ms, 1),
        }

        self._logger.info(
            "Search '%s': encode=%.0fms faiss=%.0fms rerank=%.0fms total=%.0fms",
            query,
            encode_ms,
            faiss_ms,
            rerank_ms,
            total_ms,
        )

        return [{"results": candidates, "latency": latency}]

    @functools.lru_cache(maxsize=128)
    def _cached_rerank(
        self, query: str, candidate_texts: tuple[str, ...]
    ) -> tuple[float, ...]:
        """Cross-encoder scoring with LRU cache for repeated queries."""
        pairs = [(query, text) for text in candidate_texts]
        scores = self.cross_encoder.predict(pairs)
        return tuple(float(s) for s in scores)

    def search_dense_only(self, query: str, top_k: int = 5) -> list[dict]:
        """Dense retrieval without cross-encoder reranking."""
        return self.search(query, top_k=top_k, rerank=False)
