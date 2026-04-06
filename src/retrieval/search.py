"""Semantic search engine: encode + retrieve + rerank + hybrid BM25/RRF."""

import functools
import os
import time

import joblib
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.exceptions import IndexNotFoundError
from src.logger import get_logger
from src.retrieval.encoder import SemanticEncoder
from src.retrieval.index import FAISSIndex


class SemanticSearchEngine:
    """Orchestrates dense retrieval, BM25 sparse, and hybrid RRF search."""

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

        # BM25 sparse index — persist to avoid rebuilding on startup
        bm25_path = "models/bm25_index.pkl"
        if os.path.exists(bm25_path):
            self.bm25 = joblib.load(bm25_path)
            self._logger.info("Loaded persisted BM25 index from %s", bm25_path)
        else:
            tokenized_corpus = [
                text.lower().split() for text in self.metadata["text"].tolist()
            ]
            self.bm25 = BM25Okapi(tokenized_corpus)
            os.makedirs(os.path.dirname(bm25_path), exist_ok=True)
            joblib.dump(self.bm25, bm25_path)
            self._logger.info("Built and persisted BM25 index to %s", bm25_path)

        self.cross_encoder = CrossEncoder(config["reranker"]["model_name"])
        self.top_k_retrieve = config["reranker"]["top_k_retrieve"]
        self.top_k_rerank = config["reranker"]["top_k_rerank"]
        self.default_top_k = config["search"]["default_top_k"]

        # Hybrid config with defaults
        hybrid_cfg = config.get("hybrid", {})
        self.bm25_top_k = hybrid_cfg.get("bm25_top_k", 20)
        self.dense_top_k = hybrid_cfg.get("dense_top_k", 20)
        self.rrf_k = hybrid_cfg.get("rrf_k", 60)
        self.hybrid_final_top_k = hybrid_cfg.get("final_top_k", 5)

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

    def search_bm25(self, query: str, top_k: int = 20) -> list[dict]:
        """BM25 sparse retrieval."""
        t0 = time.perf_counter()
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        bm25_ms = (time.perf_counter() - t0) * 1000

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices, 1):
            if scores[idx] <= 0:
                break
            row = self.metadata.iloc[idx]
            results.append(
                {
                    "text": row["text"],
                    "title": row["article_title"],
                    "score": float(scores[idx]),
                    "rank": rank,
                }
            )

        self._logger.info(
            "BM25 search '%s': bm25=%.0fms results=%d",
            query,
            bm25_ms,
            len(results),
        )
        return [{"results": results, "latency": {"bm25_ms": round(bm25_ms, 1)}}]

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        k: int = 60,
        rerank: bool = True,
    ) -> list[dict]:
        """Hybrid search: dense + BM25 combined via Reciprocal Rank Fusion."""
        t0 = time.perf_counter()

        # Dense retrieval
        query_vec = self.encoder.encode_query(query)
        encode_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        faiss_scores, faiss_indices = self.index.search(
            query_vec, top_k=self.dense_top_k
        )
        faiss_ms = (time.perf_counter() - t1) * 1000

        # BM25 retrieval
        t2 = time.perf_counter()
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][: self.bm25_top_k]
        bm25_ms = (time.perf_counter() - t2) * 1000

        # Reciprocal Rank Fusion
        rrf_scores: dict[int, float] = {}

        for rank, idx in enumerate(faiss_indices, 1):
            if idx < 0:
                continue
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank)

        for rank, idx in enumerate(bm25_top_indices, 1):
            if bm25_scores[idx] <= 0:
                break
            idx_int = int(idx)
            rrf_scores[idx_int] = rrf_scores.get(idx_int, 0.0) + 1.0 / (k + rank)

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        candidates = []
        for doc_idx, score in sorted_docs[:top_k] if not rerank else sorted_docs[:20]:
            row = self.metadata.iloc[doc_idx]
            candidates.append(
                {
                    "text": row["text"],
                    "title": row["article_title"],
                    "score": float(score),
                    "source": "hybrid",
                }
            )

        # Optional cross-encoder reranking
        rerank_ms = 0.0
        if rerank and candidates:
            t3 = time.perf_counter()
            texts_tuple = tuple(c["text"] for c in candidates)
            ce_scores = self._cached_rerank(query, texts_tuple)
            for i, c in enumerate(candidates):
                c["score"] = ce_scores[i]
            candidates.sort(key=lambda x: x["score"], reverse=True)
            candidates = candidates[:top_k]
            rerank_ms = (time.perf_counter() - t3) * 1000

        for rank, c in enumerate(candidates, 1):
            c["rank"] = rank

        total_ms = (time.perf_counter() - t0) * 1000
        latency = {
            "total_ms": round(total_ms, 1),
            "encode_ms": round(encode_ms, 1),
            "faiss_ms": round(faiss_ms, 1),
            "bm25_ms": round(bm25_ms, 1),
            "rerank_ms": round(rerank_ms, 1),
        }

        self._logger.info(
            "Hybrid search '%s': encode=%.0fms faiss=%.0fms bm25=%.0fms "
            "rerank=%.0fms total=%.0fms",
            query,
            encode_ms,
            faiss_ms,
            bm25_ms,
            rerank_ms,
            total_ms,
        )

        return [{"results": candidates, "latency": latency}]

    def search_dense_only(self, query: str, top_k: int = 5) -> list[dict]:
        """Dense retrieval without cross-encoder reranking."""
        return self.search(query, top_k=top_k, rerank=False)
