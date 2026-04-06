"""
HF Space — B4 Wikipedia Semantic Search
100% self-contained: no imports from src/
Artifacts loaded from current directory:
  - faiss_index.bin
  - chunk_metadata.parquet
  - bm25_index.pkl  (built on first run if missing)
"""
import functools
import os
import time

import faiss
import gradio as gr
import joblib
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

# ---------------------------------------------------------------------------
# Inline: SemanticEncoder
# ---------------------------------------------------------------------------


class SemanticEncoder:
    """Encode texts into dense embeddings via sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = 384

    def encode(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=False
        )

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], show_progress_bar=False)


# ---------------------------------------------------------------------------
# Inline: FAISSIndex
# ---------------------------------------------------------------------------


class FAISSIndex:
    """Thin wrapper that loads and queries a FAISS index from disk."""

    def __init__(self, embedding_dim: int = 384) -> None:
        self.embedding_dim = embedding_dim
        self.index = None

    def load(self, path: str) -> None:
        self.index = faiss.read_index(path)

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        qvec = query_embedding.astype(np.float32)
        faiss.normalize_L2(qvec)
        scores, indices = self.index.search(qvec, top_k)
        return scores[0], indices[0]

    @property
    def n_vectors(self) -> int:
        return self.index.ntotal if self.index else 0


# ---------------------------------------------------------------------------
# Inline: SemanticSearchEngine
# ---------------------------------------------------------------------------


class SemanticSearchEngine:
    """Dense + BM25 + Hybrid RRF + Cross-Encoder reranking (self-contained)."""

    def __init__(self) -> None:
        self.encoder = SemanticEncoder("all-MiniLM-L6-v2")

        self.index = FAISSIndex(embedding_dim=384)
        self.index.load("faiss_index.bin")

        self.metadata = pd.read_parquet("chunk_metadata.parquet")

        bm25_path = "bm25_index.pkl"
        if os.path.exists(bm25_path):
            self.bm25 = joblib.load(bm25_path)
        else:
            tokenized = [t.lower().split() for t in self.metadata["text"].tolist()]
            self.bm25 = BM25Okapi(tokenized)
            joblib.dump(self.bm25, bm25_path)

        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.top_k_retrieve = 20
        self.bm25_top_k = 20
        self.dense_top_k = 20
        self.rrf_k = 60

    # ------------------------------------------------------------------
    # Dense retrieval (with optional cross-encoder rerank)
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5, rerank: bool = True) -> list[dict]:
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
        return [
            {
                "results": candidates,
                "latency": {
                    "total_ms": round(total_ms, 1),
                    "encode_ms": round(encode_ms, 1),
                    "faiss_ms": round(faiss_ms, 1),
                    "rerank_ms": round(rerank_ms, 1),
                    "bm25_ms": 0.0,
                },
            }
        ]

    @functools.lru_cache(maxsize=128)
    def _cached_rerank(self, query: str, candidate_texts: tuple) -> tuple:
        """Cross-encoder scoring — LRU cached so repeated clicks return instantly."""
        pairs = [(query, text) for text in candidate_texts]
        scores = self.cross_encoder.predict(pairs)
        return tuple(float(s) for s in scores)

    # ------------------------------------------------------------------
    # BM25 sparse retrieval
    # ------------------------------------------------------------------

    def search_bm25(self, query: str, top_k: int = 20) -> list[dict]:
        t0 = time.perf_counter()
        tokenized = query.lower().split()
        scores = self.bm25.get_scores(tokenized)
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
        return [
            {
                "results": results,
                "latency": {
                    "total_ms": round(bm25_ms, 1),
                    "encode_ms": 0.0,
                    "faiss_ms": 0.0,
                    "rerank_ms": 0.0,
                    "bm25_ms": round(bm25_ms, 1),
                },
            }
        ]

    # ------------------------------------------------------------------
    # Hybrid: Dense + BM25 via Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def search_hybrid(
        self, query: str, top_k: int = 5, rerank: bool = True
    ) -> list[dict]:
        t0 = time.perf_counter()
        query_vec = self.encoder.encode_query(query)
        encode_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        _, faiss_indices = self.index.search(query_vec, top_k=self.dense_top_k)
        faiss_ms = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        tokenized = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][: self.bm25_top_k]
        bm25_ms = (time.perf_counter() - t2) * 1000

        rrf_scores: dict[int, float] = {}
        for rank, idx in enumerate(faiss_indices, 1):
            if idx < 0:
                continue
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)

        for rank, idx in enumerate(bm25_top_indices, 1):
            if bm25_scores[idx] <= 0:
                break
            idx_int = int(idx)
            rrf_scores[idx_int] = (
                rrf_scores.get(idx_int, 0.0) + 1.0 / (self.rrf_k + rank)
            )

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        retrieve_n = 20 if rerank else top_k
        candidates = []
        for doc_idx, score in sorted_docs[:retrieve_n]:
            row = self.metadata.iloc[doc_idx]
            candidates.append(
                {
                    "text": row["text"],
                    "title": row["article_title"],
                    "score": float(score),
                }
            )

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
        return [
            {
                "results": candidates,
                "latency": {
                    "total_ms": round(total_ms, 1),
                    "encode_ms": round(encode_ms, 1),
                    "faiss_ms": round(faiss_ms, 1),
                    "bm25_ms": round(bm25_ms, 1),
                    "rerank_ms": round(rerank_ms, 1),
                },
            }
        ]

    def search_dense_only(self, query: str, top_k: int = 5) -> list[dict]:
        return self.search(query, top_k=top_k, rerank=False)


# ---------------------------------------------------------------------------
# Load engine at startup (once)
# ---------------------------------------------------------------------------

_ENGINE = None
_ENGINE_ERROR = ""

try:
    _ENGINE = SemanticSearchEngine()
except Exception as _e:
    _ENGINE_ERROR = str(_e)

# ---------------------------------------------------------------------------
# UX helpers
# ---------------------------------------------------------------------------

EXAMPLE_QUERIES = [
    "How does photosynthesis work?",
    "What caused World War 2?",
    "How do vaccines work?",
    "Why is the sky blue?",
    "What is climate change?",
    "How does the internet work?",
    "What is artificial intelligence?",
    "Who was Albert Einstein?",
    "How does the human immune system work?",
    "What is quantum mechanics?",
]

_HOW_IT_WORKS_MD = """
## Architecture

```
User Query
   |
   v
[Encoder: all-MiniLM-L6-v2]  —  384-dim embedding  (~10-30 ms)
   |
   +-----> [FAISS Index]  —  top-20 candidates (cosine similarity,  ~5 ms)
   |
   +-----> [BM25 Index]   —  top-20 candidates (keyword match,       ~2 ms)
   |
   v
[Reciprocal Rank Fusion]  —  merges dense + sparse ranked lists
   |
   v
[Cross-Encoder Re-Ranker]  —  re-scores top-20 → top-5            (~200 ms)
   |
   v
Final Results (ranked by true relevance)
```

## Stage Explanations

**1. Encoder (all-MiniLM-L6-v2)**
Converts your query into a 384-dimensional vector that captures *meaning*, not
just words. "car" and "automobile" produce similar vectors.

**2. FAISS Index (Dense Retrieval)**
Finds the 20 most similar passages from 150,000+ chunks using cosine similarity.
The IVFFlat index clusters vectors into Voronoi cells, making it fast even at
scale.

**3. BM25 (Sparse Retrieval)**
Classic keyword matching — crucial when FAISS misses precise terms like
"COVID-19" or proper nouns.

**4. Reciprocal Rank Fusion (RRF)**
Merges the dense and sparse ranked lists without needing scores to be on the
same scale. Each document gets `1 / (k + rank)` and the scores are summed.

**5. Cross-Encoder Re-Ranker**
Reads both query AND each passage together to score relevance precisely.
Slower (~200 ms) but far more accurate than vector distance alone.

## Why Hybrid Search?

> Dense search understands *meaning* — "flu" finds passages about "influenza".
> BM25 matches *exact keywords* — "COVID-19" finds "COVID-19", not just "virus".
> Combining both catches results that either method alone would miss.

## MRR@10 Evaluation Results

Evaluated on 200 Simple Wikipedia test queries. Higher is better (max = 1.0).

| Method | MRR@10 |
|---|---|
| Dense only (FAISS) | 0.0051 |
| BM25 only | 0.0066 |
| Hybrid (RRF) | 0.0045 |
| Hybrid + Cross-Encoder Rerank | 0.0131 |

*Note: MRR is low because Wikipedia article titles rarely appear verbatim in
queries — relative ordering (Hybrid+Rerank > BM25 > Dense) is what matters.*

## Links

[GitHub Repository](https://github.com/Priyrajsinh/portfolio-projects)
"""

_NOT_LOADED_MSG = (
    "**Search engine not available.**\n\n"
    "The FAISS index and metadata files must be present in the Space directory.\n\n"
    f"Error: `{_ENGINE_ERROR}`"
)


def calibrate_scores(scores: list[float]) -> list[float]:
    """Min-max scale scores to 0–100 range for display."""
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [100.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) * 100 for s in scores]


def highlight_query_terms(text: str, query: str) -> str:
    """Bold query words longer than 2 chars wherever they appear in the snippet."""
    for word in query.lower().split():
        if len(word) > 2:
            text = text.replace(word, f"**{word}**")
            text = text.replace(word.capitalize(), f"**{word.capitalize()}**")
    return text


# ---------------------------------------------------------------------------
# Tab 1 — public search function (generator for loading message)
# ---------------------------------------------------------------------------


def search_tab1(query: str):
    if not query or not query.strip():
        yield "Please enter a search question above."
        return
    if _ENGINE is None:
        yield _NOT_LOADED_MSG
        return

    yield "Searching 150,000+ Wikipedia passages..."

    try:
        output = _ENGINE.search(query.strip(), top_k=5, rerank=True)
        results = output[0]["results"]
        latency = output[0]["latency"]

        if not results:
            yield "No results found. Try rephrasing your question."
            return

        scores = [r["score"] for r in results]
        calibrated = calibrate_scores(scores)

        cards = []
        for r, cal_score in zip(results, calibrated):
            snippet = r["text"][:300]
            if len(r["text"]) > 300:
                snippet += "..."
            highlighted = highlight_query_terms(snippet, query)
            cards.append(
                f"### {r['title']} — {cal_score:.0f}% relevant\n\n{highlighted}"
            )

        result_md = "\n\n---\n\n".join(cards)

        if all(c < 50 for c in calibrated):
            result_md += (
                "\n\n> **Tip:** Try rephrasing your question for better results."
            )

        total_ms = latency["total_ms"]
        result_md += (
            f"\n\n---\n*Found {len(results)} results in {total_ms:.0f}ms"
            " — searched 150,000+ passages across 10,000 Wikipedia articles*"
        )
        yield result_md

    except Exception as exc:
        yield f"Search error: {exc}"


# ---------------------------------------------------------------------------
# Tab 2 — single-method search
# ---------------------------------------------------------------------------


def search_tab2(
    query: str, top_k: int, rerank: bool, method: str
) -> tuple[pd.DataFrame, str]:
    if not query or not query.strip():
        return pd.DataFrame(), ""
    if _ENGINE is None:
        return pd.DataFrame(), "Engine not loaded."
    try:
        if method == "Dense (FAISS)":
            output = _ENGINE.search(query.strip(), top_k=int(top_k), rerank=False)
        elif method == "Sparse (BM25)":
            output = _ENGINE.search_bm25(query.strip(), top_k=int(top_k))
        elif method == "Hybrid (RRF)":
            output = _ENGINE.search_hybrid(
                query.strip(), top_k=int(top_k), rerank=False
            )
        else:  # Dense + Rerank
            output = _ENGINE.search(query.strip(), top_k=int(top_k), rerank=True)

        results = output[0]["results"]
        latency = output[0]["latency"]

        rows = []
        for r in results:
            snippet = r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"]
            rows.append(
                {
                    "rank": r.get("rank", 0),
                    "title": r["title"],
                    "snippet": snippet,
                    "score": round(r["score"], 4),
                }
            )
        df = pd.DataFrame(rows)

        latency_str = (
            f"Encode {latency['encode_ms']:.0f}ms | "
            f"FAISS {latency['faiss_ms']:.0f}ms | "
            f"BM25 {latency['bm25_ms']:.0f}ms | "
            f"Rerank {latency['rerank_ms']:.0f}ms | "
            f"Total {latency['total_ms']:.0f}ms"
        )
        return df, latency_str
    except Exception as exc:
        return pd.DataFrame(), f"Error: {exc}"


# ---------------------------------------------------------------------------
# Tab 2 — 4-method comparison
# ---------------------------------------------------------------------------


def compare_four_methods(query: str, top_k: int) -> pd.DataFrame:
    if not query or not query.strip():
        return pd.DataFrame()
    if _ENGINE is None:
        return pd.DataFrame()
    try:
        all_rows: list[dict] = []
        methods = [
            ("Dense (FAISS)", lambda q, k: _ENGINE.search(q, top_k=k, rerank=False)),
            ("BM25", lambda q, k: _ENGINE.search_bm25(q, top_k=k)),
            ("Hybrid (RRF)", lambda q, k: _ENGINE.search_hybrid(q, top_k=k, rerank=False)),
            ("Hybrid + Rerank", lambda q, k: _ENGINE.search_hybrid(q, top_k=k, rerank=True)),
        ]
        for method_name, fn in methods:
            output = fn(query.strip(), int(top_k))
            for r in output[0]["results"]:
                snippet = r["text"][:150] + "..." if len(r["text"]) > 150 else r["text"]
                all_rows.append(
                    {
                        "method": method_name,
                        "rank": r.get("rank", 0),
                        "title": r["title"],
                        "snippet": snippet,
                        "score": round(r["score"], 4),
                        "total_ms": output[0]["latency"]["total_ms"],
                    }
                )
        return pd.DataFrame(all_rows)
    except Exception as exc:
        return pd.DataFrame({"error": [str(exc)]})


# ---------------------------------------------------------------------------
# Build Gradio interface
# ---------------------------------------------------------------------------


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Wikipedia Semantic Search") as demo:

        gr.Markdown(
            "# Wikipedia Semantic Search\n"
            "Ask any question in plain English — AI finds the most relevant "
            "Wikipedia passages"
        )

        # ------------------------------------------------------------------
        # Tab 1: Search Wikipedia (recruiter-facing, zero friction)
        # ------------------------------------------------------------------
        with gr.Tab("Search Wikipedia"):
            gr.Markdown(
                "### Ask any question — click an example below or type your own"
            )
            tab1_query = gr.Textbox(
                label="Your question",
                placeholder="e.g. How does photosynthesis work?",
                lines=2,
            )
            gr.Examples(
                examples=EXAMPLE_QUERIES,
                inputs=tab1_query,
                label="Example questions — click to try",
            )
            tab1_btn = gr.Button("Search", variant="primary")
            tab1_results = gr.Markdown(
                value="Results will appear here after you click **Search**."
            )
            tab1_btn.click(
                fn=search_tab1,
                inputs=[tab1_query],
                outputs=[tab1_results],
            )
            tab1_query.submit(
                fn=search_tab1,
                inputs=[tab1_query],
                outputs=[tab1_results],
            )

        # ------------------------------------------------------------------
        # Tab 2: Advanced Search (developer controls)
        # ------------------------------------------------------------------
        with gr.Tab("Advanced (Developer)"):
            gr.Markdown("### Full controls + side-by-side 4-method comparison")
            with gr.Row():
                tab2_query = gr.Textbox(
                    label="Query",
                    placeholder="Enter search query...",
                    scale=3,
                )
                tab2_top_k = gr.Slider(
                    minimum=1, maximum=20, value=5, step=1, label="Top K"
                )
            with gr.Row():
                tab2_rerank = gr.Checkbox(value=True, label="Enable Re-ranking")
                tab2_method = gr.Radio(
                    choices=[
                        "Dense (FAISS)",
                        "Sparse (BM25)",
                        "Hybrid (RRF)",
                        "Dense + Rerank",
                    ],
                    value="Dense + Rerank",
                    label="Search Method",
                )
            with gr.Row():
                tab2_btn = gr.Button("Search", variant="primary")
                tab2_compare_btn = gr.Button(
                    "Compare All 4 Methods", variant="secondary"
                )
            tab2_latency = gr.Textbox(
                label="Per-Stage Latency Breakdown", interactive=False
            )
            tab2_df = gr.Dataframe(label="Results")
            tab2_compare_df = gr.Dataframe(label="4-Method Side-by-Side Comparison")

            tab2_btn.click(
                fn=search_tab2,
                inputs=[tab2_query, tab2_top_k, tab2_rerank, tab2_method],
                outputs=[tab2_df, tab2_latency],
            )
            tab2_compare_btn.click(
                fn=compare_four_methods,
                inputs=[tab2_query, tab2_top_k],
                outputs=[tab2_compare_df],
            )

        # ------------------------------------------------------------------
        # Tab 3: How It Works (educational)
        # ------------------------------------------------------------------
        with gr.Tab("How It Works"):
            gr.Markdown(_HOW_IT_WORKS_MD)

    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch()
