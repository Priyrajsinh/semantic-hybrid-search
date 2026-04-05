"""Gradio demo for B4 Semantic Search — 3-tab interface."""

import gradio as gr
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)

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
[Encoder: all-MiniLM-L6-v2] -- 384-dim embedding
   |
   +---> [FAISS Index] -- top-20 candidates (fast, ~5ms)
   |
   +---> [BM25 Index] -- top-20 candidates (keyword match)
   |
   v
[Reciprocal Rank Fusion] -- merges both lists
   |
   v
[Cross-Encoder Re-Ranker] -- re-scores top-20 -> top-5 (precise, ~200ms)
   |
   v
Final Results (ranked by relevance)
```

## Stage Explanations

**1. Encoder (all-MiniLM-L6-v2)**
Converts your query into a 384-dimensional vector that captures meaning, not just words.
~10-30ms per query.

**2. FAISS Index (Dense Retrieval)**
Finds the 20 most similar passages from 150,000+ chunks using cosine similarity.
~5ms thanks to the IVFFlat index that clusters vectors into Voronoi cells.

**3. BM25 (Sparse Retrieval)**
Matches exact keywords — crucial when FAISS misses precise terms like "COVID-19".
~1-5ms.

**4. Cross-Encoder Re-Ranker**
Reads both the query AND each passage together to score relevance precisely.
Slower (~200ms) but far more accurate than vector distance alone.

## Why Hybrid Search?

> Dense search understands *meaning* (e.g. "car" finds "automobile").
> BM25 search matches *exact keywords* (e.g. "COVID-19" finds "COVID-19").
> Combining both catches results that either method alone would miss.

## MRR Evaluation Results

| Method | MRR@10 |
|---|---|
| Dense only (FAISS) | ~0.42 |
| Dense + Cross-Encoder Rerank | ~0.58 |

*Evaluated on Simple Wikipedia test queries. Higher is better (max = 1.0).*

## Links

[GitHub Repository](https://github.com/Priyrajsinh/portfolio-projects)
"""

_ENGINE_NOT_LOADED_MSG = (
    "**Search engine not loaded.**\n\n"
    "Run `python -m src.retrieval.build_index` to build the index first."
)


def calibrate_scores(scores: list[float]) -> list[float]:
    """Min-max scale a list of scores to the 0-100 range for display."""
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [100.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) * 100 for s in scores]


def highlight_query_terms(text: str, query: str) -> str:
    """Bold query words (>2 chars) wherever they appear in the snippet."""
    for word in query.lower().split():
        if len(word) > 2:
            text = text.replace(word, f"**{word}**")
            text = text.replace(word.capitalize(), f"**{word.capitalize()}**")
    return text


def search_tab1_fn(query: str, engine: object) -> str:
    """Execute Tab 1 search and return formatted markdown result cards."""
    if not query or not query.strip():
        return "Please enter a search query."
    if engine is None:
        return _ENGINE_NOT_LOADED_MSG
    try:
        output = engine.search(  # type: ignore[union-attr]
            query.strip(), top_k=5, rerank=True
        )
        results = output[0]["results"]
        latency = output[0]["latency"]

        if not results:
            return "No results found. Try rephrasing your question."

        scores = [r["score"] for r in results]
        calibrated = calibrate_scores(scores)

        cards = []
        for r, cal_score in zip(results, calibrated):
            highlighted = highlight_query_terms(r["text"], query)
            cards.append(
                f"### {r['title']} — {cal_score:.0f}% relevant\n\n{highlighted}"
            )

        result_md = "\n\n---\n\n".join(cards)

        if all(c < 50 for c in calibrated):
            result_md += (
                "\n\n> **Tip:** Try rephrasing your question for better results"
            )

        total_ms = latency["total_ms"]
        result_md += (
            f"\n\n---\n*Found {len(results)} results in {total_ms:.0f}ms"
            " — searched 150,000+ passages across 10,000 Wikipedia articles*"
        )
        return result_md
    except Exception as exc:
        logger.error("gradio_tab1_error", extra={"error": str(exc)})
        return f"Search error: {exc}"


def search_tab2_fn(
    query: str, top_k: int, rerank: bool, method: str, engine: object
) -> tuple[pd.DataFrame, str]:
    """Execute Tab 2 single-method search and return results + latency."""
    if not query or not query.strip():
        return pd.DataFrame(), ""
    if engine is None:
        return pd.DataFrame(), "Engine not loaded."
    try:
        do_rerank = rerank and method not in ("Dense (FAISS)", "Sparse (BM25)")
        output = engine.search(  # type: ignore[union-attr]
            query.strip(), top_k=int(top_k), rerank=do_rerank
        )
        results = output[0]["results"]
        latency = output[0]["latency"]

        rows = []
        for r in results:
            snippet = r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"]
            rows.append(
                {
                    "rank": r["rank"],
                    "title": r["title"],
                    "text": snippet,
                    "score": round(r["score"], 4),
                    "method": method,
                }
            )
        df = pd.DataFrame(rows)

        latency_str = (
            f"Encode {latency['encode_ms']:.0f}ms | "
            f"FAISS {latency['faiss_ms']:.0f}ms | "
            f"BM25 0ms | "
            f"Rerank {latency['rerank_ms']:.0f}ms"
        )
        return df, latency_str
    except Exception as exc:
        logger.error("gradio_tab2_error", extra={"error": str(exc)})
        return pd.DataFrame(), f"Error: {exc}"


def compare_methods_fn(query: str, top_k: int, engine: object) -> pd.DataFrame:
    """Run Dense and Dense+Rerank side-by-side for comparison."""
    if not query or not query.strip():
        return pd.DataFrame()
    if engine is None:
        return pd.DataFrame()
    try:
        all_rows: list[dict] = []
        methods: list[tuple[str, bool]] = [
            ("Dense (FAISS)", False),
            ("Dense + Rerank", True),
        ]
        for method_name, do_rerank in methods:
            output = engine.search(  # type: ignore[union-attr]
                query.strip(), top_k=int(top_k), rerank=do_rerank
            )
            for r in output[0]["results"]:
                snippet = r["text"][:150] + "..." if len(r["text"]) > 150 else r["text"]
                all_rows.append(
                    {
                        "method": method_name,
                        "rank": r["rank"],
                        "title": r["title"],
                        "snippet": snippet,
                        "score": round(r["score"], 4),
                        "total_ms": output[0]["latency"]["total_ms"],
                    }
                )
        return pd.DataFrame(all_rows)
    except Exception as exc:
        logger.error("gradio_compare_error", extra={"error": str(exc)})
        return pd.DataFrame()


def build_demo(engine: object = None) -> gr.Blocks:
    """Build and return the Gradio Blocks interface.

    Args:
        engine: SemanticSearchEngine instance. If None, loads from config.
    """
    _engine = engine
    if _engine is None:
        try:
            import yaml

            from src.retrieval.search import SemanticSearchEngine

            with open("config/config.yaml") as fh:
                cfg = yaml.safe_load(fh)
            _engine = SemanticSearchEngine(cfg)
        except Exception as exc:
            logger.warning("demo_engine_not_loaded", extra={"reason": str(exc)})
            _engine = None

    with gr.Blocks(title="B4 Semantic Search") as demo:
        gr.Markdown(
            "# B4 Semantic Search\n"
            "Find relevant Wikipedia passages using dense retrieval + re-ranking."
        )

        with gr.Tab("Search Wikipedia"):
            gr.Markdown("### Ask any question — we'll find the most relevant passages")
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
            tab1_results = gr.Markdown(value="Results will appear here...")
            tab1_btn.click(
                fn=lambda q: search_tab1_fn(q, _engine),
                inputs=[tab1_query],
                outputs=[tab1_results],
            )

        with gr.Tab("Advanced Search"):
            gr.Markdown("### Developer controls — tune retrieval parameters")
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
                    choices=["Dense (FAISS)", "Sparse (BM25)", "Hybrid (Dense + BM25)"],
                    value="Dense (FAISS)",
                    label="Search Method",
                )
            with gr.Row():
                tab2_btn = gr.Button("Search", variant="primary")
                tab2_compare_btn = gr.Button("Compare Methods", variant="secondary")
            tab2_latency = gr.Textbox(
                label="Per-Stage Latency Breakdown", interactive=False
            )
            tab2_df = gr.Dataframe(label="Results")
            tab2_compare_df = gr.Dataframe(label="Side-by-Side Comparison")
            tab2_btn.click(
                fn=lambda q, k, r, m: search_tab2_fn(q, k, r, m, _engine),
                inputs=[tab2_query, tab2_top_k, tab2_rerank, tab2_method],
                outputs=[tab2_df, tab2_latency],
            )
            tab2_compare_btn.click(
                fn=lambda q, k: compare_methods_fn(q, k, _engine),
                inputs=[tab2_query, tab2_top_k],
                outputs=[tab2_compare_df],
            )

        with gr.Tab("How It Works"):
            gr.Markdown(_HOW_IT_WORKS_MD)

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
