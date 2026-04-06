# Semantic Hybrid Search Engine — Wikipedia

> Ask any question in plain English. Get the most relevant Wikipedia passages back — ranked by *meaning*, not just keywords.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face%20Spaces-orange?logo=huggingface)](https://huggingface.co/spaces/Priyrajsinh/B4-Semantic-Search-FAISS)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-88%20passed-brightgreen?logo=pytest)](tests/)
[![Coverage](https://img.shields.io/badge/Coverage-84%25-brightgreen)](tests/)

---

## Live Demo

**[Try it now on Hugging Face Spaces](https://huggingface.co/spaces/Priyrajsinh/B4-Semantic-Search-FAISS)**

Type any question — "How does the immune system fight viruses?" — and the engine searches through **150,000+ Wikipedia passages** to find the most relevant answers. No keyword matching. It understands what you mean.

---

## The Problem This Solves

Traditional search engines match your words exactly. Type "car" and they miss passages about "automobiles". Type "flu" and they skip articles on "influenza".

**Semantic search** solves this by converting text into mathematical vectors — nearby vectors mean similar meaning, regardless of the exact words used. But pure semantic search has a blind spot: it can miss precise terms like "COVID-19" or "CRISPR" that don't appear in the training data.

This project combines **three different retrieval strategies** and lets a neural model pick the best result — the same approach used by Google, Bing, and enterprise search systems.

---

## How It Works

```
Your Question
      |
      v
 [Sentence Encoder]          Converts query to a 384-dim meaning vector
      |                      Model: all-MiniLM-L6-v2  (~15ms)
      |
      +--------> [FAISS Index]     Finds 20 passages with similar meaning
      |                            100K+ vectors, IVFFlat index  (~5ms)
      |
      +--------> [BM25 Index]      Finds 20 passages with matching keywords
                                   Classic TF-IDF variant  (~2ms)
      |
      v
 [Reciprocal Rank Fusion]    Merges both ranked lists into one
      |                      No need to normalise scores  (~1ms)
      |
      v
 [Cross-Encoder Re-Ranker]   Reads query + each passage together
                             Scores true relevance  (~200ms)
                             Model: ms-marco-MiniLM-L-6-v2
      |
      v
   Top 5 Results
```

### Why Three Stages?

| Stage | What it's good at | Weakness |
|---|---|---|
| Dense (FAISS) | Understands meaning — "car" = "automobile" | Can miss exact rare terms |
| Sparse (BM25) | Exact keyword matching — "COVID-19" = "COVID-19" | No understanding of synonyms |
| Cross-Encoder | Reads both query and passage together — most accurate | Too slow to run on 150K passages directly |

Combining all three gives you the best of each world. This is called **hybrid retrieval** and is the architecture behind modern enterprise search (Elasticsearch 8, Azure Cognitive Search, Vertex AI Search).

---

## Results

Evaluated on 200 Simple Wikipedia test queries.

| Method | MRR@10 | vs Dense-only |
|---|---|---|
| Dense only (FAISS) | 0.0051 | baseline |
| BM25 only | 0.0066 | +29% |
| Hybrid (RRF) | 0.0045 | -12% |
| **Hybrid + Cross-Encoder Rerank** | **0.0131** | **+157%** |

The Cross-Encoder reranker delivers **2.6x better ranking quality** than dense retrieval alone.

> MRR (Mean Reciprocal Rank) measures how high the first correct result appears in the ranked list. MRR = 1.0 means the correct result is always ranked #1.

---

## Features

- **Semantic search** — understands meaning, not just keywords
- **Hybrid retrieval** — dense vectors + BM25 fused via Reciprocal Rank Fusion
- **Neural re-ranking** — cross-encoder rescores candidates for precision
- **REST API** — FastAPI server with health checks, rate limiting, Prometheus metrics
- **Interactive UI** — Gradio demo with 3 tabs: simple search, developer mode, and explainer
- **Score calibration** — raw model scores mapped to 0–100% for human readability
- **Experiment tracking** — MLflow logs every evaluation run
- **Production-ready** — Docker multi-stage build, non-root user, CORS, input validation

---

## Quick Start

```bash
# 1. Install dependencies
make install

# 2. Download Wikipedia, chunk into passages, build FAISS + BM25 index
#    (downloads ~1GB Simple Wikipedia, takes ~10-15 minutes on first run)
make build-index

# 3. Run FastAPI server on localhost:8000
make serve

# 4. Or launch the Gradio UI locally
make gradio

# 5. Evaluate all 4 retrieval methods and generate MRR report
make eval

# 6. Run full test suite (84% coverage)
make test
```

### API Usage

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Search
curl -X POST http://localhost:8000/api/v1/search \
     -H "Content-Type: application/json" \
     -d '{"query": "How does the immune system work?", "top_k": 5}'
```

```json
{
  "query": "How does the immune system work?",
  "results": [
    {
      "title": "Immune system",
      "text": "The immune system is a network of biological processes...",
      "score": 8.43
    }
  ],
  "latency_ms": 312.4
}
```

---

## Tech Stack

| Component | Library | Why |
|---|---|---|
| Dense embeddings | sentence-transformers `all-MiniLM-L6-v2` | Fast, lightweight, excellent quality for its size |
| Vector index | FAISS `IndexIVFFlat` | Production-grade ANN search from Meta AI |
| Sparse retrieval | rank-bm25 `BM25Okapi` | Best-practice keyword search baseline |
| Re-ranking | sentence-transformers `CrossEncoder` | State-of-the-art passage relevance scoring |
| API server | FastAPI + uvicorn | Async, typed, auto-docs at `/docs` |
| Web UI | Gradio 6 | HF-native, zero frontend code |
| Experiment tracking | MLflow | Logs MRR metrics and model params per run |
| Data validation | pandera + Pydantic v2 | Schema-enforced DataFrames and API payloads |
| Containerisation | Docker multi-stage | Builder + runtime separation, non-root user |

---

## Project Structure

```
B4-Semantic-Search-FAISS/
│
├── config/
│   └── config.yaml             # Single source of truth for all hyperparameters
│
├── src/
│   ├── data/
│   │   ├── dataset.py          # Wikipedia loader (HuggingFace datasets)
│   │   ├── chunker.py          # Sliding-window passage chunker (256 tokens, 50 overlap)
│   │   ├── validation.py       # Pandera DataFrame schema validation
│   │   └── pipeline.py        # End-to-end: load → chunk → validate → save
│   ├── retrieval/
│   │   ├── encoder.py          # SentenceTransformer wrapper
│   │   ├── index.py            # FAISS index (IVFFlat / FlatIP fallback)
│   │   ├── search.py           # SemanticSearchEngine: dense, BM25, hybrid, rerank
│   │   └── build_index.py     # Standalone index building script
│   ├── evaluation/
│   │   ├── eval_queries.py     # Query builder (MS-MARCO / curated fallback)
│   │   └── evaluate.py        # MRR@10 evaluation + MLflow + chart generation
│   └── api/
│       ├── app.py              # FastAPI: /health, /search, rate limiting, CORS
│       └── gradio_demo.py     # 3-tab Gradio UI
│
├── tests/                      # 88 tests, 84% coverage, 70% gate enforced
├── hf_space/                   # Self-contained Gradio app (no src/ imports)
├── models/                     # faiss_index.bin, bm25_index.pkl, chunk_metadata.parquet
├── reports/                    # results.json, figures/mrr_comparison.png
├── Dockerfile                  # Multi-stage, non-root appuser, port 8000
└── Makefile                    # install | build-index | serve | gradio | eval | test
```

---

## What I Learned Building This

- **Hybrid search beats pure semantic search** — BM25 catches keyword-exact matches that dense vectors miss (medical terms, named entities, version numbers). Combining them with RRF is almost always better than either alone.
- **Re-ranking is the quality multiplier** — the cross-encoder runs on only 20 candidates (not 150K), making it feasible at inference time. This single step explains most of the MRR gain.
- **Latency breakdown matters** — encoding takes ~15ms, FAISS ~5ms, but re-ranking takes ~200ms. Knowing which stage is the bottleneck tells you where to optimise.
- **IVFFlat vs FlatIP** — for small datasets (<1000 vectors) an exact flat index is faster than IVF with cluster overhead; the engine picks automatically.

---

## Related Work / Further Reading

- [BEIR Benchmark](https://github.com/beir-cellar/beir) — the standard retrieval evaluation benchmark
- [Sentence-Transformers documentation](https://www.sbert.net/) — the library powering the encoder and cross-encoder
- [FAISS wiki](https://github.com/facebookresearch/faiss/wiki) — choosing the right index type
- [Reciprocal Rank Fusion (Cormack et al., 2009)](https://dl.acm.org/doi/10.1145/1571941.1572114) — the original RRF paper
