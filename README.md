# B4 — Semantic Search with Sentence-Transformers + FAISS

## Live Demo

[Link to HF Space — add after deploying]

## What This Does

This project builds a production-ready semantic search engine over 10,000 Simple
Wikipedia articles (150,000+ passages). You type a question in plain English and
the engine finds the most relevant passages — understanding *meaning*, not just
keywords. It combines dense vector search (FAISS), sparse keyword search (BM25),
and a neural re-ranker (Cross-Encoder) into a hybrid pipeline evaluated with MRR.

## Architecture

```
Query
  |
  v
[Encoder: all-MiniLM-L6-v2]  →  384-dim embedding
  |
  +-----> [FAISS Index]  →  top-20 by cosine similarity   (~5 ms)
  |
  +-----> [BM25 Index]   →  top-20 by keyword match       (~2 ms)
  |
  v
[Reciprocal Rank Fusion]  →  merged candidate list
  |
  v
[Cross-Encoder Re-Ranker]  →  re-scores top-20 → top-5  (~200 ms)
  |
  v
Final Results
```

## Results

Evaluated on 200 Simple Wikipedia test queries (MS-MARCO fallback).

| Method | MRR@10 |
|---|---|
| Dense only (FAISS) | 0.0051 |
| BM25 only | 0.0066 |
| Hybrid (RRF) | 0.0045 |
| Hybrid + Cross-Encoder Rerank | 0.0131 |

Hybrid + Rerank achieves **2.6x higher MRR** than dense-only retrieval.

## Quick Start

```bash
# Install dependencies
make install

# Download Wikipedia, chunk into passages, build FAISS + BM25 index
make build-index

# Run FastAPI server on :8000
make serve

# Or launch Gradio demo
make gradio

# Evaluate all 4 retrieval methods (MRR@10)
make eval

# Run all tests (70% coverage gate)
make test
```

## Tech Stack

| Component | Library |
|---|---|
| Dense embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector index | FAISS (IndexIVFFlat) |
| Sparse retrieval | rank-bm25 (BM25Okapi) |
| Re-ranking | sentence-transformers (CrossEncoder) |
| API server | FastAPI + uvicorn |
| Web UI | Gradio |
| Experiment tracking | MLflow |
| Data validation | pandera |
| Schema validation | Pydantic v2 |
| Containerisation | Docker (multi-stage) |

## Project Structure

```
B4-Semantic-Search-FAISS/
├── config/config.yaml          # Single source of truth for all hyperparameters
├── src/
│   ├── data/                   # Wikipedia loading, chunking, validation
│   ├── retrieval/              # Encoder, FAISS index, search engine
│   ├── evaluation/             # MRR@10 evaluation pipeline
│   └── api/                    # FastAPI app + Gradio demo
├── tests/                      # Full test suite (70% coverage gate)
├── hf_space/                   # Self-contained HF Space (no src/ imports)
├── models/                     # Persisted FAISS index + BM25 + metadata
├── reports/                    # results.json + MRR comparison chart
├── Dockerfile                  # Multi-stage, non-root, port 8000
└── Makefile                    # install | build-index | serve | eval | test
```

## Deployment (HF Space)

Copy built artifacts into `hf_space/` before pushing:

```bash
cp models/faiss_index.bin      hf_space/
cp models/chunk_metadata.parquet hf_space/
cp models/bm25_index.pkl       hf_space/
```

Then push `hf_space/` to a Hugging Face Space with Gradio SDK.
