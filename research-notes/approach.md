# Research Notes — B4 Semantic Search + FAISS

## Papers I Read Before Starting
- Sentence-BERT (Reimers & Gurevych, 2019) — sentence embeddings via siamese BERT
- FAISS (Johnson et al., 2019, Facebook) — billion-scale similarity search
- BM25 (Robertson et al., 1995) — term frequency-inverse document frequency ranking

## Architecture Decisions
- all-MiniLM-L6-v2: 384-dim, fast, same encoder as P1/P6 Qdrant stores
- FAISS IndexIVFFlat (nlist=100): scalable to 1M+ docs, with IndexFlatIP fallback for small corpora
- 256-token chunks with 50-token overlap: matches P1/P6 chunking strategy
- Cross-encoder re-ranking: two-stage retrieval (fast recall + accurate precision)
- Hybrid search: dense + BM25 + RRF covers both semantic and lexical matches

## Surprising Findings
- [Fill in after building]
