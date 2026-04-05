"""Tests for FAISSIndex."""

import numpy as np
import pytest

from src.retrieval.index import FAISSIndex


def _random_vecs(n: int, dim: int = 384) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((n, dim)).astype(np.float32)


def test_build_and_search():
    vecs = _random_vecs(10)
    idx = FAISSIndex(embedding_dim=384, fallback_threshold=1000)
    idx.build(vecs)
    query = _random_vecs(1)
    scores, indices = idx.search(query, top_k=3)
    assert len(indices) == 3
    assert all(0 <= i < 10 for i in indices)


def test_save_load_roundtrip(tmp_path):
    vecs = _random_vecs(10)
    idx = FAISSIndex(embedding_dim=384, fallback_threshold=1000)
    idx.build(vecs)

    path = str(tmp_path / "faiss_index.bin")
    idx.save(path)

    idx2 = FAISSIndex(embedding_dim=384)
    idx2.load(path)
    assert idx2.n_vectors == 10


def test_n_vectors_property():
    vecs = _random_vecs(7)
    idx = FAISSIndex(embedding_dim=384, fallback_threshold=1000)
    idx.build(vecs)
    assert idx.n_vectors == 7


def test_ivf_used_above_threshold():
    """IndexIVFFlat is selected when n_vectors >= fallback_threshold."""
    import faiss

    vecs = _random_vecs(1100)
    idx = FAISSIndex(embedding_dim=384, nlist=10, fallback_threshold=1000)
    idx.build(vecs)
    assert isinstance(idx.index, faiss.IndexIVFFlat)


def test_flatip_used_below_threshold():
    """IndexFlatIP is selected when n_vectors < fallback_threshold."""
    import faiss

    vecs = _random_vecs(10)
    idx = FAISSIndex(embedding_dim=384, fallback_threshold=1000)
    idx.build(vecs)
    assert isinstance(idx.index, faiss.IndexFlatIP)


def test_search_raises_before_build():
    idx = FAISSIndex()
    query = _random_vecs(1)
    with pytest.raises(Exception):
        idx.search(query, top_k=3)
