"""Tests for SemanticEncoder."""

import numpy as np

from src.retrieval.encoder import SemanticEncoder

_encoder = SemanticEncoder()


def test_encode_shape():
    texts = ["hello world", "foo bar baz", "semantic search is great"]
    embeddings = _encoder.encode(texts, show_progress=False)
    assert embeddings.shape == (3, 384)


def test_encode_query_shape():
    embedding = _encoder.encode_query("what is machine learning?")
    assert embedding.shape == (1, 384)


def test_embeddings_normalized():
    texts = ["alpha", "beta", "gamma"]
    embeddings = _encoder.encode(texts, show_progress=False)
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
