"""FAISS index wrapper with IVFFlat / FlatIP fallback."""

import faiss
import numpy as np

from src.logger import get_logger

_logger = get_logger(__name__)


class FAISSIndex:
    """Manages FAISS index: build, save, load, and query."""

    def __init__(
        self,
        embedding_dim: int = 384,
        nlist: int = 100,
        fallback_threshold: int = 1000,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.nlist = nlist
        self.fallback_threshold = fallback_threshold
        self.index = None

    def build(self, embeddings: np.ndarray, nprobe: int = 10) -> None:
        """Build FAISS index from embeddings (normalised to unit L2)."""
        vecs = embeddings.astype(np.float32)
        faiss.normalize_L2(vecs)
        n = vecs.shape[0]

        if n >= self.fallback_threshold:
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.embedding_dim,
                self.nlist,
                faiss.METRIC_INNER_PRODUCT,
            )
            self.index.train(vecs)
            self.index.nprobe = nprobe
            self.index.add(vecs)
            index_type = "IndexIVFFlat"
        else:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(vecs)
            index_type = "IndexFlatIP"

        _logger.info("Built FAISS index (%s) with %d vectors", index_type, n)

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, indices) for the top-k nearest neighbours."""
        qvec = query_embedding.astype(np.float32)
        faiss.normalize_L2(qvec)
        scores, indices = self.index.search(qvec, top_k)
        return scores[0], indices[0]

    def save(self, path: str) -> None:
        """Persist index to disk."""
        faiss.write_index(self.index, path)

    def load(self, path: str) -> None:
        """Load index from disk."""
        self.index = faiss.read_index(path)

    @property
    def n_vectors(self) -> int:
        """Number of vectors stored in the index."""
        return self.index.ntotal
