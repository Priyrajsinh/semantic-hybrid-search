"""Sentence-Transformer encoder wrapper."""

import numpy as np
from sentence_transformers import SentenceTransformer

from src.logger import get_logger


class SemanticEncoder:
    """Encodes text into dense embeddings using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._logger = get_logger(__name__)
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = 384

    def encode(
        self,
        texts: list[str],
        batch_size: int = 256,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode a list of texts into embeddings of shape (n, 384)."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        )
        self._logger.info("Encoded %d texts -> shape %s", len(texts), embeddings.shape)
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query and return shape (1, 384)."""
        embedding = self.model.encode([query], show_progress_bar=False)
        return embedding
