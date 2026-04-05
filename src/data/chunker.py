"""Article chunking utilities."""

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


def chunk_articles(
    articles: list[dict],
    chunk_size: int = 256,
    chunk_overlap: int = 50,
) -> pd.DataFrame:
    """Split articles into overlapping text chunks via whitespace tokenization.

    Args:
        articles: List of dicts with 'title' and 'text' keys.
        chunk_size: Max tokens (whitespace-split words) per chunk.
        chunk_overlap: Tokens to overlap between consecutive chunks.

    Returns:
        DataFrame with columns: article_title, chunk_index, text.
    """
    rows = []
    step = chunk_size - chunk_overlap
    for article in articles:
        text = article.get("text", "")
        if not text or not text.strip():
            continue
        title = article["title"]
        tokens = text.split()
        idx = 0
        chunk_index = 0
        while idx < len(tokens):
            chunk_text = " ".join(tokens[idx : idx + chunk_size])
            rows.append(
                {
                    "article_title": title,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                }
            )
            chunk_index += 1
            idx += step
    df = pd.DataFrame(rows, columns=["article_title", "chunk_index", "text"])
    logger.info("Chunked %d articles into %d passages", len(articles), len(df))
    return df
