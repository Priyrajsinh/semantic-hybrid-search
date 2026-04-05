"""Article chunking utilities."""

import pandas as pd


def chunk_articles(articles: list[dict]) -> pd.DataFrame:
    """Split articles into overlapping text chunks.

    Args:
        articles: List of dicts with 'id', 'title', 'text' keys.

    Returns:
        DataFrame with columns: chunk_id, article_id, title, text.
    """
    pass
