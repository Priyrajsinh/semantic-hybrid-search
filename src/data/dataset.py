"""Wikipedia dataset loading utilities."""


def load_wikipedia(config: dict) -> list[dict]:
    """Load Wikipedia articles from HuggingFace datasets.

    Args:
        config: Data config dict (dataset_name, dataset_config, n_articles, seed).

    Returns:
        List of dicts with 'id', 'title', 'text' keys.
    """
    pass
