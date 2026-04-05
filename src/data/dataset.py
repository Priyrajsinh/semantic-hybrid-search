"""Wikipedia dataset loading utilities."""

from datasets import load_dataset

from src.logger import get_logger

logger = get_logger(__name__)


def load_wikipedia(config: dict) -> list[dict]:
    """Load Wikipedia Simple English articles from HuggingFace datasets.

    Args:
        config: Project config dict. Uses config["data"] keys:
            dataset_name, dataset_config, n_articles.

    Returns:
        List of dicts with 'title' and 'text' keys.
    """
    data_cfg = config["data"]
    ds = load_dataset(  # nosec B615
        data_cfg["dataset_name"],
        data_cfg["dataset_config"],
        split="train",
        trust_remote_code=True,
    )
    n = data_cfg["n_articles"]
    ds = ds.select(range(min(n, len(ds))))
    articles = [{"title": row["title"], "text": row["text"]} for row in ds]
    logger.info("Loaded %d articles from Wikipedia Simple English", len(articles))
    return articles
