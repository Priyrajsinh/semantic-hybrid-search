"""Build evaluation query set from MS-MARCO or curated fallback."""

import json
from pathlib import Path

from src.logger import get_logger

logger = get_logger(__name__)

EVAL_QUERIES_PATH = Path("data/processed/eval_queries.json")
MIN_QUERIES = 50


def build_eval_queries(
    articles: list[dict],
    output_path: Path = EVAL_QUERIES_PATH,
    max_queries: int = 200,
) -> list[dict]:
    """Build evaluation set from MS-MARCO filtered to our Wikipedia articles.

    Filters MS-MARCO train split to questions whose passage text overlaps
    with titles or content in our Simple English Wikipedia collection.
    Falls back to a curated set if filtering yields fewer than MIN_QUERIES.

    Args:
        articles: Loaded Wikipedia articles (list of title/text dicts).
        output_path: Destination path for eval_queries.json.
        max_queries: Maximum number of query-answer pairs to retain.

    Returns:
        List of dicts: [{"query": str, "relevant_titles": [str]}]
    """
    try:
        pairs = _filter_msmarco(articles, max_queries)
        if len(pairs) < MIN_QUERIES:
            logger.warning(
                "MS-MARCO filtering yielded %d matches (< %d). "
                "Falling back to curated queries.",
                len(pairs),
                MIN_QUERIES,
            )
            pairs = _curated_queries(articles)
    except Exception as exc:
        logger.warning("MS-MARCO load failed (%s). Using curated queries.", exc)
        pairs = _curated_queries(articles)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w") as fh:
        json.dump(pairs, fh, indent=2)
    logger.info("Saved %d eval queries to %s", len(pairs), output_path)
    return pairs


def _filter_msmarco(articles: list[dict], max_queries: int) -> list[dict]:
    """Load MS-MARCO v1.1 and filter to articles in our collection."""
    from datasets import load_dataset

    ds = load_dataset(  # nosec B615
        "ms_marco", "v1.1", split="train", trust_remote_code=True
    )
    pairs: list[dict] = []
    for row in ds:
        if len(pairs) >= max_queries:
            break
        query = row["query"]
        passages = row.get("passages", {})
        passage_texts = passages.get("passage_text", [])
        relevant = [
            a["title"]
            for a in articles
            if any(a["title"].lower() in p.lower() for p in passage_texts)
        ]
        if relevant:
            pairs.append({"query": query, "relevant_titles": relevant[:3]})
    return pairs


def _curated_queries(articles: list[dict]) -> list[dict]:
    """Return up to 50 curated queries matched against article titles."""
    candidates = [
        ("What is the capital of France?", "Paris"),
        ("Who invented the telephone?", "Alexander Graham Bell"),
        ("What is photosynthesis?", "Photosynthesis"),
        ("What is the speed of light?", "Speed of light"),
        ("Who was Albert Einstein?", "Albert Einstein"),
        ("What is DNA?", "DNA"),
        ("What is the Internet?", "Internet"),
        ("Who was William Shakespeare?", "William Shakespeare"),
        ("What is gravity?", "Gravity"),
        ("What is democracy?", "Democracy"),
        ("What is the solar system?", "Solar System"),
        ("Who was Isaac Newton?", "Isaac Newton"),
        ("What is climate change?", "Climate change"),
        ("What is the water cycle?", "Water cycle"),
        ("Who was Napoleon Bonaparte?", "Napoleon"),
        ("What is the United Nations?", "United Nations"),
        ("What is evolution?", "Evolution"),
        ("Who was Marie Curie?", "Marie Curie"),
        ("What is the human brain?", "Brain"),
        ("What is electricity?", "Electricity"),
        ("What is the Amazon River?", "Amazon River"),
        ("Who was Leonardo da Vinci?", "Leonardo da Vinci"),
        ("What is the periodic table?", "Periodic table"),
        ("What is a volcano?", "Volcano"),
        ("What is the Roman Empire?", "Roman Empire"),
        ("Who was Mahatma Gandhi?", "Mahatma Gandhi"),
        ("What is World War II?", "World War II"),
        ("What is the theory of relativity?", "Theory of relativity"),
        ("What is a black hole?", "Black hole"),
        ("What is the United States Constitution?", "United States Constitution"),
        ("Who was Charles Darwin?", "Charles Darwin"),
        ("What is oxygen?", "Oxygen"),
        ("What is the Mediterranean Sea?", "Mediterranean Sea"),
        ("Who was Cleopatra?", "Cleopatra"),
        ("What is the Renaissance?", "Renaissance"),
        ("What is a mammal?", "Mammal"),
        ("What is the French Revolution?", "French Revolution"),
        ("Who was Abraham Lincoln?", "Abraham Lincoln"),
        ("What is the Great Wall of China?", "Great Wall of China"),
        ("What is the Big Bang?", "Big Bang"),
        ("Who was Nikola Tesla?", "Nikola Tesla"),
        ("What is photon?", "Photon"),
        ("What is the Olympic Games?", "Olympic Games"),
        ("What is the Moon?", "Moon"),
        ("Who was Julius Caesar?", "Julius Caesar"),
        ("What is the Amazon rainforest?", "Amazon rainforest"),
        ("What is democracy?", "Democracy"),
        ("What is Buddhism?", "Buddhism"),
        ("What is the Industrial Revolution?", "Industrial Revolution"),
        ("What is the Atlantic Ocean?", "Atlantic Ocean"),
    ]
    title_set = {a["title"].lower() for a in articles}
    pairs = []
    for query, title in candidates:
        if title.lower() in title_set:
            pairs.append({"query": query, "relevant_titles": [title]})
    return pairs


if __name__ == "__main__":
    import yaml

    with open("config/config.yaml") as fh:
        config = yaml.safe_load(fh)
    from src.data.dataset import load_wikipedia

    articles = load_wikipedia(config)
    build_eval_queries(articles)
