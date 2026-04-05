"""End-to-end data pipeline: load → chunk → validate → save."""

import hashlib
import json
from pathlib import Path

import yaml

from src.data.chunker import chunk_articles
from src.data.dataset import load_wikipedia
from src.data.validation import validate_chunks
from src.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")


def run_pipeline(config_path: str = "config/config.yaml") -> None:
    """Run data pipeline: load Wikipedia → chunk → validate → save parquet.

    Args:
        config_path: Path to YAML config file.
    """
    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    articles = load_wikipedia(config)

    df = chunk_articles(
        articles,
        chunk_size=config["data"]["chunk_size"],
        chunk_overlap=config["data"]["chunk_overlap"],
    )
    df = validate_chunks(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    chunks_path = PROCESSED_DIR / "chunks.parquet"
    df.to_parquet(str(chunks_path), index=False)
    logger.info("Saved %d chunks to %s", len(df), chunks_path)

    checksum = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    checksums_path = RAW_DIR / "checksums.json"
    with open(str(checksums_path), "w") as fh:
        json.dump({"chunks.parquet": checksum}, fh, indent=2)
    logger.info("Saved checksum to %s", checksums_path)


if __name__ == "__main__":
    run_pipeline()
