"""Build FAISS index from processed chunks parquet."""

import time
from pathlib import Path

import mlflow
import pandas as pd
import yaml

from src.logger import get_logger
from src.retrieval.encoder import SemanticEncoder
from src.retrieval.index import FAISSIndex

_logger = get_logger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
CONFIG_PATH = Path("config/config.yaml")


def build_index() -> None:
    """Load chunks, encode, build FAISS index, and save artefacts."""
    with open(str(CONFIG_PATH)) as fh:
        config = yaml.safe_load(fh)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    chunks_path = PROCESSED_DIR / "chunks.parquet"
    df = pd.read_parquet(str(chunks_path))
    texts = df["text"].tolist()
    _logger.info("Loaded %d chunks from %s", len(texts), chunks_path)

    enc_cfg = config["encoder"]
    encoder = SemanticEncoder(model_name=enc_cfg["model_name"])

    t0 = time.time()
    embeddings = encoder.encode(
        texts,
        batch_size=enc_cfg["batch_size"],
        show_progress=True,
    )
    build_time = time.time() - t0
    _logger.info("Encoding done in %.1fs", build_time)

    idx_cfg = config["index"]
    faiss_index = FAISSIndex(
        embedding_dim=enc_cfg["embedding_dim"],
        nlist=idx_cfg["nlist"],
        fallback_threshold=idx_cfg["fallback_threshold"],
    )
    faiss_index.build(embeddings, nprobe=idx_cfg["nprobe"])

    index_path = MODELS_DIR / "faiss_index.bin"
    faiss_index.save(str(index_path))
    _logger.info("Saved FAISS index to %s", index_path)

    meta_path = MODELS_DIR / "chunk_metadata.parquet"
    df[["article_title", "chunk_index", "text"]].to_parquet(str(meta_path), index=False)
    _logger.info("Saved chunk metadata to %s", meta_path)

    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="build_index"):
        mlflow.log_params(
            {
                "encoder_model": enc_cfg["model_name"],
                "n_chunks": len(texts),
                "embedding_dim": enc_cfg["embedding_dim"],
            }
        )
        mlflow.log_metrics(
            {
                "build_time_seconds": build_time,
                "n_vectors": faiss_index.n_vectors,
            }
        )

    _logger.info("build_index complete. Total time: %.1fs", build_time)


if __name__ == "__main__":
    build_index()
