"""Scaffold tests for B4 Semantic Search project."""

import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from src.data.schemas import SearchQuery
from src.logger import get_logger


def test_config_yaml_loads():
    config_path = Path("config/config.yaml")
    assert config_path.exists(), "config/config.yaml not found"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    assert "data" in cfg
    assert "encoder" in cfg
    assert "index" in cfg


def test_get_logger_returns_logger():
    logger = get_logger("test.scaffold")
    assert isinstance(logger, logging.Logger)


def test_get_logger_no_duplicate_handlers():
    get_logger("test.dedup")
    get_logger("test.dedup")
    logger = get_logger("test.dedup")
    assert len(logger.handlers) == 1


def test_search_query_valid():
    q = SearchQuery(query="What is FAISS?", top_k=5)
    assert q.query == "What is FAISS?"
    assert q.top_k == 5


def test_search_query_default_top_k():
    q = SearchQuery(query="hello")
    assert q.top_k == 5


def test_search_query_rejects_empty_query():
    try:
        SearchQuery(query="")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


def test_search_query_rejects_top_k_too_large():
    try:
        SearchQuery(query="test", top_k=51)
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


def test_search_query_rejects_top_k_zero():
    try:
        SearchQuery(query="test", top_k=0)
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


def test_exceptions_importable():
    from src.exceptions import (
        ConfigError,
        DataLoadError,
        DataValidationError,
        IndexNotFoundError,
        ProjectBaseError,
        SearchError,
    )

    assert issubclass(DataLoadError, ProjectBaseError)
    assert issubclass(DataValidationError, ProjectBaseError)
    assert issubclass(IndexNotFoundError, ProjectBaseError)
    assert issubclass(SearchError, ProjectBaseError)
    assert issubclass(ConfigError, ProjectBaseError)


def test_chunker_importable():
    from src.data.chunker import chunk_articles

    assert callable(chunk_articles)


def test_dataset_importable():
    from src.data.dataset import load_wikipedia

    assert callable(load_wikipedia)


def test_encoder_importable():
    from src.retrieval.encoder import SemanticEncoder

    assert SemanticEncoder is not None


def test_index_importable():
    from src.retrieval.index import FAISSIndex

    assert FAISSIndex is not None


def test_search_engine_importable():
    from src.retrieval.search import SemanticSearchEngine

    assert SemanticSearchEngine is not None


def test_evaluate_importable():
    from src.evaluation.evaluate import evaluate_retrieval

    assert callable(evaluate_retrieval)


def test_api_app_importable():
    from src.api.app import app

    assert app is not None


def test_seed_importable():
    from utils.seed import set_seed

    set_seed(42)
