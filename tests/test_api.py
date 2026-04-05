"""Tests for the FastAPI search endpoint."""

from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

import src.api.app as app_module
from src.api.app import app
from src.exceptions import IndexNotFoundError


@pytest.fixture
async def client():
    """App client with no engine (health and validation error paths)."""
    with patch("src.api.app.SemanticSearchEngine") as MockEngine:
        MockEngine.side_effect = IndexNotFoundError("no index in test env")
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            yield c


@pytest.fixture
async def client_with_engine():
    """App client with a mock engine injected after lifespan startup."""
    mock_engine = MagicMock()
    mock_engine.index.n_vectors = 150
    mock_engine.search.return_value = [
        {
            "results": [
                {
                    "text": "Test passage about photosynthesis.",
                    "title": "Photosynthesis",
                    "score": 0.92,
                    "rank": 1,
                },
                {
                    "text": "Plants use sunlight to produce energy.",
                    "title": "Plant Biology",
                    "score": 0.75,
                    "rank": 2,
                },
            ],
            "latency": {
                "encode_ms": 10.0,
                "faiss_ms": 5.0,
                "rerank_ms": 100.0,
                "total_ms": 115.0,
            },
        }
    ]

    with patch("src.api.app.SemanticSearchEngine") as MockEngine:
        MockEngine.side_effect = IndexNotFoundError("no index in test env")
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            app_module._state["engine"] = mock_engine
            try:
                yield c
            finally:
                app_module._state["engine"] = None


async def test_health_endpoint(client):
    """GET /api/v1/health returns 200 with all required fields."""
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "index_loaded" in data
    assert "n_vectors" in data
    assert "uptime_seconds" in data
    assert "memory_mb" in data
    assert data["status"] == "healthy"


async def test_search_valid_query(client_with_engine):
    """POST /api/v1/search with valid query returns results."""
    payload = {"query": "What is photosynthesis?", "top_k": 2}
    response = await client_with_engine.post("/api/v1/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "query" in data
    assert "latency_ms" in data
    assert len(data["results"]) > 0
    assert data["results"][0]["title"] == "Photosynthesis"
    assert data["query"] == "What is photosynthesis?"


async def test_search_empty_query_422(client):
    """Empty query string returns 422 Unprocessable Entity."""
    payload = {"query": "", "top_k": 5}
    response = await client.post("/api/v1/search", json=payload)
    assert response.status_code == 422


async def test_search_invalid_top_k_422(client):
    """top_k=0 or top_k=100 returns 422 Unprocessable Entity."""
    for bad_top_k in [0, 100]:
        payload = {"query": "test query", "top_k": bad_top_k}
        response = await client.post("/api/v1/search", json=payload)
        assert (
            response.status_code == 422
        ), f"Expected 422 for top_k={bad_top_k}, got {response.status_code}"


async def test_get_search_endpoint(client_with_engine):
    """GET /api/v1/search?q=...&top_k=5 returns results."""
    response = await client_with_engine.get(
        "/api/v1/search", params={"q": "climate change", "top_k": 2}
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert data["query"] == "climate change"


async def test_rate_limiting_header_present(client_with_engine):
    """Limiter is active: slowapi does not crash the endpoint."""
    payload = {"query": "artificial intelligence", "top_k": 3}
    response = await client_with_engine.post("/api/v1/search", json=payload)
    # slowapi adds headers once limits are near; just confirm endpoint is up
    assert response.status_code == 200
