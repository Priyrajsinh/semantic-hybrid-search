"""FastAPI application for B4 Semantic Search."""

import time
from contextlib import asynccontextmanager

import psutil
import yaml
from fastapi import FastAPI, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.data.schemas import SearchQuery, SearchResponse, SearchResult
from src.exceptions import IndexNotFoundError, SearchError
from src.logger import get_logger
from src.retrieval.search import SemanticSearchEngine

logger = get_logger(__name__)

with open("config/config.yaml") as _fh:
    _config = yaml.safe_load(_fh)

_state: dict[str, object] = {
    "engine": None,
    "start_time": None,
}

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load SemanticSearchEngine on startup; release on shutdown."""
    _state["start_time"] = time.time()
    try:
        _engine = SemanticSearchEngine(_config)
        _state["engine"] = _engine
        logger.info(
            "startup_complete",
            extra={"n_vectors": _engine.index.n_vectors},
        )
    except IndexNotFoundError as exc:
        logger.warning("engine_not_loaded", extra={"reason": str(exc)})
    yield
    logger.info("shutdown")


app = FastAPI(
    title="B4 Semantic Search API",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,  # type: ignore[arg-type]
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_config["api"]["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])


@app.middleware("http")
async def check_content_length(request: Request, call_next):
    """Reject payloads exceeding max_payload_mb from config."""
    max_bytes = int(_config["api"]["max_payload_mb"]) * 1024 * 1024
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > max_bytes:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={
                "detail": (
                    f"Payload exceeds {_config['api']['max_payload_mb']} MB limit."
                )
            },
        )
    return await call_next(request)


@app.exception_handler(SearchError)
async def search_error_handler(request: Request, exc: SearchError) -> JSONResponse:
    logger.error("search_error", extra={"detail": str(exc)})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)},
    )


@app.get("/api/v1/health", tags=["ops"])
async def health() -> dict[str, str | bool | float | int]:
    """Return detailed health status including model load state and memory."""
    proc = psutil.Process()
    engine = _state["engine"]
    start_ts = _state["start_time"]
    start = float(start_ts) if isinstance(start_ts, float) else time.time()
    n_vecs: int = (
        engine.index.n_vectors if engine is not None else 0  # type: ignore[union-attr]
    )
    return {
        "status": "healthy",
        "model_loaded": engine is not None,
        "index_loaded": engine is not None,
        "n_vectors": n_vecs,
        "uptime_seconds": round(time.time() - start, 2),
        "memory_mb": round(proc.memory_info().rss / 1024 / 1024, 2),
    }


def _run_search(query: str, top_k: int, rerank: bool = True) -> SearchResponse:
    """Execute search via the loaded engine and return a SearchResponse."""
    engine = _state["engine"]
    if engine is None:
        raise SearchError(
            "Search engine not loaded — run build_index.py to create the FAISS index."
        )
    try:
        output = engine.search(  # type: ignore[union-attr]
            query, top_k=top_k, rerank=rerank
        )
    except Exception as exc:
        raise SearchError(str(exc)) from exc
    raw = output[0]
    results = [
        SearchResult(text=r["text"], title=r["title"], score=r["score"])
        for r in raw["results"]
    ]
    return SearchResponse(
        results=results,
        query=query,
        latency_ms=raw["latency"]["total_ms"],
    )


@app.post("/api/v1/search", response_model=SearchResponse, tags=["inference"])
@limiter.limit(_config["api"]["rate_limit_search"])
async def search_post(request: Request, body: SearchQuery) -> SearchResponse:
    """Search Wikipedia passages via POST with JSON body."""
    try:
        return _run_search(body.query, body.top_k)
    except SearchError:
        raise
    except Exception as exc:
        logger.error("search_exception", extra={"error": str(exc)})
        raise SearchError(str(exc)) from exc


@app.get("/api/v1/search", response_model=SearchResponse, tags=["inference"])
@limiter.limit(_config["api"]["rate_limit_search"])
async def search_get(
    request: Request,
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(default=5, ge=1, le=50, description="Number of results"),
) -> SearchResponse:
    """Search Wikipedia passages via GET query params — browser-friendly."""
    try:
        return _run_search(q, top_k)
    except SearchError:
        raise
    except Exception as exc:
        logger.error("search_exception", extra={"error": str(exc)})
        raise SearchError(str(exc)) from exc
