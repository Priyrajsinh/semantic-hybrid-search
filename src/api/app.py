"""FastAPI application for B4 Semantic Search."""

from fastapi import FastAPI

app = FastAPI(title="B4 Semantic Search API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}
