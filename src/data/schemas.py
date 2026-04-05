"""Pydantic v2 schemas for search API request/response."""

from pydantic import BaseModel, field_validator


class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, v: str) -> str:
        if len(v) < 1:
            raise ValueError("query must have at least 1 character")
        return v

    @field_validator("top_k")
    @classmethod
    def top_k_in_range(cls, v: int) -> int:
        if v < 1 or v > 50:
            raise ValueError("top_k must be between 1 and 50")
        return v


class SearchResult(BaseModel):
    text: str
    title: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str
    latency_ms: float
