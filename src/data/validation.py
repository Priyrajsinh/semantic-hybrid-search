"""Pandera schema for chunk metadata DataFrame."""

from pandera.pandas import Column, DataFrameSchema

chunk_metadata_schema = DataFrameSchema(
    {
        "chunk_id": Column(int),
        "article_id": Column(int),
        "title": Column(str),
        "text": Column(str),
    }
)
