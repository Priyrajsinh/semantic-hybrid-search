"""Pandera schema for chunk DataFrame validation."""

import pandas as pd
import pandera as pa
from pandera.pandas import Column, DataFrameSchema

ChunkSchema = DataFrameSchema(
    {
        "article_title": Column(str, nullable=False),
        "chunk_index": Column(int, pa.Check.ge(0), nullable=False),
        "text": Column(
            str,
            pa.Check(lambda x: len(x) >= 10, element_wise=True),
            nullable=False,
        ),
    }
)


def validate_chunks(df: pd.DataFrame) -> pd.DataFrame:
    """Validate chunk DataFrame against ChunkSchema.

    Args:
        df: DataFrame with columns article_title, chunk_index, text.

    Returns:
        Validated DataFrame (unchanged if valid).

    Raises:
        pandera.errors.SchemaError: If validation fails.
    """
    return ChunkSchema.validate(df)
