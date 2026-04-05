"""Project-wide exception hierarchy for B4."""


class ProjectBaseError(Exception):
    """Base class for all B4 project exceptions."""


class DataLoadError(ProjectBaseError):
    """Raised when raw data cannot be loaded or parsed."""


class DataValidationError(ProjectBaseError):
    """Raised when pandera schema validation fails."""


class IndexNotFoundError(ProjectBaseError):
    """Raised when a FAISS or BM25 index file is missing."""


class SearchError(ProjectBaseError):
    """Raised when a search query fails (empty index, shape mismatch, etc.)."""


class ConfigError(ProjectBaseError):
    """Raised when config/config.yaml is missing or has invalid keys."""
