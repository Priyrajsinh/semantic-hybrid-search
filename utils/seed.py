"""Reproducibility utilities."""

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
