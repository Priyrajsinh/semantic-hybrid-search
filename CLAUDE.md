# CLAUDE.md — B4 Semantic Search Project Rules

## Carry-Forward Error Prevention (from carry.md)
1. Set `target-version = ["py310"]` in `[tool.black]` in `pyproject.toml`.
2. Only import modules you actually use. Run `flake8` before every commit.
3. Run `isort src/ tests/ --profile black` before every commit.
4. Every `src/` file needs at least one test. Coverage gate: 70%.
5. Use `from pythonjsonlogger import json as jsonlogger` (not deprecated path).
6. Use `from pandera.pandas import Column, DataFrameSchema` (not top-level pandera).
7. Pydantic v2 only: `@field_validator` + `@classmethod`, never `@validator`.
8. Run ALL 5 gates before every commit: black -> isort -> flake8 -> bandit -> pytest.
9. Use Python 3.10. Verify with `py --list` before scaffold.
10. Create `.flake8` with `max-line-length = 88` matching `[tool.black]`.
11. Use built-in `open()` for file I/O, never `Path.open()`.
12. No module-level side effects (no `set_seed()` at import time).
13. HF Space must be 100% self-contained. Never import from `src/`.
14. HF Space `requirements.txt` must match every import in `app.py`.
15. Do not use `allow_flagging` in Gradio (removed in v5).
16. Use `plt.switch_backend("Agg")` after all imports for matplotlib.
17. Annotate mixed-type dicts explicitly for mypy: `dict[str, str | int]`.
18. Verify `git config user.email` = `priyrajsinh03@gmail.com` before first commit.
19. Tab 1 = non-technical UX with pre-loaded examples. Dev features in Tab 2.

## Project-Specific Rules
- `config/config.yaml` is the single source of truth. Never hardcode hyperparameters.
- Use `get_logger(__name__)` everywhere. Zero `print()` statements.
- `filterwarnings = ["error::DeprecationWarning"]` is set in pytest config — do not suppress warnings.
- All dependency versions are pinned in `requirements.txt`. Do not add unpinned deps.
- FAISS index uses IndexIVFFlat (nlist=100) with IndexFlatIP fallback when n_vectors < 1000.
- BM25 index is persisted with joblib to `models/bm25_index.pkl` — never rebuild on startup.
- Cross-encoder `predict()` is wrapped with `functools.lru_cache` for repeated queries.
