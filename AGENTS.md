# AGENTS.md

## Cursor Cloud specific instructions

This is a single Python product, **`ai-fund-manager`** — an LLM-driven discretionary fund manager. It ships a `fund` CLI (the core pipeline), a FastAPI web dashboard, and an optional Telegram bot. State lives in a local **SQLite** file (`data/fund.db`); there is no separate database server to run.

### Environment / tooling
- Python `>=3.11`, managed with **`uv`** (installed at `~/.local/bin/uv`, already on `PATH`). The startup update script runs `uv sync --extra dev`, which creates `.venv/` with runtime + dev deps (`pytest`, `ruff`).
- Run everything through the venv, e.g. `uv run <cmd>` (or activate `.venv`). The CLI entry point is `fund` (`uv run fund ...`).
- `OPENAI_API_KEY` is required only for the LLM decision step (`fund run`) and is injected from secrets. Config (`config/config.yaml`) pins `model_id: gpt-5.5`; override with `FUND_MODEL_ID` if that model is unavailable to the key.
- Config loads `.env` with `load_dotenv(override=False)`, so injected env vars win. **Do not create a placeholder `.env`** from `.env.example` — its dummy `TELEGRAM_BOT_TOKEN=...` would be picked up and cause junk Telegram send attempts. Leave Telegram vars unset to skip notifications (handled gracefully).

### Standard commands (see `pyproject.toml`)
- Lint: `uv run ruff check .` — note the repo currently has ~37 pre-existing ruff findings; the tool works, these are not environment problems.
- Test: `uv run pytest` (config in `pyproject.toml`; `asyncio_mode=auto`).
- Build: `uv build`.

### Running the product
- One-time per fresh VM: `uv run fund init` seeds the SQLite portfolio (`data/` is gitignored, so the DB does not persist across fresh clones/VMs and must be re-initialised).
- CLI examples: `fund fill <TICKER> <shares> <price> <fee> --side buy` records a trade; `fund status`, `fund transactions`, `fund report`.
- Full decision pipeline: `uv run fund run` — this fetches Yahoo Finance data for ~1600 tickers (slow, network-heavy, can be rate-limited) then calls the LLM. Use `--dry-run` and `--skip-news --skip-macro --skip-fundamentals` for faster iteration.
- Web dashboard (read-only reporting UI): `uv run uvicorn fundmgr.web.app:app --host 0.0.0.0 --port 8000`. It pulls live prices from Yahoo Finance on page load.
- A second "global simulation" fund shares the same code, selected via `FUND_CONFIG=config/config_global.yaml` (paper trading, benchmark `URTH`, DB `data/fund_global.db`, dashboard on port 8001).

### Gotchas
- `fund universe` (CLI) has a pre-existing bug (references an undefined `cfg`) and crashes — this is a code bug, not setup. The web `/universe` page works fine.
- FinBERT sentiment (`fund run` without `--skip-news`) needs the optional `torch` extra (`uv sync --extra torch`, CPU wheels) plus a ~440 MB HuggingFace model download on first use; it is intentionally not part of the default install.
- Telegram screenshot OCR (optional) needs the system package `tesseract-ocr`, which is not installed by default.
