#!/bin/bash
# Prepare a Claude Code on the web container so tests and linters can run.
#
# The fund depends on the shared FinanceData package as an editable path dep
# (see [tool.uv.sources] in pyproject.toml), which deploy/SETUP.md expects to sit
# beside this repo. A web session clones only this repo, so `uv sync` fails to
# resolve before a single test runs. Clone the sibling, then sync.
set -euo pipefail

# Local machines already have the sibling checkout and their own venv.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
FINANCEDATA_DIR="${FINANCEDATA_DIR:-$(cd "$PROJECT_DIR/.." && pwd)/FinanceData}"
FINANCEDATA_REPO="${FINANCEDATA_REPO:-https://github.com/acwesterberg/FinanceData.git}"

# Idempotent: refresh an existing checkout, clone only when absent. FinanceData
# must stay on main (deploy/SETUP.md) — the fund pins no revision of it.
if [ -d "$FINANCEDATA_DIR/.git" ]; then
  echo "FinanceData present at $FINANCEDATA_DIR — fetching latest main"
  git -C "$FINANCEDATA_DIR" fetch --depth 1 origin main \
    && git -C "$FINANCEDATA_DIR" reset --hard FETCH_HEAD \
    || echo "warning: could not refresh FinanceData; using the existing checkout"
else
  echo "Cloning FinanceData into $FINANCEDATA_DIR"
  git clone --depth 1 "$FINANCEDATA_REPO" "$FINANCEDATA_DIR"
fi

cd "$PROJECT_DIR"
uv sync --extra dev

# `uv run` is what the venv is reachable through; make that explicit for the
# session rather than leaving it to be rediscovered.
echo 'export UV_PROJECT_ENVIRONMENT="'"$PROJECT_DIR"'/.venv"' >> "${CLAUDE_ENV_FILE:-/dev/null}"

echo "Ready: uv run pytest -q | uv run ruff check src/ tests/"
