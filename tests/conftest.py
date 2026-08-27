"""Shared test fixtures."""
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_alias_store(tmp_path, monkeypatch):
    """Point the learned-alias store at a temp file for every test.

    Autouse and unconditional: importing through the web routes records aliases
    as a side effect, so without this a test run writes into the real DATA_DIR
    and then leaks into later tests — a learned name would silently outrank the
    universe ISIN map, which is exactly what aliases are designed to do.
    """
    monkeypatch.setenv("FUND_ALIAS_PATH", str(tmp_path / "ticker_aliases.json"))


@pytest.fixture(autouse=True)
def no_live_llm_calls(monkeypatch):
    """Fail any unmocked LLM call instead of billing a real one.

    Without credentials `call_llm` raises LLMError on its own, so this changes
    nothing on CI — but on a developer box with keys in .env the learning
    distillation inside `fund run` / `paper-track` would otherwise reach the API
    during a test run. Tests that want a response monkeypatch their own.
    """
    from fundmgr.engine import client

    def _refuse(*a, **k):
        raise client.LLMError("live LLM calls are disabled under test")

    monkeypatch.setattr(client, "call_llm", _refuse)


@pytest.fixture
def alias_file() -> Path:
    """The isolated alias file for the current test."""
    return Path(os.environ["FUND_ALIAS_PATH"])
