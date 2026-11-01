"""Shared test fixtures."""
import os
from datetime import datetime
from pathlib import Path

import pytest

from fundmgr import paper


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


@pytest.fixture(autouse=True)
def authenticated_dashboard_tests(monkeypatch):
    """Existing route tests exercise authenticated access; auth tests opt out per client."""
    from starlette.testclient import TestClient

    monkeypatch.setenv("FUND_WEB_USERNAME", "fund")
    monkeypatch.setenv("FUND_WEB_PASSWORD", "test-password")
    original = TestClient.__init__

    def init(self, *args, **kwargs):
        headers = dict(kwargs.get("headers") or {})
        headers.setdefault("Authorization", "Basic ZnVuZDp0ZXN0LXBhc3N3b3Jk")
        kwargs["headers"] = headers
        original(self, *args, **kwargs)

    monkeypatch.setattr(TestClient, "__init__", init)


# ── Paper-book fixtures (test_paper.py, test_accounts.py) ──────────────────────
# An isolated PAPER_DIR and mocked prices, FX and history. test_accounts.py used
# to borrow them with `pytest_plugins = ["tests.test_paper"]`, which registers
# nothing once test_paper.py has been collected first — running the two files
# in that order failed with "fixture 'paper_dir' not found".

FAKE_NATIVE = {"AAPL": 200.0, "MSFT": 500.0, "VOLV-B.ST": 300.0}
FAKE_FX = {"USD": 10.0, "SEK": 1.0}


@pytest.fixture
def paper_dir(tmp_path, monkeypatch):
    d = tmp_path / "paper"
    monkeypatch.setattr(paper, "PAPER_DIR", d)
    return d


@pytest.fixture
def mock_market(monkeypatch):
    import fundmgr.data.fx as fx
    import fundmgr.data.quotes as quotes

    monkeypatch.setattr(quotes, "live_prices", lambda tickers: {t: FAKE_NATIVE.get(t) for t in tickers})
    monkeypatch.setattr(fx, "rate_to_sek", lambda cur, store=None: FAKE_FX.get(cur.upper()))
    monkeypatch.setattr(
        paper, "detect_currency",
        lambda t: "SEK" if t.endswith(".ST") else "USD",
    )

    def fake_bench(store, symbol="URTH", **kwargs):
        today = datetime.utcnow().strftime("%Y-%m-%d")
        store.save_benchmark([{"date": today, "close": 100.0}])
        return True

    import fundmgr.data.benchmark as benchmark
    monkeypatch.setattr(benchmark, "fetch_and_cache_benchmark", fake_bench)

    def fake_history(store, tickers, lookback_days=40):
        today = datetime.utcnow().strftime("%Y-%m-%d")
        for t in tickers:
            if t in FAKE_NATIVE:
                store.save_prices(t, [{
                    "date": today, "open": FAKE_NATIVE[t], "high": FAKE_NATIVE[t],
                    "low": FAKE_NATIVE[t], "close": FAKE_NATIVE[t], "volume": 1000,
                }])

    monkeypatch.setattr(paper, "_cache_price_history", fake_history)
    monkeypatch.setattr(paper, "_search_symbol", lambda name: None)  # no network
