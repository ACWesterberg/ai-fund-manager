"""Recording a fill values the book at market, never at cost basis.

`fund fill` rewrote the day's NAV point as shares × average cost + cash, and the
real fund's fills land on the Monday the weekly run has just marked the book —
so its history fell by exactly the fees through July while prices moved, and
showed a gain only once a position was sold.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner

from fundmgr import cli as commands
from fundmgr.config import AppConfig
from fundmgr.state.models import NavPoint
from fundmgr.state.store import Store

CONFIG = Path(__file__).resolve().parents[1] / "config"
TODAY = datetime.utcnow().strftime("%Y-%m-%d")


@pytest.fixture
def book(tmp_path, monkeypatch):
    cfg = AppConfig(db_path=tmp_path / "fund.db", universe_path=CONFIG / "universe.csv")
    store = Store(cfg.db_path)
    store.initialise(50_000)
    monkeypatch.setattr(commands, "_get_store", lambda cfg_=None: (cfg, store))
    monkeypatch.setattr(commands, "fetch_and_cache_benchmark", lambda *a, **k: True)
    return store


def _prices(monkeypatch, price):
    monkeypatch.setattr("fundmgr.data.quotes.live_prices", lambda ts: {t: price for t in ts})


def _fill(*extra):
    return CliRunner().invoke(commands.cli, ["fill", "VOLV-B.ST", "10", "100", "1", *extra])


def test_a_fill_marks_todays_point_at_live_prices(book, monkeypatch):
    _prices(monkeypatch, 120.0)
    result = _fill()
    assert result.exit_code == 0, result.output
    point = book.get_nav_history()[-1]
    assert point.date == TODAY
    # Cost basis would be 48,999 + 10 × 100 = 49,999.
    assert point.portfolio_nav_sek == pytest.approx(48_999 + 10 * 120.0)


def test_without_a_price_the_runs_marked_point_survives(book, monkeypatch):
    book.upsert_nav(NavPoint(date=TODAY, portfolio_nav_sek=51_234.0,
                             benchmark_value=100.0, cash_sek=50_000.0))
    _prices(monkeypatch, None)
    result = _fill()
    assert result.exit_code == 0, result.output
    assert book.get_positions()[0].shares == 10
    assert book.get_nav_history()[-1].portfolio_nav_sek == pytest.approx(51_234.0)
    assert "left unchanged" in result.output


def test_a_valuation_error_never_fails_a_booked_fill(book, monkeypatch):
    def boom(ts):
        raise RuntimeError("quote service down")
    monkeypatch.setattr("fundmgr.data.quotes.live_prices", boom)
    result = _fill()
    assert result.exit_code == 0, result.output
    assert book.get_positions()[0].shares == 10
    assert book.get_nav_history() == []


def test_a_back_dated_fill_writes_no_nav_point(book, monkeypatch):
    _prices(monkeypatch, 120.0)
    past = (datetime.utcnow() - timedelta(days=10)).strftime("%Y-%m-%d")
    result = _fill("--date", past)
    assert result.exit_code == 0, result.output
    assert book.get_nav_history() == []
    assert "left as it is" in result.output
