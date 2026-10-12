"""Repairing NAV history that fills wrote at cost basis, and the paths that wrote it.

The real fund's every fill-day point read shares × average cost + cash, so its
history showed +6.8% while the book, marked, was up 11.7%. `remark-history`
re-values those points from cached closes; sleeve fills stop writing them.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from fundmgr import cli as commands
from fundmgr import paper
from fundmgr.config import AppConfig
from fundmgr.state.models import NavPoint, Transaction
from fundmgr.state.store import Store

CONFIG = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture
def book(tmp_path, monkeypatch):
    cfg = AppConfig(db_path=tmp_path / "fund.db", universe_path=CONFIG / "universe.csv",
                    fx_to_sek=True)
    store = Store(cfg.db_path)
    store.initialise(50_000)
    monkeypatch.setattr(commands, "_get_store", lambda cfg_=None: (cfg, store))
    return store


def _buy(store: Store, day: str, ticker: str = "VOLV-B.ST", shares: float = 10,
         price: float = 100.0) -> None:
    store.apply_fill(Transaction(
        ticker=ticker, side="buy", shares=shares, price_sek=price, fee_sek=1.0,
        source="fill", timestamp=datetime.strptime(day, "%Y-%m-%d").replace(hour=12),
    ))


def _cost_point(store: Store, day: str) -> None:
    cost = sum(p.shares * p.avg_cost_sek for p in store.get_positions())
    store.upsert_nav(NavPoint(date=day, portfolio_nav_sek=cost + store.get_cash(),
                              benchmark_value=1000.0, cash_sek=store.get_cash()))


def _close(store: Store, ticker: str, day: str, close: float) -> None:
    store.save_prices(ticker, [{"date": day, "open": close, "high": close,
                                "low": close, "close": close, "volume": 1}])


def _remark(*extra):
    return CliRunner().invoke(commands.cli, ["remark-history", *extra])


def test_dry_run_prints_without_writing(book):
    _buy(book, "2026-07-08")
    _cost_point(book, "2026-07-08")
    _close(book, "VOLV-B.ST", "2026-07-08", 120.0)

    result = _remark()
    assert result.exit_code == 0, result.output
    assert "Would re-mark 1" in result.output
    assert book.get_nav_history()[-1].portfolio_nav_sek == pytest.approx(48_999 + 1_000)


def test_apply_values_a_cost_point_at_the_cached_close(book):
    _buy(book, "2026-07-08")
    _cost_point(book, "2026-07-08")
    _close(book, "VOLV-B.ST", "2026-07-06", 120.0)  # last close before the point

    result = _remark("--apply")
    assert result.exit_code == 0, result.output
    point = book.get_nav_history()[-1]
    assert point.portfolio_nav_sek == pytest.approx(48_999 + 10 * 120.0)
    assert point.cash_sek == pytest.approx(48_999)
    assert point.benchmark_value == pytest.approx(1000.0)


def test_a_point_already_at_market_is_left_alone(book):
    _buy(book, "2026-07-08")
    book.upsert_nav(NavPoint(date="2026-07-08", portfolio_nav_sek=50_300.0,
                             benchmark_value=1000.0, cash_sek=book.get_cash()))
    _close(book, "VOLV-B.ST", "2026-07-08", 120.0)

    result = _remark("--apply")
    assert result.exit_code == 0, result.output
    assert book.get_nav_history()[-1].portfolio_nav_sek == pytest.approx(50_300.0)


def test_a_missing_or_stale_close_leaves_the_point_as_it_is(book):
    _buy(book, "2026-07-08")
    _cost_point(book, "2026-07-08")
    _close(book, "VOLV-B.ST", "2026-06-20", 120.0)  # 18 days old

    result = _remark("--apply")
    assert result.exit_code == 0, result.output
    assert "left as it is (no close for VOLV-B.ST)" in result.output
    assert book.get_nav_history()[-1].portfolio_nav_sek == pytest.approx(49_999)


def test_foreign_holdings_use_the_rate_of_the_close_date(book, monkeypatch):
    seen = []

    def rate_on(cur, date):
        seen.append((cur, date))
        return 0.5
    monkeypatch.setattr("fundmgr.data.fx.rate_to_sek_on", rate_on)
    _buy(book, "2026-07-08", ticker="NOVO-B.CO", shares=2, price=500.0)
    _cost_point(book, "2026-07-08")
    _close(book, "NOVO-B.CO", "2026-07-07", 1200.0)  # DKK

    result = _remark("--apply")
    assert result.exit_code == 0, result.output
    assert seen == [("DKK", "2026-07-07")]
    assert book.get_nav_history()[-1].portfolio_nav_sek == pytest.approx(48_999 + 2 * 600.0)


def test_a_sale_back_to_dust_closes_the_position(book):
    _buy(book, "2026-07-08", shares=1.0)
    book.apply_fill(Transaction(
        ticker="VOLV-B.ST", side="sell", shares=1.0 - 1e-13, price_sek=100.0,
        fee_sek=1.0, source="auto", timestamp=datetime(2026, 7, 9, 12),
    ))
    assert book.get_positions() == []


# ── Sleeves ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sleeve(tmp_path, monkeypatch):
    import fundmgr.data.fx as fx

    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path / "paper")
    monkeypatch.setattr(fx, "rate_to_sek", lambda cur, store=None: {"SEK": 1.0}.get(cur.upper()))
    monkeypatch.setattr(paper, "detect_currency", lambda t: "SEK")
    monkeypatch.setattr(paper, "_cache_price_history", lambda *a, **k: None)
    monkeypatch.setattr(paper, "_search_symbol", lambda name: None)
    import fundmgr.data.benchmark as benchmark
    monkeypatch.setattr(benchmark, "fetch_and_cache_benchmark", lambda *a, **k: True)
    slug, _ = paper.create_portfolio("Mirror", 100_000, "VOLV-B.ST 100%",
                                     kind="live", execute_buys=False)
    return slug


def test_a_sleeve_fill_marks_today_at_live_prices(sleeve, monkeypatch):
    monkeypatch.setattr("fundmgr.data.quotes.live_prices", lambda ts: {t: 120.0 for t in ts})
    result = CliRunner().invoke(commands.cli, ["paper-fill", sleeve, "VOLV-B.ST", "10", "100", "1"])
    assert result.exit_code == 0, result.output
    _, store = paper.open_portfolio(sleeve)
    # Cost basis would be 98,999 + 1,000.
    assert store.get_nav_history()[-1].portfolio_nav_sek == pytest.approx(98_999 + 1_200)


def test_a_sleeve_fill_without_a_quote_keeps_the_point(sleeve, monkeypatch):
    monkeypatch.setattr("fundmgr.data.quotes.live_prices", lambda ts: {t: None for t in ts})
    _, store = paper.open_portfolio(sleeve)
    before = store.get_nav_history()[-1]
    result = CliRunner().invoke(commands.cli, ["paper-fill", sleeve, "VOLV-B.ST", "10", "100", "1"])
    assert result.exit_code == 0, result.output
    assert "left unchanged" in result.output
    after = store.get_nav_history()[-1]
    assert (after.date, after.portfolio_nav_sek) == (before.date, before.portfolio_nav_sek)
