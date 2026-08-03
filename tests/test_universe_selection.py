"""Tests for large-universe selection and screener pinning."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

from fundmgr.config import ScreenerConfig, UniverseTicker
from fundmgr.data.prices import TickerFeatures
from fundmgr.data.screener import screen
from fundmgr.data.universe_selection import (
    news_watch_tickers,
    select_tickers_for_price_fetch,
    tickers_for_feature_build,
)
from fundmgr.state.store import Store


def _ticker(yahoo: str) -> UniverseTicker:
    return UniverseTicker(
        name=yahoo,
        yahoo_ticker=yahoo,
        isin="",
        country="US",
        exchange="NASDAQ",
        sector="Technology",
        enabled=True,
    )


def _feat(ticker: str, score_hint: float = 0.0) -> TickerFeatures:
    return TickerFeatures(
        ticker=ticker,
        name=ticker,
        last_price=100.0 + score_hint,
        last_date="2026-07-01",
        data_age_trading_days=1,
    )


def test_select_always_includes_held_and_pinned():
    tickers = [_ticker(f"T{i}") for i in range(100)]
    cfg = ScreenerConfig(
        price_fetch_limit=10,
        rotate_weeks=4,
        pinned_tickers=["T99"],
    )
    selected, note = select_tickers_for_price_fetch(tickers, held={"T0"}, cfg=cfg)
    symbols = {t.yahoo_ticker for t in selected}
    assert "T0" in symbols
    assert "T99" in symbols
    assert len(selected) == 10
    assert "bucket" in note


def test_select_full_universe_when_under_limit():
    tickers = [_ticker("A"), _ticker("B")]
    cfg = ScreenerConfig(price_fetch_limit=100)
    selected, note = select_tickers_for_price_fetch(tickers, held=set(), cfg=cfg)
    assert len(selected) == 2
    assert note == "full"


def test_rotation_bucket_is_stable():
    cfg = ScreenerConfig(price_fetch_limit=5, rotate_weeks=4, pinned_tickers=[])
    tickers = [_ticker(f"T{i}") for i in range(20)]
    with patch("fundmgr.data.universe_selection.datetime") as mock_dt:
        mock_dt.utcnow.return_value = datetime(2026, 7, 7)
        a, _ = select_tickers_for_price_fetch(tickers, held=set(), cfg=cfg)
        b, _ = select_tickers_for_price_fetch(tickers, held=set(), cfg=cfg)
    assert {t.yahoo_ticker for t in a} == {t.yahoo_ticker for t in b}


def test_tickers_for_feature_build_merges_cache_and_fetch(tmp_path):
    store = Store(tmp_path / "fund.db")
    store.save_prices("CACHED", [
        {"date": "2026-07-01", "open": 1, "high": 1, "low": 1, "close": 10, "volume": 100},
    ])
    universe = [_ticker("CACHED"), _ticker("FRESH")]
    out = tickers_for_feature_build(universe, [_ticker("FRESH")], store)
    assert {t.yahoo_ticker for t in out} == {"CACHED", "FRESH"}


def test_news_watch_tickers_held_plus_pinned():
    universe = [_ticker("AAPL"), _ticker("MSFT"), _ticker("ZZZ")]
    cfg = ScreenerConfig(pinned_tickers=["ZZZ"])
    out = news_watch_tickers(universe, held={"AAPL"}, cfg=cfg)
    assert {t.yahoo_ticker for t in out} == {"AAPL", "ZZZ"}


def test_screen_includes_pinned_regardless_of_score():
    features = {
        "LOW": _feat("LOW"),
        "HIGH": _feat("HIGH"),
        "PIN": _feat("PIN"),
    }
    features["HIGH"].return_20d_pct = 25.0
    features["LOW"].return_20d_pct = -5.0
    selected, screened_out = screen(
        features,
        held_tickers=set(),
        top_n=2,
        pinned_tickers={"PIN"},
    )
    assert selected.keys() == {"PIN", "HIGH"}
    assert screened_out == 1
