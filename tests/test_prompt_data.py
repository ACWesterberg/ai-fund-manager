"""
What the 2 Nov run's model flagged as bad data, fixed at the source.

Its notes: dividend yields "mis-scaled" (KCC 618%), BAVA.CO at two prices
(369.34 vs DKK 252.40), volume ratios of 0.0-0.2x, and an "Unknown" sector
near its cap. Each was real.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from fundmgr.config import UniverseTicker
from fundmgr.data import prices
from fundmgr.data.fundamentals import apply_to_features, normalise_sector
from fundmgr.data.prices import TickerFeatures
from fundmgr.engine import prompt
from fundmgr.state.models import PortfolioSnapshot, Position
from fundmgr.state.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db")
    s.initialise(1000)
    return s


def _feat(ticker, price=100.0, ccy="SEK", sector=None):
    return TickerFeatures(ticker=ticker, name=ticker, last_price=price, last_date="2026-11-02",
                          data_age_trading_days=0, currency=ccy, sector=sector)


# ── Dividend yield ───────────────────────────────────────────────────────────

def test_yahoo_dividend_yield_is_already_a_percentage(store):
    # INWI.ST on 2 Nov: dividendYield 2.88 with trailingAnnualDividendYield 0.0288.
    store.save_fundamentals("INWI.ST", {"dividend_yield": 2.88, "profit_margin": 0.05})
    feats = {"INWI.ST": _feat("INWI.ST")}
    apply_to_features(feats, store)
    assert feats["INWI.ST"].dividend_yield_pct == 2.88
    assert feats["INWI.ST"].profit_margin_pct == 5.0      # true fractions still scale


def test_criteria_read_the_yield_as_a_percentage():
    from fundmgr import evidence
    assert evidence.FUND_FIELD_META["dividend_yield"]["is_fraction"] is False


# ── Sectors ──────────────────────────────────────────────────────────────────

def test_unknown_is_not_a_sector_and_yahoo_names_join_the_universes():
    assert normalise_sector("Unknown") is None and normalise_sector("") is None
    assert normalise_sector("Healthcare") == "Health Care"
    assert normalise_sector("Financial Services") == "Financials"
    assert normalise_sector("Industrials") == "Industrials"


def test_a_missing_sector_is_filled_from_yahoo_but_a_listed_one_is_kept(store):
    store.save_fundamentals("NOBA.ST", {"sector": "Financial Services"})
    store.save_fundamentals("KCC.OL", {"sector": "Energy"})
    feats = {"NOBA.ST": _feat("NOBA.ST"), "KCC.OL": _feat("KCC.OL", sector="Industrials")}
    apply_to_features(feats, store)
    assert feats["NOBA.ST"].sector == "Financials"
    assert feats["KCC.OL"].sector == "Industrials"


def test_unclassified_holdings_are_reported_but_not_capped():
    snap = PortfolioSnapshot(positions=[Position("A.ST", 10, 100, 100), Position("B.ST", 10, 100, 100)],
                             cash_sek=0.0)
    text = prompt._sector_weights_block(snap, {"A.ST": _feat("A.ST", sector="Industrials"),
                                               "B.ST": _feat("B.ST")}, 35.0)
    assert "Unknown" not in text
    assert "Industrials" in text and "not counted against the cap" in text


# ── Portfolio prices: SEK, with the listing's own beside a foreign one ───────

def test_foreign_holdings_show_both_sek_and_native_prices():
    snap = PortfolioSnapshot(positions=[Position("BAVA.CO", 9, 332.97, 369.34),
                                        Position("NOBA.ST", 129, 89.10, 91.20)], cash_sek=0.0)
    feats = {"BAVA.CO": _feat("BAVA.CO", 252.40, "DKK"), "NOBA.ST": _feat("NOBA.ST", 91.20)}
    text = prompt._portfolio_block(snap, None, features=feats, in_sek=True)
    assert "in SEK" in text
    bava = next(ln for ln in text.splitlines() if "BAVA.CO" in ln)
    assert "369.34" in bava and "(DKK 252.40)" in bava
    noba = next(ln for ln in text.splitlines() if "NOBA.ST" in ln)
    assert "(SEK" not in noba


def test_a_native_currency_book_is_not_labelled_sek():
    snap = PortfolioSnapshot(positions=[Position("AAPL", 1, 150, 160)], cash_sek=0.0)
    text = prompt._portfolio_block(snap, None, features={"AAPL": _feat("AAPL", 160, "USD")})
    assert "Open positions:" in text and "SEK)" not in text.split("Open positions")[1]


# ── Volume: the last complete session, not the first half hour ───────────────

def _series(store, today_volume):
    today = datetime.now(timezone.utc).date()
    rows = []
    for i in range(30, 0, -1):
        d = today - timedelta(days=i)
        rows.append({"date": d.isoformat(), "open": 10, "high": 10, "low": 10, "close": 10,
                     "volume": 1_000})
    rows[-1]["volume"] = 3_000        # yesterday: a heavy session
    rows.append({"date": today.isoformat(), "open": 10, "high": 10, "low": 10, "close": 10,
                 "volume": today_volume})
    store.save_prices("X.ST", rows)
    return (today - timedelta(days=40)).isoformat()


def _ticker():
    return UniverseTicker(name="X", yahoo_ticker="X.ST", isin="", country="SE", exchange="OMXS",
                          sector="Unknown", enabled=True)


def test_volume_ignores_todays_bar_while_the_venue_trades(store, monkeypatch):
    since = _series(store, today_volume=50)
    monkeypatch.setattr(prices, "_venue_open", lambda exchange: True)
    f = prices.compute_features(_ticker(), store, since)
    assert f.rel_volume == pytest.approx(3.0)
    assert f.sector is None


def test_volume_uses_todays_bar_once_the_session_is_over(store, monkeypatch):
    since = _series(store, today_volume=2_000)
    monkeypatch.setattr(prices, "_venue_open", lambda exchange: False)
    f = prices.compute_features(_ticker(), store, since)
    assert f.rel_volume == pytest.approx(2_000 / ((19 * 1_000 + 3_000) / 20), rel=1e-2)
