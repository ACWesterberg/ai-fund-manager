"""The book must be valued, and every holding sellable, before anything is sized.

Three ways it wasn't: a missing FX rate read as 1.0 (a EUR holding at an
eleventh of its worth, a false −90% stop hit); a holding whose universe row was
disabled or never existed had no price, which blocked every trade, and could not
be sold; and a swap whose buy outranked its sell had the buy refused for the slot
or cash the sell was about to free.
"""
from __future__ import annotations

from datetime import datetime

import pytest
from click.testing import CliRunner

from fundmgr import cli as commands
from fundmgr.config import AppConfig, holding_rows
from fundmgr.data.prices import TickerFeatures
from fundmgr.engine.client import LLMError
from fundmgr.engine.schema import Action, DecisionRun
from fundmgr.guardrails.rules import apply_guardrails
from fundmgr.state.models import PortfolioSnapshot, Position
from fundmgr.state.store import Store

HEADER = "name,yahoo_ticker,isin,country,exchange,sector,enabled\n"


def act(ticker, side, weight, confidence):
    return Action(ticker=ticker, side=side, target_weight_pct=weight, sek_estimate=0,
                  confidence=confidence, thesis="A test thesis")


def guard(actions, snap, cfg, universe=None):
    tickers = {a.ticker for a in actions} | {p.ticker for p in snap.positions}
    features = {t: TickerFeatures(t, t, 100, "2026-10-28", 0, sector="Tech", country="SE")
                for t in tickers}
    decision = DecisionRun(run_id="t", market_summary="m", actions=actions, cash_target_pct=5)
    return apply_guardrails(decision, snap, features, universe or set(features), cfg)


def real_book_cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.risk.min_cash_pct, cfg.risk.max_cash_pct = 5, 10
    cfg.risk.max_positions, cfg.risk.max_sector_pct = 12, 1000
    return cfg


# ── Swaps ─────────────────────────────────────────────────────────────────────

def test_a_swap_whose_buy_outranks_its_sell_gets_the_slot_the_sell_frees():
    full = PortfolioSnapshot([Position(f"P{i}", 79, 100, 100) for i in range(12)], 5200)
    result = guard([act("NEW", "buy", 8, .8), act("P0", "sell", 0, .6)], full, real_book_cfg())
    assert {(v.action.ticker, v.status) for v in result.verdicts} == {
        ("P0", "APPROVED"), ("NEW", "APPROVED")}


def test_a_swap_whose_buy_outranks_its_sell_is_funded_by_it():
    snap = PortfolioSnapshot([Position("OLD", 950, 100, 100)], 5000)  # 5% cash, at the floor
    result = guard([act("NEW", "buy", 8, .9), act("OLD", "sell", 85, .5)], snap, real_book_cfg())
    assert [a.ticker for a in result.approved_actions] == ["OLD", "NEW"]


def test_turnover_still_binds_once_sells_go_first():
    cfg = real_book_cfg()
    cfg.risk.max_turnover_pct = 10
    full = PortfolioSnapshot([Position(f"P{i}", 79, 100, 100) for i in range(12)], 5200)
    result = guard([act("NEW", "buy", 8, .8), act("P0", "sell", 0, .6)], full, cfg)
    assert [a.ticker for a in result.approved_actions] == ["P0"]
    assert "turnover" in next(v for v in result.verdicts if v.action.ticker == "NEW").rejection_reason


# ── Holdings outside the enabled universe ─────────────────────────────────────

def test_a_holding_outside_the_universe_can_be_sold_but_not_added_to():
    snap = PortfolioSnapshot([Position("HEM.ST", 100, 100, 100)], 90000)
    sold = guard([act("HEM.ST", "sell", 0, .7)], snap, AppConfig(), universe={"OTHER.ST"})
    assert sold.approved_actions and sold.approved_actions[0].sek_estimate == 10000
    added = guard([act("HEM.ST", "buy", 15, .7)], snap, AppConfig(), universe={"OTHER.ST"})
    assert added.approved_actions == []
    assert "not in universe" in added.verdicts[0].rejection_reason


def test_holding_rows_cover_disabled_and_unlisted_symbols(tmp_path):
    path = tmp_path / "universe.csv"
    path.write_text(HEADER + "Hemnet,HEM.ST,SE1,SE,OMXS,Tech,false\n"
                    "Alfa,ALFA.ST,SE2,SE,OMXS,Tech,true\n")
    rows = {t.yahoo_ticker: t for t in holding_rows(path, {"HEM.ST", "NOKIA.HE"})}
    assert rows["HEM.ST"].name == "Hemnet" and rows["HEM.ST"].currency == "SEK"
    assert rows["NOKIA.HE"].currency == "EUR"   # from the suffix, not defaulted to SEK
    assert "ALFA.ST" not in rows


# ── Missing FX rates ──────────────────────────────────────────────────────────

def test_dashboard_prices_leave_out_a_holding_with_no_rate(monkeypatch):
    from fundmgr.data import fx
    monkeypatch.setattr(fx, "rate_to_sek", lambda cur, store=None: None if cur == "EUR" else 1.5)
    out = fx.convert_prices_to_sek({"NOKIA.HE": 4.0, "NOVO-B.CO": 10.0, "ALFA.ST": 300.0},
                                   {"NOKIA.HE": "EUR", "NOVO-B.CO": "DKK", "ALFA.ST": "SEK"})
    assert out == {"NOVO-B.CO": 15.0, "ALFA.ST": 300.0}


def test_stop_scan_skips_a_holding_it_cannot_convert(tmp_path, monkeypatch):
    from fundmgr.engine.stop_review import find_stop_breaches
    path = tmp_path / "universe.csv"
    path.write_text(HEADER + "Nokia,NOKIA.HE,FI1,FI,OMXH,Tech,true\n")
    store = Store(tmp_path / "fund.db")
    store.initialise(100000)
    store.upsert_position("NOKIA.HE", 1000, 45.0)        # SEK cost; the quote is ~4 EUR
    store.set_position_stop("NOKIA.HE", stop_pct=10)
    monkeypatch.setattr("fundmgr.data.quotes.live_price", lambda t: 4.1)
    monkeypatch.setattr("fundmgr.data.fx.rate_to_sek", lambda cur, store=None: None)
    scan = find_stop_breaches(store, AppConfig(fx_to_sek=True, universe_path=path))
    assert scan["breaches"] == []
    assert scan["skipped"] == [{"ticker": "NOKIA.HE", "reason": "no EUR/SEK rate"}]


def test_a_review_refuses_to_price_a_holding_it_cannot_convert(monkeypatch):
    from fundmgr.engine.review_common import live_price_sek
    monkeypatch.setattr("fundmgr.data.fx.rate_to_sek", lambda cur, store=None: None)
    with pytest.raises(ValueError, match="EUR/SEK"):
        live_price_sek("NOKIA.HE", Position("NOKIA.HE", 1000, 45.0), AppConfig(fx_to_sek=True),
                       None, "EUR", None, 4.1)


# ── The weekly run ────────────────────────────────────────────────────────────

@pytest.fixture
def weekly(tmp_path, monkeypatch):
    """A weekly run with every network step stubbed; records what reached the model."""
    path = tmp_path / "universe.csv"
    path.write_text(HEADER + "Alfa,ALFA.ST,SE2,SE,OMXS,Tech,true\n"
                    "Nokia,NOKIA.HE,FI1,FI,OMXH,Tech,true\n"
                    "Hemnet,HEM.ST,SE1,SE,OMXS,Tech,false\n")
    cfg = AppConfig(name="Test Nordic", fx_to_sek=True, db_path=tmp_path / "fund.db",
                    universe_path=path)
    store = Store(cfg.db_path)
    store.initialise(100000)
    seen: dict = {"telegram": [], "featured": None, "snap": None, "llm": 0}
    today = datetime.utcnow().strftime("%Y-%m-%d")

    def features(tickers, *_a):
        seen["featured"] = {t.yahoo_ticker for t in tickers}
        return {t.yahoo_ticker: TickerFeatures(t.yahoo_ticker, t.name, 100.0, today, 0,
                                               currency=t.currency, sector="Tech", country="SE")
                for t in tickers}

    def prompt(cfg_, snap, *_a, **_k):
        seen["snap"] = snap
        return "system", "user", {}

    def model(*_a):
        seen["llm"] += 1
        raise LLMError("stopped by the test")

    monkeypatch.setattr(commands, "_get_store", lambda cfg_=None: (cfg, store))
    monkeypatch.setattr(commands, "fetch_and_cache_prices",
                        lambda tickers, *_a: {t.yahoo_ticker: True for t in tickers})
    monkeypatch.setattr(commands, "fetch_and_cache_benchmark", lambda *_a, **_k: True)
    monkeypatch.setattr(commands, "build_all_features", features)
    monkeypatch.setattr(commands, "apply_to_features", lambda *_a: None)
    monkeypatch.setattr(commands, "attach_sentiment_to_features", lambda *_a, **_k: None)
    monkeypatch.setattr(commands, "build_prompt", prompt)
    monkeypatch.setattr(commands, "call_llm_consensus", model)
    monkeypatch.setattr("fundmgr.notify.send.send_telegram",
                        lambda text, **_k: seen["telegram"].append(text) or True)
    monkeypatch.setattr("fundmgr.data.fx.rate_to_sek",
                        lambda cur, store=None: 1.0 if cur == "SEK" else 11.0)

    def go():
        return CliRunner().invoke(commands.cli, ["run", "--force", "--skip-news",
                                                 "--skip-macro", "--skip-fundamentals"])
    return store, seen, go, monkeypatch


def test_the_run_stops_before_the_model_when_a_rate_is_missing(weekly):
    store, seen, go, monkeypatch = weekly
    store.upsert_position("NOKIA.HE", 100, 450.0)
    monkeypatch.setattr("fundmgr.data.fx.rate_to_sek", lambda cur, store=None: None)
    result = go()
    assert result.exit_code == 1
    assert seen["snap"] is None and seen["llm"] == 0
    assert store.get_nav_history() == []
    assert "NOKIA.HE (no EUR/SEK rate)" in seen["telegram"][-1]


def test_the_run_prices_a_holding_whose_row_was_disabled(weekly):
    store, seen, go, monkeypatch = weekly
    store.upsert_position("HEM.ST", 100, 90.0)
    go()
    assert "HEM.ST" in seen["featured"]
    assert seen["llm"] == 1
    held = {p.ticker: p for p in seen["snap"].positions}
    assert held["HEM.ST"].current_price_sek == 100.0
