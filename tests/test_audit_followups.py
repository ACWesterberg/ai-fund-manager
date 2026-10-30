"""Four follow-ups from the audit: full-exit slivers, split-rebased outcome
scores, cost-basis NAV in sleeves and accounts, and overlapping runs."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest
from click.testing import CliRunner

from fundmgr import cli as commands
from fundmgr.config import AppConfig
from fundmgr.state.models import NavPoint, RecommendationLog
from fundmgr.state.store import Store


def _iso(days_ago: int) -> str:
    return (datetime.utcnow() - timedelta(days=days_ago)).strftime("%Y-%m-%d")


# ── Full exits ────────────────────────────────────────────────────────────────

@pytest.fixture
def filler(tmp_path, monkeypatch):
    from fundmgr.engine import auto_fill
    store = Store(tmp_path / "book.db")
    store.initialise(1000)
    store.upsert_position("OLD", 100, 100)
    monkeypatch.setattr("fundmgr.config.load_universe", lambda *a: [])
    monkeypatch.setattr("fundmgr.data.market_hours.is_exchange_open", lambda *a: True)
    return auto_fill, store, AppConfig(auto_fill=True, fx_to_sek=False), monkeypatch


def test_a_full_exit_sells_every_share_after_the_price_rose(filler):
    auto_fill, store, cfg, monkeypatch = filler
    monkeypatch.setattr(auto_fill, "_fetch_price", lambda t: 102.0)  # budget was priced at 100
    auto_fill.execute_paper_fills(
        [{"ticker": "OLD", "side": "sell", "target_weight_pct": 0, "sek_estimate": 10000.0}],
        store, cfg)
    assert store.get_positions() == []
    assert store.get_transactions()[0].shares == 100


def test_a_partial_sell_still_keeps_to_its_budget(filler):
    auto_fill, store, cfg, monkeypatch = filler
    monkeypatch.setattr(auto_fill, "_fetch_price", lambda t: 100.0)
    auto_fill.execute_paper_fills(
        [{"ticker": "OLD", "side": "sell", "target_weight_pct": 5, "sek_estimate": 2600.0}],
        store, cfg)
    assert store.get_transactions()[0].gross_sek <= 2600


# ── Outcome scores across a split ─────────────────────────────────────────────

def _seed_outcome(store, price_at_decision: float, decision_close: float, eval_close: float):
    decision = _iso(40)
    target = (datetime.strptime(decision, "%Y-%m-%d") + timedelta(days=28)).strftime("%Y-%m-%d")
    store.save_recommendation(RecommendationLog(
        run_id="r1", timestamp=datetime.utcnow() - timedelta(days=40),
        prompt_snapshot=json.dumps({"user_message": ""}), llm_response="{}",
        guardrail_log="{}", actions_json="[]"))
    store.seed_outcomes_for_run("r1", json.dumps([{
        "ticker": "AAA.ST", "side": "buy", "confidence": .7, "sek_estimate": 9000.0,
        "thesis": "t"}]), prices={"AAA.ST": price_at_decision})
    store.save_prices("AAA.ST", [
        {"date": decision, "open": 0, "high": 0, "low": 0, "close": decision_close, "volume": 0},
        {"date": target, "open": 0, "high": 0, "low": 0, "close": eval_close, "volume": 0},
    ])
    store.save_benchmark([{"date": decision, "close": 1000.0}, {"date": target, "close": 1030.0}])


def test_a_split_inside_the_horizon_is_not_scored_as_a_loss(tmp_path):
    from fundmgr.engine.evaluator import evaluate_pending_outcomes
    store = Store(tmp_path / "f.db")
    # Decided at 200; a 2:1 split rebased the cached history to 100 and it ended at 110.
    _seed_outcome(store, price_at_decision=200.0, decision_close=100.0, eval_close=110.0)
    [o] = evaluate_pending_outcomes(store, lookback_days=28)
    assert o.position_return_pct == pytest.approx(10.0)
    assert o.outperformed is True


def test_a_dividend_adjustment_keeps_the_recorded_decision_price(tmp_path):
    from fundmgr.engine.evaluator import evaluate_pending_outcomes
    store = Store(tmp_path / "f.db")
    _seed_outcome(store, price_at_decision=100.0, decision_close=98.0, eval_close=110.0)
    [o] = evaluate_pending_outcomes(store, lookback_days=28)
    assert o.position_return_pct == pytest.approx(10.0)


# ── Sleeve and account NAV ────────────────────────────────────────────────────

def test_sleeve_nav_is_marked_at_market_not_cost(tmp_path):
    from fundmgr import paper
    store = Store(tmp_path / "s.db")
    store.initialise(1000)
    store.upsert_position("AAA.ST", 10, 100)
    store.save_prices("AAA.ST", [{"date": _iso(1), "open": 0, "high": 0, "low": 0,
                                  "close": 150.0, "volume": 0}])
    nav = paper.mark_nav(store, paper.cached_prices_sek(store, {"AAA.ST": "SEK"}))
    assert nav == pytest.approx(2500)
    assert store.get_nav_history()[-1].portfolio_nav_sek == pytest.approx(2500)


def test_sleeve_nav_is_left_alone_when_a_holding_has_no_price(tmp_path):
    from fundmgr import paper
    store = Store(tmp_path / "s.db")
    store.initialise(1000)
    store.upsert_position("AAA.ST", 10, 100)
    store.upsert_position("NEW.ST", 5, 200)              # just bought, nothing cached yet
    store.save_prices("AAA.ST", [{"date": _iso(1), "open": 0, "high": 0, "low": 0,
                                  "close": 150.0, "volume": 0}])
    store.upsert_nav(NavPoint(_iso(0), 3200.0, 100.0, 1000.0))
    assert paper.mark_nav(store, paper.cached_prices_sek(store, {"AAA.ST": "SEK",
                                                                 "NEW.ST": "SEK"})) is None
    assert store.get_nav_history()[-1].portfolio_nav_sek == 3200.0


# ── Overlapping runs ──────────────────────────────────────────────────────────

def test_a_second_run_of_the_same_fund_is_refused_while_one_holds_the_lock(tmp_path):
    cfg = AppConfig(db_path=tmp_path / "fund.db")
    first = commands._run_lock(cfg)
    assert first is not None
    assert commands._run_lock(cfg) is None
    first.close()
    again = commands._run_lock(cfg)
    assert again is not None
    again.close()


@pytest.mark.parametrize("hours_ago, starts", [(3, False), (30, True)])
def test_a_news_trigger_starts_no_run_within_a_day_of_the_last(tmp_path, monkeypatch,
                                                               hours_ago, starts):
    import subprocess
    cfg = AppConfig(db_path=tmp_path / "fund.db")
    cfg.data.news_feeds = ["https://example.invalid/rss"]
    store = Store(cfg.db_path)
    store.initialise(1000)
    store.upsert_position("AAA.ST", 10, 100)
    store.save_recommendation(RecommendationLog(
        run_id="r1", timestamp=datetime.utcnow() - timedelta(hours=hours_ago),
        prompt_snapshot="{}", llm_response="{}", guardrail_log="[]", actions_json="[]"))
    started = []
    monkeypatch.setattr(commands, "_get_store", lambda cfg_=None: (cfg, store))
    monkeypatch.setattr(commands, "get_enabled_tickers", lambda *_a: [])
    monkeypatch.setattr(commands, "check_news_triggers", lambda *_a, **_k: [{
        "ticker": "AAA.ST", "is_held": True, "sentiment_label": "negative",
        "sentiment_score": .95, "headline": "Profit warning"}])
    monkeypatch.setattr(subprocess, "run", lambda args, **_k: started.append(args))
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    result = CliRunner().invoke(commands.cli, ["check-news"])
    assert result.exit_code == 0, result.output
    assert bool(started) is starts
