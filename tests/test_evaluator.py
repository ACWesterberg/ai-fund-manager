import json
from datetime import datetime, timedelta

import pytest

from fundmgr.engine.evaluator import evaluate_pending_outcomes
from fundmgr.state.models import RecommendationLog
from fundmgr.state.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "test.db")


def _iso(days_ago: int) -> str:
    return (datetime.utcnow() - timedelta(days=days_ago)).strftime("%Y-%m-%d")


def _save_run(store, run_id, days_ago):
    ts = datetime.utcnow() - timedelta(days=days_ago)
    store.save_recommendation(RecommendationLog(
        run_id=run_id, timestamp=ts,
        prompt_snapshot=json.dumps({"user_message": ""}),
        llm_response="{}", guardrail_log="{}", actions_json="[]",
    ))


def test_close_near_picks_nearest_within_tolerance(store):
    store.save_prices("AAA.ST", [
        {"date": "2026-06-01", "open": 0, "high": 0, "low": 0, "close": 100.0, "volume": 0},
        {"date": "2026-06-29", "open": 0, "high": 0, "low": 0, "close": 112.0, "volume": 0},
    ])
    assert store.close_near("AAA.ST", "2026-06-28") == ("2026-06-29", 112.0)
    # 20 days away, default tolerance 7 → miss
    assert store.close_near("AAA.ST", "2026-06-15") is None


def test_evaluation_pinned_to_fixed_horizon(store):
    # Decision 40 days ago; the run evaluating it fires today, but the outcome
    # must be measured at decision + 28 days, not "now".
    decision = _iso(40)
    target = (datetime.strptime(decision, "%Y-%m-%d") + timedelta(days=28)).strftime("%Y-%m-%d")

    _save_run(store, "r1", days_ago=40)
    store.seed_outcomes_for_run(
        "r1",
        json.dumps([{"ticker": "AAA.ST", "side": "buy", "confidence": 0.7,
                     "sek_estimate": 9000.0, "thesis": "t"}]),
        prices={"AAA.ST": 100.0},
    )
    # Pinned close at the 28-day mark = 110; a much higher "live" close today = 200.
    store.save_prices("AAA.ST", [
        {"date": target, "open": 0, "high": 0, "low": 0, "close": 110.0, "volume": 0},
        {"date": _iso(0), "open": 0, "high": 0, "low": 0, "close": 200.0, "volume": 0},
    ])
    store.save_benchmark([
        {"date": decision, "close": 1000.0},
        {"date": target, "close": 1030.0},
        {"date": _iso(0), "close": 2000.0},  # after eval date — must be excluded
    ])

    evaluated = evaluate_pending_outcomes(store, lookback_days=28)
    assert len(evaluated) == 1
    o = evaluated[0]
    assert o.price_at_evaluation == pytest.approx(110.0)          # pinned, not the 200 live close
    assert o.evaluation_date == target
    assert o.position_return_pct == pytest.approx(10.0)           # 110 / 100 - 1
    assert o.benchmark_return_pct == pytest.approx(3.0)           # 1030 / 1000 within window
    assert o.outperformed is True


def test_evaluation_skips_outcome_without_decision_price(store):
    _save_run(store, "r1", days_ago=40)
    store.seed_outcomes_for_run(
        "r1",
        json.dumps([{"ticker": "AAA.ST", "side": "buy", "confidence": 0.5, "thesis": "t"}]),
        prices=None,  # unknown price → NULL → skipped, no network call
    )
    assert evaluate_pending_outcomes(store, lookback_days=28) == []
