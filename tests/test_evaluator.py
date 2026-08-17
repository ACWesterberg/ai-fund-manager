import json
from datetime import datetime, timedelta

import pytest

from fundmgr.engine.evaluator import (
    _batch_review_message,
    evaluate_pending_outcomes,
    generate_qualitative_learnings,
    parse_batch_lessons,
)
from fundmgr.state.models import DecisionOutcome, RecommendationLog
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


def _outcome(ticker, run_id="r1", ret=5.0, bench=1.0, conf=0.6, action="buy"):
    return DecisionOutcome(
        run_id=run_id, ticker=ticker, action=action, confidence=conf,
        position_return_pct=ret, benchmark_return_pct=bench,
        outperformed=ret > bench, thesis="t",
    )


# ── batch distillation ────────────────────────────────────────────────────────

def test_batch_lesson_needs_more_than_one_ticker():
    """A lesson resting on a single 28-day return is the failure mode, not a lesson."""
    outcomes = [_outcome("AAA.ST"), _outcome("BBB.ST"), _outcome("CCC.ST")]
    content = json.dumps({"lessons": [
        {"body": "Only AAA moved on the thesis.", "tickers": ["AAA.ST"]},
        {"body": "Both tanker names tracked freight rates.", "tickers": ["AAA.ST", "BBB.ST"]},
    ]})
    kept = parse_batch_lessons(content, outcomes, max_lessons=3)
    assert [b for b, _ in kept] == ["Both tanker names tracked freight rates."]
    assert kept[0][1] == {"AAA.ST", "BBB.ST"}


def test_batch_lessons_drop_tickers_outside_the_batch():
    outcomes = [_outcome("AAA.ST"), _outcome("BBB.ST")]
    content = json.dumps({"lessons": [
        # Only one of the two cited tickers was actually in this batch.
        {"body": "Hallucinated support.", "tickers": ["AAA.ST", "ZZZ.ST"]},
    ]})
    assert parse_batch_lessons(content, outcomes, max_lessons=3) == []


def test_batch_lessons_accept_empty_and_malformed():
    outcomes = [_outcome("AAA.ST"), _outcome("BBB.ST")]
    assert parse_batch_lessons(json.dumps({"lessons": []}), outcomes, 3) == []
    assert parse_batch_lessons("not json", outcomes, 3) == []
    assert parse_batch_lessons(json.dumps({}), outcomes, 3) == []


def test_batch_lessons_respect_max():
    outcomes = [_outcome(t) for t in ("AAA.ST", "BBB.ST", "CCC.ST")]
    pair = ["AAA.ST", "BBB.ST"]
    content = json.dumps({"lessons": [{"body": f"L{i}", "tickers": pair} for i in range(5)]})
    assert len(parse_batch_lessons(content, outcomes, max_lessons=2)) == 2


def test_batch_review_message_uses_the_funds_own_benchmark():
    by_run = {"r1": [_outcome("AAA.ST", ret=8.0, bench=2.0)]}
    msg = _batch_review_message(by_run, {"r1": "oil up"}, "URTH", max_lessons=3)
    assert "URTH" in msg and "OMXSPI" not in msg
    # The shared-macro trap is stated where the model will read it.
    assert "SHARED" in msg
    assert "alpha +6.0pp" in msg


def test_batch_review_message_reports_dispersion():
    by_run = {"r1": [
        _outcome("AAA.ST", ret=8.0, bench=2.0),   # +6pp
        _outcome("BBB.ST", ret=-4.0, bench=2.0),  # -6pp
    ]}
    msg = _batch_review_message(by_run, {"r1": ""}, "^OMXSPI", max_lessons=3)
    assert "2 trades, 1 beat" in msg
    assert "mean alpha +0.0pp" in msg and "spread 12.0pp" in msg


def test_generate_qualitative_learnings_without_api_key(store, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert generate_qualitative_learnings(store, [_outcome("AAA.ST")]) == []
    assert generate_qualitative_learnings(store, []) == []


def test_evaluation_skips_outcome_without_decision_price(store):
    _save_run(store, "r1", days_ago=40)
    store.seed_outcomes_for_run(
        "r1",
        json.dumps([{"ticker": "AAA.ST", "side": "buy", "confidence": 0.5, "thesis": "t"}]),
        prices=None,  # unknown price → NULL → skipped, no network call
    )
    assert evaluate_pending_outcomes(store, lookback_days=28) == []
