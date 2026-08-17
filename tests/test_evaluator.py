import json
from datetime import datetime, timedelta

import pytest

from fundmgr.config import AppConfig, default_heavy_model
from fundmgr.engine.evaluator import (
    _batch_review_message,
    evaluate_pending_outcomes,
    generate_qualitative_learnings,
    surviving_lessons,
)
from fundmgr.engine.schema import BatchLessons
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

def _reply(*lessons) -> BatchLessons:
    return BatchLessons.model_validate({
        "lessons": [{"body": b, "tickers": t} for b, t in lessons]
    })


def test_batch_lesson_needs_more_than_one_ticker():
    """A lesson resting on a single 28-day return is the failure mode, not a lesson."""
    outcomes = [_outcome("AAA.ST"), _outcome("BBB.ST"), _outcome("CCC.ST")]
    parsed = _reply(
        ("Only AAA moved on the thesis.", ["AAA.ST"]),
        ("Both tanker names tracked freight rates.", ["AAA.ST", "BBB.ST"]),
    )
    kept = surviving_lessons(parsed, outcomes, max_lessons=3)
    assert [b for b, _ in kept] == ["Both tanker names tracked freight rates."]
    assert kept[0][1] == {"AAA.ST", "BBB.ST"}


def test_batch_lessons_drop_tickers_outside_the_batch():
    outcomes = [_outcome("AAA.ST"), _outcome("BBB.ST")]
    # Only one of the two cited tickers was actually in this batch.
    parsed = _reply(("Hallucinated support.", ["AAA.ST", "ZZZ.ST"]))
    assert surviving_lessons(parsed, outcomes, max_lessons=3) == []


def test_batch_lessons_accept_an_empty_reply():
    outcomes = [_outcome("AAA.ST"), _outcome("BBB.ST")]
    assert surviving_lessons(_reply(), outcomes, 3) == []
    assert surviving_lessons(BatchLessons(), outcomes, 3) == []


def test_batch_lessons_match_tickers_case_insensitively():
    outcomes = [_outcome("AAA.ST"), _outcome("BBB.ST")]
    parsed = _reply(("Lowercased by the model.", ["aaa.st", "bbb.st"]))
    assert surviving_lessons(parsed, outcomes, 3)[0][1] == {"AAA.ST", "BBB.ST"}


def test_batch_lessons_respect_max():
    outcomes = [_outcome(t) for t in ("AAA.ST", "BBB.ST", "CCC.ST")]
    parsed = _reply(*[(f"L{i}", ["AAA.ST", "BBB.ST"]) for i in range(5)])
    assert len(surviving_lessons(parsed, outcomes, max_lessons=2)) == 2


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


def test_generate_qualitative_learnings_no_outcomes_makes_no_call(store):
    assert generate_qualitative_learnings(store, [], cfg=AppConfig()) == []


def test_generate_qualitative_learnings_survives_an_llm_failure(store, monkeypatch):
    """A failed distillation must not take the run down — outcomes stay evaluated."""
    from fundmgr.engine import client as client_mod

    def _boom(*a, **k):
        raise client_mod.LLMError("no credentials")

    monkeypatch.setattr(client_mod, "call_llm", _boom)
    assert generate_qualitative_learnings(store, [_outcome("AAA.ST")], cfg=AppConfig()) == []


def test_learning_model_is_the_heavy_reasoner_not_the_decision_model():
    """The lesson writer conditions every future decision — it is not the cheap tier."""
    cfg = AppConfig()
    cfg.llm.provider, cfg.llm.model_id = "anthropic", "claude-haiku-4-5-20251001"
    assert cfg.learning_model == default_heavy_model("anthropic")

    cfg.llm.provider, cfg.llm.model_id = "openai", "gpt-4o-mini"
    assert cfg.learning_model == default_heavy_model("openai")


def test_learning_model_can_be_pinned_across_funds():
    cfg = AppConfig()
    cfg.learning_model_id = "gpt-5.6-sol"
    cfg.llm.provider = "anthropic"
    assert cfg.learning_model == "gpt-5.6-sol"


def test_evaluation_skips_outcome_without_decision_price(store):
    _save_run(store, "r1", days_ago=40)
    store.seed_outcomes_for_run(
        "r1",
        json.dumps([{"ticker": "AAA.ST", "side": "buy", "confidence": 0.5, "thesis": "t"}]),
        prices=None,  # unknown price → NULL → skipped, no network call
    )
    assert evaluate_pending_outcomes(store, lookback_days=28) == []
