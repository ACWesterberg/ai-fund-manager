import json
from datetime import datetime, timedelta, timezone

import pytest

from fundmgr.config import AppConfig, default_heavy_model
from fundmgr.engine.evaluator import (
    _batch_review_message,
    calibration_body,
    evaluate_pending_outcomes,
    generate_learnings,
    generate_qualitative_learnings,
    surviving_lessons,
    wilson_interval,
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


# ── Calibration lesson ────────────────────────────────────────────────────────

def _stats(**buckets) -> dict:
    """buckets: bucket=(hits, n)"""
    return {
        b: {"n": n, "hits": hits, "hit_rate": hits / n if n else None}
        for b, (hits, n) in buckets.items()
    }


def test_wilson_interval_is_wide_at_small_n():
    """2-of-5 cannot distinguish a bad strategy from a good one."""
    lo, hi = wilson_interval(2, 5)
    assert lo < 0.15 and hi > 0.75
    # The same rate over 100 decisions is a real finding.
    lo, hi = wilson_interval(40, 100)
    assert lo > 0.30 and hi < 0.51


def test_wilson_interval_stays_inside_zero_one():
    assert wilson_interval(0, 5)[0] == 0.0
    assert wilson_interval(5, 5)[1] == 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_no_calibration_lesson_below_the_sample_bar():
    """The old text made a directive claim off 5 decisions."""
    assert calibration_body(_stats(high=(2, 5), low=(3, 6))) is None


def test_calibration_body_reports_intervals_not_a_breakeven_claim():
    body = calibration_body(_stats(high=(11, 25), low=(14, 25)))
    assert "95% CI" in body and "n=25" in body
    # No invented threshold, and no advice that the statistic cannot support.
    assert "breakeven" not in body.lower()
    assert "stop-loss" not in body.lower() and "stop loss" not in body.lower()
    # States what a hit rate cannot tell you.
    assert "not by how much" in body


def test_calibration_reports_every_qualifying_band():
    """The old chain was silent for the medium band and for whole ranges of the others."""
    body = calibration_body(_stats(high=(12, 25), medium=(13, 25), low=(11, 25)))
    for label in ("high (>=0.7)", "medium (0.4-0.7)", "low (<0.4)"):
        assert label in body


def test_calibration_verdict_calls_overlap_inconclusive():
    body = calibration_body(_stats(high=(12, 25), low=(13, 25)))
    assert "intervals overlap" in body
    assert "not yet separating" in body


def test_calibration_verdict_detects_informative_conviction():
    body = calibration_body(_stats(high=(24, 25), low=(3, 25)))
    assert "carrying real information" in body


def test_calibration_verdict_detects_inverted_conviction():
    body = calibration_body(_stats(high=(3, 25), low=(24, 25)))
    assert "inverted" in body


def test_calibration_single_band_makes_no_comparison_claim():
    body = calibration_body(_stats(high=(12, 25), low=(2, 4)))
    assert "Too few decisions in the other conviction bands" in body
    assert "n=4" not in body  # the under-powered band is not reported at all


def _seed_buys(store, run_id, rows):
    """rows: (confidence, outperformed)"""
    import sqlite3
    with store._conn() as conn:
        for i, (conf, won) in enumerate(rows):
            conn.execute(
                "INSERT INTO decision_outcomes (run_id, ticker, action, confidence, "
                "outperformed, source) VALUES (?, ?, 'buy', ?, ?, 'run')",
                (run_id, f"T{i}.ST", conf, 1 if won else 0),
            )


def test_two_qualifying_bands_no_longer_delete_each_other(store):
    """Both bands keyed supersede on category='calibration' — the second killed the first."""
    _seed_buys(store, "r1", [(0.8, i % 2 == 0) for i in range(25)]
                          + [(0.2, i % 2 == 0) for i in range(25)])

    created = generate_learnings(store)
    assert len(created) == 1
    active = store.get_active_learnings()
    assert len(active) == 1
    # One lesson carrying both readings, rather than one band silently lost.
    assert "high (>=0.7)" in active[0].body and "low (<0.4)" in active[0].body


def test_calibration_refresh_supersedes_the_previous_reading(store):
    _seed_buys(store, "r1", [(0.8, i % 2 == 0) for i in range(25)])
    first = generate_learnings(store)
    assert len(first) == 1

    # Same data again — the reading has not changed, so nothing is rewritten.
    assert generate_learnings(store) == []
    assert len(store.get_active_learnings()) == 1

    # New outcomes shift the rate: one active lesson, the old one superseded.
    _seed_buys(store, "r2", [(0.8, True) for _ in range(25)])
    second = generate_learnings(store)
    assert len(second) == 1
    assert len(store.get_active_learnings()) == 1
    assert store.get_active_learnings()[0].body == second[0].body


def test_unsupported_calibration_claim_is_retired(store):
    """A standing claim is in every prompt — it must not outlive its evidence."""
    from fundmgr.state.models import Learning

    store.save_learning(Learning(
        category="calibration",
        body="Your high-confidence buys have a 40% hit rate over 5 decisions.",
        created_at=datetime(2026, 8, 17, tzinfo=timezone.utc),
    ))
    _seed_buys(store, "r1", [(0.8, True), (0.8, False)])  # nowhere near the bar

    assert generate_learnings(store) == []
    assert store.get_active_learnings() == []
