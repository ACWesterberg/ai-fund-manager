"""
Tests for logging review verdicts as evaluable fund decisions.

A stop or take-profit review used to leave no trace: it was never scored, never
fed a learning, and never appeared in the "recent decisions on this name" block
of a later prompt. These cover the decision record it now writes — and, just as
importantly, the places it must stay out of.

Run: pytest tests/test_review_decisions.py -v
"""
from __future__ import annotations

import sys
import types
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

try:  # pragma: no cover - environment-dependent
    import financedata  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _stub = types.ModuleType("financedata")
    for _name in (
        "get_prices_since", "rsi", "pct_return", "ann_vol", "get_cache",
        "get_fundamentals", "ts_to_days",
    ):
        setattr(_stub, _name, lambda *a, **k: None)
    sys.modules["financedata"] = _stub

from fundmgr.engine.review_common import action_for_verdict, log_review  # noqa: E402
from fundmgr.engine.schema import StopReview, TargetReview  # noqa: E402
from fundmgr.state.models import DecisionOutcome, RecommendationLog  # noqa: E402
from fundmgr.state.store import Store  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "test.db"))


def _target(rec="raise", conf=0.8, ticker="TRUE-B.ST"):
    return TargetReview(
        ticker=ticker, recommendation=rec, confidence=conf,
        new_take_profit_pct=50.0 if rec in ("raise", "trim") else None,
        what_changed="Revisions still upward.", rationale="Momentum intact.",
    )


def _stop(rec="exit", conf=0.7, ticker="AAA.ST"):
    return StopReview(
        ticker=ticker, recommendation=rec, confidence=conf,
        what_changed="Guidance cut.", rationale="Thesis broken.",
    )


def _days_ago(n: int) -> str:
    return (datetime.utcnow() - timedelta(days=n)).strftime("%Y-%m-%d")


# ── Verdict → scored action ───────────────────────────────────────────────────

@pytest.mark.parametrize("verdict,action", [
    ("sell", "sell"),    # take-profit: bank it
    ("trim", "sell"),    # a partial sale is still a sale
    ("raise", "hold"),   # kept the position
    ("hold", "hold"),
    ("exit", "sell"),    # stop review
    ("add", "buy"),
])
def test_verdicts_map_to_the_scored_vocabulary(verdict, action):
    assert action_for_verdict(verdict) == action


def test_an_unknown_verdict_is_treated_as_a_hold():
    """Never guess a trade from a verb the evaluator doesn't know."""
    assert action_for_verdict("something-new") == "hold"


# ── The decision record ───────────────────────────────────────────────────────

def test_a_review_is_written_as_a_decision(store):
    log_review(_target("sell"), store, "target_review", native_price=20.81)
    rows = store.get_decisions_for_ticker("TRUE-B.ST")
    assert len(rows) == 1
    assert rows[0]["action"] == "sell"
    assert rows[0]["source"] == "target_review"
    assert "SELL" in rows[0]["thesis"]


def test_a_priceless_review_is_not_logged(store):
    """The evaluator skips outcomes with no decision price — don't write a dud."""
    assert log_review(_target(), store, "target_review", native_price=None) is None
    assert store.get_decisions_for_ticker("TRUE-B.ST") == []


def test_the_logged_price_is_the_one_passed_in(store):
    """Must be native: outcomes are scored against cached closes, which are native."""
    log_review(_target("sell"), store, "target_review", native_price=142.5)
    assert store.get_decisions_for_ticker("TRUE-B.ST")[0]["price_at_decision"] == 142.5


def test_re_reviewing_the_same_day_overwrites_rather_than_duplicates(store):
    log_review(_target("raise"), store, "target_review", native_price=20.0)
    log_review(_target("sell"), store, "target_review", native_price=21.0)
    rows = store.get_decisions_for_ticker("TRUE-B.ST")
    assert len(rows) == 1
    assert rows[0]["action"] == "sell", "the later verdict stands"


def test_stop_and_target_reviews_are_separate_records(store):
    log_review(_target("sell", ticker="AAA.ST"), store, "target_review", native_price=10.0)
    log_review(_stop("add", ticker="AAA.ST"), store, "stop_review", native_price=10.0)
    rows = store.get_decisions_for_ticker("AAA.ST", limit=5)
    assert {r["source"] for r in rows} == {"target_review", "stop_review"}


# ── It actually gets evaluated ────────────────────────────────────────────────

def test_a_review_decision_becomes_pending_for_evaluation(store):
    """
    The point of the whole change. Review rows have no recommendations row, so
    an inner join would drop them here and they would never be scored.
    """
    store.log_review_decision(
        ticker="TRUE-B.ST", action="sell", confidence=0.8, thesis="banked",
        price_at_decision=20.81, source="target_review", decision_date=_days_ago(30),
    )
    pending = store.get_pending_outcomes(older_than_days=28)
    assert [p.ticker for p in pending] == ["TRUE-B.ST"]
    assert pending[0].decision_date == _days_ago(30)


def test_a_recent_review_is_not_yet_due(store):
    store.log_review_decision(
        ticker="TRUE-B.ST", action="sell", confidence=0.8, thesis="banked",
        price_at_decision=20.81, source="target_review", decision_date=_days_ago(3),
    )
    assert store.get_pending_outcomes(older_than_days=28) == []


def test_run_decisions_still_evaluate_off_their_recommendation(store):
    """The LEFT JOIN must not break the existing path, which carries no decision_date."""
    store.save_recommendation(RecommendationLog(
        run_id="2026-07-01-abc", timestamp=datetime.utcnow() - timedelta(days=40),
        prompt_snapshot="{}", llm_response="{}", guardrail_log="[]",
        actions_json='[{"ticker":"AAA.ST","side":"buy"}]',
    ))
    store.seed_outcomes_for_run(
        "2026-07-01-abc",
        '[{"ticker":"AAA.ST","side":"buy","confidence":0.8,"thesis":"t"}]',
        prices={"AAA.ST": 100.0},
    )
    pending = store.get_pending_outcomes(older_than_days=28)
    assert [p.ticker for p in pending] == ["AAA.ST"]
    assert pending[0].decision_date is not None


def test_evaluation_results_survive_on_a_review_row(store):
    """update_outcome must not clear source/decision_date."""
    run_id = store.log_review_decision(
        ticker="TRUE-B.ST", action="sell", confidence=0.8, thesis="banked",
        price_at_decision=20.0, source="target_review", decision_date=_days_ago(30),
    )
    outcome = store.get_pending_outcomes(older_than_days=28)[0]
    outcome.price_at_evaluation = 18.0
    outcome.position_return_pct = -10.0
    outcome.benchmark_return_pct = 2.0
    outcome.outperformed = False
    outcome.evaluation_date = _days_ago(2)
    store.update_outcome(outcome)

    rows = store.get_decisions_for_ticker("TRUE-B.ST")
    assert rows[0]["source"] == "target_review"
    assert store.get_pending_outcomes(older_than_days=28) == [], "no longer pending"


def test_a_sell_that_beat_the_market_scored_badly():
    """Selling is 'correct' only if the stock then underperformed."""
    o = DecisionOutcome(run_id="r", ticker="T", action="sell", outperformed=True)
    assert o.was_correct is False
    o2 = DecisionOutcome(run_id="r", ticker="T", action="sell", outperformed=False)
    assert o2.was_correct is True


def test_a_raise_is_recorded_as_a_hold_and_left_ungraded():
    """Holds are deliberately ambiguous — a RAISE inherits that, like any hold."""
    o = DecisionOutcome(run_id="r", ticker="T", action="hold", outperformed=True)
    assert o.was_correct is None


# ── Where reviews must NOT appear ─────────────────────────────────────────────

def test_reviews_stay_out_of_the_calibration_buckets(store):
    """
    "Your high-confidence buy calls" must keep meaning the weekly calls. A stop
    review's ADD is a buy, but made under a different prompt about one name.
    """
    store.log_review_decision(
        ticker="AAA.ST", action="buy", confidence=0.9, thesis="add",
        price_at_decision=10.0, source="stop_review", decision_date=_days_ago(40),
    )
    outcome = store.get_pending_outcomes(older_than_days=28)[0]
    outcome.outperformed = True
    outcome.evaluation_date = _days_ago(1)
    store.update_outcome(outcome)

    assert store.get_calibration_stats() == {}, "review buys must not move the hit rate"


def test_reviews_stay_out_of_the_optimizer_training_set(store):
    store.log_review_decision(
        ticker="AAA.ST", action="sell", confidence=0.9, thesis="exit",
        price_at_decision=10.0, source="stop_review", decision_date=_days_ago(40),
    )
    outcome = store.get_pending_outcomes(older_than_days=28)[0]
    outcome.outperformed = True
    outcome.evaluation_date = _days_ago(1)
    store.update_outcome(outcome)

    assert store.get_evaluated_outcomes() == []


def test_reviews_do_not_touch_the_recommendations_table(store):
    """
    Writing there would put a single-ticker review into the dashboard's "Last
    Decision", the run counter, and the history list.
    """
    before = store.count_runs() if hasattr(store, "count_runs") else None
    log_review(_target("sell"), store, "target_review", native_price=20.0)
    assert store.get_last_recommendation() is None
    if before is not None:
        assert store.count_runs() == before
