"""
Tests for the take-profit review — the decision made when a position reaches
its target.

Before this existed, a target hit produced the words "consider trimming" and
nothing else: no verdict, and no way to move the level, so the same target
re-alerted indefinitely. A stop breach, on the other side of the trade, has
always got a full consensus review. These cover the symmetric path.

Run: pytest tests/test_target_review.py -v
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

# See tests/test_target_ratchet.py — the engine package reaches the shared
# financedata install transitively; stub it only when genuinely absent.
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

from fundmgr.engine.review_common import majority, votes_str  # noqa: E402
from fundmgr.engine.schema import TargetReview  # noqa: E402
from fundmgr.engine.target_review import (  # noqa: E402
    _detail,
    _vote,
    apply_new_target,
    format_review_text,
)
from fundmgr.state.store import Store  # noqa: E402


def _review(rec="raise", conf=0.8, new_tp=None, trim=None, ticker="TRUE-B.ST"):
    return TargetReview(
        ticker=ticker,
        recommendation=rec,
        confidence=conf,
        trim_pct=trim,
        new_take_profit_pct=new_tp,
        what_changed="Momentum intact, forward multiple still undemanding.",
        rationale="Earnings revisions continue upward.",
    )


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "test.db"))


# ── The four verdicts ─────────────────────────────────────────────────────────

def test_schema_offers_sell_trim_raise_hold():
    """'raise' is the verb the old flow lacked — holding on with a stated level."""
    for rec in ("sell", "trim", "raise", "hold"):
        assert _review(rec=rec, new_tp=50.0).recommendation == rec


def test_a_new_target_below_the_old_one_is_ignored(store):
    """A weak sample must not quietly lower the bar the position is measured against."""
    store.set_position_stop("TRUE-B.ST", stop_pct=12.0, take_profit_pct=33.0)
    assert apply_new_target(_review("raise", new_tp=25.0), store) is None
    assert store.get_effective_stops()["TRUE-B.ST"]["take_profit_pct"] == 33.0


def test_an_equal_target_is_ignored(store):
    store.set_position_stop("TRUE-B.ST", stop_pct=12.0, take_profit_pct=33.0)
    assert apply_new_target(_review("raise", new_tp=33.0), store) is None


def test_a_raise_is_written_and_reports_the_move(store):
    store.set_position_stop("TRUE-B.ST", stop_pct=12.0, take_profit_pct=33.0)
    moved = apply_new_target(_review("raise", new_tp=50.0), store)
    assert moved == (33.0, 50.0)
    levels = store.get_effective_stops()["TRUE-B.ST"]
    assert levels["take_profit_pct"] == 50.0
    assert levels["stop_pct"] == 12.0, "raising the target must not drop the stop"


def test_a_trim_also_re_targets_the_remainder(store):
    store.set_position_stop("TRUE-B.ST", stop_pct=12.0, take_profit_pct=33.0)
    moved = apply_new_target(_review("trim", new_tp=55.0, trim=40.0), store)
    assert moved == (33.0, 55.0)


def test_sell_never_writes_a_target(store):
    """The level is irrelevant once the position is meant to be closed."""
    store.set_position_stop("TRUE-B.ST", stop_pct=12.0, take_profit_pct=33.0)
    assert apply_new_target(_review("sell", new_tp=99.0), store) is None
    assert store.get_effective_stops()["TRUE-B.ST"]["take_profit_pct"] == 33.0


def test_hold_never_writes_a_target(store):
    store.set_position_stop("TRUE-B.ST", stop_pct=12.0, take_profit_pct=33.0)
    assert apply_new_target(_review("hold", new_tp=99.0), store) is None


def test_raise_without_a_level_changes_nothing(store):
    """A malformed sample that says 'raise' but names no level must be inert."""
    store.set_position_stop("TRUE-B.ST", stop_pct=12.0, take_profit_pct=33.0)
    assert apply_new_target(_review("raise", new_tp=None), store) is None
    assert store.get_effective_stops()["TRUE-B.ST"]["take_profit_pct"] == 33.0


# ── Consensus ─────────────────────────────────────────────────────────────────

def test_majority_wins_and_confidence_averages_over_agreers():
    reviews = [
        _review("raise", conf=0.9, new_tp=50.0),
        _review("raise", conf=0.7, new_tp=45.0),
        _review("sell", conf=0.99),
    ]
    consensus, votes = _vote(reviews)
    assert consensus.recommendation == "raise"
    assert consensus.confidence == pytest.approx(0.8)
    assert votes == {"raise": 2, "sell": 1}


def test_new_target_is_the_median_not_the_outlier():
    """One sample arguing for a wild level must not set it alone."""
    reviews = [
        _review("raise", conf=0.8, new_tp=45.0),
        _review("raise", conf=0.8, new_tp=50.0),
        _review("raise", conf=0.9, new_tp=250.0),
    ]
    consensus, _ = _vote(reviews)
    assert consensus.new_take_profit_pct == 50.0


def test_dissenters_do_not_contribute_their_target():
    """Only the winning camp's levels count toward the median."""
    reviews = [
        _review("raise", conf=0.8, new_tp=40.0),
        _review("raise", conf=0.8, new_tp=40.0),
        _review("trim", conf=0.9, new_tp=200.0),
    ]
    consensus, _ = _vote(reviews)
    assert consensus.recommendation == "raise"
    assert consensus.new_take_profit_pct == 40.0


def test_unanimous_sell_carries_no_target():
    consensus, votes = _vote([_review("sell", conf=0.9), _review("sell", conf=0.8)])
    assert consensus.recommendation == "sell"
    assert consensus.new_take_profit_pct is None
    assert votes == {"sell": 2}


# ── Reporting the raise ───────────────────────────────────────────────────────

def test_output_states_the_target_move():
    """The user asked to be told when a target got raised — so it must be visible."""
    out = format_review_text(_review("raise", new_tp=50.0), {"raise": 3}, 3, moved=(33.0, 50.0))
    assert "+33% → +50%" in out
    assert "RAISE" in out


def test_a_proposed_but_rejected_target_says_so():
    """A level that didn't clear the standing one must not read as applied."""
    detail = _detail(_review("raise", new_tp=25.0), moved=None)
    assert "unchanged" in detail
    assert "proposed +25%" in detail


def test_trim_reports_both_the_sale_and_the_new_level():
    detail = _detail(_review("trim", new_tp=55.0, trim=40.0), moved=(33.0, 55.0))
    assert "sell 40%" in detail
    assert "+33% → +55%" in detail


def test_sell_output_has_no_target_noise():
    assert _detail(_review("sell"), moved=None) == ""


# ── Shared plumbing (first coverage for the stop review's voting too) ─────────

def test_majority_picks_the_most_common():
    reviews = [_review("sell"), _review("sell"), _review("raise")]
    winner, agreeing, counts = majority(reviews)
    assert winner == "sell"
    assert len(agreeing) == 2
    assert counts == {"sell": 2, "raise": 1}


def test_votes_str_is_ordered_by_count():
    assert votes_str({"raise": 1, "sell": 2}, 3) == "sell×2/raise×1 of 3"
