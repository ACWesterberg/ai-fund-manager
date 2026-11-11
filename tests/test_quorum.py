"""A majority is of the samples requested, never of the ones that came back."""
from __future__ import annotations

import pytest

from fundmgr.config import AppConfig
from fundmgr.engine import client
from fundmgr.engine.client import LLMError, _aggregate_decisions, call_llm_consensus
from fundmgr.engine.review_common import majority, no_majority
from fundmgr.engine.schema import Action, DecisionRun, StopReview
from fundmgr.engine.stop_review import _vote as stop_vote


def action(ticker: str, side: str = "buy") -> Action:
    return Action(ticker=ticker, side=side, target_weight_pct=10 if side == "buy" else 0,
                  sek_estimate=10000, confidence=.8, thesis="Test")


def decision(*actions: Action) -> DecisionRun:
    return DecisionRun(run_id="r", market_summary="Test", actions=list(actions), cash_target_pct=15)


def review(rec: str, conf: float = .8, trim: float | None = None) -> StopReview:
    return StopReview(ticker="X", recommendation=rec, confidence=conf, trim_pct=trim,
                      what_changed="changed", rationale="because")


def test_two_survivors_of_five_are_not_a_majority():
    runs = [decision(action("A")), decision(action("A")), decision(action("B"))]
    consensus, votes = _aggregate_decisions(runs, requested=5)
    assert consensus.actions == []
    assert votes == {}
    consensus, votes = _aggregate_decisions(runs)
    assert [a.ticker for a in consensus.actions] == ["A"]


def test_three_agreeing_survivors_of_five_still_pass():
    runs = [decision(action("A"))] * 3 + [decision(action("B"))]
    consensus, votes = _aggregate_decisions(runs, requested=5)
    assert votes == {"A": 3}


def _patch_calls(monkeypatch, ok: int, failed: int) -> None:
    outcomes = [True] * ok + [False] * failed

    def call(system, user, cfg):
        if outcomes.pop():
            return decision(action("A")), "{}"
        raise LLMError("rate limited")

    monkeypatch.setattr(client, "call_llm", call)


@pytest.mark.parametrize("ok", [1, 2])
def test_a_minority_of_samples_raises_instead_of_deciding(monkeypatch, ok):
    _patch_calls(monkeypatch, ok=ok, failed=5 - ok)
    cfg = AppConfig()
    cfg.llm.n_samples = 5
    with pytest.raises(LLMError, match=f"Only {ok}/5"):
        call_llm_consensus("s", "u", cfg)


def test_a_majority_of_samples_still_decides(monkeypatch):
    _patch_calls(monkeypatch, ok=3, failed=2)
    cfg = AppConfig()
    cfg.llm.n_samples = 5
    consensus, _raw, votes, sampling = call_llm_consensus("s", "u", cfg)
    assert votes == {"A": 3}
    assert sampling["failed"] == 2


def test_a_split_review_holds_whatever_order_the_samples_arrived_in():
    samples = [review("exit"), review("exit"), review("hold"), review("hold"), review("trim", trim=50)]
    for order in (samples, samples[::-1], samples[2:] + samples[:2]):
        winner, agreeing, counts = majority(order, 5)
        assert winner == "hold"
        assert all(r.recommendation == "hold" for r in agreeing)
        assert no_majority(winner, counts, 5)


def test_one_survivor_of_five_is_not_a_consensus_exit():
    consensus, votes = stop_vote([review("exit", trim=None)], requested=5)
    assert consensus.recommendation == "hold"
    assert consensus.what_changed.startswith("[No majority (exit×1 of 5)")


def test_a_fallback_hold_drops_the_trim_size():
    consensus, _ = stop_vote([review("trim", trim=40), review("trim", trim=40), review("exit")], requested=5)
    assert consensus.recommendation == "hold"
    assert consensus.trim_pct is None


def test_a_real_majority_is_unchanged():
    consensus, votes = stop_vote([review("trim", trim=40)] * 3 + [review("hold")] * 2, requested=5)
    assert consensus.recommendation == "trim"
    assert consensus.trim_pct == 40
    assert votes == {"trim": 3, "hold": 2}
