"""What the learning loop is built on: a real quorum, every sample kept, one
currency per outcome, and calibration bands that match the model's own."""
from __future__ import annotations

import itertools
import json
import threading
from datetime import datetime, timedelta

import pytest

from fundmgr.config import AppConfig
from fundmgr.engine import client
from fundmgr.engine.client import LLMError, call_llm_consensus
from fundmgr.engine.evaluator import evaluate_pending_outcomes
from fundmgr.engine.schema import Action, DecisionRun
from fundmgr.state.models import RecommendationLog
from fundmgr.state.store import Store, conviction_band


def _iso(days_ago: int) -> str:
    return (datetime.utcnow() - timedelta(days=days_ago)).strftime("%Y-%m-%d")


# ── Quorum and samples ────────────────────────────────────────────────────────

def decision(*tickers: str) -> DecisionRun:
    return DecisionRun(run_id="r", market_summary="m", cash_target_pct=5, actions=[
        Action(ticker=t, side="buy", target_weight_pct=8, sek_estimate=0, confidence=.7,
               thesis="t") for t in tickers])


@pytest.fixture
def samples(monkeypatch):
    """call_llm returns the queued answers in order; an LLMError entry fails that call."""
    queue: list = []
    lock, counter = threading.Lock(), itertools.count()

    def fake(system, user, cfg, schema=DecisionRun, **_k):
        with lock:
            answer = queue[next(counter)]
        if isinstance(answer, Exception):
            raise answer
        return answer, answer.model_dump_json()

    monkeypatch.setattr(client, "call_llm", fake)
    cfg = AppConfig()
    cfg.llm.n_samples = 5
    return queue, cfg


def test_two_of_five_samples_are_not_a_decision(samples):
    queue, cfg = samples
    queue += [decision("A.ST"), decision("A.ST")] + [LLMError("parse")] * 3
    with pytest.raises(LLMError, match="Only 2 of 5 samples came back; a decision needs 3"):
        call_llm_consensus("s", "u", cfg)


def test_a_failed_sample_does_not_lower_the_bar(samples):
    """3 of 5 back: a ticker still needs 3 votes — a majority of those asked for."""
    queue, cfg = samples
    queue += [decision("A.ST", "B.ST"), decision("A.ST", "B.ST"), decision("A.ST"),
              LLMError("parse"), LLMError("timeout")]
    consensus, _raw, votes, sampling = call_llm_consensus("s", "u", cfg)
    assert [a.ticker for a in consensus.actions] == ["A.ST"]       # B had 2 of 5
    assert votes == {"A.ST": 3}
    assert sampling["succeeded"] == 3 and len(sampling["samples"]) == 3
    assert sorted(len(s["actions"]) for s in sampling["samples"]) == [1, 2, 2]


def test_every_sample_is_kept_for_the_record(samples):
    queue, cfg = samples
    queue += [decision("A.ST")] * 4 + [decision("B.ST")]
    _c, _r, _v, sampling = call_llm_consensus("s", "u", cfg)
    kept = sorted(s["actions"][0]["ticker"] for s in sampling["samples"])
    assert kept == ["A.ST"] * 4 + ["B.ST"]
    json.dumps(sampling)                                            # stored as JSON


# ── Calibration ───────────────────────────────────────────────────────────────

def test_conviction_bands_are_the_schemas():
    assert [conviction_band(c) for c in (0.80, 0.75, 0.74, 0.55, 0.54, 0.40, None)] == [
        "high", "high", "medium", "medium", "low", "low", "low"]


def _buy(store, run_id, ticker, day, conf=0.8, won=True):
    with store._conn() as conn:
        conn.execute(
            "INSERT INTO decision_outcomes (run_id, ticker, action, confidence, outperformed, "
            "source, decision_date, horizon_days) VALUES (?, ?, 'buy', ?, ?, 'run', ?, 28)",
            (run_id, ticker, conf, 1 if won else 0, day))


def test_weekly_repeats_of_one_buy_count_once_per_window(tmp_path):
    store = Store(tmp_path / "f.db")
    for week in range(4):                                  # 0, 7, 14, 21 days in: one look
        _buy(store, f"r{week}", "AAA.ST", f"2026-09-{1 + 7 * week:02d}")
    _buy(store, "r5", "AAA.ST", "2026-10-06")              # 35 days on: a new look
    _buy(store, "r0", "BBB.ST", "2026-09-01")
    stats = store.get_calibration_stats()
    assert stats["high"]["n"] == 3
    assert store.get_calibration_run_ids({"high"}) == ["r0", "r5"]


# ── One currency per outcome ──────────────────────────────────────────────────

@pytest.fixture
def eur_outcome(tmp_path):
    """A EUR stock up 10% in EUR while the euro fell 9% against the krona."""
    store = Store(tmp_path / "f.db")
    decided = _iso(40)
    target = (datetime.strptime(decided, "%Y-%m-%d") + timedelta(days=28)).strftime("%Y-%m-%d")
    store.save_recommendation(RecommendationLog(
        run_id="r1", timestamp=datetime.utcnow() - timedelta(days=40),
        prompt_snapshot=json.dumps({"user_message": ""}), llm_response="{}",
        guardrail_log="{}", actions_json="[]"))
    store.seed_outcomes_for_run("r1", json.dumps([{
        "ticker": "NOKIA.HE", "side": "buy", "confidence": .7, "sek_estimate": 9000.0,
        "thesis": "t"}]), prices={"NOKIA.HE": 10.0})

    def rows(*points):
        return [{"date": d, "open": 0, "high": 0, "low": 0, "close": c, "volume": 0} for d, c in points]

    store.save_prices("NOKIA.HE", rows((decided, 10.0), (target, 11.0)))
    store.save_benchmark([{"date": decided, "close": 1000.0}, {"date": target, "close": 1030.0}])
    return store, rows, decided, target


def test_a_foreign_outcome_is_scored_in_krona(eur_outcome):
    store, rows, decided, target = eur_outcome
    store.save_prices("EURSEK=X", rows((decided, 11.0), (target, 10.0)))
    [o] = evaluate_pending_outcomes(store, currency_of={"NOKIA.HE": "EUR"})
    assert o.position_return_pct == pytest.approx(0.0)     # 1.10 × 10/11 − 1
    assert o.outperformed is False                         # the index rose 3%


def test_a_foreign_outcome_without_its_rate_stays_pending(eur_outcome):
    store, *_ = eur_outcome
    assert evaluate_pending_outcomes(store, currency_of={"NOKIA.HE": "EUR"}) == []
    [o] = evaluate_pending_outcomes(store)                  # a native-currency sim scores it
    assert o.position_return_pct == pytest.approx(10.0)
