"""Verification of the reasoning behind a decision, separately from its return."""
import json
from datetime import datetime, timedelta

import pytest

from fundmgr.config import AppConfig
from fundmgr.engine.schema import ThesisChecks
from fundmgr.engine.thesis_check import _review_message, verify_theses
from fundmgr.state.models import DecisionOutcome, RecommendationLog
from fundmgr.state.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "test.db")


def _seed(store, ticker="AAA.ST", thesis="Order intake recovers in Q3.", conf=0.7):
    ts = datetime.utcnow() - timedelta(days=40)
    store.save_recommendation(RecommendationLog(
        run_id="r1", timestamp=ts, prompt_snapshot="{}",
        llm_response="{}", guardrail_log="{}", actions_json="[]",
    ))
    store.seed_outcomes_for_run("r1", json.dumps([
        {"ticker": ticker, "side": "buy", "confidence": conf, "thesis": thesis}
    ]), prices={ticker: 100.0})
    return [o for o in store.get_all_outcomes() if o.ticker == ticker][0]


def _news(store, ticker, headline, days_ago=10):
    store.save_news_sentiment(ticker, [{
        "headline": headline, "summary": "", "source_url": "",
        "published_at": (datetime.utcnow() - timedelta(days=days_ago)).strftime("%Y-%m-%d"),
        "sentiment_label": "neutral", "sentiment_score": 0.0,
    }])


def _reply(*checks) -> ThesisChecks:
    return ThesisChecks.model_validate({
        "checks": [{"ticker": t, "verdict": v, "evidence": e} for t, v, e in checks]
    })


def _patch_call(monkeypatch, reply):
    from fundmgr.engine import thesis_check
    captured = {}

    def _fake(cfg, user_msg, horizon_days):
        captured["msg"] = user_msg
        captured["horizon"] = horizon_days
        return reply

    monkeypatch.setattr(thesis_check, "_call_for_checks", _fake)
    return captured


def test_verdict_is_persisted_and_counted(store, monkeypatch):
    outcome = _seed(store)
    _news(store, "AAA.ST", "Q3 order intake up 18% year on year")
    _patch_call(monkeypatch, _reply(("AAA.ST", "held", "Q3 order intake up 18%")))

    counts = verify_theses(store, [outcome], cfg=AppConfig())
    assert counts == {"held": 1}

    stored = [o for o in store.get_all_outcomes() if o.ticker == "AAA.ST"][0]
    assert stored.thesis_verdict == "held"
    assert "18%" in stored.thesis_evidence


def test_the_return_is_never_shown_to_the_auditor(store, monkeypatch):
    """Given the return, a model judges the thesis by it — the exact circularity
    this check exists to break."""
    outcome = _seed(store)
    outcome.position_return_pct = -12.4
    outcome.benchmark_return_pct = 1.1
    outcome.outperformed = False
    _news(store, "AAA.ST", "Company reports weak orders")
    captured = _patch_call(monkeypatch, _reply(("AAA.ST", "broke", "weak orders")))

    verify_theses(store, [outcome], cfg=AppConfig())
    msg = captured["msg"]
    assert "-12.4" not in msg and "12.4" not in msg
    assert "1.1" not in msg
    assert "benchmark" not in msg.lower()
    # It does get the thesis and the evidence.
    assert "Order intake recovers" in msg
    assert "weak orders" in msg


def test_no_news_means_no_verdict(store, monkeypatch):
    """With nothing to judge against, the model would fall back on general
    knowledge of the company — so it is not asked."""
    outcome = _seed(store)
    called = {"n": 0}

    from fundmgr.engine import thesis_check

    def _fake(cfg, user_msg, horizon_days):
        called["n"] += 1
        return _reply(("AAA.ST", "held", "invented"))

    monkeypatch.setattr(thesis_check, "_call_for_checks", _fake)

    assert verify_theses(store, [outcome], cfg=AppConfig()) == {}
    assert called["n"] == 0
    stored = [o for o in store.get_all_outcomes() if o.ticker == "AAA.ST"][0]
    assert stored.thesis_verdict is None


def test_decisions_without_a_thesis_are_not_judged(store, monkeypatch):
    outcome = _seed(store, thesis="")
    _news(store, "AAA.ST", "Something happened")
    _patch_call(monkeypatch, _reply(("AAA.ST", "held", "x")))
    assert verify_theses(store, [outcome], cfg=AppConfig()) == {}


def test_verdicts_for_tickers_outside_the_batch_are_dropped(store, monkeypatch):
    outcome = _seed(store)
    _news(store, "AAA.ST", "Orders up")
    _patch_call(monkeypatch, _reply(("ZZZ.ST", "held", "not in this batch")))

    assert verify_theses(store, [outcome], cfg=AppConfig()) == {}


def test_verification_survives_an_llm_failure(store, monkeypatch):
    outcome = _seed(store)
    _news(store, "AAA.ST", "Orders up")
    from fundmgr.engine import thesis_check
    monkeypatch.setattr(thesis_check, "_call_for_checks", lambda cfg, msg, horizon: None)
    assert verify_theses(store, [outcome], cfg=AppConfig()) == {}


def test_review_message_keeps_each_company_to_its_own_evidence():
    outcomes = [
        DecisionOutcome(run_id="r1", ticker="AAA.ST", action="buy", thesis="A recovers",
                        decision_date="2026-07-01"),
        DecisionOutcome(run_id="r1", ticker="BBB.ST", action="buy", thesis="B expands margin",
                        decision_date="2026-07-01"),
    ]
    evidence = {
        "AAA.ST": [{"headline": "A wins contract", "published_at": "2026-07-10", "summary": ""}],
        "BBB.ST": [{"headline": "B guides margin down", "published_at": "2026-07-12", "summary": ""}],
    }
    msg = _review_message(outcomes, evidence)
    a_section = msg.split("### BBB.ST")[0]
    assert "A wins contract" in a_section
    assert "B guides margin down" not in a_section


# ── the cross-tab ─────────────────────────────────────────────────────────────

def _outcome_row(store, ticker, verdict, outperformed):
    with store._conn() as conn:
        conn.execute(
            "INSERT INTO decision_outcomes (run_id, ticker, action, thesis_verdict, "
            "outperformed, source) VALUES ('r1', ?, 'buy', ?, ?, 'run')",
            (ticker, verdict, 1 if outperformed else 0),
        )


def test_thesis_stats_separate_luck_from_skill(store):
    _outcome_row(store, "A.ST", "held", True)     # right for the right reason
    _outcome_row(store, "B.ST", "held", False)    # right reasoning, wrong timing
    _outcome_row(store, "C.ST", "broke", True)    # lucky — looks like skill on return alone
    _outcome_row(store, "D.ST", "broke", False)   # simply wrong
    _outcome_row(store, "E.ST", "unresolved", True)

    stats = store.get_thesis_stats()
    assert stats["n"] == 5
    assert stats["cells"]["held"] == {"beat": 1, "lagged": 1}
    assert stats["cells"]["broke"] == {"beat": 1, "lagged": 1}
    # unresolved is excluded from the rate rather than counted as a failure.
    assert stats["resolved"] == 4
    assert stats["hold_rate"] == pytest.approx(0.5)


def test_thesis_stats_empty(store):
    stats = store.get_thesis_stats()
    assert stats["n"] == 0 and stats["hold_rate"] is None


# ── Per-fund evaluation horizon ───────────────────────────────────────────────

def test_audit_prompt_states_the_fund_s_own_horizon():
    """A 90-day fund must not be told its theses had four weeks to resolve."""
    from fundmgr.engine.thesis_check import _system_prompt

    assert "28 days after each was made" in _system_prompt(28)
    assert "90 days after each was made" in _system_prompt(90)
    assert "4-week window" in _system_prompt(28)
    assert "13-week window" in _system_prompt(90)


def test_audit_prompt_never_claims_reports_land_inside_the_window():
    """The old wording asserted margin theses are unresolvable — true at four
    weeks, false at a quarter, where a report usually does land."""
    for horizon in (28, 90):
        prompt = _system_prompt_text(horizon)
        assert "unless such a report actually fell inside" in prompt
        assert "Do not assume one did" in prompt


def _system_prompt_text(horizon):
    from fundmgr.engine.thesis_check import _system_prompt
    return _system_prompt(horizon)


def test_the_fund_s_horizon_reaches_the_auditor(store, monkeypatch):
    outcome = _seed(store)
    _news(store, "AAA.ST", "Orders up")
    captured = _patch_call(monkeypatch, _reply(("AAA.ST", "held", "orders up")))

    cfg = AppConfig()
    cfg.evaluation_horizon_days = 90
    verify_theses(store, [outcome], cfg=cfg)
    assert captured["horizon"] == 90
