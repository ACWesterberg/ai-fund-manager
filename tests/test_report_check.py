"""Report check: a held company's report is read once, against its thesis, on evidence.

The verdict can move real money, so the tests pin what must not happen: an
invented quote voting for an exit, a report read twice at full price, and a
report silently lost because its coverage was late or the model call failed.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from click.testing import CliRunner

from fundmgr import cli as commands
from fundmgr.config import AppConfig
from fundmgr.engine import report_check as rc
from fundmgr.engine.client import LLMError
from fundmgr.engine.schema import ReportCheck, ReportQuote
from fundmgr.state.store import Store

TODAY = "2026-10-21"
SENTENCE = "Net sales decreased by 18 percent and the company lowered its full-year outlook."
COVERAGE = [{"publisher": "Cision", "published": "2026-10-21", "headline": "Interim report Q3",
             "body": f"Third quarter. {SENTENCE} Order intake was flat."}]


def _days_before(n: int) -> str:
    return (datetime.strptime(TODAY, "%Y-%m-%d") - timedelta(days=n)).strftime("%Y-%m-%d")


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "fund.db")
    s.upsert_position("HELD.ST", 100, 50.0)
    return s


@pytest.fixture
def cfg(tmp_path):
    c = AppConfig(db_path=tmp_path / "fund.db")
    c.llm.n_samples = 3
    return c


@pytest.fixture(autouse=True)
def no_market_data(monkeypatch):
    monkeypatch.setattr(rc, "technicals_block", lambda t: ("  Live: 40.00", 40.0))


def _sample(rec: str = "exit", status: str = "broken", quote: str = SENTENCE,
            conf: float = 0.8) -> ReportCheck:
    return ReportCheck(ticker="HELD.ST", thesis_status=status, recommendation=rec,
                       confidence=conf, trim_pct=None, what_changed="Outlook cut.",
                       rationale="Growth thesis contradicted.",
                       quotes=[ReportQuote(quote=quote, shows="demand falling")] if quote else [])


def _consensus(samples):
    calls = []

    def fake(system, user, cfg, schema, label):
        calls.append(user)
        return list(samples)
    fake.calls = calls
    return fake


def _run(store, cfg, consensus, coverage=COVERAGE, history=None, **kw):
    return rc.run_check(
        store, cfg, today=TODAY,
        history=history or (lambda t: [{"date": _days_before(1), "eps_estimate": 1.0,
                                        "eps_reported": 0.6, "eps_surprise_pct": -40.0}]),
        news_fn=lambda t: [], coverage_fn=lambda t, d: coverage,
        consensus=consensus, echo=lambda *_: None, **kw)


# ── Finding the report ────────────────────────────────────────────────────────

def test_only_a_recent_report_counts_as_new():
    old = [{"date": _days_before(40)}]
    assert rc.latest_report("HELD.ST", TODAY, rc.LOOKBACK_DAYS,
                            history=lambda t: old, news_fn=lambda t: []) is None
    fresh = old + [{"date": _days_before(1)}]
    found = rc.latest_report("HELD.ST", TODAY, rc.LOOKBACK_DAYS,
                             history=lambda t: fresh, news_fn=lambda t: [])
    assert found["date"] == _days_before(1) and found["source"] == "calendar"


def test_a_report_headline_stands_in_when_yahoo_has_no_date():
    news = [{"headline": "Bolaget publicerar delårsrapport", "published": _days_before(2)},
            {"headline": "CEO interview", "published": TODAY}]
    found = rc.latest_report("HELD.ST", TODAY, rc.LOOKBACK_DAYS,
                             history=lambda t: [], news_fn=lambda t: news)
    assert found == {"date": _days_before(2), "source": "news"}


# ── Evidence ──────────────────────────────────────────────────────────────────

def test_a_verdict_resting_on_a_real_sentence_stands(store, cfg):
    out = _run(store, cfg, _consensus([_sample()] * 3))
    review = out["reviewed"][0]["review"]
    assert (review.thesis_status, review.recommendation) == ("broken", "exit")
    assert review.quotes and review.quotes[0].quote == SENTENCE


def test_invented_quotes_cannot_vote_for_an_exit(store, cfg):
    invented = _sample(quote="The company announced it will wind down its core business.")
    honest = _sample("hold", "intact", conf=0.6)
    out = _run(store, cfg, _consensus([invented, invented, honest]))
    review = out["reviewed"][0]["review"]
    assert review.recommendation == "hold", "fabricated evidence decided the vote"
    assert out["reviewed"][0]["votes"] == {"hold": 3}


def test_unclear_is_no_licence_to_trade_without_evidence(store, cfg):
    out = _run(store, cfg, _consensus([_sample("exit", "unclear", quote="")] * 3))
    assert out["reviewed"][0]["review"].recommendation == "hold"


def test_the_prompt_carries_the_thesis_and_the_coverage(store, cfg):
    store.log_review_decision(ticker="HELD.ST", action="buy", confidence=0.7,
                              thesis="Order growth from defence contracts.",
                              price_at_decision=50.0, source="run")
    fake = _consensus([_sample()] * 3)
    _run(store, cfg, fake)
    user = fake.calls[0]
    assert "Order growth from defence contracts." in user
    assert SENTENCE in user and "-40.0%" in user


# ── Once, and never lost ──────────────────────────────────────────────────────

def test_a_report_is_read_once_but_can_be_re_read_on_request(store, cfg):
    fake = _consensus([_sample()] * 3)
    _run(store, cfg, fake)
    _run(store, cfg, fake)
    assert len(fake.calls) == 1, "the same report was paid for twice"
    _run(store, cfg, fake, only="HELD.ST")
    assert len(fake.calls) == 2


def test_late_coverage_is_retried_then_reported(store, cfg):
    fake = _consensus([_sample()] * 3)
    fresh = lambda t: [{"date": TODAY}]  # noqa: E731
    out = _run(store, cfg, fake, coverage=[], history=fresh)
    assert out["no_coverage"] == [] and not fake.calls
    out = _run(store, cfg, fake, coverage=COVERAGE, history=fresh)
    assert len(out["reviewed"]) == 1, "a report whose coverage arrived late was dropped"

    stale = lambda t: [{"date": _days_before(rc.COVERAGE_GRACE_DAYS)}]  # noqa: E731
    out = _run(store, cfg, fake, coverage=[], history=stale)
    assert [m["ticker"] for m in out["no_coverage"]] == ["HELD.ST"]
    out = _run(store, cfg, fake, coverage=[], history=stale)
    assert out["no_coverage"] == [], "the same 'read it yourself' was sent twice"


def test_a_failed_call_is_reported_and_retried(store, cfg):
    def failing(*a, **k):
        raise LLMError("all samples failed")
    out = _run(store, cfg, failing)
    assert [f["ticker"] for f in out["failed"]] == ["HELD.ST"]
    out = _run(store, cfg, _consensus([_sample()] * 3))
    assert len(out["reviewed"]) == 1


def test_a_dry_run_calls_nothing(store, cfg):
    fake = _consensus([_sample()] * 3)
    out = _run(store, cfg, fake, dry_run=True)
    assert not fake.calls and not any(out.values())


# ── Filed like any review ─────────────────────────────────────────────────────

def test_an_exit_is_filed_as_an_open_review(store, cfg):
    out = _run(store, cfg, _consensus([_sample()] * 3))
    open_reviews = store.get_open_reviews()
    assert [(r["ticker"], r["source"], r["verdict"]) for r in open_reviews] == \
        [("HELD.ST", "report_review", "exit")]
    assert out["reviewed"][0]["review_id"] == open_reviews[0]["review_id"]


def test_the_message_escapes_model_text(store, cfg):
    s = _sample().model_copy(update={"rationale": "<b>sell</b> & run"})
    out = _run(store, cfg, _consensus([s] * 3))
    body = rc.format_html(out)
    assert "&lt;b&gt;sell&lt;/b&gt; &amp; run" in body and "<b>sell</b>" not in body


def test_nothing_to_say_sends_nothing():
    assert rc.format_html({"reviewed": [], "no_coverage": [], "failed": []}) is None


def test_a_paused_fund_checks_nothing(store, monkeypatch):
    monkeypatch.setattr(rc, "run_check", lambda *a, **k: pytest.fail("check ran"))
    monkeypatch.setattr(commands, "_get_store", lambda cfg_=None: (AppConfig(paused=True), store))
    result = CliRunner().invoke(commands.cli, ["report-check"])
    assert result.exit_code == 0, result.output
