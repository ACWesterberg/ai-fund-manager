"""
Earnings watch: each held name's report read against the thesis it was bought on.

No network and no model: calendar, news, coverage and the LLM are fakes.
"""
from __future__ import annotations

from datetime import date, datetime

import pytest

from fundmgr.config import AppConfig
from fundmgr.engine import earnings_watch as ew
from fundmgr.engine.schema import ReportQuote, ReportThesisCheck
from fundmgr.state.models import Transaction
from fundmgr.state.store import Store

MON, TUE, WED, FRI = date(2026, 11, 2), date(2026, 11, 3), date(2026, 11, 4), date(2026, 10, 30)
RESULTS = ("Invisio's net sales rose 31 per cent to SEK 412 million and the order "
           "intake reached a record SEK 520 million in the quarter.")


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "fund.db")
    s.set_cash(50_000)
    s.apply_fill(Transaction("IVSO.ST", "buy", 30, 250.0, 3.0, "fill", datetime(2026, 10, 5, 12)))
    s.apply_fill(Transaction("NOBA.ST", "buy", 100, 89.0, 4.0, "fill", datetime(2026, 10, 5, 12)))
    s.log_review_decision("IVSO.ST", "buy", 0.7, "Order intake from European defence budgets "
                          "keeps sales growth above 25%.", 250.0, "run", "2026-10-05")
    return s


@pytest.fixture
def cfg():
    c = AppConfig()
    c.earnings_watch.enabled = True
    return c


def _cal(dates):
    return lambda t: dates.get(t)


def _no_news(t):
    return []


def _coverage(published="2026-11-03T07:30:00Z", text=RESULTS):
    def fn(ticker, report_date):
        if ticker != "IVSO.ST":
            return []
        return [{"headline": "Invisio interim report Q3", "publisher": "Cision",
                 "published": published, "body": text, "url": "https://x/1"},
                {"headline": "Invisio to report Q3 on 3 Nov", "publisher": "Cision",
                 "published": "2026-10-27T07:00:00Z", "body": "Invisio will publish...", "url": "u2"}]
    return fn


class LLM:
    def __init__(self, verdict="held", quote=RESULTS):
        self.verdict, self.quote, self.calls = verdict, quote, []

    def __call__(self, system, user, cfg, schema):
        self.calls.append((system, user, cfg))
        assert schema is ReportThesisCheck
        quotes = [ReportQuote(quote=self.quote, shows="growth")] if self.quote else []
        return ReportThesisCheck(ticker="IVSO.ST", verdict=self.verdict, confidence=0.8,
                                 what_changed="Sales +31%, record orders.",
                                 thesis_impact="Growth above 25% holds.", watch_next="",
                                 quotes=quotes), "{}"


def _run(store, cfg, today, sent, **kw):
    kw.setdefault("calendar_fn", _cal({"IVSO.ST": TUE}))
    kw.setdefault("news_fn", _no_news)
    return ew.run(store, cfg, today, send=lambda m: sent.append(m) or True,
                  names={"IVSO.ST": "Invisio"}, echo=lambda s: None, **kw)


def test_heads_up_the_evening_before_with_the_thesis(store, cfg):
    sent: list[str] = []
    out = _run(store, cfg, MON, sent, llm=LLM())
    assert out == [{"ticker": "IVSO.ST", "kind": "heads_up"}]
    assert "reports Tue 03 Nov" in sent[0] and "European defence budgets" in sent[0]
    _run(store, cfg, MON, sent, llm=LLM())
    assert len(sent) == 1                      # once per report


def test_friday_evening_warns_about_a_monday_report(store, cfg):
    sent: list[str] = []
    _run(store, cfg, FRI, sent, calendar_fn=_cal({"IVSO.ST": MON}))
    assert sent and "reports Mon 02 Nov" in sent[0]


def test_report_day_verdict_reads_only_coverage_since_the_report(store, cfg):
    sent: list[str] = []
    llm = LLM("held")
    out = _run(store, cfg, TUE, sent, coverage_fn=_coverage(), llm=llm)
    assert out == [{"ticker": "IVSO.ST", "kind": "verdict", "verdict": "held"}]
    user = llm.calls[0][1]
    assert "Invisio will publish" not in user          # the preview is not the results
    assert "European defence budgets" in user and "%" not in user.split("## Coverage")[0].split(
        "## Why")[0]                                   # no price move in the prompt
    assert llm.calls[0][2].llm.reasoning_effort == "medium"
    assert "thesis HELD" in sent[0] and "net sales rose 31 per cent" in sent[0]
    row = store.get_earnings_check("IVSO.ST", "2026-11-03")
    assert row["verdict"] == "held" and row["attempts"] == 1
    # Judged once: later slots neither call nor send again.
    _run(store, cfg, TUE, sent, coverage_fn=_coverage(), llm=llm)
    assert len(llm.calls) == 1 and len(sent) == 1


def test_a_broken_thesis_offers_review_and_a_sell_ticket_but_trades_nothing(store, cfg):
    sent: list[str] = []
    cash = store.get_cash()
    _run(store, cfg, TUE, sent, coverage_fn=_coverage(), llm=LLM("broken"))
    assert "thesis BROKEN" in sent[0]
    assert "/review IVSO.ST" in sent[0] and "/montrose_ticket IVSO.ST sell 30" in sent[0]
    assert store.get_cash() == cash and {p.ticker for p in store.get_positions()} == {"IVSO.ST", "NOBA.ST"}


def test_a_verdict_without_a_surviving_quote_is_unclear_and_retried(store, cfg):
    sent: list[str] = []
    llm = LLM("broken", quote="Invisio cut its guidance for the full year sharply today.")
    _run(store, cfg, TUE, sent, coverage_fn=_coverage(), llm=llm)
    assert sent == []                          # unclear on report day is not final
    _run(store, cfg, TUE, sent, coverage_fn=_coverage(), llm=llm)
    _run(store, cfg, WED, sent, coverage_fn=_coverage(), llm=llm)
    assert len(llm.calls) == 3 and len(sent) == 1
    assert "thesis UNCLEAR" in sent[0] and "No quote survived" in sent[0]
    # Capped: nothing more is spent on it.
    _run(store, cfg, WED, sent, coverage_fn=_coverage(), llm=llm)
    assert len(llm.calls) == 3


def test_no_coverage_means_no_call_and_eventually_a_read_it_yourself(store, cfg):
    sent: list[str] = []
    llm = LLM()
    _run(store, cfg, TUE, sent, coverage_fn=lambda t, d: [], llm=llm)
    assert llm.calls == [] and sent == []
    for day in (WED, date(2026, 11, 5), date(2026, 11, 6), date(2026, 11, 9)):
        _run(store, cfg, day, sent, calendar_fn=_cal({}), coverage_fn=lambda t, d: [], llm=llm)
    assert llm.calls == [] and len(sent) == 1 and "Read the report yourself" in sent[0]
    assert store.get_earnings_check("IVSO.ST", "2026-11-03")["verdict"] == "unclear"


def test_a_report_headline_stands_in_for_a_missing_calendar(store, cfg):
    sent: list[str] = []
    news = {"NOBA.ST": [{"headline": "Noba Bank Group Interim report January–September 2026",
                         "published": "2026-11-03T06:00:00Z"}]}
    _run(store, cfg, TUE, sent, calendar_fn=_cal({}), news_fn=lambda t: news.get(t, []),
         coverage_fn=lambda t, d: [], llm=LLM())
    row = store.get_earnings_check("NOBA.ST", "2026-11-03")
    assert row and row["found_by"] == "headline"


def test_a_name_sold_before_its_report_is_not_read(store, cfg):
    sent: list[str] = []
    store.apply_fill(Transaction("IVSO.ST", "sell", 30, 260.0, 3.0, "fill", datetime(2026, 11, 2, 12)))
    llm = LLM()
    _run(store, cfg, TUE, sent, coverage_fn=_coverage(), llm=llm)
    assert llm.calls == [] and sent == []


def test_dry_run_spends_and_sends_nothing(store, cfg):
    sent: list[str] = []
    llm = LLM()
    ew.run(store, cfg, TUE, send=lambda m: sent.append(m), dry_run=True,
           calendar_fn=_cal({"IVSO.ST": TUE}), news_fn=_no_news, coverage_fn=_coverage(),
           llm=llm, echo=lambda s: None)
    assert llm.calls == [] and sent == []


def test_theses_prefer_a_newer_hold_rationale_alongside_the_buy(store):
    store.log_review_decision("IVSO.ST", "hold", 0.6, "Margins expanding as volumes scale.",
                              255.0, "run", "2026-10-26")
    th = store.get_position_theses("IVSO.ST")
    assert th["buy"]["date"] == "2026-10-05" and th["hold"]["date"] == "2026-10-26"
    assert store.get_position_theses("NOBA.ST") == {}
