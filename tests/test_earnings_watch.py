"""
Earnings watch: the evening-before heads-up for a holding's report.

The report itself is read by report-check (tests/test_report_check.py).
"""
from __future__ import annotations

from datetime import date, datetime

import pytest

from fundmgr.config import AppConfig
from fundmgr.engine import earnings_watch as ew
from fundmgr.state.models import Transaction
from fundmgr.state.store import Store

THU, FRI, MON, TUE = date(2026, 10, 29), date(2026, 10, 30), date(2026, 11, 2), date(2026, 11, 3)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "fund.db")
    s.set_cash(50_000)
    s.apply_fill(Transaction("IVSO.ST", "buy", 30, 250.0, 3.0, "fill", datetime(2026, 10, 5, 12)))
    s.apply_fill(Transaction("NOBA.ST", "buy", 100, 89.0, 4.0, "fill", datetime(2026, 10, 5, 12)))
    s.log_review_decision("IVSO.ST", "buy", 0.7, "Order intake from European defence budgets "
                          "keeps sales growth above 25%.", 250.0, "run", "2026-10-05")
    return s


def _run(store, today, dates, sent, **kw):
    return ew.run(store, AppConfig(), today, send=lambda m: sent.append(m) or True,
                  names={"IVSO.ST": "Invisio"}, calendar_fn=lambda t: dates.get(t),
                  echo=lambda s: None, **kw)


def test_heads_up_the_evening_before_with_the_thesis_once(store):
    sent: list[str] = []
    out = _run(store, THU, {"IVSO.ST": FRI, "NOBA.ST": TUE}, sent)
    assert [o["ticker"] for o in out] == ["IVSO.ST"]
    assert "reports Fri 30 Oct" in sent[0] and "European defence budgets" in sent[0]
    assert "before the next open" in sent[0]
    _run(store, THU, {"IVSO.ST": FRI}, sent)
    assert len(sent) == 1


def test_friday_evening_warns_about_a_monday_report(store):
    sent: list[str] = []
    _run(store, FRI, {"NOBA.ST": MON}, sent)
    assert sent and "reports Mon 02 Nov" in sent[0] and "No thesis recorded" in sent[0]


def test_dry_run_sends_and_records_nothing(store):
    sent: list[str] = []
    out = _run(store, THU, {"IVSO.ST": FRI}, sent, dry_run=True)
    assert out and sent == [] and store.get_earnings_check("IVSO.ST", "2026-10-30") is None


def test_theses_prefer_a_newer_hold_rationale_alongside_the_buy(store):
    store.log_review_decision("IVSO.ST", "hold", 0.6, "Margins expanding as volumes scale.",
                              255.0, "run", "2026-10-26")
    th = store.get_position_theses("IVSO.ST")
    assert th["buy"]["date"] == "2026-10-05" and th["hold"]["date"] == "2026-10-26"
    assert store.get_position_theses("NOBA.ST") == {}
