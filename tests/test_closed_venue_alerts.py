"""A breach found while its venue is shut is not an order the owner can place.

The Nordic check-stops cron line `*/15 9-17` fires at 17:30, the minute
Stockholm closes. On Friday 2026-10-23 that cycle found BONEX.ST past its +38%
target, ran the five-sample target review and sent "ACTION NEEDED — SELL" at
17:31 — an order against a closing print, unplaceable until the next session.

Run: pytest tests/test_closed_venue_alerts.py -v
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from fundmgr import cli as commands
from fundmgr.config import AppConfig
from fundmgr.data import market_hours
from fundmgr.levels import split_by_venue
from fundmgr.state.store import Store

CONFIG = Path(__file__).resolve().parents[1] / "config"

# 17:30 and 17:15 Stockholm (CEST, UTC+2) on the Friday of the incident.
AT_CLOSE = datetime(2026, 10, 23, 15, 30, tzinfo=timezone.utc)
IN_SESSION = datetime(2026, 10, 23, 15, 15, tzinfo=timezone.utc)


def _hit(ticker: str) -> tuple:
    return (ticker, 38.4, 38.0, 332.6)


def test_a_closed_venue_holds_the_hit_back():
    actionable, held = split_by_venue([_hit("BONEX.ST")], lambda _t: False)
    assert actionable == [] and held == [_hit("BONEX.ST")]


def test_an_open_venue_keeps_the_hit():
    actionable, held = split_by_venue([_hit("BONEX.ST")], lambda _t: True)
    assert actionable == [_hit("BONEX.ST")] and held == []


def test_an_unknown_venue_fails_open():
    actionable, held = split_by_venue([_hit("XYZ")], lambda _t: None)
    assert actionable == [_hit("XYZ")] and held == []


def test_each_hit_is_judged_on_its_own_venue():
    open_ = {"EQNR.OL": True, "BONEX.ST": False}
    actionable, held = split_by_venue([_hit("EQNR.OL"), _hit("BONEX.ST")], open_.get)
    assert [h[0] for h in actionable] == ["EQNR.OL"]
    assert [h[0] for h in held] == ["BONEX.ST"]


def test_stockholm_counts_as_shut_at_the_minute_the_cron_fired():
    assert market_hours.is_exchange_open("OMXS-FN", AT_CLOSE) is False
    assert market_hours.is_exchange_open("OMXS-FN", IN_SESSION) is True


# ── The whole cycle ───────────────────────────────────────────────────────────

class _Ticker:
    def __init__(self, _symbol):
        pass

    def history(self, period="2d"):
        idx = pd.DatetimeIndex([pd.Timestamp("2026-10-22"), pd.Timestamp("2026-10-23")])
        return pd.DataFrame({"Close": [320.0, 332.6]}, index=idx)


@pytest.fixture
def real_book(tmp_path, monkeypatch):
    cfg = AppConfig(name="Test REAL", db_path=tmp_path / "fund.db",
                    universe_path=CONFIG / "universe.csv")
    store = Store(cfg.db_path)
    store.initialise(50_000)
    store.upsert_position("BONEX.ST", 13, 240.3)
    store.set_position_stop("BONEX.ST", stop_pct=15.0, take_profit_pct=38.0)
    monkeypatch.setattr(commands, "_get_store", lambda cfg_=None: (cfg, store))
    monkeypatch.setattr("yfinance.Ticker", _Ticker)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "c")
    sent: list = []
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: sent.append(a))
    reviewed: list[str] = []

    def _review(ticker, *_a, **_k):
        reviewed.append(ticker)
        return None

    monkeypatch.setattr("fundmgr.engine.target_review.review_position", _review)
    return store, sent, reviewed


def _at(when: datetime, monkeypatch):
    class _Clock(datetime):
        @classmethod
        def utcnow(cls):
            return when.replace(tzinfo=None)

    real_open = market_hours.is_exchange_open
    monkeypatch.setattr(commands, "datetime", _Clock)
    monkeypatch.setattr(market_hours, "is_exchange_open",
                        lambda code, when_=None: real_open(code, when))


def test_the_1730_cycle_neither_reviews_nor_alerts(real_book, monkeypatch):
    _, sent, reviewed = real_book
    _at(AT_CLOSE, monkeypatch)

    result = CliRunner().invoke(commands.cli, ["check-stops"])

    assert result.exit_code == 0, result.output
    assert reviewed == [] and sent == []
    assert "BONEX.ST — OMXS-FN closed" in result.output


def test_the_same_breach_in_session_is_reviewed_and_alerted(real_book, monkeypatch):
    _, sent, reviewed = real_book
    _at(IN_SESSION, monkeypatch)

    result = CliRunner().invoke(commands.cli, ["check-stops"])

    assert result.exit_code == 0, result.output
    assert reviewed == ["BONEX.ST"] and len(sent) == 1
