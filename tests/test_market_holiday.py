"""The weekly run is cron-driven and cron knows nothing about holidays; the
only calendar check in the pipeline sat in auto_fill, per ticker, after the
model had already been paid for."""
from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from fundmgr.data.market_hours import (
    dominant_calendar, is_trading_day, next_session,
)

# US Labor Day 2026 — NYSE/NASDAQ shut, Stockholm and London trading.
LABOR_DAY = datetime.datetime(2026, 9, 7, 14, 0, tzinfo=datetime.timezone.utc)
# The Tuesday after it, when everything trades.
NORMAL_DAY = datetime.datetime(2026, 9, 8, 14, 0, tzinfo=datetime.timezone.utc)

CONFIG = Path(__file__).resolve().parents[1] / "config"


class TestDominantCalendar:
    def test_nordic_universe_answers_to_stockholm(self):
        assert dominant_calendar(CONFIG / "universe.csv") == "XSTO"

    def test_buffett_universe_answers_to_the_us(self):
        assert dominant_calendar(CONFIG / "universe_buffett.csv") == "XNYS"

    def test_global_universe_answers_to_the_us_not_london(self):
        """LSE has more names than either US venue alone, but fewer than the
        two together — and LSE trades on US holidays, so counting raw MICs
        would let a US-dominant fund run into a closed market."""
        assert dominant_calendar(CONFIG / "universe_global.csv") == "XNYS"

    def test_unreadable_universe_fails_open(self):
        assert dominant_calendar(CONFIG / "does_not_exist.csv") is None


class TestTradingDay:
    def test_us_shut_on_labor_day(self):
        assert is_trading_day("XNYS", LABOR_DAY) is False

    def test_stockholm_trades_through_a_us_holiday(self):
        assert is_trading_day("XSTO", LABOR_DAY) is True

    def test_london_trades_through_a_us_holiday(self):
        """Why the grouping matters: this is what a MIC count would have picked."""
        assert is_trading_day("XLON", LABOR_DAY) is True

    def test_everything_trades_the_next_day(self):
        for mic in ("XNYS", "XSTO", "XLON"):
            assert is_trading_day(mic, NORMAL_DAY) is True, mic

    def test_session_level_not_minute_level(self):
        """A run fires at a fixed cron time and its fills follow minutes later,
        so the question is whether the market trades today at all."""
        before_open = datetime.datetime(2026, 9, 8, 6, 0, tzinfo=datetime.timezone.utc)
        assert is_trading_day("XNYS", before_open) is True

    def test_unknown_calendar_is_undeterminable_not_closed(self):
        assert is_trading_day("NOT-A-MIC", LABOR_DAY) is None


class TestNextSession:
    def test_reports_the_day_trading_resumes(self):
        assert next_session("XNYS", LABOR_DAY) == datetime.date(2026, 9, 8)

    def test_is_strictly_after_today(self):
        assert next_session("XSTO", LABOR_DAY) == datetime.date(2026, 9, 8)

    def test_unknown_calendar_returns_none(self):
        assert next_session("NOT-A-MIC", LABOR_DAY) is None


class TestRunGate:
    """_skip_on_market_holiday decides before anything is fetched or billed."""

    def _cfg(self, universe: str):
        from types import SimpleNamespace
        return SimpleNamespace(universe_path=CONFIG / universe, display_name="Test Fund")

    def test_us_fund_skips_on_a_us_holiday(self, monkeypatch):
        from fundmgr import cli
        sent = []
        monkeypatch.setattr("fundmgr.notify.send.send_telegram", lambda *a, **k: sent.append(a))
        monkeypatch.setattr("fundmgr.data.market_hours.is_trading_day",
                            lambda mic, when=None: False)
        monkeypatch.setattr("fundmgr.data.market_hours.next_session",
                            lambda mic, after=None: datetime.date(2026, 9, 8))
        assert cli._skip_on_market_holiday(self._cfg("universe_global.csv")) is True
        assert sent, "a skipped run must say so on Telegram"
        assert "2026-09-08" in sent[0][0]

    def test_nordic_fund_runs_through_a_us_holiday(self, monkeypatch):
        from fundmgr import cli
        monkeypatch.setattr("fundmgr.data.market_hours.is_trading_day",
                            lambda mic, when=None: mic != "XNYS")
        assert cli._skip_on_market_holiday(self._cfg("universe.csv")) is False

    def test_an_undeterminable_calendar_never_blocks_a_run(self, monkeypatch):
        from fundmgr import cli
        monkeypatch.setattr("fundmgr.data.market_hours.is_trading_day",
                            lambda mic, when=None: None)
        assert cli._skip_on_market_holiday(self._cfg("universe_global.csv")) is False

    def test_an_unknown_universe_never_blocks_a_run(self, monkeypatch):
        from fundmgr import cli
        assert cli._skip_on_market_holiday(self._cfg("does_not_exist.csv")) is False


def test_todays_real_schedule_is_what_we_think_it_is():
    """Pins the case that prompted this: on 2026-09-07 the Nordic real-money
    fund should run and the three US-dominant profiles should not."""
    expected = {
        "universe.csv": True,            # 09:30 CET — Stockholm open
        "universe_buffett.csv": False,   # 17:00 / 17:30 CET — US shut
        "universe_global.csv": False,    # 16:00 / 16:30 CET — US shut
    }
    for universe, should_run in expected.items():
        mic = dominant_calendar(CONFIG / universe)
        assert is_trading_day(mic, LABOR_DAY) is should_run, f"{universe} ({mic})"
