"""A daily NAV mark, /undo on Telegram, and reports that price and label the book."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from fundmgr import cli as commands
from fundmgr.config import AppConfig
from fundmgr.notify import telegram_bot as tb
from fundmgr.reporting.dashboard import format_text_report, generate_html_report
from fundmgr.state.models import NavPoint, Transaction
from fundmgr.state.store import Store

ROOT = Path(__file__).resolve().parents[1]


# ── The managed cron block ────────────────────────────────────────────────────

def test_the_daily_mark_is_installed_once_beside_hand_kept_jobs(tmp_path):
    tab = tmp_path / "tab"
    tab.write_text("30 9 * * 1 cd /x && .venv/bin/fund run >> data/logs/weekly.log 2>&1\n")
    fake = tmp_path / "bin" / "crontab"
    fake.parent.mkdir()
    fake.write_text(f'#!/usr/bin/env bash\nif [ "$1" = "-l" ]; then cat {tab}; '
                    f'elif [ "$1" = "-" ]; then cat > {tab}; fi\n')
    fake.chmod(0o755)
    env = {**os.environ, "PATH": f"{fake.parent}:{os.environ['PATH']}"}
    def run() -> str:
        return subprocess.run(["bash", "deploy/managed_cron.sh", "."], cwd=ROOT, env=env,
                              capture_output=True, text=True, check=True).stdout
    assert "updated" in run() and "already up to date" in run()
    lines = tab.read_text().splitlines()
    assert sum("fund mark" in line for line in lines) == 1
    assert sum("fund run" in line for line in lines) == 1


# ── Undo ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def book(tmp_path, monkeypatch):
    cfg = AppConfig(db_path=tmp_path / "fund.db")
    store = Store(cfg.db_path)
    store.initialise(10000)
    store.apply_fill(Transaction("AAA.ST", "buy", 10, 100, 1, "fill"))
    store.apply_fill(Transaction("BBB.ST", "buy", 5, 200, 1, "fill"))
    monkeypatch.setattr(commands, "_get_store", lambda cfg_=None: (cfg, store))
    monkeypatch.setattr(commands, "_mark_after_ledger_change", lambda *a: None)
    return cfg, store


def test_undo_shows_the_fill_it_would_reverse_without_changing_anything(book):
    _cfg, store = book
    out = CliRunner().invoke(commands.cli, ["undo-fill", "--show", "--json"]).output
    shown = json.loads(out)
    assert shown["id"] == store.last_transaction().id and "BBB.ST" in shown["summary"]
    assert len(store.get_transactions()) == 2


def test_a_stale_undo_button_cannot_reverse_a_later_fill(book):
    _cfg, store = book
    stale = store.last_transaction().id
    store.apply_fill(Transaction("CCC.ST", "buy", 1, 50, 1, "fill"))
    result = CliRunner().invoke(commands.cli, ["undo-fill", "--expect", str(stale)])
    assert result.exit_code != 0 and "newer fill" in result.output
    assert len(store.get_transactions()) == 3
    fresh = store.last_transaction().id
    assert CliRunner().invoke(commands.cli, ["undo-fill", "--expect", str(fresh)]).exit_code == 0
    assert {t.ticker for t in store.get_transactions()} == {"AAA.ST", "BBB.ST"}


class _Query:
    def __init__(self, data):
        self.data, self.edits = data, []
        self.message = SimpleNamespace(text="Undo the most recent fill?")

    async def answer(self):
        pass

    async def edit_message_reply_markup(self, reply_markup=None):
        pass

    async def edit_message_text(self, text):
        self.edits.append(text)


def test_the_undo_button_names_the_fill_it_was_shown(monkeypatch):
    calls = []
    monkeypatch.setattr(tb, "_run_cli_ok", lambda *a, **k: calls.append(a) or (True, "✓ Undone"))
    query = _Query("undo:42")
    asyncio.run(tb.undo_callback(SimpleNamespace(callback_query=query), None))
    assert calls == [("undo-fill", "--expect", "42")] and "✓ Undone" in query.edits[-1]
    keep = _Query("undo:keep")
    asyncio.run(tb.undo_callback(SimpleNamespace(callback_query=keep), None))
    assert len(calls) == 1 and "Kept" in keep.edits[-1]


# ── Reports ───────────────────────────────────────────────────────────────────

def test_the_reports_price_the_book_and_name_its_benchmark(tmp_path):
    cfg = AppConfig(db_path=tmp_path / "fund.db", benchmark="URTH")
    store = Store(cfg.db_path)
    store.initialise(10000)
    store.apply_fill(Transaction("AAA.ST", "buy", 10, 100, 0, "fill"))
    store.save_prices("AAA.ST", [{"date": "2026-10-30", "open": 0, "high": 0, "low": 0,
                                  "close": 110.0, "volume": 0}])
    store.upsert_nav(NavPoint("2026-10-01", 10000, 100, 10000))
    store.upsert_nav(NavPoint(datetime.utcnow().strftime("%Y-%m-%d"), 10100, 101, 9000))
    text = format_text_report(store, cfg)
    assert "+10.0%" in text and "URTH:" in text and "OMXSPI" not in text
    out = tmp_path / "report.html"
    generate_html_report(store, cfg, out)
    page = out.read_text()
    assert ">URTH<" in page and "OMXSPI" not in page
