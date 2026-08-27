"""
Tests for the Telegram bot's long-running command handling.

Covers the two failure modes that made /run report "Command timed out after
300s": the timeout was far shorter than a real pipeline run, and the command
ran inline so it blocked the whole bot while it waited.

Run: pytest tests/test_bot_long_commands.py -v
"""
from __future__ import annotations

import asyncio
import sys
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from fundmgr.notify import telegram_bot as tb  # noqa: E402


# ── Fakes ─────────────────────────────────────────────────────────────────────

class _FakeBot:
    def __init__(self, sink: list[str]) -> None:
        self._sink = sink

    async def send_message(self, chat_id: int, text: str) -> None:
        self._sink.append(text)


class _FakeMessage:
    def __init__(self, sink: list[str]) -> None:
        self._sink = sink

    async def reply_text(self, text: str, **kwargs) -> None:
        self._sink.append(text)


def _fake_update_context(sink: list[str], args: list[str] | None = None):
    update = types.SimpleNamespace(
        message=_FakeMessage(sink),
        effective_chat=types.SimpleNamespace(id=42),
    )
    context = types.SimpleNamespace(bot=_FakeBot(sink), args=args or [])
    return update, context


# ── Timeout budget ────────────────────────────────────────────────────────────

def test_run_timeouts_allow_a_realistic_pipeline():
    """300s was below a real Pi run; the budgets must leave real headroom."""
    assert tb.RUN_TIMEOUT >= 1800          # /run — prices + fundamentals + LLM
    assert tb.RUN_FULL_TIMEOUT > tb.RUN_TIMEOUT   # full run adds news + FinBERT
    assert tb.REVIEW_TIMEOUT >= 600


def test_timeout_is_env_overridable(monkeypatch):
    monkeypatch.setenv("FUND_RUN_TIMEOUT", "1234")
    assert tb._env_timeout("FUND_RUN_TIMEOUT", 2700) == 1234


@pytest.mark.parametrize("bad", ["", "not-a-number", "0", "-5"])
def test_bad_timeout_override_falls_back_to_default(monkeypatch, bad):
    monkeypatch.setenv("FUND_RUN_TIMEOUT", bad)
    assert tb._env_timeout("FUND_RUN_TIMEOUT", 2700) == 2700


# ── Timeout message ───────────────────────────────────────────────────────────

def test_timeout_message_reports_the_stalled_step(monkeypatch):
    """A timeout must say where the run got stuck, not just that it stopped."""
    monkeypatch.setattr(tb, "FUND_BIN", "/bin/sh")
    out = tb._run_cli(
        "-c",
        'echo "[1/6] Fetching prices…"; echo "[5/6] Computing features…"; sleep 30',
        timeout=2,
    )
    assert "timed out" in out
    assert "[5/6] Computing features" in out   # tail of the partial output
    assert "FUND_RUN_TIMEOUT" in out           # tells the user how to raise it


def test_tail_keeps_only_the_last_nonempty_lines():
    text = "\n".join(["", "one", "", "two", "three", ""])
    assert tb._tail(text, lines=2) == "two\nthree"


# ── Non-blocking execution ────────────────────────────────────────────────────

def test_cmd_run_returns_immediately_and_notifies_when_done(monkeypatch):
    """
    /run must hand off to a background task: the handler returns at once (so the
    bot keeps serving price alerts and /status) and the result arrives later.
    """
    sink: list[str] = []
    update, context = _fake_update_context(sink)
    monkeypatch.setattr(tb, "_run_cli", lambda *a, **k: (time.sleep(0.5), "ok")[1])

    async def scenario():
        started = time.monotonic()
        await tb.cmd_run(update, context)
        handler_elapsed = time.monotonic() - started
        assert handler_elapsed < 0.2, "handler blocked the event loop"

        # The loop is genuinely free while the pipeline runs.
        await asyncio.sleep(0.05)
        assert tb._bg_tasks, "background run was not scheduled"

        await asyncio.sleep(1.0)
        assert not tb._bg_tasks, "task reference was not released"

    asyncio.run(scenario())
    assert any("Running weekly decision pipeline" in m for m in sink)
    assert any("✅ Done" in m for m in sink)


def test_background_run_reports_a_timeout_back_to_the_chat(monkeypatch):
    sink: list[str] = []
    update, context = _fake_update_context(sink)
    monkeypatch.setattr(tb, "_run_cli", lambda *a, **k: "⏱ Command timed out after 45 min")

    async def scenario():
        await tb.cmd_run(update, context)
        await asyncio.sleep(0.3)

    asyncio.run(scenario())
    assert any("finished with errors" in m and "timed out" in m for m in sink)


def test_review_scan_posts_its_output_from_the_background(monkeypatch):
    """The full /review scan is backgrounded but its report must still arrive."""
    sink: list[str] = []
    update, context = _fake_update_context(sink, args=[])
    monkeypatch.setattr(tb, "_run_cli", lambda *a, **k: "TRUE-B.ST — HOLD, thesis intact")

    async def scenario():
        await tb.cmd_review(update, context)
        await asyncio.sleep(0.3)

    asyncio.run(scenario())
    assert any("thesis intact" in m for m in sink)
