"""
Tests for the Telegram bot's chat allowlist.

The bot answered any chat that found its username, and its commands record
fills, set cash and book deposits on the real book. Every update now passes a
gate that drops anything not from an allowed chat before a handler runs.

Run: pytest tests/test_bot_chat_gate.py -v
"""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from fundmgr.notify import telegram_bot as tb  # noqa: E402


def _update(chat_id: int | None) -> types.SimpleNamespace:
    chat = None if chat_id is None else types.SimpleNamespace(id=chat_id)
    return types.SimpleNamespace(effective_chat=chat)


def test_allowed_ids_combine_primary_and_extra(monkeypatch):
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "111")
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", "222, -333")
    assert tb.allowed_chat_ids() == {111, 222, -333}


def test_malformed_entries_narrow_rather_than_widen(monkeypatch):
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "...")
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", "abc,,*")
    assert tb.allowed_chat_ids() == set()


def test_unset_means_no_one(monkeypatch):
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("TELEGRAM_ALLOWED_CHAT_IDS", raising=False)
    assert tb.allowed_chat_ids() == set()


def test_is_authorised():
    assert tb.is_authorised(_update(111), {111})
    assert not tb.is_authorised(_update(999), {111})
    assert not tb.is_authorised(_update(None), {111})
    assert not tb.is_authorised(_update(111), set())


@pytest.mark.skipif(not tb._TELEGRAM_AVAILABLE, reason="python-telegram-bot not installed")
def test_gate_stops_strangers_and_passes_owner():
    gate = tb._make_gate({111})
    asyncio.run(gate(_update(111), None))
    with pytest.raises(tb.ApplicationHandlerStop):
        asyncio.run(gate(_update(999), None))
    with pytest.raises(tb.ApplicationHandlerStop):
        asyncio.run(gate(_update(None), None))


# ── Commands that arrive as formatted text ───────────────────────────────────
# A command pasted from a message that showed it as `code` reaches the bot
# without its bot_command marker; /montrose_ticket did nothing at all.

class _Msg:
    def __init__(self, text: str) -> None:
        self.text = text
        self.replies: list[str] = []

    async def reply_text(self, text: str, **kw) -> None:
        self.replies.append(text)


def test_a_formatted_command_runs_with_its_arguments():
    seen = []

    async def cmd(update, context):
        seen.append(context.args)

    msg = _Msg("/montrose_ticket IVSO.ST sell 1")
    ctx = types.SimpleNamespace(args=None)
    handler = tb._formatted_command({"montrose_ticket": cmd})
    asyncio.run(handler(types.SimpleNamespace(message=msg), ctx))
    assert seen == [["IVSO.ST", "sell", "1"]]


def test_an_unknown_formatted_command_says_so():
    msg = _Msg("/nope@expert_fund_manager_bot x")
    asyncio.run(tb._formatted_command({})(types.SimpleNamespace(message=msg),
                                          types.SimpleNamespace(args=None)))
    assert msg.replies and "/nope" in msg.replies[0]
