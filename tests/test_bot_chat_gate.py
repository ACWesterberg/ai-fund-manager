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
