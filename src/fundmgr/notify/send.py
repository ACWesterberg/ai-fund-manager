"""Minimal fire-and-forget Telegram sender, reusable outside the bot process."""
from __future__ import annotations

import os
import urllib.parse
import urllib.request


def send_telegram(text: str, parse_mode: str = "HTML") -> bool:
    """Send a Telegram message to TELEGRAM_CHAT_ID. Returns True on success.

    No-ops (returns False) if the token/chat env vars are unset, and never
    raises — notification failures must not break the run pipeline.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return False
    try:
        data = urllib.parse.urlencode(
            {"chat_id": chat_id, "text": text[:4000], "parse_mode": parse_mode}
        ).encode()
        urllib.request.urlopen(
            f"https://api.telegram.org/bot{token}/sendMessage", data, timeout=10
        )
        return True
    except Exception:
        return False
