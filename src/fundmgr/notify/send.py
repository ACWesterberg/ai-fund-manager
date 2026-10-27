"""Minimal fire-and-forget Telegram sender, reusable outside the bot process."""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request


def send_telegram(text: str, parse_mode: str = "HTML",
                  buttons: list[list[tuple[str, str]]] | None = None) -> bool:
    """Send a Telegram message to TELEGRAM_CHAT_ID. Returns True on success.

    No-ops (returns False) if the token/chat env vars are unset, and never
    raises — notification failures must not break the run pipeline.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return False
    # Try with the requested parse_mode; if it's rejected (e.g. HTTP 400 from an
    # unescaped character), retry once as plain text so the message still lands.
    for mode in (parse_mode, None):
        payload = {"chat_id": chat_id, "text": text[:4000]}
        if buttons:
            # Rows of (label, callback_data); the bot's CallbackQueryHandler answers them.
            payload["reply_markup"] = json.dumps({"inline_keyboard": [
                [{"text": label, "callback_data": data} for label, data in row] for row in buttons]})
        if mode:
            payload["parse_mode"] = mode
        try:
            urllib.request.urlopen(
                f"https://api.telegram.org/bot{token}/sendMessage",
                urllib.parse.urlencode(payload).encode(), timeout=10,
            )
            return True
        except Exception:
            continue
    return False
