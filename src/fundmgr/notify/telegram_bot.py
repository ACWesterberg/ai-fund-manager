"""
Telegram bot — notifications and fill entry for the AI Fund Manager.

Commands:
  /run           — trigger weekly decision run
  /run_full      — trigger run with news + FinBERT sentiment
  /fill TICKER SHARES PRICE FEE  — record a broker fill
  /status        — portfolio snapshot
  /report        — performance report
  /stops         — check stop-loss thresholds
  /universe      — list enabled tickers
  /help          — show this message

Run standalone:  python -m fundmgr.notify.telegram_bot
Or via systemd:  see deploy/fundmgr-bot.service
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Lazy import so the module can be imported without telegram installed ───────
try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes
    _TELEGRAM_AVAILABLE = True
except ImportError:
    _TELEGRAM_AVAILABLE = False


ROOT = Path(__file__).resolve().parents[3]
FUND_BIN = ROOT / ".venv" / "bin" / "fund"


def _run_cli(*args: str, timeout: int = 300) -> str:
    """Run a fund CLI command and return its stdout as a string."""
    cmd = [str(FUND_BIN), *args]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=ROOT,
        )
        output = result.stdout.strip()
        if result.returncode != 0 and result.stderr:
            output += f"\n\nError: {result.stderr.strip()[:500]}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return f"⏱ Command timed out after {timeout}s"
    except Exception as e:
        return f"❌ Failed to run command: {e}"


def _chunk(text: str, max_len: int = 4000) -> list[str]:
    """Split long messages into Telegram-safe chunks (4096 char limit)."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    while text:
        chunks.append(text[:max_len])
        text = text[max_len:]
    return chunks


async def _send(update: "Update", text: str) -> None:
    for chunk in _chunk(text):
        await update.message.reply_text(chunk, parse_mode=None)


# ── Command handlers ──────────────────────────────────────────────────────────

async def cmd_run(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    await update.message.reply_text("⏳ Running weekly decision pipeline (skipping news for speed)…")
    output = _run_cli("run", "--skip-news", timeout=300)
    await _send(update, output)


async def cmd_run_full(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    await update.message.reply_text("⏳ Running full pipeline with news + FinBERT (this takes a few minutes)…")
    output = _run_cli("run", timeout=600)
    await _send(update, output)


async def cmd_fill(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    """
    Usage: /fill TICKER SHARES PRICE FEE [buy|sell]
    Example: /fill VOLV-B.ST 12 291.50 2.91
             /fill SAND.ST 8 217.80 1.74 sell
    """
    args = context.args or []
    if len(args) < 4:
        await update.message.reply_text(
            "Usage: /fill TICKER SHARES PRICE FEE [buy|sell]\n"
            "Example: /fill VOLV-B.ST 12 291.50 2.91\n"
            "         /fill SAND.ST 8 217.80 1.74 sell"
        )
        return

    ticker, shares, price, fee = args[0], args[1], args[2], args[3]
    side = args[4] if len(args) > 4 else "buy"

    cli_args = ["fill", ticker, shares, price, fee, "--side", side]
    output = _run_cli(*cli_args, timeout=30)
    await _send(update, output)


async def cmd_status(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    output = _run_cli("status", timeout=30)
    await _send(update, output)


async def cmd_report(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    output = _run_cli("report", timeout=30)
    await _send(update, output)


async def cmd_stops(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    output = _run_cli("check-stops", timeout=60)
    await _send(update, output)


async def cmd_universe(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    output = _run_cli("universe", timeout=15)
    await _send(update, output)


async def cmd_help(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    await update.message.reply_text(
        "🤖 AI Fund Manager Bot\n\n"
        "/run — weekly decision run (fast, no news)\n"
        "/run_full — full run with FinBERT sentiment\n"
        "/fill TICKER SHARES PRICE FEE [side] — record a fill\n"
        "         e.g. /fill VOLV-B.ST 12 291.50 2.91\n"
        "/status — current portfolio snapshot\n"
        "/report — performance vs OMXSPI\n"
        "/stops — check stop-loss alerts\n"
        "/universe — list all enabled tickers\n"
        "/help — this message"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if not _TELEGRAM_AVAILABLE:
        log.error("python-telegram-bot not installed. Run: uv pip install python-telegram-bot")
        sys.exit(1)

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token:
        log.error("TELEGRAM_BOT_TOKEN not set in .env")
        sys.exit(1)

    log.info("Starting AI Fund Manager bot…")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("run", cmd_run))
    app.add_handler(CommandHandler("run_full", cmd_run_full))
    app.add_handler(CommandHandler("fill", cmd_fill))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("report", cmd_report))
    app.add_handler(CommandHandler("stops", cmd_stops))
    app.add_handler(CommandHandler("universe", cmd_universe))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("start", cmd_help))

    log.info("Bot polling… (Ctrl+C to stop)")
    app.run_polling(allowed_updates=["message"])


if __name__ == "__main__":
    main()
