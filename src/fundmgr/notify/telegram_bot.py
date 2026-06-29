"""
Telegram bot — notifications and fill entry for the AI Fund Manager.

Commands:
  /run           — trigger weekly decision run
  /run_full      — trigger run with news + FinBERT sentiment
  /fill TICKER SHARES PRICE FEE [buy|sell]  — record a fill manually
  /status        — portfolio snapshot
  /report        — performance report
  /stops         — check stop-loss thresholds
  /universe      — list enabled tickers
  /reject_rates  — malformed-sample & guardrail drop rates (Refine gate)
  /review [TICKER] — stop-loss review (no ticker = scan all breaches)
  /setcash AMOUNT — correct the cash balance (SEK)
  /help          — show this message

Photo messages:
  Send a screenshot of a Montrose trade confirmation — the bot will use
  GPT vision to extract the fill details and ask for confirmation before
  recording it.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from io import BytesIO
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.ext import (
        Application,
        CallbackQueryHandler,
        CommandHandler,
        ContextTypes,
        MessageHandler,
        filters,
    )
    _TELEGRAM_AVAILABLE = True
except ImportError:
    _TELEGRAM_AVAILABLE = False


ROOT     = Path(__file__).resolve().parents[3]
FUND_BIN = ROOT / ".venv" / "bin" / "fund"

# Pending vision-extracted fills awaiting user confirmation: chat_id -> data dict
_pending_fills: dict[int, dict] = {}

# Lazy-loaded ISIN -> yahoo_ticker map
_isin_map: dict[str, str] | None = None


def _get_isin_map() -> dict[str, str]:
    global _isin_map
    if _isin_map is None:
        from fundmgr.config import get_isin_map
        _isin_map = get_isin_map()
    return _isin_map


def _decision_name_map() -> dict[str, str]:
    """
    Return {company_name_lower: yahoo_ticker} for all tickers in the last
    recommendation's buy/sell actions. These are the most likely candidates
    when a user photographs a trade confirmation.
    """
    try:
        from fundmgr.config import load_config, load_universe
        from fundmgr.state.store import Store
        cfg = load_config()
        store = Store(cfg.db_path)
        last = store.get_last_recommendation()
        if not last:
            return {}
        actions = json.loads(last.actions_json)
        decision_tickers = {
            a["ticker"] for a in actions
            if a.get("side") in ("buy", "sell") and a.get("ticker")
        }
        # Build name → ticker map from universe for just those tickers
        universe = load_universe(cfg.universe_path)
        return {
            t.name.lower(): t.yahoo_ticker
            for t in universe
            if t.yahoo_ticker in decision_tickers
        }
    except Exception:
        return {}


def _name_to_ticker(company_name: str) -> tuple[str | None, str]:
    """
    Resolve company name to ticker. Returns (ticker, source) where source
    describes how the match was found.
    Try: last decision tickers → full universe fuzzy match.
    """
    name_q = company_name.lower().strip()

    # 1. Match against last decision (most reliable — small candidate set)
    decision_map = _decision_name_map()
    for name, ticker in decision_map.items():
        if name_q == name or name_q in name or name in name_q:
            return ticker, "matched from last decision"

    # 2. Full universe fuzzy fallback
    try:
        from fundmgr.config import load_config, load_universe
        universe = load_universe(load_config().universe_path)
        for t in universe:
            if t.name.lower() == name_q:
                return t.yahoo_ticker, "matched by name"
        for t in universe:
            t_name = t.name.lower()
            if name_q in t_name or t_name in name_q:
                return t.yahoo_ticker, "matched by name"
    except Exception:
        pass

    return None, ""


def _run_cli(*args: str, timeout: int = 300) -> str:
    """Run a fund CLI command and return its stdout as a string."""
    cmd = [str(FUND_BIN), *args]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
        output = result.stdout.strip()
        if result.returncode != 0 and result.stderr:
            output += f"\n\nError: {result.stderr.strip()[:500]}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return f"⏱ Command timed out after {timeout}s"
    except Exception as e:
        return f"❌ Failed to run command: {e}"


def _chunk(text: str, max_len: int = 4000) -> list[str]:
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
    await update.message.reply_text("⏳ Running weekly decision pipeline (skipping news)…")
    output = _run_cli("run", "--skip-news", timeout=300)
    # fund run sends its own formatted Telegram notification on success;
    # only echo output back here if something went wrong
    if "Error" in output or "Traceback" in output or "timed out" in output:
        await _send(update, f"⚠️ Run finished with errors:\n\n{output[:3000]}")
    else:
        await update.message.reply_text("✅ Done — decision summary sent above.")


async def cmd_run_full(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    import asyncio
    chat_id = update.effective_chat.id
    bot = context.bot
    await update.message.reply_text(
        "⏳ Running full pipeline with news + FinBERT…\n"
        "This takes 10-20 min on Pi — I'll message you when it's done."
    )

    async def _run_and_notify():
        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(None, lambda: _run_cli("run", timeout=1800))
        if "Error" in output or "Traceback" in output or "timed out" in output:
            for chunk in _chunk(f"⚠️ Run finished with errors:\n\n{output}"):
                await bot.send_message(chat_id=chat_id, text=chunk)
        else:
            await bot.send_message(chat_id=chat_id, text="✅ Full run complete — decision summary sent above.")

    asyncio.ensure_future(_run_and_notify())


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
            "         /fill SAND.ST 8 217.80 1.74 sell\n\n"
            "Or just send a screenshot of your Montrose confirmation."
        )
        return
    ticker, shares, price, fee = args[0], args[1], args[2], args[3]
    side = args[4] if len(args) > 4 else "buy"
    output = _run_cli("fill", ticker, shares, price, fee, "--side", side, timeout=30)
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


async def cmd_reject_rates(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    output = _run_cli("reject-rates", timeout=30)
    await _send(update, output)


async def cmd_setcash(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    """Usage: /setcash AMOUNT — correct the cash balance (SEK)."""
    args = context.args or []
    if not args:
        await update.message.reply_text("Usage: /setcash AMOUNT\nExample: /setcash 10382")
        return
    output = _run_cli("set-cash", args[0], timeout=30)
    await _send(update, output)


async def cmd_review(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    """/review [TICKER] — advisory stop-loss review.

    With a ticker: review that position. With no ticker: scan all holdings and
    review every position currently below its stop-loss.
    """
    args = context.args or []
    if args:
        await update.message.reply_text(f"⏳ Reviewing {args[0].upper()} (consensus)… ~1 min")
        output = _run_cli("review-stop", args[0], "--no-notify", timeout=180)
    else:
        await update.message.reply_text("⏳ Scanning holdings for stop-loss breaches and reviewing… may take a few min")
        output = _run_cli("review-stop", "--no-notify", timeout=600)
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
        "/reject_rates — malformed-sample & guardrail drop rates (Refine gate)\n"
        "/review [TICKER] — stop-loss review; no ticker = scan all breaches\n"
        "/setcash AMOUNT — correct the cash balance (SEK)\n"
        "/help — this message\n\n"
        "📸 Send a screenshot of a Montrose confirmation to auto-record a fill."
    )


# ── Screenshot fill extraction ────────────────────────────────────────────────

_OCR_SYSTEM_PROMPT = """\
You are extracting trade fill details from OCR text of a Swedish broker order \
confirmation (Avanza-style). The account is in SEK (ISK), so all monetary TOTALS \
are SEK even when the stock trades in a foreign currency (e.g. DKK). OCR may have \
minor artifacts.

CRITICAL — currency rules:
- "Kurs" / "Avslutskurs" is the execution price in the STOCK'S NATIVE currency
  (e.g. DKK). DO NOT use it as the SEK amount.
- "Köpesumma" (buy) — or "Försäljningssumma" / "Likvid" (sell) — is the SEK value
  of the shares, EXCLUDING fees. This is the cost/proceeds basis.
- Fees are in SEK and there may be SEVERAL: sum ALL of "Prel. courtage" /
  "Courtage" and "Prel. växlingsavgift" / "Växlingsavgift" (FX fee).

Return ONLY a valid JSON object — no other text, no markdown fences:
{
  "isin": "12-char ISIN (e.g. DK0062498333) — most reliable identifier, or null",
  "company_name": "company name as shown",
  "side": "buy" or "sell",
  "shares": integer number of shares (Antal),
  "amount_sek": SEK value of the shares excl. fees (Köpesumma / sale proceeds),
  "fee_sek": TOTAL fees in SEK = courtage + växlingsavgift (sum every fee line),
  "native_price": execution price per share in native currency (Avslutskurs), or null,
  "native_currency": "DKK" / "SEK" / "EUR" / etc. for the Kurs, or null,
  "trade_date": "YYYY-MM-DD from Senaste avslut / Affärsdatum, or null",
  "confidence": float 0.0-1.0
}
Set any field to null if it cannot be reliably determined.\
"""


async def _extract_fill_from_ocr(ocr_text: str, model: str, api_key: str) -> str:
    """Call cheap text LLM to extract fill details from OCR text. Returns raw response string."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key)
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _OCR_SYSTEM_PROMPT},
            {"role": "user",   "content": f"OCR text from broker screenshot:\n\n{ocr_text}"},
        ],
        max_tokens=300,
    )
    return resp.choices[0].message.content or ""


def _parse_fill_json(raw: str) -> dict | None:
    """Extract and parse the JSON object from an LLM response string."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def _ocr_image(buf: BytesIO) -> str:
    """Run pytesseract OCR on image bytes. Returns extracted text."""
    from PIL import Image, ImageEnhance
    import pytesseract

    img = Image.open(buf).convert("RGB")
    # Upscale small screenshots for better OCR accuracy
    if img.width < 1000:
        scale = 1000 / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    # Grayscale + contrast boost helps on dark broker UIs
    img = ImageEnhance.Contrast(img.convert("L")).enhance(2.0)
    return pytesseract.image_to_string(img, lang="swe+eng")


async def photo_handler(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    """Handle photo messages — OCR locally, then extract fill details via cheap text LLM."""
    await update.message.reply_text("🔍 Reading screenshot…")

    # Download highest-resolution photo Telegram provides
    photo = update.message.photo[-1]
    tg_file = await context.bot.get_file(photo.file_id)
    buf = BytesIO()
    await tg_file.download_to_memory(buf)

    # Step 1: local OCR (free, no API)
    try:
        ocr_text = _ocr_image(buf)
    except ImportError:
        await update.message.reply_text(
            "❌ pytesseract / Pillow not installed.\n"
            "On the Pi run:\n"
            "  sudo apt install tesseract-ocr tesseract-ocr-swe\n"
            "  uv pip install pytesseract Pillow"
        )
        return
    except Exception as e:
        await update.message.reply_text(f"❌ OCR failed: {e}\nUse /fill manually.")
        return

    if not ocr_text.strip():
        await update.message.reply_text(
            "❌ OCR produced no text — screenshot may be too small or low contrast.\n"
            "Use /fill TICKER SHARES PRICE FEE manually."
        )
        return

    # Step 2: cheap text LLM for structured extraction
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        await update.message.reply_text("❌ OPENAI_API_KEY not set.")
        return

    model = os.getenv("FUND_OCR_MODEL", "gpt-4o-mini")
    try:
        raw = await _extract_fill_from_ocr(ocr_text, model, api_key)
    except Exception as e:
        await update.message.reply_text(f"❌ LLM extraction error: {e}\nUse /fill manually.")
        return
    data = _parse_fill_json(raw)
    if not data:
        await update.message.reply_text(
            "❌ Could not parse fill details from screenshot.\n"
            "Use /fill TICKER SHARES PRICE FEE manually."
        )
        return

    # Derive the SEK per-share price from the SEK amount (Köpesumma): the broker
    # settles in SEK, so cost basis is amount_sek / shares — NOT the native Kurs.
    try:
        sh = float(data.get("shares") or 0)
        amt = data.get("amount_sek")
        if amt is not None and sh > 0:
            data["price_sek"] = round(float(amt) / sh, 4)
    except (TypeError, ValueError):
        pass
    data.setdefault("price_sek", None)

    # ISIN → name fallback → Yahoo ticker lookup
    isin = (data.get("isin") or "").strip().upper()
    company_name = (data.get("company_name") or "").strip()
    ticker = None
    isin_status = ""

    if isin:
        isin_map = _get_isin_map()
        ticker = isin_map.get(isin)
        if ticker:
            isin_status = "✅ matched by ISIN"
        else:
            # Fallback: decision-aware name match
            ticker, match_source = _name_to_ticker(company_name)
            if ticker:
                isin_status = f"✅ {match_source}"
            else:
                isin_status = "⚠️ not found in universe"

    confidence = float(data.get("confidence") or 0.0)
    conf_bar   = "🟢" if confidence >= 0.85 else "🟡" if confidence >= 0.60 else "🔴"

    trade_date = (data.get("trade_date") or "").strip() or None

    lines = ["🧾 <b>Extracted from screenshot</b>\n"]
    lines.append(f"Company: {company_name or '?'}")
    if isin:
        lines.append(f"ISIN: <code>{isin}</code>  {isin_status}")
    lines.append(f"Ticker: <b>{ticker or '?'}</b>")
    lines.append(f"Side: <b>{(data.get('side') or '?').upper()}</b>")
    lines.append(f"Shares: <b>{data.get('shares') or '?'}</b>")
    price_line = f"Price: <b>{data.get('price_sek') or '?'} SEK/sh</b>"
    nc = (data.get("native_currency") or "").upper()
    if nc and nc != "SEK" and data.get("native_price"):
        price_line += f"  (Kurs {data.get('native_price')} {nc})"
    lines.append(price_line)
    if data.get("amount_sek"):
        lines.append(f"Amount: <b>{data.get('amount_sek')} SEK</b>")
    lines.append(f"Fee: <b>{data.get('fee_sek') or '?'} SEK</b>  (courtage + FX)")
    if trade_date:
        lines.append(f"Date: <b>{trade_date}</b>")
    lines.append(f"\n{conf_bar} Confidence: {confidence:.0%}")

    chat_id = update.effective_chat.id

    if not ticker:
        # Store pending fill without ticker — ask user to reply with it
        _pending_fills[chat_id] = {
            "ticker":     None,
            "side":       data.get("side", "buy"),
            "shares":     data.get("shares"),
            "price":      data.get("price_sek"),
            "fee":        data.get("fee_sek", 0),
            "trade_date": trade_date,
        }
        lines.append("\n📝 Reply with the ticker (e.g. <code>SEYE.ST</code>) to confirm.")
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")
        return

    # Store pending fill and show confirm / cancel buttons
    _pending_fills[chat_id] = {
        "ticker":     ticker,
        "side":       data.get("side", "buy"),
        "shares":     data.get("shares"),
        "price":      data.get("price_sek"),
        "fee":        data.get("fee_sek", 0),
        "trade_date": trade_date,
    }

    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("✓ Record fill", callback_data="fill_confirm"),
        InlineKeyboardButton("✗ Cancel",      callback_data="fill_cancel"),
    ]])

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode="HTML",
        reply_markup=keyboard,
    )


async def fill_callback(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    """Handle confirm / cancel button presses from the photo fill flow."""
    query = update.callback_query
    await query.answer()
    chat_id = update.effective_chat.id

    if query.data == "fill_cancel":
        _pending_fills.pop(chat_id, None)
        await query.edit_message_text("Cancelled. Use /fill manually if needed.")
        return

    data = _pending_fills.pop(chat_id, None)
    if not data:
        await query.edit_message_text("No pending fill — please send a new screenshot.")
        return

    ticker     = data.get("ticker", "")
    side       = data.get("side", "buy")
    shares     = data.get("shares")
    price      = data.get("price")
    fee        = data.get("fee", 0)
    trade_date = data.get("trade_date")

    if not all([ticker, shares, price]):
        await query.edit_message_text(
            "❌ Missing required fields.\n"
            "Use /fill TICKER SHARES PRICE FEE manually."
        )
        return

    cli_args = ["fill", ticker, str(shares), str(price), str(fee), "--side", side]
    if trade_date:
        cli_args += ["--date", trade_date]
    output = _run_cli(*cli_args, timeout=30)
    await query.edit_message_text(f"✅ Fill recorded!\n\n{output}")


async def text_handler(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    """Catch a plain ticker reply when a pending fill is waiting for one."""
    chat_id = update.effective_chat.id
    pending = _pending_fills.get(chat_id)
    if not pending or pending.get("ticker"):
        return  # not waiting for a ticker — ignore

    text = (update.message.text or "").strip().upper()
    # Basic ticker sanity check: letters, digits, dots, dashes
    if not re.match(r'^[A-Z0-9][A-Z0-9\-\.]{1,15}$', text):
        await update.message.reply_text("That doesn't look like a ticker. Try again (e.g. <code>SEYE.ST</code>).", parse_mode="HTML")
        return

    pending["ticker"] = text
    _pending_fills[chat_id] = pending

    data = pending
    lines = [f"🧾 <b>Confirm fill</b>\n"]
    lines.append(f"Ticker: <b>{text}</b>")
    lines.append(f"Side: <b>{data['side'].upper()}</b>")
    lines.append(f"Shares: <b>{data['shares']}</b>")
    lines.append(f"Price: <b>{data['price']} SEK</b>")
    lines.append(f"Fee: <b>{data['fee']} SEK</b>")
    if data.get("trade_date"):
        lines.append(f"Date: <b>{data['trade_date']}</b>")

    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("✓ Record fill", callback_data="fill_confirm"),
        InlineKeyboardButton("✗ Cancel",      callback_data="fill_cancel"),
    ]])
    await update.message.reply_text("\n".join(lines), parse_mode="HTML", reply_markup=keyboard)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if not _TELEGRAM_AVAILABLE:
        log.error("python-telegram-bot not installed. Run: uv pip install python-telegram-bot")
        sys.exit(1)

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        log.error("TELEGRAM_BOT_TOKEN not set in .env")
        sys.exit(1)

    log.info("Starting AI Fund Manager bot…")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("run",      cmd_run))
    app.add_handler(CommandHandler("run_full", cmd_run_full))
    app.add_handler(CommandHandler("fill",     cmd_fill))
    app.add_handler(CommandHandler("status",   cmd_status))
    app.add_handler(CommandHandler("report",   cmd_report))
    app.add_handler(CommandHandler("stops",    cmd_stops))
    app.add_handler(CommandHandler("universe", cmd_universe))
    app.add_handler(CommandHandler("reject_rates", cmd_reject_rates))
    app.add_handler(CommandHandler("review",   cmd_review))
    app.add_handler(CommandHandler("setcash",  cmd_setcash))
    app.add_handler(CommandHandler("help",     cmd_help))
    app.add_handler(CommandHandler("start",    cmd_help))

    # Screenshot fill extraction
    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    app.add_handler(CallbackQueryHandler(fill_callback, pattern="^fill_"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    log.info("Bot polling… (Ctrl+C to stop)")
    app.run_polling(allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    main()
