"""
Portfolio photo → holdings extraction.

Feeds one or more screenshots/photos of a broker holdings screen (Montrose,
Avanza, Nordnet, IBKR…) through a vision LLM and returns structured holdings
ready to seed a live sleeve.

Two signals are combined:

  1. **Local OCR** (pytesseract, free) — a text transcription appended to the
     prompt. It reliably nails ISINs and digit strings that vision models
     occasionally soften, and costs nothing. Skipped silently when tesseract
     isn't installed.
  2. **Vision** (gpt-4o-mini by default) — reads the actual table layout, so
     rows stay intact and columns don't get shuffled the way a flat OCR dump
     does.

Ticker resolution reuses the same ladder as the rest of the app: ISIN map from
universe.csv → broker→Yahoo symbol map → company-name alias → Yahoo search.

The photo is *evidence*, not truth: every row comes back with a confidence and
is meant to land in a review table the user edits before anything is written.
"""
from __future__ import annotations

import base64
import json
import os
import re
from io import BytesIO

# Vision reads the table; OCR text is a supporting hint in the same prompt.
DEFAULT_VISION_MODEL = "gpt-4o-mini"

# Cap on images per extraction call — a holdings screen is 1–3 screenshots;
# more than this is almost certainly a mistake and burns tokens.
MAX_IMAGES = 8

# Longest edge each image is downscaled to before upload. Broker tables stay
# legible well below full retina resolution, and tokens scale with pixels.
_MAX_EDGE = 1600

_SYSTEM_PROMPT = """\
You read screenshots of stock brokerage portfolio/holdings screens and return the \
holdings as structured data. The screenshots may be in Swedish, Norwegian, Danish, \
Finnish, German or English, and may come from Montrose, Avanza, Nordnet, \
Interactive Brokers, Degiro or similar.

Return ONLY a JSON object of this exact shape:

{
  "holdings": [
    {
      "name": "company name as shown",
      "ticker": "broker ticker as shown, or null",
      "isin": "12-character ISIN if visible, else null",
      "shares": the SHARE COUNT for this row — the "Antal" / "Antal aktier" /
                "Quantity" column. This is a COUNT of shares, almost always a
                whole number, and it is NEVER a monetary amount. If the screen
                does not show a share count, return null. Do NOT substitute the
                position's value, cost or weight: a null here is correct and
                recoverable, a value here silently corrupts the whole import.
      "avg_cost_native": average purchase price per share in the row's own
                         currency (GAV / Genomsnittligt anskaffningsvärde /
                         Inköpskurs / Avg price), else null,
      "last_price_native": current/last price per share as shown, else null,
      "market_value_sek": position market value in SEK if the row shows a SEK
                          value (Marknadsvärde / Värde), else null,
      "cost_value_sek": total purchase value in SEK (Inköpsvärde / GAV × antal)
                        if shown, else null,
      "currency": "SEK" | "USD" | "EUR" | "NOK" | "DKK" | ... for the price
                  columns of this row,
      "weight_pct": portfolio weight percent if shown, else null,
      "confidence": 0.0-1.0 — how sure you are of THIS row's numbers
    }
  ],
  "cash_sek": free cash / likvida medel in SEK if visible, else null,
  "total_value_sek": total portfolio value in SEK if visible, else null,
  "account_name": account or portfolio name if visible, else null,
  "notes": "anything ambiguous, cut off, or unreadable — one short sentence"
}

Rules:
- One object per holding row. Do not invent rows, and do not merge rows.
- `shares` and `market_value_sek` must never be the same number. If you are
  tempted to put a position's value in `shares`, the share count is not visible
  — return null for it.
- Copy numbers exactly as displayed. Nordic screens use space or dot as the
  thousands separator and comma as the decimal separator: "1 234,50" is 1234.50,
  "12.500" is 12500. Strip currency symbols and % signs.
- A number in parentheses or with a leading minus is negative.
- Do NOT convert between currencies. Leave a field null rather than guessing.
- Skip summary/total rows, funds-in-transit rows and cash rows — cash goes in
  `cash_sek`, not in `holdings`.
- If a value is cut off or illegible, use null and lower that row's confidence.
"""


class PhotoExtractionError(RuntimeError):
    """Raised when the photos could not be turned into holdings at all."""


# ── Image prep ────────────────────────────────────────────────────────────────

def _prepare_image(raw: bytes) -> tuple[str, str]:
    """Downscale/normalise an uploaded image. Returns (base64_data, mime).

    Falls back to shipping the original bytes when Pillow isn't importable so a
    missing image library degrades quality, not availability.
    """
    try:
        from PIL import Image
    except ImportError:
        return base64.b64encode(raw).decode(), _guess_mime(raw)

    try:
        img = Image.open(BytesIO(raw))
        img.load()
    except Exception as e:
        raise PhotoExtractionError(f"Could not read image: {e}") from e

    # Drop alpha (JPEG can't carry it) and flatten onto white so dark-mode
    # screenshots with transparency don't come out black-on-black.
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")

    longest = max(img.width, img.height)
    if longest > _MAX_EDGE:
        scale = _MAX_EDGE / longest
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                         Image.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"


def _guess_mime(raw: bytes) -> str:
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:2] == b"\xff\xd8":
        return "image/jpeg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def ocr_text(raw: bytes) -> str:
    """Best-effort local OCR transcription of one image. '' when unavailable.

    Mirrors the Telegram fill-screenshot pipeline: upscale small images, boost
    contrast for dark broker UIs, Swedish+English language packs.
    """
    try:
        from PIL import Image, ImageEnhance
        import pytesseract
    except ImportError:
        return ""
    try:
        img = Image.open(BytesIO(raw)).convert("RGB")
        if img.width < 1000:
            scale = 1000 / img.width
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        img = ImageEnhance.Contrast(img.convert("L")).enhance(2.0)
        try:
            return pytesseract.image_to_string(img, lang="swe+eng")
        except Exception:
            return pytesseract.image_to_string(img, lang="eng")
    except Exception:
        return ""


# ── Number coercion ───────────────────────────────────────────────────────────

_NUM_CLEAN_RE = re.compile(r"[^\d,.\-]")


def to_float(value) -> float | None:
    """Parse a number the way a Nordic broker screen writes it.

    Handles "1 234,50" (space thousands, comma decimal), "1.234,50" (dot
    thousands), "1,234.50" (US), "-12,5", "(3 000)" and bare floats. Returns
    None for anything that isn't a number.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = _NUM_CLEAN_RE.sub("", text)
    if not text or text in ("-", ".", ","):
        return None
    if "," in text and "." in text:
        # Whichever separator comes last is the decimal point.
        if text.rindex(",") > text.rindex("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        # A single comma with 1-2 trailing digits is a decimal comma;
        # "1,234" with exactly 3 trailing digits is a thousands separator.
        head, _, tail = text.rpartition(",")
        text = f"{head}.{tail}" if len(tail) != 3 or not head else head + tail
    try:
        out = float(text)
    except ValueError:
        return None
    return -out if negative and out > 0 else out


# ── Extraction ────────────────────────────────────────────────────────────────

def _vision_model() -> str:
    return os.getenv("FUND_VISION_MODEL") or os.getenv("FUND_OCR_MODEL") or DEFAULT_VISION_MODEL


def extract_holdings(
    images: list[bytes],
    *,
    model: str | None = None,
    api_key: str | None = None,
    use_ocr: bool = True,
    resolve: bool = True,
) -> dict:
    """Read portfolio holdings out of one or more photos.

    Returns {holdings, cash_sek, total_value_sek, account_name, notes,
    warnings, ocr_used, model}. Each holding carries the raw extracted fields
    plus a resolved Yahoo `ticker` (None when it couldn't be resolved) and
    `avg_cost_sek` where the screen gave enough to derive it.

    resolve=False skips all ticker lookups (no network) — used by tests.
    Raises PhotoExtractionError when no holdings could be read at all.
    """
    if not images:
        raise PhotoExtractionError("No images uploaded.")
    if len(images) > MAX_IMAGES:
        raise PhotoExtractionError(
            f"Too many images ({len(images)}) — upload at most {MAX_IMAGES} per portfolio.")

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise PhotoExtractionError(
            "OPENAI_API_KEY is not set — photo import needs it to read the screenshot.")

    prepared = [_prepare_image(raw) for raw in images]

    transcripts = [t for t in ((ocr_text(raw) if use_ocr else "") for raw in images) if t.strip()]
    ocr_block = ""
    if transcripts:
        joined = "\n\n".join(f"--- image {i + 1} ---\n{t.strip()}"
                             for i, t in enumerate(transcripts))
        ocr_block = (
            "\n\nA local OCR pass produced the transcription below. It is noisy and the "
            "column order may be scrambled, but ISINs and digit strings in it are usually "
            "accurate — use it to check the numbers you read from the image:\n\n"
            f"{joined[:12000]}"
        )

    content: list[dict] = [{
        "type": "text",
        "text": (
            f"Extract every holding from {'these ' + str(len(images)) + ' screenshots' if len(images) > 1 else 'this screenshot'} "
            "of my brokerage portfolio. If several screenshots show the same account "
            "scrolled to different rows, merge them and drop duplicate rows."
            + ocr_block
        ),
    }]
    for b64, mime in prepared:
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

    chosen = model or _vision_model()
    raw_answer = _call_vision(chosen, key, content)
    data = _parse_json(raw_answer)
    if data is None:
        raise PhotoExtractionError(
            "The model did not return readable JSON for this photo. Try a sharper, "
            "straight-on screenshot of the holdings table.")

    holdings, warnings = _normalise_rows(data.get("holdings") or [])
    if not holdings:
        raise PhotoExtractionError(
            "No holdings could be read from the photo. Make sure the holdings table "
            "(name, shares, price) is fully visible and in focus.")
    if resolve:
        holdings = _resolve_tickers(holdings)

    total_value = to_float(data.get("total_value_sek"))
    warnings += _sanity_check(holdings, total_value)

    return {
        "holdings": holdings,
        "cash_sek": to_float(data.get("cash_sek")),
        "total_value_sek": total_value,
        "account_name": (data.get("account_name") or "").strip() or None,
        "notes": (data.get("notes") or "").strip(),
        "warnings": warnings,
        "ocr_used": bool(transcripts),
        "model": chosen,
    }


def _call_vision(model: str, api_key: str, content: list[dict]) -> str:
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover - dependency is declared
        raise PhotoExtractionError("The openai package is not installed.") from e

    client = OpenAI(api_key=api_key)
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "max_tokens": 3000,
        "temperature": 0.0,
    }
    try:
        resp = client.chat.completions.create(
            response_format={"type": "json_object"}, **kwargs)
    except Exception:
        # Older models / gateways reject response_format — retry without it and
        # fall back to pulling the JSON object out of the prose.
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            raise PhotoExtractionError(f"Vision model call failed: {e}") from e
    return resp.choices[0].message.content or ""


def _parse_json(raw: str) -> dict | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    for candidate in (raw, _first_json_object(raw)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"holdings": parsed}
    return None


def _first_json_object(text: str) -> str | None:
    """Brace-matched extraction of the first JSON object — tolerates fences/prose."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


_ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")

# Rows a broker draws inside the holdings table that aren't positions.
_NON_POSITION = {
    "total", "totalt", "summa", "sum", "cash", "likvida medel", "likvider",
    "saldo", "kontanter", "portfolio", "portfölj", "portfolio total",
    "totalt värde", "market value", "grand total",
}


def _normalise_rows(rows: list) -> tuple[list[dict], list[str]]:
    """Coerce raw model rows into clean holdings; collect per-row warnings."""
    out: list[dict] = []
    warnings: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        raw_ticker = str(row.get("ticker") or "").strip().upper()
        if not name and not raw_ticker:
            continue
        if name.lower() in _NON_POSITION:
            continue

        isin = str(row.get("isin") or "").strip().upper().replace(" ", "")
        if isin and not _ISIN_RE.match(isin):
            warnings.append(f"{name or raw_ticker}: '{isin}' is not a valid ISIN — ignored.")
            isin = ""

        shares = to_float(row.get("shares"))
        avg_native = to_float(row.get("avg_cost_native"))
        last_native = to_float(row.get("last_price_native"))
        value_sek = to_float(row.get("market_value_sek"))
        cost_sek = to_float(row.get("cost_value_sek"))
        currency = (str(row.get("currency") or "").strip().upper() or None)

        # A share count equal to the row's own money column means the value
        # column was read as the count. Left alone this is catastrophic and
        # silent: cost derives as value/shares = 1.00, and the dashboard then
        # multiplies a currency amount by a real share price. Drop the count
        # rather than carry a number we know is a different quantity.
        for money, label in ((value_sek, "market value"), (cost_sek, "purchase value")):
            if shares and money and abs(shares - money) < max(0.01, abs(money) * 1e-6):
                warnings.append(
                    f"{name or raw_ticker}: the share count matched this row's "
                    f"{label} exactly — read as a value, not a count. Cleared; "
                    "enter the share count yourself.")
                shares = None
                break

        # Cost basis in SEK, most trustworthy source first: the broker's own
        # SEK purchase value ÷ shares beats the native GAV, which would need an
        # FX rate we don't have for the trade date. Same reasoning as the
        # Telegram fill flow (see `fund paper-setcost`).
        avg_cost_sek = None
        basis = None
        if cost_sek and shares:
            avg_cost_sek, basis = cost_sek / shares, "cost_value_sek"
        elif avg_native and currency == "SEK":
            avg_cost_sek, basis = avg_native, "avg_cost_native"
        elif value_sek and shares and not avg_native:
            # No cost anywhere — seed at market so the position at least exists;
            # flagged so the review table can say so.
            avg_cost_sek, basis = value_sek / shares, "market_value_sek"

        confidence = to_float(row.get("confidence"))
        if confidence is None:
            confidence = 0.5
        confidence = min(1.0, max(0.0, confidence))

        if not shares:
            warnings.append(f"{name or raw_ticker}: no share count read — enter it manually.")
        if avg_cost_sek is None:
            warnings.append(f"{name or raw_ticker}: no cost basis read — enter it manually.")

        out.append({
            "name": name or raw_ticker,
            "raw_ticker": raw_ticker,
            "ticker": None,
            "isin": isin,
            "shares": shares,
            "avg_cost_sek": round(avg_cost_sek, 4) if avg_cost_sek else None,
            "avg_cost_basis": basis,
            "avg_cost_native": avg_native,
            "last_price_native": last_native,
            "market_value_sek": value_sek,
            "currency": currency,
            "weight_pct": to_float(row.get("weight_pct")),
            "confidence": round(confidence, 2),
            "resolved_via": None,
        })
    return out, warnings


def _sanity_check(holdings: list[dict], total_value_sek: float | None) -> list[str]:
    """Whole-portfolio checks for the failure modes a per-row look won't catch.

    A column misread the same way on every row leaves each row individually
    plausible, so the only place it shows up is in the totals.
    """
    out: list[str] = []
    costed = [h for h in holdings if h["shares"] and h["avg_cost_sek"]]
    if not costed:
        return out

    # A cost basis of exactly 1.00 is not a price, it is value/value.
    unit_cost = [h for h in costed if abs(h["avg_cost_sek"] - 1.0) < 1e-9]
    if len(unit_cost) == len(costed) and len(costed) > 1:
        out.append(
            "Every row came out at a cost basis of exactly 1.00 — the share-count "
            "and value columns were almost certainly confused. Check the share "
            "counts against the broker before importing.")

    # The implied book should reconcile with the total the screen reported.
    implied = sum(h["shares"] * h["avg_cost_sek"] for h in costed)
    if total_value_sek and implied > 0:
        ratio = implied / total_value_sek
        if ratio > 1.5 or ratio < 0.5:
            out.append(
                f"The rows add up to {implied:,.0f} SEK but the screen's total reads "
                f"{total_value_sek:,.0f} SEK — off by {ratio:.1f}×. Something was "
                "read from the wrong column.")
    return out


# Trading-currency → the Yahoo suffixes that currency's home listings use, best
# first. A holdings screen in one market resolves to that market's listing: a
# Swedish ISK holding AstraZeneca means AZN.ST, not AZN.L or a Frankfurt line.
_CURRENCY_SUFFIXES: dict[str, tuple[str, ...]] = {
    "SEK": (".ST",),
    "NOK": (".OL",),
    "DKK": (".CO",),
    "EUR": (".HE", ".AS", ".DE", ".PA", ".BR", ".LS", ".MI"),
    "GBP": (".L",),
    "GBX": (".L",),
    "CHF": (".SW",),
    "CAD": (".TO", ".V"),
}


def _search_symbol_preferring(name: str, currency: str | None) -> str | None:
    """Yahoo symbol search that prefers the row's own market.

    Plain search returns whichever listing Yahoo ranks first, which for Nordic
    small caps is routinely a thin Frankfurt or London line quoted in another
    currency — the price then looks real and is wrong. Preferring the suffix
    implied by the row's currency keeps the instrument the one actually held.
    """
    from fundmgr import paper

    cur = (currency or "").upper()
    preferred = _CURRENCY_SUFFIXES.get(cur, ())
    # US listings carry no suffix, so USD expresses its preference as "bare".
    prefer_bare = cur == "USD"

    if preferred or prefer_bare:
        try:
            import yfinance as yf
            quotes = yf.Search(name, max_results=10).quotes or []
        except Exception:
            quotes = []
        candidates = [
            str(q["symbol"]).upper() for q in quotes
            if q.get("symbol") and str(q.get("quoteType", "")).upper() in ("EQUITY", "ETF")
        ]
        for suffix in preferred:
            for symbol in candidates:
                if symbol.endswith(suffix):
                    return symbol
        if prefer_bare:
            for symbol in candidates:
                if "." not in symbol:
                    return symbol

    return paper._search_symbol(name)


def _resolve_tickers(holdings: list[dict]) -> list[dict]:
    """Resolve each row to a Yahoo symbol: ISIN → broker map → alias → search."""
    from fundmgr import paper

    try:
        from fundmgr.config import get_isin_map
        isin_map = get_isin_map()
    except Exception:
        isin_map = {}

    unresolved: list[dict] = []
    for h in holdings:
        if h["isin"] and isin_map.get(h["isin"]):
            h["ticker"] = isin_map[h["isin"]]
            h["resolved_via"] = "isin"
            continue
        if h["raw_ticker"]:
            h["ticker"] = paper.montrose_to_yahoo(
                h["raw_ticker"], h.get("currency"), h.get("name"))
            h["resolved_via"] = "broker ticker"
            continue
        unresolved.append(h)

    for h in unresolved:
        # Alias map first (it already carries home listings), then a
        # market-aware search so a Nordic name doesn't land on a Frankfurt line.
        alias = paper._lookup_alias(paper._clean_name(h["name"]))
        if alias:
            h["ticker"] = alias
            h["resolved_via"] = "name"
            continue
        symbol = _search_symbol_preferring(h["name"], h.get("currency"))
        if symbol:
            h["ticker"] = symbol
            h["resolved_via"] = "search"
    return holdings
