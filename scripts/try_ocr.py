#!/usr/bin/env python3
"""
Interactive OCR fill extraction tester.

Usage:
    .venv/bin/python scripts/try_ocr.py path/to/screenshot.png
    .venv/bin/python scripts/try_ocr.py path/to/screenshot.png --model gpt-4o-mini

Shows each step of the pipeline so you can see exactly what comes out.
Requires: tesseract-ocr, pytesseract, Pillow, openai, python-dotenv
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from io import BytesIO
from pathlib import Path

# Load .env so OPENAI_API_KEY etc. are available
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def hr(label: str = "") -> None:
    width = 60
    if label:
        pad = (width - len(label) - 2) // 2
        print(f"\n{'─' * pad} {label} {'─' * pad}")
    else:
        print("─" * width)


def step_ocr(image_path: Path) -> str:
    try:
        from PIL import Image, ImageEnhance
        import pytesseract
    except ImportError:
        print("❌  Missing deps. Install with:")
        print("      brew install tesseract          # macOS")
        print("      sudo apt install tesseract-ocr  # Pi/Ubuntu")
        print("      uv pip install pytesseract Pillow")
        sys.exit(1)

    hr("STEP 1 — OCR  (local, free)")
    print(f"    Image: {image_path}")

    with open(image_path, "rb") as f:
        buf = BytesIO(f.read())

    img = Image.open(buf).convert("RGB")
    print(f"    Size:  {img.width} × {img.height} px")

    # Upscale + contrast boost (same as bot)
    if img.width < 1000:
        scale = 1000 / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        print(f"    Upscaled to {img.width} × {img.height} px")

    img = ImageEnhance.Contrast(img.convert("L")).enhance(2.0)

    # Try Swedish + English; fall back to English only
    try:
        text = pytesseract.image_to_string(img, lang="swe+eng")
    except pytesseract.TesseractError:
        print("    ⚠  Swedish lang pack not found, using English only")
        text = pytesseract.image_to_string(img, lang="eng")

    print("\n    Raw OCR output:")
    for line in text.splitlines():
        if line.strip():
            print(f"      {line}")

    # Highlight any ISIN-shaped strings found
    isins = re.findall(r"\b[A-Z]{2}[A-Z0-9]{10}\b", text)
    if isins:
        print(f"\n    ISINs detected: {', '.join(isins)}")
    else:
        print("\n    ⚠  No ISIN pattern detected in OCR text")

    return text


def step_llm(ocr_text: str, model: str) -> dict | None:
    hr(f"STEP 2 — LLM extraction  ({model})")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌  OPENAI_API_KEY not set in .env")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("❌  openai not installed — uv pip install openai")
        sys.exit(1)

    SYSTEM = """\
You are extracting trade fill details from OCR text from a Swedish/Nordic broker \
confirmation screen (Montrose or similar). The text may have minor OCR artifacts.

Return ONLY a valid JSON object:
{
  "isin": "12-character ISIN or null",
  "company_name": "as shown",
  "side": "buy" or "sell",
  "shares": integer,
  "price_sek": per-share price as float,
  "fee_sek": commission in SEK as float,
  "trade_date": "YYYY-MM-DD from Affärsdatum/trade date field, or null",
  "confidence": 0.0-1.0
}
Set any field to null if it cannot be determined.\
"""

    print(f"    Sending OCR text to {model}…")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": f"OCR text:\n\n{ocr_text}"},
        ],
        max_tokens=300,
    )
    raw = resp.choices[0].message.content or ""
    tokens_used = resp.usage.total_tokens if resp.usage else "?"

    print(f"\n    Raw LLM response ({tokens_used} tokens):")
    print(f"      {raw}")

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        print("\n    ❌  No JSON object found in response")
        return None

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as e:
        print(f"\n    ❌  JSON parse error: {e}")
        return None

    return data


def step_isin_lookup(data: dict) -> str | None:
    hr("STEP 3 — ISIN lookup  (universe.csv)")

    from fundmgr.config import get_isin_map
    isin_map = get_isin_map()

    isin = (data.get("isin") or "").strip().upper()
    if not isin:
        print("    ⚠  No ISIN in extracted data — cannot look up ticker")
        return None

    ticker = isin_map.get(isin)
    if ticker:
        print(f"    {isin}  →  {ticker}  ✅")
    else:
        print(f"    {isin}  →  not in universe  ⚠")
        print("    Use /fill TICKER manually, or add to universe.csv")

    return ticker


def step_summary(data: dict, ticker: str | None) -> None:
    hr("RESULT")

    conf = float(data.get("confidence") or 0)
    conf_icon = "🟢" if conf >= 0.85 else "🟡" if conf >= 0.60 else "🔴"

    trade_date = (data.get("trade_date") or "").strip() or None

    rows = [
        ("Ticker",     ticker or "⚠  unknown"),
        ("ISIN",       data.get("isin")         or "—"),
        ("Company",    data.get("company_name") or "—"),
        ("Side",       (data.get("side") or "—").upper()),
        ("Shares",     str(data.get("shares")   or "—")),
        ("Price/sh",   f"{data.get('price_sek') or '—'} SEK"),
        ("Fee",        f"{data.get('fee_sek')   or '—'} SEK"),
        ("Trade date", trade_date or "—"),
        ("Confidence", f"{conf_icon}  {conf:.0%}"),
    ]

    for label, value in rows:
        print(f"    {label:<14}{value}")

    if ticker and all(data.get(k) for k in ("side", "shares", "price_sek", "fee_sek")):
        date_flag = f" --date {trade_date}" if trade_date else ""
        cmd = (
            f"\n    fund fill {ticker} {data['shares']} "
            f"{data['price_sek']} {data['fee_sek']} --side {data.get('side','buy')}{date_flag}"
        )
        print(f"\n    CLI command to record this fill:{cmd}")
    hr()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test OCR fill extraction on a real screenshot")
    parser.add_argument("image", type=Path, help="Path to broker confirmation screenshot")
    parser.add_argument(
        "--model", default=os.getenv("FUND_OCR_MODEL", "gpt-4o-mini"),
        help="OpenAI model for text extraction (default: gpt-4o-mini)"
    )
    args = parser.parse_args()

    if not args.image.exists():
        print(f"❌  File not found: {args.image}")
        sys.exit(1)

    print(f"\n🔍  OCR Fill Extraction Test")
    print(f"    File:  {args.image.name}")
    print(f"    Model: {args.model}")

    ocr_text = step_ocr(args.image)
    data = step_llm(ocr_text, args.model)

    if data is None:
        print("\n❌  Extraction failed — check OCR output above")
        sys.exit(1)

    ticker = step_isin_lookup(data)
    step_summary(data, ticker)


if __name__ == "__main__":
    main()
