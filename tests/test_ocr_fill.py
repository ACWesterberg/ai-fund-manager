"""
Tests for the screenshot → OCR → fill extraction pipeline.

Layers tested:
  1. ISIN map (no external deps)
  2. JSON parsing (no external deps)
  3. OCR on a synthetic image (requires: tesseract + pytesseract + Pillow)
  4. Full photo_handler flow with mocked LLM (requires: tesseract)

Run all:       pytest tests/test_ocr_fill.py -v
Run fast only: pytest tests/test_ocr_fill.py -v -m "not ocr"
"""
from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_broker_image(text: str) -> BytesIO:
    """Generate a white PNG with broker-style text for OCR testing."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (800, 500), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Use a system font if available; fall back to PIL default
    font = None
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",           # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux / Pi
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]:
        try:
            font = ImageFont.truetype(path, 28)
            break
        except (IOError, OSError):
            continue
    if font is None:
        font = ImageFont.load_default()

    draw.text((40, 40), text, fill=(0, 0, 0), font=font)
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# A realistic-looking (simplified) Montrose confirmation
SAMPLE_OCR_TEXT = """\
GENOMFORD KOPSORDER

Instrument: Volvo B
ISIN: SE0000115446
Antal aktier: 12
Kurs: 291.50 SEK
Courtage: 2.91 SEK
Totalt: 3500.91 SEK
"""

SAMPLE_LLM_JSON = {
    "isin": "SE0000115446",
    "company_name": "Volvo B",
    "side": "buy",
    "shares": 12,
    "price_sek": 291.50,
    "fee_sek": 2.91,
    "confidence": 0.95,
}


# ── 1. ISIN map ───────────────────────────────────────────────────────────────

def test_isin_map_loads():
    from fundmgr.config import get_isin_map
    m = get_isin_map()
    assert len(m) > 0, "ISIN map should not be empty"


def test_isin_resolves_volvo():
    from fundmgr.config import get_isin_map
    assert get_isin_map().get("SE0000115446") == "VOLV-B.ST"


def test_isin_resolves_sandvik():
    from fundmgr.config import get_isin_map
    assert get_isin_map().get("SE0000667891") == "SAND.ST"


def test_isin_unknown_returns_none():
    from fundmgr.config import get_isin_map
    assert get_isin_map().get("XX0000000000") is None


# ── 2. JSON parsing ───────────────────────────────────────────────────────────

def test_parse_fill_json_clean():
    from fundmgr.notify.telegram_bot import _parse_fill_json
    raw = json.dumps(SAMPLE_LLM_JSON)
    data = _parse_fill_json(raw)
    assert data is not None
    assert data["isin"] == "SE0000115446"
    assert data["shares"] == 12
    assert data["price_sek"] == pytest.approx(291.50)


def test_parse_fill_json_with_preamble():
    """LLM sometimes wraps JSON in prose — regex should still extract it."""
    from fundmgr.notify.telegram_bot import _parse_fill_json
    raw = f"Here are the details:\n{json.dumps(SAMPLE_LLM_JSON)}\nEnd."
    data = _parse_fill_json(raw)
    assert data is not None
    assert data["isin"] == "SE0000115446"


def test_parse_fill_json_invalid_returns_none():
    from fundmgr.notify.telegram_bot import _parse_fill_json
    assert _parse_fill_json("No JSON here") is None
    assert _parse_fill_json("{ not valid json }") is None
    assert _parse_fill_json("") is None


def test_parse_fill_json_null_fields():
    """Null fields in LLM response should be preserved as None."""
    from fundmgr.notify.telegram_bot import _parse_fill_json
    raw = '{"isin": null, "side": "buy", "shares": 12, "price_sek": 100.0, "fee_sek": 1.0, "confidence": 0.5}'
    data = _parse_fill_json(raw)
    assert data is not None
    assert data["isin"] is None


# ── 3. OCR (requires tesseract) ───────────────────────────────────────────────

try:
    import pytesseract as _pyt
    from PIL import Image  # noqa: F401
    _pyt.get_tesseract_version()
    _TESSERACT_OK = True
except Exception:
    _TESSERACT_OK = False

pytesseract_installed = pytest.mark.skipif(
    not _TESSERACT_OK,
    reason=(
        "tesseract not available — install with:\n"
        "  macOS: brew install tesseract\n"
        "  Pi/Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-swe\n"
        "  then: uv pip install pytesseract Pillow"
    ),
)


@pytesseract_installed
def test_ocr_returns_text():
    from fundmgr.notify.telegram_bot import _ocr_image
    buf = _make_broker_image(SAMPLE_OCR_TEXT)
    text = _ocr_image(buf)
    assert isinstance(text, str)
    assert len(text.strip()) > 0, "OCR should return non-empty text"


@pytesseract_installed
def test_ocr_finds_isin():
    """OCR on synthetic broker image should extract the ISIN."""
    import re
    from fundmgr.notify.telegram_bot import _ocr_image

    buf = _make_broker_image(SAMPLE_OCR_TEXT)
    text = _ocr_image(buf)

    # ISIN pattern: 2 capital letters + 10 alphanumeric
    found = re.findall(r"\b[A-Z]{2}[A-Z0-9]{10}\b", text)
    assert "SE0000115446" in found, (
        f"ISIN SE0000115446 not found in OCR output.\n"
        f"Got: {text!r}\n"
        f"Hint: check tesseract language packs and image contrast."
    )


@pytesseract_installed
def test_ocr_finds_shares_and_price():
    """OCR should at minimum extract the share count and price digits."""
    from fundmgr.notify.telegram_bot import _ocr_image
    buf = _make_broker_image(SAMPLE_OCR_TEXT)
    text = _ocr_image(buf)
    assert "12" in text,  f"Share count '12' not found in OCR output:\n{text}"
    assert "291" in text, f"Price '291' not found in OCR output:\n{text}"


# ── 4. Full flow — mocked LLM ─────────────────────────────────────────────────

@pytesseract_installed
@pytest.mark.asyncio
async def test_full_flow_resolves_ticker():
    """
    photo_handler extracts fill from synthetic image and resolves ISIN → ticker.
    The LLM call is mocked; OCR runs for real via pytesseract.
    """
    from fundmgr.notify.telegram_bot import _pending_fills, photo_handler

    # Build a fake Telegram photo message backed by our synthetic image
    buf = _make_broker_image(SAMPLE_OCR_TEXT)

    mock_file = MagicMock()
    async def _download(out_buf):
        out_buf.write(buf.read())
    mock_file.download_to_memory = _download

    mock_photo = MagicMock()
    mock_photo.file_id = "fake_id"

    mock_message = AsyncMock()
    mock_message.photo = [mock_photo]

    mock_update = MagicMock()
    mock_update.message = mock_message
    mock_update.effective_chat.id = 99999

    mock_context = MagicMock()
    mock_context.bot.get_file = AsyncMock(return_value=mock_file)

    # Mock _extract_fill_from_ocr so we don't hit the real OpenAI API
    with patch(
        "fundmgr.notify.telegram_bot._extract_fill_from_ocr",
        new=AsyncMock(return_value=json.dumps(SAMPLE_LLM_JSON)),
    ):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test", "FUND_OCR_MODEL": "gpt-4o-mini"}):
            await photo_handler(mock_update, mock_context)

    # Bot should have replied at least twice (initial "Reading…" + preview)
    assert mock_message.reply_text.call_count >= 2

    # The preview message should contain the resolved ticker and ISIN
    preview_call = mock_message.reply_text.call_args_list[-1]
    preview_text = preview_call.args[0] if preview_call.args else preview_call.kwargs.get("text", "")
    assert "VOLV-B.ST" in preview_text, f"Ticker not in preview:\n{preview_text}"
    assert "SE0000115446" in preview_text, f"ISIN not in preview:\n{preview_text}"

    # Pending fill should be stored ready for confirm button
    assert 99999 in _pending_fills
    fill = _pending_fills[99999]
    assert fill["ticker"] == "VOLV-B.ST"
    assert fill["shares"] == 12
    assert fill["price"] == pytest.approx(291.50)

    # Cleanup
    _pending_fills.pop(99999, None)


@pytesseract_installed
@pytest.mark.asyncio
async def test_full_flow_unknown_isin_no_pending():
    """
    When ISIN is not in universe, handler sends a manual /fill suggestion
    and does NOT store a pending fill (since ticker is unknown).
    """
    from fundmgr.notify.telegram_bot import _pending_fills, photo_handler

    buf = _make_broker_image("ISIN: XX9999999999\nAntal: 5\nKurs: 100.00 SEK\nCourtage: 1.00 SEK")

    mock_file = MagicMock()
    async def _download(out_buf):
        out_buf.write(buf.read())
    mock_file.download_to_memory = _download

    mock_message = AsyncMock()
    mock_message.photo = [MagicMock(file_id="fake")]

    mock_update = MagicMock()
    mock_update.message = mock_message
    mock_update.effective_chat.id = 88888

    mock_context = MagicMock()
    mock_context.bot.get_file = AsyncMock(return_value=mock_file)

    unknown_json = {**SAMPLE_LLM_JSON, "isin": "XX9999999999", "company_name": "Unknown Corp"}

    with patch(
        "fundmgr.notify.telegram_bot._extract_fill_from_ocr",
        new=AsyncMock(return_value=json.dumps(unknown_json)),
    ):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            await photo_handler(mock_update, mock_context)

    # No pending fill should be stored for unknown ticker
    assert 88888 not in _pending_fills

    # Reply should mention "not in universe" or manual /fill
    all_text = " ".join(
        (c.args[0] if c.args else c.kwargs.get("text", ""))
        for c in mock_message.reply_text.call_args_list
    )
    assert "not in universe" in all_text or "/fill" in all_text
