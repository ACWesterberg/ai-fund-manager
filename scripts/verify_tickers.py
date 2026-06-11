#!/usr/bin/env python3
"""
Verify candidate tickers against yfinance before adding to universe.csv.

Usage:
    .venv/bin/python scripts/verify_tickers.py

Checks each candidate: downloads ~5 days of price data.
Prints a table showing which are valid, their current price, and market cap.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed — uv pip install yfinance")
    sys.exit(1)

# ── Candidate tickers ──────────────────────────────────────────────────────────
# Format: (yahoo_ticker, name, country, sector, approx_cap)
# approx_cap: L=large (>50B SEK), M=mid (5-50B), S=small (<5B)
CANDIDATES = [
    # ── One last ticker to reach exactly 200 ─────────────────────────────────
    ("ATEA.OL",     "Atea ASA",             "NO", "Technology",              "M"),
    ("SCHO.CO",     "Schouw & Co.",         "DK", "Industrials",             "M"),
    ("SAVE.ST",     "Nordnet B",            "SE", "Financials",              "M"),
    ("BULTEN.ST",   "Bulten",               "SE", "Industrials",             "S"),
]

# ── Verify ────────────────────────────────────────────────────────────────────

def fmt_mcap(val: float | None) -> str:
    if val is None:
        return "?"
    if val >= 1e12:
        return f"{val/1e12:.1f}T"
    if val >= 1e9:
        return f"{val/1e9:.1f}B"
    return f"{val/1e6:.0f}M"


def check(ticker: str, name: str, country: str, sector: str, cap: str):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d")
        if hist.empty:
            return None, None, "❌ no price data"
        price = hist["Close"].iloc[-1]
        info  = t.info or {}
        mcap  = info.get("marketCap")
        return price, mcap, "✅"
    except Exception as e:
        return None, None, f"❌ {e}"


print(f"\n{'Ticker':<16} {'Name':<26} {'Ctry'} {'Cap'} {'Status':<6} {'Price':>10} {'Mkt Cap':>10}  Sector")
print("─" * 100)

valid, invalid = [], []
for ticker, name, country, sector, cap in CANDIDATES:
    price, mcap, status = check(ticker, name, country, sector, cap)
    price_str = f"{price:>10.2f}" if price else "         —"
    mcap_str  = f"{fmt_mcap(mcap):>10}" if mcap else "         —"
    print(f"{ticker:<16} {name:<26} {country}   {cap}   {status:<6} {price_str} {mcap_str}  {sector}")
    if "✅" in status:
        valid.append((ticker, name, country, sector, cap))
    else:
        invalid.append((ticker, name))

print(f"\n✅ {len(valid)} valid   ❌ {len(invalid)} invalid/delisted")
if invalid:
    print("  Skipping:", ", ".join(t for t, _ in invalid))

print("\nValid tickers to add to universe.csv:")
for ticker, name, country, sector, cap in valid:
    exchange = {"SE": "OMXS", "DK": "OMXC", "FI": "OMXH", "NO": "OSLO"}[country]
    print(f"  {name},{ticker},,{country},{exchange},{sector},true")
print()
