#!/usr/bin/env python3
"""
Validate a list of potential tickers against yfinance and output
universe.csv-ready rows for the ones that have price data.

Usage
-----
# From a plain text file (one ticker per line, e.g. "BETS-B.ST"):
  python tools/discover_tickers.py --file tickers.txt --exchange OMXS --sector "Consumer"

# From a Nasdaq Nordic instrument CSV export
# (download from nasdaqomxnordic.com → Shares → Listed Companies → Export):
  python tools/discover_tickers.py --nasdaq-nordic nasdaq_stockholm.csv --suffix .ST --exchange OMXS

# From a Spotlight/NGM CSV (columns: Name, Ticker/Symbol, ISIN)
  python tools/discover_tickers.py --spotlight spotlight.csv --exchange SPOTLIGHT

# Pipe known symbols directly:
  echo "CTEK.ST\nPROFF.ST\nBETS-B.ST" | python tools/discover_tickers.py --stdin --exchange OMXS

Output is appended to stdout as CSV rows ready to paste into universe.csv.
Errors / skipped tickers go to stderr.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf

# Minimum bars of history required to consider a ticker usable
MIN_BARS = 60


def _test_ticker(symbol: str) -> tuple[str, bool, str]:
    """Return (symbol, ok, reason)."""
    try:
        raw = yf.download(symbol, period="6mo", auto_adjust=True, progress=False)
        if raw.empty or len(raw) < MIN_BARS:
            return symbol, False, f"only {len(raw)} bars"
        # Grab name from fast_info
        try:
            name = yf.Ticker(symbol).fast_info.get("longName") or symbol
        except Exception:
            name = symbol
        return symbol, True, name
    except Exception as e:
        return symbol, False, str(e)


def _parse_nasdaq_nordic(path: str, suffix: str) -> list[tuple[str, str, str]]:
    """Parse Nasdaq Nordic instrument export CSV.

    Returns list of (symbol_with_suffix, isin, name).
    Nasdaq Nordic CSVs are typically semicolon-separated with columns:
    Symbol, ISIN, Name, Currency, Lot size, Market segment, ...
    """
    results = []
    with open(path, encoding="utf-8-sig") as f:
        # Detect delimiter
        sample = f.read(2048)
        f.seek(0)
        delim = ";" if sample.count(";") > sample.count(",") else ","
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            # Nasdaq Nordic uses "Symbol" or "Ticker" column
            sym = (row.get("Symbol") or row.get("Ticker") or "").strip()
            isin = (row.get("ISIN") or "").strip()
            name = (row.get("Name") or row.get("Instrument name") or sym).strip()
            if not sym:
                continue
            # Strip any existing suffix and re-apply the requested one
            base = sym.split(".")[0] if "." in sym else sym
            results.append((base + suffix, isin, name))
    return results


def _parse_spotlight(path: str) -> list[tuple[str, str, str]]:
    """Parse Spotlight Stock Market CSV.

    Their export typically has columns: Bolagsnamn, Kortnamn, ISIN
    Tickers are listed without exchange suffix; they trade on .ST in yfinance.
    """
    results = []
    with open(path, encoding="utf-8-sig") as f:
        sample = f.read(2048)
        f.seek(0)
        delim = ";" if sample.count(";") > sample.count(",") else ","
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            sym = (
                row.get("Kortnamn") or row.get("Symbol") or row.get("Ticker") or ""
            ).strip()
            isin = (row.get("ISIN") or "").strip()
            name = (row.get("Bolagsnamn") or row.get("Name") or sym).strip()
            if not sym:
                continue
            base = sym.split(".")[0] if "." in sym else sym
            results.append((base + ".ST", isin, name))
    return results


def _parse_plain(path: str, suffix: str) -> list[tuple[str, str, str]]:
    """One symbol per line; optionally suffix-less."""
    results = []
    with open(path) as f:
        for line in f:
            sym = line.strip()
            if not sym or sym.startswith("#"):
                continue
            if "." not in sym:
                sym += suffix
            results.append((sym, "", sym))
    return results


def _parse_stdin(suffix: str) -> list[tuple[str, str, str]]:
    results = []
    for line in sys.stdin:
        sym = line.strip()
        if not sym or sym.startswith("#"):
            continue
        if "." not in sym:
            sym += suffix
        results.append((sym, "", sym))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--nasdaq-nordic", metavar="CSV", help="Nasdaq Nordic instrument export CSV")
    source.add_argument("--spotlight", metavar="CSV", help="Spotlight Stock Market CSV export")
    source.add_argument("--file", metavar="TXT", help="Plain text file, one ticker per line")
    source.add_argument("--stdin", action="store_true", help="Read tickers from stdin")
    ap.add_argument("--suffix", default=".ST", help="Exchange suffix to append (default: .ST)")
    ap.add_argument("--exchange", default="OMXS", help="Exchange code for universe.csv (default: OMXS)")
    ap.add_argument("--country", default="SE", help="Country code (default: SE)")
    ap.add_argument("--sector", default="Unknown", help="Default sector for new rows")
    ap.add_argument("--workers", type=int, default=10, help="Parallel workers (default: 10)")
    ap.add_argument("--existing-universe", metavar="CSV",
                    help="Path to existing universe.csv — skip tickers already present")
    args = ap.parse_args()

    # Load existing universe to skip duplicates
    existing: set[str] = set()
    if args.existing_universe:
        with open(args.existing_universe) as f:
            for row in csv.DictReader(f):
                existing.add(row.get("yahoo_ticker", "").strip())
    else:
        # Try default location
        import os
        default = os.path.join(os.path.dirname(__file__), "..", "config", "universe.csv")
        if os.path.exists(default):
            with open(default) as f:
                for row in csv.DictReader(f):
                    existing.add(row.get("yahoo_ticker", "").strip())

    # Parse candidates
    if args.nasdaq_nordic:
        candidates = _parse_nasdaq_nordic(args.nasdaq_nordic, args.suffix)
    elif args.spotlight:
        candidates = _parse_spotlight(args.spotlight)
    elif args.file:
        candidates = _parse_plain(args.file, args.suffix)
    else:
        candidates = _parse_stdin(args.suffix)

    # Deduplicate and skip existing
    seen: set[str] = set()
    to_test: list[tuple[str, str, str]] = []
    for sym, isin, name in candidates:
        if sym in existing:
            print(f"# SKIP (already in universe): {sym}", file=sys.stderr)
            continue
        if sym in seen:
            continue
        seen.add(sym)
        to_test.append((sym, isin, name))

    print(f"# Testing {len(to_test)} candidates ({len(existing)} already in universe)…",
          file=sys.stderr)

    # Print CSV header if this is the first run (no existing universe loaded)
    if not existing:
        print("name,yahoo_ticker,isin,country,exchange,sector,enabled")

    # Validate in parallel
    sym_to_meta = {sym: (isin, name) for sym, isin, name in to_test}
    passed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_test_ticker, sym): sym for sym, _, _ in to_test}
        for fut in as_completed(futures):
            sym, ok, detail = fut.result()
            isin, hint_name = sym_to_meta[sym]
            if ok:
                # detail is the fetched name when ok=True
                display_name = detail if detail != sym else hint_name
                print(
                    f"{display_name},{sym},{isin},{args.country},"
                    f"{args.exchange},{args.sector},true"
                )
                passed += 1
            else:
                print(f"# NO DATA: {sym} — {detail}", file=sys.stderr)
                failed += 1

    print(
        f"# Done: {passed} valid, {failed} no data / too short",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
