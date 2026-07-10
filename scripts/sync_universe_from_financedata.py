#!/usr/bin/env python3
"""
Sync config/universe.csv and config/universe_global.csv from FinanceData.

FinanceData maintains the broker-tradable universe (EODHD → shared cache). This
script reads the committed snapshot at FinanceData/data/universe_snapshot.csv
(or exports from the installed financedata package) and converts rows to the
fund-manager CSV schema (yahoo_ticker, exchange, sector, …).

Outputs:
  config/universe.csv        — Nordic listings only (real-money fund)
  config/universe_global.csv — full Montrose universe (global sim fund)

Run:
  uv run python scripts/sync_universe_from_financedata.py
  FINANCEDATA_DIR=../FinanceData uv run python scripts/sync_universe_from_financedata.py
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NORDIC_CSV = ROOT / "config" / "universe.csv"
GLOBAL_CSV = ROOT / "config" / "universe_global.csv"

FIELDNAMES = ["name", "yahoo_ticker", "isin", "country", "exchange", "sector", "enabled"]

NORDIC_COUNTRIES = {"Sweden", "Denmark", "Finland", "Norway"}

COUNTRY_ISO: dict[str, str] = {
    "Sweden": "SE",
    "Denmark": "DK",
    "Finland": "FI",
    "Norway": "NO",
    "Iceland": "IS",
    "United States": "US",
    "United Kingdom": "GB",
    "Canada": "CA",
    "Germany": "DE",
    "France": "FR",
    "Switzerland": "CH",
    "Netherlands": "NL",
    "Spain": "ES",
    "Italy": "IT",
    "Belgium": "BE",
    "Austria": "AT",
    "Ireland": "IE",
    "Portugal": "PT",
    "Poland": "PL",
}

# EODHD exchange_code → Yahoo suffix (empty for US plain tickers).
YAHOO_SUFFIX: dict[str, str] = {
    "ST": ".ST",
    "CO": ".CO",
    "HE": ".HE",
    "OL": ".OL",
    "LSE": ".L",
    "SW": ".SW",
    "XETRA": ".DE",
    "PA": ".PA",
    "AS": ".AS",
    "BR": ".BR",
    "LS": ".LS",
    "MC": ".MC",
    "IR": ".IR",
    "VI": ".VI",
    "WAR": ".WA",
    "TO": ".TO",
    "V": ".V",
    "US": "",
}

# Default fund exchange label per EODHD code (US uses the lit sub-exchange).
EXCHANGE_LABEL: dict[str, str] = {
    "ST": "OMXS",
    "CO": "OMXC",
    "HE": "OMXH",
    "OL": "OSLO",
    "LSE": "LSE",
    "SW": "SIX",
    "XETRA": "XETRA",
    "PA": "EURONEXT",
    "AS": "EURONEXT",
    "BR": "EURONEXT",
    "LS": "EURONEXT",
    "MC": "EURONEXT",
    "IR": "EURONEXT",
    "VI": "EURONEXT",
    "WAR": "WSE",
    "TO": "TSX",
    "V": "TSX",
}

US_SUB_EXCHANGE: dict[str, str] = {
    "NYSE": "NYSE",
    "NASDAQ": "NASDAQ",
    "AMEX": "NYSE",
    "NYSE MKT": "NYSE",
}


def _find_financedata_dir() -> Path | None:
  for candidate in (
      os.environ.get("FINANCEDATA_DIR"),
      ROOT.parent / "FinanceData",
      Path.home() / "Documents" / "FinanceData",
      Path.home() / "FinanceData",
  ):
      if candidate and Path(candidate).is_dir():
          return Path(candidate)
  return None


def _snapshot_path(fd_dir: Path | None) -> Path | None:
    if fd_dir:
        snap = fd_dir / "data" / "universe_snapshot.csv"
        if snap.is_file():
            return snap
    return None


def _load_snapshot_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _load_existing_meta() -> dict[str, dict[str, str]]:
    """Preserve sector / fine-grained exchange from the current CSVs."""
    meta: dict[str, dict[str, str]] = {}
    for path in (NORDIC_CSV, GLOBAL_CSV):
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                yahoo = row["yahoo_ticker"].strip()
                if not yahoo:
                    continue
                prev = meta.get(yahoo, {})
                sector = row.get("sector", "").strip()
                if sector and sector.lower() != "unknown":
                    prev["sector"] = sector
                exchange = row.get("exchange", "").strip()
                if exchange:
                    prev["exchange"] = exchange
                meta[yahoo] = prev
    return meta


def _to_yahoo_ticker(exchange_code: str, ticker: str, sub_exchange: str) -> str:
    code = exchange_code.strip().upper()
    sym = ticker.strip()
    if code == "US":
        return sym
    suffix = YAHOO_SUFFIX.get(code)
    if suffix is None:
        return sym
    return f"{sym}{suffix}"


def _fund_exchange(exchange_code: str, sub_exchange: str, meta_exchange: str | None) -> str:
    if meta_exchange:
        return meta_exchange
    code = exchange_code.strip().upper()
    if code == "US":
        return US_SUB_EXCHANGE.get(sub_exchange.strip(), "NASDAQ")
    return EXCHANGE_LABEL.get(code, code)


def _convert_row(row: dict[str, str], meta: dict[str, dict[str, str]]) -> dict[str, str] | None:
    if row.get("status", "active").strip().lower() != "active":
        return None

    country_name = row.get("country", "").strip()
    country = COUNTRY_ISO.get(country_name)
    if not country:
        return None

    exchange_code = row.get("exchange_code", "").strip().upper()
    sub_exchange = row.get("exchange", "").strip()
    ticker = row.get("ticker", "").strip()
    if not ticker:
        return None

    yahoo = _to_yahoo_ticker(exchange_code, ticker, sub_exchange)
    prev = meta.get(yahoo, {})
    sector = prev.get("sector") or "Unknown"
    exchange = _fund_exchange(exchange_code, sub_exchange, prev.get("exchange"))

    return {
        "name": row.get("company_name", ticker).strip(),
        "yahoo_ticker": yahoo,
        "isin": row.get("isin", "").strip(),
        "country": country,
        "exchange": exchange,
        "sector": sector,
        "enabled": "true",
    }


def _dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        yahoo = row["yahoo_ticker"]
        if yahoo in seen:
            continue
        seen.add(yahoo)
        out.append(row)
    return out


def _sort_key(row: dict[str, str]) -> tuple:
    return (row["country"], row["exchange"], row["yahoo_ticker"])


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sync(snapshot: Path, *, dry_run: bool = False) -> dict[str, int]:
    meta = _load_existing_meta()
    raw = _load_snapshot_rows(snapshot)

    all_rows: list[dict[str, str]] = []
    for row in raw:
        converted = _convert_row(row, meta)
        if converted:
            all_rows.append(converted)
    all_rows = _dedupe(all_rows)
    all_rows.sort(key=_sort_key)

    nordic_rows = [r for r in all_rows if r["country"] in {"SE", "DK", "FI", "NO"}]
    nordic_rows.sort(key=_sort_key)

    counts = {
        "snapshot_rows": len(raw),
        "converted": len(all_rows),
        "nordic": len(nordic_rows),
        "nordic_enabled": sum(1 for r in nordic_rows if r["enabled"] == "true"),
        "global": len(all_rows),
        "global_enabled": sum(1 for r in all_rows if r["enabled"] == "true"),
        "sectors_known": sum(1 for r in all_rows if r["sector"] != "Unknown"),
    }

    if not dry_run:
        _write_csv(NORDIC_CSV, nordic_rows)
        _write_csv(GLOBAL_CSV, all_rows)

    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Path to universe_snapshot.csv (default: FinanceData/data/universe_snapshot.csv)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print counts only; do not write CSVs")
    args = parser.parse_args(argv)

    snapshot = args.snapshot
    if snapshot is None:
        fd_dir = _find_financedata_dir()
        snapshot = _snapshot_path(fd_dir)
        if snapshot is None:
            print(
                "FinanceData not found. Clone it next to this repo or set FINANCEDATA_DIR.\n"
                "  git clone https://github.com/ACWesterberg/FinanceData.git ../FinanceData",
                file=sys.stderr,
            )
            return 1

    if not snapshot.is_file():
        print(f"Snapshot not found: {snapshot}", file=sys.stderr)
        return 1

    counts = sync(snapshot, dry_run=args.dry_run)
    print(f"Source: {snapshot}")
    for key, val in counts.items():
        print(f"  {key}: {val:,}")
    if args.dry_run:
        print("(dry run — no files written)")
    else:
        print(f"Wrote {NORDIC_CSV} ({counts['nordic']:,} rows)")
        print(f"Wrote {GLOBAL_CSV} ({counts['global']:,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
