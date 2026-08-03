"""Tickers we've deliberately dropped must stay dropped.

Two ways an exclusion can silently regress: someone re-enables the row, or a
universe sync overwrites the flag from the broker snapshot. Both are pinned here.
"""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

from fundmgr.config import CONFIG_DIR, get_enabled_tickers, get_isin_map, load_universe

# Names removed from consideration, and every universe file they appear in.
EXCLUDED_TICKERS = {"HEM.ST", "0ABY.L"}  # Hemnet Group (Stockholm + LSE lines)

UNIVERSE_FILES = ["universe.csv", "universe_buffett.csv", "universe_global.csv"]


@pytest.mark.parametrize("filename", UNIVERSE_FILES)
def test_excluded_tickers_are_disabled(filename):
    path = CONFIG_DIR / filename
    enabled = {t.yahoo_ticker for t in get_enabled_tickers(path)}
    assert not (enabled & EXCLUDED_TICKERS), (
        f"{filename} re-enabled excluded ticker(s): {sorted(enabled & EXCLUDED_TICKERS)}"
    )


@pytest.mark.parametrize("filename", UNIVERSE_FILES)
def test_excluded_rows_are_kept_not_deleted(filename):
    """Disabled, not deleted — the row keeps the ISIN→ticker mapping that
    `fund fill` and reconcile use, and keeps the name resolving on the
    dashboard for any position still held."""
    path = CONFIG_DIR / filename
    all_tickers = {t.yahoo_ticker for t in load_universe(path)}
    assert "HEM.ST" in all_tickers, f"{filename} deleted the Hemnet row instead of disabling it"


def test_excluded_ticker_still_resolves_by_isin():
    """A held position must stay mappable even though it can't be bought."""
    isin_map = get_isin_map(CONFIG_DIR / "universe.csv")
    assert isin_map.get("SE0015671995") == "HEM.ST"


# ── Sync carry-forward ────────────────────────────────────────────────────────

def _load_sync_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "sync_universe_from_financedata.py"
    spec = importlib.util.spec_from_file_location("sync_universe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sync_preserves_manual_disable(tmp_path, monkeypatch):
    """The broker snapshot has no opinion on `enabled`, so a sync must not
    silently flip a hand-disabled ticker back on."""
    sync = _load_sync_module()

    nordic = tmp_path / "universe.csv"
    with nordic.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sync.FIELDNAMES)
        w.writeheader()
        w.writerow({"name": "Hemnet Group AB", "yahoo_ticker": "HEM.ST",
                    "isin": "SE0015671995", "country": "SE", "exchange": "OMXS",
                    "sector": "Technology", "enabled": "false"})
        w.writerow({"name": "Volvo B", "yahoo_ticker": "VOLV-B.ST",
                    "isin": "SE0000115446", "country": "SE", "exchange": "OMXS",
                    "sector": "Industrials", "enabled": "true"})

    monkeypatch.setattr(sync, "NORDIC_CSV", nordic)
    monkeypatch.setattr(sync, "GLOBAL_CSV", tmp_path / "universe_global.csv")

    meta = sync._load_existing_meta()
    assert meta["HEM.ST"]["enabled"] == "false"
    assert "enabled" not in meta.get("VOLV-B.ST", {})  # enabled names carry no override

    # A fresh snapshot row for the disabled ticker must come back disabled…
    row = sync._convert_row(
        {"company_name": "Hemnet Group AB", "ticker": "HEM", "isin": "SE0015671995",
         "country": "Sweden", "exchange_code": "ST", "exchange": "", "status": "active"},
        meta,
    )
    assert row is not None and row["enabled"] == "false"

    # …while an untouched ticker still defaults to enabled.
    row_ok = sync._convert_row(
        {"company_name": "Volvo B", "ticker": "VOLV-B", "isin": "SE0000115446",
         "country": "Sweden", "exchange_code": "ST", "exchange": "", "status": "active"},
        meta,
    )
    assert row_ok is not None and row_ok["enabled"] == "true"
