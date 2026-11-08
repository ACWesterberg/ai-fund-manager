"""Every enabled universe row must resolve to a currency and a trading calendar.

Rows listed as "HELSINKI"/"STOCKHOLM" instead of OMXH/OMXS had neither: Orion
Oyj (ORNBV.HE, EUR) fell back to SEK in the real fund's valuation maths, and a
candidate without a calendar made the shadow comparison refuse the whole run.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from fundmgr.config import _EXCHANGE_CURRENCY, UniverseTicker
from fundmgr.data.market_hours import _EXCHANGE_TO_CALENDAR

CONFIG = Path(__file__).resolve().parents[1] / "config"


@pytest.mark.parametrize("path", sorted(CONFIG.glob("universe*.csv")), ids=lambda p: p.name)
def test_every_enabled_row_has_a_currency_and_a_calendar(path):
    rows = [r for r in csv.DictReader(path.open()) if r["enabled"].strip().lower() == "true"]
    unknown = sorted({r["exchange"] for r in rows
                      if r["exchange"].upper() not in _EXCHANGE_CURRENCY
                      or r["exchange"].upper() not in _EXCHANGE_TO_CALENDAR})
    assert not unknown, f"exchange codes with no currency or calendar: {unknown}"


def _row(ticker: str, exchange: str) -> UniverseTicker:
    return UniverseTicker(name=ticker, yahoo_ticker=ticker, isin="", country="",
                          exchange=exchange, sector="", enabled=True)


def test_a_helsinki_listing_is_priced_in_euro():
    assert _row("ORNBV.HE", "OMXH").currency == "EUR"
    assert _row("ORNBV.HE", "HELSINKI").currency == "EUR"


def test_an_unknown_exchange_falls_back_to_the_ticker_suffix_not_to_sek():
    assert _row("XYZ.HE", "SOMEWHERE").currency == "EUR"
    assert _row("XYZ.OL", "").currency == "NOK"
    assert _row("XYZ", "SOMEWHERE").currency == "SEK"
