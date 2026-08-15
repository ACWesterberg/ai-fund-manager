"""
SEC EDGAR as a source for figures no market-data feed carries.

`.info` gives a normalised snapshot and the statements give three frames; both
stop where the filing begins. EDGAR publishes the XBRL facts a US company
actually filed — free, keyless, documented, and stamped with the fiscal period
they describe, which is the shape this codebase already stores.

It is deliberately wired to **custom metrics only**. A book defines a metric and
names the us-gaap concept behind it:

    fund paper-metric kf datacenter_revenue --unit SEK --edgar Revenues

That indirection matters. Letting EDGAR write a built-in field would put two
providers behind one key, and a series that switches definition halfway is worse
than one that is merely incomplete — a "two consecutive quarters" rule would be
comparing one source's number against another's.

Three properties, all of which exist because the alternative fails quietly:

  • **A value is stored against the period it covers**, taken from the filing's
    own `end` date, never the date it was fetched.
  • **Only quarterly durations count.** A concept carries year-to-date and
    annual facts in the same list, and a nine-month revenue read as a quarter
    would look like spectacular growth.
  • **A later filing supersedes an earlier one for the same period.** Restated
    figures are the company correcting itself, and the correction is the number
    to hold.

The SEC requires a User-Agent identifying the caller. Set FUND_SEC_USER_AGENT
to "name you@example.com"; without it requests are refused, and this module
says so rather than reporting an empty result.
"""
from __future__ import annotations

import json
import os

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
TIMEOUT = 20

# A quarterly fact spans about three months. Year-to-date and annual facts sit
# in the same list with the same concept, so the span is what separates them.
_QUARTER_DAYS = (80, 100)

# Cached ticker→CIK map, so the lookup file is fetched once per process.
_ticker_cik: dict[str, int] | None = None


class EdgarError(RuntimeError):
    """EDGAR could not be reached, or refused the request."""


def _user_agent() -> str:
    agent = (os.getenv("FUND_SEC_USER_AGENT") or "").strip()
    if not agent:
        raise EdgarError(
            "SEC requires a User-Agent identifying the caller. Set "
            "FUND_SEC_USER_AGENT=\"Your Name you@example.com\" in .env.")
    return agent


def _get(url: str) -> dict:
    import requests

    resp = requests.get(url, timeout=TIMEOUT,
                        headers={"User-Agent": _user_agent(),
                                 "Accept-Encoding": "gzip, deflate"})
    if resp.status_code == 403:
        raise EdgarError("SEC refused the request (403) — check FUND_SEC_USER_AGENT.")
    if resp.status_code == 404:
        raise EdgarError("Not found at SEC — the company may not file with them.")
    resp.raise_for_status()
    return resp.json()


def cik_for(ticker: str) -> int | None:
    """CIK for a US ticker, or None. Foreign listings are simply absent."""
    global _ticker_cik
    if _ticker_cik is None:
        _ticker_cik = _parse_ticker_map(_get(TICKER_MAP_URL))
    # EDGAR knows plain US symbols; a Yahoo suffix means another exchange.
    key = (ticker or "").strip().upper()
    if "." in key:
        return None
    return _ticker_cik.get(key)


def _parse_ticker_map(payload) -> dict[str, int]:
    """{"0": {"cik_str": 320193, "ticker": "AAPL", ...}, ...} → {ticker: cik}."""
    rows = payload.values() if isinstance(payload, dict) else payload
    out: dict[str, int] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").strip().upper()
        try:
            cik = int(row.get("cik_str"))
        except (TypeError, ValueError):
            continue
        if ticker:
            out.setdefault(ticker, cik)
    return out


def fetch_facts(ticker: str) -> dict:
    """Every XBRL fact a company has filed. {} when it does not file with SEC."""
    cik = cik_for(ticker)
    if cik is None:
        return {}
    return _get(FACTS_URL.format(cik=cik))


def quarterly_series(facts: dict, concept: str,
                     taxonomy: str = "us-gaap") -> dict[str, float]:
    """{period_end: value} for one concept, quarterly facts only.

    A concept's fact list mixes durations — three months, nine months, a full
    year — and instants. Reading a nine-month revenue as a quarter would show
    growth that never happened, so only quarter-length spans are kept.
    """
    from datetime import datetime

    node = ((facts.get("facts") or {}).get(taxonomy) or {}).get(concept) or {}
    units = node.get("units") or {}
    if not units:
        return {}

    # One concept, one unit. Prefer the currency with the most facts rather than
    # guessing USD, so a filer reporting in another currency still works.
    unit_key = max(units, key=lambda u: len(units[u] or []))
    best: dict[str, tuple[str, float]] = {}
    for fact in units.get(unit_key) or []:
        if not isinstance(fact, dict):
            continue
        start, end = fact.get("start"), fact.get("end")
        if not start or not end:
            continue          # an instant, not a duration — not a quarter
        try:
            span = (datetime.strptime(end, "%Y-%m-%d")
                    - datetime.strptime(start, "%Y-%m-%d")).days
            value = float(fact["val"])
        except (TypeError, ValueError, KeyError):
            continue
        if not _QUARTER_DAYS[0] <= span <= _QUARTER_DAYS[1]:
            continue
        filed = str(fact.get("filed") or "")
        # A later filing restating the same period is the company correcting
        # itself, and the correction is the figure to keep.
        if end not in best or filed >= best[end][0]:
            best[end] = (filed, value)
    return {end: value for end, (_filed, value) in sorted(best.items())}


def read_ticker(store, ticker: str, facts: dict | None = None) -> dict:
    """Quarterly values for every EDGAR-mapped metric this book defines.

    Returns {metric: {period_end: value}}. Empty when the book maps nothing or
    the company does not file with the SEC.
    """
    from fundmgr.evidence import custom_metrics

    mapped = {k: m["edgar"] for k, m in custom_metrics(store).items()
              if (m or {}).get("edgar")}
    if not mapped:
        return {}
    if facts is None:
        facts = fetch_facts(ticker)
    if not facts:
        return {}

    out: dict[str, dict[str, float]] = {}
    for metric, concept in mapped.items():
        series = quarterly_series(facts, concept)
        if series:
            out[metric] = series
    return out
