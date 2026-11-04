"""
Fundamentals: valuation, quality, growth, analyst and ownership figures from
yfinance `.info`, cached in the fund's own store with a 7-day TTL.

Fetched here rather than through financedata (moved 2026-11-01, phase 2 of
docs/FINANCEDATA_MIGRATION.md). Two things changed with the move: Yahoo's
insider and institution holdings now arrive with the same call, and a refresh
replaces only the `.info` keys, so the ratios `statements.refresh` derives and
folds into the same blob survive it instead of vanishing until the next
statements pass.
"""
from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from fundmgr.state.store import Store

logger = logging.getLogger(__name__)

# yfinance info key → cache key. Values are stored raw; fraction fields stay 0-1.
FIELD_MAP: dict[str, str] = {
    "trailingPE":                       "pe_ratio",
    "forwardPE":                        "forward_pe",
    "priceToBook":                      "pb_ratio",
    "enterpriseToEbitda":               "ev_to_ebitda",
    "priceToSalesTrailingTwelveMonths": "price_to_sales",
    "marketCap":                        "market_cap",
    "profitMargins":                    "profit_margin",
    "grossMargins":                     "gross_margin",
    "operatingMargins":                 "operating_margin",
    "ebitdaMargins":                    "ebitda_margin",
    "returnOnEquity":                   "roe",
    "returnOnAssets":                   "roa",
    "freeCashflow":                     "free_cash_flow",
    "operatingCashflow":                "operating_cash_flow",
    "totalDebt":                        "total_debt",
    "totalCash":                        "total_cash",
    "debtToEquity":                     "debt_to_equity",
    "revenueGrowth":                    "revenue_growth",
    "earningsGrowth":                   "earnings_growth",
    "beta":                             "beta",
    "fiftyTwoWeekHigh":                 "fifty_two_week_high",
    "fiftyTwoWeekLow":                  "fifty_two_week_low",
    "dividendYield":                    "dividend_yield",
    "targetMeanPrice":                  "analyst_target_price",
    "numberOfAnalystOpinions":          "analyst_count",
    "currency":                         "currency",
    "earningsTimestamp":                "earnings_timestamp",
    "exDividendDate":                   "ex_div_timestamp",
    "website":                          "website",
    "sector":                           "sector",
    # Share of capital, 0-1. Not votes: an A/B-share founder with most of the
    # votes on a minority of the capital reads as a small insider stake.
    "heldPercentInsiders":              "held_insiders",
    "heldPercentInstitutions":          "held_institutions",
}
TEXT_FIELDS = {"currency", "website", "sector"}


def ts_to_days(ts) -> int | None:
    """Calendar days from today to a Unix timestamp (None if it has passed)."""
    if ts is None:
        return None
    try:
        today = datetime.now(timezone.utc).date()
        target = datetime.fromtimestamp(float(ts), timezone.utc).date()
    except (TypeError, ValueError, OSError, OverflowError):
        return None
    delta = (target - today).days
    return delta if delta >= 0 else None


def _number(val) -> float | None:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def parse_info(info: dict | None) -> dict | None:
    """Map a raw `.info` payload onto the cache keys; None when Yahoo knows no such symbol.

    An unknown symbol comes back as `{'trailingPegRatio': None}` after a 404,
    not as an error, so a payload with nothing in it counts as no data rather
    than being cached as an empty entry for a week.
    """
    if not info or info.get("quoteType") == "NONE":
        return None
    data: dict = {}
    for yf_key, key in FIELD_MAP.items():
        raw = info.get(yf_key)
        if key in TEXT_FIELDS:
            data[key] = raw if isinstance(raw, str) else None
        elif key == "analyst_count":
            n = _number(raw)
            data[key] = int(n) if n is not None else None
        else:
            data[key] = _number(raw)
    return data if any(v is not None for v in data.values()) else None


def fetch_info(ticker: str) -> dict | None:
    """One ticker's fundamentals straight from Yahoo; None on any failure."""
    try:
        import yfinance as yf
        return parse_info(yf.Ticker(ticker).info)
    except Exception as exc:
        logger.debug("fundamentals fetch failed for %s: %s", ticker, exc)
        return None


def fetch_and_cache_fundamentals(
    tickers: list[str],
    store: Store,
    ttl_days: int = 7,
    max_workers: int = 12,
) -> int:
    """Refresh the tickers whose cache entry is missing or older than ttl_days.

    Only the `.info` keys are replaced; anything else in the stored blob (the
    statement-derived ratios) is kept. A failed fetch leaves the old entry
    alone, so it is retried on the next run. Returns how many were refreshed.
    """
    stale = store.get_stale_fundamentals_tickers(list(tickers), ttl_days=ttl_days)
    if not stale:
        return 0
    refreshed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_info, t): t for t in stale}
        for future in as_completed(futures):
            data = future.result()
            if data is None:
                continue
            ticker = futures[future]
            merged = store.get_fundamentals(ticker) or {}
            merged.update(data)
            store.save_fundamentals(ticker, merged)
            refreshed += 1
    return refreshed


def ownership_line(data: dict | None) -> str | None:
    """`Owners: insiders 34%, institutions 41%` from a cached blob, or None."""
    if not data:
        return None
    parts = []
    for label, key in (("insiders", "held_insiders"), ("institutions", "held_institutions")):
        val = _number(data.get(key))
        if val is not None and 0 <= val <= 1.5:
            parts.append(f"{label} {min(val, 1.0) * 100:.0f}%")
    return f"Owners: {', '.join(parts)} of capital" if parts else None


def ownership_lines(store: Store, tickers: list[str]) -> dict[str, str]:
    """Ownership lines for the tickers the cache has figures for.

    Kept out of TickerFeatures on purpose: a shadow comparison freezes each
    case's features and checks them by hash, so a new feature field would
    invalidate every archived case.
    """
    cached = store.get_all_fundamentals(list(tickers))
    lines = {t: ownership_line(cached.get(t)) for t in tickers}
    return {t: line for t, line in lines.items() if line}


# Yahoo's sector names → the universe's GICS-style ones, so a filled-in sector
# lands in the same bucket as a listed one (Healthcare and Health Care are one
# sector to the cap, not two).
_YAHOO_SECTOR = {
    "Financial Services": "Financials",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Basic Materials": "Materials",
    "Healthcare": "Health Care",
    "Communication Services": "Communication Services",
    "Technology": "Technology",
    "Industrials": "Industrials",
    "Energy": "Energy",
    "Utilities": "Utilities",
    "Real Estate": "Real Estate",
}


def normalise_sector(value: str | None) -> str | None:
    """A sector in the universe's names, or None when there is none.

    "Unknown" is not a sector. Two thirds of universe.csv carry it, and treated
    as one it pooled NOBA, WSG, NANEXA and DYVOX into a 34% "Unknown" bucket
    against the 35% cap, blocking every unclassified buy (2 Nov).
    """
    v = (value or "").strip()
    if not v or v.lower() in ("unknown", "n/a", "none"):
        return None
    return _YAHOO_SECTOR.get(v, v)


def apply_to_features(features: dict, store: Store) -> None:
    """
    Pull cached fundamentals from fund's store and attach to TickerFeatures objects.
    `features` is a dict[str, TickerFeatures] as returned by build_all_features.
    """
    cached = store.get_all_fundamentals(list(features.keys()))

    for ticker, feat in features.items():
        data = cached.get(ticker)
        if not data:
            continue
        if not feat.sector:
            feat.sector = normalise_sector(data.get("sector"))

        def _safe(val, scale=1.0):
            try:
                if val is None:
                    return None
                f = float(val) * scale
                return None if math.isnan(f) or math.isinf(f) else round(f, 2)
            except (TypeError, ValueError):
                return None

        feat.pe_ratio       = _safe(data.get("pe_ratio"))
        feat.forward_pe     = _safe(data.get("forward_pe"))
        feat.pb_ratio       = _safe(data.get("pb_ratio"))
        feat.ev_to_ebitda   = _safe(data.get("ev_to_ebitda"))
        feat.price_to_sales = _safe(data.get("price_to_sales"))
        # financedata has cached debtToEquity all along and this line was
        # missing, so the field stayed None everywhere: the screener's leverage
        # penalty never fired, the prompt's Quality line never showed D/E, and a
        # style screen asking for a conservative balance sheet had nothing to
        # read. Populating it makes all three live.
        feat.debt_to_equity = _safe(data.get("debt_to_equity"))
        feat.beta           = _safe(data.get("beta"))
        feat.analyst_count  = data.get("analyst_count")

        for frac_key, attr in (
            ("profit_margin",   "profit_margin_pct"),
            ("gross_margin",    "gross_margin_pct"),
            ("roe",             "roe_pct"),
            ("revenue_growth",  "revenue_growth_pct"),
            ("earnings_growth", "earnings_growth_pct"),
        ):
            raw = data.get(frac_key)
            setattr(feat, attr, round(raw * 100, 1) if raw is not None else None)
        # Yahoo's dividendYield is already a percentage (INWI.ST: 2.88 with
        # trailingAnnualDividendYield 0.0288), not the 0-1 fraction the other
        # ratios are. Scaled like them, the 2 Nov prompt showed KCC at 618% and
        # NOBA at 176%, and the model set every yield aside as mis-scaled.
        dy = _safe(data.get("dividend_yield"))
        feat.dividend_yield_pct = round(dy, 2) if dy is not None else None

        mc = data.get("market_cap")
        feat.market_cap_msek = round(mc / 1e6, 0) if mc is not None else None

        high = data.get("fifty_two_week_high")
        if high and high > 0 and feat.last_price:
            feat.pct_from_52w_high = round((feat.last_price / high - 1) * 100, 1)

        target = data.get("analyst_target_price")
        if target and feat.last_price and feat.last_price > 0:
            feat.analyst_target_pct = round((target / feat.last_price - 1) * 100, 1)

        feat.days_to_earnings = ts_to_days(data.get("earnings_timestamp"))
        feat.days_to_ex_div   = ts_to_days(data.get("ex_div_timestamp"))
