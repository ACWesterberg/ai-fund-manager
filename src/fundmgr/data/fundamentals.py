"""
Fundamentals cache layer.

Fetches valuation, quality, growth, and analyst data from yfinance .info
and caches it in SQLite with a weekly TTL so individual runs stay fast.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import yfinance as yf

from fundmgr.state.store import Store

# yfinance info key → our cache key
_FIELD_MAP: dict[str, str] = {
    "trailingPE":                       "pe_ratio",
    "forwardPE":                        "forward_pe",
    "priceToBook":                      "pb_ratio",
    "enterpriseToEbitda":               "ev_to_ebitda",
    "priceToSalesTrailingTwelveMonths": "price_to_sales",
    "marketCap":                        "market_cap",
    "profitMargins":                    "profit_margin",  # fraction → pct on apply
    "grossMargins":                     "gross_margin",
    "returnOnEquity":                   "roe",
    "debtToEquity":                     "debt_to_equity",
    "revenueGrowth":                    "revenue_growth",  # fraction → pct on apply
    "earningsGrowth":                   "earnings_growth",
    "beta":                             "beta",
    "fiftyTwoWeekHigh":                 "fifty_two_week_high",
    "fiftyTwoWeekLow":                  "fifty_two_week_low",
    "dividendYield":                    "dividend_yield",  # fraction → pct on apply
    "targetMeanPrice":                  "analyst_target_price",
    "numberOfAnalystOpinions":          "analyst_count",
    "currency":                         "currency",
    "earningsTimestamp":                "earnings_timestamp",  # Unix ts → days_to_earnings
    "exDividendDate":                   "ex_div_timestamp",    # Unix ts → days_to_ex_div
}

# Fields stored as fractions that should be reported as percentages
_FRACTION_FIELDS = {"profit_margin", "gross_margin", "roe", "revenue_growth", "earnings_growth", "dividend_yield"}


def _ts_to_days(ts) -> int | None:
    """Convert Unix timestamp to calendar days from today. Returns None if past or invalid."""
    if ts is None:
        return None
    try:
        today = datetime.utcnow().date()
        target = datetime.utcfromtimestamp(float(ts)).date()
        delta = (target - today).days
        return delta if delta >= 0 else None
    except (TypeError, ValueError, OSError):
        return None


def _safe(val) -> float | None:
    try:
        if val is None:
            return None
        f = float(val)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _fetch_one(ticker: str) -> tuple[str, dict | None]:
    try:
        info = yf.Ticker(ticker).info
        if not info or info.get("quoteType") == "NONE":
            return ticker, None
        data: dict = {}
        for yf_key, our_key in _FIELD_MAP.items():
            raw = info.get(yf_key)
            if our_key == "currency":
                data[our_key] = raw if isinstance(raw, str) else None
            elif our_key == "analyst_count":
                data[our_key] = int(raw) if raw is not None else None
            else:
                data[our_key] = _safe(raw)
        return ticker, data
    except Exception:
        return ticker, None


def fetch_and_cache_fundamentals(
    tickers: list[str],
    store: Store,
    ttl_days: int = 7,
    max_workers: int = 12,
) -> int:
    """
    Fetch fundamentals for any ticker whose cache is stale or missing.
    Returns count of tickers refreshed.
    """
    stale = store.get_stale_fundamentals_tickers(tickers, ttl_days=ttl_days)
    if not stale:
        return 0

    refreshed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in stale}
        for fut in as_completed(futures):
            ticker, data = fut.result()
            if data is not None:
                store.save_fundamentals(ticker, data)
                refreshed += 1

    return refreshed


def apply_to_features(features: dict, store: Store) -> None:
    """
    Pull cached fundamentals and attach them to TickerFeatures objects in-place.
    `features` is a dict[str, TickerFeatures] as returned by build_all_features.
    """
    cached = store.get_all_fundamentals(list(features.keys()))

    for ticker, feat in features.items():
        data = cached.get(ticker)
        if not data:
            continue

        feat.pe_ratio        = data.get("pe_ratio")
        feat.forward_pe      = data.get("forward_pe")
        feat.pb_ratio        = data.get("pb_ratio")
        feat.ev_to_ebitda    = data.get("ev_to_ebitda")
        feat.price_to_sales  = data.get("price_to_sales")
        feat.beta            = data.get("beta")
        feat.analyst_count   = data.get("analyst_count")

        # Convert fractions → percentages
        for frac_key, attr in (
            ("profit_margin",  "profit_margin_pct"),
            ("gross_margin",   "gross_margin_pct"),
            ("roe",            "roe_pct"),
            ("revenue_growth", "revenue_growth_pct"),
            ("earnings_growth","earnings_growth_pct"),
            ("dividend_yield", "dividend_yield_pct"),
        ):
            raw = data.get(frac_key)
            setattr(feat, attr, round(raw * 100, 1) if raw is not None else None)

        # Market cap: already in native currency units, convert to millions
        mc = data.get("market_cap")
        feat.market_cap_msek = round(mc / 1e6, 0) if mc is not None else None

        # 52-week positioning
        high = data.get("fifty_two_week_high")
        if high and high > 0 and feat.last_price:
            feat.pct_from_52w_high = round((feat.last_price / high - 1) * 100, 1)

        # Analyst upside/downside vs current price
        target = data.get("analyst_target_price")
        if target and feat.last_price and feat.last_price > 0:
            feat.analyst_target_pct = round((target / feat.last_price - 1) * 100, 1)

        # Calendar: days to next earnings and ex-dividend date (future only)
        feat.days_to_earnings = _ts_to_days(data.get("earnings_timestamp"))
        feat.days_to_ex_div = _ts_to_days(data.get("ex_div_timestamp"))
