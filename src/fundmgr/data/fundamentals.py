"""
Fundamentals cache layer.

Fetching and caching is delegated to financedata (7-day SQLite TTL).
Data is mirrored to the fund's own store so apply_to_features (which reads
from the store) continues to work unchanged.
"""
from __future__ import annotations

import math

from financedata import get_fundamentals as fd_get_fundamentals, ts_to_days

from fundmgr.state.store import Store


def fetch_and_cache_fundamentals(
    tickers: list[str],
    store: Store,
    ttl_days: int = 7,
    max_workers: int = 12,
) -> int:
    """Fetch stale fundamentals via financedata and mirror to fund's store."""
    data = fd_get_fundamentals(tickers, ttl_days=ttl_days, max_workers=max_workers)
    refreshed = 0
    for ticker, d in data.items():
        store.save_fundamentals(ticker, d)
        refreshed += 1
    return refreshed


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
        feat.beta           = _safe(data.get("beta"))
        feat.analyst_count  = data.get("analyst_count")

        for frac_key, attr in (
            ("profit_margin",   "profit_margin_pct"),
            ("gross_margin",    "gross_margin_pct"),
            ("roe",             "roe_pct"),
            ("revenue_growth",  "revenue_growth_pct"),
            ("earnings_growth", "earnings_growth_pct"),
            ("dividend_yield",  "dividend_yield_pct"),
        ):
            raw = data.get(frac_key)
            setattr(feat, attr, round(raw * 100, 1) if raw is not None else None)

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
