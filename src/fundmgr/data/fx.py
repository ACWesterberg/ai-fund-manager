"""
FX conversion to SEK (the fund's base currency).

The fund books cash and NAV in SEK, but holdings on foreign exchanges are
quoted in their native currency (DKK, EUR, NOK, USD, …). Conversion now delegates
to the shared financedata.get_fx_rate (single cached source across projects),
with a local yfinance fallback for older financedata installs.

Design notes:
  - SEK → identity (rate 1.0), no fetch.
  - On fetch failure rate_to_sek returns None so callers can degrade loudly
    rather than silently mis-converting.
  - Per-share prices stay in native currency elsewhere (so returns / RSI /
    stop-% remain internally consistent); FX is applied only where amounts
    enter the SEK books (cash at fill, NAV / market value).
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fundmgr.state.store import Store

# In-process memo so a single run doesn't even hit app_meta repeatedly.
_MEMO: dict[str, float] = {}


def _fetch_rate(currency: str) -> float | None:
    """Spot {currency}->SEK from yfinance. None on failure."""
    try:
        import yfinance as yf
        hist = yf.Ticker(f"{currency}SEK=X").history(period="5d")
        if hist is None or hist.empty:
            return None
        rate = float(hist["Close"].iloc[-1])
        return rate if rate > 0 else None
    except Exception:
        return None


def rate_to_sek(currency: str, store: "Store | None" = None) -> float | None:
    """Return the {currency}->SEK rate (1.0 for SEK). None when unavailable.

    Prefers the shared financedata.get_fx_rate (single cached source across
    projects); falls back to a local yfinance fetch if that version of
    financedata isn't installed yet.
    """
    cur = (currency or "SEK").upper()
    if cur == "SEK":
        return 1.0
    try:
        from financedata import get_fx_rate
    except ImportError:
        return _local_rate_to_sek(cur, store)
    return get_fx_rate(cur, "SEK")


def _local_rate_to_sek(cur: str, store: "Store | None" = None) -> float | None:
    """Fallback: local yfinance fetch + per-day cache (pre-financedata-FX)."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    memo_key = f"{cur}:{today}"
    if memo_key in _MEMO:
        return _MEMO[memo_key]

    meta_key = f"fxrate:{cur}:{today}"
    if store is not None:
        cached = store.get_meta(meta_key)
        if cached is not None:
            try:
                r = float(cached)
                _MEMO[memo_key] = r
                return r
            except ValueError:
                pass

    rate = _fetch_rate(cur)
    if rate is not None:
        _MEMO[memo_key] = rate
        if store is not None:
            store.set_meta(meta_key, str(rate))
    return rate


def to_sek(amount: float, currency: str, store: "Store | None" = None) -> float | None:
    """Convert `amount` in `currency` to SEK. None if the rate is unavailable."""
    rate = rate_to_sek(currency, store)
    return None if rate is None else amount * rate


def convert_prices_to_sek(
    prices: dict[str, float],
    cur_by_ticker: dict[str, str],
    store: "Store | None" = None,
) -> dict[str, float]:
    """Convert a {ticker: native_price} map to {ticker: SEK_price}.

    Each ticker's currency comes from `cur_by_ticker` (SEK / unknown pass through
    unchanged). Rates are fetched once per currency. If a rate is unavailable the
    native price is kept as-is (degrade rather than drop the position), matching
    the CLI's `... or 1.0` fallback."""
    out: dict[str, float] = {}
    rate_cache: dict[str, float] = {}
    for ticker, price in prices.items():
        cur = (cur_by_ticker.get(ticker) or "SEK").upper()
        if cur == "SEK":
            out[ticker] = price
            continue
        rate = rate_cache.get(cur)
        if rate is None:
            rate = rate_to_sek(cur, store) or 1.0
            rate_cache[cur] = rate
        out[ticker] = price * rate
    return out
