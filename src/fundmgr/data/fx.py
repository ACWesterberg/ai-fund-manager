"""
FX conversion to SEK (the fund's base currency).

The fund books cash and NAV in SEK, but holdings on foreign exchanges are
quoted in their native currency (DKK, EUR, NOK, USD, …). This converts native
amounts to SEK using daily spot rates from yfinance ({CUR}SEK=X), cached for the
day in the store's app_meta so we don't refetch per ticker.

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
    """Return the {currency}->SEK rate (1.0 for SEK), cached per day. None on failure."""
    cur = (currency or "SEK").upper()
    if cur == "SEK":
        return 1.0

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


def populate_fx(positions: list, store: "Store | None" = None, universe_path=None) -> list:
    """Set .currency and .fx_rate on each Position so SEK valuation is correct.

    Looks up each ticker's native currency from the universe and the daily
    rate from the FX cache. Unknown/unavailable → SEK identity (1.0). Returns
    the same list for convenience.
    """
    from fundmgr.config import load_config, load_universe
    if universe_path is None:
        universe_path = load_config().universe_path
    cur_map = {t.yahoo_ticker: t.currency for t in load_universe(universe_path)}
    for p in positions:
        p.currency = cur_map.get(p.ticker, "SEK")
        r = rate_to_sek(p.currency, store)
        p.fx_rate = r if r is not None else 1.0
    return positions
