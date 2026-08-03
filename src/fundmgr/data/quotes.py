"""
Live / intraday quotes — native currency.

Thin wrapper over the shared financedata.get_live_price(s) (cached, 10-min TTL),
with a local yfinance fallback so the fund keeps working if that version of
financedata isn't installed yet. Returns prices in the stock's native currency;
convert to SEK separately via fx.rate_to_sek when comparing to a SEK cost basis.
"""
from __future__ import annotations


def _local_live_price(ticker: str) -> float | None:
    """Fallback: latest close via yfinance. None on failure."""
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="2d")
        if hist is None or hist.empty:
            return None
        price = float(hist["Close"].iloc[-1])
        return price if price > 0 else None
    except Exception:
        return None


def live_price(ticker: str) -> float | None:
    """Latest price for `ticker` in its native currency. None when unavailable."""
    try:
        from financedata import get_live_price
    except ImportError:
        return _local_live_price(ticker)
    return get_live_price(ticker)


def live_prices(tickers: list[str]) -> dict[str, float | None]:
    """Batched live_price. Uses financedata's batched fetch when available."""
    try:
        from financedata import get_live_prices
    except ImportError:
        return {t: _local_live_price(t) for t in tickers}
    return get_live_prices(tickers)
