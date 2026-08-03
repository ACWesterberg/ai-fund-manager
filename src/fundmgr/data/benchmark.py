from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from fundmgr.state.store import Store


def fetch_and_cache_benchmark(
    store: Store,
    symbol: str = "^OMXSPI",
    lookback_days: int = 280,
    force_refresh: bool = False,
) -> bool:
    """Fetch the benchmark index series and cache it. Returns True on success."""
    since = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    if not force_refresh:
        rows = store.get_benchmark(since_date=since)
        if rows and rows[-1]["date"] >= (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d"):
            return True  # fresh enough

    try:
        raw = yf.download(symbol, start=since, auto_adjust=True, progress=False)
    except Exception as e:
        print(f"  Benchmark fetch error ({symbol}): {e}")
        return False

    if raw.empty:
        return False

    # Extract the Close series robustly (yfinance sometimes returns multi-level columns)
    close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    close.index = close.index.strftime("%Y-%m-%d")
    rows = [{"date": str(idx), "close": float(v)} for idx, v in close.items()]

    if not rows:
        return False

    store.save_benchmark(rows)
    return True


def get_benchmark_return_pct(
    store: Store, since_date: str, until_date: str | None = None
) -> float | None:
    """Return percentage change of benchmark from since_date to until_date (default: latest available)."""
    rows = store.get_benchmark(since_date=since_date)
    if until_date:
        rows = [r for r in rows if r["date"] <= until_date]
    if len(rows) < 2:
        return None
    start = rows[0]["close"]
    end = rows[-1]["close"]
    if start == 0:
        return None
    return round((end / start - 1) * 100, 2)


def get_benchmark_series(store: Store) -> pd.Series:
    rows = store.get_benchmark()
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series(
        [r["close"] for r in rows],
        index=pd.to_datetime([r["date"] for r in rows]),
        name="benchmark",
    )
