from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from fundmgr.config import AppConfig, UniverseTicker
from fundmgr.state.store import Store


@dataclass
class TickerFeatures:
    ticker: str
    name: str
    last_price: float
    last_date: str
    data_age_trading_days: int  # trading days since last close (0 = today / yesterday)
    return_1d_pct: float | None = None
    return_5d_pct: float | None = None
    return_20d_pct: float | None = None
    return_60d_pct: float | None = None
    vol_20d_ann_pct: float | None = None   # annualised volatility %
    rsi_14: float | None = None
    above_ma50: bool | None = None
    above_ma200: bool | None = None
    # Fundamentals (yfinance, may be None for many SE tickers)
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    dividend_yield_pct: float | None = None
    market_cap_msek: float | None = None
    # Sentiment (filled by news pipeline)
    sentiment_label: str | None = None    # positive | negative | neutral
    sentiment_score: float | None = None
    news_count: int = 0
    # FX
    currency: str = "SEK"
    needs_fx: bool = False

    @property
    def is_stale(self) -> bool:
        return self.data_age_trading_days > 5

    def to_prompt_block(self) -> str:
        """Compact text block for the LLM prompt."""
        fx_note = f"  ⚠ FX: trades in {self.currency} — SEK conversion cost applies" if self.needs_fx else ""
        lines = [f"[{self.ticker}] {self.name}"]
        lines.append(f"  Price: {self.last_price:.2f} SEK  (as of {self.last_date})")
        if fx_note:
            lines.append(fx_note)

        ret_parts = []
        if self.return_1d_pct is not None:
            ret_parts.append(f"1d {self.return_1d_pct:+.1f}%")
        if self.return_5d_pct is not None:
            ret_parts.append(f"5d {self.return_5d_pct:+.1f}%")
        if self.return_20d_pct is not None:
            ret_parts.append(f"20d {self.return_20d_pct:+.1f}%")
        if self.return_60d_pct is not None:
            ret_parts.append(f"60d {self.return_60d_pct:+.1f}%")
        if ret_parts:
            lines.append(f"  Returns: {', '.join(ret_parts)}")

        tech_parts = []
        if self.vol_20d_ann_pct is not None:
            tech_parts.append(f"vol {self.vol_20d_ann_pct:.0f}%")
        if self.rsi_14 is not None:
            tech_parts.append(f"RSI {self.rsi_14:.0f}")
        if self.above_ma50 is not None:
            tech_parts.append(f"{'above' if self.above_ma50 else 'below'} MA50")
        if self.above_ma200 is not None:
            tech_parts.append(f"{'above' if self.above_ma200 else 'below'} MA200")
        if tech_parts:
            lines.append(f"  Technical: {', '.join(tech_parts)}")

        fund_parts = []
        if self.pe_ratio is not None:
            fund_parts.append(f"P/E {self.pe_ratio:.1f}x")
        if self.pb_ratio is not None:
            fund_parts.append(f"P/B {self.pb_ratio:.1f}x")
        if self.dividend_yield_pct is not None:
            fund_parts.append(f"yield {self.dividend_yield_pct:.1f}%")
        if self.market_cap_msek is not None:
            fund_parts.append(f"mkt cap {self.market_cap_msek:,.0f}M SEK")
        if fund_parts:
            lines.append(f"  Fundamentals: {', '.join(fund_parts)}")

        if self.sentiment_label:
            lines.append(
                f"  Sentiment: {self.sentiment_label} ({self.sentiment_score:.2f}) "
                f"from {self.news_count} headlines"
            )
        else:
            lines.append("  Sentiment: no recent news")

        if self.is_stale:
            lines.append(f"  ⚠ DATA STALE ({self.data_age_trading_days} trading days old)")

        return "\n".join(lines)


def _rsi(closes: pd.Series, period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    delta = closes.diff().dropna()
    gains = delta.clip(lower=0).rolling(period).mean()
    losses = (-delta).clip(lower=0).rolling(period).mean()
    if losses.iloc[-1] == 0:
        return 100.0
    rs = gains.iloc[-1] / losses.iloc[-1]
    return round(100 - (100 / (1 + rs)), 1)


def _pct_return(closes: pd.Series, periods: int) -> float | None:
    if len(closes) < periods + 1:
        return None
    val = (closes.iloc[-1] / closes.iloc[-1 - periods] - 1) * 100
    return round(val, 2)


def _ann_vol(closes: pd.Series, periods: int = 20) -> float | None:
    if len(closes) < periods + 1:
        return None
    log_ret = (closes / closes.shift(1)).apply(lambda x: x if x > 0 else float("nan")).apply(
        lambda x: __import__("math").log(x) if not __import__("math").isnan(x) else float("nan")
    )
    std = log_ret.iloc[-periods:].std()
    if pd.isna(std):
        return None
    return round(std * math.sqrt(252) * 100, 1)


def _count_trading_days_since(last_date_str: str) -> int:
    """Approximate business days between last_date and today."""
    last = datetime.strptime(last_date_str, "%Y-%m-%d").date()
    today = datetime.utcnow().date()
    if last >= today:
        return 0
    count = 0
    cur = last + timedelta(days=1)
    while cur <= today:
        if cur.weekday() < 5:  # Mon-Fri
            count += 1
        cur += timedelta(days=1)
    return count


def fetch_and_cache_prices(
    tickers: list[UniverseTicker],
    store: Store,
    lookback_days: int = 252,
    force_refresh: bool = False,
) -> dict[str, bool]:
    """Batch-fetch prices for all tickers; return dict ticker -> success."""
    symbols = [t.yahoo_ticker for t in tickers]
    since = (datetime.utcnow() - timedelta(days=lookback_days + 10)).strftime("%Y-%m-%d")

    # Determine which need refreshing
    to_fetch = []
    for t in tickers:
        if force_refresh:
            to_fetch.append(t.yahoo_ticker)
            continue
        latest = store.latest_price_date(t.yahoo_ticker)
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if latest is None or latest < today:
            to_fetch.append(t.yahoo_ticker)

    if not to_fetch:
        return {t.yahoo_ticker: True for t in tickers}

    results: dict[str, bool] = {}

    # yfinance batch download
    try:
        raw = yf.download(
            to_fetch,
            start=since,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        print(f"  yfinance download error: {e}")
        return {sym: False for sym in to_fetch}

    if raw.empty:
        return {sym: False for sym in to_fetch}

    # Normalise to a (date, ticker) -> close DataFrame regardless of yfinance version.
    # Older yfinance: MultiIndex columns (price_type, ticker) — "Close" at level 0.
    # Newer yfinance: MultiIndex columns (ticker, price_type) OR flat for single ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0).unique().tolist()
        if "Close" in lvl0:
            # (price_type, ticker) layout
            closes_df = raw["Close"]
        else:
            # (ticker, price_type) layout — swap levels then index by "Close"
            closes_df = raw.swaplevel(axis=1)["Close"]
    else:
        # Single ticker, flat columns
        closes_df = raw[["Close"]].rename(columns={"Close": to_fetch[0]})

    for sym in to_fetch:
        try:
            if sym not in closes_df.columns:
                results[sym] = False
                continue
            series = closes_df[sym].dropna()
            if series.empty:
                results[sym] = False
                continue
            # Build full OHLCV rows by slicing from raw for this ticker
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    lvl0 = raw.columns.get_level_values(0).unique().tolist()
                    if "Close" in lvl0:
                        df_sym = raw.xs(sym, axis=1, level=1)
                    else:
                        df_sym = raw.xs(sym, axis=1, level=0)
                else:
                    df_sym = raw.copy()
                if isinstance(df_sym.columns, pd.MultiIndex):
                    df_sym.columns = df_sym.columns.get_level_values(0)
            except Exception:
                # Fall back to close-only rows
                df_sym = series.to_frame("Close")

            df_sym = df_sym[~df_sym["Close"].isna()] if "Close" in df_sym.columns else df_sym
            df_sym.index = df_sym.index.strftime("%Y-%m-%d")
            rows = [
                {
                    "date": str(idx),
                    "open": float(row["Open"]) if "Open" in row and not pd.isna(row["Open"]) else None,
                    "high": float(row["High"]) if "High" in row and not pd.isna(row["High"]) else None,
                    "low": float(row["Low"]) if "Low" in row and not pd.isna(row["Low"]) else None,
                    "close": float(row["Close"]) if not pd.isna(row["Close"]) else None,
                    "volume": float(row["Volume"]) if "Volume" in row and not pd.isna(row["Volume"]) else None,
                }
                for idx, row in df_sym.iterrows()
                if "Close" in row and not pd.isna(row["Close"])
            ]
            if rows:
                store.save_prices(sym, rows)
                results[sym] = True
            else:
                results[sym] = False
        except Exception:
            results[sym] = False

    # Tickers that were already cached
    for t in tickers:
        if t.yahoo_ticker not in results:
            results[t.yahoo_ticker] = store.latest_price_date(t.yahoo_ticker) is not None

    return results


def compute_features(
    ticker: UniverseTicker,
    store: Store,
    since_date: str,
) -> TickerFeatures | None:
    rows = store.get_prices(ticker.yahoo_ticker, since_date=since_date)
    if not rows:
        return None

    closes = pd.Series(
        [r["close"] for r in rows],
        index=pd.to_datetime([r["date"] for r in rows]),
    )

    last_date = rows[-1]["date"]
    last_price = rows[-1]["close"]
    age = _count_trading_days_since(last_date)

    feat = TickerFeatures(
        ticker=ticker.yahoo_ticker,
        name=ticker.name,
        last_price=last_price,
        last_date=last_date,
        data_age_trading_days=age,
        currency=ticker.currency,
        needs_fx=ticker.needs_fx,
        return_1d_pct=_pct_return(closes, 1),
        return_5d_pct=_pct_return(closes, 5),
        return_20d_pct=_pct_return(closes, 20),
        return_60d_pct=_pct_return(closes, 60),
        vol_20d_ann_pct=_ann_vol(closes, 20),
        rsi_14=_rsi(closes),
        above_ma50=bool(last_price > closes.rolling(50).mean().iloc[-1]) if len(closes) >= 50 else None,
        above_ma200=bool(last_price > closes.rolling(200).mean().iloc[-1]) if len(closes) >= 200 else None,
    )

    # Fundamentals via yfinance (best-effort)
    try:
        info = yf.Ticker(ticker.yahoo_ticker).fast_info
        feat.pe_ratio = _safe_float(getattr(info, "pe_ratio", None))
        feat.market_cap_msek = _safe_float(getattr(info, "market_cap", None), scale=1e-6)
    except Exception:
        pass

    try:
        full_info = yf.Ticker(ticker.yahoo_ticker).info
        feat.pb_ratio = _safe_float(full_info.get("priceToBook"))
        dy = full_info.get("dividendYield")
        feat.dividend_yield_pct = round(dy * 100, 2) if dy else None
    except Exception:
        pass

    return feat


def _safe_float(val, scale: float = 1.0) -> float | None:
    try:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return None
        return round(float(val) * scale, 2)
    except (TypeError, ValueError):
        return None


def build_all_features(
    tickers: list[UniverseTicker],
    store: Store,
    cfg: AppConfig,
    fetch_result: dict[str, bool],
) -> dict[str, TickerFeatures]:
    since = (datetime.utcnow() - timedelta(days=cfg.data.lookback_days + 10)).strftime("%Y-%m-%d")
    features: dict[str, TickerFeatures] = {}
    for t in tickers:
        if not fetch_result.get(t.yahoo_ticker, False):
            continue
        feat = compute_features(t, store, since)
        if feat is not None:
            features[t.yahoo_ticker] = feat
    return features
