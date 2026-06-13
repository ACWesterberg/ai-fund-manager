from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf  # still needed for batch price download

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
    # ── Valuation (filled from fundamentals cache) ───────────────────────────
    pe_ratio: float | None = None           # trailing P/E
    forward_pe: float | None = None         # forward P/E
    pb_ratio: float | None = None           # price / book
    ev_to_ebitda: float | None = None       # enterprise value / EBITDA
    price_to_sales: float | None = None     # trailing P/S
    market_cap_msek: float | None = None    # market cap, millions (native currency)
    dividend_yield_pct: float | None = None
    # ── Quality / profitability ───────────────────────────────────────────────
    gross_margin_pct: float | None = None
    profit_margin_pct: float | None = None
    roe_pct: float | None = None            # return on equity
    debt_to_equity: float | None = None
    # ── Growth ───────────────────────────────────────────────────────────────
    revenue_growth_pct: float | None = None
    earnings_growth_pct: float | None = None
    # ── Market / risk ────────────────────────────────────────────────────────
    beta: float | None = None
    pct_from_52w_high: float | None = None  # negative = below high
    # ── Analyst consensus ────────────────────────────────────────────────────
    analyst_target_pct: float | None = None  # mean target % upside vs current price
    analyst_count: int | None = None
    # ── Calendar events (filled from fundamentals cache) ─────────────────────
    days_to_earnings: int | None = None      # trading days until next earnings report
    days_to_ex_div: int | None = None        # calendar days until next ex-dividend date
    # ── Volume ───────────────────────────────────────────────────────────────
    rel_volume: float | None = None          # latest day volume / 20d avg volume
    # ── Classification ───────────────────────────────────────────────────────
    sector: str | None = None               # GICS sector from universe CSV
    # ── Sentiment (filled by news pipeline) ──────────────────────────────────
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
        lines = [f"[{self.ticker}] {self.name}"]

        price_line = f"  Price: {self.last_price:.2f}  (as of {self.last_date})"
        if self.needs_fx:
            price_line += f"  [FX: {self.currency}]"
        if self.market_cap_msek is not None:
            price_line += f"  mktcap {self.market_cap_msek:,.0f}M"
        lines.append(price_line)

        # Returns
        ret_parts = []
        for label, val in (("1d", self.return_1d_pct), ("5d", self.return_5d_pct),
                           ("20d", self.return_20d_pct), ("60d", self.return_60d_pct)):
            if val is not None:
                ret_parts.append(f"{label} {val:+.1f}%")
        if self.pct_from_52w_high is not None:
            ret_parts.append(f"52wH {self.pct_from_52w_high:+.1f}%")
        if ret_parts:
            lines.append(f"  Returns: {', '.join(ret_parts)}")

        # Technical
        tech_parts = []
        if self.vol_20d_ann_pct is not None:
            tech_parts.append(f"vol {self.vol_20d_ann_pct:.0f}%")
        if self.rsi_14 is not None:
            tech_parts.append(f"RSI {self.rsi_14:.0f}")
        if self.above_ma50 is not None:
            tech_parts.append(f"{'▲' if self.above_ma50 else '▼'} MA50")
        if self.above_ma200 is not None:
            tech_parts.append(f"{'▲' if self.above_ma200 else '▼'} MA200")
        if self.beta is not None:
            tech_parts.append(f"β {self.beta:.2f}")
        if tech_parts:
            lines.append(f"  Technical: {', '.join(tech_parts)}")

        # Valuation
        val_parts = []
        for label, val in (
            ("P/E", self.pe_ratio), ("fP/E", self.forward_pe),
            ("P/B", self.pb_ratio), ("EV/EBITDA", self.ev_to_ebitda),
            ("P/S", self.price_to_sales),
        ):
            if val is not None:
                val_parts.append(f"{label} {val:.1f}x")
        if val_parts:
            lines.append(f"  Valuation: {', '.join(val_parts)}")

        # Quality / profitability
        qual_parts = []
        if self.gross_margin_pct is not None:
            qual_parts.append(f"gm {self.gross_margin_pct:.1f}%")
        if self.profit_margin_pct is not None:
            qual_parts.append(f"nm {self.profit_margin_pct:.1f}%")
        if self.roe_pct is not None:
            qual_parts.append(f"ROE {self.roe_pct:.1f}%")
        if self.debt_to_equity is not None:
            qual_parts.append(f"D/E {self.debt_to_equity:.1f}")
        if qual_parts:
            lines.append(f"  Quality: {', '.join(qual_parts)}")

        # Growth
        grow_parts = []
        if self.revenue_growth_pct is not None:
            grow_parts.append(f"rev {self.revenue_growth_pct:+.1f}%")
        if self.earnings_growth_pct is not None:
            grow_parts.append(f"earn {self.earnings_growth_pct:+.1f}%")
        if grow_parts:
            lines.append(f"  Growth (YoY): {', '.join(grow_parts)}")

        # Analyst consensus
        if self.analyst_target_pct is not None:
            count_str = f" ({self.analyst_count} analysts)" if self.analyst_count else ""
            lines.append(f"  Analysts: mean target {self.analyst_target_pct:+.1f}% upside{count_str}")

        # Calendar events
        cal_parts = []
        if self.days_to_earnings is not None:
            cal_parts.append(f"earnings in {self.days_to_earnings}d")
        if self.days_to_ex_div is not None:
            cal_parts.append(f"ex-div in {self.days_to_ex_div}d")
        if self.dividend_yield_pct is not None:
            cal_parts.append(f"yield {self.dividend_yield_pct:.1f}%")
        if cal_parts:
            lines.append(f"  Calendar: {', '.join(cal_parts)}")

        # Volume
        if self.rel_volume is not None:
            lines.append(f"  Volume: {self.rel_volume:.1f}x 20d avg")

        # Sector / classification
        if self.sector:
            lines.append(f"  Sector: {self.sector}")

        # Sentiment
        if self.sentiment_label:
            lines.append(
                f"  Sentiment: {self.sentiment_label} ({self.sentiment_score:.2f}) "
                f"from {self.news_count} headlines"
            )
        else:
            lines.append("  Sentiment: no recent news")

        if self.is_stale:
            lines.append(f"  ⚠ DATA STALE ({self.data_age_trading_days}d old)")

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

    volumes = pd.Series([r["volume"] for r in rows if r.get("volume") is not None])
    rel_vol = None
    if len(volumes) >= 21:
        avg_20d = volumes.iloc[-21:-1].mean()
        if avg_20d > 0:
            rel_vol = round(float(volumes.iloc[-1]) / avg_20d, 2)

    return TickerFeatures(
        ticker=ticker.yahoo_ticker,
        name=ticker.name,
        last_price=last_price,
        last_date=last_date,
        data_age_trading_days=age,
        currency=ticker.currency,
        needs_fx=ticker.needs_fx,
        sector=ticker.sector or None,
        return_1d_pct=_pct_return(closes, 1),
        return_5d_pct=_pct_return(closes, 5),
        return_20d_pct=_pct_return(closes, 20),
        return_60d_pct=_pct_return(closes, 60),
        vol_20d_ann_pct=_ann_vol(closes, 20),
        rsi_14=_rsi(closes),
        above_ma50=bool(last_price > closes.rolling(50).mean().iloc[-1]) if len(closes) >= 50 else None,
        above_ma200=bool(last_price > closes.rolling(200).mean().iloc[-1]) if len(closes) >= 200 else None,
        rel_volume=rel_vol,
    )


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
