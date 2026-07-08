from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from financedata import get_prices_since, rsi, pct_return, ann_vol, get_cache as get_fd_cache
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
    # Recent headlines (+ optional content snippet) for the LLM to interpret
    # directly, rather than relying only on the aggregated FinBERT label above.
    headlines: list[dict] = field(default_factory=list)
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

        # Sentiment + recent headlines (raw text for the LLM to interpret directly).
        # The FinBERT label is a hint; the headlines/snippets are the primary signal.
        if self.sentiment_label:
            lines.append(
                f"  Sentiment (FinBERT): {self.sentiment_label} ({self.sentiment_score:.2f}) "
                f"from {self.news_count} headlines"
            )
        else:
            lines.append("  Sentiment (FinBERT): no recent news")

        if self.headlines:
            lines.append("  Recent headlines:")
            for h in self.headlines:
                date = (h.get("published_at") or "")[:10]
                label = h.get("sentiment_label")
                tag = f"[{label[:3].upper()}] " if label else ""
                date_str = f"{date} " if date else ""
                headline = (h.get("headline") or "").strip()
                lines.append(f"    - {date_str}{tag}{headline}")
                summary = (h.get("summary") or "").strip()
                if summary:
                    lines.append(f"      {summary}")

        if self.is_stale:
            lines.append(f"  ⚠ DATA STALE ({self.data_age_trading_days}d old)")

        return "\n".join(lines)


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
    """Delegate price fetching to financedata; return dict ticker -> success."""
    symbols = [t.yahoo_ticker for t in tickers]
    since = (datetime.utcnow() - timedelta(days=lookback_days + 10)).strftime("%Y-%m-%d")

    results = get_prices_since(symbols, since=since, force_refresh=force_refresh)

    # Mirror from shared cache to fund's store so compute_features (which reads
    # from store) continues to work without changes.
    fd_cache = get_fd_cache()
    for sym, ok in results.items():
        if ok:
            rows = fd_cache.get_prices(sym, since_date=since)
            if rows:
                store.save_prices(sym, rows)

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
        return_1d_pct=pct_return(closes, 1),
        return_5d_pct=pct_return(closes, 5),
        return_20d_pct=pct_return(closes, 20),
        return_60d_pct=pct_return(closes, 60),
        vol_20d_ann_pct=ann_vol(closes, 20),
        rsi_14=rsi(closes),
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
