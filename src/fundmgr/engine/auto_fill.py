"""
Auto-fill engine for the paper trading simulation.

After each fund run, if auto_fill=True (global fund config), this module:
1. Waits until all relevant markets are open (or uses latest available price)
2. Fetches the open/current price for each approved action
3. Records fills via store.apply_fill() — same path as manual fills
4. Records a NAV snapshot
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import yfinance as yf

from fundmgr.config import AppConfig
from fundmgr.data.benchmark import get_benchmark_return_pct
from fundmgr.state.models import NavPoint, Transaction
from fundmgr.state.store import Store


def _fetch_price(ticker: str) -> float | None:
    """Fetch the latest available price for a ticker."""
    try:
        info = yf.Ticker(ticker).fast_info
        price = getattr(info, "last_price", None) or getattr(info, "open", None)
        return float(price) if price and price > 0 else None
    except Exception:
        return None


def _compute_shares(nav_sek: float, target_weight_pct: float, price_sek: float) -> float:
    """Compute how many whole shares to buy for a given target weight."""
    target_sek = nav_sek * target_weight_pct / 100.0
    return max(0.0, target_sek / price_sek)


def execute_paper_fills(
    actions: list[dict],
    store: Store,
    cfg: AppConfig,
    *,
    max_wait_secs: int = 0,
) -> list[str]:
    """
    Execute paper fills for all approved buy/sell actions.

    actions: list of action dicts from guardrail_result.approved_actions
    max_wait_secs: if > 0, poll until price is available (for use at market open)

    Returns list of log lines describing what was executed.
    """
    log: list[str] = []
    positions = store.get_positions()
    pos_map = {p.ticker: p for p in positions}
    nav = sum(p.shares * (p.current_price_sek or p.avg_cost_sek) for p in positions) + store.get_cash()

    for action in actions:
        ticker = action.get("ticker", "")
        side = action.get("side", "")
        target_weight_pct = action.get("target_weight_pct", 0.0)

        if side not in ("buy", "sell"):
            continue

        # Fetch price with optional wait
        price = None
        waited = 0
        while price is None:
            price = _fetch_price(ticker)
            if price or waited >= max_wait_secs:
                break
            time.sleep(30)
            waited += 30

        if not price:
            log.append(f"  ⚠ {ticker}: could not fetch price — skipped")
            continue

        if side == "buy":
            current_weight = (
                (pos_map[ticker].shares * price / nav * 100) if ticker in pos_map else 0.0
            )
            weight_gap = target_weight_pct - current_weight
            if weight_gap <= 0:
                log.append(f"  {ticker}: already at/above target weight — skipped")
                continue
            buy_sek = nav * weight_gap / 100.0
            if buy_sek < cfg.risk.min_trade_sek:
                log.append(f"  {ticker}: trade size {buy_sek:.0f} SEK below minimum — skipped")
                continue
            shares = buy_sek / price
            fee = cfg.fees.calc(buy_sek)

        elif side == "sell":
            if ticker not in pos_map:
                log.append(f"  {ticker}: not held — skipped sell")
                continue
            pos = pos_map[ticker]
            if target_weight_pct == 0:
                # Full sell
                shares = pos.shares
            else:
                target_sek = nav * target_weight_pct / 100.0
                current_sek = pos.shares * price
                sell_sek = current_sek - target_sek
                if sell_sek < cfg.risk.min_trade_sek:
                    log.append(f"  {ticker}: sell size {sell_sek:.0f} SEK below minimum — skipped")
                    continue
                shares = sell_sek / price
            fee = cfg.fees.calc(shares * price)

        txn = Transaction(
            ticker=ticker,
            side=side,
            shares=round(shares, 4),
            price_sek=round(price, 4),
            fee_sek=round(fee, 4),
            source="auto",
            timestamp=datetime.now(timezone.utc),
        )
        store.apply_fill(txn)
        log.append(
            f"  ✓ {'Bought' if side == 'buy' else 'Sold'} {shares:.2f} × {ticker} "
            f"@ {price:.2f} SEK (fee {fee:.2f})"
        )

    # Record NAV snapshot after all fills
    try:
        from fundmgr.state.store import Store as _S
        bench_rows = store.get_benchmark()
        bench_val = bench_rows[-1]["close"] if bench_rows else 0.0
        positions_after = store.get_positions()
        cash_after = store.get_cash()
        nav_after = sum(p.shares * p.avg_cost_sek for p in positions_after) + cash_after
        store.upsert_nav(NavPoint(
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            portfolio_nav_sek=nav_after,
            benchmark_value=bench_val,
            cash_sek=cash_after,
        ))
    except Exception:
        pass

    return log
