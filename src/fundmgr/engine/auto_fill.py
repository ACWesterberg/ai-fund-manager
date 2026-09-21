"""
Auto-fill engine for the paper trading simulation.

After each fund run, if auto_fill=True (global fund config), this module:
1. Skips any action whose exchange is closed (weekend/holiday/off-hours) so
   fills are never booked at stale prices — and sends a Telegram reminder
2. Fetches the current price for each remaining approved action
3. Records fills via store.apply_fill() — same path as manual fills
4. Records a NAV snapshot
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import NamedTuple

from fundmgr.config import AppConfig
from fundmgr.levels import (
    SKIP_AT_TARGET,
    SKIP_BELOW_MIN,
    SKIP_MARKET_CLOSED,
    SKIP_NOT_HELD,
    SKIP_NO_PRICE,
    SKIP_RISK,
)
from fundmgr.state.models import NavPoint, Transaction
from fundmgr.state.store import Store


class FillOutcome(NamedTuple):
    """What the filler did: log lines, and why each action that didn't fill didn't.

    `skipped` maps ticker -> a `levels.SKIP_*` reason. It exists because a
    caller cannot infer the reason from the book: check-stops reads unsold
    shares and used to call every one of them a closed market, which is true of
    a European name after its venue shuts and false of a fill that simply had
    no price.
    """
    log: list[str]
    skipped: dict[str, str]


def _fetch_price(ticker: str) -> float | None:
    """Fetch the latest available price for a ticker (shared cache via financedata)."""
    from fundmgr.data.quotes import live_price
    return live_price(ticker)


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
    notify_skips: bool = True,
) -> FillOutcome:
    """
    Execute paper fills for all approved buy/sell actions.

    actions: list of action dicts from guardrail_result.approved_actions
    max_wait_secs: if > 0, poll until price is available (for use at market open)
    notify_skips: if True, send a Telegram reminder when fills are skipped
                  because their exchange is closed (holiday/off-hours)

    Returns a FillOutcome: the log lines describing what was executed, and the
    reason every skipped action was skipped.
    """
    log: list[str] = []
    skipped: dict[str, str] = {}
    prices: dict[str, float] = {}

    def quote(ticker: str) -> float | None:
        if ticker in prices:
            return prices[ticker]
        price = _fetch_price(ticker)
        if price is None or not math.isfinite(price) or price <= 0:
            return None
        if cfg.fx_to_sek:
            from fundmgr.data.fx import rate_to_sek
            rate = rate_to_sek(currency_by_ticker.get(ticker, "SEK"), store)
            if rate is None or not math.isfinite(rate) or rate <= 0:
                return None
            price *= rate
        prices[ticker] = price
        return price

    def marked_nav() -> float | None:
        value = store.get_cash()
        for position in store.get_positions():
            price = quote(position.ticker)
            if price is None:
                return None
            value += position.shares * price
        return value

    # Ticker -> exchange code, so we can check each venue's trading calendar.
    from fundmgr.config import load_universe
    from fundmgr.data.market_hours import is_exchange_open
    universe = load_universe(cfg.universe_path)
    exch_by_ticker = {t.yahoo_ticker: t.exchange for t in universe}
    currency_by_ticker = {t.yahoo_ticker: t.currency for t in universe}
    skipped_closed: list[tuple[str, str, str]] = []  # (ticker, side, exchange)

    for action in actions:
        ticker = action.get("ticker", "")
        side = action.get("side", "")
        target_weight_pct = action.get("target_weight_pct", 0.0)

        if side not in ("buy", "sell"):
            continue

        # Skip fills when the stock's exchange is closed (weekend/holiday/off-hours)
        # so we never book a trade at a stale price. None = unknown venue → fail open.
        exchange = exch_by_ticker.get(ticker, "")
        if is_exchange_open(exchange) is False:
            skipped_closed.append((ticker, side, exchange or "?"))
            skipped[ticker] = SKIP_MARKET_CLOSED
            log.append(f"  ⏸ {ticker}: {exchange or '?'} closed — fill skipped")
            continue

        # Fetch price with optional wait
        price = None
        waited = 0
        while price is None:
            price = quote(ticker)
            if price or waited >= max_wait_secs:
                break
            time.sleep(30)
            waited += 30

        if not price:
            skipped[ticker] = SKIP_NO_PRICE
            log.append(f"  ⚠ {ticker}: could not fetch price — skipped")
            continue

        pos_map = {p.ticker: p for p in store.get_positions()}
        nav = marked_nav()
        if (nav is None or nav <= 0) and not (side == "sell" and target_weight_pct == 0):
            skipped[ticker] = SKIP_NO_PRICE
            log.append(f"  ⚠ {ticker}: incomplete portfolio valuation — skipped")
            continue

        if side == "buy":
            target_weight_pct = min(target_weight_pct, cfg.risk.max_position_pct)
            current_weight = (
                (pos_map[ticker].shares * price / nav * 100) if ticker in pos_map else 0.0
            )
            weight_gap = target_weight_pct - current_weight
            if weight_gap <= 0:
                skipped[ticker] = SKIP_AT_TARGET
                log.append(f"  {ticker}: already at/above target weight — skipped")
                continue
            budget = action.get("sek_estimate")
            if budget is None or not math.isfinite(budget) or budget <= 0:
                skipped[ticker] = SKIP_RISK
                log.append(f"  ⚠ {ticker}: missing approved trade budget — skipped")
                continue
            buy_sek = min(budget, nav * weight_gap / 100.0)
            if store.get_cash() - buy_sek - cfg.fees.calc(buy_sek) < nav * cfg.risk.min_cash_pct / 100:
                skipped[ticker] = SKIP_RISK
                log.append(f"  ⚠ {ticker}: cash floor including fees — skipped")
                continue
            if buy_sek < cfg.risk.min_trade_sek:
                skipped[ticker] = SKIP_BELOW_MIN
                log.append(f"  {ticker}: trade size {buy_sek:.0f} SEK below minimum — skipped")
                continue
            shares = math.floor(buy_sek / price * 10000) / 10000
            fee = cfg.fees.calc(shares * price)

        elif side == "sell":
            if ticker not in pos_map:
                skipped[ticker] = SKIP_NOT_HELD
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
                    skipped[ticker] = SKIP_BELOW_MIN
                    log.append(f"  {ticker}: sell size {sell_sek:.0f} SEK below minimum — skipped")
                    continue
                shares = sell_sek / price
            if "sek_estimate" in action:
                budget = action["sek_estimate"]
                if not math.isfinite(budget) or budget <= 0:
                    skipped[ticker] = SKIP_RISK
                    continue
                shares = min(shares, math.floor(budget / price * 10000) / 10000)
            fee = cfg.fees.calc(shares * price)

        if shares <= 0:
            skipped[ticker] = SKIP_BELOW_MIN
            continue
        txn = Transaction(
            ticker=ticker,
            side=side,
            shares=shares,
            price_sek=price,
            fee_sek=fee,
            source="auto",
            timestamp=datetime.now(timezone.utc),
        )
        try:
            store.apply_fill(txn)
        except ValueError as exc:
            skipped[ticker] = SKIP_RISK
            log.append(f"  ⚠ {ticker}: {exc} — skipped")
            continue
        log.append(
            f"  ✓ {'Bought' if side == 'buy' else 'Sold'} {shares:.2f} × {ticker} "
            f"@ {price:.2f} SEK (fee {fee:.2f})"
        )

    # Reminder: orders that didn't execute because their market was closed.
    if skipped_closed and notify_skips:
        from fundmgr.notify.send import send_telegram
        lines = [
            f"<b>{cfg.display_name}</b>\n⏸ Fills skipped — market closed",
            f"{len(skipped_closed)} order(s) not executed (holiday/off-hours):",
        ]
        lines += [f"  {tk}  {sd.upper()}  [{ex}]" for tk, sd, ex in skipped_closed]
        lines.append("They'll be reconsidered on the next run.")
        send_telegram("\n".join(lines))

    # Record NAV snapshot after all fills
    try:
        bench_rows = store.get_benchmark()
        bench_val = bench_rows[-1]["close"] if bench_rows else 0.0
        cash_after = store.get_cash()
        nav_after = marked_nav()
        if nav_after is None:
            log.append("  ⚠ NAV not saved: incomplete portfolio valuation")
            return FillOutcome(log, skipped)
        store.upsert_nav(NavPoint(
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            portfolio_nav_sek=nav_after,
            benchmark_value=bench_val,
            cash_sek=cash_after,
        ))
    except Exception as exc:
        log.append(f"  ⚠ Could not save NAV: {exc}")

    return FillOutcome(log, skipped)
