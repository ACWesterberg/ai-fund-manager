"""
Take-profit review — decides what to do when a position reaches its target.

Hitting a take-profit is a decision point, not a notification. The old behaviour
said "consider trimming" and left the level untouched, so the same target kept
firing while nothing resolved it. This mirrors the stop-loss review on the other
side of the trade: SELL (bank it), TRIM (bank part, re-target the rest), RAISE
(conviction has grown — hold it all at a higher target) or HOLD.

Unlike the stop review, this one does write: a RAISE or TRIM persists its new
take-profit level, because leaving the old one standing is precisely the bug
this exists to fix. It still executes no trades — a human sells.
"""
from __future__ import annotations

import html

from fundmgr.config import AppConfig, load_universe
from fundmgr.engine.review_common import (
    context_blocks,
    live_price_sek,
    log_review,
    majority,
    run_consensus,
    technicals_block,
    votes_str,
)
from fundmgr.engine.schema import TargetReview
from fundmgr.state.store import Store


def find_target_hits(store: Store, cfg: AppConfig | None = None) -> dict:
    """Scan holdings for positions at or above their take-profit level.

    Mirrors find_stop_breaches: returns {"hits": [...], "skipped": [...]} where
    hits are {ticker, chg, take_profit_pct, live} and skipped are
    {ticker, reason} for holdings that couldn't be evaluated — so a position
    with no target on record is reported rather than silently passed over.
    """
    from fundmgr.data.fx import rate_to_sek
    from fundmgr.data.quotes import live_price

    fx_on = bool(cfg and cfg.fx_to_sek)
    cur_by_ticker = {}
    if fx_on:
        cur_by_ticker = {t.yahoo_ticker: t.currency for t in load_universe(cfg.universe_path)}
    fx_cache: dict[str, float] = {}

    level_map = store.get_effective_stops()
    hits: list[dict] = []
    skipped: list[dict] = []
    for p in store.get_positions():
        tp_pct = level_map.get(p.ticker, {}).get("take_profit_pct")
        if not tp_pct or not p.avg_cost_sek:
            skipped.append({"ticker": p.ticker, "reason": "no target on record"})
            continue
        live = live_price(p.ticker)
        if live is None:
            skipped.append({"ticker": p.ticker, "reason": "price unavailable"})
            continue
        cur = cur_by_ticker.get(p.ticker, "SEK")
        if fx_on and cur != "SEK":  # native→SEK to compare with SEK cost basis
            rate = fx_cache.get(cur) or rate_to_sek(cur, store) or 1.0
            fx_cache[cur] = rate
            live = live * rate
        chg = (live / p.avg_cost_sek - 1) * 100
        if chg >= tp_pct:
            hits.append({"ticker": p.ticker, "chg": chg, "take_profit_pct": tp_pct, "live": live})
    return {"hits": hits, "skipped": skipped}


def _build_review_prompt(
    ticker: str, store: Store, cfg: AppConfig, live_price: float | None
) -> tuple[str, str, float | None] | None:
    """Assemble (system, user, native_price), or None if the ticker isn't held.

    The native price is what the decision gets logged at — outcomes are scored
    against cached closes, which are native, never SEK-converted.
    """
    pos = next((p for p in store.get_positions() if p.ticker == ticker), None)
    if pos is None:
        return None

    uni = load_universe(cfg.universe_path)
    name = {t.yahoo_ticker: t.name for t in uni}.get(ticker, ticker)
    currency = {t.yahoo_ticker: t.currency for t in uni}.get(ticker, "SEK")
    levels = store.get_effective_stops().get(ticker, {})
    tp_pct = levels.get("take_profit_pct")

    technicals, tech_live = technicals_block(ticker)
    live = live_price_sek(ticker, pos, cfg, store, currency, live_price, tech_live)
    chg = (live / pos.avg_cost_sek - 1) * 100 if pos.avg_cost_sek else 0.0
    mkt_value = pos.shares * live

    decisions_block, news_block = context_blocks(ticker, store)

    system = (
        "You are the portfolio manager for a real-money equity fund. A position "
        "has reached its take-profit target. Decide whether to bank the gain or "
        "to keep holding at a higher target — and if you keep holding, say what "
        "the new target is and why the upside justifies it. Letting a winner run "
        "is legitimate; leaving a stale target in place is not. This is advisory "
        "for the sale — a human executes — but the target you set is recorded."
    )

    tp_str = f"+{tp_pct:.0f}%" if tp_pct else "n/a"
    user = f"""# Take-Profit Review — {ticker} ({name})

## Position
Shares: {pos.shares:.0f}  |  Avg cost: {pos.avg_cost_sek:.2f}  |  Live: {live:.2f}
Change since entry: {chg:+.1f}%   (take-profit level {tp_str} — now reached)
Market value: {mkt_value:,.0f} SEK

## Most recent decisions on this name
{decisions_block}

## Current technicals
{technicals}

## Recent news / sentiment (14d)
{news_block}

## Your task
The target has been reached at {chg:+.1f}%. Choose SELL (bank the whole gain),
TRIM (bank part — set trim_pct — and set new_take_profit_pct for the remainder),
RAISE (conviction has grown; hold it all and set a higher new_take_profit_pct),
or HOLD (rare: keep the current target as-is).

A new_take_profit_pct is measured as % gain from the original entry price, so it
must be greater than {tp_str} to mean anything. Justify it on the fundamentals or
momentum actually in evidence above — not on the fact that the price has risen.
Return the TargetReview JSON."""
    return system, user, tech_live


def _vote(reviews: list[TargetReview]) -> tuple[TargetReview, dict[str, int]]:
    """Majority-vote the recommendation; average confidence among agreeing reviews.

    The new target is the median of what the agreeing reviews proposed, so one
    outlier arguing for a wildly higher level cannot set it alone.
    """
    winner, agreeing, counts = majority(reviews)
    best = max(agreeing, key=lambda r: r.confidence)

    targets = sorted(r.new_take_profit_pct for r in agreeing if r.new_take_profit_pct is not None)
    new_tp = targets[len(targets) // 2] if targets else None

    consensus = TargetReview(
        ticker=best.ticker,
        recommendation=winner,
        confidence=round(sum(r.confidence for r in agreeing) / len(agreeing), 3),
        trim_pct=best.trim_pct,
        new_take_profit_pct=new_tp,
        what_changed=best.what_changed,
        rationale=best.rationale,
    )
    return consensus, counts


def review_position(
    ticker: str,
    store: Store,
    cfg: AppConfig,
    live_price: float | None = None,
    log_decision: bool = True,
) -> tuple[TargetReview, dict[str, int]] | None:
    """Run an N-sample consensus take-profit review for `ticker`.

    Returns (consensus TargetReview, vote breakdown), or None if the ticker
    isn't held. Raises LLMError if every sample fails.

    The verdict is logged as a decision so it is evaluated ~4 weeks later like
    any other call — SELL scored as a sale, RAISE and HOLD as holds, whose
    return is measured but not graded correct/incorrect.
    """
    ticker = ticker.upper()
    built = _build_review_prompt(ticker, store, cfg, live_price)
    if built is None:
        return None
    system, user, native_price = built
    consensus, votes = _vote(run_consensus(system, user, cfg, TargetReview, "target-review"))
    if log_decision:
        log_review(consensus, store, "target_review", native_price)
    return consensus, votes


def apply_new_target(review: TargetReview, store: Store) -> tuple[float, float] | None:
    """
    Persist a raised take-profit level. Returns (old_pct, new_pct) if it moved.

    Only ever raises: a review that comes back with a level at or below the
    current one leaves it alone, so a weak sample cannot quietly lower the bar
    a position is measured against. Nothing is written for SELL — the level is
    irrelevant once the position is meant to be closed.
    """
    if review.recommendation not in ("raise", "trim"):
        return None
    new_pct = review.new_take_profit_pct
    if new_pct is None:
        return None

    prior = store.get_effective_stops().get(review.ticker, {})
    old_pct = prior.get("take_profit_pct")
    if old_pct is not None and new_pct <= old_pct:
        return None

    store.set_position_stop(
        review.ticker,
        stop_pct=prior.get("stop_pct"),
        take_profit_pct=new_pct,
    )
    return (old_pct, new_pct) if old_pct is not None else (0.0, new_pct)


_REC_EMOJI = {"sell": "💰", "trim": "🟠", "raise": "🎯", "hold": "⏸"}


def _detail(r: TargetReview, moved: tuple[float, float] | None) -> str:
    """The part that says what actually changed: how much to sell, and the new level."""
    bits = []
    if r.recommendation == "trim" and r.trim_pct:
        bits.append(f"sell {r.trim_pct:.0f}%")
    if moved:
        old, new = moved
        bits.append(f"target {old:+.0f}% → {new:+.0f}%")
    elif r.new_take_profit_pct and r.recommendation in ("raise", "trim"):
        # Proposed but not applied — it did not clear the standing level.
        bits.append(f"target unchanged (proposed +{r.new_take_profit_pct:.0f}%)")
    return ("  " + ", ".join(bits)) if bits else ""


def format_review_text(
    r: TargetReview, votes: dict[str, int], n: int, moved: tuple[float, float] | None = None
) -> str:
    return (
        f"{_REC_EMOJI.get(r.recommendation,'')} TARGET REVIEW {r.ticker}: "
        f"{r.recommendation.upper()}{_detail(r, moved)}  "
        f"(conf {r.confidence:.2f}; consensus {votes_str(votes, n)})\n"
        f"  What changed: {r.what_changed}\n"
        f"  Rationale: {r.rationale}"
    )


def format_review_html(
    r: TargetReview, votes: dict[str, int], n: int, moved: tuple[float, float] | None = None
) -> str:
    return (
        f"{_REC_EMOJI.get(r.recommendation,'')} <b>Target review {html.escape(r.ticker)}: "
        f"{r.recommendation.upper()}{html.escape(_detail(r, moved))}</b>  "
        f"<i>(conf {r.confidence:.2f}; {votes_str(votes, n)})</i>\n"
        f"<i>Changed:</i> {html.escape(r.what_changed)}\n"
        f"<i>Why:</i> {html.escape(r.rationale)}"
    )
