"""
Stop-loss review — focused, advisory reassessment of a single position after its
stop level is breached.

Triggered automatically by `check-stops` when a stop hits on a non-auto-fill
(real-money) fund, and runnable on demand via `fund review-stop TICKER`.

It does NOT execute anything: it gathers the position, the most recent decision
on the name (so a recent "add more" thesis is weighed), current technicals and
news, asks the LLM with N-sample consensus, and returns an EXIT / TRIM / HOLD /
ADD recommendation for the human to act on manually.
"""
from __future__ import annotations

import html

from fundmgr.config import AppConfig, load_universe
from fundmgr.engine.review_common import (
    context_blocks,
    live_price_sek,
    majority,
    run_consensus,
    technicals_block,
    votes_str,
)
from fundmgr.engine.schema import StopReview
from fundmgr.state.store import Store


def find_stop_breaches(store: Store, cfg: AppConfig | None = None) -> dict:
    """Scan holdings for stop-loss breaches.

    Returns {"breaches": [...], "skipped": [...]} where breaches are dicts
    {ticker, chg, stop_pct, live} and skipped are {ticker, reason} for holdings
    that couldn't be evaluated (no stop on record, or price unavailable) — so the
    caller can surface them rather than silently report "no breaches".

    Stops fall back to the level decided at buy time (get_effective_stops), so
    positions whose stop was never persisted are still checked. Live prices are
    converted native→SEK (when cfg.fx_to_sek) to match the SEK cost basis.
    """
    from fundmgr.data.fx import rate_to_sek
    from fundmgr.data.quotes import live_price

    fx_on = bool(cfg and cfg.fx_to_sek)
    cur_by_ticker = {}
    if fx_on:
        cur_by_ticker = {t.yahoo_ticker: t.currency for t in load_universe(cfg.universe_path)}
    fx_cache: dict[str, float] = {}

    stop_map = store.get_effective_stops()
    breaches: list[dict] = []
    skipped: list[dict] = []
    for p in store.get_positions():
        stop_pct = stop_map.get(p.ticker, {}).get("stop_pct")
        if not stop_pct or not p.avg_cost_sek:
            skipped.append({"ticker": p.ticker, "reason": "no stop on record"})
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
        if chg <= -stop_pct:
            breaches.append({"ticker": p.ticker, "chg": chg, "stop_pct": stop_pct, "live": live})
    return {"breaches": breaches, "skipped": skipped}


def _build_review_prompt(ticker: str, store: Store, cfg: AppConfig, live_price: float | None) -> tuple[str, str] | None:
    """Assemble (system, user) for the review, or None if the ticker isn't held."""
    pos = next((p for p in store.get_positions() if p.ticker == ticker), None)
    if pos is None:
        return None

    uni = load_universe(cfg.universe_path)
    name = {t.yahoo_ticker: t.name for t in uni}.get(ticker, ticker)
    currency = {t.yahoo_ticker: t.currency for t in uni}.get(ticker, "SEK")
    stop = store.get_effective_stops().get(ticker, {})
    stop_pct = stop.get("stop_pct")

    technicals, tech_live = technicals_block(ticker)
    live = live_price_sek(ticker, pos, cfg, store, currency, live_price, tech_live)
    chg = (live / pos.avg_cost_sek - 1) * 100 if pos.avg_cost_sek else 0.0
    mkt_value = pos.shares * live

    # The thesis the stop may be contradicting, plus recent news.
    decisions_block, news_block = context_blocks(ticker, store)

    system = (
        "You are the risk manager for a real-money equity fund. A stop-loss level "
        "has just been breached on a held position. Reassess, with discipline, "
        "whether to act. Be decisive but weigh the most recent thesis on the name "
        "against what has actually changed. This is advisory — a human executes."
    )

    stop_str = f"-{stop_pct:.0f}%" if stop_pct else "n/a"
    user = f"""# Stop-Loss Review — {ticker} ({name})

## Position
Shares: {pos.shares:.0f}  |  Avg cost: {pos.avg_cost_sek:.2f}  |  Live: {live:.2f}
Change since entry: {chg:+.1f}%   (stop level {stop_str} — now breached)
Market value: {mkt_value:,.0f} SEK

## Most recent decisions on this name
{decisions_block}

## Current technicals
{technicals}

## Recent news / sentiment (14d)
{news_block}

## Your task
The stop has been hit. Choose EXIT (sell all), TRIM (reduce), HOLD (the move was
noise, conviction intact), or ADD (conviction unchanged or improved). If your most
recent decision here was to ADD, address that tension head-on in `what_changed`.
Return the StopReview JSON."""
    return system, user


def _vote(reviews: list[StopReview]) -> tuple[StopReview, dict[str, int]]:
    """Majority-vote the recommendation; average confidence among agreeing reviews."""
    winner, agreeing, counts = majority(reviews)
    best = max(agreeing, key=lambda r: r.confidence)
    consensus = StopReview(
        ticker=best.ticker,
        recommendation=winner,
        confidence=round(sum(r.confidence for r in agreeing) / len(agreeing), 3),
        trim_pct=best.trim_pct,
        what_changed=best.what_changed,
        rationale=best.rationale,
    )
    return consensus, counts


def review_position(
    ticker: str, store: Store, cfg: AppConfig, live_price: float | None = None
) -> tuple[StopReview, dict[str, int]] | None:
    """Run an N-sample consensus stop-loss review for `ticker`.

    Returns (consensus StopReview, vote breakdown), or None if the ticker isn't
    held. Raises LLMError if every sample fails.
    """
    ticker = ticker.upper()
    built = _build_review_prompt(ticker, store, cfg, live_price)
    if built is None:
        return None
    system, user = built

    return _vote(run_consensus(system, user, cfg, StopReview, "stop-review"))


_REC_EMOJI = {"exit": "🔴", "trim": "🟠", "hold": "⏸", "add": "🟢"}


def format_review_text(r: StopReview, votes: dict[str, int], n: int) -> str:
    trim = f" {r.trim_pct:.0f}%" if r.recommendation == "trim" and r.trim_pct else ""
    return (
        f"{_REC_EMOJI.get(r.recommendation,'')} STOP REVIEW {r.ticker}: "
        f"{r.recommendation.upper()}{trim}  (conf {r.confidence:.2f}; consensus {votes_str(votes, n)})\n"
        f"  What changed: {r.what_changed}\n"
        f"  Rationale: {r.rationale}"
    )


def format_review_html(r: StopReview, votes: dict[str, int], n: int) -> str:
    trim = f" {r.trim_pct:.0f}%" if r.recommendation == "trim" and r.trim_pct else ""
    return (
        f"{_REC_EMOJI.get(r.recommendation,'')} <b>Stop review {html.escape(r.ticker)}: "
        f"{r.recommendation.upper()}{trim}</b>  <i>(conf {r.confidence:.2f}; {votes_str(votes, n)})</i>\n"
        f"<i>Changed:</i> {html.escape(r.what_changed)}\n"
        f"<i>Why:</i> {html.escape(r.rationale)}"
    )
