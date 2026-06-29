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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

from fundmgr.config import AppConfig, load_universe
from fundmgr.engine.client import LLMError, call_llm
from fundmgr.engine.schema import StopReview
from fundmgr.state.store import Store


def find_stop_breaches(store: Store) -> dict:
    """Scan holdings for stop-loss breaches.

    Returns {"breaches": [...], "skipped": [...]} where breaches are dicts
    {ticker, chg, stop_pct, live} and skipped are {ticker, reason} for holdings
    that couldn't be evaluated (no stop on record, or price unavailable) — so the
    caller can surface them rather than silently report "no breaches".

    Stops fall back to the level decided at buy time (get_effective_stops), so
    positions whose stop was never persisted are still checked.
    """
    import yfinance as yf

    stop_map = store.get_effective_stops()
    breaches: list[dict] = []
    skipped: list[dict] = []
    for p in store.get_positions():
        stop_pct = stop_map.get(p.ticker, {}).get("stop_pct")
        if not stop_pct or not p.avg_cost_sek:
            skipped.append({"ticker": p.ticker, "reason": "no stop on record"})
            continue
        try:
            hist = yf.Ticker(p.ticker).history(period="2d")
            live = float(hist["Close"].iloc[-1]) if hist is not None and not hist.empty else None
        except Exception:
            live = None
        if live is None:
            skipped.append({"ticker": p.ticker, "reason": "price unavailable"})
            continue
        chg = (live / p.avg_cost_sek - 1) * 100
        if chg <= -stop_pct:
            breaches.append({"ticker": p.ticker, "chg": chg, "stop_pct": stop_pct, "live": live})
    return {"breaches": breaches, "skipped": skipped}


def _technicals_block(ticker: str) -> tuple[str, float | None]:
    """Compact technicals summary from yfinance history. Returns (text, live_price)."""
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="6mo")
    except Exception:
        return "  (technicals unavailable)", None
    if hist is None or hist.empty:
        return "  (technicals unavailable)", None

    close = hist["Close"]
    live = float(close.iloc[-1])

    def ret(days: int) -> float | None:
        return (live / float(close.iloc[-days]) - 1) * 100 if len(close) > days else None

    ma50 = float(close.tail(50).mean()) if len(close) >= 50 else None
    ma200 = float(close.tail(200).mean()) if len(close) >= 200 else None

    # RSI(14)
    rsi = None
    if len(close) >= 15:
        delta = close.diff()
        gain = delta.clip(lower=0).tail(14).mean()
        loss = (-delta.clip(upper=0)).tail(14).mean()
        if loss and loss > 0:
            rsi = 100 - 100 / (1 + gain / loss)
        elif gain:
            rsi = 100.0

    hi = float(close.tail(21).max()) if len(close) >= 1 else None
    lo = float(close.tail(21).min()) if len(close) >= 1 else None

    lines = [f"  Live: {live:.2f}"]
    r5, r20 = ret(5), ret(20)
    if r5 is not None:  lines.append(f"  5d return:  {r5:+.1f}%")
    if r20 is not None: lines.append(f"  20d return: {r20:+.1f}%")
    if ma50 is not None:
        lines.append(f"  vs 50d MA:  {(live/ma50-1)*100:+.1f}% ({'above' if live>=ma50 else 'below'})")
    if ma200 is not None:
        lines.append(f"  vs 200d MA: {(live/ma200-1)*100:+.1f}% ({'above' if live>=ma200 else 'below'})")
    if rsi is not None:  lines.append(f"  RSI(14): {rsi:.0f}")
    if hi and lo:        lines.append(f"  1-month range: {lo:.2f} – {hi:.2f}")
    return "\n".join(lines), live


def _build_review_prompt(ticker: str, store: Store, cfg: AppConfig, live_price: float | None) -> tuple[str, str] | None:
    """Assemble (system, user) for the review, or None if the ticker isn't held."""
    pos = next((p for p in store.get_positions() if p.ticker == ticker), None)
    if pos is None:
        return None

    names = {t.yahoo_ticker: t.name for t in load_universe(cfg.universe_path)}
    name = names.get(ticker, ticker)
    stop = store.get_effective_stops().get(ticker, {})
    stop_pct = stop.get("stop_pct")

    technicals, tech_live = _technicals_block(ticker)
    live = live_price or tech_live or pos.avg_cost_sek
    chg = (live / pos.avg_cost_sek - 1) * 100 if pos.avg_cost_sek else 0.0
    mkt_value = pos.shares * live

    # Recent decisions on this name — the thesis the stop may be contradicting.
    decisions = store.get_decisions_for_ticker(ticker, limit=3)
    if decisions:
        dlines = []
        for d in decisions:
            ts = (d.get("timestamp") or "")[:10]
            conf = d.get("confidence")
            conf_s = f", conf {conf:.2f}" if conf is not None else ""
            dlines.append(f"  {ts}: {str(d.get('action','')).upper()}{conf_s} — \"{(d.get('thesis') or '').strip()}\"")
        decisions_block = "\n".join(dlines)
    else:
        decisions_block = "  (no logged decisions on this name)"

    # Recent cached news / sentiment (last 14 days), if any.
    since = (datetime.now(timezone.utc) - timedelta(days=14)).strftime("%Y-%m-%d")
    news = store.get_recent_news(ticker, since)
    if news:
        nlines = [
            f"  [{(n.get('sentiment_label') or '?')[:3]}] {(n.get('headline') or '')[:140]}"
            for n in news[:6]
        ]
        news_block = "\n".join(nlines)
    else:
        news_block = "  (no recent cached news)"

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
    counts = Counter(r.recommendation for r in reviews)
    winner, _ = counts.most_common(1)[0]
    agreeing = [r for r in reviews if r.recommendation == winner]
    best = max(agreeing, key=lambda r: r.confidence)
    consensus = StopReview(
        ticker=best.ticker,
        recommendation=winner,
        confidence=round(sum(r.confidence for r in agreeing) / len(agreeing), 3),
        trim_pct=best.trim_pct,
        what_changed=best.what_changed,
        rationale=best.rationale,
    )
    return consensus, dict(counts)


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

    n = max(1, cfg.llm.n_samples)
    reviews: list[StopReview] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(call_llm, system, user, cfg, StopReview) for _ in range(n)]
        for fut in as_completed(futures):
            try:
                parsed, _raw = fut.result()
                reviews.append(parsed)
            except LLMError as e:
                errors.append(str(e))

    if not reviews:
        raise LLMError(f"All {n} stop-review call(s) failed: {'; '.join(errors)}")
    return _vote(reviews)


_REC_EMOJI = {"exit": "🔴", "trim": "🟠", "hold": "⏸", "add": "🟢"}


def _votes_str(votes: dict[str, int], n: int) -> str:
    parts = [f"{rec}×{cnt}" for rec, cnt in sorted(votes.items(), key=lambda kv: -kv[1])]
    return f"{'/'.join(parts)} of {n}"


def format_review_text(r: StopReview, votes: dict[str, int], n: int) -> str:
    trim = f" {r.trim_pct:.0f}%" if r.recommendation == "trim" and r.trim_pct else ""
    return (
        f"{_REC_EMOJI.get(r.recommendation,'')} STOP REVIEW {r.ticker}: "
        f"{r.recommendation.upper()}{trim}  (conf {r.confidence:.2f}; consensus {_votes_str(votes, n)})\n"
        f"  What changed: {r.what_changed}\n"
        f"  Rationale: {r.rationale}"
    )


def format_review_html(r: StopReview, votes: dict[str, int], n: int) -> str:
    trim = f" {r.trim_pct:.0f}%" if r.recommendation == "trim" and r.trim_pct else ""
    return (
        f"{_REC_EMOJI.get(r.recommendation,'')} <b>Stop review {html.escape(r.ticker)}: "
        f"{r.recommendation.upper()}{trim}</b>  <i>(conf {r.confidence:.2f}; {_votes_str(votes, n)})</i>\n"
        f"<i>Changed:</i> {html.escape(r.what_changed)}\n"
        f"<i>Why:</i> {html.escape(r.rationale)}"
    )
