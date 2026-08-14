"""
Shared plumbing for single-position reviews.

A stop breach and a take-profit hit ask opposite questions but need the same
evidence — current technicals, the recent thesis on the name, recent news — and
the same N-sample consensus machinery. Keeping that in one place is what stops
the two reviews from drifting into disagreeing views of the same position.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

from fundmgr.config import AppConfig
from fundmgr.engine.client import LLMError, call_llm
from fundmgr.state.store import Store


def technicals_block(ticker: str) -> tuple[str, float | None]:
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


def context_blocks(ticker: str, store: Store) -> tuple[str, str]:
    """(recent decisions, recent news) blocks for a position under review."""
    decisions = store.get_decisions_for_ticker(ticker, limit=3)
    if decisions:
        dlines = []
        for d in decisions:
            ts = (d.get("timestamp") or "")[:10]
            conf = d.get("confidence")
            conf_s = f", conf {conf:.2f}" if conf is not None else ""
            # Mark which decisions came from an intra-week review rather than a
            # weekly run, so a reviewer can tell a whole-portfolio call from a
            # single-name one made under different information.
            src = d.get("source")
            tag = " (review)" if src and src != "run" else ""
            dlines.append(
                f"  {ts}: {str(d.get('action','')).upper()}{tag}{conf_s} — "
                f"\"{(d.get('thesis') or '').strip()}\""
            )
        decisions_block = "\n".join(dlines)
    else:
        decisions_block = "  (no logged decisions on this name)"

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

    return decisions_block, news_block


def live_price_sek(
    ticker: str,
    pos,
    cfg: AppConfig,
    store: Store,
    currency: str,
    live_price: float | None,
    tech_live: float | None,
) -> float:
    """Resolve the position's live price in SEK, to compare against SEK cost basis."""
    if live_price is not None:
        return live_price                      # already SEK (caller converted)
    live = tech_live or pos.avg_cost_sek
    if cfg.fx_to_sek and tech_live is not None and currency != "SEK":
        from fundmgr.data.fx import rate_to_sek
        live = tech_live * (rate_to_sek(currency, store) or 1.0)  # native→SEK
    return live


def run_consensus(system: str, user: str, cfg: AppConfig, schema, label: str) -> list:
    """Fire N samples in parallel and return every parsed review that came back."""
    n = max(1, cfg.llm.n_samples)
    parsed_all: list = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(call_llm, system, user, cfg, schema) for _ in range(n)]
        for fut in as_completed(futures):
            try:
                parsed, _raw = fut.result()
                parsed_all.append(parsed)
            except LLMError as e:
                errors.append(str(e))

    if not parsed_all:
        raise LLMError(f"All {n} {label} call(s) failed: {'; '.join(errors)}")
    return parsed_all


def majority(reviews: list) -> tuple[str, list, dict[str, int]]:
    """(winning recommendation, the reviews backing it, full vote counts)."""
    counts = Counter(r.recommendation for r in reviews)
    winner, _ = counts.most_common(1)[0]
    agreeing = [r for r in reviews if r.recommendation == winner]
    return winner, agreeing, dict(counts)


# Review verbs → the buy/sell/hold vocabulary decision_outcomes is scored in.
# A partial sale is scored as a sale: the question the evaluator answers for a
# sell — "did the stock then underperform?" — is the right one for a trim too.
# Everything that keeps the position is a hold, which the evaluator measures the
# return of but deliberately does not grade correct/incorrect.
_ACTION_FOR_VERDICT = {
    # take-profit review
    "sell": "sell", "trim": "sell", "raise": "hold",
    # stop review
    "exit": "sell", "add": "buy",
    "hold": "hold",
}


def action_for_verdict(verdict: str) -> str:
    """The decision_outcomes action a review verdict is recorded as."""
    return _ACTION_FOR_VERDICT.get(verdict, "hold")


def log_review(review, store: Store, source: str, native_price: float | None) -> str | None:
    """Record a review verdict as an evaluable decision. Returns the run_id."""
    if native_price is None:
        return None      # the evaluator skips priceless outcomes; don't write one
    return store.log_review_decision(
        ticker=review.ticker,
        action=action_for_verdict(review.recommendation),
        confidence=review.confidence,
        thesis=f"[{source}: {review.recommendation.upper()}] {review.rationale}",
        price_at_decision=native_price,
        source=source,
    )


def votes_str(votes: dict[str, int], n: int) -> str:
    parts = [f"{rec}×{cnt}" for rec, cnt in sorted(votes.items(), key=lambda kv: -kv[1])]
    return f"{'/'.join(parts)} of {n}"
