from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fundmgr.config import AppConfig, UniverseTicker
from fundmgr.data.benchmark import get_benchmark_return_pct
from fundmgr.data.prices import TickerFeatures
from fundmgr.state.models import Learning, PortfolioSnapshot
from fundmgr.state.store import Store


def _load_mandate(path: Path) -> str:
    return path.read_text().strip()


def _portfolio_block(snap: PortfolioSnapshot, benchmark_return: float | None) -> str:
    lines = ["## Current Portfolio State"]
    lines.append(f"NAV: {snap.nav_sek:,.0f} SEK  |  Cash: {snap.cash_sek:,.0f} SEK ({snap.cash_pct:.1f}%)")

    bench_str = f"{benchmark_return:+.1f}%" if benchmark_return is not None else "n/a"
    lines.append(f"Benchmark return since inception: {bench_str}")
    lines.append("")

    if snap.positions:
        lines.append("Open positions:")
        for p in sorted(snap.positions, key=lambda x: x.market_value_sek, reverse=True):
            w = snap.weight_pct(p.ticker)
            pnl = p.unrealised_pnl_pct
            lines.append(
                f"  {p.ticker:<16} {p.shares:>8.0f} shares  "
                f"avg {p.avg_cost_sek:>8.2f}  "
                f"now {p.current_price_sek:>8.2f}  "
                f"({pnl:+.1f}%)  weight {w:.1f}%"
            )
    else:
        lines.append("No open positions — fully in cash.")

    return "\n".join(lines)


def _risk_limits_block(cfg: AppConfig, snap: PortfolioSnapshot) -> str:
    return (
        "## Risk Limits (hard constraints)\n"
        f"  Max single-name weight: {cfg.risk.max_position_pct}%\n"
        f"  Max open positions: {cfg.risk.max_positions}\n"
        f"  Cash range: {cfg.risk.min_cash_pct}% – {cfg.risk.max_cash_pct}%\n"
        f"  Min trade size: {cfg.risk.min_trade_sek:,.0f} SEK\n"
        f"  Max turnover this run: {cfg.risk.max_turnover_pct}% of NAV "
        f"(≈{snap.nav_sek * cfg.risk.max_turnover_pct / 100:,.0f} SEK)\n"
        f"  Fee: {cfg.fees.rate*100:.2f}% per trade "
        f"(min {cfg.fees.min_sek:.0f} SEK, max {cfg.fees.max_sek:.0f} SEK)\n"
        "  FX cost: non-SEK stocks (DK/NO/FI) incur a 0.10% currency conversion spread (Montrose rate) on top of brokerage.\n"
        "  Prefer SEK-denominated stocks when conviction is equal. Only buy foreign stocks for clear alpha.\n"
        "Guardrails enforce these mechanically — size your recommendations within them."
    )


def _learnings_block(learnings: list[Learning]) -> str:
    if not learnings:
        return ""
    lines = ["## Past Performance Reflections", "These are lessons distilled from your prior decisions. Factor them in."]
    for l in learnings[:8]:  # cap at 8 to keep prompt tight
        lines.append(f"  [{l.category.upper()}] {l.body}")
    return "\n".join(lines)


_CANDIDATE_LIMIT = 50  # non-held tickers shown to LLM per run


def _signal_score(f: TickerFeatures) -> float:
    """
    Simple composite signal to surface the most actionable candidates.
    Higher = more worth the LLM's attention.
    """
    score = 0.0
    # Momentum
    r20 = f.return_20d_pct or 0.0
    r5  = f.return_5d_pct  or 0.0
    score += min(r20 / 5, 4.0)   # capped contribution from 20d return
    score += min(r5  / 2, 2.0)   # recent momentum bonus
    # Sentiment signal
    if f.sentiment_label == "positive":
        score += 3.0
    elif f.sentiment_label == "negative":
        score += 1.5  # still worth seeing — potential sell or avoid signal
    # RSI extremes (oversold = potential entry, overbought = potential exit)
    if f.rsi_14 is not None:
        if f.rsi_14 < 35:
            score += 2.5   # oversold — contrarian opportunity
        elif f.rsi_14 > 70:
            score += 1.0   # extended — watch for trim / avoid entry
    # Trend alignment
    if f.above_ma50 and f.above_ma200:
        score += 1.0
    # Penalise stale data
    if f.is_stale:
        score -= 3.0
    return score


def _features_block(
    features: dict[str, TickerFeatures],
    current_tickers: set[str],
) -> str:
    held = {t: f for t, f in features.items() if t in current_tickers}
    rest = {t: f for t, f in features.items() if t not in current_tickers}

    # Rank non-held candidates by signal, take top N
    top_candidates = sorted(rest.values(), key=_signal_score, reverse=True)[:_CANDIDATE_LIMIT]

    total_universe = len(features)
    shown = len(held) + len(top_candidates)

    lines = ["## Universe — Ticker Feature Blocks"]
    lines.append(
        f"(Showing {shown} of {total_universe} tickers: all {len(held)} held ★ "
        f"+ top {len(top_candidates)} candidates by signal score. "
        f"Remaining {total_universe - shown} had no notable signal this run.)\n"
    )

    for f in held.values():
        lines.append("★ " + f.to_prompt_block())
        lines.append("")

    for f in top_candidates:
        lines.append("  " + f.to_prompt_block())
        lines.append("")

    return "\n".join(lines)


def build_prompt(
    cfg: AppConfig,
    snap: PortfolioSnapshot,
    features: dict[str, TickerFeatures],
    store: Store,
    run_id: str,
    macro_block: str = "",
) -> tuple[str, str]:
    """
    Returns (system_message, user_message).
    The system message is the mandate; the user message is the full context.
    """
    mandate = _load_mandate(cfg.mandate_path)

    # Add structured output instruction to mandate
    system = (
        mandate
        + "\n\n---\n"
        + "Return ONLY a valid JSON object matching the DecisionRun schema. "
        + "No markdown, no explanation outside the JSON."
    )

    # Benchmark return since first NAV entry
    nav_history = store.get_nav_history()
    bench_return = None
    if nav_history:
        first_date = nav_history[0].date
        bench_return = get_benchmark_return_pct(store, since_date=first_date)

    current_tickers = {p.ticker for p in snap.positions}
    learnings = store.get_active_learnings()

    sections = [
        f"# Weekly Decision Run\nRun ID: {run_id}\nDate: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n",
    ]

    if macro_block:
        sections.append(macro_block)
        sections.append("")

    sections += [
        _portfolio_block(snap, bench_return),
        "",
        _risk_limits_block(cfg, snap),
        "",
    ]

    if learnings:
        sections.append(_learnings_block(learnings))
        sections.append("")

    sections.append(_features_block(features, current_tickers))

    sections.append(
        f"## Your Task\n"
        f"Review the above and return a DecisionRun JSON with your buy/sell/hold decisions. "
        f"Run ID must be: {run_id}"
    )

    user = "\n".join(sections)
    return system, user


def snapshot_to_dict(snap: PortfolioSnapshot, system: str, user: str) -> str:
    """Serialise full prompt context for the recommendation log."""
    return json.dumps({
        "nav_sek": snap.nav_sek,
        "cash_sek": snap.cash_sek,
        "positions": [
            {
                "ticker": p.ticker,
                "shares": p.shares,
                "avg_cost_sek": p.avg_cost_sek,
                "current_price_sek": p.current_price_sek,
            }
            for p in snap.positions
        ],
        "system_message": system,
        "user_message": user,
    })
