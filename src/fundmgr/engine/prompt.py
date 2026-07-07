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


def _sector_weights_block(
    snap: PortfolioSnapshot,
    features: dict[str, TickerFeatures],
    max_sector_pct: float,
) -> str:
    """Compute current sector weights from held positions and feature sector tags."""
    if not snap.positions:
        return ""
    nav = snap.nav_sek
    if nav <= 0:
        return ""

    sector_values: dict[str, float] = {}
    for p in snap.positions:
        feat = features.get(p.ticker)
        sector = feat.sector if feat and feat.sector else "Unknown"
        sector_values[sector] = sector_values.get(sector, 0.0) + p.market_value_sek

    if not sector_values:
        return ""

    lines = [f"  Current sector exposure (cap {max_sector_pct:.0f}%):"]
    for sector, val in sorted(sector_values.items(), key=lambda x: x[1], reverse=True):
        pct = val / nav * 100
        headroom = max_sector_pct - pct
        flag = " ⚠ near cap" if headroom < 5 else ""
        lines.append(f"    {sector:<30} {pct:>5.1f}%  (headroom {headroom:.1f}%){flag}")
    return "\n".join(lines)


def _risk_limits_block(
    cfg: AppConfig,
    snap: PortfolioSnapshot,
    features: dict[str, TickerFeatures],
) -> str:
    sector_block = _sector_weights_block(snap, features, cfg.risk.max_sector_pct)
    lines = [
        "## Risk Limits (hard constraints)",
        f"  Max single-name weight: {cfg.risk.max_position_pct}%",
        f"  Max sector weight: {cfg.risk.max_sector_pct}% of NAV per GICS sector",
        f"  Max open positions: {cfg.risk.max_positions}",
        f"  Cash range: {cfg.risk.min_cash_pct}% – {cfg.risk.max_cash_pct}%",
        f"  Min trade size: {cfg.risk.min_trade_sek:,.0f} SEK",
        f"  Max turnover this run: {cfg.risk.max_turnover_pct}% of NAV "
        f"(≈{snap.nav_sek * cfg.risk.max_turnover_pct / 100:,.0f} SEK)",
        f"  Fee: {cfg.fees.rate*100:.2f}% per trade "
        f"(min {cfg.fees.min_sek:.0f} SEK, max {cfg.fees.max_sek:.0f} SEK)",
        "  FX cost: non-SEK stocks incur a 0.10% currency conversion spread on top of brokerage.",
        "  Prefer SEK-denominated stocks when conviction is equal. Only buy foreign stocks for clear alpha.",
    ]
    if sector_block:
        lines.append(sector_block)
    lines.append("Guardrails enforce these mechanically — size your recommendations within them.")
    return "\n".join(lines)


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
    # Penalise earnings imminent (binary event risk)
    if f.days_to_earnings is not None:
        if 0 <= f.days_to_earnings <= 2:
            score -= 4.0
        elif 0 <= f.days_to_earnings <= 5:
            score -= 1.5
    # Volume confirmation
    if f.rel_volume is not None and f.rel_volume > 2.5:
        score += 1.5
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
) -> tuple[str, str, dict[str, str]]:
    """
    Returns (system_message, user_message, fields).

    The system message is the mandate; the user message is the full context.
    `fields` is the same context broken out into the shared, retrieval-ready
    keys {mandate, macro, portfolio_state, risk_limits, universe, learnings}
    so the training corpus is KNN/optimizer-ready at write time rather than
    reconstructed from the flattened string later.

    Note: `risk_limits` is the block *as rendered* — it embeds live sector
    exposure derived from current positions, not just static config caps.
    """
    mandate = _load_mandate(cfg.mandate_path)

    # Optimized decision guidance (from `fund optimize`, once compiled)
    from fundmgr.engine.optimizer import load_guidance
    guidance = load_guidance(cfg)
    guidance_section = (
        "\n\n---\n## Decision Guidance (optimized from your realized outcomes)\n" + guidance
        if guidance else ""
    )

    # Add structured output instruction to mandate
    system = (
        mandate
        + guidance_section
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

    # Discrete blocks — captured once, used both for the flat prompt and the
    # fielded snapshot so the two can never drift.
    portfolio_state = _portfolio_block(snap, bench_return)
    risk_limits     = _risk_limits_block(cfg, snap, features)
    learnings_block = _learnings_block(learnings)
    universe        = _features_block(features, current_tickers)

    fields = {
        "mandate":         mandate,
        "macro":           macro_block,
        "portfolio_state": portfolio_state,
        "risk_limits":     risk_limits,
        "universe":        universe,
        "learnings":       learnings_block,
    }

    sections = [
        f"# Weekly Decision Run\nRun ID: {run_id}\nDate: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n",
    ]

    if macro_block:
        sections.append(macro_block)
        sections.append("")

    sections += [
        portfolio_state,
        "",
        risk_limits,
        "",
    ]

    if learnings:
        sections.append(learnings_block)
        sections.append("")

    sections.append(universe)

    sections.append(
        f"## Your Task\n"
        f"Review the above and return a DecisionRun JSON with your buy/sell/hold decisions. "
        f"Run ID must be: {run_id}"
    )

    user = "\n".join(sections)
    return system, user, fields


SNAPSHOT_VERSION = 2  # v1 = flat system/user only; v2 = + fields + regime


def snapshot_to_dict(
    snap: PortfolioSnapshot,
    system: str,
    user: str,
    fields: dict[str, str] | None = None,
    cfg: "AppConfig | None" = None,
) -> str:
    """Serialise full prompt context for the recommendation log.

    v2 augments the flat system/user strings (kept for exact replay/audit) with
    the fielded context and the run regime. The regime carries a nullable
    `session` key so a single loader can serve both this repo and swing-trader.
    """
    out: dict = {
        "snapshot_version": SNAPSHOT_VERSION,
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
        # Flat — exact replay / audit
        "system_message": system,
        "user_message": user,
        # Fielded — retrieval / optimizer ready
        "fields": fields,
    }
    if cfg is not None:
        from fundmgr.engine.optimizer import guidance_fingerprint
        out["regime"] = {
            "capital_sek":  cfg.capital_sek,
            "provider":     cfg.llm.provider,
            "model_id":     cfg.llm.model_id,
            "n_samples":    cfg.llm.n_samples,
            "session":      None,  # nullable shared key (swing-trader sets us/european)
            "config_hash":  cfg.config_hash(),
            "guidance_hash": guidance_fingerprint(cfg),  # None when no MIPRO guidance active
        }
    return json.dumps(out)
