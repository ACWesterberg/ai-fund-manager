from __future__ import annotations

from datetime import datetime

from fundmgr.config import AppConfig
from fundmgr.data.prices import TickerFeatures
from fundmgr.engine.schema import Action, DecisionRun
from fundmgr.guardrails.rules import GuardrailResult, shares_for_action
from fundmgr.state.models import PortfolioSnapshot


def format_action_list(
    decision: DecisionRun,
    guardrail: GuardrailResult,
    snap: PortfolioSnapshot,
    features: dict[str, TickerFeatures],
    cfg: AppConfig,
    vote_counts: dict[str, int] | None = None,
    n_samples: int = 1,
) -> str:
    nav = snap.nav_sek
    lines: list[str] = []

    consensus_mode = vote_counts is not None and n_samples > 1
    consensus_tag = f"  ({n_samples}-run consensus)" if consensus_mode else ""

    lines.append("═" * 60)
    lines.append(f"  Weekly Action List — {datetime.utcnow().strftime('%Y-%m-%d')}{consensus_tag}")
    lines.append(f"  Run ID: {decision.run_id}")
    lines.append("═" * 60)
    lines.append("")

    # Market summary
    lines.append("  Market summary (GPT-5.6-sol):")
    for sentence in decision.market_summary.split(". "):
        if sentence.strip():
            lines.append(f"  \"{sentence.strip()}.\"")
    lines.append("")

    # Approved trades
    buys  = [a for a in guardrail.approved_actions if a.side == "buy"]
    sells = [a for a in guardrail.approved_actions if a.side == "sell"]
    holds = [a for a in guardrail.approved_actions if a.side == "hold"]

    total_turnover = 0.0

    def _vote_badge(ticker: str) -> str:
        if not consensus_mode:
            return ""
        v = vote_counts.get(ticker, 0)  # type: ignore[union-attr]
        return f"  [{v}/{n_samples}]"

    if buys:
        lines.append("  ── BUYS " + "─" * 50)
        for a in sorted(buys, key=lambda x: x.sek_estimate, reverse=True):
            n_shares = shares_for_action(a, snap, features)
            feat = features.get(a.ticker)
            price_str = f"@ ~{feat.last_price:.2f}" if feat else ""
            shares_str = f"{n_shares} shares" if n_shares is not None else "? shares"
            fee = cfg.fees.calc(a.sek_estimate)
            total_turnover += a.sek_estimate
            conf_bar = "█" * round(a.confidence * 5)
            lines.append(
                f"  BUY  {a.ticker:<16} {shares_str:<12} {price_str:<14}"
                f"≈ {a.sek_estimate:>7,.0f} SEK  fee {fee:.2f}  [{conf_bar:<5}] {a.confidence:.2f}{_vote_badge(a.ticker)}"
            )
            lines.append(f"       → {a.thesis}")
            if a.stop_loss_pct:
                lines.append(f"       Stop-loss: -{a.stop_loss_pct:.1f}%")
        lines.append("")

    if sells:
        lines.append("  ── SELLS " + "─" * 49)
        for a in sorted(sells, key=lambda x: x.sek_estimate, reverse=True):
            n_shares = shares_for_action(a, snap, features)
            feat = features.get(a.ticker)
            price_str = f"@ ~{feat.last_price:.2f}" if feat else ""
            shares_str = f"{n_shares} shares" if n_shares is not None else "? shares"
            fee = cfg.fees.calc(a.sek_estimate)
            total_turnover += a.sek_estimate
            conf_bar = "█" * round(a.confidence * 5)
            lines.append(
                f"  SELL {a.ticker:<16} {shares_str:<12} {price_str:<14}"
                f"≈ {a.sek_estimate:>7,.0f} SEK  fee {fee:.2f}  [{conf_bar:<5}] {a.confidence:.2f}{_vote_badge(a.ticker)}"
            )
            lines.append(f"       → {a.thesis}")
        lines.append("")

    if holds:
        lines.append("  ── HOLDS " + "─" * 49)
        for a in sorted(holds, key=lambda x: x.ticker):
            feat = features.get(a.ticker)
            price_str = f"{feat.last_price:.2f} SEK" if feat else "n/a"
            lines.append(f"  HOLD {a.ticker:<16}  {price_str:<12}  {a.thesis[:70]}{_vote_badge(a.ticker)}")
        lines.append("")

    # Vetoed actions
    vetoed = [v for v in guardrail.verdicts if not v.approved]
    if vetoed:
        lines.append("  ── VETOED BY GUARDRAILS " + "─" * 33)
        for v in vetoed:
            lines.append(f"  ✗ {v.action.ticker:<16} ({v.action.side.upper():<4}) — {v.rejection_reason}")
        lines.append("")

    clipped = [v for v in guardrail.verdicts if v.clipped]
    if clipped:
        lines.append("  ── CLIPPED " + "─" * 46)
        for v in clipped:
            lines.append(f"  ↓ {v.action.ticker:<16}  {v.clip_note}")
        lines.append("")

    # Summary footer
    total_fee = sum(cfg.fees.calc(a.sek_estimate) for a in guardrail.approved_actions if a.side != "hold")
    lines.append("─" * 60)
    lines.append(f"  NAV:               {nav:>12,.0f} SEK")
    lines.append(f"  Cash target:       {guardrail.cash_target_pct:>11.1f}%"
                 + ("  ← clamped by guardrail" if guardrail.cash_clamped else ""))
    lines.append(f"  Estimated turnover:{total_turnover:>12,.0f} SEK ({total_turnover/nav*100:.1f}% of NAV)")
    lines.append(f"  Estimated fees:    {total_fee:>12.2f} SEK")
    lines.append("")
    lines.append("  Next step: execute trades at broker, then run:")
    lines.append("    fund fill <TICKER> <SHARES> <PRICE> <FEE>")
    lines.append("═" * 60)

    if decision.notes:
        lines.append(f"\n  GPT notes: {decision.notes}")

    return "\n".join(lines)
