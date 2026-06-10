from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from fundmgr.config import AppConfig
from fundmgr.data.prices import TickerFeatures
from fundmgr.engine.schema import Action, DecisionRun
from fundmgr.state.models import PortfolioSnapshot


@dataclass
class GuardrailVerdict:
    action: Action
    approved: bool
    clipped: bool = False
    rejection_reason: str = ""
    clip_note: str = ""

    @property
    def status(self) -> str:
        if not self.approved:
            return "REJECTED"
        if self.clipped:
            return "CLIPPED"
        return "APPROVED"


@dataclass
class GuardrailResult:
    verdicts: list[GuardrailVerdict] = field(default_factory=list)
    cash_target_pct: float = 0.0
    cash_clamped: bool = False
    approved_actions: list[Action] = field(default_factory=list)

    def to_log(self) -> list[dict]:
        return [
            {
                "ticker": v.action.ticker,
                "side": v.action.side,
                "status": v.status,
                "reason": v.rejection_reason or v.clip_note or "ok",
            }
            for v in self.verdicts
        ]


def apply_guardrails(
    decision: DecisionRun,
    snap: PortfolioSnapshot,
    features: dict[str, TickerFeatures],
    universe_tickers: set[str],
    cfg: AppConfig,
) -> GuardrailResult:
    """
    Run all risk checks against the LLM decision.
    Returns approved (and possibly clipped) actions plus a full audit log.
    """
    result = GuardrailResult()
    approved: list[Action] = []
    verdicts: list[GuardrailVerdict] = []

    nav = snap.nav_sek
    current_positions = {p.ticker for p in snap.positions if p.shares > 0}

    for action in decision.actions:
        verdict = _check_action(
            action, snap, features, universe_tickers, current_positions, cfg, nav
        )
        verdicts.append(verdict)
        if verdict.approved:
            approved.append(verdict.action)

    # Turnover cap: if aggregate trade value exceeds max, drop lowest-confidence trades
    approved = _apply_turnover_cap(approved, nav, cfg)

    # Cash target clamping
    cash_target = decision.cash_target_pct
    cash_clamped = False
    if cash_target < cfg.risk.min_cash_pct:
        cash_target = cfg.risk.min_cash_pct
        cash_clamped = True
    elif cash_target > cfg.risk.max_cash_pct:
        cash_target = cfg.risk.max_cash_pct
        cash_clamped = True

    result.verdicts = verdicts
    result.approved_actions = approved
    result.cash_target_pct = cash_target
    result.cash_clamped = cash_clamped
    return result


def _check_action(
    action: Action,
    snap: PortfolioSnapshot,
    features: dict[str, TickerFeatures],
    universe_tickers: set[str],
    current_positions: set[str],
    cfg: AppConfig,
    nav: float,
) -> GuardrailVerdict:
    v = GuardrailVerdict(action=action, approved=True)

    # 1. Universe check
    if action.ticker not in universe_tickers:
        v.approved = False
        v.rejection_reason = f"Ticker {action.ticker!r} not in universe"
        return v

    # Holds never require further checks
    if action.side == "hold":
        return v

    # 2. Stale data block (buys only)
    if action.side == "buy":
        feat = features.get(action.ticker)
        if feat is None:
            v.approved = False
            v.rejection_reason = "No price data available"
            return v
        if feat.data_age_trading_days > cfg.risk.stale_after_days:
            v.approved = False
            v.rejection_reason = f"Stale data ({feat.data_age_trading_days} trading days old)"
            return v

    # 3. Min trade size
    if action.sek_estimate < cfg.risk.min_trade_sek and action.side != "hold":
        v.approved = False
        v.rejection_reason = (
            f"Trade size {action.sek_estimate:.0f} SEK below minimum {cfg.risk.min_trade_sek:.0f} SEK"
        )
        return v

    # 4. Max position weight — clip the target weight
    if action.side == "buy" and action.target_weight_pct > cfg.risk.max_position_pct:
        clipped_weight = cfg.risk.max_position_pct
        clipped_sek = nav * clipped_weight / 100
        # Adjust sek_estimate proportionally
        action = Action(
            ticker=action.ticker,
            side=action.side,
            target_weight_pct=clipped_weight,
            sek_estimate=clipped_sek,
            confidence=action.confidence,
            thesis=action.thesis,
            stop_loss_pct=action.stop_loss_pct,
            take_profit_pct=action.take_profit_pct,
        )
        v.action = action
        v.clipped = True
        v.clip_note = (
            f"Weight clipped from {v.action.target_weight_pct:.1f}% to {clipped_weight:.1f}% (max_position_pct)"
        )

    # 5. New position count limit
    if action.side == "buy" and action.ticker not in current_positions:
        if len(current_positions) >= cfg.risk.max_positions:
            v.approved = False
            v.rejection_reason = f"Max positions ({cfg.risk.max_positions}) already reached"
            return v

    # 6. Cash floor check for buys
    if action.side == "buy":
        projected_cash = snap.cash_sek - action.sek_estimate
        projected_cash_pct = projected_cash / nav * 100 if nav > 0 else 0
        if projected_cash_pct < cfg.risk.min_cash_pct:
            v.approved = False
            v.rejection_reason = (
                f"Would breach min cash floor: projected cash {projected_cash_pct:.1f}% < {cfg.risk.min_cash_pct}%"
            )
            return v

    return v


def _apply_turnover_cap(
    actions: list[Action],
    nav: float,
    cfg: AppConfig,
) -> list[Action]:
    """Drop lowest-confidence non-hold trades until aggregate turnover is within cap."""
    max_turnover = nav * cfg.risk.max_turnover_pct / 100
    trades = [a for a in actions if a.side != "hold"]
    holds = [a for a in actions if a.side == "hold"]

    total = sum(a.sek_estimate for a in trades)
    if total <= max_turnover:
        return actions

    # Sort by confidence descending; drop lowest-confidence trades until within cap
    trades_sorted = sorted(trades, key=lambda a: a.confidence, reverse=True)
    kept: list[Action] = []
    running = 0.0
    for trade in trades_sorted:
        if running + trade.sek_estimate <= max_turnover:
            kept.append(trade)
            running += trade.sek_estimate

    return holds + kept


def shares_for_action(
    action: Action,
    snap: PortfolioSnapshot,
    features: dict[str, TickerFeatures],
) -> int | None:
    """
    Calculate the number of whole shares to trade for a buy/sell action.
    Returns None if price data is unavailable.
    """
    feat = features.get(action.ticker)
    if feat is None:
        return None

    price = feat.last_price
    if price <= 0:
        return None

    if action.side == "buy":
        shares = math.floor(action.sek_estimate / price)
    else:  # sell
        # Sell down to target weight
        target_value = snap.nav_sek * action.target_weight_pct / 100
        current_pos = next((p for p in snap.positions if p.ticker == action.ticker), None)
        if current_pos is None:
            return 0
        current_value = current_pos.shares * price
        sell_value = current_value - target_value
        if sell_value <= 0:
            return 0
        shares = math.floor(sell_value / price)

    return max(0, shares)
