from __future__ import annotations

import math
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field

from fundmgr import regions, styles
from fundmgr.config import AppConfig
from fundmgr.data.prices import TickerFeatures
from fundmgr.engine.schema import Action, DecisionRun
from fundmgr.state.models import PortfolioSnapshot, Position


# The allocation dials a buy is measured against. Ceilings only, in both cases:
# a guardrail can refuse a trade but cannot invent one, so the floor half of a
# mix is a brief to the model and a number reported back (see fundmgr.allocation).
_MIX_DIALS = (
    (regions, "region_targets", "region_tolerance_pct"),
    (styles, "style_targets", "style_tolerance_pct"),
)


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
    projected = deepcopy(snap)
    mix_ceilings = _mix_ceilings(cfg)
    counts = Counter(a.ticker for a in decision.actions)
    turnover = 0.0

    # Reserve cash, exposure and turnover as each trade is accepted. There is
    # no later pruning that could remove a sell another trade depended upon.
    for action in sorted(decision.actions, key=lambda a: a.confidence, reverse=True):
        if counts[action.ticker] > 1:
            verdicts.append(GuardrailVerdict(action, False, rejection_reason="Duplicate ticker"))
            continue
        current_positions = {p.ticker for p in projected.positions if p.shares > 0}
        verdict = _check_action(
            action, projected, features, universe_tickers, current_positions, cfg, nav,
            mix_ceilings,
        )
        verdicts.append(verdict)
        if not verdict.approved:
            continue
        action = verdict.action
        if action.side != "hold":
            if turnover + action.sek_estimate > nav * cfg.risk.max_turnover_pct / 100 + 1e-9:
                verdict.approved = False
                verdict.rejection_reason = "Aggregate turnover cap exceeded"
                continue
            turnover += action.sek_estimate
            pos = next((p for p in projected.positions if p.ticker == action.ticker), None)
            current = pos.market_value_sek if pos else 0.0
            delta = action.sek_estimate * (1 if action.side == "buy" else -1)
            if pos is None:
                pos = Position(action.ticker, 0, 1, 1)
                projected.positions.append(pos)
            # Synthetic shares keep the projected snapshot in book currency.
            pos.current_price_sek = 1.0
            pos.shares = max(0.0, current + delta)
            projected.cash_sek -= delta + cfg.fees.calc(action.sek_estimate)
        approved.append(action)

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


def _mix_ceilings(cfg: AppConfig) -> list[tuple]:
    """(dial, targets, ceilings) for every mix this run was given, or [].

    Computed once per run: every buy is measured against the same set, so a
    classification cannot shift underneath a decision mid-pass.
    """
    out = []
    for dial, targets_attr, tolerance_attr in _MIX_DIALS:
        targets = getattr(cfg.risk, targets_attr, None) or {}
        if targets:
            tolerance = getattr(cfg.risk, tolerance_attr, dial.DEFAULT_TOLERANCE_PCT)
            out.append((dial, targets, dial.ceilings(targets, tolerance)))
    return out


def _check_action(
    action: Action,
    snap: PortfolioSnapshot,
    features: dict[str, TickerFeatures],
    universe_tickers: set[str],
    current_positions: set[str],
    cfg: AppConfig,
    nav: float,
    mix_ceilings: list[tuple] | None = None,
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
        if feat is None or not math.isfinite(feat.last_price) or feat.last_price <= 0:
            v.approved = False
            v.rejection_reason = "No price data available"
            return v
        if feat.data_age_trading_days > cfg.risk.stale_after_days:
            v.approved = False
            v.rejection_reason = f"Stale data ({feat.data_age_trading_days} trading days old)"
            return v

    if not math.isfinite(nav) or nav <= 0:
        return GuardrailVerdict(action, False, rejection_reason="Invalid portfolio NAV")
    if any(p.shares > 0 and (not math.isfinite(p.current_price_sek) or p.current_price_sek <= 0)
           for p in snap.positions):
        return GuardrailVerdict(action, False, rejection_reason="Missing portfolio valuation")

    requested_weight = action.target_weight_pct
    target_weight = (min(requested_weight, cfg.risk.max_position_pct)
                     if action.side == "buy" else requested_weight)
    current_value = sum(p.market_value_sek for p in snap.positions if p.ticker == action.ticker)
    target_value = nav * target_weight / 100
    trade_value = (max(0.0, target_value - current_value) if action.side == "buy"
                   else max(0.0, current_value - target_value))
    action = action.model_copy(update={"target_weight_pct": target_weight, "sek_estimate": trade_value})
    v.action = action
    if target_weight != requested_weight:
        v.clipped = True
        v.clip_note = f"Weight clipped from {requested_weight:.1f}% to {target_weight:.1f}% (max_position_pct)"

    if trade_value <= 0:
        return GuardrailVerdict(action, False, rejection_reason="Already at target or position not held")
    if trade_value < cfg.risk.min_trade_sek:
        v.approved = False
        v.rejection_reason = f"Trade size {trade_value:.0f} SEK below minimum {cfg.risk.min_trade_sek:.0f} SEK"
        return v

    # 5. Sector concentration cap
    if action.side == "buy":
        feat = features.get(action.ticker)
        if feat and feat.sector:
            sector = feat.sector
            sector_value_now = sum(
                p.market_value_sek
                for p in snap.positions
                if p.shares > 0
                and features.get(p.ticker) is not None
                and features[p.ticker].sector == sector
            )
            projected_sector_pct = (sector_value_now + action.sek_estimate) / nav * 100 if nav > 0 else 0
            if projected_sector_pct > cfg.risk.max_sector_pct:
                v.approved = False
                v.rejection_reason = (
                    f"Sector cap breach: {sector} would reach {projected_sector_pct:.1f}% "
                    f"(max {cfg.risk.max_sector_pct:.0f}%)"
                )
                return v

    # 6. Allocation mixes — regional and style ceilings
    #
    # Same shape as the sector cap and for the same reason — a mix is only a mix
    # if something enforces it — but only the ceiling is enforceable. Nothing
    # here can make the book buy Nordics or buy quality; it can only stop it
    # buying past the band. A bucket the mix never named has no entry and is not
    # checked, which is also what keeps an unclassifiable name unblocked.
    if action.side == "buy" and mix_ceilings:
        feat = features.get(action.ticker)
        for dial, targets, tops in mix_ceilings:
            code = dial.bucket_of(feat)
            ceiling = tops.get(code)
            if ceiling is None:
                continue
            bucket_value_now = sum(
                p.market_value_sek
                for p in snap.positions
                if p.shares > 0 and dial.bucket_of(features.get(p.ticker)) == code
            )
            projected_pct = (
                (bucket_value_now + action.sek_estimate) / nav * 100 if nav > 0 else 0
            )
            if projected_pct > ceiling + 1e-9:
                label = dial.label_of(code)
                v.approved = False
                v.rejection_reason = (
                    f"{dial.SCHEME.name} bucket excluded by the allocation mix: "
                    f"{label} is set to 0%"
                    if ceiling <= 0 else
                    f"{dial.SCHEME.name} cap breach: {label} would reach "
                    f"{projected_pct:.1f}% (target {targets.get(code, 0.0):.0f}%, "
                    f"cap {ceiling:.0f}%)"
                )
                return v

    # 7. New position count limit
    if action.side == "buy" and action.ticker not in current_positions:
        if len(current_positions) >= cfg.risk.max_positions:
            v.approved = False
            v.rejection_reason = f"Max positions ({cfg.risk.max_positions}) already reached"
            return v

    # 8. Cash floor check for buys
    if action.side == "buy":
        projected_cash = snap.cash_sek - action.sek_estimate - cfg.fees.calc(action.sek_estimate)
        projected_cash_pct = projected_cash / nav * 100 if nav > 0 else 0
        if projected_cash_pct < cfg.risk.min_cash_pct:
            v.approved = False
            v.rejection_reason = (
                f"Would breach min cash floor: projected cash {projected_cash_pct:.1f}% < {cfg.risk.min_cash_pct}%"
            )
            return v

    return v


def shares_for_action(
    action: Action,
    snap: PortfolioSnapshot,
    features: dict[str, TickerFeatures],
    cfg: AppConfig | None = None,
) -> int | None:
    """
    Calculate the number of whole shares to trade for a buy/sell action.
    Returns None if price data is unavailable.
    """
    feat = features.get(action.ticker)
    if feat is None:
        return None

    price = feat.last_price
    if cfg is not None and cfg.fx_to_sek and feat.currency != "SEK":
        from fundmgr.data.fx import rate_to_sek
        rate = rate_to_sek(feat.currency)
        if rate is None or not math.isfinite(rate) or rate <= 0:
            return None
        price *= rate
    if not math.isfinite(price) or price <= 0:
        return None
    if action.side == "hold":
        return 0
    shares = math.floor(action.sek_estimate / price)
    if action.side == "sell":
        position = next((p for p in snap.positions if p.ticker == action.ticker), None)
        shares = min(shares, math.floor(position.shares)) if position else 0
    return max(0, shares)
