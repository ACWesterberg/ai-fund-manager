"""
Guardrail unit tests — these are the safety-critical rules; they must be bulletproof.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from fundmgr.config import AppConfig, FeeConfig, RiskConfig
from fundmgr.data.prices import TickerFeatures
from fundmgr.engine.schema import Action, DecisionRun
from fundmgr.guardrails.rules import apply_guardrails
from fundmgr.state.models import PortfolioSnapshot, Position

UNIVERSE = {"VOLV-B.ST", "SAND.ST", "ERIC-B.ST", "ABB.ST", "HM-B.ST", "INVE-B.ST"}

def _cfg(**overrides) -> AppConfig:
    cfg = AppConfig()
    cfg.risk = RiskConfig(
        max_position_pct=overrides.get("max_position_pct", 18),
        max_positions=overrides.get("max_positions", 5),
        min_cash_pct=overrides.get("min_cash_pct", 12),
        max_cash_pct=overrides.get("max_cash_pct", 25),
        min_trade_sek=overrides.get("min_trade_sek", 2500),
        max_turnover_pct=overrides.get("max_turnover_pct", 25),
        stale_after_days=overrides.get("stale_after_days", 5),
    )
    cfg.fees = FeeConfig()
    return cfg


def _snap(cash: float = 40_000, positions: list[Position] | None = None) -> PortfolioSnapshot:
    pos = positions or []
    snap = PortfolioSnapshot(positions=pos, cash_sek=cash)
    for p in pos:
        p.current_price_sek = p.avg_cost_sek  # simplify: current = cost
    return snap


def _feat(ticker: str, price: float = 100.0, stale: bool = False) -> TickerFeatures:
    return TickerFeatures(
        ticker=ticker,
        name=ticker,
        last_price=price,
        last_date="2026-06-09",
        data_age_trading_days=10 if stale else 1,
    )


def _decision(actions: list[Action], cash_target: float = 15.0) -> DecisionRun:
    return DecisionRun(
        run_id="test-001",
        market_summary="Test run.",
        actions=actions,
        cash_target_pct=cash_target,
        notes="",
    )


def _buy(ticker: str, weight: float, sek: float, confidence: float = 0.8) -> Action:
    return Action(
        ticker=ticker, side="buy",
        target_weight_pct=weight, sek_estimate=sek,
        confidence=confidence, thesis="Test thesis.",
    )


def _sell(ticker: str, weight: float, sek: float, confidence: float = 0.8) -> Action:
    return Action(
        ticker=ticker, side="sell",
        target_weight_pct=weight, sek_estimate=sek,
        confidence=confidence, thesis="Test thesis.",
    )


def _hold(ticker: str) -> Action:
    return Action(
        ticker=ticker, side="hold",
        target_weight_pct=10.0, sek_estimate=0,
        confidence=0.5, thesis="No change.",
    )


# ── Universe check ────────────────────────────────────────────────────────────

def test_out_of_universe_ticker_rejected():
    cfg = _cfg()
    snap = _snap(cash=40_000)
    features = {"VOLV-B.ST": _feat("VOLV-B.ST", 300)}
    decision = _decision([_buy("TSLA", 10, 5_000)])  # not in universe
    result = apply_guardrails(decision, snap, features, UNIVERSE, cfg)
    assert len(result.approved_actions) == 0
    assert result.verdicts[0].rejection_reason is not None


# ── Stale data ────────────────────────────────────────────────────────────────

def test_stale_data_blocks_buy():
    cfg = _cfg(stale_after_days=5)
    snap = _snap(cash=40_000)
    features = {"VOLV-B.ST": _feat("VOLV-B.ST", stale=True)}  # 10 days old
    decision = _decision([_buy("VOLV-B.ST", 10, 5_000)])
    result = apply_guardrails(decision, snap, features, UNIVERSE, cfg)
    assert len(result.approved_actions) == 0
    assert "stale" in result.verdicts[0].rejection_reason.lower()


def test_stale_data_does_not_block_sell():
    cfg = _cfg(stale_after_days=5)
    pos = [Position("VOLV-B.ST", shares=20, avg_cost_sek=300)]
    snap = _snap(cash=40_000, positions=pos)
    snap.positions[0].current_price_sek = 300
    features = {"VOLV-B.ST": _feat("VOLV-B.ST", price=300, stale=True)}
    decision = _decision([_sell("VOLV-B.ST", 0, 3_000)])
    result = apply_guardrails(decision, snap, features, UNIVERSE, cfg)
    approved_sells = [a for a in result.approved_actions if a.side == "sell"]
    assert len(approved_sells) == 1  # sell goes through despite stale data


# ── Min trade size ────────────────────────────────────────────────────────────

def test_min_trade_size_blocks_small_buy():
    cfg = _cfg(min_trade_sek=2500)
    snap = _snap(cash=40_000)
    features = {"VOLV-B.ST": _feat("VOLV-B.ST", 300)}
    decision = _decision([_buy("VOLV-B.ST", 5, 1_000)])  # 1000 < 2500
    result = apply_guardrails(decision, snap, features, UNIVERSE, cfg)
    assert len(result.approved_actions) == 0
    assert "minimum" in result.verdicts[0].rejection_reason.lower()


def test_min_trade_size_passes_large_buy():
    cfg = _cfg(min_trade_sek=2500)
    snap = _snap(cash=40_000)
    features = {"VOLV-B.ST": _feat("VOLV-B.ST", 300)}
    decision = _decision([_buy("VOLV-B.ST", 10, 5_000)])
    result = apply_guardrails(decision, snap, features, UNIVERSE, cfg)
    assert len(result.approved_actions) == 1


# ── Max position weight ───────────────────────────────────────────────────────

def test_max_position_weight_clips():
    cfg = _cfg(max_position_pct=18)
    snap = _snap(cash=40_000)  # nav = 40_000, 18% = 7_200
    features = {"VOLV-B.ST": _feat("VOLV-B.ST", 300)}
    decision = _decision([_buy("VOLV-B.ST", 30, 12_000)])  # wants 30%, above limit
    result = apply_guardrails(decision, snap, features, UNIVERSE, cfg)
    assert len(result.approved_actions) == 1
    assert result.verdicts[0].clipped
    assert result.approved_actions[0].target_weight_pct == pytest.approx(18.0)


# ── Max positions count ───────────────────────────────────────────────────────

def test_max_positions_blocks_new_entry():
    cfg = _cfg(max_positions=3)
    # Already at max: 3 positions
    positions = [
        Position("SAND.ST", 10, 200),
        Position("ERIC-B.ST", 20, 80),
        Position("ABB.ST", 5, 400),
    ]
    snap = _snap(cash=20_000, positions=positions)
    features = {
        "HM-B.ST": _feat("HM-B.ST", 160),
        "INVE-B.ST": _feat("INVE-B.ST", 380),
    }
    decision = _decision([
        _buy("HM-B.ST", 10, 5_000),
        _buy("INVE-B.ST", 10, 5_000),
    ])
    result = apply_guardrails(decision, snap, features, UNIVERSE, cfg)
    approved_new = [a for a in result.approved_actions if a.side == "buy"]
    assert len(approved_new) == 0  # both new entries blocked


# ── Cash floor ────────────────────────────────────────────────────────────────

def test_min_cash_blocks_buy_that_breaches_floor():
    cfg = _cfg(min_cash_pct=12)
    snap = _snap(cash=5_000)  # nav ≈ 5_000, 12% floor = 600 SEK
    features = {"VOLV-B.ST": _feat("VOLV-B.ST", 300)}
    # Buying 4_500 would leave only 500 SEK cash (10%) — below 12% floor
    decision = _decision([_buy("VOLV-B.ST", 10, 4_500)])
    result = apply_guardrails(decision, snap, features, UNIVERSE, cfg)
    assert len(result.approved_actions) == 0
    assert "cash floor" in result.verdicts[0].rejection_reason.lower()


# ── Turnover cap ──────────────────────────────────────────────────────────────

def test_turnover_cap_drops_low_confidence_trades():
    cfg = _cfg(max_turnover_pct=25)
    snap = _snap(cash=40_000)  # 25% = 10_000 SEK cap
    features = {
        "VOLV-B.ST": _feat("VOLV-B.ST", 300),
        "SAND.ST": _feat("SAND.ST", 200),
        "ERIC-B.ST": _feat("ERIC-B.ST", 80),
    }
    decision = _decision([
        _buy("VOLV-B.ST", 10, 6_000, confidence=0.9),   # kept (higher confidence)
        _buy("SAND.ST",   10, 5_000, confidence=0.5),   # dropped (lower confidence, over cap)
        _buy("ERIC-B.ST", 10, 5_000, confidence=0.4),   # dropped (lowest confidence)
    ])
    result = apply_guardrails(decision, snap, features, UNIVERSE, cfg)
    buys = [a for a in result.approved_actions if a.side == "buy"]
    total = sum(a.sek_estimate for a in buys)
    assert total <= 10_000
    # Highest confidence buy should survive
    tickers = [a.ticker for a in buys]
    assert "VOLV-B.ST" in tickers


# ── Cash target clamping ──────────────────────────────────────────────────────

def test_cash_target_clamped_to_min():
    cfg = _cfg(min_cash_pct=12)
    snap = _snap(cash=40_000)
    decision = _decision([_hold("VOLV-B.ST")], cash_target=5.0)  # below min
    result = apply_guardrails(decision, snap, {}, UNIVERSE, cfg)
    assert result.cash_target_pct == pytest.approx(12.0)
    assert result.cash_clamped


def test_cash_target_clamped_to_max():
    cfg = _cfg(max_cash_pct=25)
    snap = _snap(cash=40_000)
    decision = _decision([_hold("VOLV-B.ST")], cash_target=60.0)  # above max
    result = apply_guardrails(decision, snap, {}, UNIVERSE, cfg)
    assert result.cash_target_pct == pytest.approx(25.0)
    assert result.cash_clamped


def test_cash_target_within_range_not_clamped():
    cfg = _cfg(min_cash_pct=12, max_cash_pct=25)
    snap = _snap(cash=40_000)
    decision = _decision([_hold("VOLV-B.ST")], cash_target=15.0)
    result = apply_guardrails(decision, snap, {}, UNIVERSE, cfg)
    assert result.cash_target_pct == pytest.approx(15.0)
    assert not result.cash_clamped


# ── Holds always pass ─────────────────────────────────────────────────────────

def test_hold_always_approved():
    cfg = _cfg()
    snap = _snap(cash=100)  # nearly empty cash — holds still go through
    decision = _decision([_hold("VOLV-B.ST")])
    result = apply_guardrails(decision, snap, {}, UNIVERSE, cfg)
    holds = [a for a in result.approved_actions if a.side == "hold"]
    assert len(holds) == 1
