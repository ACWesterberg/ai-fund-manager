"""
Guardrail unit tests — these are the safety-critical rules; they must be bulletproof.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from fundmgr import regions, styles
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


def test_clip_note_records_the_weight_that_was_asked_for():
    """The note is the audit trail for a clip, so it has to name the weight the
    model actually requested — not repeat the capped one on both sides."""
    cfg = _cfg(max_position_pct=18)
    snap = _snap(cash=40_000)
    features = {"VOLV-B.ST": _feat("VOLV-B.ST", 300)}
    decision = _decision([_buy("VOLV-B.ST", 30, 12_000)])
    result = apply_guardrails(decision, snap, features, UNIVERSE, cfg)

    note = result.verdicts[0].clip_note
    assert note == "Weight clipped from 30.0% to 18.0% (max_position_pct)"


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


# ── Regional allocation ceiling ───────────────────────────────────────────────
#
# Only the ceiling is enforceable. Nothing here can make a book buy Nordics; it
# can only stop it buying past the band. Tests that appear to check a floor are
# checking that the floor is *not* enforced.

def _regional_cfg(targets: dict, tolerance: float = 10.0, **overrides) -> AppConfig:
    cfg = _cfg(**overrides)
    cfg.risk.region_targets = targets
    cfg.risk.region_tolerance_pct = tolerance
    return cfg


def _geo_feat(ticker: str, country: str, price: float = 100.0) -> TickerFeatures:
    feat = _feat(ticker, price=price)
    feat.country = country
    return feat


def _geo_features() -> dict[str, TickerFeatures]:
    return {
        "VOLV-B.ST": _geo_feat("VOLV-B.ST", "SE"),
        "SAND.ST":   _geo_feat("SAND.ST", "SE"),
        "ERIC-B.ST": _geo_feat("ERIC-B.ST", "SE"),
        "ABB.ST":    _geo_feat("ABB.ST", "US"),
        "HM-B.ST":   _geo_feat("HM-B.ST", "DE"),
    }


def test_buy_within_the_regional_band_is_approved():
    cfg = _regional_cfg({"nordics": 30.0}, min_cash_pct=0)
    snap = _snap(cash=100_000)
    decision = _decision([_buy("VOLV-B.ST", 25.0, 25_000)])
    result = apply_guardrails(decision, snap, _geo_features(), UNIVERSE, cfg)
    assert result.verdicts[0].approved


def test_buy_past_the_regional_ceiling_is_rejected():
    cfg = _regional_cfg({"nordics": 30.0}, min_cash_pct=0, max_position_pct=50)
    snap = _snap(cash=100_000)
    decision = _decision([_buy("VOLV-B.ST", 45.0, 45_000)])
    result = apply_guardrails(decision, snap, _geo_features(), UNIVERSE, cfg)
    v = result.verdicts[0]
    assert not v.approved
    assert "Nordics" in v.rejection_reason
    assert "cap 40%" in v.rejection_reason


def test_the_ceiling_counts_what_the_region_already_holds():
    """Two buys that each fit still cannot both fit — the second sees the first."""
    cfg = _regional_cfg({"nordics": 30.0}, min_cash_pct=0)
    snap = _snap(cash=65_000, positions=[Position("VOLV-B.ST", 350, 100.0)])
    decision = _decision([_buy("SAND.ST", 10.0, 10_000)])
    result = apply_guardrails(decision, snap, _geo_features(), UNIVERSE, cfg)
    assert not result.verdicts[0].approved
    assert "Nordics" in result.verdicts[0].rejection_reason


def test_a_region_the_mix_never_named_is_unconstrained():
    """Asking for 30% Nordics says nothing about where the other 70% goes."""
    cfg = _regional_cfg({"nordics": 30.0}, min_cash_pct=0)
    snap = _snap(cash=100_000)
    decision = _decision([_buy("HM-B.ST", 16.0, 16_000)])   # Germany — not named
    result = apply_guardrails(decision, snap, _geo_features(), UNIVERSE, cfg)
    assert result.verdicts[0].approved


def test_a_zero_target_excludes_the_region_outright():
    """"No North America" must not permit a tolerance band's worth of it."""
    cfg = _regional_cfg({"north_america": 0.0}, min_cash_pct=0)
    snap = _snap(cash=100_000)
    decision = _decision([_buy("ABB.ST", 5.0, 5_000)])
    result = apply_guardrails(decision, snap, _geo_features(), UNIVERSE, cfg)
    v = result.verdicts[0]
    assert not v.approved
    assert "set to 0%" in v.rejection_reason


def test_a_sell_out_of_a_capped_region_is_never_blocked():
    """Selling is how an over-weight region gets back inside its band."""
    cfg = _regional_cfg({"nordics": 10.0}, min_cash_pct=0)
    snap = _snap(cash=20_000, positions=[Position("VOLV-B.ST", 800, 100.0)])
    decision = _decision([_sell("VOLV-B.ST", 5.0, 40_000)])
    result = apply_guardrails(decision, snap, _geo_features(), UNIVERSE, cfg)
    assert result.verdicts[0].approved


def test_a_region_left_short_is_not_forced_anywhere():
    """The floor is advisory: a run that buys nothing Nordic still passes."""
    cfg = _regional_cfg({"nordics": 30.0}, min_cash_pct=0)
    snap = _snap(cash=100_000)
    decision = _decision([_buy("HM-B.ST", 15.0, 15_000)])
    result = apply_guardrails(decision, snap, _geo_features(), UNIVERSE, cfg)
    assert result.verdicts[0].approved
    assert not [v for v in result.verdicts if not v.approved]


def test_no_mix_means_no_regional_check():
    cfg = _cfg(min_cash_pct=0)
    snap = _snap(cash=100_000)
    decision = _decision([_buy("VOLV-B.ST", 15.0, 15_000)])
    result = apply_guardrails(decision, snap, _geo_features(), UNIVERSE, cfg)
    assert result.verdicts[0].approved


def test_a_name_with_no_country_is_measured_as_other():
    cfg = _regional_cfg({regions.OTHER_CODE: 0.0}, min_cash_pct=0)
    snap = _snap(cash=100_000)
    features = _geo_features() | {"INVE-B.ST": _feat("INVE-B.ST")}  # no country set
    decision = _decision([_buy("INVE-B.ST", 5.0, 5_000)])
    result = apply_guardrails(decision, snap, features, UNIVERSE, cfg)
    assert not result.verdicts[0].approved


def test_the_region_is_measured_on_the_clipped_trade_not_the_requested_one():
    """A 40% request clipped to 12% is a 12% trade — and a 15% ceiling clears it."""
    cfg = _regional_cfg({"nordics": 10.0}, tolerance=5.0, max_position_pct=12, min_cash_pct=0)
    snap = _snap(cash=100_000)
    decision = _decision([_buy("VOLV-B.ST", 40.0, 40_000)])
    result = apply_guardrails(decision, snap, _geo_features(), UNIVERSE, cfg)
    v = result.verdicts[0]
    assert v.clipped and v.approved
    assert v.action.target_weight_pct == 12.0


def test_a_clipped_trade_still_breaching_its_region_is_rejected():
    cfg = _regional_cfg({"nordics": 10.0}, tolerance=0.0, max_position_pct=12, min_cash_pct=0)
    snap = _snap(cash=100_000)
    decision = _decision([_buy("VOLV-B.ST", 40.0, 40_000)])
    result = apply_guardrails(decision, snap, _geo_features(), UNIVERSE, cfg)
    v = result.verdicts[0]
    assert not v.approved
    assert "Nordics would reach 12.0%" in v.rejection_reason


# ── Style allocation ceiling ──────────────────────────────────────────────────
#
# Same machinery as the regional cap over a judgement instead of a fact, so the
# rule these tests defend is that a name is only ever capped on a positive
# finding. A missing fundamental is not a finding.

def _style_cfg(targets: dict, tolerance: float = 10.0, **overrides) -> AppConfig:
    cfg = _cfg(**overrides)
    cfg.risk.style_targets = targets
    cfg.risk.style_tolerance_pct = tolerance
    return cfg


def _compounder(ticker: str) -> TickerFeatures:
    feat = _feat(ticker)
    feat.roe_pct, feat.profit_margin_pct = 28.0, 22.0
    feat.debt_to_equity, feat.revenue_growth_pct = 60.0, 9.0
    return feat


def _lossmaker(ticker: str) -> TickerFeatures:
    feat = _feat(ticker)
    feat.profit_margin_pct = -30.0
    return feat


def _style_features() -> dict[str, TickerFeatures]:
    return {
        "VOLV-B.ST": _compounder("VOLV-B.ST"),
        "SAND.ST": _compounder("SAND.ST"),
        "ERIC-B.ST": _lossmaker("ERIC-B.ST"),
        "ABB.ST": _lossmaker("ABB.ST"),
        "HM-B.ST": _feat("HM-B.ST"),            # no figures on file at all
    }


def test_buy_within_the_style_band_is_approved():
    cfg = _style_cfg({"speculative": 20.0}, min_cash_pct=0)
    snap = _snap(cash=100_000)
    decision = _decision([_buy("ERIC-B.ST", 15.0, 15_000)])
    result = apply_guardrails(decision, snap, _style_features(), UNIVERSE, cfg)
    assert result.verdicts[0].approved


def test_buy_past_the_style_ceiling_is_rejected():
    cfg = _style_cfg({"speculative": 20.0}, min_cash_pct=0, max_position_pct=50)
    snap = _snap(cash=100_000)
    decision = _decision([_buy("ERIC-B.ST", 40.0, 40_000)])
    result = apply_guardrails(decision, snap, _style_features(), UNIVERSE, cfg)
    v = result.verdicts[0]
    assert not v.approved
    assert "Higher-risk / speculative" in v.rejection_reason
    assert "cap 30%" in v.rejection_reason


def test_the_style_ceiling_counts_what_the_book_already_holds():
    cfg = _style_cfg({"speculative": 12.0}, tolerance=0.0, min_cash_pct=0)
    snap = _snap(cash=90_000, positions=[Position("ERIC-B.ST", 100, 100.0)])
    decision = _decision([_buy("ABB.ST", 10.0, 10_000)])
    result = apply_guardrails(decision, snap, _style_features(), UNIVERSE, cfg)
    assert not result.verdicts[0].approved
    assert "would reach 20.0%" in result.verdicts[0].rejection_reason


def test_a_style_the_mix_never_named_is_unconstrained():
    cfg = _style_cfg({"speculative": 20.0}, min_cash_pct=0)
    snap = _snap(cash=100_000)
    decision = _decision([_buy("VOLV-B.ST", 16.0, 16_000)])   # quality — not named
    result = apply_guardrails(decision, snap, _style_features(), UNIVERSE, cfg)
    assert result.verdicts[0].approved


def test_a_name_with_no_figures_is_never_blocked_by_a_style_cap():
    """Rejecting a buy because the fundamentals cache had not filled would be
    failing closed on a gap rather than on a finding."""
    cfg = _style_cfg({"speculative": 0.0}, min_cash_pct=0)
    snap = _snap(cash=100_000)
    decision = _decision([_buy("HM-B.ST", 16.0, 16_000)])
    result = apply_guardrails(decision, snap, _style_features(), UNIVERSE, cfg)
    assert result.verdicts[0].approved


def test_an_explicit_zero_on_unclassified_does_block_it():
    """"Don't buy what you can't see" is a real instruction, and the only way an
    unclassified name is ever refused."""
    cfg = _style_cfg({styles.UNCLASSIFIED: 0.0}, min_cash_pct=0)
    snap = _snap(cash=100_000)
    decision = _decision([_buy("HM-B.ST", 16.0, 16_000)])
    result = apply_guardrails(decision, snap, _style_features(), UNIVERSE, cfg)
    assert not result.verdicts[0].approved
    assert "set to 0%" in result.verdicts[0].rejection_reason


def test_an_excluded_style_refuses_the_buy_outright():
    cfg = _style_cfg({"speculative": 0.0}, min_cash_pct=0)
    snap = _snap(cash=100_000)
    decision = _decision([_buy("ERIC-B.ST", 5.0, 5_000)])
    result = apply_guardrails(decision, snap, _style_features(), UNIVERSE, cfg)
    assert not result.verdicts[0].approved


def test_no_style_mix_means_no_style_check():
    cfg = _cfg(min_cash_pct=0)
    snap = _snap(cash=100_000)
    decision = _decision([_buy("ERIC-B.ST", 15.0, 15_000)])
    result = apply_guardrails(decision, snap, _style_features(), UNIVERSE, cfg)
    assert result.verdicts[0].approved


def test_a_buy_must_clear_both_dials():
    """Region and style are checked independently; either can refuse."""
    cfg = _cfg(min_cash_pct=0, max_position_pct=50)
    cfg.risk.region_targets = {"nordics": 30.0}
    cfg.risk.region_tolerance_pct = 10.0
    cfg.risk.style_targets = {"speculative": 10.0}
    cfg.risk.style_tolerance_pct = 0.0
    features = _style_features()
    for feat in features.values():
        feat.country = "SE"
    snap = _snap(cash=100_000)

    # Inside the regional band, past the style one.
    result = apply_guardrails(
        _decision([_buy("ERIC-B.ST", 40.0, 40_000)]), snap, features, UNIVERSE, cfg)
    assert "Style cap breach" in result.verdicts[0].rejection_reason

    # Inside the style band (quality is unnamed), past the regional one.
    result = apply_guardrails(
        _decision([_buy("VOLV-B.ST", 50.0, 50_000)]), snap, features, UNIVERSE, cfg)
    assert "Regional cap breach" in result.verdicts[0].rejection_reason
