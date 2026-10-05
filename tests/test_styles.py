"""
Style allocation — the classifier, and the rule that keeps it honest.

A region is a fact on the universe row; a style is a reading of fundamentals
that may not have landed. So the property these tests defend above all others is
that a name is bucketed on a **positive finding, never on an absence**: missing
figures mean `unclassified`, which is reported, never targeted by accident and
never blocked. The shared mix arithmetic is covered in test_regions.py — this
file is about the judgement and the seams around it.
"""
from __future__ import annotations

import pytest

from fundmgr import allocation, styles
from fundmgr.data.prices import TickerFeatures
from fundmgr.data.screener import screen
from fundmgr.engine.schema import Action
from fundmgr.state.models import PortfolioSnapshot, Position


def _feat(ticker: str = "X", **kw) -> TickerFeatures:
    return TickerFeatures(
        ticker=ticker, name=ticker, last_price=100.0, last_date="2026-09-16",
        data_age_trading_days=1, **kw,
    )


def _compounder(ticker: str = "Q", **kw) -> TickerFeatures:
    base = dict(roe_pct=28, profit_margin_pct=22, debt_to_equity=60, revenue_growth_pct=9)
    base.update(kw)
    return _feat(ticker, **base)


# ── Classification ────────────────────────────────────────────────────────────

def test_a_compounder_is_quality_and_says_why():
    code, why = styles.classify(_compounder())
    assert code == "quality"
    assert "ROE 28.0%" in why and "net margin 22.0%" in why


def test_the_quality_screen_needs_every_signal_it_can_see_to_pass():
    """One failing leg is enough — this is a screen, not a score."""
    assert styles.style_of(_compounder(roe_pct=4)) != "quality"
    assert styles.style_of(_compounder(profit_margin_pct=2)) != "quality"
    assert styles.style_of(_compounder(debt_to_equity=200)) != "quality"
    assert styles.style_of(_compounder(revenue_growth_pct=-5)) != "quality"


def test_two_good_ratios_are_not_enough_evidence_for_quality():
    """A lucky pair of ratios is not a Buffett screen; it is missing data."""
    assert styles.style_of(_feat(roe_pct=30, profit_margin_pct=25)) == styles.UNCLASSIFIED


def test_three_of_four_signals_can_carry_a_quality_verdict():
    assert styles.style_of(_feat(roe_pct=30, profit_margin_pct=25, debt_to_equity=40)) == "quality"


@pytest.mark.parametrize("kw,expected_in_reason", [
    ({"profit_margin_pct": -30}, "loss-making"),
    ({"roe_pct": -5}, "negative ROE"),
    ({"debt_to_equity": 400}, "D/E 400"),
])
def test_a_fundamental_risk_flag_is_speculative_on_its_own(kw, expected_in_reason):
    code, why = styles.classify(_feat(**kw))
    assert code == "speculative"
    assert expected_in_reason in why


def test_a_loss_making_hyper_grower_is_speculative_not_growth():
    """For a risk-appetite dial that distinction is the entire point."""
    assert styles.style_of(_feat(profit_margin_pct=-40, revenue_growth_pct=120)) == "speculative"


def test_extreme_volatility_alone_marks_a_name_speculative():
    code, why = styles.classify(_feat(vol_20d_ann_pct=95))
    assert code == "speculative"
    assert "vol 95%" in why


def test_a_deep_drawdown_alone_marks_a_name_speculative():
    assert styles.style_of(_feat(pct_from_52w_high=-70)) == "speculative"


def test_volatility_never_overrides_a_quality_pass():
    """A compounder in a violent quarter is still a compounder."""
    assert styles.style_of(_compounder(vol_20d_ann_pct=95)) == "quality"


def test_a_fundamental_flag_does_override_a_quality_pass():
    assert styles.style_of(_compounder(debt_to_equity=400)) == "speculative"


def test_a_profitable_fast_grower_that_misses_the_screen_is_growth():
    code, why = styles.classify(
        _feat(roe_pct=8, profit_margin_pct=6, debt_to_equity=50, revenue_growth_pct=30))
    assert code == "growth"
    assert "revenue +30.0%" in why


def test_moderate_volatility_is_not_a_verdict():
    assert styles.style_of(_feat(vol_20d_ann_pct=45)) == styles.UNCLASSIFIED


def test_a_name_with_no_figures_is_unclassified_and_gives_no_reason():
    assert styles.classify(_feat()) == (styles.UNCLASSIFIED, "")


def test_a_missing_feature_is_unclassified_rather_than_a_breach():
    """The guardrails call this with whatever the features dict had, or None."""
    assert styles.bucket_of(None) == styles.UNCLASSIFIED


def test_styles_of_and_reasons_of_agree_on_every_name():
    features = {"Q": _compounder("Q"), "S": _feat("S", profit_margin_pct=-10), "U": _feat("U")}
    codes, reasons = styles.styles_of(features), styles.reasons_of(features)
    assert codes == {"Q": "quality", "S": "speculative", "U": styles.UNCLASSIFIED}
    assert reasons["U"] == "" and reasons["S"]


# ── The mix, over a judgement ─────────────────────────────────────────────────

def test_clean_targets_accepts_the_style_codes():
    assert styles.clean_targets({"quality": 40, "speculative": 20}) == {
        "quality": 40.0, "speculative": 20.0}


def test_a_region_code_is_not_a_style_code():
    assert styles.clean_targets({"nordics": 30}) == {}


def test_an_impossible_style_mix_names_the_dial():
    with pytest.raises(ValueError, match="Style targets add up"):
        styles.clean_targets({"quality": 60, "growth": 60})


def test_an_excluded_style_still_drops_out_of_the_candidate_list():
    assert styles.excluded({"speculative": 0.0}) == {"speculative"}


def test_unclassified_is_never_dropped_from_the_candidate_list():
    """A cache miss must not hollow out the screen — the cap still bites in the
    guardrails, where the book rather than the candidate list is measured."""
    assert styles.excluded({styles.UNCLASSIFIED: 0.0}) == set()
    assert styles.ceilings({styles.UNCLASSIFIED: 0.0}, 10) == {styles.UNCLASSIFIED: 0.0}


def test_the_prompt_block_says_the_tag_can_be_argued_with():
    block = styles.prompt_block({"quality": 40.0}, 10, {}, {"quality": 20})
    assert "Buffett-style quality" in block and "30–50%" in block
    assert "not from judgement" in block
    assert "guardrails reject" in block and "nothing can force a buy" in block


def test_the_prompt_block_owns_up_to_what_it_could_not_classify():
    block = styles.prompt_block(
        {"quality": 40.0}, 10, {}, {"quality": 20, styles.UNCLASSIFIED: 33})
    assert "33 candidate(s) below are tagged Unclassified" in block
    assert "under-count" in block


def test_the_prompt_block_is_empty_without_a_mix():
    assert styles.prompt_block({}, 10, {}, {}) == ""


def test_projection_buckets_the_book_a_run_would_leave():
    snap = PortfolioSnapshot(positions=[], cash_sek=100_000)
    actions = [
        Action(ticker="Q", side="buy", target_weight_pct=30, sek_estimate=30_000,
               confidence=0.8, thesis="t"),
        Action(ticker="S", side="buy", target_weight_pct=15, sek_estimate=15_000,
               confidence=0.6, thesis="t"),
    ]
    got = styles.projected_exposure(
        snap, actions, {"Q": "quality", "S": "speculative"})
    assert got == {"quality": 30.0, "speculative": 15.0}


def test_exposure_places_an_unclassifiable_holding_in_unclassified():
    p = Position(ticker="X", shares=1, avg_cost_sek=50_000)
    p.current_price_sek = 50_000
    snap = PortfolioSnapshot(positions=[p], cash_sek=50_000)
    assert round(styles.exposure(snap, {})[styles.UNCLASSIFIED]) == 50


# ── Screener quotas ───────────────────────────────────────────────────────────

def _mixed_features() -> dict[str, TickerFeatures]:
    """Ten strong speculative names and three dull compounders — the shape that
    makes a quality target unbuildable when momentum alone decides."""
    feats = {
        f"SPEC{i}": _feat(f"SPEC{i}", profit_margin_pct=-20, return_20d_pct=50 - i)
        for i in range(10)
    }
    feats.update({
        f"QUAL{i}": _compounder(f"QUAL{i}", return_20d_pct=-20 - i) for i in range(3)
    })
    return feats


def test_without_a_quota_the_ranking_crowds_quality_out():
    selected, _ = screen(_mixed_features(), set(), top_n=6)
    assert all(t.startswith("SPEC") for t in selected)


def test_a_style_quota_reserves_slots_for_the_quality_names():
    selected, _ = screen(_mixed_features(), set(), top_n=6, style_quotas={"quality": 2})
    assert len([t for t in selected if t.startswith("QUAL")]) == 2
    assert len(selected) == 6


def test_an_excluded_style_is_dropped_from_the_candidate_list():
    selected, _ = screen(_mixed_features(), set(), top_n=10,
                         excluded_styles={"speculative"})
    assert set(selected) == {"QUAL0", "QUAL1", "QUAL2"}


def test_a_held_name_survives_its_style_being_excluded():
    selected, _ = screen(_mixed_features(), {"SPEC3"}, top_n=5,
                         excluded_styles={"speculative"})
    assert "SPEC3" in selected


def test_both_dials_reserve_against_the_same_list():
    """A Nordic compounder settles a Nordic slot and a quality slot at once."""
    feats = {
        "SE-Q": _compounder("SE-Q", country="SE", return_20d_pct=-30),
        "US-S": _feat("US-S", country="US", profit_margin_pct=-20, return_20d_pct=60),
        "US-S2": _feat("US-S2", country="US", profit_margin_pct=-25, return_20d_pct=55),
    }
    selected, _ = screen(feats, set(), top_n=2,
                         region_quotas={"nordics": 1}, style_quotas={"quality": 1})
    assert "SE-Q" in selected
    assert len(selected) == 2


def test_the_surer_dial_is_reserved_first_when_both_are_squeezed():
    """Geography is a fact and a style is a reading, so a squeezed screen keeps
    the region reservation and lets the style one give."""
    feats = {
        "SE-X": _feat("SE-X", country="SE", return_20d_pct=-40),
        "US-Q": _compounder("US-Q", country="US", return_20d_pct=-30),
        "US-S": _feat("US-S", country="US", profit_margin_pct=-20, return_20d_pct=90),
    }
    selected, _ = screen(feats, set(), top_n=1,
                         region_quotas={"nordics": 1}, style_quotas={"quality": 1})
    assert set(selected) == {"SE-X"}


def test_screening_without_style_arguments_is_unchanged():
    feats = _mixed_features()
    assert screen(feats, set(), top_n=5) == screen(
        feats, set(), top_n=5, style_quotas=None, excluded_styles=None)


# ── The two dials share one implementation ────────────────────────────────────

def test_both_dials_are_schemes_over_the_same_arithmetic():
    from fundmgr import regions
    for dial in (regions, styles):
        assert isinstance(dial.SCHEME, allocation.Scheme)
        assert dial.SCHEME.fallback in dial.SCHEME.codes
        assert dial.bucket_of(None) == dial.SCHEME.fallback


def test_debt_to_equity_reaches_the_feature_from_the_cache():
    """It was declared, scored by the screener and rendered in the prompt, but
    never read off the cache — so the balance-sheet leg of every check that
    mentions it was silently inert."""
    import tempfile
    from pathlib import Path

    from fundmgr.data.fundamentals import apply_to_features
    from fundmgr.state.store import Store

    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "t.db")
        store.initialise(1000)
        store.save_fundamentals("X", {"debt_to_equity": 275.0, "profit_margin": 0.05})
        features = {"X": _feat("X")}
        apply_to_features(features, store)

    assert features["X"].debt_to_equity == 275.0
    assert styles.style_of(features["X"]) == "speculative"
