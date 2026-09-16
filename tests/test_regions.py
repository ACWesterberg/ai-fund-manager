"""
Regional allocation — the mapping, the arithmetic, and the screener quotas that
make a target reachable rather than merely stated.

The asymmetry these tests keep honest: a ceiling is mechanical and a floor is
not. Anything that claims to enforce a floor is claiming a guardrail can invent
a trade.
"""
from __future__ import annotations

import pytest

from fundmgr import regions
from fundmgr.data.prices import TickerFeatures
from fundmgr.data.screener import screen
from fundmgr.engine.schema import Action
from fundmgr.state.models import PortfolioSnapshot, Position


def _feat(ticker: str, country: str, r20: float = 0.0) -> TickerFeatures:
    return TickerFeatures(
        ticker=ticker, name=ticker, last_price=100.0, last_date="2026-09-15",
        data_age_trading_days=1, country=country, return_20d_pct=r20,
    )


def _pos(ticker: str, value: float) -> Position:
    p = Position(ticker=ticker, shares=1.0, avg_cost_sek=value)
    p.current_price_sek = value
    return p


# ── Mapping ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("country,expected", [
    ("SE", "nordics"), ("no", "nordics"), (" DK ", "nordics"), ("FI", "nordics"),
    ("US", "north_america"), ("CA", "north_america"),
    ("GB", "uk_ireland"), ("IE", "uk_ireland"),
    ("DE", "europe"), ("FR", "europe"), ("PL", "europe"),
    ("JP", "asia_pacific"),
])
def test_country_maps_to_its_region(country, expected):
    assert regions.region_of(country) == expected


@pytest.mark.parametrize("bad", ["", None, "ZZ", "SE0001234567", "   "])
def test_unmapped_country_falls_into_other(bad):
    """Universe rows carry blanks and the odd ISIN in the country column."""
    assert regions.region_of(bad) == regions.OTHER_CODE


def test_every_universe_country_is_classified():
    """A country the real universes carry must not land in `other` unnoticed."""
    from fundmgr.config import CONFIG_DIR, get_enabled_tickers

    unmapped = set()
    for path in CONFIG_DIR.glob("universe*.csv"):
        for t in get_enabled_tickers(path):
            code = (t.country or "").strip().upper()
            if len(code) == 2 and code.isalpha() and regions.region_of(code) == regions.OTHER_CODE:
                unmapped.add(code)
    assert not unmapped, f"universe countries with no region: {sorted(unmapped)}"


# ── Validating a submitted mix ────────────────────────────────────────────────

def test_clean_targets_keeps_known_regions_in_region_order():
    out = regions.clean_targets({"north_america": 40, "nordics": "30"})
    assert list(out) == ["nordics", "north_america"]
    assert out == {"nordics": 30.0, "north_america": 40.0}


@pytest.mark.parametrize("raw", [
    {"atlantis": 30},          # unknown region
    {"nordics": "thirty"},     # unparseable
    {"nordics": -5},           # negative
    {"nordics": 140},          # past 100
    {"nordics": ""},           # blank field = no opinion
    {"nordics": None},
])
def test_clean_targets_drops_anything_unusable(raw):
    assert regions.clean_targets(raw) == {}


def test_explicit_zero_survives_cleaning():
    """Blank is "no opinion"; 0 is "none of this" and must not be dropped."""
    assert regions.clean_targets({"north_america": 0}) == {"north_america": 0.0}


def test_targets_over_one_hundred_percent_are_refused():
    with pytest.raises(ValueError, match="cannot exceed 100"):
        regions.clean_targets({"nordics": 60, "north_america": 50})


def test_targets_summing_to_exactly_one_hundred_are_fine():
    assert sum(regions.clean_targets({"nordics": 30, "north_america": 70}).values()) == 100


@pytest.mark.parametrize("raw,expected", [
    (None, regions.DEFAULT_TOLERANCE_PCT), ("", regions.DEFAULT_TOLERANCE_PCT),
    ("junk", regions.DEFAULT_TOLERANCE_PCT), (-1, regions.DEFAULT_TOLERANCE_PCT),
    (101, regions.DEFAULT_TOLERANCE_PCT), (5, 5.0), ("7.5", 7.5), (0, 0.0),
])
def test_tolerance_falls_back_to_the_default_when_unusable(raw, expected):
    assert regions.clean_tolerance(raw) == expected


# ── Bands ─────────────────────────────────────────────────────────────────────

def test_ceiling_is_target_plus_tolerance():
    assert regions.ceilings({"nordics": 30.0}, 10) == {"nordics": 40.0}


def test_an_excluded_region_is_capped_at_zero_not_at_the_tolerance():
    """"No North America" must not permit a tenth of the book there."""
    assert regions.ceilings({"north_america": 0.0}, 10) == {"north_america": 0.0}


def test_floor_never_goes_negative():
    assert regions.floors({"nordics": 5.0}, 10) == {"nordics": 0.0}


def test_only_named_regions_get_a_band():
    caps = regions.ceilings({"nordics": 30.0}, 10)
    assert "europe" not in caps and "north_america" not in caps


def test_excluded_lists_only_the_zeroed_regions():
    assert regions.excluded({"nordics": 30.0, "north_america": 0.0}) == {"north_america"}


# ── Candidate quotas ──────────────────────────────────────────────────────────

def test_quota_is_proportional_to_the_target():
    assert regions.candidate_quotas({"nordics": 30.0}, 120) == {"nordics": 36}


def test_a_small_target_still_gets_enough_names_to_choose_between():
    assert regions.candidate_quotas({"nordics": 2.0}, 75)["nordics"] == regions.MIN_REGION_SLOTS


def test_an_excluded_region_reserves_nothing():
    assert regions.candidate_quotas({"north_america": 0.0}, 120) == {}


def test_quotas_never_outgrow_the_candidate_list():
    quotas = regions.candidate_quotas(
        {"nordics": 25.0, "north_america": 25.0, "europe": 25.0, "uk_ireland": 25.0}, 8
    )
    assert sum(quotas.values()) <= 8
    assert all(n >= 1 for n in quotas.values())


def test_no_targets_reserve_nothing():
    assert regions.candidate_quotas({}, 120) == {}


# ── Exposure ──────────────────────────────────────────────────────────────────

def test_exposure_sums_nav_weights_by_region():
    snap = PortfolioSnapshot(
        positions=[_pos("A.ST", 30_000), _pos("AAPL", 20_000), _pos("B.ST", 10_000)],
        cash_sek=40_000,
    )
    mapping = {"A.ST": "nordics", "AAPL": "north_america", "B.ST": "nordics"}
    got = regions.exposure(snap, mapping)
    assert round(got["nordics"]) == 40
    assert round(got["north_america"]) == 20


def test_a_position_with_no_mapping_lands_in_other():
    snap = PortfolioSnapshot(positions=[_pos("X", 50_000)], cash_sek=50_000)
    assert round(regions.exposure(snap, {})[regions.OTHER_CODE]) == 50


def test_projection_reads_target_weights_off_a_clean_slate():
    snap = PortfolioSnapshot(positions=[], cash_sek=100_000)
    actions = [
        Action(ticker="A.ST", side="buy", target_weight_pct=18, sek_estimate=18_000,
               confidence=0.8, thesis="t"),
        Action(ticker="AAPL", side="buy", target_weight_pct=12, sek_estimate=12_000,
               confidence=0.7, thesis="t"),
    ]
    got = regions.projected_exposure(
        snap, actions, {"A.ST": "nordics", "AAPL": "north_america"})
    assert got == {"nordics": 18.0, "north_america": 12.0}


def test_projection_writes_the_runs_targets_over_the_book_it_inherits():
    """A sell to 5% is where that name ends up, not what leaves the book."""
    snap = PortfolioSnapshot(
        positions=[_pos("A.ST", 30_000), _pos("AAPL", 20_000)], cash_sek=50_000)
    actions = [Action(ticker="A.ST", side="sell", target_weight_pct=5,
                      sek_estimate=25_000, confidence=0.9, thesis="t")]
    got = regions.projected_exposure(
        snap, actions, {"A.ST": "nordics", "AAPL": "north_america"})
    assert got["nordics"] == 5.0
    assert round(got["north_america"]) == 20      # untouched names keep their weight


def test_a_hold_does_not_rewrite_the_weight_a_position_already_has():
    snap = PortfolioSnapshot(positions=[_pos("A.ST", 30_000)], cash_sek=70_000)
    actions = [Action(ticker="A.ST", side="hold", target_weight_pct=0,
                      sek_estimate=0, confidence=0.9, thesis="t")]
    got = regions.projected_exposure(snap, actions, {"A.ST": "nordics"})
    assert round(got["nordics"]) == 30


# ── Reporting ─────────────────────────────────────────────────────────────────

def test_mix_rows_label_each_region_against_its_band():
    rows = {r["code"]: r for r in regions.mix_rows(
        {"nordics": 30.0, "north_america": 20.0, "europe": 0.0}, 10,
        {"nordics": 31.0, "north_america": 4.0, "uk_ireland": 12.0},
    )}
    assert rows["nordics"]["status"] == "on_target"
    assert rows["north_america"]["status"] == "short"
    assert rows["europe"]["status"] == "excluded"
    assert rows["uk_ireland"]["status"] == "unconstrained"


def test_a_book_already_past_its_ceiling_is_reported_as_over():
    """Guardrails cannot produce this — an inherited book can."""
    rows = {r["code"]: r for r in regions.mix_rows({"nordics": 30.0}, 10, {"nordics": 55.0})}
    assert rows["nordics"]["status"] == "over"


def test_shortfalls_name_the_region_and_what_it_was_offered():
    rows = regions.mix_rows({"nordics": 30.0}, 10, {"nordics": 4.0}, {"nordics": 2})
    lines = regions.shortfalls(rows)
    assert len(lines) == 1
    assert "Nordics" in lines[0] and "30% target" in lines[0] and "2 candidate" in lines[0]


def test_an_on_target_run_reports_no_shortfall():
    assert regions.shortfalls(regions.mix_rows({"nordics": 30.0}, 10, {"nordics": 28.0})) == []


def test_prompt_block_is_empty_without_a_mix():
    assert regions.prompt_block({}, 10, {}, {}) == ""


def test_prompt_block_states_which_bound_is_enforced():
    block = regions.prompt_block({"nordics": 30.0}, 10, {"nordics": 0.0}, {"nordics": 20})
    assert "Nordics" in block and "20–40%" in block
    # The model must not be left thinking something will fill the gap for it.
    assert "guardrails reject" in block
    assert "nothing can force a buy" in block


def test_prompt_block_says_an_excluded_region_is_not_buyable():
    block = regions.prompt_block({"north_america": 0.0}, 10, {}, {})
    assert "EXCLUDED" in block


# ── Screener quotas ───────────────────────────────────────────────────────────

def _mixed_features() -> dict[str, TickerFeatures]:
    """Ten strong US names and three weak Nordic ones — the shape that makes a
    regional target unbuildable when the ranking alone decides."""
    feats = {f"US{i}": _feat(f"US{i}", "US", r20=50 - i) for i in range(10)}
    feats.update({f"SE{i}": _feat(f"SE{i}", "SE", r20=-20 - i) for i in range(3)})
    return feats


def test_without_a_quota_the_ranking_crowds_a_region_out():
    selected, _ = screen(_mixed_features(), set(), top_n=6)
    assert all(t.startswith("US") for t in selected)


def test_a_quota_reserves_slots_for_a_region_the_ranking_would_skip():
    selected, _ = screen(_mixed_features(), set(), top_n=6, region_quotas={"nordics": 2})
    nordic = [t for t in selected if t.startswith("SE")]
    assert len(nordic) == 2
    assert len(selected) == 6          # reserved slots come out of top_n, not on top of it


def test_a_quota_takes_the_best_of_its_region():
    selected, _ = screen(_mixed_features(), set(), top_n=6, region_quotas={"nordics": 1})
    assert "SE0" in selected           # the strongest of the three weak names


def test_an_excluded_region_is_dropped_from_the_candidate_list():
    selected, _ = screen(
        _mixed_features(), set(), top_n=10, excluded_regions={"north_america"})
    assert set(selected) == {"SE0", "SE1", "SE2"}


def test_a_held_name_survives_its_region_being_excluded():
    """You must be able to sell what you own, wherever it is listed."""
    selected, _ = screen(
        _mixed_features(), {"US3"}, top_n=5, excluded_regions={"north_america"})
    assert "US3" in selected


def test_a_pinned_name_survives_its_region_being_excluded():
    selected, _ = screen(
        _mixed_features(), set(), top_n=5, pinned_tickers={"US4"},
        excluded_regions={"north_america"})
    assert "US4" in selected


def test_held_names_count_towards_their_regions_quota():
    selected, _ = screen(_mixed_features(), {"SE0"}, top_n=6, region_quotas={"nordics": 2})
    assert len([t for t in selected if t.startswith("SE")]) == 2


def test_screening_without_regional_arguments_is_unchanged():
    feats = _mixed_features()
    assert screen(feats, set(), top_n=5) == screen(
        feats, set(), top_n=5, region_quotas=None, excluded_regions=None)
