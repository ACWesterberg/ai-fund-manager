"""
Regional allocation — choosing a book's geographic mix, and holding it there.

A mandate's universe is a list of countries, but a book built off it has no
geography of its own. The screener ranks on momentum, trend and fundamentals,
so a global fund's candidate list — and therefore its book — lands wherever the
strongest signals happened to be that week. "I want roughly 30% Nordics" was
not expressible anywhere in the pipeline.

This module is the single place geography is defined, so the three layers that
have to agree on it cannot drift: the screener reserves candidate slots per
region, the prompt states the mix, and the guardrails reject a buy that would
push a region past its cap. The arithmetic behind all three — which bound binds,
what an explicit 0% means, how slots are reserved — lives in `allocation.py` and
is shared with the style dial.

A region's country code is a *fact* on the universe row, which is what makes
this dial cheap and stable: it never goes missing and never changes between
runs. `styles.py` is the same machinery over a judgement, and says so.
"""
from __future__ import annotations

from typing import Iterable, Mapping

from fundmgr import allocation
from fundmgr.allocation import Bucket, Scheme
from fundmgr.state.models import PortfolioSnapshot

DEFAULT_TOLERANCE_PCT = allocation.DEFAULT_TOLERANCE_PCT
MIN_REGION_SLOTS = allocation.MIN_BUCKET_SLOTS
OTHER_CODE = "other"

# Grouped by how an allocator thinks about them rather than by continent: the
# Nordic market is the one this fund family treats as a bloc, and the UK trades
# on its own calendar and currency whatever its geography.
_COUNTRIES: dict[str, frozenset[str]] = {
    "nordics": frozenset({"SE", "NO", "DK", "FI", "IS"}),
    "north_america": frozenset({"US", "CA"}),
    "uk_ireland": frozenset({"GB", "IE"}),
    "europe": frozenset({
        "DE", "FR", "NL", "BE", "AT", "CH", "ES", "PT", "IT", "PL", "LU",
        "GR", "CZ", "HU", "RO", "SK", "SI", "EE", "LV", "LT", "TR", "CY", "MT",
    }),
    "asia_pacific": frozenset({
        "JP", "HK", "CN", "TW", "KR", "SG", "AU", "NZ", "IN", "TH", "MY", "ID",
    }),
    OTHER_CODE: frozenset(),
}

SCHEME = Scheme(
    name="Regional",
    buckets=(
        Bucket("nordics", "Nordics"),
        Bucket("north_america", "North America"),
        Bucket("uk_ireland", "UK & Ireland"),
        Bucket("europe", "Europe ex-Nordics"),
        Bucket("asia_pacific", "Asia-Pacific"),
        # Anything the universe leaves blank, malformed or outside the groups
        # above. Reportable and targetable like the rest — a book can be held to
        # "0% of whatever I could not classify" — but never a country's home.
        Bucket(OTHER_CODE, "Other / unclassified"),
    ),
    fallback=OTHER_CODE,
    heading=(
        "Regional allocation target (% of NAV, each name's region is tagged in "
        "the universe below):"
    ),
    plural="Regions",
)

REGIONS: tuple[Bucket, ...] = SCHEME.buckets
REGION_BY_CODE: dict[str, Bucket] = {b.code: b for b in SCHEME.buckets}
_REGION_OF_COUNTRY: dict[str, str] = {
    country: code for code, countries in _COUNTRIES.items() for country in countries
}


def region_of(country: str | None) -> str:
    """Region code for a universe row's country, `other` when unmapped."""
    code = (country or "").strip().upper()
    return _REGION_OF_COUNTRY.get(code, OTHER_CODE)


def label_of(code: str) -> str:
    return SCHEME.label_of(code)


def bucket_of(feat: object | None) -> str:
    """This dial's bucket for one feature row — the guardrails' uniform entry
    point across both mixes. A missing feature is `other`, never a breach."""
    return region_of(getattr(feat, "country", None))


def regions_of(features: Mapping[str, object]) -> dict[str, str]:
    """{ticker: region code} for a feature set, read off each row's country."""
    return {
        ticker: region_of(getattr(feat, "country", None))
        for ticker, feat in features.items()
    }


def options() -> list[dict]:
    return SCHEME.options()


def clean_targets(raw: Mapping[str, object] | None) -> dict[str, float]:
    return allocation.clean_targets(raw, SCHEME)


def clean_tolerance(raw: object, default: float = DEFAULT_TOLERANCE_PCT) -> float:
    return allocation.clean_tolerance(raw, default)


def ceilings(targets: Mapping[str, float], tolerance_pct: float) -> dict[str, float]:
    return allocation.ceilings(targets, tolerance_pct)


def floors(targets: Mapping[str, float], tolerance_pct: float) -> dict[str, float]:
    return allocation.floors(targets, tolerance_pct)


def excluded(targets: Mapping[str, float]) -> set[str]:
    return allocation.excluded(targets)


def candidate_quotas(targets: Mapping[str, float], top_n: int) -> dict[str, int]:
    return allocation.candidate_quotas(targets, top_n)


def exposure(
    snap: PortfolioSnapshot, region_by_ticker: Mapping[str, str]
) -> dict[str, float]:
    return allocation.exposure(snap, region_by_ticker, OTHER_CODE)


def projected_exposure(
    snap: PortfolioSnapshot, actions: Iterable, region_by_ticker: Mapping[str, str]
) -> dict[str, float]:
    return allocation.projected_exposure(snap, actions, region_by_ticker, OTHER_CODE)


def candidate_counts(region_by_ticker: Mapping[str, str]) -> dict[str, int]:
    return allocation.bucket_counts(region_by_ticker)


def mix_rows(
    targets: Mapping[str, float],
    tolerance_pct: float,
    achieved: Mapping[str, float],
    candidates: Mapping[str, int] | None = None,
) -> list[dict]:
    return allocation.mix_rows(targets, tolerance_pct, achieved, SCHEME, candidates)


def shortfalls(rows: Iterable[Mapping]) -> list[str]:
    return allocation.shortfalls(rows)


def prompt_block(
    targets: Mapping[str, float],
    tolerance_pct: float,
    current: Mapping[str, float],
    candidates: Mapping[str, int] | None = None,
) -> str:
    return allocation.prompt_block(targets, tolerance_pct, current, SCHEME, candidates)
