"""
Regional allocation — choosing a book's geographic mix, and holding it there.

A mandate's universe is a list of countries, but a book built off it has no
geography of its own. The screener ranks on momentum, trend and fundamentals,
so a global fund's candidate list — and therefore its book — lands wherever the
strongest signals happened to be that week. "I want roughly 30% Nordics" was
not expressible anywhere in the pipeline.

This module is the single place geography is defined, so the three layers that
have to agree on it cannot drift:

  • the screener, which reserves candidate slots per region. This is the step
    that makes a target reachable at all: a model asked for 30% Nordics out of
    a top-120 list holding four Nordic names cannot deliver it, however the
    prompt is worded.
  • the prompt, which states the mix, the band around it and where the book
    currently stands.
  • the guardrails, which reject a buy that would push a region past its cap.

Ceilings bind and floors do not, and the asymmetry is real rather than
laziness: a guardrail can refuse a trade, it cannot invent one. Rejecting a US
buy does not produce a Nordic buy, so "at least 30% Nordics" is a brief to the
model and a number reported back afterwards, while "at most 40% Nordics" is
mechanical. Everything here that reports a mix says which side it enforced.

A region nobody named is unconstrained — asking for 30% Nordics says nothing
about where the other 70% goes. An explicit 0% is a different instruction:
that is "none of this", so it caps the region at zero rather than at the
tolerance band, and drops its names from the candidate list before the model is
charged for reading them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

from fundmgr.state.models import PortfolioSnapshot


@dataclass(frozen=True)
class Region:
    code: str
    label: str
    countries: frozenset[str]


# Grouped by how an allocator thinks about them rather than by continent: the
# Nordic market is the one this fund family treats as a bloc, and the UK trades
# on its own calendar and currency whatever its geography.
REGIONS: tuple[Region, ...] = (
    Region("nordics", "Nordics", frozenset({"SE", "NO", "DK", "FI", "IS"})),
    Region("north_america", "North America", frozenset({"US", "CA"})),
    Region("uk_ireland", "UK & Ireland", frozenset({"GB", "IE"})),
    Region("europe", "Europe ex-Nordics", frozenset({
        "DE", "FR", "NL", "BE", "AT", "CH", "ES", "PT", "IT", "PL", "LU",
        "GR", "CZ", "HU", "RO", "SK", "SI", "EE", "LV", "LT", "TR", "CY", "MT",
    })),
    Region("asia_pacific", "Asia-Pacific", frozenset({
        "JP", "HK", "CN", "TW", "KR", "SG", "AU", "NZ", "IN", "TH", "MY", "ID",
    })),
    # Anything the universe leaves blank, malformed or outside the groups above.
    # Reportable and targetable like the rest — a book can be held to "0% of
    # whatever I could not classify" — but it is never a country's home.
    Region("other", "Other / unclassified", frozenset()),
)

OTHER_CODE = "other"
REGION_BY_CODE: dict[str, Region] = {r.code: r for r in REGIONS}
_REGION_OF_COUNTRY: dict[str, str] = {
    country: r.code for r in REGIONS for country in r.countries
}

# Floor on reserved candidate slots for any region carrying a positive target.
# A 5% target against top_n=75 is four names; below roughly this the model is
# choosing from too few to have chosen at all.
MIN_REGION_SLOTS = 3

DEFAULT_TOLERANCE_PCT = 10.0


def region_of(country: str | None) -> str:
    """Region code for a universe row's country, `other` when unmapped."""
    code = (country or "").strip().upper()
    return _REGION_OF_COUNTRY.get(code, OTHER_CODE)


def label_of(code: str) -> str:
    region = REGION_BY_CODE.get(code)
    return region.label if region else code


def regions_of(features: Mapping[str, object]) -> dict[str, str]:
    """{ticker: region code} for a feature set, read off each row's country."""
    return {
        ticker: region_of(getattr(feat, "country", None))
        for ticker, feat in features.items()
    }


def options() -> list[dict]:
    """The selectable regions, for a form or an API payload."""
    return [{"code": r.code, "label": r.label} for r in REGIONS]


def clean_targets(raw: Mapping[str, object] | None) -> dict[str, float]:
    """Keep only known regions with a usable percentage, in REGIONS order.

    This is the boundary between a web form and a guardrail, so anything
    unrecognised or unparseable is dropped rather than carried forward — a
    typo'd region must fall back to "unconstrained", never reach the cap
    arithmetic as a string. A blank field is "no opinion" and disappears; an
    explicit 0 is an instruction and survives.

    Raises ValueError when the targets sum past 100% of NAV, which is a mix no
    book can be built to and is worth saying out loud rather than clamping into
    something the user did not ask for.
    """
    out: dict[str, float] = {}
    for region in REGIONS:
        value = (raw or {}).get(region.code)
        if value is None or value == "":
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed < 0 or parsed > 100:
            continue
        out[region.code] = round(parsed, 2)

    total = sum(out.values())
    if total > 100.0 + 1e-9:
        raise ValueError(
            f"Regional targets add up to {total:.0f}% of NAV — they cannot exceed 100%."
        )
    return out


def clean_tolerance(raw: object, default: float = DEFAULT_TOLERANCE_PCT) -> float:
    """The band around each target, in percentage points of NAV."""
    if raw is None or raw == "":
        return default
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return default
    return parsed if 0 <= parsed <= 100 else default


def ceilings(targets: Mapping[str, float], tolerance_pct: float) -> dict[str, float]:
    """The hard per-region NAV ceilings a target mix implies.

    An explicit 0% is an exclusion rather than a band: it would be absurd for
    "no North America" to permit a tenth of the book there because the
    tolerance said so.
    """
    return {
        code: (0.0 if target <= 0 else min(100.0, target + tolerance_pct))
        for code, target in targets.items()
    }


def floors(targets: Mapping[str, float], tolerance_pct: float) -> dict[str, float]:
    """The per-region NAV floors a target mix implies. Advisory — see module docstring."""
    return {code: max(0.0, target - tolerance_pct) for code, target in targets.items()}


def excluded(targets: Mapping[str, float]) -> set[str]:
    """Regions the mix asks for none of."""
    return {code for code, target in targets.items() if target <= 0}


def candidate_quotas(targets: Mapping[str, float], top_n: int) -> dict[str, int]:
    """Candidate slots the screener reserves per region.

    Proportional to the target, because that is the weakest guarantee that
    still works: a candidate list whose own regional mix matches the one being
    asked for can always be built to it, and the unreserved remainder still
    adds whatever else scored well. Ranking alone gives no such guarantee —
    the momentum of one week decides the geography of the book.
    """
    if not targets or top_n <= 0:
        return {}
    quotas = {
        code: max(MIN_REGION_SLOTS, math.ceil(top_n * target / 100))
        for code, target in targets.items() if target > 0
    }
    total = sum(quotas.values())
    if total > top_n:
        # More reserved than there are slots (a small top_n, or the minimum
        # kicking in across many regions). Scale back proportionally and leave
        # every named region at least one name rather than dropping it out of
        # sight entirely.
        quotas = {
            code: max(1, int(n * top_n / total)) for code, n in quotas.items()
        }
    return quotas


def exposure(
    snap: PortfolioSnapshot, region_by_ticker: Mapping[str, str]
) -> dict[str, float]:
    """Region NAV weights of the book as it stands."""
    out: dict[str, float] = {}
    for p in snap.positions:
        if p.shares <= 0:
            continue
        code = region_by_ticker.get(p.ticker, OTHER_CODE)
        out[code] = out.get(code, 0.0) + snap.weight_pct(p.ticker)
    return out


def projected_exposure(
    snap: PortfolioSnapshot,
    actions: Iterable,
    region_by_ticker: Mapping[str, str],
) -> dict[str, float]:
    """Region NAV weights of the book a run would leave behind.

    An action's `target_weight_pct` is where that name ends up — a buy's new
    weight, a sell's remaining weight — so the projection is today's book with
    the run's own targets written over it. That reads correctly both for a
    clean-slate what-if, where every name arrives from an action, and for a
    sleeve review, where most names keep the weight they already had.
    """
    weights = {
        p.ticker: snap.weight_pct(p.ticker) for p in snap.positions if p.shares > 0
    }
    for action in actions:
        if getattr(action, "side", "") == "hold":
            continue
        weights[action.ticker] = float(action.target_weight_pct)

    out: dict[str, float] = {}
    for ticker, weight in weights.items():
        code = region_by_ticker.get(ticker, OTHER_CODE)
        out[code] = out.get(code, 0.0) + weight
    return out


def candidate_counts(region_by_ticker: Mapping[str, str]) -> dict[str, int]:
    """How many candidates each region contributes to a screened set."""
    out: dict[str, int] = {}
    for code in region_by_ticker.values():
        out[code] = out.get(code, 0) + 1
    return out


def mix_rows(
    targets: Mapping[str, float],
    tolerance_pct: float,
    achieved: Mapping[str, float],
    candidates: Mapping[str, int] | None = None,
) -> list[dict]:
    """Target versus outcome, one row per region that carries either.

    `status` is the reportable verdict: `over` cannot happen through the
    guardrails and means the book was already past its ceiling before the run;
    `short` is the ordinary way a mix misses, since nothing can force a buy.
    """
    tops = ceilings(targets, tolerance_pct)
    bottoms = floors(targets, tolerance_pct)
    codes = [
        r.code for r in REGIONS
        if r.code in targets or achieved.get(r.code) or (candidates or {}).get(r.code)
    ]

    rows = []
    for code in codes:
        target = targets.get(code)
        got = round(achieved.get(code, 0.0), 1)
        row = {
            "code": code,
            "label": label_of(code),
            "target_pct": target,
            "floor_pct": bottoms.get(code),
            "ceiling_pct": tops.get(code),
            "achieved_pct": got,
            "candidates": (candidates or {}).get(code, 0),
        }
        if target is None:
            row["status"] = "unconstrained"
        elif target <= 0:
            row["status"] = "excluded" if got <= 0 else "over"
        elif got + 1e-9 < bottoms[code]:
            row["status"] = "short"
        elif got > tops[code] + 1e-9:
            row["status"] = "over"
        else:
            row["status"] = "on_target"
        rows.append(row)
    return rows


def shortfalls(rows: Iterable[Mapping]) -> list[str]:
    """Human-readable lines for the regions a run did not reach."""
    out = []
    for row in rows:
        if row.get("status") == "short":
            out.append(
                f"{row['label']}: {row['achieved_pct']:.1f}% against a "
                f"{row['target_pct']:.0f}% target "
                f"({row['candidates']} candidate(s) were offered)"
            )
        elif row.get("status") == "over":
            out.append(
                f"{row['label']}: {row['achieved_pct']:.1f}% against a "
                f"{row['target_pct']:.0f}% target — already past its ceiling "
                f"before this run"
            )
    return out


def prompt_block(
    targets: Mapping[str, float],
    tolerance_pct: float,
    current: Mapping[str, float],
    candidates: Mapping[str, int] | None = None,
) -> str:
    """The regional brief as the model sees it, or "" when no mix is set."""
    if not targets:
        return ""

    tops = ceilings(targets, tolerance_pct)
    bottoms = floors(targets, tolerance_pct)
    lines = [
        "  Regional allocation target (% of NAV, each name's region is tagged "
        "in the universe below):",
    ]
    for region in REGIONS:
        if region.code not in targets:
            continue
        target = targets[region.code]
        now = current.get(region.code, 0.0)
        offered = (candidates or {}).get(region.code)
        band = (
            "EXCLUDED — do not buy here"
            if target <= 0
            else f"target {target:.0f}%  allowed {bottoms[region.code]:.0f}–{tops[region.code]:.0f}%"
        )
        line = f"    {region.label:<22} {band}  ·  now {now:.1f}%"
        if offered is not None:
            line += f"  ·  {offered} candidate(s) below"
        lines.append(line)

    named = ", ".join(label_of(c) for c in targets)
    lines += [
        f"    Regions not listed ({100 - sum(targets.values()):.0f}% of NAV is "
        f"unallocated by this mix) are unconstrained — only {named} are managed here."
        if sum(targets.values()) < 100 else
        "    Every region is named above; the mix accounts for the whole book.",
        "    The upper bound is mechanical: guardrails reject a buy that would take a "
        "region past it. The lower bound is not — nothing can force a buy, so a region "
        "left short simply comes back short. Build the mix here or it does not happen.",
    ]
    return "\n".join(lines)
