"""
Holding a book to a mix — the arithmetic every allocation dial shares.

Two dials use this: `regions.py` (geography, read off the universe row) and
`styles.py` (risk/quality character, computed from reported fundamentals). The
semantics below are subtle enough that two copies would drift, and a drift here
is a book quietly built to something other than what was asked for.

The rules, once:

  • **Ceilings bind, floors do not.** A guardrail can refuse a trade; it cannot
    invent one. Rejecting a US buy never produces a Nordic buy, so "at least
    30%" is a brief to the model and a number reported back afterwards, while
    "at most 40%" is mechanical. Anything claiming to enforce a floor is
    claiming a guardrail can create a buy.
  • **A bucket nobody named is unconstrained.** Asking for 30% Nordics says
    nothing about where the other 70% goes.
  • **An explicit 0% is an exclusion, not a band.** It would be absurd for "no
    North America" to permit a tenth of the book there because the tolerance
    said so, so a zero caps at zero — and the screener drops that bucket rather
    than paying to show names whose buys would only be rejected.
  • **Reserved candidate slots are what make a target reachable.** The screener
    score is blind to both dials, so without a proportional reservation the mix
    is unbuildable rather than merely hard.

A scheme's `fallback` bucket is where anything unclassifiable lands. For
geography that is a blank or malformed country code; for style it is a name
whose fundamentals were not on file. Either way the fallback is reported like
any other bucket, and a scheme decides for itself whether it can be targeted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

from fundmgr.state.models import PortfolioSnapshot

DEFAULT_TOLERANCE_PCT = 10.0

# Floor on reserved candidate slots for any bucket carrying a positive target.
# A 5% target against top_n=75 is four names; below roughly this the model is
# choosing from too few to have chosen at all.
MIN_BUCKET_SLOTS = 3


@dataclass(frozen=True)
class Bucket:
    code: str
    label: str


@dataclass(frozen=True)
class Scheme:
    """One allocation dial: its buckets, where the unclassifiable go, and how it
    introduces itself in a prompt."""
    name: str
    buckets: tuple[Bucket, ...]
    fallback: str
    heading: str
    plural: str          # what to call these buckets in prose ("Regions", "Styles")

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(b.code for b in self.buckets)

    def label_of(self, code: str) -> str:
        for b in self.buckets:
            if b.code == code:
                return b.label
        return code

    def options(self) -> list[dict]:
        """The selectable buckets, for a form or an API payload."""
        return [{"code": b.code, "label": b.label} for b in self.buckets]


def clean_targets(raw: Mapping[str, object] | None, scheme: Scheme) -> dict[str, float]:
    """Keep only known buckets with a usable percentage, in scheme order.

    This is the boundary between a web form and a guardrail, so anything
    unrecognised or unparseable is dropped rather than carried forward — a
    typo'd bucket must fall back to "unconstrained", never reach the cap
    arithmetic as a string. A blank field is "no opinion" and disappears; an
    explicit 0 is an instruction and survives.

    Raises ValueError when the targets sum past 100% of NAV, which is a mix no
    book can be built to and is worth saying out loud rather than clamping into
    something the user did not ask for.
    """
    out: dict[str, float] = {}
    for bucket in scheme.buckets:
        value = (raw or {}).get(bucket.code)
        if value is None or value == "":
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed < 0 or parsed > 100:
            continue
        out[bucket.code] = round(parsed, 2)

    total = sum(out.values())
    if total > 100.0 + 1e-9:
        raise ValueError(
            f"{scheme.name} targets add up to {total:.0f}% of NAV — they cannot exceed 100%."
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
    """The hard per-bucket NAV ceilings a target mix implies."""
    return {
        code: (0.0 if target <= 0 else min(100.0, target + tolerance_pct))
        for code, target in targets.items()
    }


def floors(targets: Mapping[str, float], tolerance_pct: float) -> dict[str, float]:
    """The per-bucket NAV floors a target mix implies. Advisory — see module docstring."""
    return {code: max(0.0, target - tolerance_pct) for code, target in targets.items()}


def excluded(targets: Mapping[str, float]) -> set[str]:
    """Buckets the mix asks for none of."""
    return {code for code, target in targets.items() if target <= 0}


def candidate_quotas(
    targets: Mapping[str, float], top_n: int, min_slots: int = MIN_BUCKET_SLOTS
) -> dict[str, int]:
    """Candidate slots the screener reserves per bucket.

    Proportional to the target, because that is the weakest guarantee that
    still works: a candidate list whose own mix matches the one being asked for
    can always be built to it, and the unreserved remainder still goes to
    whatever scored well. Ranking alone gives no such guarantee.
    """
    if not targets or top_n <= 0:
        return {}
    quotas = {
        code: max(min_slots, math.ceil(top_n * target / 100))
        for code, target in targets.items() if target > 0
    }
    total = sum(quotas.values())
    if total > top_n:
        # More reserved than there are slots (a small top_n, or the minimum
        # kicking in across many buckets). Scale back proportionally and leave
        # every named bucket at least one name rather than dropping it out of
        # sight entirely.
        quotas = {code: max(1, int(n * top_n / total)) for code, n in quotas.items()}
    return quotas


def exposure(
    snap: PortfolioSnapshot, bucket_by_ticker: Mapping[str, str], fallback: str
) -> dict[str, float]:
    """Bucket NAV weights of the book as it stands."""
    out: dict[str, float] = {}
    for p in snap.positions:
        if p.shares <= 0:
            continue
        code = bucket_by_ticker.get(p.ticker, fallback)
        out[code] = out.get(code, 0.0) + snap.weight_pct(p.ticker)
    return out


def projected_exposure(
    snap: PortfolioSnapshot,
    actions: Iterable,
    bucket_by_ticker: Mapping[str, str],
    fallback: str,
) -> dict[str, float]:
    """Bucket NAV weights of the book a run would leave behind.

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
        code = bucket_by_ticker.get(ticker, fallback)
        out[code] = out.get(code, 0.0) + weight
    return out


def bucket_counts(bucket_by_ticker: Mapping[str, str]) -> dict[str, int]:
    """How many candidates each bucket contributes to a screened set."""
    out: dict[str, int] = {}
    for code in bucket_by_ticker.values():
        out[code] = out.get(code, 0) + 1
    return out


def mix_rows(
    targets: Mapping[str, float],
    tolerance_pct: float,
    achieved: Mapping[str, float],
    scheme: Scheme,
    candidates: Mapping[str, int] | None = None,
) -> list[dict]:
    """Target versus outcome, one row per bucket that carries either.

    `status` is the reportable verdict: `over` cannot happen through the
    guardrails and means the book was already past its ceiling before the run;
    `short` is the ordinary way a mix misses, since nothing can force a buy.
    """
    tops = ceilings(targets, tolerance_pct)
    bottoms = floors(targets, tolerance_pct)
    codes = [
        b.code for b in scheme.buckets
        if b.code in targets or achieved.get(b.code) or (candidates or {}).get(b.code)
    ]

    rows = []
    for code in codes:
        target = targets.get(code)
        got = round(achieved.get(code, 0.0), 1)
        row = {
            "code": code,
            "label": scheme.label_of(code),
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
    """Human-readable lines for the buckets a run did not reach."""
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
    scheme: Scheme,
    candidates: Mapping[str, int] | None = None,
    extra_lines: Iterable[str] = (),
    closing: bool = True,
) -> str:
    """The brief as the model sees it, or "" when no mix is set.

    `closing` carries the which-bound-binds paragraph. It is the same sentence
    for every dial, so a run with two mixes set prints it once rather than
    teaching the model the same rule twice in consecutive paragraphs.
    """
    if not targets:
        return ""

    tops = ceilings(targets, tolerance_pct)
    bottoms = floors(targets, tolerance_pct)
    lines = [f"  {scheme.heading}"]
    for bucket in scheme.buckets:
        if bucket.code not in targets:
            continue
        target = targets[bucket.code]
        now = current.get(bucket.code, 0.0)
        offered = (candidates or {}).get(bucket.code)
        band = (
            "EXCLUDED — do not buy here"
            if target <= 0
            else f"target {target:.0f}%  allowed "
                 f"{bottoms[bucket.code]:.0f}–{tops[bucket.code]:.0f}%"
        )
        line = f"    {bucket.label:<26} {band}  ·  now {now:.1f}%"
        if offered is not None:
            line += f"  ·  {offered} candidate(s) below"
        lines.append(line)

    named = ", ".join(scheme.label_of(c) for c in targets)
    total = sum(targets.values())
    lines.append(
        f"    {scheme.plural} not listed ({100 - total:.0f}% of NAV is unallocated "
        f"by this mix) are unconstrained — only {named} are managed here."
        if total < 100 else
        f"    Every one of the {scheme.plural.lower()} is named above; the mix "
        f"accounts for the whole book."
    )
    lines.extend(f"    {line}" for line in extra_lines)
    if closing:
        lines.append(
            "    The upper bound is mechanical: guardrails reject a buy that would take a "
            "bucket past it. The lower bound is not — nothing can force a buy, so a bucket "
            "left short simply comes back short. Build the mix here or it does not happen."
        )
    return "\n".join(lines)
