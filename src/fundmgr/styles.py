"""
Style allocation — asking a book for some quality and some risk, in proportion.

The five profiles bundle style into the *universe*: `config_buffett_*.yaml`
brings 96 names that already passed a quality screen, and the Global profiles
bring 17k that passed nothing. So "40% Buffett-style compounders and 20% higher
-risk names" was unaskable — the Buffett universe has no risk sleeve in it to
reach, and the global universe has no quality screen to lean on. The mandates
cap risky *names* (small caps at 8%, micro-caps and >100% vol at 5%, one at a
time) but never say how much of the book should be risk-on.

This is the same machinery as `regions.py` over `allocation.py`. What differs,
and why it is stated everywhere this dial surfaces:

  **A region is a fact; a style is a judgement.** `country` sits on the universe
  row, never missing and never changing. A style is computed here from reported
  fundamentals on a 7-day TTL plus realised volatility, so a name can be
  unclassifiable this week and classified the next.

That asymmetry sets one hard rule: **a name is only ever bucketed on a positive
finding, never on an absence.** Missing fundamentals mean `unclassified`, which
is reported, never targeted, and never blocked — rejecting a buy because the
cache had not filled would be failing closed on a gap, and "blocking a run on a
guess is worse than the spend it saves" is already this repo's position. The
honest cost is that the ceiling under-counts: an unclassified name that is in
truth speculative is not counted against the speculative cap. Everything that
reports a style mix says so rather than implying the cap is airtight.

Thresholds deliberately reuse the ones `screener.py` already scores on and the
criteria `mandate_buffett.md` states in prose (high ROE, durable margins, a
conservative balance sheet, growth that is not shrinking), so the classifier and
the mandate it is named after cannot quietly disagree.

Selection is always *within* the profile's own universe. Asking a Global run for
40% Buffett-style gets the highest-quality names in the global universe, not
names imported from `universe_buffett.csv` — mandate and universe travel
together, and a dial that broke that would be mixing two funds.
"""
from __future__ import annotations

from typing import Iterable, Mapping

from fundmgr import allocation
from fundmgr.allocation import Bucket, Scheme
from fundmgr.state.models import PortfolioSnapshot

DEFAULT_TOLERANCE_PCT = allocation.DEFAULT_TOLERANCE_PCT
UNCLASSIFIED = "unclassified"

SCHEME = Scheme(
    name="Style",
    buckets=(
        Bucket("quality", "Buffett-style quality"),
        Bucket("growth", "Growth"),
        Bucket("speculative", "Higher-risk / speculative"),
        # Never a verdict, only an absence of one. Targetable so a book can be
        # held to "0% of what I cannot see", which is a real preference — but a
        # name lands here because the data was missing, so nothing else treats
        # it as a finding.
        Bucket(UNCLASSIFIED, "Unclassified (no data)"),
    ),
    fallback=UNCLASSIFIED,
    heading=(
        "Style allocation target (% of NAV, each name's style is tagged in the "
        "universe below with the figures behind it):"
    ),
    plural="Styles",
)

STYLE_BY_CODE: dict[str, Bucket] = {b.code: b for b in SCHEME.buckets}

# ── Thresholds ────────────────────────────────────────────────────────────────
# Quality: the Buffett mandate's own screen, in the fields this repo actually
# carries. All four must pass, and at least QUALITY_MIN_SIGNALS of them must be
# on file — three of four is enough evidence, one lucky ratio is not.
QUALITY_ROE_PCT = 15.0
QUALITY_MARGIN_PCT = 10.0
QUALITY_MAX_DEBT_TO_EQUITY = 150.0
QUALITY_MIN_REVENUE_GROWTH_PCT = 0.0      # compounders do not shrink
QUALITY_MIN_SIGNALS = 3

# Hard risk: a positive statement about the business, enough on its own.
RISK_MAX_DEBT_TO_EQUITY = 250.0

# Soft risk: a statement about the tape, not the business. On its own it marks a
# name speculative only at an extreme, and it never overrides a quality pass —
# a compounder in a violent quarter is still a compounder.
RISK_VOL_PCT = 80.0
RISK_DRAWDOWN_PCT = -60.0

GROWTH_REVENUE_PCT = 15.0
GROWTH_EARNINGS_PCT = 20.0


def classify(feat: object) -> tuple[str, str]:
    """(style code, the figures that decided it).

    Order of resolution: a fundamental risk flag wins outright, then the quality
    screen, then growth, then nothing. A loss-making hyper-grower is speculative
    rather than growth — for a risk-appetite dial that is the whole point.
    """
    roe = getattr(feat, "roe_pct", None)
    margin = getattr(feat, "profit_margin_pct", None)
    debt = getattr(feat, "debt_to_equity", None)
    rev = getattr(feat, "revenue_growth_pct", None)
    earn = getattr(feat, "earnings_growth_pct", None)
    vol = getattr(feat, "vol_20d_ann_pct", None)
    drawdown = getattr(feat, "pct_from_52w_high", None)

    hard_risk = []
    if margin is not None and margin < 0:
        hard_risk.append(f"loss-making (net margin {margin:.1f}%)")
    if roe is not None and roe < 0:
        hard_risk.append(f"negative ROE ({roe:.1f}%)")
    if debt is not None and debt > RISK_MAX_DEBT_TO_EQUITY:
        hard_risk.append(f"D/E {debt:.0f}")
    if hard_risk:
        return "speculative", ", ".join(hard_risk)

    quality_checks = (
        (roe, roe is not None and roe >= QUALITY_ROE_PCT, f"ROE {roe:.1f}%" if roe is not None else ""),
        (margin, margin is not None and margin >= QUALITY_MARGIN_PCT,
         f"net margin {margin:.1f}%" if margin is not None else ""),
        (debt, debt is not None and debt <= QUALITY_MAX_DEBT_TO_EQUITY,
         f"D/E {debt:.0f}" if debt is not None else ""),
        (rev, rev is not None and rev >= QUALITY_MIN_REVENUE_GROWTH_PCT,
         f"revenue {rev:+.1f}%" if rev is not None else ""),
    )
    present = [c for c in quality_checks if c[0] is not None]
    if len(present) >= QUALITY_MIN_SIGNALS and all(c[1] for c in present):
        return "quality", ", ".join(c[2] for c in present)

    soft_risk = []
    if vol is not None and vol > RISK_VOL_PCT:
        soft_risk.append(f"vol {vol:.0f}% annualised")
    if drawdown is not None and drawdown <= RISK_DRAWDOWN_PCT:
        soft_risk.append(f"{drawdown:.0f}% from its 52w high")
    if soft_risk:
        return "speculative", ", ".join(soft_risk)

    growth = []
    if rev is not None and rev >= GROWTH_REVENUE_PCT:
        growth.append(f"revenue {rev:+.1f}%")
    if earn is not None and earn >= GROWTH_EARNINGS_PCT:
        growth.append(f"earnings {earn:+.1f}%")
    if growth:
        return "growth", ", ".join(growth)

    # Deliberately not a verdict. Everything downstream treats this as "we could
    # not say", which is why it is never blocked and never counted against a cap.
    return UNCLASSIFIED, ""


def style_of(feat: object) -> str:
    return classify(feat)[0]


def bucket_of(feat: object | None) -> str:
    """This dial's bucket for one feature row — the guardrails' uniform entry
    point across both mixes. A missing feature is `unclassified`, never a
    breach."""
    return style_of(feat)


def label_of(code: str) -> str:
    return SCHEME.label_of(code)


def styles_of(features: Mapping[str, object]) -> dict[str, str]:
    """{ticker: style code} for a feature set."""
    return {ticker: style_of(feat) for ticker, feat in features.items()}


def reasons_of(features: Mapping[str, object]) -> dict[str, str]:
    """{ticker: the figures behind its tag}, for a prompt that can be argued with."""
    return {ticker: classify(feat)[1] for ticker, feat in features.items()}


def options() -> list[dict]:
    return SCHEME.options()


def clean_targets(raw: Mapping[str, object] | None) -> dict[str, float]:
    return allocation.clean_targets(raw, SCHEME)


def clean_tolerance(raw: object, default: float = DEFAULT_TOLERANCE_PCT) -> float:
    return allocation.clean_tolerance(raw, default)


def ceilings(targets: Mapping[str, float], tolerance_pct: float) -> dict[str, float]:
    return allocation.ceilings(targets, tolerance_pct)


def excluded(targets: Mapping[str, float]) -> set[str]:
    """Styles the mix asks for none of.

    `unclassified` is never excluded from the *candidate list* even when it is
    targeted at 0: dropping every name whose fundamentals had not landed would
    hollow out the screen on a cache miss rather than on a finding. A zero there
    still caps at zero in the guardrails, where the book is measured.
    """
    return allocation.excluded(targets) - {UNCLASSIFIED}


def candidate_quotas(targets: Mapping[str, float], top_n: int) -> dict[str, int]:
    return allocation.candidate_quotas(targets, top_n)


def exposure(
    snap: PortfolioSnapshot, style_by_ticker: Mapping[str, str]
) -> dict[str, float]:
    return allocation.exposure(snap, style_by_ticker, UNCLASSIFIED)


def projected_exposure(
    snap: PortfolioSnapshot, actions: Iterable, style_by_ticker: Mapping[str, str]
) -> dict[str, float]:
    return allocation.projected_exposure(snap, actions, style_by_ticker, UNCLASSIFIED)


def candidate_counts(style_by_ticker: Mapping[str, str]) -> dict[str, int]:
    return allocation.bucket_counts(style_by_ticker)


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
    closing: bool = True,
) -> str:
    unclassified = (candidates or {}).get(UNCLASSIFIED, 0)
    extra = [
        "The tag is computed from reported fundamentals and realised volatility, "
        "not from judgement. Where you think it is wrong, argue the name on its "
        "merits in the thesis — but the ceiling is mechanical and a buy past it is "
        "rejected whatever the thesis says.",
    ]
    if unclassified:
        extra.append(
            f"{unclassified} candidate(s) below are tagged Unclassified: the figures "
            "to place them were missing, not contradictory. They count towards no "
            "target and are blocked by none, so the caps above under-count rather "
            "than bind tightly — read a style target as a tilt, not a guarantee."
        )
    return allocation.prompt_block(
        targets, tolerance_pct, current, SCHEME, candidates,
        extra_lines=extra, closing=closing,
    )
