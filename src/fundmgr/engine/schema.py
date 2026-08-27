from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Action(BaseModel):
    ticker: str = Field(description="Yahoo Finance ticker, e.g. VOLV-B.ST")
    side: Literal["buy", "sell", "hold"] = Field(description="Desired action this run")
    target_weight_pct: float = Field(
        ge=0, le=100,
        description=(
            "Desired portfolio weight after this trade, as % of NAV. "
            "For sells: 0 = full exit, partial sell = new lower weight. "
            "For holds: current weight (no trade occurs). "
            "Must respect mandate sizing rules: high conviction 10-15%, medium 5-9%, starter 3-5%."
        ),
    )
    sek_estimate: float = Field(
        ge=0,
        description="Approximate SEK value of the trade (0 for holds). Used for guardrail checks.",
    )
    confidence: float = Field(
        ge=0, le=1,
        description=(
            "Conviction level 0.0–1.0. "
            "Do not recommend buys below 0.40. "
            "0.75+ = high conviction, 0.55-0.74 = medium, 0.40-0.54 = starter."
        ),
    )
    thesis: str = Field(
        max_length=500,
        description=(
            "1–3 sentences. Required for buys and sells; optional for holds. "
            "Must answer: Why now? What is the edge? What would break this thesis?"
        ),
    )
    stop_loss_pct: float | None = Field(
        default=None, ge=0, le=50,
        description=(
            "Recommended for all buys: % decline from entry price that invalidates the thesis. "
            "Typical range 8–15%. Tighter for momentum plays, wider for value."
        ),
    )
    take_profit_pct: float | None = Field(
        default=None, ge=0, le=200,
        description="Optional upside target as % gain from entry. Helps define the risk/reward.",
    )

    @field_validator("ticker")
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        return v.upper()


class DecisionRun(BaseModel):
    run_id: str = Field(description="Unique run identifier, e.g. 2026-06-15-abc123")
    market_summary: str = Field(
        max_length=1000,
        description="2-3 sentence read of current market conditions relevant to the portfolio",
    )
    actions: list[Action] = Field(
        description="One entry per ticker you have a view on. Omit tickers with no view.",
        min_length=1,
    )
    cash_target_pct: float = Field(
        ge=0, le=100,
        description="Desired cash allocation as % of NAV after all trades",
    )
    notes: str = Field(
        default="",
        max_length=1000,
        description="Any concerns, data quality issues, or tickers you'd like added to the universe",
    )


class TargetReview(BaseModel):
    """Reassessment of a position that has reached its take-profit level.

    A target hit is a decision, not a notification: either the gain is banked,
    or the reason for holding on is a higher target that says so explicitly.
    """
    ticker: str = Field(description="The position under review")
    recommendation: Literal["sell", "trim", "raise", "hold"] = Field(
        description=(
            "sell = take the profit, exit the whole position; "
            "trim = bank part of the gain and let the rest run (set trim_pct and "
            "new_take_profit_pct); "
            "raise = conviction has grown, keep it all and set a higher "
            "new_take_profit_pct; "
            "hold = keep as-is at the current target (use sparingly — it leaves "
            "the target breached and re-alerting)."
        ),
    )
    confidence: float = Field(
        ge=0, le=1, description="Conviction in this recommendation, 0.0-1.0."
    )
    trim_pct: float | None = Field(
        default=None, ge=0, le=100,
        description="If recommendation=trim, what % of the current position to sell.",
    )
    new_take_profit_pct: float | None = Field(
        default=None, ge=0, le=300,
        description=(
            "Required for 'raise' and 'trim': the new take-profit level, as % gain "
            "from the original entry price. Must exceed the current level — this is "
            "the target the position will be measured against from now on."
        ),
    )
    what_changed: str = Field(
        max_length=400,
        description="What has materially changed (or not) since the most recent decision on this name.",
    )
    rationale: str = Field(
        max_length=600,
        description="1-4 sentences: why bank the gain now, or what justifies a higher target.",
    )

    @field_validator("ticker")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()


class StopReview(BaseModel):
    """Focused reassessment of a single position after its stop-loss is hit."""
    ticker: str = Field(description="The position under review")
    recommendation: Literal["exit", "trim", "hold", "add"] = Field(
        description=(
            "exit = sell the whole position; trim = reduce it; "
            "hold = keep as-is (the stop move was noise); "
            "add = conviction unchanged or improved, buy more."
        ),
    )
    confidence: float = Field(
        ge=0, le=1, description="Conviction in this recommendation, 0.0-1.0."
    )
    trim_pct: float | None = Field(
        default=None, ge=0, le=100,
        description="If recommendation=trim, what % of the current position to sell.",
    )
    what_changed: str = Field(
        max_length=400,
        description="What has materially changed (or not) since the most recent decision on this name.",
    )
    rationale: str = Field(
        max_length=600,
        description="1-4 sentences: why this call, weighing the recent thesis against the stop being hit.",
    )

    @field_validator("ticker")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()


class Lesson(BaseModel):
    """One lesson distilled from a batch of evaluated outcomes."""
    body: str = Field(
        max_length=400,
        description=(
            "At most 2 sentences. Name a signal that was checkable before entry and "
            "specific enough to change a future decision. 'Monitor macro indicators' "
            "changes nothing and is not a lesson."
        ),
    )
    tickers: list[str] = Field(
        description=(
            "The tickers from this batch that support the lesson. At least two — a "
            "pattern seen in a single 28-day return is noise, not a lesson."
        ),
    )

    @field_validator("tickers")
    @classmethod
    def _upper_all(cls, v: list[str]) -> list[str]:
        return [t.upper() for t in v]


class BatchLessons(BaseModel):
    """The lessons a single evaluation batch yields — often none."""
    lessons: list[Lesson] = Field(
        default_factory=list,
        description=(
            "Empty whenever the batch shows no repeated, thesis-level pattern. That is "
            "the expected result for most batches: an empty list is a better answer "
            "than a plausible story."
        ),
    )


class ThesisCheck(BaseModel):
    """Whether one decision's stated thesis came true, judged on evidence."""
    ticker: str = Field(description="The ticker this verdict is for")
    verdict: Literal["held", "broke", "unresolved"] = Field(
        description=(
            "held = the specific claim in the thesis demonstrably came true; "
            "broke = it demonstrably did not; "
            "unresolved = the evidence available does not settle it. "
            "unresolved is the correct answer whenever the claim is about something "
            "the evidence below cannot show, and it is the most common answer."
        ),
    )
    evidence: str = Field(
        max_length=300,
        description=(
            "The specific evidence the verdict rests on — a headline, a reported "
            "figure, a stated guidance change. For 'unresolved', say what would "
            "have settled it. Never cite the share price move as evidence: that is "
            "the outcome being explained, not evidence about the reasoning."
        ),
    )

    @field_validator("ticker")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()


class ThesisChecks(BaseModel):
    """Verdicts for a batch of matured decisions."""
    checks: list[ThesisCheck] = Field(default_factory=list)
