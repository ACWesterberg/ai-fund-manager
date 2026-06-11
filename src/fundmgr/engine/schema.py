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
