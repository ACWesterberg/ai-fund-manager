from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Action(BaseModel):
    ticker: str = Field(description="Yahoo Finance ticker, e.g. VOLV-B.ST")
    side: Literal["buy", "sell", "hold"] = Field(description="Desired action this week")
    target_weight_pct: float = Field(
        ge=0, le=100,
        description="Desired portfolio weight after this trade, as % of NAV (0-100)",
    )
    sek_estimate: float = Field(
        ge=0,
        description="Approximate SEK value of the trade (0 for holds)",
    )
    confidence: float = Field(
        ge=0, le=1,
        description="Conviction level 0-1. Only recommend if genuinely above 0.4.",
    )
    thesis: str = Field(
        max_length=500,
        description="1-3 sentence rationale. Why now? What is the edge?",
    )
    stop_loss_pct: float | None = Field(
        default=None, ge=0, le=50,
        description="Optional: % drop from entry that would invalidate the thesis",
    )
    take_profit_pct: float | None = Field(
        default=None, ge=0, le=200,
        description="Optional: % gain target",
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
