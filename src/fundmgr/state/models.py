from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class Position:
    ticker: str
    shares: float
    avg_cost_sek: float
    current_price_sek: float = 0.0
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def market_value_sek(self) -> float:
        return self.shares * self.current_price_sek

    @property
    def unrealised_pnl_sek(self) -> float:
        return (self.current_price_sek - self.avg_cost_sek) * self.shares

    @property
    def unrealised_pnl_pct(self) -> float:
        if self.avg_cost_sek == 0:
            return 0.0
        return (self.current_price_sek / self.avg_cost_sek - 1) * 100


@dataclass
class PortfolioSnapshot:
    positions: list[Position]
    cash_sek: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def equity_sek(self) -> float:
        return sum(p.market_value_sek for p in self.positions)

    @property
    def nav_sek(self) -> float:
        return self.equity_sek + self.cash_sek

    @property
    def cash_pct(self) -> float:
        if self.nav_sek == 0:
            return 100.0
        return self.cash_sek / self.nav_sek * 100

    def weight_pct(self, ticker: str) -> float:
        if self.nav_sek == 0:
            return 0.0
        for p in self.positions:
            if p.ticker == ticker:
                return p.market_value_sek / self.nav_sek * 100
        return 0.0


@dataclass
class Transaction:
    ticker: str
    side: Literal["buy", "sell"]
    shares: float
    price_sek: float
    fee_sek: float
    source: Literal["fill", "recommended"]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    id: int | None = None

    @property
    def gross_sek(self) -> float:
        return self.shares * self.price_sek

    @property
    def net_sek(self) -> float:
        if self.side == "buy":
            return -(self.gross_sek + self.fee_sek)
        return self.gross_sek - self.fee_sek


@dataclass
class NavPoint:
    date: str  # YYYY-MM-DD
    portfolio_nav_sek: float
    benchmark_value: float
    cash_sek: float

    @property
    def equity_sek(self) -> float:
        return self.portfolio_nav_sek - self.cash_sek


@dataclass
class RecommendationLog:
    run_id: str
    timestamp: datetime
    prompt_snapshot: str   # full JSON-serialised prompt context
    llm_response: str      # raw LLM text response
    guardrail_log: str     # JSON log of guardrail decisions
    actions_json: str      # final actions after guardrails applied
    sampling_log: str = "" # JSON: {requested, succeeded, failed, errors} per-run sample health
    id: int | None = None


@dataclass
class DecisionOutcome:
    """Retrospective evaluation of a single LLM action, filled in ~4 weeks later."""
    run_id: str
    ticker: str
    action: Literal["buy", "sell", "hold"]
    confidence: float | None = None
    price_at_decision: float | None = None
    price_at_evaluation: float | None = None
    benchmark_return_pct: float | None = None
    position_return_pct: float | None = None
    outperformed: bool | None = None
    evaluation_date: str | None = None
    thesis: str | None = None
    id: int | None = None

    @property
    def was_correct(self) -> bool | None:
        """A buy/sell is 'correct' if it outperformed the benchmark; hold is correct if it would have underperformed."""
        if self.outperformed is None:
            return None
        if self.action in ("buy",):
            return self.outperformed
        if self.action == "sell":
            return not self.outperformed  # selling was good if the stock then underperformed
        return None  # hold is ambiguous


@dataclass
class Learning:
    """A distilled lesson injected into future prompts to improve LLM decisions."""
    category: str           # calibration | sector_bias | timing | general
    body: str               # plain-text lesson (≤3 sentences)
    run_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    superseded_by: int | None = None
    id: int | None = None
