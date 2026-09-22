"""Frozen, paired prompt experiments. Never books fills or promotes guidance.

Decisions are recorded before outcome data is supplied to the offline scorer.
The production prompt assembly, consensus and guardrails are reused. Execution
is an explicit buy-and-hold research convention, not a broker-fill simulation.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from fundmgr import regions, styles
from fundmgr.config import AppConfig, FeeConfig, LLMConfig, RiskConfig
from fundmgr.data.prices import TickerFeatures
from fundmgr.engine.client import call_llm_consensus
from fundmgr.engine.prompt import assemble_system_prompt
from fundmgr.guardrails.rules import apply_guardrails
from fundmgr.state.models import PortfolioSnapshot, Position

VERSION = 1
CONVENTION = "fractional-shares_at-frozen-quotes_fees_buy-and-hold_v1"
FORWARD_CONVENTION = "fractional-shares_at-next-common-close_fees_total-return_v1"
Number = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    version: Literal[1] = VERSION
    fund_id: str
    decision_time: datetime
    horizon_days: int = Field(gt=0)
    benchmark: str
    basis: Literal["SEK", "synthetic_native"]
    mandate: str
    guidance: str
    system: str
    user: str
    llm: LLMConfig
    risk: RiskConfig
    fees: FeeConfig
    cash: Number
    positions: list[Position]
    features: dict[str, TickerFeatures]
    universe: list[str]
    fx_rates: dict[str, Positive]

    def config(self) -> AppConfig:
        return AppConfig(llm=self.llm, risk=self.risk, fees=self.fees,
                         fx_to_sek=self.basis == "SEK", benchmark=self.benchmark,
                         evaluation_horizon_days=self.horizon_days)

    def snapshot(self) -> PortfolioSnapshot:
        return PortfolioSnapshot(positions=self.positions, cash_sek=self.cash)

    def price(self, ticker: str) -> float:
        feat = self.features[ticker]
        rate = 1.0 if self.basis == "synthetic_native" or feat.currency == "SEK" else self.fx_rates.get(feat.currency)
        if rate is None:
            raise ValueError(f"Missing decision-time FX for {ticker} ({feat.currency})")
        value = feat.last_price * rate
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid decision-time mark for {ticker}")
        return value


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def capture_case(cfg, snap, features, universe, fields, system, user, fx_rates) -> dict:
    """Archive actual inputs without fetching data or evaluating future returns."""
    return {
        "version": VERSION, "fund_id": cfg.db_path.stem,
        "decision_time": snap.timestamp.replace(tzinfo=timezone.utc).isoformat()
            if snap.timestamp.tzinfo is None else snap.timestamp.isoformat(),
        "horizon_days": cfg.evaluation_horizon_days, "benchmark": cfg.benchmark,
        "basis": "SEK" if cfg.fx_to_sek else "synthetic_native",
        "mandate": fields["mandate"], "guidance": fields.get("guidance", ""),
        "system": system, "user": user,
        "llm": asdict(cfg.llm), "risk": asdict(cfg.risk), "fees": asdict(cfg.fees),
        "cash": snap.cash_sek,
        "positions": [{"ticker": p.ticker, "shares": p.shares,
                       "avg_cost_sek": p.avg_cost_sek, "current_price_sek": p.current_price_sek,
                       "updated_at": p.updated_at.isoformat()} for p in snap.positions],
        "features": {t: asdict(f) for t, f in features.items()},
        "universe": sorted(universe), "fx_rates": dict(fx_rates),
    }


def load_case(snapshot: dict) -> Case:
    if not snapshot.get("evaluation_case"):
        raise ValueError("Run lacks frozen evaluation context; collect a new run, not today's replacement data")
    case = Case.model_validate(snapshot["evaluation_case"])
    if case.decision_time.tzinfo is None:
        raise ValueError("Decision time must have a timezone")
    if assemble_system_prompt(case.mandate, case.guidance) != case.system:
        raise ValueError("Frozen system prompt does not match the shared prompt builder")
    if snapshot.get("system_message") != case.system or snapshot.get("user_message") != case.user:
        raise ValueError("Frozen evaluation inputs disagree with the recorded prompt")
    digest(case.model_dump(mode="json"))  # Reject non-finite dataclass inputs, too.
    if case.llm.n_samples < 1 or case.snapshot().nav_sek <= 0:
        raise ValueError("Invalid sample count or portfolio NAV")
    held = set()
    for p in case.positions:
        if p.ticker in held or p.shares < 0 or p.ticker not in case.features:
            raise ValueError("Missing/duplicate/invalid held position in frozen features")
        held.add(p.ticker)
        if not math.isclose(p.current_price_sek, case.price(p.ticker), rel_tol=1e-6):
            raise ValueError(f"Position and quote/FX marks disagree for {p.ticker}")
    for ticker, feat in case.features.items():
        if ticker != feat.ticker:
            raise ValueError("Feature ticker identity mismatch")
        case.price(ticker)  # Preflight every alternative before paying for calls.
    return case


def project_allocation(case: Case, actions: list[dict]) -> dict:
    """Project approved targets with fees and cumulative feasibility checks.

    All target weights use decision NAV, matching the guardrails' convention.
    Sells release cash before buys. Omitted/held names remain owned; cash earns
    zero. Fractional shares isolate allocation quality from lot-size artifacts.
    An infeasible batch is reported invalid, never silently treated as cash.
    """
    cfg, snap = case.config(), case.snapshot()
    nav = snap.nav_sek
    shares = {p.ticker: p.shares for p in case.positions if p.shares > 0}
    cash, fees, turnover = case.cash, 0.0, 0.0
    seen, buys, trades = set(), set(), []
    for action in sorted(actions, key=lambda a: (a["side"] != "sell", a["ticker"])):
        ticker, side = action["ticker"], action["side"]
        if ticker in seen:
            raise ValueError(f"Duplicate action for {ticker}")
        seen.add(ticker)
        if side == "hold":
            continue
        if ticker not in case.features:
            raise ValueError(f"No frozen price for {ticker}")
        price = case.price(ticker)
        current = shares.get(ticker, 0.0)
        target = nav * action["target_weight_pct"] / 100 / price
        delta = target - current
        if side == "sell" and (current <= 0 or delta >= 0):
            raise ValueError(f"Sell is not a reduction of an owned position: {ticker}")
        if side == "buy" and delta <= 0:
            raise ValueError(f"Buy does not increase the position: {ticker}")
        gross = abs(delta) * price
        if gross + 1e-8 < cfg.risk.min_trade_sek:
            raise ValueError(f"Actual trade is below the minimum: {ticker}")
        fee = cfg.fees.calc(gross)
        if not math.isfinite(fee) or fee < 0:
            raise ValueError("Invalid fee configuration")
        cash -= delta * price + fee
        if cash < -1e-7:
            raise ValueError("Projected batch spends unavailable cash")
        shares[ticker] = target
        fees += fee
        turnover += gross
        trades.append({"ticker": ticker, "side": side, "shares": abs(delta),
                       "price": price, "gross": gross, "fee": fee})
        if side == "buy":
            buys.add(ticker)
    shares = {t: q for t, q in shares.items() if q > 1e-10}
    values = {t: q * case.price(t) for t, q in shares.items()}
    eps = 1e-7
    if turnover > nav * cfg.risk.max_turnover_pct / 100 + eps:
        raise ValueError("Actual turnover exceeds the limit")
    if buys:
        if cash < nav * cfg.risk.min_cash_pct / 100 - eps:
            raise ValueError("Cumulative buys breach the cash floor after fees")
        if len(shares) > cfg.risk.max_positions:
            raise ValueError("Cumulative buys exceed the position count limit")
        for ticker in buys:
            if values[ticker] > nav * cfg.risk.max_position_pct / 100 + eps:
                raise ValueError(f"Position cap exceeded: {ticker}")
            sector = case.features[ticker].sector
            if not sector:
                raise ValueError(f"Cannot verify sector cap for buy: {ticker}")
            if any(not case.features[t].sector for t in shares):
                raise ValueError("Cannot verify sector cap with unclassified holdings")
            sector_value = sum(v for t, v in values.items() if case.features[t].sector == sector)
            if sector_value > nav * cfg.risk.max_sector_pct / 100 + eps:
                raise ValueError(f"Cumulative sector cap exceeded: {sector}")
        for dial, attr in ((regions, "region"), (styles, "style")):
            ceilings = dial.ceilings(getattr(cfg.risk, f"{attr}_targets"),
                                     getattr(cfg.risk, f"{attr}_tolerance_pct"))
            for bucket in {dial.bucket_of(case.features[t]) for t in buys}:
                if bucket not in ceilings:
                    continue
                value = sum(v for t, v in values.items() if dial.bucket_of(case.features[t]) == bucket)
                if value > nav * ceilings[bucket] / 100 + eps:
                    raise ValueError(f"Cumulative {attr} cap exceeded: {bucket}")
    return {"shares": shares, "cash": cash, "fees": fees, "turnover": turnover,
            "initial_nav": nav, "post_trade_nav": sum(values.values()) + cash, "trades": trades}


def validate_candidate(case: Case, candidate: dict) -> datetime:
    """Validate lineage before reserving a run or making paid calls."""
    expected = {"fund_id": case.fund_id, "task_model": case.llm.model_id,
                "provider": case.llm.provider, "horizon_days": case.horizon_days,
                "mandate": case.mandate}
    for key, value in expected.items():
        if candidate.get(key) != value:
            raise ValueError(f"Candidate {key} does not match the frozen case")
    incumbent_hash = hashlib.sha256(case.guidance.encode()).hexdigest()[:12] if case.guidance else None
    if candidate.get("incumbent_guidance_hash") != incumbent_hash:
        raise ValueError("Candidate was compiled against a different incumbent")
    if candidate.get("status") != "pending_evaluation" or not str(candidate.get("instructions", "")).strip():
        raise ValueError("Expected an inactive candidate with instructions")
    candidate_created = datetime.fromisoformat(candidate["created_at"])
    if candidate_created.tzinfo is None:
        raise ValueError("Candidate timestamp must have a timezone")
    return candidate_created


def compare_guidance(snapshot: dict, candidate: dict) -> dict:
    """Call both arms on identical frozen input. No future labels are accepted."""
    case = load_case(snapshot)
    candidate_created = validate_candidate(case, candidate)
    candidate_hash = digest(candidate)
    start = datetime.now(timezone.utc)
    if case.decision_time > start:
        raise ValueError("Cannot compare a snapshot dated in the future")
    arms = {}
    for name, instructions in (("incumbent", case.guidance), ("candidate", candidate["instructions"])):
        arm = {"system": assemble_system_prompt(case.mandate, instructions)}
        try:
            decision, raw, votes, sampling = call_llm_consensus(arm["system"], case.user, case.config())
            arm.update(decision=decision.model_dump(), response=raw, votes=votes, sampling=sampling)
            if sampling.get("failed") or sampling.get("succeeded") != case.llm.n_samples:
                raise ValueError("Incomplete consensus sample set; comparison is not scoreable")
            verdicts = apply_guardrails(decision, case.snapshot(), case.features, set(case.universe), case.config())
            approved = [a.model_dump() for a in verdicts.approved_actions]
            arm.update(approved=approved, guardrails=verdicts.to_log())
            arm["allocation"] = project_allocation(case, approved)
            arm["status"] = "ready"
        except Exception as exc:
            arm.update(status="invalid", error=f"{type(exc).__name__}: {exc}")
        arms[name] = arm
    completed = datetime.now(timezone.utc)
    # Retrospective replay can diagnose implementation but cannot earn promotion.
    forward = (candidate_created <= case.decision_time
               and start.date() == case.decision_time.astimezone(timezone.utc).date()
               and completed.date() == start.date())
    payload = {"version": VERSION, "case": case.model_dump(mode="json"),
               "case_hash": digest(case.model_dump(mode="json")), "candidate": candidate,
               "candidate_hash": candidate_hash, "started_at": start.isoformat(),
               "completed_at": completed.isoformat(), "arms": arms,
               "mode": "same_day_shadow" if forward else "retrospective_diagnostic",
               "execution_convention": CONVENTION, "promotion_eligible": False}
    return {**payload, "comparison_hash": digest(payload)}


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: date
    prices: dict[str, Number]
    benchmark: Positive


class Outcomes(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_hash: str
    basis: Literal["SEK", "synthetic_native"]
    benchmark: str
    source: str = Field(min_length=1)
    observations: list[Observation] = Field(min_length=2)


def score_comparison(comparison: dict, outcomes: Outcomes) -> dict:
    """Deterministic net portfolio returns on shared daily closing valuations.

    Require all calendar dates (carry non-trading closes forward explicitly)
    and all shown candidates. No missing mark is silently valued as zero or
    excluded, even when that candidate wasn't picked by either arm.
    """
    payload = {k: v for k, v in comparison.items() if k != "comparison_hash"}
    if comparison.get("comparison_hash") != digest(payload):
        raise ValueError("Comparison was modified after decisions were recorded")
    case = Case.model_validate(comparison["case"])
    if outcomes.case_hash != comparison["case_hash"] or digest(case.model_dump(mode="json")) != outcomes.case_hash:
        raise ValueError("Outcomes belong to a different frozen case")
    if outcomes.basis != case.basis or outcomes.benchmark != case.benchmark:
        raise ValueError("Outcome currency convention or benchmark mismatch")
    if comparison.get("execution_convention") not in {CONVENTION, FORWARD_CONVENTION}:
        raise ValueError("Unsupported execution convention")
    start = case.decision_time.astimezone(timezone.utc).date()
    end = start + timedelta(days=case.horizon_days)
    if end > datetime.now(timezone.utc).date():
        raise ValueError("Outcome horizon has not matured")
    expected_dates = [start + timedelta(days=i) for i in range(case.horizon_days + 1)]
    if [p.date for p in outcomes.observations] != expected_dates:
        raise ValueError("Provide one ordered valuation for every calendar day of the exact horizon")
    required = set(case.features)
    for point in outcomes.observations:
        if set(point.prices) != required:
            raise ValueError(f"Incomplete or extra candidate marks on {point.date}")
    for ticker, price in outcomes.observations[0].prices.items():
        if not math.isclose(price, case.price(ticker), rel_tol=1e-6):
            raise ValueError(f"Starting mark changed for {ticker}")
    if set(comparison["arms"]) != {"incumbent", "candidate"}:
        raise ValueError("Both arms are required")
    scores = {}
    benchmark_return = (outcomes.observations[-1].benchmark / outcomes.observations[0].benchmark - 1) * 100
    for name, arm in comparison["arms"].items():
        if arm["status"] != "ready":
            raise ValueError(f"{name} decision is invalid; cannot compare performance")
        plan = arm["allocation"]
        nav = plan["initial_nav"]
        values = [plan["post_trade_nav"]] + [
            plan["cash"] + sum(q * point.prices[t] for t, q in plan["shares"].items())
            for point in outcomes.observations[1:]
        ]
        peak, drawdown = nav, 0.0
        for value in values:
            peak = max(peak, value)
            drawdown = max(drawdown, (1 - value / peak) * 100)
        net_return = (values[-1] / nav - 1) * 100
        scores[name] = {"net_return_pct": net_return, "excess_return_pp": net_return - benchmark_return,
                        "max_daily_drawdown_pct": drawdown, "fees": plan["fees"],
                        "turnover_pct": plan["turnover"] / nav * 100, "terminal_nav": values[-1]}
    return {"version": VERSION, "comparison_hash": comparison["comparison_hash"],
            "candidate_hash": comparison["candidate_hash"], "case_hash": outcomes.case_hash,
            "outcomes_hash": digest(outcomes.model_dump(mode="json")), "source": outcomes.source,
            "decision_date": start.isoformat(), "horizon_end": end.isoformat(),
            "horizon_days": case.horizon_days, "basis": case.basis,
            "benchmark_return_pct": benchmark_return, "scores": scores,
            "candidate_advantage_pp": scores["candidate"]["net_return_pct"] - scores["incumbent"]["net_return_pct"],
            "mode": comparison["mode"], "promotion_eligible": False,
            "limitation": "One buy-and-hold case is not promotion evidence; independent forward periods are required."}


def write_report(path: Path, payload: dict) -> None:
    """Never overwrite prior decisions or scores. Serialize before creating."""
    import os
    import tempfile

    encoded = json.dumps(payload, indent=2, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            file.write(encoded + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.link(temporary, path)  # Atomic publication, fails if destination exists.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
