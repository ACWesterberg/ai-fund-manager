"""Offline evidence audit and descriptive aggregation of forward experiments.

No network/model calls, statistical significance claims, or promotion decisions.
Non-overlapping windows reduce repeated exposure; they do not prove independence.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median

from fundmgr.engine.experiments import (
    FORWARD_CONVENTION, digest, load_case, score_comparison, validate_candidate,
)
from fundmgr.engine.forward import _check_hash, derive_forward, execution_date


def _read(path):
    return json.loads(path.read_text())


def audit_case(directory: Path) -> dict:
    """Recompute results from archived histories; never trust summary numbers."""
    record = {"path": str(directory.resolve()), "status": "invalid", "group": None}
    try:
        registration = _read(directory / "registration.json")
        _check_hash(registration, "registration_hash")
        case = load_case(registration["snapshot"])
        created = validate_candidate(case, registration["candidate"])
        registered = datetime.fromisoformat(registration["registered_at"])
        if (not created <= case.decision_time <= registered
                or registered - case.decision_time >= timedelta(hours=2)
                or registration.get("execution_convention") != FORWARD_CONVENTION):
            raise ValueError("Candidate/registration chronology is invalid")
        identity = {
            "fund_id": case.fund_id, "candidate_hash": digest(registration["candidate"]),
            "incumbent_hash": digest(case.guidance), "mandate_hash": digest(case.mandate),
            "model": case.model_dump(mode="json")["llm"],
            "risk": case.model_dump(mode="json")["risk"],
            "fees": case.model_dump(mode="json")["fees"],
            "benchmark": case.benchmark, "basis": case.basis, "horizon_days": case.horizon_days,
            "benchmark_currency": registration["benchmark_currency"],
            "benchmark_calendar": registration["benchmark_calendar"],
            "execution_convention": registration["execution_convention"],
        }
        record.update(group=digest(identity), identity=identity,
                      registration_hash=registration["registration_hash"],
                      run_id=registration["run_id"],
                      decision_time=case.decision_time.astimezone(timezone.utc).isoformat())
        if not (directory / "comparison.json").exists():
            record.update(status="recording_incomplete", error="No completed comparison")
            return record
        pair = _read(directory / "comparison.json")
        _check_hash(pair, "comparison_hash")
        if (pair.get("registration_hash") != registration["registration_hash"]
                or pair["case_hash"] != digest(case.model_dump(mode="json"))
                or pair["candidate_hash"] != identity["candidate_hash"]
                or pair["case"] != case.model_dump(mode="json")
                or pair["candidate"] != registration["candidate"]
                or pair.get("mode") != "registered_forward"):
            raise ValueError("Comparison lineage mismatch")
        started, completed = (datetime.fromisoformat(pair[k]) for k in ("started_at", "completed_at"))
        if not registered <= started <= completed:
            raise ValueError("Comparison predates registration")
        start = execution_date(registration, pair)
        end = start + timedelta(days=case.horizon_days)
        record.update(start=start.isoformat(), end=end.isoformat(), comparison_hash=pair["comparison_hash"])
        if set(pair["arms"]) != {"incumbent", "candidate"} or any(a["status"] != "ready" for a in pair["arms"].values()):
            record.update(status="invalid_decision", error="Incomplete or infeasible model arms")
            return record
        if end >= datetime.now(timezone.utc).date():
            record["status"] = "waiting"
            return record
        if not (directory / "result.json").exists():
            record.update(status="missing_outcomes", error="Matured comparison lacks a completed score")
            return record
        result = _read(directory / "result.json")
        match = None
        for history_path in sorted(directory.glob("collections/*/history.json")):
            bundle = _read(history_path)
            if digest(bundle) == result.get("history_hash"):
                match = (history_path.parent, bundle)
                break
        if match is None:
            raise ValueError("Score's archived history is missing or modified")
        attempt, bundle = match
        derived, outcomes = derive_forward(registration, pair, bundle)
        if _read(attempt / "valuation_case.json") != derived:
            raise ValueError("Archived valuation case differs from reconstructed allocations")
        if _read(attempt / "outcomes.json") != outcomes.model_dump(mode="json"):
            raise ValueError("Archived outcomes differ from reconstructed history")
        score = score_comparison(derived, outcomes)
        expected = dict(score, original_comparison_hash=pair["comparison_hash"],
                        registration_hash=registration["registration_hash"], history_hash=digest(bundle),
                        execution_convention=registration["execution_convention"],
                        recorded_decision_time=case.decision_time.isoformat())
        if any(result.get(key) != value for key, value in expected.items()):
            raise ValueError("Stored score differs from recomputed evidence")
        record.update(status="scored", score=score, history_hash=digest(bundle))
    except Exception as exc:
        record.update(status="invalid", error=f"{type(exc).__name__}: {exc}")
    return record


def summarize(records: list[dict], min_periods: int = 8) -> list[dict]:
    """Select non-overlapping periods by time BEFORE inspecting their scores.

    Failed/missing periods reserve their windows too, preventing selection of a
    convenient overlapping winner in their place. Earliest entry wins; ties use
    decision time and registration hash. No aggregate compounded return is made.
    """
    if min_periods < 2:
        raise ValueError("min_periods must be at least 2")
    groups = defaultdict(list)
    for record in records:
        if record.get("group"):
            groups[record["group"]].append(record)
    reports = []
    for group, members in sorted(groups.items()):
        selected, overlaps, last_end = [], [], None
        eligible = sorted((r for r in members if r.get("start") and r["status"] != "duplicate"),
                          key=lambda r: (r["start"], r["decision_time"], r["registration_hash"]))
        for record in eligible:
            if last_end is None or record["start"] > last_end:
                selected.append(record)
                last_end = record["end"]
            else:
                overlaps.append(record["registration_hash"])
        matured = [r for r in selected if r["end"] < datetime.now(timezone.utc).date().isoformat()]
        scored = [r for r in matured if r["status"] == "scored"]
        problems = [r for r in members if r["status"] not in {"scored", "waiting", "duplicate"}]
        metrics = None
        if scored:
            advantages = [r["score"]["candidate_advantage_pp"] for r in scored]
            metrics = {"mean_advantage_pp": mean(advantages), "median_advantage_pp": median(advantages),
                       "worst_advantage_pp": min(advantages), "wins": sum(a > 0 for a in advantages),
                       "ties": sum(a == 0 for a in advantages), "win_fraction": sum(a > 0 for a in advantages) / len(advantages)}
            for arm in ("candidate", "incumbent"):
                values = [r["score"]["scores"][arm] for r in scored]
                metrics[arm] = {"mean_net_return_pct": mean(v["net_return_pct"] for v in values),
                    "mean_excess_return_pp": mean(v["excess_return_pp"] for v in values),
                    "worst_period_return_pct": min(v["net_return_pct"] for v in values),
                    "mean_drawdown_pct": mean(v["max_daily_drawdown_pct"] for v in values),
                    "worst_drawdown_pct": max(v["max_daily_drawdown_pct"] for v in values),
                    "mean_turnover_pct": mean(v["turnover_pct"] for v in values),
                    "mean_fees": mean(v["fees"] for v in values)}
            metrics["mean_drawdown_change_pp"] = metrics["candidate"]["mean_drawdown_pct"] - metrics["incumbent"]["mean_drawdown_pct"]
            metrics["mean_turnover_change_pp"] = metrics["candidate"]["mean_turnover_pct"] - metrics["incumbent"]["mean_turnover_pct"]
        status = "insufficient_periods"
        if problems:
            status = "incomplete_evidence"
        elif len(scored) >= min_periods:
            status = "mixed_results"
            if (metrics["mean_advantage_pp"] > 0 and metrics["median_advantage_pp"] > 0
                    and metrics["win_fraction"] > .5 and metrics["mean_drawdown_change_pp"] <= 0
                    and metrics["candidate"]["worst_drawdown_pct"] <= metrics["incumbent"]["worst_drawdown_pct"]
                    and metrics["mean_turnover_change_pp"] <= 0):
                status = "promising_descriptive"
        reports.append({"group": group, "identity": members[0]["identity"], "status": status,
            "counts": dict(Counter(r["status"] for r in members)), "registered_cases": len(members),
            "selected_periods": len(selected), "matured_selected_periods": len(matured),
            "scored_selected_periods": len(scored), "coverage": len(scored) / len(matured) if matured else None,
            "selected_registrations": [r["registration_hash"] for r in selected],
            "overlapping_registrations": overlaps, "metrics": metrics, "promotion_eligible": False})
    return reports


def aggregate_evidence(roots: list[Path], min_periods: int = 8) -> dict:
    """Scan full fund experiment directories, including unscored reservations."""
    if min_periods < 2:
        raise ValueError("min_periods must be at least 2")
    directories = set()
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"Experiment root does not exist: {root}")
        directories.update(p.resolve() for p in root.iterdir() if p.is_dir())
    records = [audit_case(p) for p in sorted(directories)]
    seen = {}
    for record in records:
        # Copies of an experiment must never multiply its contribution.
        identity = (record.get("group"), record.get("run_id"))
        if record.get("group"):
            if identity in seen:
                first = seen[identity]
                keys = ("registration_hash", "comparison_hash", "history_hash", "status", "score")
                if all(first.get(k) == record.get(k) for k in keys):
                    record.update(status="duplicate", error="Repeated fund/candidate run identity")
                else:
                    record.update(status="invalid", error="Conflicting copies of the same experiment")
                    first.update(status="invalid", error="Conflicting copies of the same experiment")
            else:
                seen[identity] = record
    reports = summarize(records, min_periods)
    payload = {"version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
               "roots": sorted(str(r.resolve()) for r in roots), "min_periods": min_periods,
               "selection": "earliest entry; strictly non-overlapping inclusive date windows; failed windows retained",
               "counts": dict(Counter(r["status"] for r in records)), "groups": reports, "cases": records,
               "promotion_eligible": False,
               "limitations": ["Non-overlap does not establish statistical independence or significance.",
                   "Minimum period count is an exploratory threshold, not a validated promotion rule.",
                   "Each fund/candidate/incumbent/config/horizon/basis is evaluated separately; no pooled return.",
                   "Coverage refers only to supplied roots; deleted or omitted experiments cannot be detected.",
                   "Promising is descriptive only; no automatic promotion or guidance changes."]}
    return {**payload, "report_hash": digest(payload)}
