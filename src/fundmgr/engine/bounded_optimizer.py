"""Small instruction-only searches with durable, conservative request budgets.

No DSPy bootstrapping, hidden retries, or zero-scored API failures. Reservations
are deliberately not refunded: a timed-out request may still have been billed.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import tempfile
from dataclasses import asdict
from contextlib import contextmanager
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from pydantic import BaseModel, Field

from fundmgr.engine.client import call_llm, _schema_hint
from fundmgr.engine.experiments import digest
from fundmgr.engine.prompt import assemble_system_prompt
from fundmgr.engine.schema import DecisionRun

logger = logging.getLogger(__name__)
VERSION = "bounded_instructions_v1"


class Proposal(BaseModel):
    instructions: str = Field(min_length=1, max_length=4000,
        description="Concise decision guidance, at most 4000 characters; no examples or output schema changes")


class SearchStopped(RuntimeError):
    pass


def _save(path: Path, state: dict):
    payload = {**state, "checksum": digest(state)}
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
            tmp = Path(file.name)
            json.dump(payload, file, indent=2, allow_nan=False)
            file.flush()
            os.fsync(file.fileno())
        tmp.replace(path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def _load(path: Path):
    state = json.loads(path.read_text())
    checksum = state.pop("checksum")
    if checksum != digest(state):
        raise ValueError("Checkpoint checksum mismatch")
    return state


def identity(cfg):
    from fundmgr.engine.optimizer import load_guidance
    return {"version": VERSION, "fund": cfg.db_path.stem, "llm": asdict(cfg.llm),
            "config_hash": cfg.config_hash(), "mandate": cfg.mandate_path.read_text().strip(),
            "guidance": load_guidance(cfg), "horizon": cfg.evaluation_horizon_days,
            "prompt_model": cfg.optimizer.prompt_model_id or cfg.llm.model_id,
            "output_tokens": cfg.optimizer.max_output_tokens,
            "reasoning_effort": cfg.optimizer.reasoning_effort,
            "validation_runs": cfg.optimizer.validation_runs}


def reservation(system, user, schema, output_tokens):
    # Byte-based upper estimate for text tokenization, plus the schema sent both
    # in text and structured output, and a generous protocol allowance. This is
    # an admission-control unit, not measured usage or a guaranteed dollar cap.
    return len((system + user + 2 * _schema_hint(schema)).encode("utf-8")) + 8192 + output_tokens


def make_plan(cfg, examples):
    from fundmgr.engine.optimizer import INPUT_FIELDS, METRIC_VERSION
    from fundmgr.engine.research_costs import evidence_periods
    if cfg.optimizer.validation_runs < 1 or cfg.optimizer.max_output_tokens < 1:
        raise ValueError("Validation run count and output token limit must be positive")
    ordered = sorted(examples, key=lambda e: (e.get("run_id", ""), e.get("source", "")))
    split = max(1, int(len(ordered) * .8))
    train, val = ordered[:split], ordered[split:][:cfg.optimizer.validation_runs]
    if not val:
        raise ValueError("Need separate training and validation runs")
    ident = identity(cfg)
    # Training-only summaries keep proposal cost bounded. Validation labels and
    # contexts never enter the instruction-writing request.
    summaries = [{"run_id": e.get("run_id"), "source": e.get("source"),
                  "alpha": sorted(e["ticker_alphas"].items())[:20],
                  "learnings": e.get("learnings", "")[:1200]} for e in train[-6:]]
    proposal_system = (
        "Write one concise improvement to an investment decision prompt using the supplied training summaries. "
        "These are noisy directional outcomes, not causal proof. Preserve the mandate and runtime risk limits. "
        "Do not invent side labels, change field meanings, or include demonstrations. "
        "Only buy/sell/hold are valid; target_weight_pct is post-trade position weight; "
        "sek_estimate is trade value, not position value. Return instructions as structured JSON."
    )
    proposal_user = json.dumps({"mandate": ident["mandate"], "current_guidance": ident["guidance"],
                                "training_summaries": summaries}, ensure_ascii=False)
    cases = [{"run_id": e.get("run_id"), "source": e.get("source", ""),
              "fields": {k: e[k] for k in INPUT_FIELDS}, "ticker_alphas": e["ticker_alphas"]} for e in val]
    return {"identity": ident, "metric_version": METRIC_VERSION,
            "evidence_periods": evidence_periods(ordered, cfg.db_path.stem), "proposal_system": proposal_system,
            "proposal_user": proposal_user, "cases": cases,
            "training_runs": [{"run_id": e.get("run_id"), "source": e.get("source", "")} for e in train]}


def task_input(case):
    return case.get("search_input", raw_task_input(case))


def raw_task_input(case):
    return "\n\n".join(case["fields"][k] for k in ("macro", "portfolio_state", "risk_limits", "universe", "learnings"))


def plan_cost(plan):
    ident = plan["identity"]
    output = ident["output_tokens"]
    if plan.get("context_mode") == "compare":
        total = sum(reservation(assemble_system_prompt(case["fields"]["mandate"], ident["guidance"]),
                                user, DecisionRun, output)
                    for case in plan["cases"]
                    for user in (raw_task_input(case), case["compact_input"]))
        return {"planned_calls": 2 * len(plan["cases"]), "reserved_token_estimate": total,
                "max_output_tokens_per_call": output, "max_output_tokens_total": 2 * len(plan["cases"]) * output}
    total = reservation(plan["proposal_system"], plan["proposal_user"], Proposal, output)
    for case in plan["cases"]:
        for guidance in (ident["guidance"], "x" * 16000):  # max 4000 Unicode chars -> <=16000 UTF-8 bytes
            total += reservation(assemble_system_prompt(case["fields"]["mandate"], guidance),
                                 task_input(case), DecisionRun, output)
    return {"planned_calls": 1 + 2 * len(plan["cases"]), "reserved_token_estimate": total,
            "max_output_tokens_per_call": output,
            "max_output_tokens_total": (1 + 2 * len(plan["cases"])) * output}


def checkpoint_path(cfg, plan):
    return cfg.optimizer.compiled_dir / "searches" / cfg.db_path.stem / (digest(plan) + ".json")


@contextmanager
def _cache_lock(path, enabled):
    if not enabled:
        yield
        return
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SearchStopped("Identical evaluation is running in another search; resume later") from exc
        yield


def run_search(cfg, plan, path: Path, *, resume=False, retry_failed=False, force_search=False):
    import fcntl
    from fundmgr.engine.optimizer import decision_metric, save_guidance_candidate, guidance_versions

    if cfg.optimizer.max_calls < 1 or cfg.optimizer.max_total_tokens < 1:
        raise ValueError("Call and token budgets must be positive")
    if plan["identity"] != identity(cfg):
        raise ValueError("Checkpoint fund, guidance, model or search settings changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / "fund.lock").open("a") as fund_lock, path.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(fund_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SearchStopped("This fund optimization is already running") from exc
        if path.exists():
            state = _load(path)
            if state["plan"] != plan:
                raise ValueError("Checkpoint plan mismatch")
            if state["status"] == "complete":
                logger.info("Search already complete: %s", state.get("candidate_path") or "incumbent retained")
                return bool(state.get("candidate_path"))
            if not resume:
                raise SearchStopped(f"Unfinished search exists. Resume with --resume {path}; no calls made")
        else:
            if resume:
                raise ValueError("Resume checkpoint does not exist")
            from fundmgr.engine.research_costs import evidence_gate
            gate = evidence_gate(cfg, plan)
            if plan.get("context_mode") != "compare" and not force_search and not gate["eligible"]:
                logger.info("Search skipped: %d new own-fund decision dates; need %d. No paid calls.",
                            len(gate["new_periods"]), gate["required"])
                return False
            cost = plan_cost(plan)
            if cost["planned_calls"] > cfg.optimizer.max_calls or cost["reserved_token_estimate"] > cfg.optimizer.max_total_tokens:
                raise SearchStopped(f"Plan exceeds budget: {cost}. Inspect --dry-run or explicitly change limits")
            state = {"version": VERSION, "plan": plan, "status": "running", "attempts": [], "results": {},
                     "evidence_gate": gate, "forced_search": force_search}
            _save(path, state)
        logger.info("Optimization checkpoint: %s", path)
        # A saved candidate survives a crash between publication and checkpoint completion.
        for candidate in guidance_versions(cfg)["candidates"]:
            if candidate.get("optimization_run_id") == digest(plan):
                state.update(status="complete", candidate_path=candidate["_path"])
                _save(path, state)
                return True

        def request(key, system, user, schema, model):
            if key in state["results"]:
                return schema.model_validate(state["results"][key]["parsed"])
            task_cfg = copy.deepcopy(cfg)
            task_cfg.llm.model_id = model
            task_cfg.llm.max_tokens = plan["identity"]["output_tokens"]
            task_cfg.llm.reasoning_effort = plan["identity"]["reasoning_effort"]
            task_cfg.llm.n_samples = 1
            descriptor = {"transport_version": 1, "llm": asdict(task_cfg.llm),
                          "system": system, "user": user, "schema": schema.model_json_schema(),
                          "schema_hint": _schema_hint(schema)}
            cache_key = digest(descriptor)
            cache_path = cfg.optimizer.compiled_dir / "request_cache" / (cache_key + ".json")
            enabled = cfg.optimizer.reuse_evaluations and schema is DecisionRun
            with _cache_lock(cache_path, enabled):
                if enabled and cache_path.exists():
                    cached = _load(cache_path)
                    if cached["request"] != descriptor:
                        raise ValueError("Cached request identity mismatch")
                    parsed = schema.model_validate(cached["parsed"])
                    state["results"][key] = {"parsed": cached["parsed"], "raw": cached["raw"],
                        "origin": "cache", "cache_key": cache_key, "source_checkpoint": cached["source_checkpoint"]}
                    _save(path, state)
                    logger.info("Optimizer cache hit: %s (no paid call)", key)
                    return parsed
                prior = [a for a in state["attempts"] if a["key"] == key]
                if prior and not retry_failed:
                    raise SearchStopped("Failed or interrupted request requires --retry-failed; it may already have been billed")
                amount = reservation(system, user, schema, plan["identity"]["output_tokens"])
                used = sum(a["reserved_tokens"] for a in state["attempts"])
                if len(state["attempts"]) >= cfg.optimizer.max_calls or used + amount > cfg.optimizer.max_total_tokens:
                    raise SearchStopped("Budget exhausted; completed calls are saved. Limits are cumulative across resumes")
                attempt = {"key": key, "model": model, "reserved_tokens": amount, "status": "in_flight", "usage": None}
                state["attempts"].append(attempt)
                _save(path, state)
                logger.info("Optimizer call %d/%d: %s (reserved tokens %d/%d)", len(state["attempts"]),
                            cfg.optimizer.max_calls, key, used + amount, cfg.optimizer.max_total_tokens)
                def record_usage(usage):
                    attempt["usage"] = usage
                    _save(path, state)  # Preserve counters even if subsequent parsing fails.
                try:
                    parsed, raw = call_llm(system, user, task_cfg, schema=schema, max_retries=0,
                                           on_usage=record_usage)
                    parsed = schema.model_validate(parsed.model_dump())
                    state["results"][key] = {"parsed": parsed.model_dump(), "raw": raw,
                                             "origin": "provider", "cache_key": cache_key}
                    attempt["status"] = "complete"
                    _save(path, state)
                    if enabled:
                        _save(cache_path, {"request": descriptor, "parsed": parsed.model_dump(), "raw": raw,
                                           "source_checkpoint": str(path), "source_attempt": len(state["attempts"])-1})
                    return parsed
                except BaseException as exc:
                    attempt.update(status="failed", error=f"{type(exc).__name__}: {str(exc)[:500]}")
                    state["status"] = "stopped"
                    _save(path, state)
                    raise

        try:
            if plan.get("context_mode") == "compare":
                from fundmgr.engine.context_compaction import compare_decisions
                comparisons = []
                for index, case in enumerate(plan["cases"]):
                    system = assemble_system_prompt(case["fields"]["mandate"], plan["identity"]["guidance"])
                    full = request(f"full:{index}", system, raw_task_input(case), DecisionRun,
                                   plan["identity"]["llm"]["model_id"])
                    compact = request(f"compact:{index}", system, case["compact_input"], DecisionRun,
                                      plan["identity"]["llm"]["model_id"])
                    comparisons.append({"run_id": case["run_id"], "source": case["source"],
                                        **compare_decisions(full, compact)})
                state.update(status="complete", comparisons=comparisons, candidate_path=None)
                _save(path, state)
                return False
            proposal = request("proposal", plan["proposal_system"], plan["proposal_user"], Proposal,
                               plan["identity"]["prompt_model"])
            scores = {}
            for name, guidance in (("incumbent", plan["identity"]["guidance"]), ("candidate", proposal.instructions)):
                scores[name] = []
                for index, case in enumerate(plan["cases"]):
                    decision = request(f"{name}:{index}", assemble_system_prompt(case["fields"]["mandate"], guidance),
                                       task_input(case), DecisionRun, plan["identity"]["llm"]["model_id"])
                    scores[name].append(decision_metric(SimpleNamespace(ticker_alphas=case["ticker_alphas"]),
                                                       SimpleNamespace(decision=decision)))
            state["scores"] = scores
            if identity(cfg) != plan["identity"]:
                raise SearchStopped("Active guidance/config changed; candidate not published")
            candidate_path = None
            if mean(scores["candidate"]) > mean(scores["incumbent"]):
                class Program:
                    def save(self, target):
                        Path(target).write_text(json.dumps({"kind": VERSION, "instructions": proposal.instructions}))
                candidate_path = save_guidance_candidate(cfg, Program(), {
                    "instructions": proposal.instructions, "optimization_run_id": digest(plan),
                    "search_method": VERSION, "search_scores": scores,
                    "context_mode": plan.get("context_mode", "full"),
                    "context_version": plan.get("context_version"),
                    "task_model": cfg.llm.model_id, "prompt_model": plan["identity"]["prompt_model"],
                    "training_runs": plan["training_runs"],
                    "validation_runs": [{"run_id": c["run_id"], "source": c["source"]} for c in plan["cases"]],
                    "search_settings": {"max_output_tokens": cfg.optimizer.max_output_tokens,
                                        "reasoning_effort": cfg.optimizer.reasoning_effort},
                    "checkpoint": str(path), "requests": len(state["attempts"]),
                    "reserved_tokens": sum(a["reserved_tokens"] for a in state["attempts"]),
                })
            state.update(status="complete", candidate_path=str(candidate_path) if candidate_path else None)
            _save(path, state)
            logger.info("Search complete: %s", candidate_path or "candidate did not beat incumbent; no artifact promoted")
            return candidate_path is not None
        except BaseException:
            state["status"] = "stopped"
            _save(path, state)
            raise
