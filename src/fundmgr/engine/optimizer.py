"""
MIPROv2 optimization of the weekly decision instructions (DeepSwing-style).

Builds run-level training examples from the fielded prompt snapshots that
`build_prompt` already persists (snapshot v2; v1 rows are reconstructed from the
flat strings), scored against the realized per-ticker alphas in
`decision_outcomes`. MIPROv2 searches instruction space for the WeeklyDecision
signature in `dspy_program.py`: a heavy prompt model writes candidate
instructions, the configured decision model evaluates them against history, and
the winner is saved as an inactive candidate. The active guidance loaded by
`build_prompt` is unchanged until a separately evaluated candidate is promoted.

dspy is an optional dependency (`uv sync --extra optimize`). Everything except
run_optimization() works without it.
"""
from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from fundmgr.config import AppConfig

if TYPE_CHECKING:
    from fundmgr.state.store import Store

logger = logging.getLogger(__name__)

# Scales the mean realized alpha (fraction) before the tanh squash. Per-run the
# mean is diluted across every ticker with a known outcome, so a smaller swing
# should already saturate: at k=25 a ±2pp mean alpha lands near the extremes.
_ALPHA_METRIC_SCALE = 25.0

# This remains a directional-alpha search surrogate, not a portfolio-return
# evaluator. Historical thesis verdicts describe a different decision's claim;
# they must never reward or penalize a candidate's new reasoning.
METRIC_VERSION = "directional_alpha_v2"

INPUT_FIELDS = ("mandate", "macro", "portfolio_state", "risk_limits", "universe", "learnings")


# ── Artifacts ────────────────────────────────────────────────────────────────

def compiled_program_path(cfg: AppConfig) -> Path:
    # Keyed by DB stem, not mandate stem: the GPT and Claude sims share a
    # mandate file but learn from their own outcomes, so artifacts are per-fund.
    return cfg.optimizer.compiled_dir / f"{cfg.db_path.stem}_weekly_decision.json"


def guidance_path(cfg: AppConfig) -> Path:
    return cfg.optimizer.compiled_dir / f"{cfg.db_path.stem}_guidance.json"


def load_guidance(cfg: AppConfig) -> str:
    """Optimized decision guidance for the prompt, or "" when none compiled yet."""
    path = guidance_path(cfg)
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text())
        return str(data.get("instructions", "")).strip()
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load guidance from %s: %s", path, exc)
        return ""


def guidance_versions(cfg: AppConfig) -> dict:
    """Active guidance, archived versions and inactive candidates for this fund.

    Returns current (or None), history and candidates. Each entry contains its
    instructions and metadata; historical/candidate lists are newest first.
    """
    def _read(path: Path) -> dict | None:
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        if not str(data.get("instructions", "")).strip():
            return None
        data["_filename"] = path.name
        return data

    current = _read(guidance_path(cfg)) if guidance_path(cfg).exists() else None

    history: list[dict] = []
    compiled_dir = cfg.optimizer.compiled_dir
    if compiled_dir.exists():
        stem = cfg.db_path.stem
        for p in sorted(compiled_dir.glob(f"{stem}_guidance_*.json"), reverse=True):
            entry = _read(p)
            if entry:
                history.append(entry)

    candidates = []
    for p in sorted(candidate_directory(cfg).glob("*/guidance.json"), reverse=True):
        entry = _read(p)
        if entry:
            entry["_path"] = str(p)
            candidates.append(entry)
    return {"current": current, "history": history, "candidates": candidates}


def guidance_fingerprint(cfg: AppConfig) -> str | None:
    """Short hash of the active guidance instructions, or None when none is applied.

    Recorded in each run's snapshot regime so score-runs can compare guided vs
    unguided weeks (and which guidance version) — the A/B signal that tells us
    whether the optimizer actually helps.
    """
    guidance = load_guidance(cfg)
    if not guidance:
        return None
    import hashlib
    return hashlib.sha256(guidance.encode()).hexdigest()[:12]


# ── Metric ───────────────────────────────────────────────────────────────────

def _sided(side: str, value: float) -> float:
    """A buy earns the value, a sell earns its inverse, a hold earns nothing."""
    if side == "buy":
        return value
    if side == "sell":
        return -value
    return 0.0


def decision_metric(example, prediction, trace=None) -> float:
    """Directional-alpha search score, independent of historical thesis labels.

    This interim surrogate scores known ticker/side choices, not allocation,
    costs or feasibility. Compiled candidates require separate evaluation
    through the live decision pipeline before they can become active guidance.
    """
    alphas: dict[str, float] = dict(getattr(example, "ticker_alphas", None) or {})
    if not alphas:
        return 0.5

    decision = getattr(prediction, "decision", None)
    actions = getattr(decision, "actions", None) or []
    realized = 0.0
    seen: set[str] = set()
    for action in actions:
        ticker = str(getattr(action, "ticker", "")).upper()
        side = str(getattr(action, "side", "")).lower()
        # Duplicate actions must not multiply an observed outcome's reward.
        if ticker in seen:
            return 0.0
        seen.add(ticker)
        alpha = alphas.get(ticker)
        if alpha is not None:
            realized += _sided(side, alpha)

    signal = (realized / len(alphas) / 100.0) * _ALPHA_METRIC_SCALE
    return 0.5 + 0.5 * math.tanh(signal)


# ── Trainset ─────────────────────────────────────────────────────────────────

def build_trainset(store: "Store", source: str = "") -> list[dict]:
    """
    One example per run that has (a) at least one evaluated outcome and (b) a
    recoverable fielded context. Inputs match WeeklyDecision's fields; the
    per-ticker alphas feed the search metric; thesis verdicts remain diagnostic metadata.

    `source` labels which fund an example came from, for pooled trainsets.
    """
    alphas_by_run: dict[str, dict[str, float]] = {}
    theses_by_run: dict[str, dict[str, str]] = {}
    for o in store.get_evaluated_outcomes():
        if o.position_return_pct is None or o.benchmark_return_pct is None:
            continue
        ticker = o.ticker.upper()
        alphas_by_run.setdefault(o.run_id, {})[ticker] = float(
            o.position_return_pct - o.benchmark_return_pct
        )
        if o.thesis_verdict:
            theses_by_run.setdefault(o.run_id, {})[ticker] = o.thesis_verdict

    examples: list[dict] = []
    for run_id, alphas in sorted(alphas_by_run.items()):
        rec = store.get_recommendation_by_run_id(run_id)
        if not rec:
            continue
        try:
            snapshot = json.loads(rec.prompt_snapshot)
        except json.JSONDecodeError:
            continue
        fields = fields_from_snapshot(snapshot)
        if not fields:
            continue
        examples.append({
            **fields,
            "ticker_alphas": alphas,
            "ticker_theses": theses_by_run.get(run_id, {}),
            "run_id": run_id,
            "source": source,
        })

    return examples


def build_pooled_trainset(cfg: AppConfig) -> list[dict]:
    """This fund's examples plus those of any fund named in `optimizer.pool_configs`.

    Pooling is defensible here in a way it is not for performance measurement.
    Correlated funds do not give independent evidence about whether *this book*
    has an edge — they all trade the same weeks. But the optimizer is not
    measuring the book, it is searching for instructions that make the decision
    task go better, and that question is shared across funds: examples from a
    different universe are extra evidence about how to reason, not a contaminated
    read on one portfolio. It also buys elapsed time, which is the binding
    constraint — five funds accrue five examples a week between them, not one.

    Each fund keeps its own guidance artifact regardless; only the trainset is
    shared. Failures to open a pooled fund are logged and skipped, never fatal:
    a missing sibling config must not stop this fund optimizing.
    """
    from fundmgr.config import CONFIG_DIR, load_config
    from fundmgr.state.store import Store

    examples = build_trainset(Store(cfg.db_path), source=cfg.db_path.stem)
    seen = {cfg.db_path}

    for name in cfg.optimizer.pool_configs:
        path = Path(name)
        if not path.is_absolute():
            path = CONFIG_DIR / name
        try:
            other = load_config(path)
        except Exception as exc:
            logger.warning("Optimizer: cannot pool from %s: %s", name, exc)
            continue
        if other.db_path in seen or not other.db_path.exists():
            continue
        seen.add(other.db_path)
        pooled = build_trainset(Store(other.db_path), source=other.db_path.stem)
        logger.info("Optimizer: pooled %d example(s) from %s", len(pooled), name)
        examples.extend(pooled)

    # Chronological across funds, so the held-out split stays a forward split
    # rather than becoming a random sample of history.
    return sorted(examples, key=lambda e: e["run_id"])


def fields_from_snapshot(snapshot: dict) -> dict[str, str] | None:
    """Fielded WeeklyDecision inputs from a stored prompt snapshot (v2 direct, v1 reconstructed)."""
    fields = snapshot.get("fields")
    if isinstance(fields, dict) and fields.get("universe"):
        return {k: str(fields.get(k, "") or "") for k in INPUT_FIELDS}

    user_msg = snapshot.get("user_message", "")
    system_msg = snapshot.get("system_message", "")
    if not user_msg:
        return None
    universe = extract_section(user_msg, "## Universe")
    if not universe:
        return None
    return {
        "mandate":         system_msg.split("\n\n---\n")[0].strip(),
        "macro":           extract_section(user_msg, "## Global Macro Context"),
        "portfolio_state": extract_section(user_msg, "## Current Portfolio State"),
        "risk_limits":     extract_section(user_msg, "## Risk Limits"),
        "universe":        universe,
        "learnings":       extract_section(user_msg, "## Past Performance Reflections"),
    }


def extract_section(text: str, header: str) -> str:
    """Return the block starting at `header` up to the next '## ' heading."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith(header):
            block = [line]
            for row in lines[i + 1:]:
                if row.startswith("## "):
                    break
                block.append(row)
            return "\n".join(block).strip()
    return ""


_PRICE_RE = re.compile(r"Price:\s*([0-9]+(?:\.[0-9]+)?)")


def price_from_snapshot(snapshot: dict, ticker: str) -> float | None:
    """Parse the ticker's decision-time share price out of a stored prompt snapshot."""
    user_msg = snapshot.get("user_message", "")
    marker = f"[{ticker.upper()}]"
    lines = user_msg.split("\n")
    for i, line in enumerate(lines):
        if marker in line:
            for row in lines[i:i + 8]:
                m = _PRICE_RE.search(row)
                if m:
                    return float(m.group(1))
                if not row.strip():
                    break
            return None
    return None


# ── Optimization run ─────────────────────────────────────────────────────────

def _default_prompt_model(provider: str) -> str:
    from fundmgr.config import default_heavy_model
    return default_heavy_model(provider)


def run_optimization(
    cfg: AppConfig,
    store: "Store",
    min_outcomes: int | None = None,
    min_examples: int | None = None,
) -> bool:
    """
    Run MIPROv2 over the run-level trainset and persist the winning instructions
    as an inactive candidate. Returns True if a candidate was saved.
    """
    try:
        import dspy
        from dspy.teleprompt import MIPROv2
    except ImportError:
        logger.error("dspy is not installed — run: uv sync --extra optimize")
        return False

    from fundmgr.engine.dspy_program import WeeklyDecision, build_lm

    min_outcomes = min_outcomes if min_outcomes is not None else cfg.optimizer.min_outcomes
    min_examples = min_examples if min_examples is not None else cfg.optimizer.min_examples

    evaluated = store.get_evaluated_outcomes()
    if len(evaluated) < min_outcomes:
        logger.info("Optimizer: only %d evaluated outcomes, need %d — skipping", len(evaluated), min_outcomes)
        return False

    raw = build_pooled_trainset(cfg)
    if len(raw) < min_examples:
        logger.info("Optimizer: only %d usable run examples, need %d — skipping", len(raw), min_examples)
        return False

    trainset = [dspy.Example(**ex).with_inputs(*INPUT_FIELDS) for ex in raw]
    split = max(1, int(len(trainset) * 0.8))
    train, val = trainset[:split], trainset[split:] or trainset[-1:]

    prompt_model_id = cfg.optimizer.prompt_model_id or _default_prompt_model(cfg.llm.provider)
    # Two roles: the task model runs candidate programs against history (many
    # calls → the configured decision-tier model); the prompt model writes the
    # candidate instructions (few calls → the heaviest reasoner).
    task_lm = build_lm(cfg)
    prompt_lm = build_lm(cfg, model_id=prompt_model_id)

    program = dspy.ChainOfThought(WeeklyDecision)

    logger.info(
        "Optimizer: MIPROv2 with %d train / %d val runs (task=%s, prompt=%s)",
        len(train), len(val), cfg.llm.model_id, prompt_model_id,
    )

    try:
        dspy.configure(lm=task_lm)
        optimizer = MIPROv2(
            metric=decision_metric,
            prompt_model=prompt_lm,
            task_model=task_lm,
            auto="light",
            num_threads=1,
        )
        compiled = optimizer.compile(
            program,
            trainset=train,
            valset=val,
            requires_permission_to_run=False,
        )
    except Exception as exc:
        logger.error("Optimizer: MIPROv2 failed: %s", exc, exc_info=True)
        return False

    instructions = _compiled_instructions(compiled)
    if not instructions:
        logger.error("Optimizer: compiled program carries no instructions — nothing saved")
        return False

    candidate = save_guidance_candidate(cfg, compiled, {
        "n_train_runs": len(train),
        "n_val_runs": len(val),
        "n_outcomes": len(evaluated),
        "training_runs": [{"run_id": e.get("run_id"), "source": e.get("source")} for e in raw[:split]],
        "validation_runs": [{"run_id": e.get("run_id"), "source": e.get("source")} for e in raw[split:]],
        "pooled_from": sorted({e.get("source", "") for e in raw} - {""}),
        "task_model": cfg.llm.model_id,
        "prompt_model": prompt_model_id,
        "instructions": instructions,
    })
    logger.info("Optimizer: saved inactive candidate to %s; active guidance unchanged", candidate)
    return True


def candidate_directory(cfg: AppConfig) -> Path:
    """Separate from active/history globs: compilation is not promotion."""
    return cfg.optimizer.compiled_dir / "candidates" / cfg.db_path.stem


def save_guidance_candidate(cfg: AppConfig, compiled, metadata: dict) -> Path:
    """Persist an immutable candidate, publishing its manifest only after save.

    No active artifact is changed. A later evaluation/promotion workflow must
    compare this candidate with its recorded incumbent on unseen inputs.
    """
    from uuid import uuid4

    now = datetime.now(timezone.utc)
    directory = candidate_directory(cfg) / (now.strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid4().hex[:12])
    directory.mkdir(parents=True, exist_ok=False)
    compiled.save(str(directory / "program.json"))
    manifest = directory / "guidance.json"
    payload = {
        **metadata,
        "created_at": now.isoformat(),
        "status": "pending_evaluation",
        "metric_version": METRIC_VERSION,
        "candidate_id": directory.name,
        "fund_id": cfg.db_path.stem,
        "horizon_days": cfg.evaluation_horizon_days,
        "provider": cfg.llm.provider,
        "mandate": cfg.mandate_path.read_text().strip(),
        "incumbent_guidance_hash": guidance_fingerprint(cfg),
        "config_hash": cfg.config_hash(),
    }
    temporary = directory / "guidance.tmp"
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(manifest)
    return manifest


def _compiled_instructions(compiled) -> str:
    for _, predictor in compiled.named_predictors():
        instructions = str(getattr(predictor.signature, "instructions", "") or "").strip()
        if instructions:
            return instructions
    return ""
