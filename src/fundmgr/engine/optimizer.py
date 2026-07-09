"""
MIPROv2 optimization of the weekly decision instructions (DeepSwing-style).

Builds run-level training examples from the fielded prompt snapshots that
`build_prompt` already persists (snapshot v2; v1 rows are reconstructed from the
flat strings), scored against the realized per-ticker alphas in
`decision_outcomes`. MIPROv2 searches instruction space for the WeeklyDecision
signature in `dspy_program.py`: a heavy prompt model writes candidate
instructions, the configured decision model evaluates them against history, and
the winner is saved to a git-backed guidance artifact that `build_prompt`
appends to the mandate on every subsequent run.

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
    """Current + archived optimized guidance for this fund, for the web view.

    Returns {"current": {...}|None, "history": [{...}, ...]} where each entry is
    the guidance JSON (instructions + metadata: created_at, models, run counts).
    History is newest-first, parsed from the timestamped archive files.
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

    return {"current": current, "history": history}


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

def decision_metric(example, prediction, trace=None) -> float:
    """
    Reward a predicted DecisionRun by the alpha it would have captured.

    Each example carries `ticker_alphas`: realized return-vs-benchmark (in
    percentage points, over the ~28-day evaluation window) for every ticker
    whose outcome from that run is known. A predicted buy "earns" that alpha, a
    sell earns the inverse (exiting before underperformance is good), and a
    hold — or omitting the ticker — earns nothing. The mean over all known
    tickers is squashed to (0, 1), so sitting out beats acting badly, acting
    well beats sitting out, and the *size* of each move drives the optimization.
    """
    alphas: dict[str, float] = dict(getattr(example, "ticker_alphas", None) or {})
    if not alphas:
        return 0.5

    decision = getattr(prediction, "decision", None)
    actions = getattr(decision, "actions", None) or []

    realized = 0.0
    for action in actions:
        ticker = str(getattr(action, "ticker", "")).upper()
        alpha = alphas.get(ticker)
        if alpha is None:
            continue
        side = str(getattr(action, "side", "")).lower()
        if side == "buy":
            realized += alpha
        elif side == "sell":
            realized -= alpha

    mean_alpha = realized / len(alphas) / 100.0
    return 0.5 + 0.5 * math.tanh(mean_alpha * _ALPHA_METRIC_SCALE)


# ── Trainset ─────────────────────────────────────────────────────────────────

def build_trainset(store: "Store") -> list[dict]:
    """
    One example per run that has (a) at least one evaluated outcome and (b) a
    recoverable fielded context. Inputs match WeeklyDecision's fields; the
    per-ticker alphas ride along for the metric.
    """
    alphas_by_run: dict[str, dict[str, float]] = {}
    for o in store.get_evaluated_outcomes():
        if o.position_return_pct is None or o.benchmark_return_pct is None:
            continue
        alphas_by_run.setdefault(o.run_id, {})[o.ticker.upper()] = float(
            o.position_return_pct - o.benchmark_return_pct
        )

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
        examples.append({**fields, "ticker_alphas": alphas, "run_id": run_id})

    return examples


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
    return "claude-opus-4-8" if provider == "anthropic" else "gpt-5.5"


def run_optimization(
    cfg: AppConfig,
    store: "Store",
    min_outcomes: int | None = None,
    min_examples: int | None = None,
) -> bool:
    """
    Run MIPROv2 over the run-level trainset and persist the winning instructions
    as the guidance artifact. Returns True if a new artifact was saved.
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

    raw = build_trainset(store)
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

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    out_path = compiled_program_path(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.rename(out_path.with_name(f"{out_path.stem}_{stamp}.json"))
    compiled.save(str(out_path))

    g_path = guidance_path(cfg)
    if g_path.exists():
        g_path.rename(g_path.with_name(f"{g_path.stem}_{stamp}.json"))
    g_path.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_train_runs": len(train),
        "n_val_runs": len(val),
        "n_outcomes": len(evaluated),
        "task_model": cfg.llm.model_id,
        "prompt_model": prompt_model_id,
        "instructions": instructions,
    }, indent=2))

    logger.info("Optimizer: saved compiled program to %s and guidance to %s", out_path, g_path)
    return True


def _compiled_instructions(compiled) -> str:
    for _, predictor in compiled.named_predictors():
        instructions = str(getattr(predictor.signature, "instructions", "") or "").strip()
        if instructions:
            return instructions
    return ""
