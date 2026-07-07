"""
SKETCH / PROTOTYPE — not wired into the live run path.

A DSPy re-expression of the weekly decision step that currently lives in
`engine/prompt.py` (prompt construction) + `engine/client.py` (LLM call,
parsing, N-sample consensus).

Goal of this file: show what the program *shape* looks like so we can decide
whether to adopt DSPy before committing to it. The live run path still goes
through prompt.py + client.py; however, `engine/optimizer.py` (fund optimize)
now uses WeeklyDecision + build_lm from here to MIPRO-optimize the decision
instructions offline. `dspy` is an optional extra: uv sync --extra optimize.

What this demonstrates, in order of near-term value:
  1. A typed Signature mirroring build_prompt's inputs -> DecisionRun output.
  2. A Module that adds self-correction (dspy.Refine) for the JSON + risk
     limits the model currently has no feedback loop on.
  3. Model portability — the SAME program runs on GPT-5.5 or Opus by swapping
     the LM, which is exactly what makes the sim head-to-head fair.
  4. Where N=3 consensus and KNNFewShot (decision_outcomes) plug in later.

None of this needs training data except step 4's optimizer, which is opt-in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fundmgr.engine.schema import DecisionRun

if TYPE_CHECKING:
    from fundmgr.config import AppConfig

try:
    import dspy
except ImportError:  # keep the module importable for review without the dep
    dspy = None


# ── 1. Signature ────────────────────────────────────────────────────────────
# The contract for the step. Today this is implicit: build_prompt() bakes the
# context into a string and the schema_hint lives in client.py. A Signature
# makes inputs/outputs explicit and decouples them from prompt wording, so an
# optimizer can later rewrite the *instructions* without us touching call sites.
#
# We keep the rich context blocks (portfolio, risk limits, universe features,
# learnings) as separate inputs rather than one giant string. That lets DSPy
# reason about each field and lets KNNFewShot retrieve on the situation later.

if dspy is not None:

    class WeeklyDecision(dspy.Signature):
        """Allocate a long-only equity portfolio for the coming week.

        Apply the mandate and the past-performance reflections. Stay strictly
        within the stated risk limits. Return one action per ticker you have a
        view on; omit tickers with no view.
        """

        mandate: str = dspy.InputField(desc="Fund mandate / investing philosophy")
        macro: str = dspy.InputField(desc="Macro + market context for this run")
        portfolio_state: str = dspy.InputField(desc="Current NAV, cash, open positions, P&L")
        risk_limits: str = dspy.InputField(desc="Hard constraints: position/sector caps, cash band, turnover, fees")
        universe: str = dspy.InputField(desc="Per-ticker feature blocks (held + top candidates by signal)")
        learnings: str = dspy.InputField(desc="Distilled lessons from prior decisions")

        # Typed output — DSPy will coerce/validate against the existing Pydantic
        # model, replacing the hand-rolled JSON parsing + fence-stripping in
        # client.py's _call_anthropic.
        decision: DecisionRun = dspy.OutputField(desc="The week's buy/sell/hold decisions")


# ── 2. Module with self-correction ──────────────────────────────────────────
# This is the no-training-data win. Today the model gets one shot; malformed
# JSON or out-of-limit sizing is only caught *after* the call (parse error ->
# LLMError, or the guardrails silently reject the trade). dspy.Refine reruns
# the prediction with feedback until a reward function is satisfied, so the
# model fixes its own constraint violations before guardrails ever see them.


def _risk_reward(args, pred) -> float:
    """Reward used by Refine. 1.0 = clean; lower = nudge the model to retry.

    This is a soft pre-check that mirrors (a subset of) the real guardrails so
    the model self-corrects the cheap, obvious violations. The mechanical
    guardrails downstream remain the source of truth.
    """
    d: DecisionRun = pred.decision
    score = 1.0
    # Single-name weight cap (mirror cfg.risk.max_position_pct; hardcoded here
    # only because the sketch has no cfg in scope).
    if any(a.target_weight_pct > 18 for a in d.actions):
        score -= 0.5
    # Buys below the mandate's conviction floor shouldn't appear.
    if any(a.side == "buy" and a.confidence < 0.40 for a in d.actions):
        score -= 0.3
    # Cash target should sit inside the configured band (5–10%).
    if not (5 <= d.cash_target_pct <= 10):
        score -= 0.2
    return max(score, 0.0)


if dspy is not None:

    class FundManager(dspy.Module):
        """The weekly decision program. One predictor + bounded self-correction."""

        def __init__(self, max_retries: int = 2):
            super().__init__()
            # ChainOfThought gives the model room to reason before emitting the
            # structured decision — analogous to gpt-5.5 reasoning_effort, but
            # provider-agnostic.
            predictor = dspy.ChainOfThought(WeeklyDecision)
            self.decide = dspy.Refine(
                module=predictor,
                N=max_retries,
                reward_fn=_risk_reward,
                threshold=1.0,
            )

        def forward(self, **inputs) -> DecisionRun:
            return self.decide(**inputs).decision


# ── 3. Model portability — what makes the head-to-head fair ─────────────────
# Right now both sims share build_prompt but call different SDKs through two
# branches in client.py (each with its own quirks: temperature handling, JSON
# fences, structured-outputs vs schema_hint). DSPy collapses that to one LM
# abstraction, so GPT-5.5 vs Opus run the *identical* program and we're
# comparing the models, not two slightly different prompt paths.


def build_lm(cfg: "AppConfig", model_id: str | None = None):
    """Map our AppConfig.llm onto a dspy.LM. Mirrors client.py's per-provider quirks.

    `model_id` overrides cfg.llm.model_id (same provider) — used by the optimizer
    to run a heavier prompt-writing model next to the decision-tier task model.
    """
    if dspy is None:
        raise RuntimeError("dspy not installed — install with: uv sync --extra optimize")

    model_id = model_id or cfg.llm.model_id
    model = f"{cfg.llm.provider}/{model_id}"  # e.g. "openai/gpt-5.5", "anthropic/claude-opus-4-8"
    kwargs = {"max_tokens": cfg.llm.max_tokens}

    # Reproduce the temperature carve-outs already in client.py.
    temp_free = model_id.startswith(
        ("gpt-5", "o1", "o3", "o4", "claude-opus-4", "claude-sonnet-4", "claude-haiku-4", "claude-fable")
    )
    if not temp_free:
        kwargs["temperature"] = cfg.llm.temperature
    if model_id.startswith(("gpt-5", "o1", "o3", "o4")):
        # OpenAI reasoning models reject small completion budgets via litellm.
        kwargs["max_tokens"] = max(kwargs["max_tokens"], 16000)
        if cfg.llm.reasoning_effort:
            kwargs["reasoning_effort"] = cfg.llm.reasoning_effort

    return dspy.LM(model, **kwargs)


# ── 4. Where consensus + optimization plug in (later) ───────────────────────
# Consensus: DSPy does NOT replace _aggregate_decisions. We'd still run the
# module N times and feed the resulting DecisionRun objects to the existing
# majority-vote aggregator. Sketch:
#
#     fm = FundManager()
#     with dspy.context(lm=build_lm(cfg)):
#         runs = [fm(**inputs) for _ in range(cfg.llm.n_samples)]  # parallelize as today
#     consensus, votes = _aggregate_decisions(runs)               # unchanged
#
# KNNFewShot (uses decision_outcomes as it accumulates — no training run):
#
#     trainset = load_examples_from_decision_outcomes(store)   # dspy.Example per past run
#     fm.decide = KNNFewShot(k=4, trainset=trainset).compile(fm.decide)
#
# MIPROv2 (the year-out goal, gated on an unbiased trainset + a metric):
#
#     metric = lambda ex, pred, _: ex.score   # excess-return label from score_runs
#     fm = MIPROv2(metric=metric).compile(fm, trainset=trainset)
#
# Adoption path: ship steps 1–3 first (zero data, immediate: fair comparison +
# fewer malformed/illegal outputs), then layer 4 on as data arrives.


def to_example(fields: dict, score: float):
    """Map a logged run's fielded snapshot to a dspy.Example for optimization.

    As of snapshot_version 2 the corpus stores `fields` directly (see
    prompt.snapshot_to_dict), so this lines up 1:1 with WeeklyDecision's inputs
    — no reconstruction from the flattened strings needed.
    """
    if dspy is None:
        raise RuntimeError("dspy not installed — this is a sketch")
    return dspy.Example(score=score, **fields).with_inputs(
        "mandate", "macro", "portfolio_state", "risk_limits", "universe", "learnings"
    )
