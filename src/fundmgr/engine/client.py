from __future__ import annotations

import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from pydantic import ValidationError

from fundmgr.config import AppConfig
from fundmgr.engine.schema import Action, DecisionRun


class LLMError(Exception):
    pass


def _schema_hint(schema: type = DecisionRun) -> str:
    """Schema instruction appended to the system message for BOTH providers.

    Keeping this identical across providers is what makes the model head-to-head
    fair: each model sees the same schema text — and the instructions embedded in
    its field descriptions (conviction floor, sizing rules, etc.) — through the
    same in-context channel. For OpenAI, native structured outputs additionally
    enforce conformance, but only as a parse safety net on top of this text, not
    as a substitute for it.
    """
    return (
        "\n\nRespond with a single JSON object matching this schema:\n"
        + schema.model_json_schema().__repr__()
    )


def call_llm(system: str, user: str, cfg: AppConfig, schema: type = DecisionRun) -> tuple:
    """
    Call the configured LLM and return (parsed `schema` instance, raw response text).
    Uses OpenAI structured outputs when provider=openai, JSON-mode for Anthropic.
    `schema` defaults to DecisionRun (the weekly decision); pass another Pydantic
    model for focused calls like the stop-loss review.
    Raises LLMError on failure.
    """
    if cfg.llm.provider == "openai":
        return _call_openai(system, user, cfg, schema)
    elif cfg.llm.provider == "anthropic":
        return _call_anthropic(system, user, cfg, schema)
    else:
        raise LLMError(f"Unknown LLM provider: {cfg.llm.provider!r}")


# ── OpenAI ────────────────────────────────────────────────────────────────────

def _call_openai(system: str, user: str, cfg: AppConfig, schema: type = DecisionRun) -> tuple:
    try:
        from openai import OpenAI
    except ImportError:
        raise LLMError("openai package not installed. Run: uv pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMError("OPENAI_API_KEY not set in environment / .env file")

    client = OpenAI(api_key=api_key)

    # gpt-5+ / o-series use max_completion_tokens and don't support custom temperature
    _NEW_API_PREFIXES = ("gpt-5", "o1", "o3", "o4")
    new_api = any(cfg.llm.model_id.startswith(p) for p in _NEW_API_PREFIXES)
    token_kwargs = (
        {"max_completion_tokens": cfg.llm.max_tokens}
        if new_api
        else {"max_tokens": cfg.llm.max_tokens}
    )
    temp_kwargs = {} if new_api else {"temperature": cfg.llm.temperature}
    reasoning_kwargs = (
        {"reasoning_effort": cfg.llm.reasoning_effort}
        if cfg.llm.reasoning_effort and new_api
        else {}
    )

    try:
        # Same in-context schema text as the Anthropic path (fair head-to-head);
        # response_format additionally enforces conformance as a parse safety net.
        response = client.beta.chat.completions.parse(
            model=cfg.llm.model_id,
            messages=[
                {"role": "system", "content": system + _schema_hint(schema)},
                {"role": "user", "content": user},
            ],
            response_format=schema,
            **token_kwargs,
            **temp_kwargs,
            **reasoning_kwargs,
        )
    except Exception as e:
        raise LLMError(f"OpenAI API call failed: {e}") from e

    choice = response.choices[0]
    raw_text = choice.message.content or ""

    if choice.message.parsed is None:
        # Structured outputs failed — fall back to JSON parse
        try:
            parsed = schema.model_validate_json(raw_text)
        except (ValidationError, json.JSONDecodeError) as e:
            raise LLMError(f"Failed to parse LLM response: {e}\n\nRaw response:\n{raw_text}") from e
    else:
        parsed = choice.message.parsed

    return parsed, raw_text


# ── Anthropic ─────────────────────────────────────────────────────────────────

def _call_anthropic(system: str, user: str, cfg: AppConfig, schema: type = DecisionRun) -> tuple:
    try:
        import anthropic
    except ImportError:
        raise LLMError("anthropic package not installed. Run: uv pip install anthropic")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMError("ANTHROPIC_API_KEY not set in environment / .env file")

    client = anthropic.Anthropic(api_key=api_key)

    # Same in-context schema text as the OpenAI path (fair head-to-head).
    schema_hint = _schema_hint(schema)

    # Claude 4+ models (opus-4-x, sonnet-4-x, haiku-4-x) do not accept temperature
    _TEMPERATURE_FREE = ("claude-opus-4", "claude-sonnet-4", "claude-haiku-4", "claude-fable")
    use_temperature = not any(cfg.llm.model_id.startswith(p) for p in _TEMPERATURE_FREE)

    try:
        kwargs = dict(
            model=cfg.llm.model_id,
            max_tokens=cfg.llm.max_tokens,
            system=system + schema_hint,
            messages=[{"role": "user", "content": user}],
        )
        if use_temperature:
            kwargs["temperature"] = cfg.llm.temperature
        response = client.messages.create(**kwargs)
    except Exception as e:
        raise LLMError(f"Anthropic API call failed: {e}") from e

    raw_text = response.content[0].text if response.content else ""

    # Strip markdown code fences if present
    clean = raw_text.strip()
    if clean.startswith("```"):
        clean = "\n".join(clean.split("\n")[1:])
        if clean.endswith("```"):
            clean = clean[: clean.rfind("```")]

    try:
        parsed = schema.model_validate_json(clean)
    except (ValidationError, json.JSONDecodeError) as e:
        raise LLMError(f"Failed to parse Anthropic response: {e}\n\nRaw response:\n{raw_text}") from e

    return parsed, raw_text


# ── Consensus sampling ────────────────────────────────────────────────────────

def _aggregate_decisions(runs: list[DecisionRun]) -> tuple[DecisionRun, dict[str, int]]:
    """
    Majority-vote aggregate of N DecisionRun objects.

    A ticker makes it into the consensus only if strictly more than half the runs
    agree on the same side (buy/sell/hold). Numeric fields are averaged across
    the agreeing runs; the thesis is taken from the highest-confidence agreeing run.

    Returns (consensus_decision, {ticker: n_agreeing_runs}).
    """
    n = len(runs)
    threshold = n // 2 + 1  # e.g. 2 out of 3, 3 out of 5

    all_tickers: set[str] = {a.ticker for run in runs for a in run.actions}

    consensus_actions: list[Action] = []
    vote_counts: dict[str, int] = {}

    for ticker in all_tickers:
        per_run = [a for run in runs for a in run.actions if a.ticker == ticker]
        if not per_run:
            continue

        best_side, best_n = Counter(a.side for a in per_run).most_common(1)[0]
        if best_n < threshold:
            continue

        agreeing = [a for a in per_run if a.side == best_side]

        stop_losses    = [a.stop_loss_pct    for a in agreeing if a.stop_loss_pct    is not None]
        take_profits   = [a.take_profit_pct  for a in agreeing if a.take_profit_pct  is not None]
        best_action    = max(agreeing, key=lambda a: a.confidence)

        consensus_actions.append(Action(
            ticker=ticker,
            side=best_side,
            target_weight_pct=round(sum(a.target_weight_pct for a in agreeing) / len(agreeing), 1),
            sek_estimate=round(sum(a.sek_estimate for a in agreeing) / len(agreeing), 0),
            confidence=round(sum(a.confidence for a in agreeing) / len(agreeing), 3),
            thesis=best_action.thesis,
            stop_loss_pct=round(sum(stop_losses) / len(stop_losses), 1) if stop_losses else None,
            take_profit_pct=round(sum(take_profits) / len(take_profits), 1) if take_profits else None,
        ))
        vote_counts[ticker] = best_n

    seen_notes: list[str] = []
    for r in runs:
        if r.notes and r.notes not in seen_notes:
            seen_notes.append(r.notes)

    consensus = DecisionRun(
        run_id=runs[0].run_id,
        market_summary=runs[0].market_summary,
        actions=consensus_actions,
        cash_target_pct=round(sum(r.cash_target_pct for r in runs) / len(runs), 1),
        notes=" | ".join(seen_notes)[:1000],
    )
    return consensus, vote_counts


def call_llm_consensus(
    system: str,
    user: str,
    cfg: AppConfig,
) -> tuple[DecisionRun, str, dict[str, int] | None, dict]:
    """
    Call the LLM cfg.llm.n_samples times in parallel and return a majority-vote
    consensus DecisionRun.

    Returns (decision, raw_for_db, vote_counts, sampling) where:
      - vote_counts is None when n_samples <= 1 (passthrough mode)
      - vote_counts is {ticker: n_agreeing_runs} when consensus was used
      - sampling is {requested, succeeded, failed, errors} — per-run sample
        health, persisted so the malformed-output rate can be measured (the
        data the Refine gate is waiting on).
    """
    n = cfg.llm.n_samples
    if n <= 1:
        decision, raw = call_llm(system, user, cfg)
        return decision, raw, None, {"requested": 1, "succeeded": 1, "failed": 0, "errors": []}

    results:       list[DecisionRun] = []
    raw_responses: list[str]         = []
    errors:        list[str]         = []

    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(call_llm, system, user, cfg) for _ in range(n)]
        for fut in as_completed(futures):
            try:
                decision, raw = fut.result()
                results.append(decision)
                raw_responses.append(raw)
            except LLMError as e:
                errors.append(str(e))

    if not results:
        raise LLMError(f"All {n} LLM calls failed: {'; '.join(errors)}")

    if errors:
        # Partial failure — note it but continue with what we have
        print(f"  ⚠ {len(errors)}/{n} LLM call(s) failed; building consensus from {len(results)} run(s).")

    sampling = {
        "requested": n,
        "succeeded": len(results),
        "failed":    len(errors),
        "errors":    [e[:200] for e in errors],  # truncated for storage
    }

    if len(results) == 1:
        return results[0], raw_responses[0], {a.ticker: 1 for a in results[0].actions}, sampling

    consensus, vote_counts = _aggregate_decisions(results)
    return consensus, consensus.model_dump_json(), vote_counts, sampling
