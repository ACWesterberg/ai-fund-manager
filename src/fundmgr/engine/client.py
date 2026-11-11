from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import ValidationError

from fundmgr.config import AppConfig
from fundmgr.engine.schema import Action, DecisionRun

logger = logging.getLogger(__name__)


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


def _report_usage(response, provider, callback):
    """Report provider counters without treating absent usage as zero spend."""
    if callback is None:
        return
    usage = getattr(response, "usage", None)
    if usage is None:
        callback(None)
        return
    raw = usage.model_dump() if hasattr(usage, "model_dump") else vars(usage)
    if provider == "openai":
        inputs, outputs = raw.get("prompt_tokens"), raw.get("completion_tokens")
        cached = (raw.get("prompt_tokens_details") or {}).get("cached_tokens")
        reasoning = (raw.get("completion_tokens_details") or {}).get("reasoning_tokens")
        cache_write = (raw.get("prompt_tokens_details") or {}).get("cache_write_tokens")
    else:
        # Anthropic's input_tokens excludes cache reads and writes.
        cache_write, cached = raw.get("cache_creation_input_tokens"), raw.get("cache_read_input_tokens")
        inputs = raw.get("input_tokens")
        if inputs is not None:
            inputs += (cache_write or 0) + (cached or 0)
        outputs, reasoning = raw.get("output_tokens"), None
    callback({"provider": provider, "response_id": getattr(response, "id", None),
              "model": getattr(response, "model", None), "input_tokens": inputs,
              "output_tokens": outputs, "cached_input_tokens": cached,
              "cache_write_input_tokens": cache_write, "reasoning_tokens": reasoning,
              "raw": raw})


def call_llm(system: str, user: str, cfg: AppConfig, schema: type = DecisionRun,
             *, max_retries: int | None = None, on_usage=None) -> tuple:
    """
    Call the configured LLM and return (parsed `schema` instance, raw response text).
    Uses OpenAI structured outputs when provider=openai, JSON-mode for Anthropic.
    `schema` defaults to DecisionRun (the weekly decision); pass another Pydantic
    model for focused calls like the stop-loss review.
    Raises LLMError on failure.
    """
    retry_kwargs = {} if max_retries is None else {"max_retries": max_retries}
    if on_usage is not None:
        retry_kwargs["on_usage"] = on_usage
    if cfg.llm.provider == "openai":
        return _call_openai(system, user, cfg, schema, **retry_kwargs)
    elif cfg.llm.provider == "anthropic":
        return _call_anthropic(system, user, cfg, schema, **retry_kwargs)
    else:
        raise LLMError(f"Unknown LLM provider: {cfg.llm.provider!r}")


# ── OpenAI ────────────────────────────────────────────────────────────────────

def _call_openai(system: str, user: str, cfg: AppConfig, schema: type = DecisionRun,
                 *, max_retries: int | None = None, on_usage=None) -> tuple:
    try:
        from openai import OpenAI
    except ImportError:
        raise LLMError("openai package not installed. Run: uv pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMError("OPENAI_API_KEY not set in environment / .env file")

    client = OpenAI(api_key=api_key, **({"max_retries": max_retries} if max_retries is not None else {}))

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
        completion = getattr(e, "completion", None)
        if completion is not None:
            _report_usage(completion, "openai", on_usage)
        raise LLMError(f"OpenAI API call failed: {e}") from e

    _report_usage(response, "openai", on_usage)
    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise LLMError("OpenAI response exceeded the output token limit")
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

# Server-side refusal fallback (beta): a request the safety classifiers decline
# is re-run on the named model inside the same call, instead of costing the
# sample. Only for models that can decline this way; the run log names any
# sample another model answered.
REFUSAL_FALLBACK: dict[str, str] = {
    "claude-opus-5-5": "claude-opus-4-8",
    "claude-fable-5-2": "claude-opus-4-8",
    "claude-sonnet-5-5": "claude-sonnet-5",
}

CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")

_CLAUDE_ID = re.compile(r"^claude-(opus|sonnet|haiku|fable|mythos)-(\d+)(?:-(\d{1,2}))?(?:-|$)")


def claude_capabilities(model_id: str) -> dict[str, bool]:
    """Which request knobs a Claude model takes, read from its family and version.

    A hard-coded prefix list went stale with every release: `claude-opus-5-5`
    matched neither the no-temperature list (sampling params are a 400 from
    4.7 on) nor the effort list (so it would have run at its medium default),
    and `claude-sonnet-5` was being sent a temperature.

    - sampling: only pre-Claude-4 IDs (`claude-3-…`) still take a temperature.
    - effort:  adaptive thinking + output_config.effort, from Opus/Sonnet 4.6,
               Haiku 5 and every Fable/Mythos.
    """
    m = _CLAUDE_ID.match(model_id or "")
    if not m:
        return {"sampling": True, "effort": False}
    family, major, minor = m.group(1), int(m.group(2)), int(m.group(3) or 0)
    if family in ("fable", "mythos"):
        effort = True
    elif family == "haiku":
        effort = major >= 5
    else:
        effort = (major, minor) >= (4, 6)
    return {"sampling": False, "effort": effort}


def _call_anthropic(system: str, user: str, cfg: AppConfig, schema: type = DecisionRun,
                    *, max_retries: int | None = None, on_usage=None) -> tuple:
    try:
        import anthropic
    except ImportError:
        raise LLMError("anthropic package not installed. Run: uv pip install anthropic")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMError("ANTHROPIC_API_KEY not set in environment / .env file")

    client = anthropic.Anthropic(api_key=api_key, **({"max_retries": max_retries} if max_retries is not None else {}))

    # Same in-context schema text as the OpenAI path (fair head-to-head).
    schema_hint = _schema_hint(schema)

    caps = claude_capabilities(cfg.llm.model_id)
    try:
        kwargs = dict(
            model=cfg.llm.model_id,
            max_tokens=cfg.llm.max_tokens,
            system=system + schema_hint,
            messages=[{"role": "user", "content": user}],
        )
        if caps["sampling"]:
            kwargs["temperature"] = cfg.llm.temperature
        if caps["effort"] and cfg.llm.reasoning_effort:
            # Reasoning parity with the OpenAI funds: adaptive thinking plus an
            # explicit effort (their equivalent of gpt-5's reasoning_effort).
            # Always sent when configured: Opus 5.5 defaults to medium, every
            # earlier model to high, so leaving it out would quietly lower it.
            effort = cfg.llm.reasoning_effort
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": effort if effort in CLAUDE_EFFORTS else "high"}
        # Stream: with adaptive thinking on, thinking tokens share the max_tokens
        # budget, so the budget is set high; the SDK requires streaming above
        # ~21k non-streaming (10-min guard). Streaming accumulates the same
        # Message and sidesteps that guard.
        fallback = REFUSAL_FALLBACK.get(cfg.llm.model_id)
        if fallback:
            stream = client.beta.messages.stream(
                **kwargs, betas=["server-side-fallback-2026-07-01"],
                fallbacks=[{"model": fallback}])
        else:
            stream = client.messages.stream(**kwargs)
        with stream as live:
            response = live.get_final_message()
    except Exception as e:
        raise LLMError(f"Anthropic API call failed: {e}") from e

    _report_usage(response, "anthropic", on_usage)
    if getattr(response, "stop_reason", None) == "max_tokens":
        raise LLMError("Anthropic response exceeded the output token limit")
    if getattr(response, "stop_reason", None) == "refusal":
        details = getattr(response, "stop_details", None)
        raise LLMError(f"{cfg.llm.model_id} declined the request "
                       f"(category: {getattr(details, 'category', None) or 'unstated'})")
    served = getattr(response, "model", cfg.llm.model_id)
    if served != cfg.llm.model_id:
        # A paired fund's comparison is a difference in models; a sample another
        # model answered has to show up in the log, not pass as the configured one.
        logger.warning("%s declined; %s answered this sample instead", cfg.llm.model_id, served)
        print(f"  ⚠ {cfg.llm.model_id} declined; {served} answered this sample (refusal fallback)")

    # With thinking enabled the first block is a thinking block — take the text.
    raw_text = next(
        (b.text for b in response.content if getattr(b, "type", None) == "text"),
        "",
    )

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

def _aggregate_decisions(
    runs: list[DecisionRun], requested: int | None = None,
) -> tuple[DecisionRun, dict[str, int]]:
    """
    Majority-vote aggregate of N DecisionRun objects.

    A ticker makes it into the consensus only if strictly more than half the runs
    agree on the same side (buy/sell/hold). Numeric fields are averaged across
    the agreeing runs; the thesis is taken from the highest-confidence agreeing run.

    "Half" is of the samples *requested*, not the ones that came back: with 3 of
    5 failing, two agreeing survivors are 2-of-5, not a majority.

    Returns (consensus_decision, {ticker: n_agreeing_runs}).
    """
    n = max(len(runs), requested or 0)
    threshold = n // 2 + 1  # e.g. 2 out of 3, 3 out of 5

    all_tickers: set[str] = {a.ticker for run in runs for a in run.actions}

    consensus_actions: list[Action] = []
    vote_counts: dict[str, int] = {}

    for ticker in sorted(all_tickers):
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

        consensus_actions.append(best_action.model_copy(update={
            "target_weight_pct": round(sum(a.target_weight_pct for a in agreeing) / len(agreeing), 1),
            "sek_estimate": round(sum(a.sek_estimate for a in agreeing) / len(agreeing), 0),
            "confidence": round(sum(a.confidence for a in agreeing) / len(agreeing), 3),
            "stop_loss_pct": round(sum(stop_losses) / len(stop_losses), 1) if stop_losses else None,
            "take_profit_pct": round(sum(take_profits) / len(take_profits), 1) if take_profits else None,
        }))
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

    if len(results) * 2 <= n:
        # A lone survivor of five used to pass through as the whole decision.
        raise LLMError(
            f"Only {len(results)}/{n} LLM call(s) succeeded — no majority is possible: "
            f"{'; '.join(e[:200] for e in errors)}"
        )
    if errors:
        print(f"  ⚠ {len(errors)}/{n} LLM call(s) failed; building consensus from {len(results)} run(s).")

    sampling = {
        "requested": n,
        "succeeded": len(results),
        "failed":    len(errors),
        "errors":    [e[:200] for e in errors],  # truncated for storage
        # Every individual run, not just the vote. The minority proposals are
        # the free counterfactuals, and agreement is a confidence signal that
        # cannot be measured once only the merged consensus survives.
        "samples":   [r.model_dump(mode="json") for r in results],
    }

    consensus, vote_counts = _aggregate_decisions(results, requested=n)
    sampling["vote_counts"] = dict(vote_counts)
    return consensus, consensus.model_dump_json(), vote_counts, sampling
