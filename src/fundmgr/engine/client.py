from __future__ import annotations

import json
import os
from datetime import datetime

from pydantic import ValidationError

from fundmgr.config import AppConfig
from fundmgr.engine.schema import DecisionRun


class LLMError(Exception):
    pass


def call_llm(system: str, user: str, cfg: AppConfig) -> tuple[DecisionRun, str]:
    """
    Call the configured LLM and return (parsed DecisionRun, raw response text).
    Uses OpenAI structured outputs when provider=openai, JSON-mode for Anthropic.
    Raises LLMError on failure.
    """
    if cfg.llm.provider == "openai":
        return _call_openai(system, user, cfg)
    elif cfg.llm.provider == "anthropic":
        return _call_anthropic(system, user, cfg)
    else:
        raise LLMError(f"Unknown LLM provider: {cfg.llm.provider!r}")


# ── OpenAI ────────────────────────────────────────────────────────────────────

def _call_openai(system: str, user: str, cfg: AppConfig) -> tuple[DecisionRun, str]:
    try:
        from openai import OpenAI
    except ImportError:
        raise LLMError("openai package not installed. Run: uv pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMError("OPENAI_API_KEY not set in environment / .env file")

    client = OpenAI(api_key=api_key)

    try:
        # Use structured outputs — guarantees schema conformance
        response = client.beta.chat.completions.parse(
            model=cfg.llm.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=DecisionRun,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
        )
    except Exception as e:
        raise LLMError(f"OpenAI API call failed: {e}") from e

    choice = response.choices[0]
    raw_text = choice.message.content or ""

    if choice.message.parsed is None:
        # Structured outputs failed — fall back to JSON parse
        try:
            parsed = DecisionRun.model_validate_json(raw_text)
        except (ValidationError, json.JSONDecodeError) as e:
            raise LLMError(f"Failed to parse LLM response: {e}\n\nRaw response:\n{raw_text}") from e
    else:
        parsed = choice.message.parsed

    return parsed, raw_text


# ── Anthropic ─────────────────────────────────────────────────────────────────

def _call_anthropic(system: str, user: str, cfg: AppConfig) -> tuple[DecisionRun, str]:
    try:
        import anthropic
    except ImportError:
        raise LLMError("anthropic package not installed. Run: uv pip install anthropic")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMError("ANTHROPIC_API_KEY not set in environment / .env file")

    client = anthropic.Anthropic(api_key=api_key)

    # Append explicit JSON schema instruction since Anthropic doesn't yet support
    # OpenAI-style structured outputs natively
    schema_hint = (
        "\n\nRespond with a single JSON object matching this schema:\n"
        + DecisionRun.model_json_schema().__repr__()
    )

    try:
        response = client.messages.create(
            model=cfg.llm.model_id,
            max_tokens=cfg.llm.max_tokens,
            system=system + schema_hint,
            messages=[{"role": "user", "content": user}],
            temperature=cfg.llm.temperature,
        )
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
        parsed = DecisionRun.model_validate_json(clean)
    except (ValidationError, json.JSONDecodeError) as e:
        raise LLMError(f"Failed to parse Anthropic response: {e}\n\nRaw response:\n{raw_text}") from e

    return parsed, raw_text
