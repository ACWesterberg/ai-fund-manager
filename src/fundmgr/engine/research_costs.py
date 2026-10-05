"""Offline optimizer accounting and evidence-based scheduling, without prices."""
from __future__ import annotations

from datetime import date
from pathlib import Path


def evidence_periods(examples, fund):
    periods = set()
    for example in examples:
        if example.get("source") != fund:
            continue
        try:
            periods.add(date.fromisoformat(str(example.get("run_id", ""))[:10]).isoformat())
        except ValueError:
            continue
    return sorted(periods)


def evidence_gate(cfg, plan):
    from fundmgr.engine.bounded_optimizer import _load
    if cfg.optimizer.min_new_periods < 1:
        raise ValueError("min_new_periods must be positive")
    seen, searches = set(), 0
    root = cfg.optimizer.compiled_dir / "searches" / cfg.db_path.stem
    for path in sorted(root.glob("*.json")):
        state = _load(path)
        if state["plan"].get("context_mode") == "compare" or not state.get("attempts"):
            continue
        searches += 1
        previous = state["plan"]
        seen.update(previous.get("evidence_periods", evidence_periods(
            previous.get("training_runs", []) + previous.get("cases", []), cfg.db_path.stem)))
    current = set(plan.get("evidence_periods", evidence_periods(
        plan["training_runs"] + plan["cases"], cfg.db_path.stem)))
    new = sorted(current - seen)
    return {"eligible": not searches or len(new) >= cfg.optimizer.min_new_periods,
            "new_periods": new, "required": cfg.optimizer.min_new_periods,
            "paid_searches": searches, "first_search": not searches}


def usage_report(root: Path):
    """Count paid attempts once, never charge a cached response to its consumer.

    Tokens are provider counters. Cache/reasoning counters are subsets, not
    additions to the input/output totals. Missing provider usage stays unknown.
    """
    from fundmgr.engine.bounded_optimizer import _load
    report = {"searches": 0, "attempts": 0, "local_cache_hits": 0,
              "unknown_usage_attempts": 0, "reserved_tokens": 0, "by_model": {}}
    if not root.exists():
        return report
    for path in sorted(root.glob("*/*.json")):
        state = _load(path)
        report["searches"] += 1
        report["local_cache_hits"] += sum(r.get("origin") == "cache" for r in state["results"].values())
        for attempt in state["attempts"]:
            report["attempts"] += 1
            report["reserved_tokens"] += attempt["reserved_tokens"]
            usage = attempt.get("usage")
            if not usage or usage.get("input_tokens") is None or usage.get("output_tokens") is None:
                report["unknown_usage_attempts"] += 1
                continue
            ident = state["plan"]["identity"]
            model = attempt.get("model") or (ident["prompt_model"] if attempt["key"] == "proposal" else ident["llm"]["model_id"])
            key = f"{ident['llm']['provider']}/{model}"
            totals = report["by_model"].setdefault(key, {"requests_with_usage": 0, "input_tokens": 0,
                "output_tokens": 0, "cached_input_tokens": 0, "cache_write_input_tokens": 0,
                "reasoning_tokens": 0})
            totals["requests_with_usage"] += 1
            for field in ("input_tokens", "output_tokens", "cached_input_tokens", "cache_write_input_tokens", "reasoning_tokens"):
                value = usage.get(field)
                # Older checkpoints retained raw OpenAI counters but omitted
                # cache writes in normalization. Recover without rewriting them.
                if (value is None and field == "cache_write_input_tokens"
                        and ident['llm']['provider'] == 'openai'):
                    value = ((usage.get('raw') or {}).get('prompt_tokens_details') or {}).get('cache_write_tokens')
                if value is not None:
                    if not isinstance(value, int) or value < 0:
                        raise ValueError(f"Invalid provider usage in {path}")
                    totals[field] += value
    return report
