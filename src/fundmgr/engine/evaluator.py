"""
Retrospective evaluation of past decisions.
Runs automatically as part of fund run when outcomes are ≥28 days old.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import yfinance as yf

from fundmgr.data.benchmark import get_benchmark_return_pct
from fundmgr.state.models import DecisionOutcome, Learning
from fundmgr.state.store import Store


def evaluate_pending_outcomes(store: Store, lookback_days: int = 28) -> list[DecisionOutcome]:
    """
    For each pending outcome older than lookback_days, fetch current price,
    compute return vs benchmark, and persist. Returns the evaluated outcomes.
    """
    pending = store.get_pending_outcomes(older_than_days=lookback_days)
    if not pending:
        return []

    evaluated: list[DecisionOutcome] = []
    for outcome in pending:
        if not outcome.price_at_decision:
            continue

        try:
            current_price = yf.Ticker(outcome.ticker).fast_info.last_price
            if not current_price:
                continue
        except Exception:
            continue

        position_return = (current_price / outcome.price_at_decision - 1) * 100

        # Benchmark over the same window as the position: decision date -> now.
        if not outcome.decision_date:
            continue
        bench_return = get_benchmark_return_pct(store, since_date=outcome.decision_date)

        outperformed = None
        if bench_return is not None:
            outperformed = position_return > bench_return

        outcome.price_at_evaluation = current_price
        outcome.position_return_pct = round(position_return, 2)
        outcome.benchmark_return_pct = bench_return
        outcome.outperformed = outperformed
        outcome.evaluation_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        store.update_outcome(outcome)
        evaluated.append(outcome)

    return evaluated


def repair_outcomes(store: Store, dry_run: bool = False) -> dict[str, int]:
    """
    Repair decision_outcomes rows poisoned by the old seeding bug, which stored
    the trade's SEK estimate as price_at_decision (and, for evaluated rows,
    measured the benchmark since the start of the cache instead of the run date).

    For each row, the true decision-time share price is re-derived from the
    run's stored prompt snapshot. Evaluated rows are then recomputed over the
    correct window (decision date -> evaluation date); rows whose price can't be
    recovered get price_at_decision cleared so they are skipped, not mis-scored.
    """
    from fundmgr.engine.optimizer import price_from_snapshot

    stats = {"checked": 0, "price_fixed": 0, "recomputed": 0, "reset_pending": 0, "unrecoverable": 0}
    snapshots: dict[str, dict] = {}

    for outcome in store.get_all_outcomes():
        stats["checked"] += 1

        if outcome.run_id not in snapshots:
            rec = store.get_recommendation_by_run_id(outcome.run_id)
            try:
                snapshots[outcome.run_id] = json.loads(rec.prompt_snapshot) if rec else {}
            except json.JSONDecodeError:
                snapshots[outcome.run_id] = {}

        true_price = price_from_snapshot(snapshots[outcome.run_id], outcome.ticker)
        evaluated = outcome.outperformed is not None

        if true_price is None:
            # Can't recover the entry price — neutralise the row instead of
            # letting a wrong price keep producing wrong returns.
            if outcome.price_at_decision is not None or evaluated:
                stats["unrecoverable"] += 1
                if not dry_run:
                    store.set_outcome_price_at_decision(outcome.id, None)
                    if evaluated:
                        store.clear_outcome_evaluation(outcome.id)
            continue

        price_wrong = (
            outcome.price_at_decision is None
            or abs(outcome.price_at_decision - true_price) / true_price > 0.005
        )
        if price_wrong:
            stats["price_fixed"] += 1
            if not dry_run:
                store.set_outcome_price_at_decision(outcome.id, true_price)

        if not evaluated or not (price_wrong or outcome.benchmark_return_pct is None):
            continue

        # Recompute the evaluated row over the correct window.
        bench = get_benchmark_return_pct(
            store,
            since_date=outcome.decision_date or "2000-01-01",
            until_date=outcome.evaluation_date,
        )
        if outcome.price_at_evaluation and bench is not None:
            stats["recomputed"] += 1
            if dry_run:
                continue
            position_return = (outcome.price_at_evaluation / true_price - 1) * 100
            outcome.price_at_decision = true_price
            outcome.position_return_pct = round(position_return, 2)
            outcome.benchmark_return_pct = bench
            outcome.outperformed = position_return > bench
            store.update_outcome(outcome)
        else:
            # No usable evaluation-side data — send it back to pending so the
            # evaluator redoes it properly on the next run.
            stats["reset_pending"] += 1
            if not dry_run:
                store.clear_outcome_evaluation(outcome.id)

    return stats


def generate_qualitative_learnings(store: Store, outcomes: list[DecisionOutcome]) -> list[Learning]:
    """
    For each evaluated outcome, call GPT-4o-mini with the original thesis +
    macro context + actual return to produce a concrete, actionable lesson.
    """
    if not outcomes:
        return []

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return []

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except ImportError:
        return []

    # Group by run_id to fetch each rec log once
    by_run: dict[str, list[DecisionOutcome]] = {}
    for o in outcomes:
        by_run.setdefault(o.run_id, []).append(o)

    new_learnings: list[Learning] = []

    for run_id, run_outcomes in by_run.items():
        rec = store.get_recommendation_by_run_id(run_id)
        macro_summary = ""
        if rec:
            try:
                llm_data = json.loads(rec.llm_response)
                macro_summary = llm_data.get("market_summary", "")
            except Exception:
                pass

        for outcome in run_outcomes:
            learning_text = _call_gpt_for_learning(client, outcome, macro_summary)
            if not learning_text:
                continue

            learning = Learning(
                category="qualitative",
                body=learning_text,
                run_ids=[run_id],
                created_at=datetime.now(timezone.utc),
            )
            learning.id = store.save_learning(learning)
            new_learnings.append(learning)

    return new_learnings


def _call_gpt_for_learning(client, outcome: DecisionOutcome, macro_summary: str) -> str | None:
    """Ask GPT-4o-mini to write one actionable lesson from a completed trade."""
    ret = outcome.position_return_pct
    bench = outcome.benchmark_return_pct
    beat = "BEAT" if outcome.outperformed else "UNDERPERFORMED"
    alpha = f"{abs((ret or 0) - (bench or 0)):.1f}pp" if bench is not None else "unknown alpha"

    user_msg = (
        f"Trade: {outcome.action.upper()} {outcome.ticker}\n"
        f"Confidence at entry: {outcome.confidence:.0%}\n"
        f"Original thesis: {outcome.thesis or '(not recorded)'}\n"
        f"Macro context at entry: {macro_summary or '(not recorded)'}\n"
        f"Return over holding period: {ret:+.1f}%\n"
        f"OMXSPI benchmark same period: {bench:+.1f}%\n"
        f"Outcome: {beat} benchmark by {alpha}\n\n"
        "Write 1–2 sentences: what specifically worked or failed, and what concrete signal "
        "to look for (or avoid) next time. Reference the thesis or macro context — "
        "do not just restate the numbers."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a trading coach analysing the outcome of a single stock trade "
                        "made by an AI fund manager. Write a precise, actionable lesson for future trades. "
                        "Maximum 2 sentences. Be specific — generic advice is useless."
                    ),
                },
                {"role": "user", "content": user_msg},
            ],
            max_tokens=150,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


def generate_learnings(store: Store, min_sample: int = 5) -> list[Learning]:
    """
    Generate statistical calibration learnings from hit-rate data and save them.
    Returns newly created Learning objects.
    """
    stats = store.get_calibration_stats()
    new_learnings: list[Learning] = []

    for bucket, data in stats.items():
        if data["n"] < min_sample or data["hit_rate"] is None:
            continue

        hit_rate = data["hit_rate"]
        n = data["n"]

        if bucket == "high" and hit_rate < 0.5:
            body = (
                f"Your high-confidence buy calls (≥0.7 conviction) have a {hit_rate:.0%} hit rate "
                f"over {n} decisions — below the 50% breakeven threshold. "
                "Consider widening stop-losses or reducing position sizing on high-confidence calls."
            )
            category = "calibration"
        elif bucket == "high" and hit_rate >= 0.65:
            body = (
                f"Your high-confidence buy calls (≥0.7 conviction) have a strong {hit_rate:.0%} hit rate "
                f"over {n} decisions. Continue sizing up on high-conviction ideas."
            )
            category = "calibration"
        elif bucket == "low" and hit_rate > 0.5:
            body = (
                f"Your low-confidence buy calls (<0.4 conviction) are actually performing well "
                f"({hit_rate:.0%} hit rate over {n} decisions). "
                "You may be undersizing these — consider bumping conviction thresholds."
            )
            category = "calibration"
        else:
            continue

        existing = [l for l in store.get_active_learnings() if l.category == category]

        learning = Learning(
            category=category,
            body=body,
            run_ids=[],
            created_at=datetime.now(timezone.utc),
        )
        new_id = store.save_learning(learning)
        learning.id = new_id

        for old in existing:
            if old.id:
                store.supersede_learning(old.id, new_id)

        new_learnings.append(learning)

    return new_learnings
