"""
Retrospective evaluation of past decisions.
Runs automatically as part of fund run when outcomes are ≥28 days old.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import yfinance as yf

from fundmgr.data.benchmark import get_benchmark_return_pct
from fundmgr.state.models import DecisionOutcome, Learning
from fundmgr.state.store import Store


def evaluate_pending_outcomes(store: Store, lookback_days: int = 28) -> int:
    """
    For each pending outcome older than lookback_days, fetch current price,
    compute return vs benchmark, and persist. Returns count updated.
    """
    pending = store.get_pending_outcomes(older_than_days=lookback_days)
    if not pending:
        return 0

    updated = 0
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

        bench_return = get_benchmark_return_pct(store, since_date=outcome.evaluation_date or "2000-01-01")

        outperformed = None
        if bench_return is not None:
            outperformed = position_return > bench_return

        outcome.price_at_evaluation = current_price
        outcome.position_return_pct = round(position_return, 2)
        outcome.benchmark_return_pct = bench_return
        outcome.outperformed = outperformed
        outcome.evaluation_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        store.update_outcome(outcome)
        updated += 1

    return updated


def generate_learnings(store: Store, min_sample: int = 5) -> list[Learning]:
    """
    Generate plain-text learnings from calibration stats and save them.
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

        # Check if we already have an active calibration learning to supersede
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
