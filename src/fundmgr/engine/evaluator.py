"""
Retrospective evaluation of past decisions.
Runs automatically as part of fund run when outcomes are ≥28 days old.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import yfinance as yf

from fundmgr.data.benchmark import get_benchmark_return_pct
from fundmgr.state.models import DecisionOutcome, Learning
from fundmgr.state.store import Store


def evaluate_pending_outcomes(store: Store, lookback_days: int = 28) -> list[DecisionOutcome]:
    """
    For each pending outcome older than lookback_days, compute return vs
    benchmark over a *fixed* horizon and persist. Returns the evaluated outcomes.

    The evaluation price is the cached close nearest to decision_date +
    lookback_days, so every outcome is a true `lookback_days` outcome regardless
    of when the run that evaluates it happens to fire. When the cache has no
    close near that target (thin history), it falls back to the live price at
    "now" — the previous behaviour — so an outcome is still recorded.
    """
    pending = store.get_pending_outcomes(older_than_days=lookback_days)
    if not pending:
        return []

    evaluated: list[DecisionOutcome] = []
    for outcome in pending:
        if not outcome.price_at_decision or not outcome.decision_date:
            continue

        eval_price, eval_date = _evaluation_price(store, outcome, lookback_days)
        if eval_price is None:
            continue

        position_return = (eval_price / outcome.price_at_decision - 1) * 100

        # Benchmark over the same window as the position: decision date -> eval date.
        bench_return = get_benchmark_return_pct(
            store, since_date=outcome.decision_date, until_date=eval_date
        )

        outperformed = None
        if bench_return is not None:
            outperformed = position_return > bench_return

        outcome.price_at_evaluation = eval_price
        outcome.position_return_pct = round(position_return, 2)
        outcome.benchmark_return_pct = bench_return
        outcome.outperformed = outperformed
        outcome.evaluation_date = eval_date

        store.update_outcome(outcome)
        evaluated.append(outcome)

    return evaluated


def _evaluation_price(
    store: Store, outcome: DecisionOutcome, horizon_days: int
) -> tuple[float | None, str]:
    """(price, date) to evaluate an outcome at: pinned cached close, else live price."""
    target = (
        datetime.strptime(outcome.decision_date, "%Y-%m-%d") + timedelta(days=horizon_days)
    ).strftime("%Y-%m-%d")

    near = store.close_near(outcome.ticker, target, max_delta_days=7)
    if near is not None:
        return near[1], near[0]

    # Fallback: live price at "now" (thin cache — better a rough outcome than none).
    try:
        current_price = yf.Ticker(outcome.ticker).fast_info.last_price
    except Exception:
        current_price = None
    return (
        (float(current_price), datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        if current_price
        else (None, "")
    )


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


_LEARNING_MODEL = "gpt-4o-mini"
MAX_BATCH_LESSONS = 3       # most a single evaluation batch may contribute
MIN_SUPPORTING_TICKERS = 2  # a lesson must repeat across names to count as one


def generate_qualitative_learnings(
    store: Store,
    outcomes: list[DecisionOutcome],
    benchmark_label: str | None = None,
    max_lessons: int = MAX_BATCH_LESSONS,
) -> list[Learning]:
    """
    Distil at most `max_lessons` lessons from a whole evaluation batch, in one call.

    Deliberately not one lesson per trade. A 28-day single-name return against
    the index is mostly noise, and a model asked to account for one such number
    in isolation will always find a story — in practice the run's macro summary,
    which every trade in that run shares, so the same variable ends up
    "explaining" the winners and the losers alike. Judging the batch together
    lets the model see when the names moved as one (that is beta, not a thesis
    error), and requiring `MIN_SUPPORTING_TICKERS` behind every lesson means a
    pattern has to repeat before it is written down. Returning nothing is a
    valid, and common, outcome.
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

    by_run: dict[str, list[DecisionOutcome]] = {}
    for o in outcomes:
        by_run.setdefault(o.run_id, []).append(o)

    macro_by_run = {run_id: _macro_summary(store, run_id) for run_id in by_run}
    user_msg = _batch_review_message(by_run, macro_by_run, benchmark_label, max_lessons)

    content = _call_gpt_for_batch_lessons(client, user_msg, max_lessons)
    if not content:
        return []

    new_learnings: list[Learning] = []
    for body, tickers in parse_batch_lessons(content, outcomes, max_lessons):
        run_ids = sorted({
            o.run_id for o in outcomes if o.ticker.upper() in tickers
        })
        learning = Learning(
            category="qualitative",
            body=body,
            run_ids=run_ids,
            created_at=datetime.now(timezone.utc),
        )
        learning.id = store.save_learning(learning)
        new_learnings.append(learning)

    return new_learnings


def _macro_summary(store: Store, run_id: str) -> str:
    rec = store.get_recommendation_by_run_id(run_id)
    if not rec:
        return ""
    try:
        return json.loads(rec.llm_response).get("market_summary", "")
    except Exception:
        return ""


def _alpha(o: DecisionOutcome) -> float | None:
    if o.position_return_pct is None or o.benchmark_return_pct is None:
        return None
    return o.position_return_pct - o.benchmark_return_pct


def _batch_review_message(
    by_run: dict[str, list[DecisionOutcome]],
    macro_by_run: dict[str, str],
    benchmark_label: str | None,
    max_lessons: int,
) -> str:
    """The batch as one table, plus the dispersion stats that expose a common move."""
    bench_name = benchmark_label or "the benchmark"
    lines: list[str] = []

    for run_id, run_outcomes in sorted(by_run.items()):
        macro = macro_by_run.get(run_id) or "(not recorded)"
        lines.append(f"### Run {run_id}")
        lines.append(f"Macro context at entry (SHARED by every trade in this run): {macro}")
        for o in sorted(run_outcomes, key=lambda x: (_alpha(x) is None, _alpha(x) or 0.0)):
            alpha = _alpha(o)
            head = f"- {o.action.upper()} {o.ticker} | conf {o.confidence:.0%}"
            if alpha is None:
                lines.append(f"{head} | alpha unknown")
            else:
                lines.append(
                    f"{head} | return {o.position_return_pct:+.1f}% | "
                    f"{bench_name} {o.benchmark_return_pct:+.1f}% | alpha {alpha:+.1f}pp"
                )
            lines.append(f"    thesis: {o.thesis or '(not recorded)'}")
        lines.append("")

    alphas = [a for a in (_alpha(o) for rl in by_run.values() for o in rl) if a is not None]
    if alphas:
        beat = sum(1 for a in alphas if a > 0)
        mean = sum(alphas) / len(alphas)
        spread = max(alphas) - min(alphas)
        lines.append(
            f"Batch dispersion: {len(alphas)} trades, {beat} beat {bench_name} and "
            f"{len(alphas) - beat} lagged; mean alpha {mean:+.1f}pp, spread {spread:.1f}pp."
        )
        lines.append("")

    lines.append(
        f"Identify at most {max_lessons} lessons that hold across this batch. "
        f"Each must be supported by at least {MIN_SUPPORTING_TICKERS} of the tickers above. "
        "If the batch shows no repeated, thesis-level pattern, return an empty list."
    )
    return "\n".join(lines)


_BATCH_SYSTEM_PROMPT = (
    "You are a trading coach reviewing a batch of completed trades made by an AI fund "
    "manager. You look for patterns that repeat across trades. You do not explain "
    "individual results.\n\n"
    "Rules:\n"
    "1. A single trade's 28-day return against an index is mostly noise. Never write a "
    "lesson resting on one trade.\n"
    "2. If most of the batch moved the same way, that is the market or the sector, not the "
    "theses. Do not convert a common move into a lesson about individual reasoning — the "
    "dispersion stats tell you when this is the case.\n"
    "3. The macro context is identical for every trade within a run. It therefore cannot "
    "explain why one trade in that run beat and another lagged. Never use the same macro "
    "factor to account for both a winner and a loser.\n"
    "4. A lesson must name a signal that was checkable before entry and specific enough to "
    "change a future decision. 'Monitor macro indicators' changes nothing and is not a lesson.\n"
    "5. Returning zero lessons is correct whenever the batch shows no repeated pattern. This "
    "is the expected result for most batches — an empty list is a better answer than a "
    "plausible story.\n\n"
    'Respond with JSON: {"lessons": [{"body": "<= 2 sentences", "tickers": ["AAA.ST", "BBB.ST"]}]}'
)


def _call_gpt_for_batch_lessons(client, user_msg: str, max_lessons: int) -> str | None:
    try:
        resp = client.chat.completions.create(
            model=_LEARNING_MODEL,
            messages=[
                {"role": "system", "content": _BATCH_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            max_tokens=100 + 160 * max_lessons,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


def parse_batch_lessons(
    content: str, outcomes: list[DecisionOutcome], max_lessons: int
) -> list[tuple[str, set[str]]]:
    """
    (body, supporting tickers) for each lesson the model returned that survives
    the evidence bar: real tickers from this batch, at least
    MIN_SUPPORTING_TICKERS of them, and a non-empty body.

    The bar is enforced here rather than trusted to the prompt — a model told
    "at least two tickers" will still cite one, and a lesson resting on one
    28-day return is the exact failure mode this pipeline exists to avoid.
    """
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return []

    known = {o.ticker.upper() for o in outcomes}
    kept: list[tuple[str, set[str]]] = []

    for entry in payload.get("lessons", []) or []:
        if not isinstance(entry, dict):
            continue
        body = str(entry.get("body", "") or "").strip()
        tickers = {
            str(t).upper() for t in (entry.get("tickers") or []) if isinstance(t, (str, int))
        } & known
        if body and len(tickers) >= MIN_SUPPORTING_TICKERS:
            kept.append((body, tickers))

    return kept[:max_lessons]


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
