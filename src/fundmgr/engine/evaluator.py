"""
Retrospective evaluation of past decisions.
Runs automatically as part of fund run when outcomes are ≥28 days old.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import yfinance as yf

from fundmgr.data.benchmark import get_benchmark_return_pct
from fundmgr.state.models import DecisionOutcome, Learning
from fundmgr.state.store import Store

if TYPE_CHECKING:
    from fundmgr.config import AppConfig
    from fundmgr.engine.schema import BatchLessons

logger = logging.getLogger(__name__)


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
        outcome.horizon_days = lookback_days

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


MAX_BATCH_LESSONS = 3       # most a single evaluation batch may contribute
MIN_SUPPORTING_TICKERS = 2  # a lesson must repeat across names to count as one


def generate_qualitative_learnings(
    store: Store,
    outcomes: list[DecisionOutcome],
    cfg: "AppConfig | None" = None,
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

    Runs on `cfg.learning_model` — the provider's heavy reasoner, not a cheap
    one. This text conditions every subsequent decision the fund makes, and
    batching turned the cost from one call per trade into one call per batch,
    so the quality is worth far more here than the tokens saved.
    """
    if not outcomes:
        return []

    if cfg is None:
        from fundmgr.config import load_config
        cfg = load_config()

    by_run: dict[str, list[DecisionOutcome]] = {}
    for o in outcomes:
        by_run.setdefault(o.run_id, []).append(o)

    macro_by_run = {run_id: _macro_summary(store, run_id) for run_id in by_run}
    user_msg = _batch_review_message(
        by_run, macro_by_run, benchmark_label or cfg.benchmark, max_lessons
    )

    parsed = _call_for_batch_lessons(cfg, user_msg, cfg.evaluation_horizon_days)
    if parsed is None:
        return []

    new_learnings: list[Learning] = []
    for body, tickers in surviving_lessons(parsed, outcomes, max_lessons):
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
            # Paper books seed outcomes without a confidence — the weekly runs
            # always carry one, so this must degrade rather than raise.
            conf = f"{o.confidence:.0%}" if o.confidence is not None else "n/a"
            head = f"- {o.action.upper()} {o.ticker} | conf {conf}"
            if alpha is None:
                lines.append(f"{head} | alpha unknown")
            else:
                lines.append(
                    f"{head} | return {o.position_return_pct:+.1f}% | "
                    f"{bench_name} {o.benchmark_return_pct:+.1f}% | alpha {alpha:+.1f}pp"
                )
            lines.append(f"    thesis: {o.thesis or '(not recorded)'}")
            if o.thesis_verdict:
                lines.append(
                    f"    thesis {o.thesis_verdict.upper()}: {o.thesis_evidence or ''}"
                )
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


def _batch_system_prompt(horizon_days: int) -> str:
    return (
        "You are a trading coach reviewing a batch of completed trades made by an AI fund "
        "manager. You look for patterns that repeat across trades. You do not explain "
        "individual results.\n\n"
        "Rules:\n"
        "1. A single trade's " + str(horizon_days) + "-day return against an index is mostly "
        "noise. Never write a "
        "lesson resting on one trade.\n"
        "2. If most of the batch moved the same way, that is the market or the sector, not the "
        "theses. Do not convert a common move into a lesson about individual reasoning — the "
        "dispersion stats tell you when this is the case.\n"
        "3. The macro context is identical for every trade within a run. It therefore cannot "
        "explain why one trade in that run beat and another lagged. Never use the same macro "
        "factor to account for both a winner and a loser.\n"
        "4. A lesson must name a signal that was checkable before entry and specific enough to "
        "change a future decision. 'Monitor macro indicators' changes nothing and is not a lesson.\n"
        "5. Where a thesis verdict is shown, it says whether the reasoning held, judged on "
        "company news and not on the return. A thesis that HELD while the position lagged is a "
        "timing or sizing lesson, not a research one. A thesis that BROKE while the position "
        "beat is luck — never write a lesson endorsing the reasoning behind it. UNRESOLVED "
        "means the claim was not checkable, which is itself worth a lesson if theses are "
        "routinely written that way.\n"
        "6. Returning zero lessons is correct whenever the batch shows no repeated pattern. This "
        "is the expected result for most batches — an empty list is a better answer than a "
        "plausible story."
)


def _call_for_batch_lessons(
    cfg: "AppConfig", user_msg: str, horizon_days: int
) -> "BatchLessons | None":
    """One structured call on the fund's learning model. None on any failure.

    Runs through `call_llm` so this inherits the provider handling the decision
    path already has — structured outputs on OpenAI, JSON mode on Anthropic, and
    the temperature/token carve-outs each model family needs.
    """
    from dataclasses import replace

    from fundmgr.engine.client import LLMError, call_llm
    from fundmgr.engine.schema import BatchLessons

    # Same provider and credentials, heavy model, modest budget: the reply is at
    # most a few sentences however much reasoning went into it.
    learning_cfg = replace(
        cfg, llm=replace(cfg.llm, model_id=cfg.learning_model, n_samples=1)
    )

    try:
        parsed, _ = call_llm(
            _batch_system_prompt(horizon_days), user_msg, learning_cfg, schema=BatchLessons
        )
    except LLMError as exc:
        logger.warning("Learning distillation failed on %s: %s", learning_cfg.llm.model_id, exc)
        return None
    return parsed


def surviving_lessons(
    parsed: "BatchLessons", outcomes: list[DecisionOutcome], max_lessons: int
) -> list[tuple[str, set[str]]]:
    """
    (body, supporting tickers) for each returned lesson that clears the evidence
    bar: real tickers from this batch, at least MIN_SUPPORTING_TICKERS of them,
    and a non-empty body.

    Enforced here rather than trusted to the prompt or the schema — a model told
    "at least two tickers" will still cite one, or cite a name that was not in
    the batch, and a lesson resting on one 28-day return is the exact failure
    mode this pipeline exists to avoid.
    """
    known = {o.ticker.upper() for o in outcomes}
    kept: list[tuple[str, set[str]]] = []

    for lesson in parsed.lessons:
        body = (lesson.body or "").strip()
        tickers = set(lesson.tickers) & known
        if body and len(tickers) >= MIN_SUPPORTING_TICKERS:
            kept.append((body, tickers))

    return kept[:max_lessons]


CALIBRATION_MIN_SAMPLE = 20  # evaluated buys in a bucket before its rate is reported

_BUCKET_LABELS = {
    "high":   "high (>=0.7)",
    "medium": "medium (0.4-0.7)",
    "low":    "low (<0.4)",
}


def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a hit rate.

    Wilson rather than the normal approximation because these samples are small
    and the rates sit near the ends, where the naive interval runs past 0 and 1
    and badly understates how little a dozen trades tell you.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = hits / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _calibration_verdict(bands: dict[str, tuple[float, float]]) -> str:
    """What the intervals licence saying about conviction — and nothing more.

    Only a non-overlap between the high and low bands says anything at this
    sample size. Everything else is "not yet distinguishable", which is a
    finding worth stating: it stops the model reading its own confidence
    number as evidence.
    """
    high, low = bands.get("high"), bands.get("low")
    if not high or not low:
        return (
            "Too few decisions in the other conviction bands to compare them yet, so "
            "this rate says nothing about whether your conviction is informative."
        )
    if high[0] > low[1]:
        return (
            "High-conviction calls beat low-conviction ones with no overlap in their "
            "intervals, so your stated conviction is carrying real information — it is "
            "reasonable to let it drive sizing."
        )
    if high[1] < low[0]:
        return (
            "High-conviction calls do WORSE than low-conviction ones, with no overlap in "
            "their intervals. Your stated conviction is inverted: treat a strong prior as "
            "a reason to re-examine the thesis, not to size up."
        )
    return (
        "The intervals overlap, so stated conviction is not yet separating winners from "
        "losers. Set your confidence to reflect the evidence you actually have rather "
        "than to justify a position, and do not let it drive sizing until it does."
    )


def calibration_body(
    stats: dict, min_sample: int = CALIBRATION_MIN_SAMPLE, horizon_days: int = 28
) -> str | None:
    """One calibration lesson from hit-rate stats, or None when nothing is supported.

    Reports every conviction band that clears `min_sample`, each with its
    interval, and says only what the intervals licence. Deliberately makes no
    claim about a "breakeven" hit rate: a hit rate is the share of calls that
    beat the benchmark, and whether the strategy makes money depends on the size
    of the wins against the losses, which this statistic does not measure.
    """
    reported, bands = [], {}
    for bucket in ("high", "medium", "low"):
        data = stats.get(bucket) or {}
        n, hits = data.get("n", 0), data.get("hits")
        if n < min_sample or hits is None:
            continue
        lo, hi = wilson_interval(hits, n)
        bands[bucket] = (lo, hi)
        reported.append(
            f"{_BUCKET_LABELS[bucket]} {hits / n:.0%} [95% CI {lo:.0%}-{hi:.0%}, n={n}]"
        )

    if not reported:
        return None

    return (
        "Calibration of your buy calls, measured as the share that beat the benchmark "
        f"over {horizon_days} days: {'; '.join(reported)}. {_calibration_verdict(bands)} "
        "Note this counts only how often you were right, not by how much — it cannot "
        "tell you whether the book is profitable."
    )


def generate_learnings(
    store: Store, min_sample: int = CALIBRATION_MIN_SAMPLE, horizon_days: int = 28
) -> list[Learning]:
    """
    Refresh the fund's single calibration lesson from its hit-rate stats.

    One lesson covering every conviction band, not one per band. The per-band
    version keyed its supersede on `category`, which was "calibration" for all
    of them, so two bands qualifying in the same pass meant the second silently
    deactivated the first — the fund could never hold a reading on more than one
    band, and which one survived depended on dict order.

    When the evidence no longer supports any claim, the standing lesson is
    retired rather than left running: it is injected into every prompt, so an
    unsupported claim left active keeps steering decisions.
    """
    body = calibration_body(
        store.get_calibration_stats(), min_sample=min_sample, horizon_days=horizon_days
    )

    if body is None:
        store.deactivate_learnings(category="calibration")
        return []

    existing = [
        lrn for lrn in store.get_active_learnings() if lrn.category == "calibration"
    ]
    if any(lrn.body == body for lrn in existing):
        return []  # unchanged reading — keep the original timestamp

    learning = Learning(
        category="calibration",
        body=body,
        run_ids=[],
        created_at=datetime.now(timezone.utc),
    )
    learning.id = store.save_learning(learning)

    for old in existing:
        if old.id:
            store.supersede_learning(old.id, learning.id)

    return [learning]
