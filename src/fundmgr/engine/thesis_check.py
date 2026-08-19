"""
Verification of the reasoning behind a decision, separately from its return.

A 28-day price move against an index is a very noisy verdict on a decision: it
takes years of weekly runs to separate skill from variance at that signal level.
Whether the *claim* the thesis made came true is a different question, checkable
in weeks, and largely idiosyncratic rather than driven by the market factor that
dominates the return.

The four combinations are what make this worth recording. A thesis that held
while the position lagged is a timing or sizing problem. A thesis that broke
while the position won is luck — and, judged on return alone, it is
indistinguishable from skill, so it is the cell that quietly teaches the wrong
lesson. Feeding both signals to the distiller lets a lesson say which happened.

The hard requirement here is that a verdict rests on evidence. A model asked
"did this thesis come true" with nothing but the return in front of it will
answer from the return, which reproduces exactly the circularity this is meant
to break. So the evidence is the ticker's own news over the holding window, the
price move is withheld, and "unresolved" is stated to be the expected answer.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fundmgr.state.models import DecisionOutcome
from fundmgr.state.store import Store

if TYPE_CHECKING:
    from fundmgr.config import AppConfig
    from fundmgr.engine.schema import ThesisChecks

logger = logging.getLogger(__name__)

MAX_HEADLINES_PER_TICKER = 12

_SYSTEM_PROMPT = (
    "You are auditing the reasoning behind stock decisions made by a fund manager, "
    "four weeks after each was made. For each decision you are given the thesis "
    "stated at entry and the news published about that company since. You judge one "
    "thing only: did the specific claim the thesis made come true?\n\n"
    "Rules:\n"
    "1. You are NOT judging whether the decision made money. You are not told the "
    "return, and you must not guess it. A thesis can hold while the stock falls and "
    "break while it rises — separating those two is the entire purpose of this task.\n"
    "2. A verdict of 'held' or 'broke' requires evidence in the material below that "
    "bears on the claim. Cite it.\n"
    "3. 'unresolved' is the correct answer whenever the claim concerns something the "
    "material cannot show — most theses about margins, multi-quarter growth or "
    "valuation will be unresolved on four weeks of headlines. This is the most "
    "common verdict and returning it is doing the job properly, not failing it.\n"
    "4. A vague thesis that makes no falsifiable claim is 'unresolved'. Do not "
    "reconstruct a claim it did not make.\n"
    "5. Judge each decision only on its own company's evidence."
)


def verify_theses(
    store: Store,
    outcomes: list[DecisionOutcome],
    cfg: "AppConfig | None" = None,
    lookback_days: int = 28,
) -> dict[str, str]:
    """Judge each matured decision's thesis and persist the verdict.

    Returns {verdict: count} for logging. Outcomes with no recorded thesis, or
    with no news to judge against, are left unchecked rather than guessed at.
    """
    judgeable = [o for o in outcomes if (o.thesis or "").strip() and o.id]
    if not judgeable:
        return {}

    if cfg is None:
        from fundmgr.config import load_config
        cfg = load_config()

    evidence = {o.ticker: _news_window(store, o, lookback_days) for o in judgeable}
    # Nothing to judge against is not a verdict — say nothing rather than let the
    # model fall back on what it knows about the company in general.
    judgeable = [o for o in judgeable if evidence.get(o.ticker)]
    if not judgeable:
        logger.info("Thesis check: no news in the holding window for any decision — skipped")
        return {}

    parsed = _call_for_checks(cfg, _review_message(judgeable, evidence))
    if parsed is None:
        return {}

    by_ticker = {o.ticker.upper(): o for o in judgeable}
    counts: dict[str, str] = {}
    for check in parsed.checks:
        outcome = by_ticker.get(check.ticker)
        if outcome is None or not outcome.id:
            continue
        store.set_thesis_verdict(outcome.id, check.verdict, check.evidence)
        outcome.thesis_verdict = check.verdict
        outcome.thesis_evidence = check.evidence
        counts[check.verdict] = counts.get(check.verdict, 0) + 1

    return counts


def _news_window(store: Store, outcome: DecisionOutcome, lookback_days: int) -> list[dict]:
    """The ticker's cached news from the decision date on, newest first."""
    since = outcome.decision_date or ""
    if not since:
        return []
    try:
        items = store.get_recent_news(outcome.ticker, since_date=since)
    except Exception as exc:
        logger.warning("Thesis check: news lookup failed for %s: %s", outcome.ticker, exc)
        return []
    return items[:MAX_HEADLINES_PER_TICKER]


def _review_message(outcomes: list[DecisionOutcome], evidence: dict[str, list[dict]]) -> str:
    lines: list[str] = []
    for o in outcomes:
        lines.append(f"### {o.ticker} — {o.action.upper()} on {o.decision_date or 'unknown date'}")
        lines.append(f"Thesis stated at entry: {o.thesis}")
        lines.append("News published since:")
        for item in evidence.get(o.ticker, []):
            published = (item.get("published_at") or "")[:10]
            headline = item.get("headline") or ""
            summary = (item.get("summary") or "").strip()
            lines.append(f"  - [{published}] {headline}")
            if summary:
                lines.append(f"      {summary[:200]}")
        lines.append("")

    lines.append(
        "For each ticker above, return a verdict on whether its thesis came true, "
        "citing the evidence it rests on. Return 'unresolved' wherever the news does "
        "not settle the claim."
    )
    return "\n".join(lines)


def _call_for_checks(cfg: "AppConfig", user_msg: str) -> "ThesisChecks | None":
    from dataclasses import replace

    from fundmgr.engine.client import LLMError, call_llm
    from fundmgr.engine.schema import ThesisChecks

    # Same reasoning tier as the lesson writer: this judgement conditions what
    # the fund learns, and it runs once per evaluation batch.
    audit_cfg = replace(cfg, llm=replace(cfg.llm, model_id=cfg.learning_model, n_samples=1))

    try:
        parsed, _ = call_llm(_SYSTEM_PROMPT, user_msg, audit_cfg, schema=ThesisChecks)
    except LLMError as exc:
        logger.warning("Thesis check failed on %s: %s", audit_cfg.llm.model_id, exc)
        return None
    return parsed
