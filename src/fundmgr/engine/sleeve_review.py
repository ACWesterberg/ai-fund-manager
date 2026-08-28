"""
Live-sleeve review — "given where this book is now, what do I do with it?"

A sleeve imported through `fund paper-import` carries exactly one decision: the
plan it was created from. Nothing ever writes a second one, so its dashboard
shows the import-time picks forever while the daily watches (kill criteria,
capex trigger, earnings, drift) only ever alert to Telegram. This module closes
that loop: it re-runs the real decision pipeline against the sleeve's *current*
positions and writes the result back into the sleeve's own store, where the
existing dashboard panel picks it up unchanged.

Two things a sleeve doesn't have, and where they come from:

  • A universe. A sleeve only caches price history for what it holds, so it
    cannot propose new names on its own. Each review borrows a *source profile*
    (one of the config/*.yaml funds) for its universe, mandate and risk limits.
    The choice is remembered on the sleeve after the first review, and is
    overridable per run.

  • A geography. Every universe row already carries a `country`, so a review is
    scoped either globally or to one nation. Scope is not just a filter: it
    decides how many candidates can be brought to a buyable freshness in one
    run (guardrails reject buys on data older than risk.stale_after_days), so
    a single-country review covers its market properly while a global one leans
    on whatever the source profile's weekly rotation has recently refreshed.

What the model is shown: the sleeve's own standing work — kill criterion and
numeric kill rules, logged kill signals and the last judged verdict, the add
criterion, target and review prices, the add-signal gates as computed by
`addsignal` (the same numbers the dashboard's Add-signals panel shows), the
reported figures standing behind each fundamentals kill rule so the line can
actually be checked rather than taken on trust, review horizons, plan weights
and drift — plus what was previously claimed about each
name and whether `thesis_check` found the claim survived, and the book's own
held/broke against beat/lagged record as calibration. All of it is evidence, none of it binding: a
target price the model disagrees with is a number it must argue against in the
thesis, not a gate that silently blocks a trade. Guardrails remain the only
hard constraint.

A review writes to the book, not only to its decision log: stop and take-profit
levels, and for a name it opens the kill and add criteria, target price and
sizing plan that the watches and the add gates read. So dismissing a decision
unwinds those writes too (`dismiss` / `revert_plan_writes`) — but only where the
value is still exactly what the decision left, since an edit you made afterwards
is yours to keep.

Funding model: the sleeve's NAV is fixed — no new capital. A buy has to be paid
for out of the book, so guardrails see a snapshot with the run's own sells
already settled (`_fund_from_sells`). That is what makes "sell A, add B" pass
the cash floor in a fully-deployed sleeve instead of being rejected outright.
"""
from __future__ import annotations

import copy
import json
import time
import uuid
from datetime import datetime, timedelta
from functools import lru_cache

from fundmgr.config import AppConfig, UniverseTicker, get_enabled_tickers
from fundmgr.data.fundamentals import apply_to_features
from fundmgr.data.news import (
    attach_sentiment_to_features, fetch_news, score_and_cache_sentiment,
)
from fundmgr.data.prices import build_all_features, fetch_and_cache_prices
from fundmgr.data.screener import screen
from fundmgr.data.universe_selection import tickers_for_feature_build
from fundmgr.engine.client import call_llm_consensus
from fundmgr.engine.prompt import build_prompt, snapshot_to_dict
from fundmgr.engine.whatif import MAX_RUNS, MODEL_OPTIONS, load_profile_config, list_profiles
from fundmgr.guardrails.rules import apply_guardrails
from fundmgr.levels import merged_levels
from fundmgr.state.models import PortfolioSnapshot, RecommendationLog
from fundmgr.state.store import Store

# Profile a sleeve reviews against when nothing else is stored: the widest
# universe in the repo, so add-on suggestions aren't boxed into one region.
DEFAULT_REVIEW_CONFIG = "config_global.yaml"

# Ceiling on candidate tickers pulled into a sleeve's own price cache per run.
# Sized so one mid-size country (SE 949, GB 3797 rotate; NO/FI/DK fit whole)
# is covered in a single pass without mirroring a 17k universe into every
# sleeve DB.
DEFAULT_MAX_CANDIDATES = 750

META_CONFIG = "paper_review_config"
META_COUNTRY = "paper_review_country"
META_RISK = "paper_review_risk"

# Risk caps a sleeve may carry its own value for. Everything else — sector caps,
# minimum trade size, staleness — stays the source profile's, because those are
# properties of the market and the broker rather than of this book's mandate.
#
# max_turnover_pct is the one that actually bites. A weekly rebalance drifts a
# book; a sleeve review swaps positions, and a swap costs turnover twice — an
# 18% exit plus an 18% replacement is 36% against a profile cap of 25%. Left
# inherited, the cap truncates exactly the paired trades a review exists to
# produce, and the funding pass then drops the orphaned buy, so the review
# returns nothing at all.
OVERRIDABLE_RISK = ("max_turnover_pct", "max_position_pct", "max_positions", "min_cash_pct")

COUNTRY_NAMES = {
    "AT": "Austria", "BE": "Belgium", "CA": "Canada", "CH": "Switzerland",
    "DE": "Germany", "DK": "Denmark", "ES": "Spain", "FI": "Finland",
    "FR": "France", "GB": "United Kingdom", "IE": "Ireland", "IT": "Italy",
    "NL": "Netherlands", "NO": "Norway", "PL": "Poland", "PT": "Portugal",
    "SE": "Sweden", "US": "United States",
}


def _country_label(code: str) -> str:
    return COUNTRY_NAMES.get(code, code)


# ── Scope ─────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=8)
def list_scopes(config_name: str = DEFAULT_REVIEW_CONFIG) -> tuple[dict, ...]:
    """Countries selectable for a review of this profile's universe.

    Only well-formed ISO-alpha-2 codes are offered — some universe rows carry a
    blank or an ISIN in the country column, and those names stay reachable via
    the global scope rather than under a junk heading of their own.

    Cached: universes are static CSVs and the global one is 17k rows, which is
    not something to re-parse on every dashboard render.
    """
    counts: dict[str, int] = {}
    for t in get_enabled_tickers(load_profile_config(config_name).universe_path):
        code = (t.country or "").strip().upper()
        if len(code) == 2 and code.isalpha():
            counts[code] = counts.get(code, 0) + 1
    return tuple(
        {"code": code, "label": _country_label(code), "count": n}
        for code, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )


def review_defaults(store: Store) -> dict:
    """This sleeve's stored review scope and risk overrides, else the global
    profile's."""
    config_name = store.get_meta(META_CONFIG) or DEFAULT_REVIEW_CONFIG
    if config_name not in {p["config"] for p in list_profiles()}:
        config_name = DEFAULT_REVIEW_CONFIG
    return {
        "config": config_name,
        "country": store.get_meta(META_COUNTRY) or "",
        "risk": stored_risk(store),
    }


def clean_risk(raw: dict | None) -> dict:
    """Keep only known, numeric, positive risk caps.

    This is the boundary between user input and a guardrail, so anything
    unrecognised or unparseable is dropped rather than carried forward: a
    typo'd cap must fall back to the profile's, never reach apply_guardrails
    as a string or a zero that silently forbids every trade.
    """
    out: dict = {}
    for key in OVERRIDABLE_RISK:
        value = (raw or {}).get(key)
        if value is None or value == "":
            continue
        try:
            parsed = int(value) if key == "max_positions" else float(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            out[key] = parsed
    return out


def stored_risk(store: Store) -> dict:
    """This sleeve's own risk caps, as far as it sets any."""
    try:
        return clean_risk(json.loads(store.get_meta(META_RISK) or "{}"))
    except (ValueError, TypeError):
        return {}


def apply_risk_overrides(cfg: AppConfig, overrides: dict) -> tuple[AppConfig, dict]:
    """cfg with this sleeve's own caps in place of the profile's.

    Returns (cfg, applied) where `applied` names only the caps that actually
    changed something, so a review can report what it ran under rather than
    leaving the operator to infer it from the profile.
    """
    clean = clean_risk(overrides)
    if not clean:
        return cfg, {}
    cfg = copy.copy(cfg)
    cfg.risk = copy.copy(cfg.risk)
    applied = {}
    for key, value in clean.items():
        before = getattr(cfg.risk, key)
        value = int(value) if key == "max_positions" else float(value)
        if value != before:
            applied[key] = {"from": before, "to": value}
        setattr(cfg.risk, key, value)
    return cfg, applied


# ── Candidate selection ───────────────────────────────────────────────────────

def _scoped_universe(cfg: AppConfig, country: str | None) -> list[UniverseTicker]:
    tickers = get_enabled_tickers(cfg.universe_path)
    if not country:
        return tickers
    want = country.strip().upper()
    scoped = [t for t in tickers if (t.country or "").strip().upper() == want]
    if not scoped:
        raise ValueError(
            f"No enabled tickers for country {want!r} in {cfg.universe_path.name} — "
            f"pick another scope or a profile whose universe covers it."
        )
    return scoped


def _select_candidates(
    universe: list[UniverseTicker],
    must_have: set[str],
    source_store: Store,
    max_candidates: int,
) -> tuple[list[UniverseTicker], str]:
    """Pick which universe rows to work this run, warm names first.

    Price history is shared across funds, so a name the source profile refreshed
    recently costs nothing to bring in and is fresh enough to clear the staleness
    guardrail; a cold name costs a fetch and may still land stale. Ordering warm
    before cold means a capped run spends its budget where buys can actually be
    approved, and a whole small country still fits under the cap either way.
    """
    by_yahoo = {t.yahoo_ticker: t for t in universe}
    held = [by_yahoo[y] for y in sorted(must_have & by_yahoo.keys())]

    pool = [t for t in universe if t.yahoo_ticker not in must_have]
    if len(pool) <= max_candidates:
        return held + pool, f"full scope ({len(pool)} candidates)"

    warm = source_store.tickers_with_price_cache([t.yahoo_ticker for t in pool])
    ordered = [t for t in pool if t.yahoo_ticker in warm]
    ordered += [t for t in pool if t.yahoo_ticker not in warm]
    kept = ordered[:max_candidates]
    n_warm = sum(1 for t in kept if t.yahoo_ticker in warm)
    return (
        held + kept,
        f"capped at {max_candidates} of {len(pool)} ({n_warm} already warm in "
        f"{source_store.db_path.name})",
    )


# ── Sleeve context ────────────────────────────────────────────────────────────

def _kill_hits(store: Store, ticker: str) -> list[str]:
    with store._conn() as conn:
        rows = conn.execute(
            "SELECT key, value FROM app_meta WHERE key LIKE ? ORDER BY key DESC LIMIT 3",
            (f"paper_killhit:{ticker}:%",),
        ).fetchall()
    return [f"{r['key'].rsplit(':', 1)[1]}: {r['value']}" for r in rows]


def _kill_verdict(store: Store, ticker: str) -> dict:
    """The judge's most recent read of one kill criterion, or {}."""
    try:
        return json.loads(store.get_meta(f"paper_killverdict:{ticker}") or "{}")
    except (ValueError, TypeError):
        return {}


def _rule_line(store: Store, ticker: str, rule: dict) -> str:
    """One position's numeric kill rule, with the reported figures behind it.

    A threshold on its own is untestable from the prompt: the model could read
    "gross_margin below 45 for 2 consecutive quarters" and have no idea whether
    it is tripping. `evidence.sustained_breach` answers the rule exactly as
    written — streak counted from the newest period back, thin history never a
    pass — so the reading travels with the line.
    """
    parts = []
    if rule.get("max_drawdown_pct"):
        anchor = rule.get("anchor_price_sek")
        anchored = f" from {anchor:,.0f} SEK ({rule.get('anchor_date', '?')})" if anchor else ""
        parts.append(f"drawdown past {rule['max_drawdown_pct']:.0f}%{anchored}")
    if rule.get("price_below"):
        parts.append(f"price below {rule['price_below']:,.2f} {rule.get('currency', '')}".strip())
    if rule.get("price_above"):
        parts.append(f"price above {rule['price_above']:,.2f} {rule.get('currency', '')}".strip())
    for f in rule.get("fundamentals") or []:
        if not isinstance(f, dict) or not f.get("metric"):
            continue
        quarters = int(f.get("quarters") or 1)
        span = f" for {quarters} consecutive quarters" if quarters > 1 else ""
        value = f.get("value")
        shown = f"{value:g}" if isinstance(value, (int, float)) else str(value or "")
        line = f"{f['metric']} {f.get('op', '')} {shown}{span}".strip()
        parts.append(line + _breach_note(store, ticker, f))
    return " · ".join(parts)


def _breach_note(store: Store, ticker: str, condition: dict) -> str:
    """Where one fundamentals condition currently stands, in reported figures.

    Silence would be read as "fine", so an unmeasurable rule says so: no
    periods on file is *unknown*, which is a different thing from a rule that
    has been checked and cleared.
    """
    try:
        from fundmgr import evidence
        state = evidence.sustained_breach(
            store, ticker, condition["metric"], condition.get("op", "below"),
            float(condition.get("value") or 0), int(condition.get("quarters") or 1),
        )
        history = evidence.metric_history(store, ticker, condition["metric"])[:3]
    except Exception:
        return ""

    if not state["periods_seen"]:
        return " [no reported periods on file — this rule cannot be checked]"
    readings = ", ".join(f"{r['period']} {r['value']:g}" for r in history)
    verdict = ("BREACHED" if state["hit"]
               else f"streak {state['streak']} of {state['quarters']}")
    return f" [{verdict}; {readings}]"


def _add_rows(store: Store) -> dict[str, dict]:
    """Add-signal assessment per ticker — the same computation the dashboard's
    Add-signals panel renders. Never fatal: a review is still worth having
    without it, and this reaches out to prices and fundamentals."""
    try:
        from fundmgr import addsignal
        return {r["ticker"]: r for r in addsignal.evaluate_all(store)}
    except Exception:
        return {}


_VERDICT_LABEL = {
    "held": "HELD",
    "broke": "BROKE",
    "unresolved": "unresolved (the evidence did not settle it)",
}


def _thesis_history(store: Store, ticker: str, limit: int = 2) -> list[str]:
    """What was previously claimed about this name, and whether it survived.

    `thesis_check` audits a matured decision's *claim* against the news in its
    holding window, deliberately without seeing the return — so a verdict here
    is independent of whether the position made money. That makes it the one
    piece of history worth putting in front of a fresh decision: a thesis that
    broke twice on the same name is a pattern, and one that held while the
    position lagged says the read was right and the sizing or timing was not.
    """
    out = []
    for d in store.get_decisions_for_ticker(ticker, limit=limit):
        thesis = (d.get("thesis") or "").strip()
        if not thesis:
            continue
        when = str(d.get("timestamp") or "")[:10]
        conf = d.get("confidence")
        head = (f"      past call {when} {str(d.get('action', '')).upper()}"
                + (f" (conf {conf:.2f})" if conf is not None else "") + ": ")
        out.append(head + f"\"{thesis[:200]}\"")
        verdict = d.get("thesis_verdict")
        if verdict:
            label = _VERDICT_LABEL.get(verdict, verdict)
            evidence = (d.get("thesis_evidence") or "").strip()
            out.append(f"          → thesis {label}"
                       + (f" — {evidence[:200]}" if evidence else ""))
    return out


def _thesis_record(store: Store) -> list[str]:
    """The book's own thesis-vs-return cross-tab, as calibration.

    Held-and-lagged and broke-and-beat are the cells that matter: judged on
    return alone the second is indistinguishable from skill, and a model that
    cannot see the split will keep learning the wrong lesson from it.
    """
    try:
        stats = store.get_thesis_stats()
    except Exception:
        return []
    if not stats.get("n"):
        return []

    cells = stats["cells"]
    parts = []
    for verdict in ("held", "broke", "unresolved"):
        cell = cells.get(verdict)
        if cell:
            parts.append(f"{verdict} {cell['beat']} beat / {cell['lagged']} lagged")
    lines = [
        "### This book's own thesis record (audited without sight of the return)",
        "  " + " · ".join(parts),
    ]
    rate = stats.get("hold_rate")
    if rate is not None:
        lines.append(f"  Reasoning held on {rate*100:.0f}% of {stats['resolved']} "
                     f"resolved calls.")
    lines += [
        "  A thesis that held while the position lagged was a sizing or timing "
        "error, not a bad read; one that broke while the position won was luck. "
        "Weigh your own confidence accordingly.",
        "",
    ]
    return lines


def _add_lines(row: dict) -> list[str]:
    """One position's add plan and where each gate currently stands.

    The gates are reported as computed, pass and fail alike — a HOLD whose only
    blocker is a stale target is a different situation from one the valuation
    genuinely rejects, and the model can only tell those apart if it sees which
    gate stopped it.
    """
    out = []
    if row.get("add_criterion"):
        out.append(f"      add criterion: {row['add_criterion']}")
    if row.get("target_price"):
        stale = " ⚠ STALE — a material event postdates it" if row.get("valuation_status") == "stale" else ""
        out.append(f"      target price: {row['target_price']:,.2f} "
                   f"(set {row.get('target_set_at') or '?'}){stale}")
    if row.get("expected_annual_return") is not None:
        out.append(f"      implied return: {row['expected_annual_return']:.0f}%/yr over "
                   f"{row.get('months_left') or 0:.0f}m — gate {row.get('return_gate', 0):.0f}% "
                   f"({'clears' if row.get('valuation_ok') else 'does not clear'})")
    if row.get("dislocation_pct") is not None:
        out.append(f"      vs review price {row.get('review_price') or 0:,.2f}: "
                   f"{row['dislocation_pct']:+.1f}% — gate {row.get('dislocation_gate', 0):+.0f}% "
                   f"({'dislocated' if row.get('dislocated') else 'not dislocated'})")
    if row.get("proof_status"):
        # proof_reason usually already names the confirmation date; only fall
        # back to the raw stamp when it doesn't, so the line never says it twice.
        detail = row.get("proof_reason") or (
            f"confirmed {row['proof_confirmed_at']}" if row.get("proof_confirmed_at") else "")
        out.append(f"      proof: {row['proof_status']}" + (f" — {detail}" if detail else ""))
    if row.get("max_weight_pct"):
        out.append(f"      add plan: book {row.get('book', '?')}, "
                   f"max weight {row['max_weight_pct']:.1f}%, "
                   f"tranche +{row.get('tranche_pct') or 0:.1f}pp, "
                   f"room {row.get('weight_room') or 0:.1f}pp")
    if row.get("state"):
        out.append(f"      ADD SIGNAL: {row['state'].upper()}"
                   + (f" — {row['why']}" if row.get("why") else ""))
    return out


def _sleeve_block(meta: dict, store: Store, snap: PortfolioSnapshot) -> str:
    """The monitoring state the daily watches alert on, as prompt context.

    Same stored metadata the dashboard's Watch panel renders, so what the model
    reasons over and what you see on the page can't drift apart.
    """
    from fundmgr import watchplan

    targets = json.loads(store.get_meta("paper_target_weights") or "{}")
    kills = watchplan.get_kill_text(store)
    notes = json.loads(store.get_meta("paper_position_notes") or "{}")
    capex = json.loads(store.get_meta("paper_capex_kill") or "{}")
    rules = watchplan.get_kill_rules(store)
    horizons = watchplan.get_horizons(store)
    adds = _add_rows(store)

    lines = [
        f"## This Sleeve — {meta['name']}",
        "Real positions held at a broker, imported as a plan and filled by hand. "
        "The plan below is the thesis you are reviewing, not a constraint: a target "
        "weight is what was originally intended, and departing from it is a decision "
        "you may take and must justify.",
        "",
    ]

    if meta.get("base_prompt"):
        lines += ["### Original sleeve mandate", meta["base_prompt"].strip(), ""]

    if targets or kills or adds:
        lines.append("### Plan, live weight, and the criteria you set for each name")
        held = {p.ticker: p for p in snap.positions if p.shares > 0}
        for ticker in sorted(set(targets) | set(kills) | set(held) | set(adds)):
            target = targets.get(ticker)
            weight = snap.weight_pct(ticker) if ticker in held else None
            parts = [f"  {ticker:<14}"]
            parts.append(f"target {target:>5.1f}%" if target is not None else "target    —")
            parts.append(f"live {weight:>5.1f}%" if weight is not None else "live    — (not filled)")
            if target and weight is not None and target > 0:
                drift = weight / target
                if drift >= 1.5 or drift <= 0.5:
                    parts.append(f"⚠ drift {drift:.2f}× target")
            lines.append("  ".join(parts))
            if kills.get(ticker):
                lines.append(f"      kill criterion: {kills[ticker]}")
            rule_text = _rule_line(store, ticker, rules.get(ticker) or {})
            if rule_text:
                lines.append(f"      kill rule: {rule_text}")
            verdict = _kill_verdict(store, ticker)
            if verdict.get("verdict"):
                lines.append(f"      last kill check ({verdict.get('date', '?')}): "
                             f"{verdict['verdict']}"
                             + (f" — {verdict['reason']}" if verdict.get("reason") else ""))
            for hit in _kill_hits(store, ticker):
                lines.append(f"      ⚠ KILL SIGNAL LOGGED — {hit}")
            lines += _add_lines(adds.get(ticker) or {})
            lines += _thesis_history(store, ticker)
            horizon = horizons.get(ticker) or {}
            if horizon.get("review_date"):
                lines.append(f"      review horizon: {horizon['review_date']}"
                             + (f" — {horizon['label']}" if horizon.get("label") else ""))
            if notes.get(ticker, {}).get("cluster"):
                lines.append(f"      cluster: {notes[ticker]['cluster']}")
            nxt = notes.get(ticker, {}).get("next_earnings")
            if nxt:
                lines.append(f"      next earnings: {nxt}")
        lines.append("")
        lines += [
            "These criteria, target prices and add signals are the operator's own "
            "standing work, computed the same way the dashboard computes them — not "
            "instructions you must follow. Use them as evidence. Where you disagree "
            "with a target price, a gate, or an ADD/HOLD signal, say so explicitly in "
            "the thesis and give your reason; an unexplained departure from a level "
            "the operator set is worse than no recommendation.",
            "",
        ]

    lines += _thesis_record(store)

    if capex.get("trigger"):
        status = store.get_meta("paper_capex_status") or "none"
        lines += [
            "### Portfolio-level capex kill criterion",
            f"  Trigger: {capex['trigger']}",
            f"  Action if triggered: {capex.get('action', '—')}",
            f"  Current status: {status}",
            "",
        ]
    return "\n".join(lines).rstrip()


def _task_block(run_id: str, scope_label: str, snap: PortfolioSnapshot, cfg: AppConfig) -> str:
    turnover = snap.nav_sek * cfg.risk.max_turnover_pct / 100
    return (
        "## Your Task\n"
        "Decide what to do with this sleeve *now*. Three jobs, one JSON answer:\n"
        "  1. Every position above: hold, trim (sell to a lower target weight) or "
        "exit (sell to 0). A kill signal logged against a name is a strong prior to "
        "exit, but you own the call — say so in the thesis either way. Where a past "
        "thesis on the name was audited, that verdict is about the claim and not the "
        "return: a thesis that BROKE is reason to distrust a restatement of it, and "
        "one that HELD while the position lagged argues for sizing or patience "
        "rather than for exiting.\n"
        "  2. Adding to a name already held, where its add criterion, target price "
        "and gates support it. An ADD or STRONG-ADD signal is a prior in favour, a "
        "HOLD one against, and a stale target price means the valuation gate could "
        "not be evaluated at all rather than that it failed — treat that as missing "
        "evidence, not as a rejection.\n"
        f"  3. Add-ons from the candidate universe below ({scope_label}). A new name "
        "must earn its place against what it displaces — including against adding to "
        "a name you already own and already understand.\n\n"
        "Every name you open must arrive with its monitoring plan: kill_criterion, "
        "add_criterion, target_price, max_weight_pct and tranche_pct, plus "
        "next_earnings where you know it. These are not paperwork — the daily "
        "watches, the add gates and the next review all read them, and a position "
        "without them is one nothing can watch and nothing can later justify adding "
        "to. Write both criteria so they could be checked against a report or a "
        "headline rather than re-argued: name the metric and the level. Set "
        "target_price in the stock's own trading currency, at the level your thesis "
        "actually implies, since the expected return it gives is what any later add "
        "is measured against.\n\n"
        f"There is NO new capital. NAV is {snap.nav_sek:,.0f} SEK and stays there, so "
        f"every buy is funded from cash on hand ({snap.cash_sek:,.0f} SEK) plus the "
        "proceeds of the sells you recommend in this same run. Recommending a buy "
        "without freeing the money for it is not a plan — pair them.\n"
        f"Total traded value this run is capped at {cfg.risk.max_turnover_pct:.0f}% of NAV "
        f"(≈{turnover:,.0f} SEK); trades beyond the cap are dropped lowest-confidence "
        "first, so lead with your best ideas.\n\n"
        f"Return a DecisionRun JSON. Run ID must be: {run_id}"
    )


# ── Undoing a decision you didn't act on ──────────────────────────────────────

# Book state a review writes, all of it {ticker: value} maps in app_meta. The
# stop levels live in their own table and are handled alongside.
_PLAN_KEYS = (
    "paper_kill_criteria", "paper_add_criteria", "paper_add_plan",
    "paper_target_weights", "paper_position_notes",
)
_REVERT_KEY = "paper_review_revert"


def _plan_state(store: Store, tickers: set[str]) -> dict:
    """The book's plan and levels for these tickers, as they stand now."""
    out: dict = {}
    for key in _PLAN_KEYS:
        try:
            data = json.loads(store.get_meta(key) or "{}")
        except (ValueError, TypeError):
            data = {}
        out[key] = {t: data.get(t) for t in tickers}
    stops = store.get_position_stops()
    out["stops"] = {t: stops.get(t) for t in tickers}
    return out


def _record_revert(store: Store, run_id: str, before: dict, after: dict) -> None:
    store.set_meta(f"{_REVERT_KEY}:{run_id}", json.dumps({"before": before, "after": after}))


def revert_plan_writes(store: Store, run_id: str, forward: bool = False) -> list[str]:
    """Undo the plan and levels a decision wrote — or re-apply them.

    Only where the current value is still exactly what the decision left. If you
    have since edited a criterion by hand, that edit is yours and stays: a
    dismissal is a statement about the decision, not a licence to overwrite work
    done after it. Returns the tickers actually changed.
    """
    try:
        saved = json.loads(store.get_meta(f"{_REVERT_KEY}:{run_id}") or "{}")
    except (ValueError, TypeError):
        return []
    source, target = ("before", "after") if forward else ("after", "before")
    expected, wanted = saved.get(source) or {}, saved.get(target) or {}
    if not wanted:
        return []

    touched: set[str] = set()
    for key in _PLAN_KEYS:
        want, have_expected = wanted.get(key) or {}, expected.get(key) or {}
        if not want and not have_expected:
            continue
        try:
            data = json.loads(store.get_meta(key) or "{}")
        except (ValueError, TypeError):
            continue
        changed = False
        for ticker in set(want) | set(have_expected):
            if data.get(ticker) != have_expected.get(ticker):
                continue                      # edited since — leave it alone
            value = want.get(ticker)
            if value is None:
                data.pop(ticker, None)
            else:
                data[ticker] = value
            changed = True
            touched.add(ticker)
        if changed:
            store.set_meta(key, json.dumps(data))

    stops_now = store.get_position_stops()
    for ticker, level in (wanted.get("stops") or {}).items():
        if stops_now.get(ticker) != (expected.get("stops") or {}).get(ticker):
            continue
        if level is None:
            store.clear_position_stop(ticker)
        else:
            store.set_position_stop(ticker, stop_pct=level.get("stop_pct"),
                                    take_profit_pct=level.get("take_profit_pct"))
        touched.add(ticker)
    return sorted(touched)


def dismiss(store: Store, run_id: str, reason: str = "") -> tuple[bool, list[str]]:
    """Mark a decision as not acted on and undo what it wrote to the book.

    Returns (ok, tickers_reverted). The decision itself stays on file and keeps
    being scored — only its side effects on the plan are unwound, because a
    criterion or a target price installed by a call you declined would otherwise
    keep steering the watches.
    """
    if not store.dismiss_recommendation(run_id, reason):
        return False, []
    return True, revert_plan_writes(store, run_id)


def restore(store: Store, run_id: str) -> tuple[bool, list[str]]:
    """Undo a dismissal, putting back the plan the decision installed."""
    if not store.restore_recommendation(run_id):
        return False, []
    return True, revert_plan_writes(store, run_id, forward=True)


# ── Monitoring for a newly opened name ────────────────────────────────────────

def _install_monitoring(store: Store, approved, held: set[str]) -> list[str]:
    """Write a new position's kill/add criteria, target price and sizing plan.

    A name the review opens has no monitoring behind it — no kill line for the
    daily watches, no add criterion or target price for the gates, no target
    weight for the drift check. It would sit in the book unwatched until someone
    wrote all of that by hand, which is exactly the step that does not get done.

    Only for approved buys, and only where the model actually supplied a value:
    a field left blank is left alone rather than cleared, so re-running a review
    never quietly erases a criterion you have since refined yourself. Target
    weights follow the decision both ways — a name opened joins the plan, a name
    fully exited leaves it, so the plan keeps describing the book that is meant
    to exist.
    """
    from fundmgr import addsignal, watchplan

    installed: list[str] = []
    targets = json.loads(store.get_meta("paper_target_weights") or "{}")
    notes = json.loads(store.get_meta("paper_position_notes") or "{}")
    changed = False

    for action in approved:
        if action.side == "sell" and action.target_weight_pct <= 0:
            if targets.pop(action.ticker, None) is not None:
                changed = True
            continue
        if action.side != "buy":
            continue

        targets[action.ticker] = round(float(action.target_weight_pct), 4)
        changed = True

        if action.ticker in held:
            continue        # already has whatever plan it was opened with

        try:
            if action.kill_criterion:
                watchplan.set_position_plan(
                    store, action.ticker, kill_criterion=action.kill_criterion)
            if action.add_criterion:
                watchplan.set_add_text(store, action.ticker, action.add_criterion)
            if action.target_price or action.max_weight_pct or action.tranche_pct:
                addsignal.set_plan(
                    store, action.ticker,
                    target_price=action.target_price,
                    max_weight_pct=action.max_weight_pct,
                    tranche_pct=action.tranche_pct,
                )
            if action.next_earnings:
                notes.setdefault(action.ticker, {})["next_earnings"] = action.next_earnings
                changed = True
        except Exception:
            # A malformed criterion must not cost the decision that carried it.
            continue
        installed.append(action.ticker)

    if changed:
        store.set_meta("paper_target_weights", json.dumps(targets))
        store.set_meta("paper_position_notes", json.dumps(notes))
    return installed


# ── Sizing ────────────────────────────────────────────────────────────────────

def _price_sek(ticker: str, snap: PortfolioSnapshot, features: dict,
               meta: dict, store: Store) -> float | None:
    """One name's price in SEK — the unit NAV, weights and cost basis are in.

    A held position already carries it, valued the same way the dashboard values
    the book. Anything else is converted from the feature's native quote, whose
    currency comes from the sleeve's own map when it knows the name and from the
    Yahoo suffix when it doesn't — never a bare default, which would silently
    price a Stockholm name in dollars.
    """
    from fundmgr import paper

    for position in snap.positions:
        if position.ticker == ticker and position.current_price_sek > 0:
            return position.current_price_sek

    feature = features.get(ticker)
    if feature is None or not feature.last_price:
        return None
    currency = (meta.get("currency_map") or {}).get(ticker) or paper.detect_currency(ticker)
    try:
        return paper.to_sek_price(feature.last_price, currency, store)
    except Exception:
        return None


def _share_counts(actions, snap: PortfolioSnapshot, features: dict,
                  meta: dict, store: Store) -> dict[str, dict]:
    """Whole shares to trade per approved action, keyed by ticker.

    A target weight is the right way to *decide* and a poor way to *execute*:
    the broker wants a number of shares. Computed here, at decision time,
    because it depends on the book as it stood when the call was made — deriving
    it later from a book that has since moved would quietly answer a different
    question.

    A full exit reports the whole holding, fractions included, since that is
    what closing it actually takes. A trim floors to whole shares, and can never
    exceed what is held.
    """
    import math

    held = {p.ticker: p.shares for p in snap.positions if p.shares > 0}
    out: dict[str, dict] = {}
    for action in actions:
        price = _price_sek(action.ticker, snap, features, meta, store)
        if not price or price <= 0:
            continue
        if action.side == "sell":
            have = held.get(action.ticker, 0.0)
            if have <= 0:
                continue
            if action.target_weight_pct <= 0:
                shares = have                      # close it out, fractions and all
            else:
                keep = (snap.nav_sek * action.target_weight_pct / 100) / price
                shares = min(have, math.floor(max(0.0, have - keep)))
            if shares <= 0:
                continue
        elif action.side == "buy":
            shares = math.floor(action.sek_estimate / price)
            if shares <= 0:
                continue
        else:
            continue
        out[action.ticker] = {"shares": round(shares, 4), "price_sek": round(price, 2)}
    return out


# ── Funding ───────────────────────────────────────────────────────────────────

def _fund_from_sells(snap: PortfolioSnapshot, decision, cfg: AppConfig) -> PortfolioSnapshot:
    """Settle the run's own sells before the guardrails price its buys.

    Guardrails check a buy against cash on hand and reject anything that would
    breach the cash floor. A fully-deployed sleeve holds almost no cash, so
    without this every add-on is rejected no matter how well funded the paired
    sell leaves it. Applying the sells first — shares down, proceeds to cash,
    NAV unchanged — models settlement and lets "sell A, add B" through, while
    an unfunded buy still fails the same floor it always did.

    Sells themselves are unaffected: guardrails check them for universe
    membership and minimum trade size only, neither of which reads the
    snapshot.
    """
    by_ticker = {p.ticker: p for p in snap.positions}
    positions = [copy.copy(p) for p in snap.positions]
    cash = snap.cash_sek

    for action in decision.actions:
        if action.side != "sell":
            continue
        pos = by_ticker.get(action.ticker)
        if pos is None or pos.shares <= 0 or pos.current_price_sek <= 0:
            continue
        value = pos.shares * pos.current_price_sek
        # Trust the smaller of the model's own estimate and a full exit: a
        # target weight of 0 means everything, and an oversized sek_estimate
        # must not conjure proceeds the position cannot produce.
        proceeds = value if action.target_weight_pct <= 0 else min(action.sek_estimate, value)
        if proceeds <= 0:
            continue
        fee = cfg.fees.calc(proceeds)
        cash += proceeds - fee
        remaining = max(0.0, value - proceeds)
        for p in positions:
            if p.ticker == action.ticker:
                p.shares = remaining / p.current_price_sek if p.current_price_sek else 0.0

    return PortfolioSnapshot(
        positions=[p for p in positions if p.shares > 0],
        cash_sek=cash,
        timestamp=snap.timestamp,
    )


def _drop_unfunded_buys(guardrails, snap: PortfolioSnapshot, cfg: AppConfig) -> set[str]:
    """Re-check funding once the turnover cap has had its say.

    Guardrails price each buy against the sells the model *proposed*, then the
    turnover cap drops trades lowest-confidence-first — and it can drop the very
    sell that was paying for a buy it keeps, leaving an approved buy with no
    money behind it. It also checks each buy independently, so several
    individually affordable buys can overdraw the book together.

    So walk the surviving buys best-first against a running balance and drop the
    ones the book can no longer fund. Returns the tickers dropped, for the
    caller to annotate. The floor is the guardrails' own formula, deliberately —
    a buy should never fail here over a rounding difference, only over funding
    that genuinely went away.
    """
    approved = guardrails.approved_actions
    sells = [a for a in approved if a.side == "sell"]
    cash = _fund_from_sells(snap, _SellsOnly(sells), cfg).cash_sek
    nav = snap.nav_sek

    kept: list = [a for a in approved if a.side != "buy"]
    dropped: set[str] = set()
    for action in sorted((a for a in approved if a.side == "buy"),
                         key=lambda a: a.confidence, reverse=True):
        projected = cash - action.sek_estimate
        if nav > 0 and projected / nav * 100 < cfg.risk.min_cash_pct:
            dropped.add(action.ticker)
            continue
        cash = projected
        kept.append(action)

    if dropped:
        guardrails.approved_actions = kept
    return dropped


class _SellsOnly:
    """Minimal stand-in so _fund_from_sells can price a bare list of sells."""

    def __init__(self, actions: list):
        self.actions = actions


def _cache_news_for_held(cfg: AppConfig, store: Store, working: list[UniverseTicker],
                         must_have: set[str], enabled: bool) -> None:
    """Cache news for the sleeve's own names, held and planned.

    Two things need it and neither works without it. Sentiment on a held name is
    part of deciding whether to keep it — and a sleeve store, unlike a fund's,
    is never filled by a weekly run. More importantly the thesis audit
    (`thesis_check`, run later by `fund paper-track`) judges a decision's claim
    against the news published in its holding window, read from this same cache:
    with nothing cached it reports "no evidence" forever and no verdict is ever
    reached, so the review would show a thesis record that could never fill in.

    Held names only — a few dozen at most, unlike the candidate pool — and never
    fatal: a review without sentiment is still a review.
    """
    if not enabled or not cfg.data.news_feeds:
        return
    ours = [t for t in working if t.yahoo_ticker in must_have]
    if not ours:
        return
    try:
        ticker_news = fetch_news(cfg.data.news_feeds, ours, max_age_hours=72)
        if ticker_news and cfg.data.sentiment.enabled:
            score_and_cache_sentiment(
                ticker_news, store, cfg.data.sentiment.model, cfg.data.sentiment.device,
            )
    except Exception:
        pass


# ── Review ────────────────────────────────────────────────────────────────────

def _sek_snapshot(meta: dict, store: Store) -> PortfolioSnapshot:
    """The sleeve's live book in SEK.

    Position cost basis is already SEK; live prices come through the sleeve's
    own FX path (the same one the dashboard values the book with), so NAV here
    and NAV on the page agree. Falls back to cost for anything unpriceable
    rather than dropping the position and hiding it from the review.
    """
    from fundmgr import paper

    positions = store.get_positions()
    prices: dict[str, float] = {}
    if positions:
        tickers = [p.ticker for p in positions]
        try:
            from fundmgr.data.quotes import live_prices
            native = {t: px for t, px in live_prices(tickers).items() if px}
            prices = paper.sek_prices_for(store, tickers, meta["currency_map"], native)
        except Exception:
            prices = {}
    for p in positions:
        p.current_price_sek = prices.get(p.ticker) or p.avg_cost_sek
    return PortfolioSnapshot(positions=positions, cash_sek=store.get_cash())


def review_sleeve(
    slug: str,
    config_name: str | None = None,
    country: str | None = None,
    provider: str | None = None,
    model_id: str | None = None,
    n_runs: int = 1,
    include_macro: bool = True,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    refresh_prices: bool = True,
    risk: dict | None = None,
    dry_run: bool = False,
) -> dict:
    """Re-decide one sleeve against its current book and a scoped universe.

    Writes the result into the sleeve's own store as a recommendation (so the
    dashboard's decision panel renders it) and seeds outcomes so the sleeve's
    learnings loop starts scoring these calls. Blocking — the web layer runs it
    in a background thread. `dry_run` returns the same result without writing.

    `risk` overrides the profile's caps for this run (see OVERRIDABLE_RISK) and,
    like the scope, is remembered on the sleeve for the next one.
    """
    from fundmgr import paper

    started = time.time()
    n_runs = max(1, min(MAX_RUNS, int(n_runs)))

    meta, store = paper.open_portfolio(slug)
    defaults = review_defaults(store)
    config_name = config_name or defaults["config"]
    country = (country if country is not None else defaults["country"]) or None
    risk = defaults["risk"] if risk is None else clean_risk(risk)

    cfg = load_profile_config(config_name)
    cfg, risk_applied = apply_risk_overrides(cfg, risk)
    cfg = copy.copy(cfg)
    cfg.llm = copy.copy(cfg.llm)
    if provider and model_id:
        cfg.llm.provider = provider
        cfg.llm.model_id = model_id
    cfg.llm.n_samples = n_runs

    universe = _scoped_universe(cfg, country)
    scope_label = (
        f"{_country_label(country.upper())} only — {len(universe)} names"
        if country else f"global — {len(universe)} names"
    )

    snap = _sek_snapshot(meta, store)
    if not snap.positions and not paper.plan_tickers(store):
        raise ValueError(
            f"Sleeve {slug!r} has no positions and no plan to review — "
            "record a fill or import a plan first."
        )

    must_have = {p.ticker for p in snap.positions} | paper.plan_tickers(store)
    source_store = Store(cfg.db_path)
    working, selection_note = _select_candidates(universe, must_have, source_store, max_candidates)

    # Held names may sit outside the scoped universe (a Nordic-scoped review of
    # a sleeve holding ASML.AS). Price them anyway — a position you can't see is
    # a position you can't sell.
    by_yahoo = {t.yahoo_ticker: t for t in get_enabled_tickers(cfg.universe_path)}
    working += [by_yahoo[t] for t in sorted(must_have - {w.yahoo_ticker for w in working})
                if t in by_yahoo]

    fetched: dict[str, bool] = {}
    if refresh_prices:
        fetched = fetch_and_cache_prices(working, store, cfg.data.lookback_days)

    # build_all_features only looks at tickers flagged True, so flag everything
    # that reached the cache — freshly fetched or already stored from an earlier
    # review. compute_features still returns None for anything with no rows, so
    # a failed cold fetch drops out here rather than becoming an empty feature.
    fetched_ok = [t for t in working if fetched.get(t.yahoo_ticker)]
    feature_tickers = tickers_for_feature_build(working, fetched_ok, store)
    features = build_all_features(
        feature_tickers, store, cfg, {t.yahoo_ticker: True for t in feature_tickers},
    )
    apply_to_features(features, store)
    _cache_news_for_held(cfg, store, working, must_have, refresh_prices)
    attach_sentiment_to_features(
        features, store,
        since_date=(datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d"),
    )
    if not features:
        raise RuntimeError(
            f"No price history could be built for {slug!r} under this scope — "
            "try a wider scope or leave the price refresh on."
        )

    held_tickers = {p.ticker for p in snap.positions}
    screened, _ = screen(
        features, held_tickers,
        top_n=cfg.screener.top_n,
        pinned_tickers=set(cfg.screener.pinned_tickers) & set(features),
    )

    macro_block = ""
    if include_macro:
        try:
            from fundmgr.data.macro_context import (
                build_macro_block, fetch_macro_headlines, fetch_macro_indicators,
            )
            macro_block = build_macro_block(
                fetch_macro_indicators(),
                fetch_macro_headlines(cfg.data.macro_feeds) if cfg.data.macro_feeds else [],
            )
        except Exception:
            macro_block = ""  # macro is context, not a hard dependency

    run_id = f"review-{slug}-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
    system_msg, user_msg, fields = build_prompt(
        cfg, snap, screened, store, run_id,
        macro_block=macro_block,
        extra_context=_sleeve_block(meta, store, snap),
        task_override=_task_block(run_id, scope_label, snap, cfg),
        heading=f"Live Sleeve Review — {meta['name']}",
    )

    decision, raw_response, vote_counts, sampling = call_llm_consensus(system_msg, user_msg, cfg)

    # Guardrails price the buys against a book where this run's sells have
    # settled — see _fund_from_sells. Held and planned names join the scoped
    # universe so an existing position is never rejected as "not in universe".
    funded = _fund_from_sells(snap, decision, cfg)
    universe_tickers = {t.yahoo_ticker for t in universe} | must_have
    guardrails = apply_guardrails(decision, funded, features, universe_tickers, cfg)
    unfunded = _drop_unfunded_buys(guardrails, snap, cfg)

    approved = {(a.ticker, a.side) for a in guardrails.approved_actions}
    sizing = _share_counts(guardrails.approved_actions, snap, features, meta, store)
    name_by_ticker = {f.ticker: f.name for f in features.values()}
    targets = json.loads(store.get_meta("paper_target_weights") or "{}")

    actions = []
    for v in guardrails.verdicts:
        a = v.action
        feat = features.get(a.ticker)
        actions.append({
            "ticker": a.ticker,
            "name": name_by_ticker.get(a.ticker, a.ticker),
            "side": a.side,
            "target_weight_pct": a.target_weight_pct,
            "sek_estimate": round(a.sek_estimate),
            "confidence": a.confidence,
            "thesis": a.thesis,
            "stop_loss_pct": a.stop_loss_pct,
            "take_profit_pct": a.take_profit_pct,
            "votes": vote_counts.get(a.ticker) if vote_counts else None,
            "status": v.status,
            "reason": v.rejection_reason or v.clip_note or "",
            "last_price": feat.last_price if feat else None,
            # What to actually type at the broker. A target weight decides;
            # a share count executes.
            "shares": (sizing.get(a.ticker) or {}).get("shares"),
            "price_sek": (sizing.get(a.ticker) or {}).get("price_sek"),
            "held": a.ticker in held_tickers,
            # The monitoring a new name arrives with, so it can be checked
            # before it is acted on rather than discovered on the watch page.
            "plan": {
                "kill_criterion": a.kill_criterion,
                "add_criterion": a.add_criterion,
                "target_price": a.target_price,
                "max_weight_pct": a.max_weight_pct,
                "tranche_pct": a.tranche_pct,
                "next_earnings": a.next_earnings,
            } if (a.side == "buy" and a.ticker not in held_tickers
                  and any((a.kill_criterion, a.add_criterion, a.target_price))) else None,
            # An approved buy in a name the plan never had is the add-on this
            # review exists to surface — flag it so the UI can say so.
            "add_on": a.side == "buy" and a.ticker not in held_tickers and a.ticker not in targets,
            "approved": (a.ticker, a.side) in approved,
        })
    # Both post-verdict passes drop actions the per-action checks approved, so
    # say which one did it rather than leaving a silently un-approved row.
    for row in actions:
        if row["status"] == "APPROVED" and not row["approved"]:
            row["status"] = "DROPPED"
            row["reason"] = (
                "Unfunded once the turnover cap dropped the sells paying for it"
                if row["ticker"] in unfunded and row["side"] == "buy"
                else "Dropped by turnover cap (lower confidence than kept trades)"
            )

    buys = [r for r in actions if r["side"] == "buy" and r["approved"]]
    sells = [r for r in actions if r["side"] == "sell" and r["approved"]]
    # A swap costs turnover twice, so the cap truncates paired trades before it
    # touches anything else. Report what it cost rather than leaving an empty
    # review to be explained by reading per-action rejection strings.
    wanted = sum(a.sek_estimate for a in decision.actions if a.side != "hold")
    kept = sum(r["sek_estimate"] for r in buys + sells)
    turnover = {
        "cap_pct": cfg.risk.max_turnover_pct,
        "cap_sek": round(snap.nav_sek * cfg.risk.max_turnover_pct / 100),
        "proposed_sek": round(wanted),
        "kept_sek": round(kept),
        "dropped": sum(1 for r in actions
                       if r["status"] == "DROPPED" and "turnover cap" in r["reason"]),
    }
    ages = sorted(f.data_age_trading_days for f in screened.values())

    result = {
        "id": run_id,
        "slug": slug,
        "sleeve": meta["name"],
        "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "elapsed_s": None,
        "dry_run": dry_run,
        "scope": {
            "config": config_name,
            "profile": cfg.display_name,
            "country": (country or "").upper(),
            "label": scope_label,
            "universe": cfg.universe_path.name,
            "selection": selection_note,
        },
        "model": {
            "provider": cfg.llm.provider,
            "model_id": cfg.llm.model_id,
            "n_runs": n_runs,
            "reasoning_effort": cfg.llm.reasoning_effort,
        },
        "sampling": sampling,
        "consensus": vote_counts is not None,
        "market_summary": decision.market_summary,
        "notes": decision.notes,
        "cash_target_pct": guardrails.cash_target_pct,
        "nav_sek": round(snap.nav_sek),
        "cash_sek": round(snap.cash_sek),
        "funded_cash_sek": round(funded.cash_sek),
        "risk": {
            "profile": config_name,
            "overrides": risk,
            "applied": risk_applied,
            "max_turnover_pct": cfg.risk.max_turnover_pct,
            "max_position_pct": cfg.risk.max_position_pct,
            "max_positions": cfg.risk.max_positions,
            "min_cash_pct": cfg.risk.min_cash_pct,
        },
        "turnover": turnover,
        "buy_count": len(buys),
        "sell_count": len(sells),
        "add_on_count": sum(1 for r in buys if r["add_on"]),
        "hold_count": sum(1 for r in actions if r["side"] == "hold" and r["approved"]),
        "actions": sorted(actions, key=lambda r: (not r["approved"], -r["target_weight_pct"])),
        "data": {
            "universe_size": len(universe),
            "candidates_worked": len(working),
            "features_built": len(features),
            "candidates_to_llm": len(screened),
            "prices_refreshed": sum(1 for ok in fetched.values() if ok),
            "macro_included": bool(macro_block),
            "median_age_days": ages[len(ages) // 2] if ages else None,
            "stale_count": sum(1 for f in screened.values() if f.is_stale),
        },
    }
    result["elapsed_s"] = round(time.time() - started, 1)

    if not dry_run:
        # Share counts ride along in the stored decision rather than being
        # re-derived when it is read back: they depend on the book as it stood
        # at decision time, and a later reader would compute them against a
        # book that has moved on.
        stored_actions = []
        for a in guardrails.approved_actions:
            row = a.model_dump()
            row.update(sizing.get(a.ticker) or {})
            stored_actions.append(row)
        actions_json = json.dumps(stored_actions)
        store.save_recommendation(RecommendationLog(
            run_id=run_id,
            timestamp=datetime.utcnow(),
            prompt_snapshot=snapshot_to_dict(snap, system_msg, user_msg, fields, cfg),
            llm_response=raw_response,
            guardrail_log=json.dumps(guardrails.to_log()),
            actions_json=actions_json,
            sampling_log=json.dumps(sampling),
        ))
        store.seed_outcomes_for_run(
            run_id, actions_json,
            prices={a.ticker: features[a.ticker].last_price
                    for a in guardrails.approved_actions
                    if a.ticker in features and features[a.ticker].last_price},
        )
        # Everything below writes to the book's plan, not just its decision log.
        # Snapshot the affected names first so dismissing this decision can undo
        # exactly what it did — a criterion or target price installed by a call
        # you declined would otherwise keep steering the watches.
        touched = {a.ticker for a in guardrails.approved_actions}
        plan_before = _plan_state(store, touched)

        # Persist the stop and take-profit the model just argued for. Without
        # this the levels are discarded at the end of the run and the *next*
        # review sees a book with no levels on it — build_prompt renders stored
        # stops in the portfolio block, so a review that drops them starves its
        # own successor. merged_levels also re-targets on holds and keeps a
        # level the model simply didn't restate.
        existing_levels = store.get_effective_stops()
        for action in guardrails.approved_actions:
            merged = merged_levels(action, existing_levels.get(action.ticker))
            if merged:
                store.set_position_stop(
                    action.ticker, stop_pct=merged[0], take_profit_pct=merged[1])
            elif action.side == "sell" and action.target_weight_pct == 0:
                store.clear_position_stop(action.ticker)
        _install_monitoring(store, guardrails.approved_actions, held_tickers)
        _record_revert(store, run_id, plan_before, _plan_state(store, touched))

        # Remember the scope and caps so the next review — and the form —
        # default to them.
        store.set_meta(META_CONFIG, config_name)
        store.set_meta(META_COUNTRY, (country or "").upper())
        store.set_meta(META_RISK, json.dumps(risk))

    return result


__all__ = [
    "DEFAULT_MAX_CANDIDATES", "DEFAULT_REVIEW_CONFIG", "MAX_RUNS", "MODEL_OPTIONS",
    "list_profiles", "list_scopes", "review_defaults", "review_sleeve",
]
