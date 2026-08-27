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
from fundmgr.data.news import attach_sentiment_to_features
from fundmgr.data.prices import build_all_features, fetch_and_cache_prices
from fundmgr.data.screener import screen
from fundmgr.data.universe_selection import tickers_for_feature_build
from fundmgr.engine.client import call_llm_consensus
from fundmgr.engine.prompt import build_prompt, snapshot_to_dict
from fundmgr.engine.whatif import MAX_RUNS, MODEL_OPTIONS, load_profile_config, list_profiles
from fundmgr.guardrails.rules import apply_guardrails
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
    """This sleeve's stored review scope, falling back to the global profile."""
    config_name = store.get_meta(META_CONFIG) or DEFAULT_REVIEW_CONFIG
    if config_name not in {p["config"] for p in list_profiles()}:
        config_name = DEFAULT_REVIEW_CONFIG
    return {"config": config_name, "country": store.get_meta(META_COUNTRY) or ""}


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


def _sleeve_block(meta: dict, store: Store, snap: PortfolioSnapshot) -> str:
    """The monitoring state the daily watches alert on, as prompt context.

    Same stored metadata the dashboard's Watch panel renders, so what the model
    reasons over and what you see on the page can't drift apart.
    """
    targets = json.loads(store.get_meta("paper_target_weights") or "{}")
    kills = json.loads(store.get_meta("paper_kill_criteria") or "{}")
    notes = json.loads(store.get_meta("paper_position_notes") or "{}")
    capex = json.loads(store.get_meta("paper_capex_kill") or "{}")

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

    if targets or kills:
        lines.append("### Plan, live weight and kill lines")
        held = {p.ticker: p for p in snap.positions if p.shares > 0}
        for ticker in sorted(set(targets) | set(kills) | set(held)):
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
                lines.append(f"      kill: {kills[ticker]}")
            for hit in _kill_hits(store, ticker):
                lines.append(f"      ⚠ KILL SIGNAL LOGGED — {hit}")
            if notes.get(ticker, {}).get("cluster"):
                lines.append(f"      cluster: {notes[ticker]['cluster']}")
            nxt = notes.get(ticker, {}).get("next_earnings")
            if nxt:
                lines.append(f"      next earnings: {nxt}")
        lines.append("")

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
        "Decide what to do with this sleeve *now*. Two jobs, one JSON answer:\n"
        "  1. Every position above: hold, trim (sell to a lower target weight) or "
        "exit (sell to 0). A kill signal logged against a name is a strong prior to "
        "exit, but you own the call — say so in the thesis either way.\n"
        f"  2. Add-ons from the candidate universe below ({scope_label}). A new name "
        "must earn its place against what it displaces.\n\n"
        f"There is NO new capital. NAV is {snap.nav_sek:,.0f} SEK and stays there, so "
        f"every buy is funded from cash on hand ({snap.cash_sek:,.0f} SEK) plus the "
        "proceeds of the sells you recommend in this same run. Recommending a buy "
        "without freeing the money for it is not a plan — pair them.\n"
        f"Total traded value this run is capped at {cfg.risk.max_turnover_pct:.0f}% of NAV "
        f"(≈{turnover:,.0f} SEK); trades beyond the cap are dropped lowest-confidence "
        "first, so lead with your best ideas.\n\n"
        f"Return a DecisionRun JSON. Run ID must be: {run_id}"
    )


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
    dry_run: bool = False,
) -> dict:
    """Re-decide one sleeve against its current book and a scoped universe.

    Writes the result into the sleeve's own store as a recommendation (so the
    dashboard's decision panel renders it) and seeds outcomes so the sleeve's
    learnings loop starts scoring these calls. Blocking — the web layer runs it
    in a background thread. `dry_run` returns the same result without writing.
    """
    from fundmgr import paper

    started = time.time()
    n_runs = max(1, min(MAX_RUNS, int(n_runs)))

    meta, store = paper.open_portfolio(slug)
    defaults = review_defaults(store)
    config_name = config_name or defaults["config"]
    country = (country if country is not None else defaults["country"]) or None

    cfg = load_profile_config(config_name)
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
            "held": a.ticker in held_tickers,
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
        actions_json = json.dumps([a.model_dump() for a in guardrails.approved_actions])
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
        # Remember the scope so the next review — and the form — default to it.
        store.set_meta(META_CONFIG, config_name)
        store.set_meta(META_COUNTRY, (country or "").upper())

    return result


__all__ = [
    "DEFAULT_MAX_CANDIDATES", "DEFAULT_REVIEW_CONFIG", "MAX_RUNS", "MODEL_OPTIONS",
    "list_profiles", "list_scopes", "review_defaults", "review_sleeve",
]
