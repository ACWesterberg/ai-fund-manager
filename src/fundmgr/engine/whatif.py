"""
What-if portfolio generation — "what would a fresh portfolio look like today?"

Runs a fund profile's full decision pipeline against a synthetic all-cash
portfolio so the model builds a portfolio from scratch under the same mandate,
universe, risk limits and learnings as the real weekly run. Each profile is one
of the existing config/*.yaml files, so the mandate and universe always travel
together (the Buffett screen brings its curated universe, the Nordic fund its
own, etc.).

Market data comes from the profile's own price/fundamentals/sentiment caches —
the same data the fund's last weekly run saw — so no bulk refetch is needed and
a what-if run costs only the LLM calls. The model can be sampled N times with
majority-vote consensus (call_llm_consensus), letting the model effectively
argue with itself before settling on a portfolio.

Results are written as JSON under data/whatif/ and NEVER into any fund
database: no recommendation rows, no outcome seeds, no NAV points, so the
learning/calibration loop never sees hypothetical trades.
"""
from __future__ import annotations

import copy
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from fundmgr.config import CONFIG_DIR, DATA_DIR, AppConfig, get_enabled_tickers, load_config
from fundmgr.data.benchmark import fetch_and_cache_benchmark
from fundmgr.data.fundamentals import fetch_and_cache_fundamentals
from fundmgr.data.fundamentals import apply_to_features
from fundmgr.data.news import attach_sentiment_to_features
from fundmgr.data.prices import build_all_features, fetch_and_cache_prices
from fundmgr.data.screener import screen
from fundmgr.data.universe_selection import tickers_for_feature_build
from fundmgr.engine.client import call_llm_consensus
from fundmgr.engine.prompt import build_prompt
from fundmgr.guardrails.rules import apply_guardrails
from fundmgr.state.models import PortfolioSnapshot
from fundmgr.state.store import Store

logger = logging.getLogger(__name__)

WHATIF_DIR = DATA_DIR / "whatif"

# Models offered in the dashboard dropdown. First element of each tuple is the
# provider expected by engine.client; the label is what the UI shows.
MODEL_OPTIONS: list[dict] = [
    {"provider": "openai",    "model_id": "gpt-5.6-sol",     "label": "GPT-5.6-sol"},
    {"provider": "openai",    "model_id": "gpt-5.5",         "label": "GPT-5.5"},
    {"provider": "anthropic", "model_id": "claude-opus-4-8", "label": "Claude Opus 4.8"},
    {"provider": "anthropic", "model_id": "claude-sonnet-5", "label": "Claude Sonnet 5"},
]

MAX_RUNS = 5  # cap on consensus samples per what-if generation

# Deployment directive appended to the user message when the full-deploy toggle
# is on. The rendered risk-limits block already carries the lifted caps; this
# states the intent in words so the model doesn't hedge into cash anyway.
_FULL_DEPLOY_DIRECTIVE = (
    "\n\n## Deployment Instruction\n"
    "Deploy the ENTIRE amount in this single run. Hold no cash back: the cash "
    "target is 0% and there is no staged-entry turnover cap. Size positions so "
    "the approved buys account for essentially all of NAV. Do not phase in over "
    "future runs — this is a one-shot allocation of the full amount."
)


def list_profiles() -> list[dict]:
    """Fund profiles = the config/*.yaml files. Each bundles mandate + universe
    + risk limits, so selecting a profile can never mix e.g. the Buffett mandate
    with the Nordic universe."""
    profiles = []
    for path in sorted(CONFIG_DIR.glob("config*.yaml")):
        try:
            raw = yaml.safe_load(path.read_text()) or {}
        except Exception:
            continue
        llm = raw.get("llm", {})
        risk = raw.get("risk", {})
        min_trade = float(risk.get("min_trade_sek", 2500))
        max_pos_pct = float(risk.get("max_position_pct", 18))
        profiles.append({
            "config": path.name,
            "name": raw.get("name") or path.stem.replace("config_", "").replace("config", "main"),
            "mandate": Path(raw.get("mandate_path", "config/mandate.md")).name,
            "universe": Path(raw.get("universe_path", "config/universe.csv")).name,
            "benchmark": raw.get("benchmark", ""),
            "capital_sek": raw.get("capital_sek", 0),
            "default_provider": llm.get("provider", "openai"),
            "default_model": llm.get("model_id", ""),
            "default_n_samples": llm.get("n_samples", 1),
            # Smallest amount that still supports a max-weight position clearing
            # the min trade size — the form warns below this.
            "min_amount_sek": round(min_trade / (max_pos_pct / 100)) if max_pos_pct > 0 else round(min_trade),
            "staged_turnover_pct": float(risk.get("cold_start_turnover_pct", 50)),
        })
    return profiles


def load_profile_config(config_name: str) -> AppConfig:
    """Resolve a profile config by bare filename only — no path components —
    so the web layer can pass user input through safely."""
    if Path(config_name).name != config_name or not config_name.endswith(".yaml"):
        raise ValueError(f"Invalid profile config name: {config_name!r}")
    path = CONFIG_DIR / config_name
    if not path.exists():
        raise ValueError(f"Unknown profile config: {config_name!r}")
    return load_config(path)


def _build_features_from_cache(cfg: AppConfig, store: Store) -> tuple[dict, int]:
    """Compute TickerFeatures for every universe ticker that has cached price
    history in this profile's store. Returns (features, universe_size)."""
    tickers = get_enabled_tickers(cfg.universe_path)
    feature_tickers = tickers_for_feature_build(tickers, [], store)
    fetch_result = {t.yahoo_ticker: True for t in feature_tickers}
    features = build_all_features(feature_tickers, store, cfg, fetch_result)
    apply_to_features(features, store)
    since_news = (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d")
    attach_sentiment_to_features(features, store, since_date=since_news)
    return features, len(tickers)


def refresh_candidates(
    cfg: AppConfig, store: Store, candidates: dict
) -> tuple[dict, dict]:
    """Refetch prices, fundamentals and news for the screened candidates only.

    A what-if is triggered on demand and read from the profile's caches, which
    the fund last filled on its weekly run — up to six days stale, and on the
    global profiles far worse, since 17k tickers rotate through a 2.5k weekly
    price budget over eight weeks. A portfolio built for "today" on prices from
    a fortnight ago is not the thing the user asked for.

    Refreshing the whole universe is not the answer either: on the global
    profiles that is 17k fetches for one what-if. The screened candidates are
    the only tickers the model is ever shown, so those are the ones refreshed —
    around a hundred, seconds rather than hours.

    The honest limit: candidates are *selected* on cached data, so a name that
    would have screened in on fresh prices can still be missed. What the model
    sees is current; what it was offered was chosen a few days ago.

    Returns (features, report).
    """
    tickers = _universe_subset(cfg, set(candidates))
    symbols = [t.yahoo_ticker for t in tickers]
    report: dict = {"tickers": len(symbols)}
    if not tickers:
        return candidates, report

    fetch_result = fetch_and_cache_prices(tickers, store, cfg.data.lookback_days, force_refresh=True)
    report["prices_ok"] = sum(1 for v in fetch_result.values() if v)

    try:
        stale = store.get_stale_fundamentals_tickers(symbols, ttl_days=7)
        report["fundamentals_refreshed"] = (
            fetch_and_cache_fundamentals(symbols, store, ttl_days=7, max_workers=12) if stale else 0
        )
    except Exception as exc:
        logger.warning("What-if: fundamentals refresh failed: %s", exc)
        report["fundamentals_refreshed"] = 0

    try:
        report["benchmark_ok"] = bool(
            fetch_and_cache_benchmark(store, cfg.benchmark, cfg.data.lookback_days, True)
        )
    except Exception as exc:
        logger.warning("What-if: benchmark refresh failed: %s", exc)
        report["benchmark_ok"] = False

    # Rebuild features over the refreshed rows.
    features = build_all_features(tickers, store, cfg, fetch_result)
    apply_to_features(features, store)
    since_news = (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d")
    attach_sentiment_to_features(features, store, since_date=since_news)
    return features, report


def _universe_subset(cfg: AppConfig, symbols: set[str]) -> list:
    return [t for t in get_enabled_tickers(cfg.universe_path) if t.yahoo_ticker in symbols]


def _freshness(features: dict) -> dict:
    ages = sorted(f.data_age_trading_days for f in features.values())
    if not ages:
        return {"median_age_days": None, "max_age_days": None, "stale_count": 0}
    return {
        "median_age_days": ages[len(ages) // 2],
        "max_age_days": ages[-1],
        "stale_count": sum(1 for f in features.values() if f.is_stale),
    }


def deployment_floors(cfg: AppConfig) -> tuple[float, float]:
    """(hard_floor, comfortable_floor) in SEK for the amount being placed.

    Below the hard floor no trade can clear risk.min_trade_sek at all. Below the
    comfortable floor a position sized at the max single-name weight still comes
    out under the min trade size, so the book degenerates to a couple of names.
    """
    hard = cfg.risk.min_trade_sek
    comfortable = (
        cfg.risk.min_trade_sek / (cfg.risk.max_position_pct / 100)
        if cfg.risk.max_position_pct > 0 else hard
    )
    return hard, comfortable


def generate_whatif(
    config_name: str,
    provider: str | None = None,
    model_id: str | None = None,
    n_runs: int = 1,
    include_macro: bool = True,
    capital_sek: float | None = None,
    deploy_full: bool = False,
    refresh_prices: bool = True,
) -> dict:
    """
    Generate a hypothetical from-scratch portfolio for one fund profile.

    provider/model_id override the profile's configured LLM (both or neither);
    n_runs sets consensus sampling (1 = single shot).

    capital_sek overrides how much is being placed (defaults to the profile's
    own capital). deploy_full puts the whole amount to work in this one run —
    it lifts the staged-entry turnover cap to 100% and drops the cash floor to
    zero, so the result is a fully-invested book rather than a first tranche.

    refresh_prices refetches prices, fundamentals and news for the screened
    candidates before the model sees them, so an on-demand what-if is built on
    today's market rather than the profile's last weekly run. Turn it off for a
    fast, cache-only run.

    Blocking — call from a background thread in the web layer.
    """
    started = time.time()
    n_runs = max(1, min(MAX_RUNS, int(n_runs)))

    cfg = load_profile_config(config_name)
    profile_name = cfg.display_name
    profile_capital = cfg.capital_sek

    # Model override + sample count. copy so the module-level config cache in
    # load_config callers is never mutated.
    cfg = copy.copy(cfg)
    cfg.llm = copy.copy(cfg.llm)
    if provider and model_id:
        cfg.llm.provider = provider
        cfg.llm.model_id = model_id
    cfg.llm.n_samples = n_runs

    # Amount to place
    if capital_sek is not None:
        capital_sek = float(capital_sek)
        if capital_sek <= 0:
            raise ValueError("Amount to place must be greater than 0.")
        cfg.capital_sek = capital_sek

    hard_floor, comfortable_floor = deployment_floors(cfg)
    if cfg.capital_sek < hard_floor:
        raise ValueError(
            f"{cfg.capital_sek:,.0f} SEK is below this profile's minimum trade size "
            f"({hard_floor:,.0f} SEK) — no trade could clear it. Place at least "
            f"{comfortable_floor:,.0f} SEK for a workable book."
        )
    undersized = cfg.capital_sek < comfortable_floor

    store = Store(cfg.db_path)

    features, universe_size = _build_features_from_cache(cfg, store)
    if not features:
        raise RuntimeError(
            f"No cached price data for profile {config_name!r} — "
            "run that fund at least once (fund run) so its caches are warm."
        )

    pinned = set(cfg.screener.pinned_tickers)
    screened_features, _ = screen(features, set(), top_n=cfg.screener.top_n, pinned_tickers=pinned)

    refresh_report: dict = {"refreshed": False}
    if refresh_prices:
        fresh, refresh_report = refresh_candidates(cfg, store, screened_features)
        refresh_report["refreshed"] = True
        if fresh:
            # Re-screen on the refreshed numbers so ordering and any staleness
            # gate reflect today, not the fund's last weekly run.
            screened_features, _ = screen(
                fresh, set(), top_n=cfg.screener.top_n, pinned_tickers=pinned
            )

    macro_block = ""
    if include_macro:
        try:
            from fundmgr.data.macro_context import (
                build_macro_block, fetch_macro_headlines, fetch_macro_indicators,
            )
            indicators = fetch_macro_indicators()
            headlines = fetch_macro_headlines(cfg.data.macro_feeds) if cfg.data.macro_feeds else []
            macro_block = build_macro_block(indicators, headlines)
        except Exception:
            macro_block = ""  # macro is context, not a hard dependency

    # Synthetic clean-slate book: the full amount in cash, zero positions.
    snap = PortfolioSnapshot(positions=[], cash_sek=cfg.capital_sek)

    effective_cfg = copy.copy(cfg)
    effective_cfg.risk = copy.copy(cfg.risk)
    if deploy_full:
        # Put everything to work now: no staged-entry cap, no cash held back.
        # max_cash_pct goes to 0 too, otherwise the guardrail would happily
        # approve a cash target up to the mandate's ceiling.
        effective_cfg.risk.max_turnover_pct = 100.0
        effective_cfg.risk.min_cash_pct = 0.0
        effective_cfg.risk.max_cash_pct = 0.0
    elif snap.cash_pct >= cfg.risk.cold_start_cash_threshold:
        # Staged: same cold-start turnover lift the real run would get, so this
        # reads as a realistic first tranche rather than the whole portfolio.
        effective_cfg.risk.max_turnover_pct = cfg.risk.cold_start_turnover_pct

    run_id = f"whatif-{datetime.utcnow().strftime('%Y-%m-%d')}-{uuid.uuid4().hex[:6]}"
    system_msg, user_msg, _fields = build_prompt(
        effective_cfg, snap, screened_features, store, run_id, macro_block=macro_block
    )
    if deploy_full:
        user_msg += _FULL_DEPLOY_DIRECTIVE

    decision, _raw, vote_counts, sampling = call_llm_consensus(system_msg, user_msg, effective_cfg)

    universe_tickers = {t.yahoo_ticker for t in get_enabled_tickers(cfg.universe_path)}
    guardrails = apply_guardrails(decision, snap, features, universe_tickers, effective_cfg)

    approved = {(a.ticker, a.side) for a in guardrails.approved_actions}
    name_by_ticker = {f.ticker: f.name for f in features.values()}

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
            "approved": (a.ticker, a.side) in approved,
        })
    # Turnover-cap drops happen after per-action verdicts — reflect them
    for row in actions:
        if row["status"] == "APPROVED" and not row["approved"]:
            row["status"] = "DROPPED"
            row["reason"] = "Dropped by turnover cap (lower confidence than kept trades)"

    buys = [r for r in actions if r["side"] == "buy" and r["approved"]]
    invested_pct = round(sum(r["target_weight_pct"] for r in buys), 1)

    result = {
        "id": run_id,
        "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "elapsed_s": None,  # filled below
        "profile": {
            "config": config_name,
            "name": profile_name,
            "mandate": cfg.mandate_path.name,
            "universe": cfg.universe_path.name,
            "benchmark": cfg.benchmark,
            "capital_sek": cfg.capital_sek,
        },
        "deployment": {
            "placed_sek": cfg.capital_sek,
            "profile_capital_sek": profile_capital,
            "amount_overridden": capital_sek is not None and capital_sek != profile_capital,
            "full_deploy": deploy_full,
            "turnover_cap_pct": effective_cfg.risk.max_turnover_pct,
            "min_cash_pct": effective_cfg.risk.min_cash_pct,
            "undersized": undersized,
            "comfortable_floor_sek": round(comfortable_floor),
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
        "invested_pct": invested_pct,
        "buy_count": len(buys),
        "actions": sorted(actions, key=lambda r: (not r["approved"], -r["target_weight_pct"])),
        "data": {
            "universe_size": universe_size,
            "features_from_cache": len(features),
            "candidates_to_llm": len(screened_features),
            "macro_included": bool(macro_block),
            # What was refetched for this run, so a portfolio built on stale
            # caches is never mistaken for one built on today's market.
            "refresh": refresh_report,
            **_freshness(screened_features),
        },
    }
    result["elapsed_s"] = round(time.time() - started, 1)

    WHATIF_DIR.mkdir(parents=True, exist_ok=True)
    (WHATIF_DIR / f"{run_id}.json").write_text(json.dumps(result, indent=2))
    return result


def list_results(limit: int = 20) -> list[dict]:
    """Past what-if results, newest first."""
    if not WHATIF_DIR.exists():
        return []
    paths = sorted(WHATIF_DIR.glob("whatif-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    results = []
    for p in paths[:limit]:
        try:
            results.append(json.loads(p.read_text()))
        except Exception:
            continue
    return results
