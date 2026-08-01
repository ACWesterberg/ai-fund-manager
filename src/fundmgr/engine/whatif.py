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
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from fundmgr.config import CONFIG_DIR, DATA_DIR, AppConfig, get_enabled_tickers, load_config
from fundmgr.data.fundamentals import apply_to_features
from fundmgr.data.news import attach_sentiment_to_features
from fundmgr.data.prices import build_all_features
from fundmgr.data.screener import screen
from fundmgr.data.universe_selection import tickers_for_feature_build
from fundmgr.engine.client import call_llm_consensus
from fundmgr.engine.prompt import build_prompt
from fundmgr.guardrails.rules import apply_guardrails
from fundmgr.state.models import PortfolioSnapshot
from fundmgr.state.store import Store

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
        })
    return profiles


def _load_profile_config(config_name: str) -> AppConfig:
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


def _freshness(features: dict) -> dict:
    ages = sorted(f.data_age_trading_days for f in features.values())
    if not ages:
        return {"median_age_days": None, "max_age_days": None, "stale_count": 0}
    return {
        "median_age_days": ages[len(ages) // 2],
        "max_age_days": ages[-1],
        "stale_count": sum(1 for f in features.values() if f.is_stale),
    }


def generate_whatif(
    config_name: str,
    provider: str | None = None,
    model_id: str | None = None,
    n_runs: int = 1,
    include_macro: bool = True,
) -> dict:
    """
    Generate a hypothetical from-scratch portfolio for one fund profile.

    provider/model_id override the profile's configured LLM (both or neither);
    n_runs sets consensus sampling (1 = single shot). Blocking — call from a
    background thread in the web layer.
    """
    started = time.time()
    n_runs = max(1, min(MAX_RUNS, int(n_runs)))

    cfg = _load_profile_config(config_name)
    profile_name = cfg.display_name

    # Model override + sample count. copy so the module-level config cache in
    # load_config callers is never mutated.
    cfg = copy.copy(cfg)
    cfg.llm = copy.copy(cfg.llm)
    if provider and model_id:
        cfg.llm.provider = provider
        cfg.llm.model_id = model_id
    cfg.llm.n_samples = n_runs

    store = Store(cfg.db_path)

    features, universe_size = _build_features_from_cache(cfg, store)
    if not features:
        raise RuntimeError(
            f"No cached price data for profile {config_name!r} — "
            "run that fund at least once (fund run) so its caches are warm."
        )

    pinned = set(cfg.screener.pinned_tickers)
    screened_features, _ = screen(features, set(), top_n=cfg.screener.top_n, pinned_tickers=pinned)

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

    # Synthetic clean-slate book: full capital, zero positions. cash_pct = 100
    # trips the same cold-start turnover lift the real run would get.
    snap = PortfolioSnapshot(positions=[], cash_sek=cfg.capital_sek)
    effective_cfg = cfg
    if snap.cash_pct >= cfg.risk.cold_start_cash_threshold:
        effective_cfg = copy.copy(cfg)
        effective_cfg.risk = copy.copy(cfg.risk)
        effective_cfg.risk.max_turnover_pct = cfg.risk.cold_start_turnover_pct

    run_id = f"whatif-{datetime.utcnow().strftime('%Y-%m-%d')}-{uuid.uuid4().hex[:6]}"
    system_msg, user_msg, _fields = build_prompt(
        effective_cfg, snap, screened_features, store, run_id, macro_block=macro_block
    )

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
