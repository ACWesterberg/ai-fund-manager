"""
Simulation dashboard route factory — call make_sim_router() for each sim config.
Each router gets isolated state (config, store, price cache) via closures.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import urlparse

import yfinance as yf
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader

from fundmgr.config import load_config, load_universe
from fundmgr.reporting.dashboard import compute_stats, nav_chart_json
from fundmgr.state.store import Store

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_PRICE_TTL = 300  # seconds


def _logo_domain(website: str | None) -> str | None:
    if not website:
        return None
    try:
        netloc = urlparse(website).netloc
        return netloc.removeprefix("www.") or None
    except Exception:
        return None


def make_sim_router(config_filename: str, prefix: str, sim_label: str, sim_accent: str = "amber") -> APIRouter:
    """
    Create a simulation APIRouter with its own isolated config/store/price cache.

    config_filename: relative to repo root, e.g. "config/config_global.yaml"
    prefix:          URL prefix, e.g. "/sim" or "/sim-claude"
    sim_label:       Human-readable label shown in the simulation banner
    sim_accent:      Tailwind colour key ("amber" or "violet") for banner styling
    """
    ROOT = Path(__file__).resolve().parents[3]
    config_path = ROOT / config_filename

    _sim_cfg = None
    _sim_store = None
    _price_cache: dict[frozenset, tuple[float, dict[str, float]]] = {}

    jinja_env = Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)), autoescape=True)

    router = APIRouter(prefix=prefix)

    def _get_deps():
        nonlocal _sim_cfg, _sim_store
        if _sim_cfg is None:
            _sim_cfg = load_config(config_path)
            _sim_store = Store(_sim_cfg.db_path)
        return _sim_cfg, _sim_store

    def _fetch_live_prices(tickers: list[str]) -> dict[str, float]:
        if not tickers:
            return {}
        key = frozenset(tickers)
        now = time.time()
        if key in _price_cache:
            ts, data = _price_cache[key]
            if now - ts < _PRICE_TTL:
                return data
        try:
            if len(tickers) == 1:
                raw = yf.download(tickers, period="2d", auto_adjust=True, progress=False)
                close = raw.get("Close", raw.iloc[:, :1])
                if hasattr(close, "columns"):
                    close = close.iloc[:, 0]
                close = close.dropna()
                result = {tickers[0]: float(close.iloc[-1])} if not close.empty else {}
            else:
                raw = yf.download(tickers, period="2d", auto_adjust=True, progress=False)
                close_df = raw.get("Close")
                result = {}
                if close_df is not None and not close_df.empty:
                    for t in tickers:
                        if t in close_df.columns:
                            series = close_df[t].dropna()
                            if not series.empty:
                                result[t] = float(series.iloc[-1])
            if result:
                _price_cache[key] = (now, result)
            return result
        except Exception:
            return {}

    def _render(template_name: str, context: dict) -> HTMLResponse:
        tmpl = jinja_env.get_template(template_name)
        return HTMLResponse(tmpl.render(**context))

    def _sim_base_ctx() -> dict:
        return {
            "is_simulation": True,
            "sim_prefix": prefix,
            "sim_label": sim_label,
            "sim_accent": sim_accent,
            "api_base": prefix,
        }

    @router.get("/", response_class=HTMLResponse)
    def sim_index(request: Request):
        cfg, store = _get_deps()
        positions = store.get_positions()
        cash = store.get_cash()
        fees_paid = store.total_fees_paid()
        nav_history = store.get_nav_history()
        stats = compute_stats(nav_history, cfg.capital_sek)

        universe = load_universe(cfg.universe_path)
        name_map = {t.yahoo_ticker: t.name for t in universe}
        cur_by_ticker = {t.yahoo_ticker: t.currency for t in universe}

        live_prices = _fetch_live_prices([p.ticker for p in positions])
        if cfg.fx_to_sek:
            from fundmgr.data.fx import convert_prices_to_sek
            live_prices = convert_prices_to_sek(live_prices, cur_by_ticker, store)
        live_market_value = sum(live_prices.get(p.ticker, p.avg_cost_sek) * p.shares for p in positions)
        nav = live_market_value + cash

        fund_domains: dict[str, str | None] = {}
        if positions:
            cached_fund = store.get_all_fundamentals([p.ticker for p in positions])
            for ticker, fdata in cached_fund.items():
                fund_domains[ticker] = _logo_domain(fdata.get("website"))

        positions_data = []
        for p in sorted(positions, key=lambda x: x.shares * x.avg_cost_sek, reverse=True):
            live = live_prices.get(p.ticker)
            cost_value = round(p.shares * p.avg_cost_sek, 0)
            current_value = round(p.shares * live, 0) if live else None
            pnl_sek = round(current_value - cost_value, 0) if current_value is not None else None
            pnl_pct = round((live / p.avg_cost_sek - 1) * 100, 1) if live else None
            weight_val = current_value if current_value is not None else cost_value
            positions_data.append({
                "ticker": p.ticker,
                "name": name_map.get(p.ticker, p.ticker),
                "shares": p.shares,
                "avg_cost": p.avg_cost_sek,
                "cost_value": cost_value,
                "current_price": round(live, 2) if live else None,
                "current_value": current_value,
                "pnl_sek": pnl_sek,
                "pnl_pct": pnl_pct,
                "weight_pct": round(weight_val / nav * 100, 1) if nav > 0 else 0,
                "logo_domain": fund_domains.get(p.ticker),
            })

        cash_pct = round(cash / nav * 100, 1) if nav > 0 else 100.0
        pnl_sek = round(nav - cfg.capital_sek, 0)
        pnl_pct = round((nav / cfg.capital_sek - 1) * 100, 2) if cfg.capital_sek else 0.0

        last_run = None
        last_rec = store.get_last_recommendation()
        if last_rec:
            try:
                llm_data = json.loads(last_rec.llm_response)
                actions = json.loads(last_rec.actions_json)
                last_run = {
                    "run_id": last_rec.run_id,
                    "timestamp": last_rec.timestamp.strftime("%Y-%m-%d %H:%M"),
                    "market_summary": llm_data.get("market_summary", ""),
                    "notes": llm_data.get("notes", ""),
                    "buys":  sum(1 for a in actions if a.get("side") == "buy"),
                    "sells": sum(1 for a in actions if a.get("side") == "sell"),
                    "holds": sum(1 for a in actions if a.get("side") == "hold"),
                    "actions": [
                        {
                            "ticker": a.get("ticker", ""),
                            "side": a.get("side", ""),
                            "sek_estimate": round(a.get("sek_estimate", 0)),
                            "target_weight_pct": a.get("target_weight_pct", 0),
                            "confidence": a.get("confidence", 0),
                            "thesis": a.get("thesis", ""),
                            "stop_loss_pct": a.get("stop_loss_pct"),
                        }
                        for a in actions if a.get("side") in ("buy", "sell")
                    ],
                }
            except Exception:
                pass

        return _render("index.html", {
            "request": request,
            "positions": positions_data,
            "cash": cash,
            "cash_pct": cash_pct,
            "nav": nav,
            "fees_paid": fees_paid,
            "stats": stats,
            "has_history": len(nav_history) >= 2,
            "last_run": last_run,
            "pnl_sek": pnl_sek,
            "pnl_pct": pnl_pct,
            "active_page": "portfolio",
            **_sim_base_ctx(),
        })

    @router.get("/history", response_class=HTMLResponse)
    def sim_history(request: Request):
        cfg, store = _get_deps()
        with store._conn() as conn:
            rows = conn.execute(
                "SELECT run_id, timestamp, actions_json, guardrail_log FROM recommendations ORDER BY timestamp DESC LIMIT 20"
            ).fetchall()

        recommendations = []
        for r in rows:
            try:
                actions = json.loads(r["actions_json"])
            except Exception:
                actions = []
            market_summary, notes = "", ""
            try:
                with store._conn() as conn2:
                    row2 = conn2.execute(
                        "SELECT llm_response FROM recommendations WHERE run_id=?", (r["run_id"],)
                    ).fetchone()
                if row2:
                    lr = json.loads(row2["llm_response"])
                    market_summary = lr.get("market_summary", "")
                    notes = lr.get("notes", "")
            except Exception:
                pass
            trade_actions = [
                {
                    "ticker": a.get("ticker", ""),
                    "side": a.get("side", ""),
                    "sek_estimate": round(a.get("sek_estimate", 0)),
                    "target_weight_pct": a.get("target_weight_pct", 0),
                    "confidence": a.get("confidence", 0),
                    "thesis": a.get("thesis", ""),
                    "stop_loss_pct": a.get("stop_loss_pct"),
                }
                for a in actions if a.get("side") in ("buy", "sell")
            ]
            recommendations.append({
                "run_id": r["run_id"],
                "timestamp": r["timestamp"][:10],
                "action_count": len(actions),
                "buys": sum(1 for a in actions if a.get("side") == "buy"),
                "sells": sum(1 for a in actions if a.get("side") == "sell"),
                "holds": sum(1 for a in actions if a.get("side") == "hold"),
                "market_summary": market_summary,
                "notes": notes,
                "trade_actions": trade_actions,
                "trade_actions_json": json.dumps(trade_actions),
            })

        return _render("history.html", {
            "request": request,
            "recommendations": recommendations,
            "active_page": "history",
            **_sim_base_ctx(),
        })

    @router.get("/transactions", response_class=HTMLResponse)
    def sim_transactions(request: Request):
        cfg, store = _get_deps()
        txns = store.get_transactions(limit=50)
        txn_data = [
            {
                "date": t.timestamp.strftime("%Y-%m-%d %H:%M"),
                "ticker": t.ticker,
                "side": t.side.upper(),
                "shares": t.shares,
                "price": t.price_sek,
                "gross": round(t.gross_sek, 0),
                "fee": t.fee_sek,
                "source": t.source,
            }
            for t in txns
        ]
        return _render("transactions.html", {
            "request": request,
            "transactions": txn_data,
            "total_fees": store.total_fees_paid(),
            "active_page": "transactions",
            **_sim_base_ctx(),
        })

    @router.get("/api/nav")
    def sim_api_nav():
        cfg, store = _get_deps()
        nav_history = store.get_nav_history()
        return json.loads(nav_chart_json(nav_history))

    @router.get("/api/stats")
    def sim_api_stats():
        cfg, store = _get_deps()
        nav_history = store.get_nav_history()
        return compute_stats(nav_history, cfg.capital_sek)

    return router
