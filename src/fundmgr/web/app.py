from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import yfinance as yf

_price_cache: dict[frozenset, tuple[float, dict[str, float]]] = {}
_PRICE_TTL = 300  # seconds


def _logo_domain(website: str | None) -> str | None:
    """Extract bare domain from a company website URL, e.g. 'https://www.volvo.com/' → 'volvo.com'."""
    if not website:
        return None
    try:
        netloc = urlparse(website).netloc
        return netloc.removeprefix("www.") or None
    except Exception:
        return None

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Environment, FileSystemLoader

from fundmgr.config import load_config, load_universe
from fundmgr.reporting.dashboard import compute_stats, nav_chart_json
from fundmgr.state.store import Store

TEMPLATES_DIR = Path(__file__).parent / "templates"

app = FastAPI(title="AI Fund Manager", docs_url=None, redoc_url=None)

# Global simulation sub-routes at /sim/*
from fundmgr.web.sim import router as sim_router  # noqa: E402
app.include_router(sim_router)

# Use Jinja2 directly — Starlette's Jinja2Templates has a cache bug on Python 3.14
_jinja_env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True)


def _render(template_name: str, context: dict) -> HTMLResponse:
    tmpl = _jinja_env.get_template(template_name)
    return HTMLResponse(tmpl.render(**context))

# Loaded once at startup; shared across requests
_cfg = None
_store = None


def _get_deps():
    global _cfg, _store
    if _cfg is None:
        _cfg = load_config()
        _store = Store(_cfg.db_path)
    return _cfg, _store


# ── Pages ─────────────────────────────────────────────────────────────────────

def _fetch_live_prices(tickers: list[str]) -> dict[str, float]:
    """Fetch current prices, cached for 5 minutes to avoid blocking on every page load."""
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


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    cfg, store = _get_deps()
    positions = store.get_positions()
    cash = store.get_cash()
    fees_paid = store.total_fees_paid()
    nav_history = store.get_nav_history()
    stats = compute_stats(nav_history, cfg.capital_sek)

    # Fetch live prices for held positions
    live_prices = _fetch_live_prices([p.ticker for p in positions])

    # NAV: live if available, else cost-basis fallback
    live_market_value = sum(live_prices.get(p.ticker, p.avg_cost_sek) * p.shares for p in positions)
    nav = live_market_value + cash

    universe = load_universe(cfg.universe_path)
    name_map = {t.yahoo_ticker: t.name for t in universe}

    # Pull website domains from fundamentals cache for logo resolution
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

    # Last decision summary for right rail
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
                        "shares_est": round(a.get("sek_estimate", 0) / 1) if a.get("sek_estimate") else None,
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

    pnl_sek = round(nav - cfg.capital_sek, 0)
    pnl_pct = round((nav / cfg.capital_sek - 1) * 100, 2) if cfg.capital_sek else 0.0

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
    })


@app.get("/history", response_class=HTMLResponse)
async def history(request: Request):
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
        try:
            llm_data = json.loads(r["guardrail_log"]) if r["guardrail_log"] else {}
        except Exception:
            llm_data = {}
        # Pull market_summary + notes from the recommendations table llm_response column
        market_summary = ""
        notes = ""
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
    })


@app.get("/transactions", response_class=HTMLResponse)
async def transactions(request: Request):
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
    })


# ── JSON API (for Plotly charts) ───────────────────────────────────────────────

@app.get("/api/nav")
async def api_nav():
    cfg, store = _get_deps()
    nav_history = store.get_nav_history()
    return json.loads(nav_chart_json(nav_history))


@app.get("/api/stats")
async def api_stats():
    cfg, store = _get_deps()
    nav_history = store.get_nav_history()
    return compute_stats(nav_history, cfg.capital_sek)


@app.get("/api/positions")
async def api_positions():
    cfg, store = _get_deps()
    positions = store.get_positions()
    cash = store.get_cash()
    total_cost = sum(p.shares * p.avg_cost_sek for p in positions)
    nav = total_cost + cash
    return {
        "positions": [
            {
                "ticker": p.ticker,
                "shares": p.shares,
                "avg_cost_sek": p.avg_cost_sek,
                "cost_value_sek": round(p.shares * p.avg_cost_sek, 2),
                "weight_pct": round(p.shares * p.avg_cost_sek / nav * 100, 1) if nav > 0 else 0,
            }
            for p in positions
        ],
        "cash_sek": cash,
        "nav_sek": nav,
    }


@app.get("/universe", response_class=HTMLResponse)
async def universe(request: Request):
    cfg, store = _get_deps()
    tickers = load_universe(cfg.universe_path)
    by_exchange: dict[str, list] = {}
    for t in tickers:
        label = t.exchange
        by_exchange.setdefault(label, []).append({
            "name": t.name,
            "ticker": t.yahoo_ticker,
            "isin": t.isin,
            "country": t.country,
            "sector": t.sector,
            "enabled": t.enabled,
        })
    # Sort each group: enabled first, then alpha
    for rows in by_exchange.values():
        rows.sort(key=lambda r: (not r["enabled"], r["name"].lower()))
    exchanges = sorted(by_exchange.keys())
    total_enabled = sum(1 for t in tickers if t.enabled)
    return _render("universe.html", {
        "request": request,
        "by_exchange": by_exchange,
        "exchanges": exchanges,
        "total": len(tickers),
        "total_enabled": total_enabled,
        "active_page": "universe",
    })


# ── GitHub webhook deploy endpoint ────────────────────────────────────────────
# Alternative to Tailscale SSH: GitHub sends a POST here on push to `deploy`.
# Requires DEPLOY_WEBHOOK_SECRET in .env and the Pi's port 8000 accessible
# (e.g. via Cloudflare Tunnel or port forwarding).

ROOT_DIR = Path(__file__).resolve().parents[3]


@app.post("/deploy")
async def github_webhook(
    request: Request,
    x_hub_signature_256: str = Header(default=""),
    x_github_event: str = Header(default=""),
):
    webhook_secret = os.getenv("DEPLOY_WEBHOOK_SECRET", "")
    if not webhook_secret:
        raise HTTPException(status_code=503, detail="Webhook not configured (DEPLOY_WEBHOOK_SECRET not set)")

    body = await request.body()

    # Verify GitHub HMAC signature
    expected = "sha256=" + hmac.new(
        webhook_secret.encode(), body, hashlib.sha256  # type: ignore[attr-defined]
    ).hexdigest()
    if not hmac.compare_digest(expected, x_hub_signature_256):
        raise HTTPException(status_code=401, detail="Invalid signature")

    if x_github_event != "push":
        return JSONResponse({"status": "ignored", "event": x_github_event})

    payload = json.loads(body)
    ref = payload.get("ref", "")
    deploy_branch = os.getenv("DEPLOY_BRANCH", "deploy")

    if ref != f"refs/heads/{deploy_branch}":
        return JSONResponse({"status": "ignored", "ref": ref})

    # Fire deploy script in background — don't block the response
    script = ROOT_DIR / "deploy" / "deploy.sh"
    subprocess.Popen(
        ["bash", str(script)],
        env={**os.environ, "DEPLOY_BRANCH": deploy_branch},
        stdout=open(ROOT_DIR / "data" / "logs" / "deploy.log", "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    commit = payload.get("after", "")[:8]
    pusher = payload.get("pusher", {}).get("name", "unknown")
    return JSONResponse({"status": "deploying", "commit": commit, "pusher": pusher})
