from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Environment, FileSystemLoader

from fundmgr.config import load_config
from fundmgr.reporting.dashboard import compute_stats, nav_chart_json
from fundmgr.state.store import Store

TEMPLATES_DIR = Path(__file__).parent / "templates"

app = FastAPI(title="AI Fund Manager", docs_url=None, redoc_url=None)

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

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    cfg, store = _get_deps()
    positions = store.get_positions()
    cash = store.get_cash()
    fees_paid = store.total_fees_paid()
    nav_history = store.get_nav_history()
    stats = compute_stats(nav_history, cfg.capital_sek)

    # Attach live prices where available (best-effort)
    nav = sum(p.shares * p.avg_cost_sek for p in positions) + cash  # cost-basis fallback
    if nav_history:
        nav = nav_history[-1].portfolio_nav_sek

    positions_data = [
        {
            "ticker": p.ticker,
            "shares": p.shares,
            "avg_cost": p.avg_cost_sek,
            "cost_value": round(p.shares * p.avg_cost_sek, 0),
            "weight_pct": round(p.shares * p.avg_cost_sek / nav * 100, 1) if nav > 0 else 0,
        }
        for p in sorted(positions, key=lambda x: x.shares * x.avg_cost_sek, reverse=True)
    ]

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
        recommendations.append({
            "run_id": r["run_id"],
            "timestamp": r["timestamp"][:10],
            "action_count": len(actions),
            "buys": sum(1 for a in actions if a.get("side") == "buy"),
            "sells": sum(1 for a in actions if a.get("side") == "sell"),
            "holds": sum(1 for a in actions if a.get("side") == "hold"),
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
