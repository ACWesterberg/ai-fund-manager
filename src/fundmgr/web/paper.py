"""
Paper portfolio web section (/paper) — create a portfolio by setting a value
and pasting the picks an LLM produced; each portfolio is then rendered with the
same dashboard templates as the funds, backed by its own store.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from jinja2 import Environment, FileSystemLoader

from fundmgr import paper
from fundmgr.reporting.dashboard import compute_stats, nav_chart_json


def _logo_domain(website: str | None) -> str | None:
    if not website:
        return None
    try:
        from urllib.parse import urlparse
        netloc = urlparse(website).netloc
        return netloc.removeprefix("www.") or None
    except Exception:
        return None

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_PRICE_TTL = 300  # seconds
_price_cache: dict[str, tuple[float, dict[str, float]]] = {}

jinja_env = Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)), autoescape=True)

router = APIRouter(prefix="/paper")


def _render(template_name: str, context: dict) -> HTMLResponse:
    tmpl = jinja_env.get_template(template_name)
    return HTMLResponse(tmpl.render(**context))


def _live_prices_sek(slug: str, store, meta: dict, tickers: list[str]) -> dict[str, float]:
    """Live prices in SEK for a portfolio's holdings, cached for 5 minutes."""
    if not tickers:
        return {}
    now = time.time()
    if slug in _price_cache:
        ts, data = _price_cache[slug]
        if now - ts < _PRICE_TTL and set(tickers) <= set(data):
            return data
    try:
        from fundmgr.data.quotes import live_prices
        native = {t: p for t, p in live_prices(tickers).items() if p}
        result = paper.sek_prices_for(store, tickers, meta["currency_map"], native)
        if result:
            _price_cache[slug] = (now, result)
        return result
    except Exception:
        return {}


def _base_ctx(meta: dict) -> dict:
    prefix = f"/paper/{meta['slug']}"
    return {
        "is_simulation": True,
        "sim_prefix": prefix,
        "sim_label": meta["name"],
        "sim_accent": "emerald",
        "sim_banner": f"PAPER PORTFOLIO — {meta['name']}"
                      + (f" · picks by {meta['model_label']}" if meta["model_label"] else "")
                      + " · Real market prices · Not real money",
        "api_base": prefix,
        "paper_prefix": prefix,
        "paper_name": meta["name"],
        "fund_label": meta["name"],
    }


def _portfolio_summaries() -> list[dict]:
    out = []
    for meta in paper.list_portfolios():
        _, store = paper.open_portfolio(meta["slug"])
        navs = store.get_nav_history()
        nav = navs[-1].portfolio_nav_sek if navs else meta["capital_sek"]
        pnl_pct = (nav / meta["capital_sek"] - 1) * 100 if meta["capital_sek"] else 0.0
        out.append({
            **meta,
            "nav": round(nav),
            "pnl_pct": round(pnl_pct, 2),
            "n_positions": len(store.get_positions()),
            "n_learnings": len(store.get_active_learnings()),
        })
    return out


# ── List + create ─────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
def paper_home(request: Request):
    return _render("paper.html", {
        "request": request,
        "portfolios": _portfolio_summaries(),
        "active_page": "paper",
        "error": None,
        "form": {},
    })


@router.post("/create")
async def paper_create(
    request: Request,
    name: str = Form(...),
    capital_sek: float = Form(...),
    holdings_text: str = Form(...),
    base_prompt: str = Form(""),
    model_label: str = Form(""),
    benchmark: str = Form(paper.DEFAULT_BENCHMARK),
):
    try:
        slug, _log = paper.create_portfolio(
            name=name,
            capital_sek=capital_sek,
            holdings_text=holdings_text,
            base_prompt=base_prompt,
            model_label=model_label,
            benchmark=benchmark.strip() or paper.DEFAULT_BENCHMARK,
        )
    except ValueError as e:
        return _render("paper.html", {
            "request": request,
            "portfolios": _portfolio_summaries(),
            "active_page": "paper",
            "error": str(e),
            "form": {
                "name": name, "capital_sek": capital_sek,
                "holdings_text": holdings_text, "base_prompt": base_prompt,
                "model_label": model_label, "benchmark": benchmark,
            },
        })
    return RedirectResponse(url=f"/paper/{slug}", status_code=303)


@router.post("/preview")
async def paper_preview(request: Request):
    """Parse the pasted picks (no price fetches) so the form can show what was understood."""
    body = await request.json()
    try:
        holdings = paper.normalise_weights(paper.parse_holdings(body.get("holdings_text", "")))
        return {"ok": True, "holdings": holdings,
                "total_weight": round(sum(h["weight_pct"] for h in holdings), 1)}
    except ValueError as e:
        return {"ok": False, "error": str(e)}


@router.post("/{slug}/delete")
def paper_delete(slug: str):
    try:
        paper.delete_portfolio(slug)
        _price_cache.pop(slug, None)
    except KeyError:
        pass
    return RedirectResponse(url="/paper", status_code=303)


# ── Per-portfolio dashboard (reuses the fund templates) ───────────────────────

def _not_found() -> HTMLResponse:
    return HTMLResponse("<h1>Paper portfolio not found</h1>", status_code=404)


@router.get("/{slug}", response_class=HTMLResponse)
def paper_index(request: Request, slug: str):
    try:
        meta, store = paper.open_portfolio(slug)
    except KeyError:
        return _not_found()

    positions = store.get_positions()
    cash = store.get_cash()
    fees_paid = store.total_fees_paid()
    nav_history = store.get_nav_history()
    stats = compute_stats(nav_history, meta["capital_sek"])

    live_prices = _live_prices_sek(slug, store, meta, [p.ticker for p in positions])
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
            "name": p.ticker,
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
    pnl_sek = round(nav - meta["capital_sek"], 0)
    pnl_pct = round((nav / meta["capital_sek"] - 1) * 100, 2) if meta["capital_sek"] else 0.0

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
                        "sek_estimate": round(a.get("sek_estimate") or 0),
                        "target_weight_pct": a.get("target_weight_pct", 0),
                        "confidence": a.get("confidence") or 0,
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
        **_base_ctx(meta),
    })


@router.get("/{slug}/transactions", response_class=HTMLResponse)
def paper_transactions(request: Request, slug: str):
    try:
        meta, store = paper.open_portfolio(slug)
    except KeyError:
        return _not_found()
    txns = store.get_transactions(limit=100)
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
        **_base_ctx(meta),
    })


@router.get("/{slug}/learnings", response_class=HTMLResponse)
def paper_learnings(request: Request, slug: str):
    try:
        meta, store = paper.open_portfolio(slug)
    except KeyError:
        return _not_found()
    learnings = store.get_active_learnings()
    by_category: dict[str, list[dict]] = {}
    for lrn in learnings:
        by_category.setdefault(lrn.category, []).append({
            "body": lrn.body,
            "created": lrn.created_at.strftime("%Y-%m-%d"),
            "run_count": len(lrn.run_ids),
        })
    return _render("learnings.html", {
        "request": request,
        "total": len(learnings),
        "by_category": by_category,
        "categories": sorted(by_category.keys()),
        "active_page": "learnings",
        **_base_ctx(meta),
    })


@router.get("/{slug}/prompt", response_class=HTMLResponse)
def paper_prompt(request: Request, slug: str):
    try:
        meta, store = paper.open_portfolio(slug)
    except KeyError:
        return _not_found()
    pasted = store.get_meta("paper_pasted_text") or ""
    mandate = meta["base_prompt"] or "(no base prompt was saved for this portfolio)"
    if pasted:
        mandate += "\n\n── Pasted picks ──\n\n" + pasted
    return _render("prompt.html", {
        "request": request,
        "mandate": mandate,
        "mandate_filename": f"asked of {meta['model_label']}" if meta["model_label"] else "base prompt",
        "guidance": {"current": None, "history": []},
        "active_page": "prompt",
        **_base_ctx(meta),
    })


@router.get("/{slug}/api/nav")
def paper_api_nav(slug: str):
    try:
        _meta, store = paper.open_portfolio(slug)
    except KeyError:
        return {"data": [], "layout": {}}
    return json.loads(nav_chart_json(store.get_nav_history()))


@router.get("/{slug}/api/stats")
def paper_api_stats(slug: str):
    try:
        meta, store = paper.open_portfolio(slug)
    except KeyError:
        return {}
    return compute_stats(store.get_nav_history(), meta["capital_sek"])
