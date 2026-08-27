"""
Portfolio dashboard sections backed by the paper-portfolio engine.

Two sections share one engine (fundmgr.paper) and the same per-book dashboards:

  • /paper — pasted LLM picks, tracked at real prices as a simulation
             ("Not real money").
  • /live  — real monitored sleeves imported from a structured answer
             (`fund paper-import`, kind="live"). Real-money framing, plus a
             Watch-status panel surfacing the capex kill criterion, upcoming
             earnings and weight drift the daily watches alert on.

Both are produced by make_portfolio_router(); each closes over its own URL
prefix and book `kind` so the list, framing and create/import path differ while
the per-slug dashboards (index / transactions / learnings / prompt) are shared.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from fundmgr import paper, watchplan
from fundmgr.engine import sleeve_review
from fundmgr.reporting.dashboard import benchmark_label, compute_stats, nav_chart_json

# Largest single upload accepted for photo import. Phone photos of a holdings
# screen land around 2–5 MB; anything past this is a mistake, not a portfolio.
_MAX_UPLOAD_BYTES = 12 * 1024 * 1024


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

# Single-slot review job registry, same shape as the What-If Lab's: a review is
# an LLM-cost-bearing run of several minutes, so it goes to a daemon thread and
# the page polls. One at a time across all sleeves — the Pi doesn't need a queue.
_review_lock = threading.Lock()
_review_job: dict | None = None  # {id, slug, status, params, result, error, started}


class ReviewRequest(BaseModel):
    """Scope for one sleeve review. Empty country = global."""
    config: str | None = None
    country: str = ""
    provider: str | None = None
    model_id: str | None = None
    n_runs: int = Field(default=1, ge=1, le=sleeve_review.MAX_RUNS)
    include_macro: bool = True
    refresh_prices: bool = True
    # Per-sleeve risk caps. Empty = keep whatever the sleeve already stores;
    # the engine cleans and validates before any of it reaches a guardrail.
    risk: dict = Field(default_factory=dict)


def _run_review(job_id: str, slug: str, req: ReviewRequest) -> None:
    global _review_job
    try:
        result = sleeve_review.review_sleeve(
            slug,
            config_name=req.config,
            country=req.country,
            provider=req.provider,
            model_id=req.model_id,
            n_runs=req.n_runs,
            include_macro=req.include_macro,
            refresh_prices=req.refresh_prices,
            risk=req.risk or None,
        )
        with _review_lock:
            if _review_job and _review_job["id"] == job_id:
                _review_job.update(status="done", result=result)
    except Exception as e:
        with _review_lock:
            if _review_job and _review_job["id"] == job_id:
                _review_job.update(status="error", error=str(e))


jinja_env = Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)), autoescape=True)


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


def _portfolio_summaries(kind: str) -> list[dict]:
    out = []
    for meta in paper.list_portfolios(kind=kind):
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


def _watch_status(store, positions_data: list[dict]) -> dict | None:
    """Monitoring state for the dashboard's Watch panel: the portfolio capex
    kill criterion, per-position weight drift, upcoming earnings, kill lines
    (text + numeric) and time horizons.

    Reads only stored metadata + the live weights already computed for the
    positions table (no extra network). Returns None when the book carries no
    monitoring config."""
    capex_cfg = json.loads(store.get_meta("paper_capex_kill") or "{}")
    targets = json.loads(store.get_meta("paper_target_weights") or "{}")
    notes = json.loads(store.get_meta("paper_position_notes") or "{}")
    kills = json.loads(store.get_meta("paper_kill_criteria") or "{}")
    add_texts = watchplan.get_add_text(store)
    rules = watchplan.get_kill_rules(store)
    horizons = watchplan.get_horizons(store)
    verdicts = _kill_verdicts(store, list(kills))
    # Anything held is worth a row even with no plan on it yet — that's how the
    # first kill criterion gets set. Only a book with nothing at all opts out.
    if not (capex_cfg or targets or kills or add_texts or rules or horizons
            or positions_data):
        return None

    capex = None
    if capex_cfg.get("trigger"):
        flagged = paper._recent_capex_signals(store, paper._CAPEX_WINDOW_DAYS)
        hypers = capex_cfg.get("hyperscalers") or paper.DEFAULT_HYPERSCALERS
        capex = {
            "status": store.get_meta("paper_capex_status") or "none",
            "count": len(flagged),
            "hyperscalers": [{"ticker": h, "signal": flagged.get(h)} for h in hypers],
            "trigger": capex_cfg.get("trigger", ""),
            "action": capex_cfg.get("action", ""),
        }

    # Show the whole plan (target tickers ∪ anything held), so intended stocks,
    # weights and kill lines are visible before they're bought; live weight and
    # drift fill in for held positions.
    weight_by = {p["ticker"]: p["weight_pct"] for p in positions_data}
    pnl_by = {p["ticker"]: p.get("pnl_pct") for p in positions_data}
    price_by = {p["ticker"]: p.get("current_price") for p in positions_data}
    cost_by = {p["ticker"]: p.get("avg_cost") for p in positions_data}
    tickers = list(dict.fromkeys(
        list(targets) + [p["ticker"] for p in positions_data] + sorted(watchplan.plan_tickers(store))
    ))
    rows = []
    for t in tickers:
        tgt = targets.get(t)
        held = t in weight_by
        weight = weight_by.get(t, 0.0)
        ratio = (weight / tgt) if (tgt and held) else None
        note = notes.get(t) or {}
        rule = rules.get(t) or {}
        horizon = horizons.get(t) or {}
        verdict = verdicts.get(t) or {}
        analysis = watchplan.get_analysis(store, t)
        applied = {(f["metric"], f["op"]) for f in (rule.get("fundamentals") or [])}
        pending = [s for s in watchplan.suggested_rules(analysis, store)
                   if (s["metric"], s["op"]) not in applied]
        left = watchplan.days_left(horizon.get("review_date"))
        # Live P&L against the drawdown line, so the panel shows how close a
        # position is to its kill without waiting for the daily watch. It goes
        # through watchplan.drawdown_for rather than comparing P&L directly:
        # the panel and the watch have to breach on the same number, and P&L is
        # measured from cost while the line is measured from the review anchor.
        pnl = pnl_by.get(t)
        max_dd = rule.get("max_drawdown_pct")
        dd = watchplan.drawdown_for(rule, price_sek=price_by.get(t),
                                    avg_cost_sek=cost_by.get(t))
        rows.append({
            "ticker": t,
            "held": held,
            "weight_pct": weight,
            "target_pct": tgt,
            "ratio": round(ratio, 2) if ratio else None,
            "over": bool(ratio and ratio >= 1.5),
            "next_earnings": note.get("next_earnings", ""),
            "watch": note.get("watch", ""),
            "kill": kills.get(t, ""),
            "add_criterion": add_texts.get(t, ""),
            "add_conditions": (watchplan.get_add_analysis(store, t) or {}).get("conditions") or [],
            "kill_verdict": verdict.get("verdict", ""),
            "kill_unverifiable": verdict.get("verdict") == "insufficient",
            "kill_reason": verdict.get("reason", ""),
            "kill_unchecked": verdict.get("unchecked", []),
            "kill_checked_on": verdict.get("date", ""),
            "conditions": (analysis or {}).get("conditions") or [],
            "logic": (analysis or {}).get("logic", ""),
            "pending_rules": pending,
            "max_drawdown_pct": max_dd,
            "price_below": rule.get("price_below"),
            "price_above": rule.get("price_above"),
            "fundamental_rules": rule.get("fundamentals") or [],
            "rule_currency": rule.get("currency") or "",
            "pnl_pct": pnl,
            "drawdown_pct": round(dd["pct"], 1) if dd["pct"] is not None else None,
            "drawdown_basis": dd["basis"],
            "anchor_date": dd["anchor_date"],
            "dd_breached": bool(max_dd and dd["pct"] is not None
                                and dd["pct"] <= -abs(max_dd)),
            "horizon_date": horizon.get("review_date", ""),
            "horizon_label": horizon.get("label", ""),
            "horizon_note": horizon.get("note", ""),
            "days_left": left,
            "horizon_due": left is not None and left <= 0,
            "horizon_near": left is not None and 0 < left <= 30,
        })
    # Most urgent first: kill lines breached, then horizons due/near, then drift.
    rows.sort(key=lambda r: (
        0 if r["dd_breached"] else 1,
        0 if r["horizon_due"] else (1 if r["horizon_near"] else 2),
        0 if r["over"] else 1,
        -(r["target_pct"] or 0),
        r["ticker"],
    ))
    return {
        "capex": capex,
        "rows": rows,
        "held_count": len(weight_by),
        "planned_count": len(tickers),
        "alerts": sum(1 for r in rows if r["dd_breached"] or r["horizon_due"] or r["over"]),
    }


def _add_status(store) -> dict | None:
    """Add signals for the dashboard, or None when no position carries a plan.

    The engine already answers "which gate stopped this?" for every position;
    this only shapes it for rendering. The panel exists because the answer was
    previously reachable only from the CLI or a Telegram alert — so a HOLD had
    no visible explanation anywhere you would actually look.
    """
    from fundmgr import addsignal

    rows = addsignal.evaluate_all(store)
    if not rows:
        return None

    actionable = {addsignal.ADD, addsignal.STRONG_ADD}
    return {
        "rows": rows,
        "books": list(addsignal.BOOK_GATES),
        "gates": addsignal.BOOK_GATES,
        "actionable": sum(1 for r in rows if r["state"] in actionable),
        "stale": sum(1 for r in rows if r["valuation_status"] == "stale"
                     or r["proof_status"] == "stale"),
    }


def _kill_verdicts(store, tickers: list[str]) -> dict[str, dict]:
    """Last judge verdict per ticker, for the Watch panel's criterion column.

    Surfaces the case that matters most: a criterion the judge could not check
    from the available evidence. Without this, an unverifiable kill line looks
    identical to one that is being watched and is holding.
    """
    out: dict[str, dict] = {}
    for ticker in tickers:
        raw = store.get_meta(f"paper_killverdict:{ticker}")
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        out[ticker] = {
            "verdict": data.get("verdict", ""),
            "reason": data.get("reason", ""),
            "unchecked": data.get("unchecked") or [],
            "date": data.get("date", ""),
            "coverage": data.get("coverage") or {},
        }
    return out


def _profile_risk(config_name: str) -> dict:
    """The source profile's overridable caps, for the review form's placeholders."""
    try:
        cfg = sleeve_review.load_profile_config(config_name)
    except ValueError:
        return {}
    return {k: getattr(cfg.risk, k) for k in sleeve_review.OVERRIDABLE_RISK}


def _not_found() -> HTMLResponse:
    return HTMLResponse("<h1>Portfolio not found</h1>", status_code=404)


def make_portfolio_router(prefix: str, kind: str, section_label: str,
                          accent: str, real: bool, home_template: str) -> APIRouter:
    """Build a portfolio dashboard router for one section.

    prefix:        URL prefix, e.g. "/paper" or "/live"
    kind:          book type filter — "paper" or "live"
    section_label: human name shown on the list page
    accent:        template accent key ("emerald" for paper, "sky" for live)
    real:          real-money framing + Watch panel when True
    home_template: list/create template ("paper.html" or "live.html")
    """
    router = APIRouter(prefix=prefix)

    def _base_ctx(meta: dict) -> dict:
        book_prefix = f"{prefix}/{meta['slug']}"
        model = meta.get("model_label")
        if real:
            banner = (f"LIVE SLEEVE — {meta['name']}"
                      + (f" · picks by {model}" if model else "")
                      + " · Real positions · Monitored: kill criteria · earnings · drift")
        else:
            banner = ("PAPER PORTFOLIO — " + meta["name"]
                      + (f" · picks by {model}" if model else "")
                      + " · Real market prices · Not real money")
        return {
            "is_simulation": True,
            "sim_prefix": book_prefix,
            "sim_label": meta["name"],
            "sim_accent": accent,
            "sim_banner": banner,
            "api_base": book_prefix,
            # per-book sidebar sub-nav (see base.html)
            "book_section": kind,
            "book_prefix": book_prefix,
            "book_name": meta["name"],
            "benchmark_label": benchmark_label(meta.get("benchmark")),
            # live sleeves get a "Record fill" form on the dashboard
            "fill_action": f"{book_prefix}/fill" if real else None,
            # kill criteria + time horizons are editable on every book
            "watchplan_action": f"{book_prefix}/watchplan",
            # add plans are a real-sleeve concern — a paper book has no tranches
            "addplan_action": f"{book_prefix}/addplan" if real else None,
        }

    def _home_ctx(request: Request, error: str | None = None, form: dict | None = None) -> dict:
        return {
            "request": request,
            "portfolios": _portfolio_summaries(kind),
            "active_page": kind,
            "section_label": section_label,
            "error": error,
            "form": form or {},
        }

    # ── List + create/import ────────────────────────────────────────────────

    @router.get("/", response_class=HTMLResponse)
    def home(request: Request):
        return _render(home_template, _home_ctx(request))

    @router.post("/create")
    async def create(
        request: Request,
        name: str = Form(...),
        capital_sek: float = Form(...),
        holdings_text: str = Form(...),
        base_prompt: str = Form(""),
        model_label: str = Form(""),
        benchmark: str = Form(paper.DEFAULT_BENCHMARK),
        holdings_json: str = Form(""),
    ):
        holdings_override = None
        if holdings_json.strip():
            try:
                parsed = json.loads(holdings_json)
                if isinstance(parsed, list) and parsed:
                    holdings_override = parsed
            except json.JSONDecodeError:
                pass
        try:
            slug, _log = paper.create_portfolio(
                name=name, capital_sek=capital_sek, holdings_text=holdings_text,
                base_prompt=base_prompt, model_label=model_label,
                benchmark=benchmark.strip() or paper.DEFAULT_BENCHMARK,
                holdings_override=holdings_override, kind=kind,
            )
        except ValueError as e:
            return _render(home_template, _home_ctx(request, error=str(e), form={
                "name": name, "capital_sek": capital_sek,
                "holdings_text": holdings_text, "base_prompt": base_prompt,
                "model_label": model_label, "benchmark": benchmark,
            }))
        return RedirectResponse(url=f"{prefix}/{slug}", status_code=303)

    @router.post("/import")
    async def do_import(
        request: Request,
        json_text: str = Form(...),
        name: str = Form(""),
        capital_sek: str = Form(""),
        benchmark: str = Form(paper.DEFAULT_BENCHMARK),
        model_label: str = Form("Claude Fable"),
    ):
        """Create a real monitored sleeve from a structured LLM answer (JSON)."""
        # Parse the whole paste as JSON first; only fall back to block extraction
        # (for JSON wrapped in prose/fences), since the object itself contains
        # nested arrays that _extract_json_block would otherwise grab.
        data = None
        for candidate in (json_text.strip(), paper._extract_json_block(json_text)):
            if not candidate:
                continue
            try:
                data = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
        if data is None:
            return _render(home_template, _home_ctx(
                request, error="Could not parse JSON — paste the full portfolio object.",
                form={"json_text": json_text, "name": name}))
        if not isinstance(data, dict) or not data.get("positions"):
            return _render(home_template, _home_ctx(
                request, error="JSON has no 'positions' array to import.",
                form={"json_text": json_text, "name": name}))

        parsed = paper.parse_structured_portfolio(data)
        try:
            cap = float(capital_sek) if capital_sek.strip() else parsed["capital_sek"]
        except (ValueError, AttributeError):
            cap = parsed["capital_sek"]
        if not cap:
            return _render(home_template, _home_ctx(
                request, error="No capital found in the JSON — enter a starting value.",
                form={"json_text": json_text, "name": name}))
        try:
            slug, _log = paper.create_portfolio(
                name=name.strip() or parsed["name"],
                capital_sek=float(cap),
                holdings_text=json.dumps(data, indent=2, ensure_ascii=False),
                model_label=model_label,
                benchmark=benchmark.strip() or paper.DEFAULT_BENCHMARK,
                holdings_override=parsed["holdings_override"],
                position_meta=parsed["position_meta"],
                capex_kill=parsed["capex_kill"],
                kind="live",
                execute_buys=False,  # import the plan; positions fill from trades
            )
        except ValueError as e:
            return _render(home_template, _home_ctx(
                request, error=str(e), form={"json_text": json_text, "name": name}))
        return RedirectResponse(url=f"{prefix}/{slug}", status_code=303)

    @router.post("/preview")
    async def preview(request: Request):
        """Parse pasted picks and resolve names to tickers (no price fetches)."""
        body = await request.json()
        try:
            holdings = paper.resolve_holdings(paper.parse_holdings(body.get("holdings_text", "")))
            holdings = paper.normalise_weights(holdings)
            unresolved = sum(1 for h in holdings if not h.get("ticker"))
            return {"ok": True, "holdings": holdings, "unresolved": unresolved,
                    "total_weight": round(sum(h["weight_pct"] for h in holdings), 1)}
        except ValueError as e:
            return {"ok": False, "error": str(e)}

    # ── Photo import ────────────────────────────────────────────────────────

    @router.post("/extract-photos")
    async def extract_photos(photos: list[UploadFile] = File(...)):
        """Read holdings out of uploaded portfolio screenshots.

        Returns rows for the review table — nothing is written until the user
        submits them to /create-from-photo, so a misread is always caught by a
        human before it becomes a position.
        """
        images: list[bytes] = []
        for upload in photos:
            raw = await upload.read()
            if not raw:
                continue
            if len(raw) > _MAX_UPLOAD_BYTES:
                return JSONResponse({
                    "ok": False,
                    "error": f"'{upload.filename}' is larger than "
                             f"{_MAX_UPLOAD_BYTES // (1024 * 1024)} MB — "
                             "a screenshot rather than a full-resolution photo works best.",
                }, status_code=400)
            images.append(raw)

        if not images:
            return JSONResponse({"ok": False, "error": "No image received."}, status_code=400)

        from fundmgr.vision.portfolio_photo import PhotoExtractionError, extract_holdings
        try:
            result = await run_in_threadpool(extract_holdings, images)
        except PhotoExtractionError as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"Photo import failed: {e}"},
                                status_code=500)

        holdings = result["holdings"]
        unresolved = sum(1 for h in holdings if not h.get("ticker"))
        return {
            "ok": True,
            "holdings": holdings,
            "cash_sek": result["cash_sek"],
            "total_value_sek": result["total_value_sek"],
            "account_name": result["account_name"],
            "notes": result["notes"],
            "warnings": result["warnings"],
            "unresolved": unresolved,
            "ocr_used": result["ocr_used"],
            "model": result["model"],
        }

    @router.post("/create-from-photo")
    async def create_from_photo(
        request: Request,
        name: str = Form(...),
        holdings_json: str = Form(...),
        cash_sek: str = Form("0"),
        benchmark: str = Form(paper.DEFAULT_BENCHMARK),
        model_label: str = Form(""),
    ):
        """Create a sleeve from the reviewed photo rows, seeded at cost basis."""
        def _fail(msg: str) -> HTMLResponse:
            return _render(home_template, _home_ctx(request, error=msg, form={"name": name}))

        try:
            rows = json.loads(holdings_json)
        except json.JSONDecodeError:
            return _fail("Could not read the reviewed holdings — try the upload again.")
        if not isinstance(rows, list) or not rows:
            return _fail("No holdings to import — upload a photo and review the rows first.")

        cash = paper._opt_float(cash_sek) or 0.0
        cost_total = 0.0
        for row in rows:
            shares = paper._opt_float(row.get("shares"))
            cost = paper._opt_float(row.get("avg_cost_sek"))
            if shares and cost:
                cost_total += shares * cost
        if cost_total <= 0:
            return _fail("Every row is missing its share count or cost basis — "
                         "fill those in on the review table before importing.")

        try:
            slug, log = paper.create_portfolio(
                name=name,
                capital_sek=cost_total + cash,
                holdings_text=json.dumps(rows, indent=2, ensure_ascii=False),
                model_label=model_label.strip() or "portfolio photo",
                benchmark=benchmark.strip() or paper.DEFAULT_BENCHMARK,
                holdings_override=rows,
                kind=kind,
                seed_holdings=True,
            )
        except ValueError as e:
            return _fail(str(e))

        # Remember how each row resolved, so the next screenshot carrying the
        # same ISIN, broker ticker or name needs no correction. Reviewed rows
        # are human-confirmed, which is exactly what makes them worth learning.
        from fundmgr import aliases
        corrected = 0
        for row in rows:
            ticker = str(row.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            if aliases.learn(ticker, isin=row.get("isin"),
                             raw_ticker=row.get("raw_ticker"), name=row.get("name")):
                if str(row.get("suggested_ticker") or "").strip().upper() != ticker:
                    corrected += 1

        skipped = sum(1 for line in log if line.startswith("⚠"))
        msg = f"Imported {name} from photo — {len(rows) - skipped} holdings seeded at cost."
        if skipped:
            msg += f" {skipped} row(s) skipped — see the log."
        if corrected:
            msg += (f" Remembered {corrected} ticker correction(s) for next time.")
        from urllib.parse import urlencode
        return RedirectResponse(
            url=f"{prefix}/{slug}?" + urlencode({"msg": msg, "ok": 1}), status_code=303)

    @router.post("/{slug}/delete")
    def delete(slug: str):
        try:
            paper.delete_portfolio(slug)
            _price_cache.pop(slug, None)
        except KeyError:
            pass
        return RedirectResponse(url=prefix, status_code=303)

    @router.post("/{slug}/fill")
    async def record_fill(
        slug: str,
        ticker: str = Form(...),
        shares: str = Form(...),
        price_sek: str = Form(...),
        fee_sek: str = Form("0"),
        side: str = Form("buy"),
        trade_date: str = Form(""),
    ):
        """Record a real broker fill into the sleeve (buy or sell, price in SEK).

        Seeds pre-existing holdings (buy at your average cost) and trims (sell),
        the browser equivalent of `/pfill` / `fund paper-fill`."""
        from datetime import datetime as _dt, timezone as _tz
        from fundmgr.state.models import NavPoint, Transaction

        def _back(msg: str, ok: int) -> RedirectResponse:
            from urllib.parse import urlencode
            return RedirectResponse(
                url=f"{prefix}/{slug}?" + urlencode({"msg": msg, "ok": ok}),
                status_code=303)

        try:
            meta, store = paper.open_portfolio(slug)
        except KeyError:
            return _not_found()

        tkr = (ticker or "").strip().upper()
        side = "sell" if side == "sell" else "buy"
        try:
            n_shares = float(str(shares).replace(",", "."))
            price = float(str(price_sek).replace(",", "."))
            fee = float(str(fee_sek).replace(",", ".") or 0)
        except ValueError:
            return _back("Shares, price and fee must be numbers.", 0)
        if not tkr or n_shares <= 0 or price <= 0:
            return _back("Enter a ticker, positive shares and a positive SEK price.", 0)

        ts = _dt.now(_tz.utc)
        if trade_date.strip():
            try:
                ts = _dt.strptime(trade_date.strip(), "%Y-%m-%d").replace(
                    hour=12, tzinfo=_tz.utc)
            except ValueError:
                return _back(f"Bad date '{trade_date}' — use YYYY-MM-DD.", 0)

        tkr, snap_note = paper.snap_ticker_to_plan(store, tkr)
        currency = meta["currency_map"].get(tkr, "SEK")
        store.apply_fill(Transaction(
            ticker=tkr, side=side, shares=n_shares, price_sek=price, fee_sek=fee,
            source="fill", currency=currency, timestamp=ts,
        ))
        _price_cache.pop(slug, None)
        try:
            bench_rows = store.get_benchmark()
            nav_cost = sum(p.shares * p.avg_cost_sek for p in store.get_positions()) + store.get_cash()
            store.upsert_nav(NavPoint(
                date=ts.strftime("%Y-%m-%d"),
                portfolio_nav_sek=nav_cost,
                benchmark_value=bench_rows[-1]["close"] if bench_rows else 0.0,
                cash_sek=store.get_cash(),
            ))
        except Exception:
            pass
        verb = "Bought" if side == "buy" else "Sold"
        tail = f" ({snap_note})" if snap_note else ""
        return _back(f"{verb} {n_shares:g} × {tkr} @ {price:,.2f} SEK in {meta['name']}.{tail}", 1)

    @router.post("/{slug}/watchplan")
    async def save_watchplan(
        slug: str,
        ticker: str = Form(...),
        kill_criterion: str = Form(""),
        add_criterion: str = Form(""),
        max_drawdown_pct: str = Form(""),
        price_below: str = Form(""),
        price_above: str = Form(""),
        horizon_date: str = Form(""),
        horizon_months: str = Form(""),
        horizon_note: str = Form(""),
        re_anchor: str = Form(""),
        remove: str = Form(""),
    ):
        """Set (or clear) one position's kill criteria and time horizon."""
        from urllib.parse import urlencode

        def _back(msg: str, ok: int) -> RedirectResponse:
            return RedirectResponse(
                url=f"{prefix}/{slug}?" + urlencode({"msg": msg, "ok": ok}) + "#watch",
                status_code=303)

        try:
            meta, store = paper.open_portfolio(slug)
        except KeyError:
            return _not_found()

        tkr = (ticker or "").strip().upper()
        if not tkr:
            return _back("Pick a ticker to set a kill criterion for.", 0)

        if remove:
            watchplan.clear_position_plan(store, tkr)
            return _back(f"Cleared the watch plan for {tkr}.", 1)

        if horizon_date.strip() and not watchplan.parse_horizon(horizon_date.strip()):
            return _back(f"Bad horizon date '{horizon_date}' — use YYYY-MM-DD.", 0)

        # The editor pre-fills the existing horizon date, and an explicit date
        # beats a month count. So a date that comes back *unchanged* is not a
        # choice — it is the old value riding along, and it silently swallowed
        # the months the investor just typed. Months win in that case.
        stored_date = (watchplan.get_horizons(store).get(tkr) or {}).get("review_date", "")
        if horizon_months.strip() and horizon_date.strip() == stored_date:
            horizon_date = ""

        previous_criterion = watchplan.get_kill_text(store).get(tkr, "")
        previous_add = watchplan.get_add_text(store).get(tkr, "")
        currency = meta["currency_map"].get(tkr)
        plan = watchplan.set_position_plan(
            store, tkr,
            kill_criterion=kill_criterion,
            # "" clears a rule, so pass the raw form values straight through.
            max_drawdown_pct=max_drawdown_pct,
            price_below=price_below,
            price_above=price_above,
            currency=currency,
            review_date=horizon_date,
            horizon_months=horizon_months,
            horizon_note=horizon_note,
            re_anchor=bool(re_anchor),
        )

        # Read the criterion back at save time, so how it will be judged is
        # visible now rather than discovered from a verdict tomorrow.
        if kill_criterion.strip() and kill_criterion.strip() != previous_criterion:
            analysis = await run_in_threadpool(
                lambda: watchplan.analyse_criterion(kill_criterion.strip(), store=store))
            watchplan.save_analysis(store, tkr, analysis)
        elif not kill_criterion.strip():
            watchplan.save_analysis(store, tkr, None)

        # The ADD criterion gets the same treatment, with the framing that says
        # these are conditions which must hold rather than ones that falsify.
        add_text = watchplan.set_add_text(store, tkr, add_criterion)
        if add_text and add_text != previous_add:
            add_analysis = await run_in_threadpool(
                lambda: watchplan.analyse_criterion(add_text, kind="add", store=store))
            watchplan.save_add_analysis(store, tkr, add_analysis)
        elif not add_text:
            watchplan.save_add_analysis(store, tkr, None)

        bits = []
        if plan["kill_criterion"]:
            bits.append("kill criterion")
        if add_text:
            bits.append("ADD criterion")
        rules = plan["kill_rules"]
        if rules.get("max_drawdown_pct"):
            # Say what the line is measured from — the whole point of the
            # anchor is that it is visible, not inferred.
            anchored = (f" from {rules['anchor_price_sek']:,.2f} SEK on "
                        f"{rules.get('anchor_date', '')}"
                        if rules.get("anchor_price_sek") else " from cost")
            bits.append(f"-{rules['max_drawdown_pct']:g}% drawdown line{anchored}")
        if rules.get("price_below"):
            bits.append(f"floor {rules['price_below']:g}")
        if rules.get("price_above"):
            bits.append(f"target {rules['price_above']:g}")
        if plan["horizon"].get("review_date"):
            bits.append(f"horizon {plan['horizon']['review_date']}")
        if not bits:
            return _back(f"Cleared the watch plan for {tkr}.", 1)
        return _back(f"{tkr}: saved {', '.join(bits)}.", 1)

    @router.post("/{slug}/watchplan/apply-rules")
    async def apply_suggested_rules(slug: str, ticker: str = Form(...)):
        """Turn the thresholds found in a criterion into deterministic rules.

        The criterion text stays exactly as written — this only adds the
        machine-checkable half alongside it, so the same line is both judged in
        context and compared against a number every run.
        """
        from urllib.parse import urlencode

        try:
            _meta, store = paper.open_portfolio(slug)
        except KeyError:
            return _not_found()

        tkr = (ticker or "").strip().upper()
        suggestions = watchplan.suggested_rules(watchplan.get_analysis(store, tkr), store)
        if not suggestions:
            msg, ok = f"No checkable thresholds found in {tkr}'s criterion.", 0
        else:
            existing = (watchplan.get_kill_rules(store).get(tkr) or {}).get("fundamentals") or []
            watchplan.set_position_plan(
                store, tkr, fundamentals=existing + suggestions)
            names = ", ".join(f"{s['label']} {s['op']} {s['value']:g}" for s in suggestions)
            msg, ok = f"{tkr}: now checking {names} every run.", 1
        return RedirectResponse(
            url=f"{prefix}/{slug}?" + urlencode({"msg": msg, "ok": ok}) + "#watch",
            status_code=303)

    @router.post("/{slug}/addplan")
    async def save_addplan(
        slug: str,
        ticker: str = Form(...),
        book: str = Form(""),
        max_weight_pct: str = Form(""),
        tranche_pct: str = Form(""),
        target_price: str = Form(""),
        review_price: str = Form(""),
        anchor_live: str = Form(""),
        proof: str = Form(""),
        remove: str = Form(""),
    ):
        """Set (or clear) one position's add plan — the web twin of
        `fund paper-add-plan`.

        Proof and the review price are separate submits rather than fields,
        because both are dated confirmations rather than settings: stamping one
        as a side effect of saving an unrelated field is how a stale anchor
        starts looking fresh.
        """
        from urllib.parse import urlencode

        from fundmgr import addsignal

        def _back(msg: str, ok: int) -> RedirectResponse:
            return RedirectResponse(
                url=f"{prefix}/{slug}?" + urlencode({"msg": msg, "ok": ok}) + "#adds",
                status_code=303)

        try:
            meta, store = paper.open_portfolio(slug)
        except KeyError:
            return _not_found()

        tkr = (ticker or "").strip().upper()
        if not tkr:
            return _back("Pick a ticker to set an add plan for.", 0)

        if remove:
            addsignal.clear_plan(store, tkr)
            return _back(f"Cleared the add plan for {tkr}.", 1)

        if proof:
            confirmed = proof == "yes"
            addsignal.set_plan(store, tkr, proof_confirmed=confirmed)
            return _back(
                f"{tkr}: fundamental proof {'confirmed' if confirmed else 'withdrawn'}."
                + (" The target price still dates from before the last report — "
                   "restate it to clear the valuation gate."
                   if confirmed and addsignal.valuation_status(store, tkr)["status"] == "stale"
                   else ""), 1)

        anchored = review_price
        if anchor_live:
            # Dislocation compares SEK prices, so the anchor must be SEK. The
            # daily track refreshes the cache, so that is the normal path; a
            # live quote covers a plan set on a ticker not yet tracked.
            price = watchplan._current_price_sek(store, tkr)
            if not price:
                try:
                    from fundmgr.data.quotes import live_price
                    native = live_price(tkr)
                    currency = meta["currency_map"].get(tkr, "SEK")
                    price = (native if currency == "SEK"
                             else paper.to_sek_price(native, currency, store)) if native else None
                except Exception:
                    price = None
            if not price:
                return _back(f"No price available for {tkr} to anchor the review at.", 0)
            anchored = price

        try:
            plan = addsignal.set_plan(
                store, tkr, book=book or None, max_weight_pct=max_weight_pct,
                tranche_pct=tranche_pct, target_price=target_price,
                review_price=anchored)
        except ValueError as e:
            return _back(str(e), 0)

        bits = []
        if plan.get("book"):
            bits.append(f"book {plan['book']}")
        if plan.get("max_weight_pct"):
            bits.append(f"max {plan['max_weight_pct']:g}%")
        if plan.get("tranche_pct"):
            bits.append(f"tranche {plan['tranche_pct']:g}%")
        if plan.get("target_price"):
            bits.append(f"target {plan['target_price']:,.2f}")
        if plan.get("review_price"):
            bits.append(f"review anchor {plan['review_price']:,.2f}")
        if not bits:
            return _back(f"Cleared the add plan for {tkr}.", 1)
        return _back(f"{tkr}: saved {', '.join(bits)}.", 1)
    # ── Sleeve review (live sleeves only) ───────────────────────────────────

    @router.post("/{slug}/review")
    def start_review(slug: str, req: ReviewRequest):
        """Kick off a background re-decision of this sleeve."""
        global _review_job
        if not real:
            raise HTTPException(status_code=404, detail="Reviews run on live sleeves only")
        try:
            _meta, store = paper.open_portfolio(slug)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"No sleeve {slug!r}") from None

        config_name = req.config or sleeve_review.review_defaults(store)["config"]
        if config_name not in {pr["config"] for pr in sleeve_review.list_profiles()}:
            raise HTTPException(status_code=400, detail=f"Unknown profile: {config_name}")
        if req.country and req.country.upper() not in {
            sc["code"] for sc in sleeve_review.list_scopes(config_name)
        }:
            raise HTTPException(status_code=400, detail=f"Unknown scope: {req.country}")
        if req.provider or req.model_id:
            valid = {(m["provider"], m["model_id"]) for m in sleeve_review.MODEL_OPTIONS}
            if (req.provider, req.model_id) not in valid:
                raise HTTPException(status_code=400, detail="Unknown provider/model combination")

        with _review_lock:
            if _review_job and _review_job["status"] == "running":
                raise HTTPException(
                    status_code=409,
                    detail=f"A review of {_review_job['slug']!r} is already running")
            job_id = uuid.uuid4().hex[:12]
            _review_job = {
                "id": job_id, "slug": slug, "status": "running",
                "params": req.model_dump(), "result": None, "error": None,
                "started": time.time(),
            }
        threading.Thread(target=_run_review, args=(job_id, slug, req), daemon=True).start()
        return {"job_id": job_id}

    @router.get("/{slug}/review/scopes")
    def review_scopes(slug: str, config: str = ""):
        """Countries selectable for one source profile — the scope dropdown
        repopulates from here when the profile changes."""
        if not real:
            raise HTTPException(status_code=404, detail="Reviews run on live sleeves only")
        config_name = config or sleeve_review.DEFAULT_REVIEW_CONFIG
        if config_name not in {pr["config"] for pr in sleeve_review.list_profiles()}:
            raise HTTPException(status_code=400, detail=f"Unknown profile: {config_name}")
        return {"scopes": list(sleeve_review.list_scopes(config_name))}

    @router.get("/{slug}/review/jobs/{job_id}")
    def review_job_status(slug: str, job_id: str):
        with _review_lock:
            if not _review_job or _review_job["id"] != job_id:
                raise HTTPException(status_code=404, detail="Unknown job")
            return {
                "id": _review_job["id"],
                "slug": _review_job["slug"],
                "status": _review_job["status"],
                "params": _review_job["params"],
                "elapsed_s": round(time.time() - _review_job["started"], 1),
                "result": _review_job["result"],
                "error": _review_job["error"],
            }

    # ── Per-book dashboard ──────────────────────────────────────────────────

    @router.get("/{slug}", response_class=HTMLResponse)
    def index(request: Request, slug: str):
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

        flash = None
        msg = request.query_params.get("msg")
        if msg:
            flash = {"msg": msg, "ok": request.query_params.get("ok") == "1"}

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
                        # Holds included: a review that decided to keep the book
                        # as it is has decided something, and an empty table for
                        # it reads as "the run went nowhere".
                        for a in actions
                    ],
                }
            except Exception as e:
                # Never silently: a decision that is on disk but unrenderable
                # looked exactly like no decision at all.
                last_run = {
                    "run_id": last_rec.run_id,
                    "timestamp": last_rec.timestamp.strftime("%Y-%m-%d %H:%M"),
                    "market_summary": "",
                    "notes": f"This decision could not be rendered: {e}",
                    "buys": 0, "sells": 0, "holds": 0, "actions": [],
                }

        review = None
        if real:
            defaults = sleeve_review.review_defaults(store)
            try:
                scopes = sleeve_review.list_scopes(defaults["config"])
            except ValueError:
                scopes = ()
            # A review runs for minutes in a background thread. Without this the
            # page has no idea one is in flight or just finished, so navigating
            # away loses the result — and, worse, loses the error when it failed.
            with _review_lock:
                job = dict(_review_job) if _review_job and _review_job["slug"] == slug else None
            review = {
                "slug": slug,
                "job": {
                    "id": job["id"],
                    "status": job["status"],
                    "elapsed_s": round(time.time() - job["started"], 1),
                } if job else None,
                "action": f"{prefix}/{slug}/review",
                "jobs_url": f"{prefix}/{slug}/review/jobs",
                "scopes_url": f"{prefix}/{slug}/review/scopes",
                "profiles": sleeve_review.list_profiles(),
                "scopes": list(scopes),
                "models": sleeve_review.MODEL_OPTIONS,
                "max_runs": sleeve_review.MAX_RUNS,
                "config": defaults["config"],
                "country": defaults["country"],
                "risk": defaults["risk"],
                # The profile's own caps, so the form can show what an empty
                # override field will actually run under.
                "profile_risk": _profile_risk(defaults["config"]),
            }

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
            # Live sleeves only — paper books keep the plain simulation framing.
            "watch": _watch_status(store, positions_data) if real else None,
            "adds": _add_status(store) if real else None,
            "review": review,
            "flash": flash,
            **_base_ctx(meta),
        })

    @router.get("/{slug}/history", response_class=HTMLResponse)
    def history(request: Request, slug: str):
        """Every decision on this book, newest first.

        The dashboard shows only the latest one, and its "All decisions" link
        used to point at /history — the *main fund's* history — so a sleeve
        review that saved perfectly still looked like it had gone nowhere.
        """
        try:
            meta, store = paper.open_portfolio(slug)
        except KeyError:
            return _not_found()

        with store._conn() as conn:
            rows = conn.execute(
                "SELECT run_id, timestamp, llm_response, actions_json "
                "FROM recommendations ORDER BY timestamp DESC LIMIT 50"
            ).fetchall()

        recommendations = []
        for r in rows:
            try:
                actions = json.loads(r["actions_json"])
            except Exception:
                actions = []
            try:
                llm = json.loads(r["llm_response"] or "{}")
            except Exception:
                llm = {}
            # Holds are carried too: a review whose whole answer was "keep
            # everything" is a decision, and showing an empty card for it is
            # what makes a saved run look lost.
            recommendations.append({
                "run_id": r["run_id"],
                "timestamp": (r["timestamp"] or "")[:16].replace("T", " "),
                "action_count": len(actions),
                "buys": sum(1 for a in actions if a.get("side") == "buy"),
                "sells": sum(1 for a in actions if a.get("side") == "sell"),
                "holds": sum(1 for a in actions if a.get("side") == "hold"),
                "market_summary": llm.get("market_summary", ""),
                "notes": llm.get("notes", ""),
                "is_review": r["run_id"].startswith("review-"),
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
                    for a in actions
                ],
            })

        return _render("book_history.html", {
            "request": request,
            "recommendations": recommendations,
            "active_page": "history",
            **_base_ctx(meta),
        })

    @router.get("/{slug}/transactions", response_class=HTMLResponse)
    def transactions(request: Request, slug: str):
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
    def learnings(request: Request, slug: str):
        try:
            meta, store = paper.open_portfolio(slug)
        except KeyError:
            return _not_found()
        # Shared builder, not a local copy — a portfolio's page must report the
        # same injected/retained split as the funds' (this route had drifted).
        from fundmgr.config import AppConfig
        from fundmgr.web.views import learnings_context

        cfg = AppConfig()
        cfg.name = meta["name"]
        return _render("learnings.html", {
            "request": request,
            **learnings_context(cfg, store),
            **_base_ctx(meta),
        })

    @router.get("/{slug}/prompt", response_class=HTMLResponse)
    def prompt(request: Request, slug: str):
        try:
            meta, store = paper.open_portfolio(slug)
        except KeyError:
            return _not_found()
        pasted = store.get_meta("paper_pasted_text") or ""
        mandate = meta["base_prompt"] or "(no base prompt was saved for this portfolio)"
        if pasted:
            mandate += "\n\n── Pasted picks ──\n\n" + pasted

        kills = json.loads(store.get_meta("paper_kill_criteria") or "{}")
        held = {p.ticker for p in store.get_positions()}
        kill_rows = []
        for ticker, criterion in sorted(kills.items()):
            if not criterion:
                continue
            hits = []
            with store._conn() as conn:
                rows = conn.execute(
                    "SELECT key, value FROM app_meta WHERE key LIKE ? ORDER BY key DESC",
                    (f"paper_killhit:{ticker}:%",),
                ).fetchall()
            for r in rows:
                hits.append({"date": r["key"].rsplit(":", 1)[1], "reason": r["value"]})
            kill_rows.append({
                "ticker": ticker, "criterion": criterion,
                "held": ticker in held, "hits": hits,
            })

        return _render("prompt.html", {
            "request": request,
            "mandate": mandate,
            "mandate_filename": f"asked of {meta['model_label']}" if meta["model_label"] else "base prompt",
            "guidance": {"current": None, "history": []},
            "kill_rows": kill_rows,
            "active_page": "prompt",
            **_base_ctx(meta),
        })

    @router.get("/{slug}/api/nav")
    def api_nav(slug: str):
        try:
            meta, store = paper.open_portfolio(slug)
        except KeyError:
            return {"data": [], "layout": {}}
        return json.loads(nav_chart_json(
            store.get_nav_history(), benchmark_label(meta.get("benchmark"))))

    @router.get("/{slug}/api/stats")
    def api_stats(slug: str):
        try:
            meta, store = paper.open_portfolio(slug)
        except KeyError:
            return {}
        return compute_stats(store.get_nav_history(), meta["capital_sek"])

    return router


# Paper portfolios (pasted sims) and Live sleeves (real monitored books) — same
# engine + per-book dashboards, different framing and list/create path.
router = make_portfolio_router(
    "/paper", "paper", "Paper Portfolios", "emerald", real=False, home_template="paper.html")
live_router = make_portfolio_router(
    "/live", "live", "Live Sleeves", "sky", real=True, home_template="live.html")
