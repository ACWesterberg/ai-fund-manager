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
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from jinja2 import Environment, FileSystemLoader
from starlette.concurrency import run_in_threadpool

from fundmgr import paper, watchplan
from fundmgr.reporting.dashboard import compute_stats, nav_chart_json

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
    rules = watchplan.get_kill_rules(store)
    horizons = watchplan.get_horizons(store)
    verdicts = _kill_verdicts(store, list(kills))
    # Anything held is worth a row even with no plan on it yet — that's how the
    # first kill criterion gets set. Only a book with nothing at all opts out.
    if not (capex_cfg or targets or kills or rules or horizons or positions_data):
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
        pending = [s for s in watchplan.suggested_rules(analysis)
                   if (s["metric"], s["op"]) not in applied]
        left = watchplan.days_left(horizon.get("review_date"))
        # Live P&L against the drawdown line, so the panel shows how close a
        # position is to its kill without waiting for the daily watch.
        pnl = pnl_by.get(t)
        max_dd = rule.get("max_drawdown_pct")
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
            "dd_breached": bool(max_dd and pnl is not None and pnl <= -abs(max_dd)),
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
            # live sleeves get a "Record fill" form on the dashboard
            "fill_action": f"{book_prefix}/fill" if real else None,
            # kill criteria + time horizons are editable on every book
            "watchplan_action": f"{book_prefix}/watchplan",
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
        max_drawdown_pct: str = Form(""),
        price_below: str = Form(""),
        price_above: str = Form(""),
        horizon_date: str = Form(""),
        horizon_months: str = Form(""),
        horizon_note: str = Form(""),
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

        previous_criterion = watchplan.get_kill_text(store).get(tkr, "")
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
        )

        # Read the criterion back at save time, so how it will be judged is
        # visible now rather than discovered from a verdict tomorrow.
        if kill_criterion.strip() and kill_criterion.strip() != previous_criterion:
            analysis = await run_in_threadpool(watchplan.analyse_criterion,
                                               kill_criterion.strip())
            watchplan.save_analysis(store, tkr, analysis)
        elif not kill_criterion.strip():
            watchplan.save_analysis(store, tkr, None)

        bits = []
        if plan["kill_criterion"]:
            bits.append("kill criterion")
        rules = plan["kill_rules"]
        if rules.get("max_drawdown_pct"):
            bits.append(f"-{rules['max_drawdown_pct']:g}% drawdown line")
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
        suggestions = watchplan.suggested_rules(watchplan.get_analysis(store, tkr))
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
            # Live sleeves only — paper books keep the plain simulation framing.
            "watch": _watch_status(store, positions_data) if real else None,
            "flash": flash,
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
            _meta, store = paper.open_portfolio(slug)
        except KeyError:
            return {"data": [], "layout": {}}
        return json.loads(nav_chart_json(store.get_nav_history()))

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
