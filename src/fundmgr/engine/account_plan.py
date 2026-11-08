"""
Monitoring plans for a broker account — the half of a decision you did not make.

You choose what to buy. What a trade almost never arrives with is the rest: what
would prove it wrong, what would justify adding, what price it is worth, how far
it may fall before the idea needs re-reading, and when to look again. A live
sleeve gets those from the review that opened the name (`sleeve_review` asks for
them on every buy); an account has no review, so this writes them after the fact,
for names you already own.

One call per account, all requested names together, so each plan is written
knowing the rest of the book. Each name comes back with:

  • kill_criterion / add_criterion — checkable text, judged daily against news
    and figures by the same watches a live sleeve uses;
  • target_price (trading currency) — installed both as the take-profit line the
    kill-rule watch alerts on and as the add plan's valuation target;
  • max_drawdown_pct — the review line, anchored to today's price;
  • horizon_months — when to re-run the thesis;
  • a stance (hold / add / trim / exit) with its reason — advice, recorded as the
    account's latest decision so the dashboard shows it, and never executed.

Writes only into empty fields unless `overwrite` is set: a criterion you have
since rewritten by hand is yours, and regenerating must not quietly replace it.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from fundmgr import paper
from fundmgr.state.models import RecommendationLog
from fundmgr.state.store import Store


class PositionPlan(BaseModel):
    ticker: str = Field(description="The Yahoo ticker exactly as given in the holdings list")
    stance: Literal["hold", "add", "trim", "exit"] = Field(
        description="What you would do with this position today, given everything shown")
    stance_reason: str = Field(
        max_length=400,
        description="1–2 sentences: why that stance now. Name the figure or event it rests on.")
    thesis: str = Field(
        max_length=500,
        description=("The most plausible reason to own this, in 1–3 sentences. The investor "
                     "did not say why they bought it — write the case the business supports, "
                     "not a guess at their motive."))
    kill_criterion: str = Field(
        max_length=400,
        description=("What would falsify the thesis, checkable against a report or headline: "
                     "name the metric and the level (e.g. 'organic growth below 3% for two "
                     "quarters'). Never a share-price move."))
    add_criterion: str = Field(
        max_length=400,
        description=("What must still be true before more money goes in — conditions that "
                     "must HOLD, with metric and level."))
    target_price: float | None = Field(
        default=None, gt=0,
        description=("Fair value in the stock's own trading currency, at the level the thesis "
                     "implies over the horizon. Reaching it is the prompt to take profit."))
    max_drawdown_pct: float | None = Field(
        default=None, ge=5, le=60,
        description=("Fall from today's price at which the idea must be re-read (not an "
                     "automatic sell). Wider for volatile names, tighter for stable ones."))
    horizon_months: int | None = Field(
        default=None, ge=1, le=36, description="Months until the thesis should be re-run")
    suggested_weight_pct: float | None = Field(
        default=None, ge=0, le=100,
        description="For trim/add: the % of account value you would move it to. 0 for exit.")
    next_earnings: str | None = Field(
        default=None, description="Next report date YYYY-MM-DD if you know it, else null")


class AccountPlans(BaseModel):
    summary: str = Field(
        max_length=800,
        description="2–4 sentences on the account as a whole: concentration, what to watch.")
    plans: list[PositionPlan]


_SYSTEM = """\
You are the monitoring analyst for a private investor's own brokerage account. \
The investor makes every trade themselves; nothing you write is executed. Your \
job is to give each holding the plan a disciplined fund manager would have \
written when opening it — so a daily watch can tell the investor, without being \
asked, when a position has done its job, when its story has broken, and when it \
deserves more money.

Rules:
- Every criterion must be checkable from a company report, a reported figure or \
a news event. Name the metric and the level. "Execution falters" is not a \
criterion; "EBIT margin below 12% in a quarterly report" is.
- A kill criterion is about the business, never the share price — the price \
line is max_drawdown_pct, and hitting it means re-read, not sell.
- target_price is in the stock's own trading currency (as shown per holding), \
at the level your thesis actually implies. Don't anchor on the investor's cost.
- The stance is your honest view today. Hold is a decision too; say what it \
rests on. Recommend trim or exit only for a reason you can name: thesis broken, \
valuation stretched past the target, or a position too large for the account.
- Use only what is shown and what you reliably know about the company. Where \
the figures are missing, say so in the reason instead of inventing them.
- Return one plan for every ticker in the holdings list, using the ticker exactly.
"""


def _fundamentals_block(store: Store, ticker: str) -> str:
    try:
        from fundmgr.evidence import current_metric, field_meta
    except Exception:
        return ""
    bits = []
    for key, meta in field_meta(store).items():
        if meta.get("custom"):
            continue
        try:
            value = current_metric(store, ticker, key)
        except Exception:
            value = None
        if value is None:
            continue
        unit = "%" if meta.get("is_fraction") else meta.get("unit", "")
        bits.append(f"{meta.get('label', key)} {value:,.2f}{unit}")
    return "; ".join(bits)


def _latest_native(store: Store, ticker: str) -> float | None:
    rows = store.get_prices(ticker)
    return rows[-1]["close"] if rows else None


def build_context(meta: dict, store: Store, tickers: list[str],
                  headlines: bool = True) -> str:
    from fundmgr import accounts

    names = accounts.names(store)
    currency_map = meta["currency_map"]
    positions = {p.ticker: p for p in store.get_positions()}
    native = {t: _latest_native(store, t) for t in positions}
    sek = paper.sek_prices_for(store, list(positions), currency_map,
                               {t: v for t, v in native.items() if v})
    value = {t: p.shares * sek.get(t, p.avg_cost_sek) for t, p in positions.items()}
    cash = store.get_cash()
    nav = sum(value.values()) + cash

    lines = [
        f"## Account: {meta['name']}",
        f"Value {nav:,.0f} SEK, of which cash {cash:,.0f} SEK "
        f"({(cash / nav * 100) if nav else 0:.1f}%). {len(positions)} holdings.",
        "",
        "## Whole book (for context — weights by market value)",
    ]
    for t, p in sorted(positions.items(), key=lambda kv: -value[kv[0]]):
        lines.append(f"- {t} ({names.get(t, t)}): {value[t] / nav * 100 if nav else 0:.1f}%")

    lines += ["", "## Holdings to plan"]
    for t in tickers:
        p = positions.get(t)
        if p is None:
            continue
        cur = currency_map.get(t, "SEK")
        px = native.get(t)
        pnl = (sek[t] / p.avg_cost_sek - 1) * 100 if t in sek and p.avg_cost_sek else None
        lines.append(f"\n### {t} — {names.get(t, t)}")
        lines.append(
            f"Shares {p.shares:g}; cost {p.avg_cost_sek:,.2f} SEK/share; "
            + (f"last price {px:,.2f} {cur}; " if px else "no recent price; ")
            + (f"P&L {pnl:+.1f}% on cost; " if pnl is not None else "")
            + f"weight {value[t] / nav * 100 if nav else 0:.1f}%")
        fund = _fundamentals_block(store, t)
        if fund:
            lines.append(f"Fundamentals: {fund}")
        if headlines:
            news = paper._recent_headlines(t, max_items=5)
            if news:
                lines.append("Recent headlines:\n" + "\n".join(f"  - {h}" for h in news))
    lines += ["", "Write a monitoring plan for each holding listed under "
              "'Holdings to plan'. Return AccountPlans JSON."]
    return "\n".join(lines)


def _plan_config():
    from fundmgr.config import load_config
    path = os.getenv("FUND_ACCOUNT_PLAN_CONFIG")
    return load_config(Path(path) if path else None)


def generate(meta: dict, store: Store, tickers: list[str], cfg=None) -> AccountPlans:
    from fundmgr.engine.client import call_llm
    cfg = cfg or _plan_config()
    parsed, _raw = call_llm(_SYSTEM, build_context(meta, store, tickers), cfg,
                            schema=AccountPlans)
    return parsed


def install(store: Store, meta: dict, plan: PositionPlan, *, overwrite: bool = False,
            analyse: bool = True) -> list[str]:
    """Write one plan into the book. Returns the fields actually written."""
    from fundmgr import addsignal, watchplan

    t = plan.ticker.strip().upper()
    currency = meta["currency_map"].get(t)
    rule = watchplan.get_kill_rules(store).get(t) or {}
    written: list[str] = []
    kw: dict = {}
    if plan.kill_criterion.strip() and (overwrite or not watchplan.get_kill_text(store).get(t)):
        kw["kill_criterion"] = plan.kill_criterion.strip()
    if plan.max_drawdown_pct and (overwrite or not rule.get("max_drawdown_pct")):
        kw["max_drawdown_pct"] = plan.max_drawdown_pct
    if plan.target_price and (overwrite or not rule.get("price_above")):
        kw["price_above"] = plan.target_price
    if plan.horizon_months and (overwrite or not watchplan.get_horizons(store).get(t)):
        kw["horizon_months"] = plan.horizon_months
    if kw:
        # A regenerated plan is a fresh review, so its drawdown line measures
        # from today; a first plan anchors to today regardless.
        watchplan.set_position_plan(
            store, t, currency=currency,
            re_anchor=overwrite and "max_drawdown_pct" in kw, **kw)
        written += list(kw)
        if analyse and "kill_criterion" in kw:
            watchplan.save_analysis(store, t, watchplan.analyse_criterion(
                kw["kill_criterion"], store=store))

    if plan.add_criterion.strip() and (overwrite or not watchplan.get_add_text(store).get(t)):
        text = watchplan.set_add_text(store, t, plan.add_criterion)
        written.append("add_criterion")
        if analyse and text:
            watchplan.save_add_analysis(store, t, watchplan.analyse_criterion(
                text, kind="add", store=store))
    if plan.target_price and (overwrite or not addsignal.get_plan(store, t).get("target_price")):
        addsignal.set_plan(store, t, target_price=plan.target_price)
        written.append("target_price")

    notes = json.loads(store.get_meta("paper_position_notes") or "{}")
    note = dict(notes.get(t) or {})
    if plan.thesis.strip() and (overwrite or not note.get("thesis")):
        note["thesis"] = plan.thesis.strip()
        written.append("thesis")
    if plan.next_earnings:
        note["next_earnings"] = plan.next_earnings
    note["stance"] = plan.stance
    note["stance_reason"] = plan.stance_reason
    note["stance_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    notes[t] = note
    store.set_meta("paper_position_notes", json.dumps(notes))
    return written


_SIDE = {"hold": "hold", "add": "buy", "trim": "sell", "exit": "sell"}


def _record(store: Store, slug: str, result: AccountPlans, weights: dict[str, float],
            model: str) -> str:
    """Keep the stances as the account's latest decision, so the dashboard shows
    them. Advice only: nothing seeds an outcome or touches the ledger."""
    now = datetime.now(timezone.utc)
    # A fill-triggered plan and a button press can land in the same second.
    run_id = f"plan-{slug}-{now.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
    actions = []
    for p in result.plans:
        weight = weights.get(p.ticker, 0.0)
        target = 0.0 if p.stance == "exit" else (
            p.suggested_weight_pct if p.suggested_weight_pct is not None
            and p.stance in ("add", "trim") else weight)
        actions.append({
            "ticker": p.ticker, "side": _SIDE[p.stance], "target_weight_pct": round(target, 2),
            "confidence": None, "sek_estimate": 0,
            "thesis": f"{p.stance.upper()}: {p.stance_reason}",
            "kill_criterion": p.kill_criterion, "add_criterion": p.add_criterion,
            "target_price": p.target_price, "stop_loss_pct": None,
        })
    store.save_recommendation(RecommendationLog(
        run_id=run_id, timestamp=now,
        prompt_snapshot=json.dumps({"kind": "account_plan", "model": model}),
        llm_response=json.dumps({
            "market_summary": result.summary,
            "notes": "Monitoring plan and stance for each holding — advice only; "
                     "you place every trade.",
        }),
        guardrail_log="{}",
        actions_json=json.dumps(actions),
    ))
    return run_id


def write_plans(slug: str, tickers: list[str] | None = None, *, overwrite: bool = False,
                notify: bool = True, cfg=None) -> list[dict]:
    """Generate and install plans for `tickers` (default: every holding).

    Returns one dict per plan with the model's fields plus `written` (what was
    actually installed) and `currency`. Raises on an LLM failure — a caller that
    must not fail (a sync) catches it and reports the gap.
    """
    meta, store = paper.open_portfolio(slug)
    held = [p.ticker for p in store.get_positions()]
    wanted = [t for t in (tickers or held) if t in held]
    if not wanted:
        return []
    try:
        paper._cache_price_history(store, wanted, lookback_days=10)
    except Exception:
        pass

    cfg = cfg or _plan_config()
    result = generate(meta, store, wanted, cfg=cfg)

    native = {t: _latest_native(store, t) for t in held}
    sek = paper.sek_prices_for(store, held, meta["currency_map"],
                               {t: v for t, v in native.items() if v})
    positions = store.get_positions()
    nav = sum(p.shares * sek.get(p.ticker, p.avg_cost_sek) for p in positions) + store.get_cash()
    weights = {p.ticker: p.shares * sek.get(p.ticker, p.avg_cost_sek) / nav * 100
               for p in positions} if nav else {}

    out: list[dict] = []
    seen: set[str] = set()
    for plan in result.plans:
        plan.ticker = plan.ticker.strip().upper()
        if plan.ticker not in wanted or plan.ticker in seen:
            continue
        seen.add(plan.ticker)
        written = install(store, meta, plan, overwrite=overwrite)
        out.append({**plan.model_dump(), "written": written,
                    "currency": meta["currency_map"].get(plan.ticker, "")})
    result.plans = [p for p in result.plans if p.ticker in seen]
    _record(store, slug, result, weights, cfg.llm.model_id)

    if notify and out:
        from fundmgr.notify.send import send_telegram
        send_telegram(format_plans_message(meta["name"], result.summary, out,
                                           missing=[t for t in wanted if t not in seen]))
    return out


def format_plans_message(name: str, summary: str, plans: list[dict],
                         missing: list[str] | None = None) -> str:
    from fundmgr.accounts import _esc, plan_lines

    act = [p for p in plans if p["stance"] != "hold"]
    lines = [f"<b>🏦 {_esc(name)} — monitoring plan</b>", _esc(summary)]
    if act:
        lines.append("\n<b>Worth acting on:</b>")
    for p in act + [p for p in plans if p["stance"] == "hold"]:
        lines.append(f"\n<b>{_esc(p['ticker'])}</b>")
        lines += ["  " + s for s in plan_lines(p)]
    if missing:
        lines.append(f"\n⚠ No plan came back for: {', '.join(_esc(t) for t in missing)}")
    lines.append("\nAdvice only — nothing is traded. Edit any line on the dashboard.")
    return "\n".join(lines)


def refresh_all(*, overwrite: bool = False, notify: bool = True) -> list[str]:
    """Re-run the plan for every account. Weekly cron: fills plans for anything
    still unplanned and refreshes every stance; existing criteria stay unless
    `overwrite`."""
    from fundmgr import accounts
    log: list[str] = []
    for meta in accounts.list_accounts():
        try:
            plans = write_plans(meta["slug"], overwrite=overwrite, notify=notify)
            acting = sum(1 for p in plans if p["stance"] != "hold")
            log.append(f"{meta['name']}: {len(plans)} plan(s), {acting} worth acting on")
        except Exception as e:
            log.append(f"⚠ {meta['name']}: {e}")
    return log or ["No broker accounts yet."]
