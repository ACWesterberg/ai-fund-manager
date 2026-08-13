"""
Add signals — the other half of a monitoring system.

A kill criterion asks "has the thesis become materially worse?". An add
criterion asks a different question: "has the probability-weighted return become
materially better?" That happens when the business improves, when the price
falls while the business does not, or both — and the two need labelling
differently, because only one of them makes the company more valuable.

The naive versions of each are traps. "Good news → buy" adds to an already
expensive stock; "price down 20% → buy" automates catching falling knives. So a
signal is never a trigger on its own:

    (proof OR dislocation) AND valuation AND weight AND NOT killed

The gates carry the weight here, not the triggers.

  • **Proof** is company-specific and mostly lives in the quarterly report —
    organic growth, ARR, adjusted margins. Very little of it is in any price
    feed, so it is a human-confirmed flag with a date, refreshed at each print.
  • **Dislocation** is measured against the last thesis-confirmed *review
    price*, never against original cost. A position up 300% that falls 25% from
    its review price is dislocated; measured from cost it looks untouched.
  • **Valuation** is the gate that stops chasing: the expected annualised
    return implied by your own target review price over the time remaining.
  • **Weight** stops a good idea becoming accidental concentration.

The dangerous failure mode this module exists to prevent: a stock halves because
intrinsic value halved, and the engine cheerfully compares the new price with a
target set before the news and calls it cheap. So a target price that predates a
material event is STALE, and a stale valuation suppresses every add signal.

Kill always wins. ADD and KILL never coexist.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone

from fundmgr.state.store import Store

ADD_PLAN_KEY = "paper_add_plan"

# Per-book gates. A riskier book must clear a wider dislocation and a higher
# expected return before an add is entertained at all.
BOOK_GATES: dict[str, dict[str, float]] = {
    "A":   {"dislocation_pct": -12.0, "min_expected_annual_return": 15.0},
    "A/B": {"dislocation_pct": -15.0, "min_expected_annual_return": 20.0},
    "B":   {"dislocation_pct": -20.0, "min_expected_annual_return": 25.0},
}
DEFAULT_BOOK = "A"

# States, least to most actionable. KILL overrides everything above it.
HOLD, ADD_WATCH, ADD, STRONG_ADD, KILL = (
    "hold", "add-watch", "add", "strong-add", "kill")

# Events that invalidate a target review price. All are recorded in app_meta by
# the existing watches, keyed with the date they happened.
_STALE_EVENT_PREFIXES = (
    "paper_earnhint:",   # a report landed
    "paper_earnpost:",   # …and the check-the-print reminder fired
    "paper_killhit:",    # a kill criterion may have triggered
)


# ── Stored plan ───────────────────────────────────────────────────────────────

def _load(store: Store) -> dict:
    try:
        data = json.loads(store.get_meta(ADD_PLAN_KEY) or "{}")
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def get_plans(store: Store) -> dict[str, dict]:
    return {t: v for t, v in _load(store).items() if isinstance(v, dict)}


def get_plan(store: Store, ticker: str) -> dict:
    return get_plans(store).get(ticker.upper(), {})


def _num(value) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        return float(str(value).replace(",", ".").strip())
    except (TypeError, ValueError):
        return None


def _today() -> date:
    return datetime.now(timezone.utc).date()


def set_plan(
    store: Store,
    ticker: str,
    *,
    book: str | None = None,
    max_weight_pct=None,
    tranche_pct=None,
    target_price=None,
    review_price=None,
    proof_confirmed: bool | None = None,
    today: date | None = None,
) -> dict:
    """Set the add-side plan for one position. Only provided fields change.

    Setting a target price stamps it with today's date — that stamp is what
    staleness is measured against, so it must never be back-dated silently.
    Setting a review price does the same: it is the anchor dislocation is
    measured from, and re-anchoring is an explicit act of confirming the thesis
    at the current price.
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        raise ValueError("Ticker is required.")
    stamp = (today or _today()).isoformat()

    plans = get_plans(store)
    plan = dict(plans.get(ticker) or {})

    if book is not None:
        cleaned = book.strip().upper().replace("AB", "A/B")
        if cleaned and cleaned not in BOOK_GATES:
            raise ValueError(f"Unknown book '{book}' — use one of {', '.join(BOOK_GATES)}.")
        plan["book"] = cleaned or DEFAULT_BOOK
    for field, value in (("max_weight_pct", max_weight_pct),
                         ("tranche_pct", tranche_pct)):
        if value is not None:
            parsed = _num(value)
            if parsed is None or parsed <= 0:
                plan.pop(field, None)
            else:
                plan[field] = parsed

    if target_price is not None:
        parsed = _num(target_price)
        if parsed is None or parsed <= 0:
            plan.pop("target_price", None)
            plan.pop("target_set_at", None)
        else:
            plan["target_price"] = parsed
            plan["target_set_at"] = stamp

    if review_price is not None:
        parsed = _num(review_price)
        if parsed is None or parsed <= 0:
            plan.pop("review_price", None)
            plan.pop("review_date", None)
        else:
            plan["review_price"] = parsed
            plan["review_date"] = stamp

    if proof_confirmed is not None:
        if proof_confirmed:
            plan["proof_confirmed_at"] = stamp
        else:
            plan.pop("proof_confirmed_at", None)

    if plan:
        plans[ticker] = plan
    else:
        plans.pop(ticker, None)
    store.set_meta(ADD_PLAN_KEY, json.dumps(plans))
    return plan


def clear_plan(store: Store, ticker: str) -> None:
    plans = get_plans(store)
    if plans.pop((ticker or "").strip().upper(), None) is not None:
        store.set_meta(ADD_PLAN_KEY, json.dumps(plans))


# ── Gates ─────────────────────────────────────────────────────────────────────

def expected_annual_return(target_price: float, current_price: float,
                           months_left: float) -> float | None:
    """Annualised return implied by reaching the target over the time left.

    (target / price) ^ (12 / months) − 1, as a percentage. None when it cannot
    be computed — a horizon that has already passed cannot be annualised, and
    pretending otherwise would manufacture a huge number from a small gap.
    """
    if not target_price or not current_price or current_price <= 0:
        return None
    if not months_left or months_left <= 0:
        return None
    try:
        return ((target_price / current_price) ** (12.0 / months_left) - 1) * 100
    except (OverflowError, ValueError, ZeroDivisionError):
        return None


def valuation_status(store: Store, ticker: str, plan: dict | None = None,
                     today: date | None = None) -> dict:
    """Whether the target review price can be trusted.

    Returns {status, reason, events}: "unset" with no target, "stale" when a
    thesis-changing event was recorded after the target was set, else "fresh".
    Staleness reads the same app_meta breadcrumbs the earnings and kill watches
    already leave, so no new bookkeeping is needed to notice a report landed.
    """
    plan = get_plan(store, ticker) if plan is None else plan
    set_at = plan.get("target_set_at")
    if not plan.get("target_price") or not set_at:
        return {"status": "unset", "reason": "no target review price set", "events": []}

    ticker = ticker.upper()
    events: list[str] = []
    with store._conn() as conn:
        rows = conn.execute("SELECT key FROM app_meta WHERE key LIKE ?",
                            (f"%:{ticker}:%",)).fetchall()
    for row in rows:
        key = row["key"]
        if not key.startswith(_STALE_EVENT_PREFIXES):
            continue
        when = key.rsplit(":", 1)[-1]
        # Keys end in the event date; anything after the stamp invalidates it.
        if len(when) == 10 and when > set_at:
            events.append(key)

    if events:
        return {"status": "stale",
                "reason": f"{len(events)} thesis-changing event(s) since {set_at}",
                "events": sorted(events)}
    return {"status": "fresh", "reason": f"set {set_at}", "events": []}


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(store: Store, ticker: str, *, price: float | None = None,
             weight_pct: float | None = None, killed: bool = False,
             today: date | None = None) -> dict:
    """Full add-signal assessment for one position.

    Every gate reports its own pass/fail and why, so the dashboard can explain a
    HOLD rather than just asserting it — "which gate stopped this?" is the
    question a monitoring system has to answer.
    """
    from fundmgr import watchplan

    today = today or _today()
    ticker = ticker.upper()
    plan = get_plan(store, ticker)
    book = plan.get("book") or DEFAULT_BOOK
    gates = BOOK_GATES.get(book, BOOK_GATES[DEFAULT_BOOK])

    # Proof: human-confirmed at the last report. Almost none of these metrics
    # (organic growth, ARR, adjusted margins) exist in any price feed.
    proof_at = plan.get("proof_confirmed_at")
    proof = bool(proof_at)

    # Dislocation from the thesis-confirmed anchor, never from cost.
    review_price = plan.get("review_price")
    dislocation_pct = None
    if review_price and price:
        dislocation_pct = (price / review_price - 1) * 100
    dislocated = (dislocation_pct is not None
                  and dislocation_pct <= gates["dislocation_pct"])

    # Valuation gate, over whatever time the horizon has left.
    valuation = valuation_status(store, ticker, plan, today)
    horizon = watchplan.get_horizons(store).get(ticker) or {}
    days_left = watchplan.days_left(horizon.get("review_date"), today)
    months_left = (days_left / 30.44) if days_left and days_left > 0 else None
    expected = expected_annual_return(plan.get("target_price"), price, months_left)
    valuation_ok = (valuation["status"] == "fresh" and expected is not None
                    and expected >= gates["min_expected_annual_return"])

    # Weight gate: is there room for a tranche under the ceiling?
    max_weight = plan.get("max_weight_pct")
    tranche = plan.get("tranche_pct") or 1.0
    room = None
    if max_weight is not None and weight_pct is not None:
        room = max_weight - weight_pct
    weight_ok = room is None or room >= tranche
    over_max = (max_weight is not None and weight_pct is not None
                and weight_pct > max_weight)

    state, why = _classify(killed, proof, dislocated, valuation_ok, weight_ok,
                           valuation, expected, gates, room, tranche, over_max)

    return {
        "ticker": ticker,
        "state": state,
        "why": why,
        "book": book,
        "proof": proof,
        "proof_confirmed_at": proof_at,
        "dislocation_pct": round(dislocation_pct, 1) if dislocation_pct is not None else None,
        "dislocation_gate": gates["dislocation_pct"],
        "dislocated": dislocated,
        "review_price": review_price,
        "review_date": plan.get("review_date"),
        "target_price": plan.get("target_price"),
        "target_set_at": plan.get("target_set_at"),
        "valuation_status": valuation["status"],
        "valuation_reason": valuation["reason"],
        "expected_annual_return": round(expected, 1) if expected is not None else None,
        "return_gate": gates["min_expected_annual_return"],
        "valuation_ok": valuation_ok,
        "months_left": round(months_left, 1) if months_left else None,
        "weight_pct": weight_pct,
        "max_weight_pct": max_weight,
        "tranche_pct": tranche,
        "weight_room": round(room, 1) if room is not None else None,
        "weight_ok": weight_ok,
        "over_max_weight": over_max,
        "price": price,
    }


def _classify(killed, proof, dislocated, valuation_ok, weight_ok,
              valuation, expected, gates, room, tranche, over_max):
    """The state machine, with the reason the state was reached."""
    if killed:
        return KILL, "kill criterion active — add signals are suppressed"
    if over_max:
        return HOLD, "above max weight — concentration review, not an add"
    if not (proof or dislocated):
        return HOLD, "no fundamental proof and no price dislocation"

    # A stale target is the failure this module exists to prevent: the price
    # fell because value fell, and the old target makes it look cheap.
    if valuation["status"] == "stale":
        return ADD_WATCH, f"valuation is stale ({valuation['reason']}) — recalculate fair value"
    if valuation["status"] == "unset":
        return ADD_WATCH, "no target review price — set one before adding"
    if not valuation_ok:
        got = f"{expected:.0f}%" if expected is not None else "n/a"
        return HOLD, (f"expected return {got} is below the "
                      f"{gates['min_expected_annual_return']:.0f}% gate for this book")
    if not weight_ok:
        return HOLD, (f"only {room:.1f}pp of room below max weight, "
                      f"tranche is {tranche:.1f}pp")

    if proof and dislocated:
        return STRONG_ADD, "fundamental proof and price dislocation together"
    if proof:
        return ADD, "fundamental proof confirmed, gates clear"
    return ADD_WATCH, "price dislocation without fundamental proof — needs review"


def evaluate_all(store: Store, today: date | None = None) -> list[dict]:
    """Assess every position carrying an add plan, most actionable first."""
    from fundmgr import paper, watchplan

    plans = get_plans(store)
    if not plans:
        return []

    positions = store.get_positions()
    currency_map = json.loads(store.get_meta("paper_currency_map") or "{}")
    tickers = sorted({p.ticker for p in positions} | set(plans))

    native: dict[str, float] = {}
    for t in tickers:
        rows = store.get_prices(t)
        if rows and rows[-1].get("close"):
            native[t] = float(rows[-1]["close"])
    prices = paper.sek_prices_for(store, tickers, currency_map, native)

    market_value = {p.ticker: p.shares * prices.get(p.ticker, p.avg_cost_sek)
                    for p in positions}
    nav = sum(market_value.values()) + store.get_cash()

    killed = {row["ticker"] for row in watchplan.evaluate_kill_rules(store) if row["hit"]}
    texts = watchplan.get_kill_text(store)
    for t in texts:
        raw = store.get_meta(f"paper_killverdict:{t}")
        if raw:
            try:
                if json.loads(raw).get("verdict") == "yes":
                    killed.add(t)
            except json.JSONDecodeError:
                pass

    out = []
    for ticker in sorted(plans):
        weight = (market_value.get(ticker, 0.0) / nav * 100) if nav > 0 else None
        out.append(evaluate(store, ticker, price=prices.get(ticker),
                            weight_pct=weight, killed=ticker in killed, today=today))
    order = {STRONG_ADD: 0, ADD: 1, ADD_WATCH: 2, KILL: 3, HOLD: 4}
    out.sort(key=lambda r: (order.get(r["state"], 9), r["ticker"]))
    return out


# ── Watch ─────────────────────────────────────────────────────────────────────

_STATE_EMOJI = {STRONG_ADD: "🟢🟢", ADD: "🟢", ADD_WATCH: "🔵", KILL: "🚨", HOLD: ""}


def check_add_signals(slug: str, store: Store | None = None,
                      today: date | None = None) -> list[str]:
    """Alert when a position becomes addable, once per state change.

    Transition-based like the kill watches: an ADD that stays ADD for six weeks
    is one message, not forty. HOLD and KILL are never announced here — the kill
    watches own that side.
    """
    if store is None:
        from fundmgr.paper import open_portfolio
        meta, store = open_portfolio(slug)
        name = meta["name"]
    else:
        name = store.get_meta("paper_name") or slug

    rows = evaluate_all(store, today)
    if not rows:
        return []

    log: list[str] = []
    fired: list[dict] = []
    for row in rows:
        ticker, state = row["ticker"], row["state"]
        previous = store.get_meta(f"paper_addstate:{ticker}") or HOLD
        if state == previous:
            continue
        store.set_meta(f"paper_addstate:{ticker}", state)
        if state in (ADD, STRONG_ADD, ADD_WATCH):
            fired.append(row)
            log.append(f"{_STATE_EMOJI[state]} {ticker}: {state.upper()} — {row['why']}")
        else:
            log.append(f"  {ticker}: {previous} → {state}")

    if fired:
        from fundmgr.notify.send import send_telegram
        lines = [f"<b>{name} — add signals</b>"]
        for row in fired:
            lines.append(f"\n{_STATE_EMOJI[row['state']]} <b>{row['ticker']}</b>: "
                         f"{row['state'].upper()}")
            lines.append(f"  {row['why']}")
            if row["expected_annual_return"] is not None:
                lines.append(f"  Expected {row['expected_annual_return']:.0f}%/yr to "
                             f"{row['target_price']:,.0f} over {row['months_left']:.0f}m "
                             f"(gate {row['return_gate']:.0f}%)")
            if row["dislocation_pct"] is not None:
                lines.append(f"  {row['dislocation_pct']:+.1f}% vs review price "
                             f"{row['review_price']:,.0f} (gate "
                             f"{row['dislocation_gate']:+.0f}%)")
            if row["state"] in (ADD, STRONG_ADD) and row["max_weight_pct"]:
                new_weight = (row["weight_pct"] or 0) + row["tranche_pct"]
                lines.append(f"  Suggested +{row['tranche_pct']:.1f}pp → "
                             f"{new_weight:.1f}% (max {row['max_weight_pct']:.1f}%)")
        lines.append("\nA suggestion, not an instruction — verify before acting.")
        send_telegram("\n".join(lines))
    return log
