"""
Per-position watch plan: kill criteria and time horizons.

A holding carries three kinds of pre-registered exit condition, all optional
and all editable from the live dashboard:

  • **Kill criterion (text)** — the observable fact that would falsify the
    thesis ("loses the Apple socket", "capex guide cut"). Judged daily against
    fresh headlines by `paper.check_kill_criteria`.
  • **Kill rules (numeric)** — hard lines that need no interpretation: a max
    drawdown from cost, a price floor, a price ceiling. Checked here, against
    prices, deterministically.
  • **Time horizon** — the date the thesis is supposed to have played out by.
    Nearing it is the cue to run fresh analysis rather than to sell, so the
    alerts escalate as it approaches (30 / 14 / 7 / 1 days, then on the day).

Everything lives in the portfolio's own `app_meta`, alongside the plan keys
`paper_kill_criteria` / `paper_target_weights` that the JSON import writes, so
a book imported from a structured answer and one built from a photo end up in
exactly the same shape.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone

from fundmgr.state.store import Store

# app_meta keys
KILL_TEXT_KEY = "paper_kill_criteria"      # {ticker: "text"} — shared with the JSON import
KILL_RULES_KEY = "paper_kill_rules"        # {ticker: {max_drawdown_pct, price_below, price_above, fundamentals[], currency}}
HORIZONS_KEY = "paper_horizons"            # {ticker: {review_date, label, note, set_at}}
ANALYSIS_KEY = "paper_killanalysis"        # {ticker: {logic, conditions[], criterion}} — per-ticker key

# Days-remaining buckets for horizon alerts. One alert per bucket per horizon:
# the tightest bucket the date falls into fires, so a horizon set 10 days out
# opens at "14 days" rather than replaying every earlier stage.
HORIZON_STAGES = (30, 14, 7, 1, 0)

# A drawdown kill re-arms only after the position recovers this many
# percentage points above its kill line — stops a position sitting exactly on
# the line from alerting every single day.
_KILL_RESET_BUFFER = 2.0


# ── Reading / writing the plan ────────────────────────────────────────────────

def _load(store: Store, key: str) -> dict:
    try:
        data = json.loads(store.get_meta(key) or "{}")
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def get_kill_text(store: Store) -> dict[str, str]:
    return {t: str(v or "") for t, v in _load(store, KILL_TEXT_KEY).items()}


def get_kill_rules(store: Store) -> dict[str, dict]:
    return {t: v for t, v in _load(store, KILL_RULES_KEY).items() if isinstance(v, dict)}


def get_horizons(store: Store) -> dict[str, dict]:
    return {t: v for t, v in _load(store, HORIZONS_KEY).items() if isinstance(v, dict)}


def _num(value) -> float | None:
    """Positive-or-None float coercion for user-entered rule fields.

    Price floors, targets and drawdown percentages are all positive by
    construction, so a 0 or a negative is a typo rather than a line.
    """
    out = _signed_num(value)
    return out if out is not None and out > 0 else None


def _signed_num(value) -> float | None:
    """Any-sign float coercion — for thresholds where 0 and negatives are real
    (free cash flow below 0, margin above -5)."""
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        return float(str(value).replace(",", ".").strip())
    except (TypeError, ValueError):
        return None


def parse_horizon(review_date: str = "", months: str | float | None = None,
                  today: date | None = None) -> tuple[str, str] | None:
    """Turn user input into (ISO review date, label).

    Accepts an explicit YYYY-MM-DD, or a number of months from today, or both
    (the explicit date wins). Returns None when neither is usable.
    """
    today = today or datetime.now(timezone.utc).date()
    text = (review_date or "").strip()
    if text:
        try:
            parsed = datetime.strptime(text, "%Y-%m-%d").date()
        except ValueError:
            return None
        months_out = round((parsed - today).days / 30.44)
        label = f"{months_out} months" if months_out >= 2 else f"{max(0, (parsed - today).days)} days"
        return parsed.isoformat(), label

    n = _num(months)
    if not n:
        return None
    total = today.month - 1 + int(round(n))
    year = today.year + total // 12
    month = total % 12 + 1
    # Clamp to the last valid day of the target month (31 Jan + 1 month → 28/29 Feb).
    day = today.day
    while day > 28:
        try:
            return date(year, month, day).isoformat(), f"{int(round(n))} months"
        except ValueError:
            day -= 1
    return date(year, month, day).isoformat(), f"{int(round(n))} months"


def set_position_plan(
    store: Store,
    ticker: str,
    *,
    kill_criterion: str | None = None,
    max_drawdown_pct=None,
    price_below=None,
    price_above=None,
    fundamentals: list[dict] | None = None,
    currency: str | None = None,
    review_date: str | None = None,
    horizon_months=None,
    horizon_note: str | None = None,
    today: date | None = None,
) -> dict:
    """Set (or clear) one position's kill criterion, kill rules and horizon.

    Every field is optional and independent: passing None leaves that part of
    the plan untouched, passing "" (or 0) clears it. Returns the resulting plan
    row for that ticker.

    Editing a rule re-arms its alert, so a tightened line can fire again today
    instead of staying suppressed by the previous rule's state.
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        raise ValueError("Ticker is required.")

    if kill_criterion is not None:
        texts = get_kill_text(store)
        cleaned = kill_criterion.strip()
        if cleaned:
            texts[ticker] = cleaned
        else:
            texts.pop(ticker, None)
        store.set_meta(KILL_TEXT_KEY, json.dumps(texts))

    if any(v is not None for v in (max_drawdown_pct, price_below, price_above,
                                   fundamentals)):
        rules = get_kill_rules(store)
        rule = dict(rules.get(ticker) or {})
        for field, value in (("max_drawdown_pct", max_drawdown_pct),
                             ("price_below", price_below),
                             ("price_above", price_above)):
            if value is None:
                continue
            parsed = _num(value)
            if parsed is None:
                rule.pop(field, None)
            else:
                rule[field] = parsed
        if fundamentals is not None:
            cleaned = _clean_fundamental_rules(fundamentals)
            if cleaned:
                rule["fundamentals"] = cleaned
            else:
                rule.pop("fundamentals", None)
        if currency:
            rule["currency"] = currency.strip().upper()
        rule = {k: v for k, v in rule.items() if k == "currency" or v is not None}
        if any(k in rule for k in ("max_drawdown_pct", "price_below", "price_above",
                                   "fundamentals")):
            rule["set_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            rules[ticker] = rule
        else:
            rules.pop(ticker, None)
        store.set_meta(KILL_RULES_KEY, json.dumps(rules))
        # Re-arm: a changed line deserves a fresh look today.
        store.set_meta(f"paper_killrule_state:{ticker}", "clear")

    if review_date is not None or horizon_months is not None or horizon_note is not None:
        horizons = get_horizons(store)
        current = dict(horizons.get(ticker) or {})
        if review_date is not None or horizon_months is not None:
            parsed = parse_horizon(review_date or "", horizon_months, today=today)
            if parsed is None:
                horizons.pop(ticker, None)
                store.set_meta(f"paper_horizon_stage:{ticker}", "")
                current = {}
            else:
                iso, label = parsed
                if current.get("review_date") != iso:
                    store.set_meta(f"paper_horizon_stage:{ticker}", "")  # re-arm on a new date
                current.update({
                    "review_date": iso,
                    "label": label,
                    "set_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })
        if horizon_note is not None and current:
            current["note"] = horizon_note.strip()
        if current:
            horizons[ticker] = current
        store.set_meta(HORIZONS_KEY, json.dumps(horizons))

    return plan_for(store, ticker)


def _clean_fundamental_rules(rules: list[dict]) -> list[dict]:
    """Keep only well-formed {metric, op, value} entries naming a known metric."""
    from fundmgr.evidence import FUND_FIELD_META

    out, seen = [], set()
    for entry in rules or []:
        if not isinstance(entry, dict):
            continue
        metric = str(entry.get("metric") or "").strip()
        op = str(entry.get("op") or "").strip().lower()
        value = _signed_num(entry.get("value"))
        if metric not in FUND_FIELD_META or op not in ("below", "above") or value is None:
            continue
        key = (metric, op)
        if key in seen:
            continue
        seen.add(key)
        out.append({"metric": metric, "op": op, "value": value,
                    "note": str(entry.get("note") or "").strip()})
    return out


def clear_position_plan(store: Store, ticker: str) -> None:
    """Remove every watch-plan entry for one ticker."""
    ticker = (ticker or "").strip().upper()
    for key, loader in ((KILL_TEXT_KEY, get_kill_text),
                        (KILL_RULES_KEY, get_kill_rules),
                        (HORIZONS_KEY, get_horizons)):
        data = loader(store)
        if ticker in data:
            data.pop(ticker)
            store.set_meta(key, json.dumps(data))
    store.set_meta(f"paper_killrule_state:{ticker}", "")
    store.set_meta(f"paper_horizon_stage:{ticker}", "")
    save_analysis(store, ticker, None)


def plan_for(store: Store, ticker: str) -> dict:
    ticker = ticker.upper()
    horizon = get_horizons(store).get(ticker) or {}
    return {
        "ticker": ticker,
        "kill_criterion": get_kill_text(store).get(ticker, ""),
        "kill_rules": get_kill_rules(store).get(ticker) or {},
        "horizon": horizon,
        "days_left": days_left(horizon.get("review_date")),
    }


def days_left(review_date: str | None, today: date | None = None) -> int | None:
    if not review_date:
        return None
    try:
        target = datetime.strptime(str(review_date), "%Y-%m-%d").date()
    except ValueError:
        return None
    return (target - (today or datetime.now(timezone.utc).date())).days


def plan_tickers(store: Store) -> set[str]:
    """Every ticker carrying any part of a watch plan."""
    return set(get_kill_text(store)) | set(get_kill_rules(store)) | set(get_horizons(store))


# ── Reading a criterion back at you ───────────────────────────────────────────

_ANALYSIS_SYSTEM = """\
You take a pre-registered kill criterion for a stock position — one sentence \
written by an investor — and report how it decomposes, so the investor can see \
how it will be read before relying on it.

Split it into its separate conditions and describe how they combine. Then, for
each condition, say how it could be checked:

  "fundamentals" - it is a threshold on ONE of the metrics listed below, and
                   that metric genuinely measures what the condition is about.
                   Fill in metric, op and value.
  "news"         - it is an observable event or disclosure that could plausibly
                   appear in news coverage or an earnings write-up.
  "manual"       - it needs the full report, a private figure, or a judgement
                   no feed will settle (segment-level detail, deal-level
                   returns, management intent).

Rules:
- Only use "fundamentals" for a metric in the list. Never invent a metric key.
- If the available metric is only an approximation of what was written, still
  use it, and say exactly how it differs in `note`. Example: a criterion about
  *recurring* revenue growth can use total revenue growth as a proxy — note it.
- Values are plain numbers in the unit shown: percentages as 8 (not 0.08).
- `op` is "below" or "above".
- Do not soften a "manual" condition into "news" to look useful. An honest
  "manual" is the point of this exercise.

Reply with a JSON object only:
{"logic": "<how the conditions combine, e.g. '(A and B) or C'>",
 "conditions": [
   {"id": "A", "text": "<the condition in the investor's own words>",
    "checkable": "fundamentals"|"news"|"manual",
    "metric": "<key from the list, or null>",
    "op": "below"|"above"|null,
    "value": <number or null>,
    "note": "<how the check differs from what was written, or ''>"}
 ]}\
"""


def analyse_criterion(criterion: str, model: str | None = None) -> dict | None:
    """Decompose a kill criterion into its conditions and how each can be checked.

    Runs once when a criterion is saved, so the investor sees the reading at
    entry time rather than discovering it from a verdict a day later. Returns
    None when it can't run (no API key, call failed) — the criterion is still
    saved and still judged; only the preview is missing.
    """
    criterion = (criterion or "").strip()
    if not criterion or not os.getenv("OPENAI_API_KEY"):
        return None

    from fundmgr.evidence import FUND_FIELD_META

    catalogue = "\n".join(f"  {key} — {meta['label']}"
                          for key, meta in FUND_FIELD_META.items())
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        kwargs = {
            "model": model or os.getenv("FUND_KILL_MODEL", "gpt-4o-mini"),
            "messages": [
                {"role": "system", "content": _ANALYSIS_SYSTEM},
                {"role": "user", "content": (
                    f"Available fundamentals metrics:\n{catalogue}\n\n"
                    f"Kill criterion:\n{criterion}")},
            ],
            "max_tokens": 700,
            "temperature": 0.0,
        }
        try:
            resp = client.chat.completions.create(
                response_format={"type": "json_object"}, **kwargs)
        except Exception:
            resp = client.chat.completions.create(**kwargs)
        raw = (resp.choices[0].message.content or "").strip()
    except Exception:
        return None

    return _parse_analysis(raw, criterion)


def _parse_analysis(raw: str, criterion: str) -> dict | None:
    """Validate the analyser's reply, dropping anything malformed."""
    from fundmgr.evidence import FUND_FIELD_META
    from fundmgr.paper import extract_json_object

    data = None
    for candidate in (raw, extract_json_object(raw or "")):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(data, dict) or not isinstance(data.get("conditions"), list):
        return None

    conditions = []
    for i, cond in enumerate(data["conditions"]):
        if not isinstance(cond, dict):
            continue
        text = str(cond.get("text") or "").strip()
        if not text:
            continue
        checkable = str(cond.get("checkable") or "manual").strip().lower()
        if checkable not in ("fundamentals", "news", "manual"):
            checkable = "manual"

        metric = cond.get("metric")
        metric = str(metric).strip() if metric else None
        op = str(cond.get("op") or "").strip().lower()
        value = _signed_num(cond.get("value"))
        # A fundamentals condition is only trustworthy if every part of the
        # comparison survived validation — otherwise it degrades to "news".
        if checkable == "fundamentals" and not (
                metric in FUND_FIELD_META and op in ("below", "above") and value is not None):
            checkable = "news" if metric else "manual"
            metric, op, value = None, None, None

        conditions.append({
            "id": str(cond.get("id") or chr(65 + i))[:2],
            "text": text,
            "checkable": checkable,
            "metric": metric if checkable == "fundamentals" else None,
            "op": op if checkable == "fundamentals" else None,
            "value": value if checkable == "fundamentals" else None,
            "note": str(cond.get("note") or "").strip(),
        })

    if not conditions:
        return None
    return {
        "criterion": criterion,
        "logic": str(data.get("logic") or "").strip(),
        "conditions": conditions,
        "analysed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def save_analysis(store: Store, ticker: str, analysis: dict | None) -> None:
    key = f"{ANALYSIS_KEY}:{ticker.upper()}"
    store.set_meta(key, json.dumps(analysis) if analysis else "")


def get_analysis(store: Store, ticker: str) -> dict | None:
    raw = store.get_meta(f"{ANALYSIS_KEY}:{ticker.upper()}")
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    # A stale analysis of a since-reworded criterion is worse than none.
    if data.get("criterion") != get_kill_text(store).get(ticker.upper(), ""):
        return None
    return data


def suggested_rules(analysis: dict | None) -> list[dict]:
    """The fundamentals conditions an analysis found, ready to become rules."""
    if not analysis:
        return []
    from fundmgr.evidence import FUND_FIELD_META
    out = []
    for cond in analysis["conditions"]:
        if cond["checkable"] != "fundamentals":
            continue
        out.append({
            "metric": cond["metric"],
            "label": FUND_FIELD_META[cond["metric"]]["label"],
            "op": cond["op"],
            "value": cond["value"],
            "note": cond["note"],
        })
    return out


# ── Numeric kill-rule watch ───────────────────────────────────────────────────

def _latest_native_prices(store: Store, tickers: list[str]) -> dict[str, float]:
    """Latest cached close per ticker, falling back to a live quote.

    track_portfolio refreshes the price cache before the watches run, so the
    cache is the normal path; the live fallback covers a ticker whose history
    hasn't been fetched yet (e.g. a plan entry that isn't held).
    """
    out: dict[str, float] = {}
    missing: list[str] = []
    for t in tickers:
        rows = store.get_prices(t)
        if rows and rows[-1].get("close"):
            out[t] = float(rows[-1]["close"])
        else:
            missing.append(t)
    if missing:
        try:
            from fundmgr.data.quotes import live_prices
            out.update({t: p for t, p in live_prices(missing).items() if p})
        except Exception:
            pass
    return out


def evaluate_kill_rules(store: Store) -> list[dict]:
    """Evaluate every numeric kill rule against current prices.

    Returns one row per rule-carrying ticker: {ticker, hit, reasons, drawdown_pct,
    price_native, price_sek, currency, rules}. Pure — no alerts, no state
    writes — so the dashboard and the watch share one implementation.
    """
    rules = get_kill_rules(store)
    if not rules:
        return []

    from fundmgr import paper

    currency_map = _load(store, "paper_currency_map")
    positions = {p.ticker: p for p in store.get_positions()}
    tickers = sorted(rules)
    native = _latest_native_prices(store, tickers)
    prices_sek = paper.sek_prices_for(store, tickers, currency_map, native)

    rows: list[dict] = []
    for ticker in tickers:
        rule = rules[ticker]
        pos = positions.get(ticker)
        price_native = native.get(ticker)
        price_sek = prices_sek.get(ticker)
        currency = rule.get("currency") or currency_map.get(ticker) or "SEK"

        drawdown = None
        if pos and pos.avg_cost_sek and price_sek:
            drawdown = (price_sek / pos.avg_cost_sek - 1) * 100

        reasons: list[str] = []
        max_dd = rule.get("max_drawdown_pct")
        if max_dd and drawdown is not None and drawdown <= -abs(max_dd):
            reasons.append(
                f"down {drawdown:+.1f}% from cost — past the -{abs(max_dd):.0f}% kill line")
        floor = rule.get("price_below")
        if floor and price_native is not None and price_native <= floor:
            reasons.append(f"price {price_native:,.2f} {currency} at/below the {floor:,.2f} floor")
        ceiling = rule.get("price_above")
        if ceiling and price_native is not None and price_native >= ceiling:
            reasons.append(f"price {price_native:,.2f} {currency} at/above the {ceiling:,.2f} target")

        # Fundamentals lines — the thresholds lifted out of the text criterion.
        # A metric with nothing cached is reported as unread, never as passing.
        from fundmgr.evidence import FUND_FIELD_META, current_metric
        fundamental_rows = []
        for f_rule in rule.get("fundamentals") or []:
            label = FUND_FIELD_META.get(f_rule["metric"], {}).get("label", f_rule["metric"])
            actual = current_metric(store, ticker, f_rule["metric"])
            unit = "%" if FUND_FIELD_META.get(f_rule["metric"], {}).get("is_fraction") else ""
            hit = actual is not None and (
                actual <= f_rule["value"] if f_rule["op"] == "below"
                else actual >= f_rule["value"])
            if hit:
                reasons.append(
                    f"{label} {actual:,.1f}{unit} is {f_rule['op']} the "
                    f"{f_rule['value']:,.1f}{unit} line")
            fundamental_rows.append({
                **f_rule, "label": label, "actual": actual, "unit": unit,
                "hit": hit, "unread": actual is None,
            })

        rows.append({
            "ticker": ticker,
            "hit": bool(reasons),
            "reasons": reasons,
            "drawdown_pct": round(drawdown, 1) if drawdown is not None else None,
            "price_native": price_native,
            "price_sek": price_sek,
            "currency": currency,
            "held": pos is not None,
            "rules": rule,
            "fundamentals": fundamental_rows,
        })
    return rows


def _recovered(row: dict) -> bool:
    """True when a previously-hit position has cleared its lines with buffer.

    The buffer keeps a position hovering on its kill line from re-alerting the
    moment it ticks back and forth across it.
    """
    rule = row["rules"]
    max_dd = rule.get("max_drawdown_pct")
    if max_dd and row["drawdown_pct"] is not None:
        if row["drawdown_pct"] <= -abs(max_dd) + _KILL_RESET_BUFFER:
            return False
    floor = rule.get("price_below")
    if floor and row["price_native"] is not None:
        if row["price_native"] <= floor * (1 + _KILL_RESET_BUFFER / 100):
            return False
    ceiling = rule.get("price_above")
    if ceiling and row["price_native"] is not None:
        if row["price_native"] >= ceiling * (1 - _KILL_RESET_BUFFER / 100):
            return False
    # Fundamentals move quarterly, not tick by tick, so no buffer is needed —
    # but a still-breached metric must keep the watch armed.
    if any(f["hit"] for f in row.get("fundamentals") or []):
        return False
    return True


def check_kill_rules(slug: str, store: Store | None = None) -> list[str]:
    """Alert when a position crosses one of its numeric kill lines.

    Transition-based like the drift watch: fires on the crossing, then re-arms
    once the position recovers past the line by a small buffer. Needs no LLM,
    so it works with no API keys configured at all.
    """
    if store is None:
        from fundmgr.paper import open_portfolio
        meta, store = open_portfolio(slug)
        name = meta["name"]
    else:
        name = store.get_meta("paper_name") or slug

    rows = evaluate_kill_rules(store)
    if not rows:
        return []

    log: list[str] = []
    fired: list[dict] = []
    for row in rows:
        ticker = row["ticker"]
        state = store.get_meta(f"paper_killrule_state:{ticker}") or "clear"
        if row["hit"] and state != "hit":
            store.set_meta(f"paper_killrule_state:{ticker}", "hit")
            store.set_meta(
                f"paper_killhit:{ticker}:{datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
                "; ".join(row["reasons"]))
            fired.append(row)
            log.append(f"🚨 {ticker}: kill rule hit — {'; '.join(row['reasons'])}")
        elif state == "hit" and _recovered(row):
            store.set_meta(f"paper_killrule_state:{ticker}", "clear")
            log.append(f"✓ {ticker}: back above its kill line — watch re-armed")

    if fired:
        from fundmgr.notify.send import send_telegram
        texts = get_kill_text(store)
        lines = [f"<b>🚨 {name} — kill criterion hit</b>"]
        for row in fired:
            held = "" if row["held"] else " (plan only — not held)"
            lines.append(f"\n<b>{row['ticker']}</b>{held}")
            for reason in row["reasons"]:
                lines.append(f"  • {reason}")
            if texts.get(row["ticker"]):
                lines.append(f"  Thesis kill: {texts[row['ticker']]}")
        lines.append("\nRun fresh analysis before acting.")
        send_telegram("\n".join(lines))
    return log


# ── Time-horizon watch ────────────────────────────────────────────────────────

def _stage_for(days: int) -> int | None:
    """The tightest alert bucket a days-remaining value falls into.

    None when the horizon is further out than the widest bucket — nothing to
    say yet, which is the normal state for most of a horizon's life.
    """
    for stage in sorted(HORIZON_STAGES):
        if days <= stage:
            return stage
    return None


def horizon_rows(store: Store, today: date | None = None) -> list[dict]:
    """Every horizon with its days remaining and urgency, soonest first."""
    today = today or datetime.now(timezone.utc).date()
    held = {p.ticker for p in store.get_positions()}
    rows = []
    for ticker, horizon in get_horizons(store).items():
        left = days_left(horizon.get("review_date"), today)
        if left is None:
            continue
        rows.append({
            "ticker": ticker,
            "review_date": horizon.get("review_date"),
            "label": horizon.get("label", ""),
            "note": horizon.get("note", ""),
            "days_left": left,
            "due": left <= 0,
            "near": 0 < left <= 30,
            "held": ticker in held,
        })
    rows.sort(key=lambda r: r["days_left"])
    return rows


def check_horizons(slug: str, store: Store | None = None,
                   today: date | None = None) -> list[str]:
    """Alert as each position's time horizon approaches, then on the day.

    Escalates through the 30/14/7/1-day buckets and fires once per bucket, so a
    horizon set months out produces four nudges total rather than a daily
    countdown. The message asks for fresh analysis — a horizon reached is a
    review trigger, not a sell signal.
    """
    if store is None:
        from fundmgr.paper import open_portfolio
        meta, store = open_portfolio(slug)
        name = meta["name"]
    else:
        name = store.get_meta("paper_name") or slug

    today = today or datetime.now(timezone.utc).date()
    rows = horizon_rows(store, today)
    if not rows:
        return []

    log: list[str] = []
    fired: list[tuple[dict, int]] = []
    for row in rows:
        ticker = row["ticker"]
        stage = _stage_for(row["days_left"])
        if stage is None:
            continue  # still further out than the first nudge
        raw = store.get_meta(f"paper_horizon_stage:{ticker}")
        try:
            already = int(raw) if raw else None
        except ValueError:
            already = None
        if already is not None and stage >= already:
            continue  # this bucket (or a tighter one) has already fired
        store.set_meta(f"paper_horizon_stage:{ticker}", str(stage))
        fired.append((row, stage))
        left = row["days_left"]
        when = "due today" if left == 0 else (f"overdue by {-left}d" if left < 0
                                              else f"in {left}d")
        log.append(f"⏳ {ticker}: horizon {when} ({row['review_date']}) — fresh analysis due")

    if fired:
        from fundmgr.notify.send import send_telegram
        texts = get_kill_text(store)
        lines = [f"<b>⏳ {name} — time horizon check</b>"]
        for row, _stage in fired:
            left = row["days_left"]
            if left > 0:
                headline = f"{left} day{'s' if left != 1 else ''} left"
            elif left == 0:
                headline = "horizon reached today"
            else:
                headline = f"{-left} day{'s' if left != -1 else ''} past horizon"
            held = "" if row["held"] else " (plan only — not held)"
            lines.append(f"\n<b>{row['ticker']}</b>{held}: {headline} "
                         f"({row['review_date']}{', ' + row['label'] if row['label'] else ''})")
            if row["note"]:
                lines.append(f"  Note: {row['note']}")
            if texts.get(row["ticker"]):
                lines.append(f"  Kill criterion: {texts[row['ticker']]}")
        lines.append("\nTime to re-run the thesis: does it still hold, or has it played out?")
        send_telegram("\n".join(lines))
    return log
