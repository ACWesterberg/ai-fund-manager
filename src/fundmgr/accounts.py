"""
Broker accounts — real books you trade yourself, which the app only watches.

A live sleeve exists to carry out a plan: it was imported from a set of picks,
and a review re-decides it. An *account* is the other way round. You decide what
to buy and sell in Montrose; the app's job is to notice what you did, write a
monitoring plan for anything new (kill and add criteria, a target price, a max
drop, a review horizon) and then tell you on Telegram when one of those lines
says it is time to act. Nothing here ever places or proposes an order to the
broker.

An account is a paper-engine book with kind="account", so every daily watch
`fund paper-track` runs on live sleeves — kill criteria, numeric kill lines,
horizons, add signals, earnings, drift — runs on it unchanged.

How the book learns what you traded: a **holdings snapshot**. Whatever produced
it (a screenshot read by the photo importer, a JSON export, a Montrose connector
run elsewhere), `sync_account` diffs it against the stored book and records the
difference as fills:

  • a name that appears → a buy at the snapshot's own cost basis;
  • more shares        → a buy of the difference, priced from the change in the
                         snapshot's total cost when it has one;
  • fewer shares / gone → a sell of the difference at the snapshot's price, else
                         the latest live price.

The snapshot is the truth and the book follows it, including cash: a difference
the fills do not explain is a deposit or withdrawal, so it moves the account's
contributed capital rather than showing up as profit.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from fundmgr import paper
from fundmgr.state.models import NavPoint, Transaction
from fundmgr.state.store import Store

KIND = "account"

# {ISIN: ticker} as each holding was first matched, so a later snapshot carrying
# the same ISIN lands on the same listing instead of reading as close + open.
ISIN_KEY = "account_isins"
NAMES_KEY = "account_names"
LAST_SYNC_KEY = "account_last_sync"

_SHARE_EPS = 1e-6

# A buy price implied by the change in total cost has to be within this factor
# of the market price to be believed. Outside it the snapshot's cost column is
# almost certainly a different figure (fees folded in differently, a misread),
# and booking it would put a wild number into the cost basis.
_IMPLIED_PRICE_BAND = (0.5, 2.0)

_TICKER_KEYS = ("yahoo_ticker", "yahoo")
_RAW_TICKER_KEYS = ("ticker", "symbol", "kortnamn", "short_name")
_SHARE_KEYS = ("shares", "quantity", "qty", "antal", "volume", "units")
_AVG_KEYS = ("avg_cost_sek", "gav_sek", "average_price_sek", "avg_price_sek")
_COST_KEYS = ("cost_value_sek", "inkopsvarde", "inköpsvärde", "acquisition_value_sek",
              "purchase_value_sek")
_VALUE_KEYS = ("market_value_sek", "marknadsvarde", "marknadsvärde", "value_sek")
_PRICE_KEYS = ("price_sek", "last_price_sek", "senast_sek")
_LIST_KEYS = ("holdings", "positions", "innehav")
_CASH_KEYS = ("cash_sek", "cash", "likvida_medel", "available_cash_sek")


def list_accounts() -> list[dict]:
    return paper.list_portfolios(kind=KIND)


# ── Snapshot parsing ──────────────────────────────────────────────────────────

def _first(row: dict, keys: tuple[str, ...]):
    for k in keys:
        if row.get(k) not in (None, ""):
            return row[k]
    return None


def _num(value) -> float | None:
    from fundmgr.vision.portfolio_photo import to_float
    return to_float(value)


def parse_snapshot(data) -> dict:
    """Normalise a holdings snapshot into {holdings, cash_sek, account_name}.

    Accepts a bare list of rows or an object with a holdings/positions list, and
    the column names a broker export, the photo importer or a hand-written JSON
    would use. Rows come back with isin / raw_ticker / name / ticker (a Yahoo
    symbol only when the row said so explicitly) and SEK money columns.
    """
    if isinstance(data, str):
        text = data.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            block = paper.extract_json_object(text) or paper._extract_json_block(text)
            if not block:
                raise ValueError("Could not read the snapshot — paste JSON with a "
                                 "holdings list.") from None
            data = json.loads(block)

    if isinstance(data, list):
        rows, cash, name = data, None, None
    elif isinstance(data, dict):
        rows = next((data[k] for k in _LIST_KEYS if isinstance(data.get(k), list)), None)
        if rows is None:
            raise ValueError("The snapshot has no holdings list "
                             f"(expected one of: {', '.join(_LIST_KEYS)}).")
        cash = _num(_first(data, _CASH_KEYS))
        name = (data.get("account_name") or data.get("name") or "").strip() or None
    else:
        raise ValueError("A snapshot must be a JSON list or object.")

    holdings: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        shares = _num(_first(row, _SHARE_KEYS))
        if shares is None or shares < 0:
            continue
        if "raw_ticker" in row:
            # The photo importer's (and this parser's own) shape: `ticker` is the
            # resolved Yahoo symbol, `raw_ticker` what the broker printed.
            raw = str(row.get("raw_ticker") or "").strip().upper()
            explicit = str(row.get("ticker") or _first(row, _TICKER_KEYS) or "").strip().upper()
        else:
            raw = str(_first(row, _RAW_TICKER_KEYS) or "").strip().upper()
            explicit = str(_first(row, _TICKER_KEYS) or "").strip().upper()
        name_ = str(row.get("name") or "").strip()
        isin = str(row.get("isin") or "").strip().upper().replace(" ", "")
        if not (explicit or raw or isin or name_):
            continue
        cost_value = _num(_first(row, _COST_KEYS))
        avg = _num(_first(row, _AVG_KEYS))
        if avg is None and cost_value and shares:
            avg = cost_value / shares
        value = _num(_first(row, _VALUE_KEYS))
        price = _num(_first(row, _PRICE_KEYS))
        if price is None and value and shares:
            price = value / shares
        holdings.append({
            "ticker": explicit or None,
            "raw_ticker": raw,
            "isin": isin,
            "name": name_ or explicit or raw,
            "currency": (str(row.get("currency") or "").strip().upper() or None),
            "shares": shares,
            "avg_cost_sek": avg,
            "cost_value_sek": cost_value if cost_value is not None
                              else (avg * shares if avg else None),
            "price_sek": price,
        })
    return {"holdings": holdings, "cash_sek": cash, "account_name": name}


def resolve_snapshot(store: Store | None, holdings: list[dict]) -> list[dict]:
    """Give every snapshot row a Yahoo ticker, preferring what the book already
    holds: an ISIN seen before maps back to the listing it was booked under, and
    a raw ticker that *is* a held symbol is taken as it stands."""
    known = json.loads(store.get_meta(ISIN_KEY) or "{}") if store else {}
    held = {p.ticker for p in store.get_positions()} if store else set()
    pending: list[dict] = []
    for h in holdings:
        if h.get("ticker"):
            continue
        if h.get("isin") and known.get(h["isin"]):
            h["ticker"] = known[h["isin"]]
        elif h.get("raw_ticker") in held:
            h["ticker"] = h["raw_ticker"]
        else:
            pending.append(h)
    if pending:
        from fundmgr.vision.portfolio_photo import _resolve_tickers
        _resolve_tickers(pending)
    return holdings


def _remember_isins(store: Store, holdings: list[dict]) -> None:
    """Remember each holding's ISIN and the company name the broker printed —
    a position row carries only a ticker, and a plan written for "SAAB-B.ST"
    with no name is a plan written blind."""
    known = json.loads(store.get_meta(ISIN_KEY) or "{}")
    names = json.loads(store.get_meta(NAMES_KEY) or "{}")
    for h in holdings:
        if h.get("isin") and h.get("ticker"):
            known[h["isin"]] = h["ticker"]
        if h.get("ticker") and h.get("name") and h["name"] != h["ticker"]:
            names[h["ticker"]] = h["name"]
    store.set_meta(ISIN_KEY, json.dumps(known))
    store.set_meta(NAMES_KEY, json.dumps(names))


def names(store: Store) -> dict[str, str]:
    return json.loads(store.get_meta(NAMES_KEY) or "{}")


# ── Creating an account ───────────────────────────────────────────────────────

def create_account(name: str, snapshot, *, benchmark: str = paper.DEFAULT_BENCHMARK,
                   source: str = "holdings snapshot") -> tuple[str, list[str]]:
    """Create an account book from a snapshot, seeded at the broker's cost basis.

    Rows that cannot be resolved to a ticker, or carry no cost basis, are skipped
    and named in the log — a position booked at a guessed cost would report a
    P&L the broker does not.
    """
    parsed = parse_snapshot(snapshot)
    holdings = resolve_snapshot(None, parsed["holdings"])
    log: list[str] = []
    rows = []
    for h in holdings:
        if not h.get("ticker"):
            log.append(f"⚠ {h['name'] or h['raw_ticker']}: no ticker match — skipped")
            continue
        if not h.get("shares"):
            continue
        cost = h.get("avg_cost_sek") or h.get("price_sek")
        if not cost:
            log.append(f"⚠ {h['ticker']}: no cost basis or price — skipped")
            continue
        rows.append({"ticker": h["ticker"], "name": h["name"], "shares": h["shares"],
                     "avg_cost_sek": cost, "isin": h.get("isin")})
    if not rows:
        raise ValueError("Nothing in the snapshot could be booked — every row was "
                         "unresolved or missing its share count and cost.")
    cash = parsed.get("cash_sek") or 0.0
    capital = sum(r["shares"] * r["avg_cost_sek"] for r in rows) + cash
    slug, create_log = paper.create_portfolio(
        name=name or parsed.get("account_name") or "Broker account",
        capital_sek=capital,
        holdings_text=json.dumps(rows, indent=2, ensure_ascii=False),
        model_label=source,
        benchmark=benchmark or paper.DEFAULT_BENCHMARK,
        holdings_override=rows,
        kind=KIND,
        seed_holdings=True,
    )
    _meta, store = paper.open_portfolio(slug)
    _remember_isins(store, rows)
    store.set_meta(LAST_SYNC_KEY, datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return slug, log + create_log


# ── Syncing ───────────────────────────────────────────────────────────────────

def _market_prices_sek(store: Store, tickers: list[str], currency_map: dict) -> dict[str, float]:
    """Live SEK prices, falling back to the latest cached close."""
    if not tickers:
        return {}
    try:
        from fundmgr.data.quotes import live_prices
        native = {t: p for t, p in live_prices(tickers).items() if p}
    except Exception:
        native = {}
    for t in tickers:
        if t not in native:
            rows = store.get_prices(t)
            if rows:
                native[t] = rows[-1]["close"]
    return paper.sek_prices_for(store, tickers, currency_map, native)


def _buy_price(old_shares: float, old_avg: float, row: dict, delta: float,
               market: float | None) -> tuple[float | None, str]:
    """SEK price for `delta` newly bought shares, and where it came from."""
    if row.get("avg_cost_sek") and row.get("shares"):
        implied = (row["shares"] * row["avg_cost_sek"] - old_shares * old_avg) / delta
        if implied > 0 and (market is None or
                            _IMPLIED_PRICE_BAND[0] <= implied / market <= _IMPLIED_PRICE_BAND[1]):
            return implied, "broker cost"
    if row.get("price_sek"):
        return row["price_sek"], "snapshot price"
    if market:
        return market, "live price"
    return None, ""


def sync_account(slug: str, snapshot, *, notify: bool = True,
                 write_plans: bool = True, dry_run: bool = False) -> dict:
    """Bring an account book in line with a holdings snapshot.

    Returns {changes, cash, warnings, plans, message}. `changes` lists one entry
    per ticker whose share count moved: kind (opened / added / trimmed / closed),
    shares, price_sek and where that price came from. With `dry_run` the diff is
    computed and nothing is written or sent.
    """
    meta, store = paper.open_portfolio(slug)
    if meta.get("kind") != KIND:
        raise ValueError(f"'{slug}' is a {meta.get('kind')} book, not a broker account.")
    parsed = parse_snapshot(snapshot)
    holdings = resolve_snapshot(store, parsed["holdings"])

    warnings: list[str] = []
    wanted: dict[str, dict] = {}
    for h in holdings:
        t = h.get("ticker")
        if not t:
            warnings.append(f"{h['name'] or h['raw_ticker']}: no ticker match — ignored "
                            "(fix it in the snapshot or record it by hand)")
            continue
        if t in wanted:     # the same listing twice (e.g. two lots) — merge
            prev = wanted[t]
            total = prev["shares"] + h["shares"]
            cost = (prev.get("cost_value_sek") or 0) + (h.get("cost_value_sek") or 0)
            prev.update(shares=total, cost_value_sek=cost or None,
                        avg_cost_sek=(cost / total) if cost and total else prev.get("avg_cost_sek"))
            continue
        wanted[t] = dict(h)

    positions = {p.ticker: p for p in store.get_positions()}
    currency_map = dict(meta["currency_map"])
    for t in wanted:
        if t not in currency_map:
            currency_map[t] = wanted[t].get("currency") or paper.detect_currency(t)
    market = _market_prices_sek(store, sorted(set(wanted) | set(positions)), currency_map)

    changes: list[dict] = []
    for t, pos in positions.items():
        row = wanted.get(t)
        new_shares = row["shares"] if row else 0.0
        delta = pos.shares - new_shares
        if delta <= _SHARE_EPS:
            continue
        price, basis = None, ""
        if row and row.get("price_sek"):
            price, basis = row["price_sek"], "snapshot price"
        elif market.get(t):
            price, basis = market[t], "live price"
        else:
            price, basis = pos.avg_cost_sek, "cost (no price available)"
            warnings.append(f"{t}: no price for the sale — booked at cost, so its "
                            "realised P&L reads zero")
        closed = new_shares <= _SHARE_EPS
        changes.append({
            "ticker": t, "kind": "closed" if closed else "trimmed",
            # Never round past what is held, or the ledger refuses the oversell.
            "side": "sell", "shares": pos.shares if closed else min(round(delta, 6), pos.shares),
            "price_sek": price,
            "basis": basis, "avg_cost_sek": pos.avg_cost_sek,
            "remaining": max(new_shares, 0.0),
        })
    for t, row in wanted.items():
        pos = positions.get(t)
        old = pos.shares if pos else 0.0
        delta = row["shares"] - old
        if delta <= _SHARE_EPS:
            continue
        price, basis = _buy_price(old, pos.avg_cost_sek if pos else 0.0, row, delta,
                                  market.get(t))
        if not price:
            warnings.append(f"{t}: {delta:g} new shares but no cost or price to book "
                            "them at — skipped")
            continue
        changes.append({
            "ticker": t, "kind": "opened" if pos is None else "added", "side": "buy",
            "shares": round(delta, 6), "price_sek": price, "basis": basis,
            "remaining": row["shares"], "name": row.get("name"),
        })

    result = {"changes": changes, "warnings": warnings, "plans": [],
              "cash": None, "message": ""}
    if dry_run:
        result["message"] = format_sync_message(meta["name"], changes, None, warnings, [])
        return result

    now = datetime.now(timezone.utc)
    capital = meta["capital_sek"]
    # Sells first, so their proceeds are in the book before anything is bought.
    for ch in sorted(changes, key=lambda c: c["side"] != "sell"):
        if ch["side"] == "buy":
            need = ch["shares"] * ch["price_sek"]
            short = need - store.get_cash()
            if short > 0:
                # The broker had the money; the mirror just hadn't been told. A
                # deposit is contributed capital, not return.
                store.adjust_cash(round(short, 2) + 0.01)
                capital += round(short, 2) + 0.01
        store.apply_fill(Transaction(
            ticker=ch["ticker"], side=ch["side"], shares=ch["shares"],
            price_sek=round(ch["price_sek"], 4), fee_sek=0.0, source="sync",
            currency=currency_map.get(ch["ticker"], "SEK"), timestamp=now,
        ))

    # Match the broker's own cost basis where it gave one, whatever price the
    # delta was booked at — P&L on the dashboard should read what Montrose reads.
    held_now = {p.ticker: p for p in store.get_positions()}
    for t, row in wanted.items():
        p = held_now.get(t)
        avg = row.get("avg_cost_sek")
        if p and avg and abs(p.avg_cost_sek - avg) / avg > 1e-4:
            # Refunds (or charges) the difference, which is right: the mirror
            # paid its guess, the broker paid this.
            paper.set_cost_basis(store, t, avg)

    cash_note = None
    if parsed.get("cash_sek") is not None:
        diff = parsed["cash_sek"] - store.get_cash()
        if abs(diff) >= 1:
            store.adjust_cash(diff)
            capital += diff
            cash_note = diff
    store.set_meta("paper_capital_sek", str(round(capital, 2)))
    result["cash"] = {"balance_sek": store.get_cash(), "unexplained_sek": cash_note}

    store.set_meta("paper_currency_map", json.dumps(currency_map))
    _remember_isins(store, list(wanted.values()))
    _update_targets(store, changes, market)
    for ch in changes:
        if ch["kind"] == "closed":
            _retire_plan(store, ch["ticker"])

    if changes:
        try:
            paper._cache_price_history(store, [c["ticker"] for c in changes
                                               if c["side"] == "buy"])
        except Exception as e:
            warnings.append(f"price history refresh failed: {e}")
        _snapshot_nav(store, market)
    store.set_meta(LAST_SYNC_KEY, now.isoformat(timespec="seconds"))

    opened = [c["ticker"] for c in changes if c["kind"] == "opened"]
    if write_plans and opened:
        from fundmgr.engine import account_plan
        try:
            result["plans"] = account_plan.write_plans(slug, opened, notify=False)
        except Exception as e:
            warnings.append(f"monitoring plan not written: {e}")

    result["message"] = format_sync_message(meta["name"], changes, cash_note, warnings,
                                            result["plans"], store=store)
    if notify and (changes or warnings):
        from fundmgr.notify.send import send_telegram
        send_telegram(result["message"])
    return result


def _update_targets(store: Store, changes: list[dict], market: dict[str, float]) -> None:
    """A name you open joins the drift watch at the weight you bought it at; a
    name you close leaves it. Adds and trims move the target with you — the
    sizing is your decision, and the watch is there to say when the market has
    changed it, not to argue with the trade you just made."""
    targets = json.loads(store.get_meta("paper_target_weights") or "{}")
    positions = {p.ticker: p for p in store.get_positions()}
    nav = sum(p.shares * market.get(p.ticker, p.avg_cost_sek) for p in positions.values()) \
        + store.get_cash()
    for ch in changes:
        t = ch["ticker"]
        if ch["kind"] == "closed":
            targets.pop(t, None)
            continue
        p = positions.get(t)
        if p and nav > 0:
            targets[t] = round(p.shares * market.get(t, p.avg_cost_sek) / nav * 100, 4)
            store.set_meta(f"paper_drift_state:{t}", "under")
    store.set_meta("paper_target_weights", json.dumps(targets))


def _retire_plan(store: Store, ticker: str) -> None:
    """A position you closed has nothing left to watch. Its kill rules would
    otherwise keep alerting as "plan only — not held" on a name you are out of."""
    from fundmgr import addsignal, watchplan
    watchplan.clear_position_plan(store, ticker)
    watchplan.set_add_text(store, ticker, "")
    addsignal.clear_plan(store, ticker)


def _snapshot_nav(store: Store, market: dict[str, float]) -> None:
    positions = store.get_positions()
    nav = sum(p.shares * market.get(p.ticker, p.avg_cost_sek) for p in positions) \
        + store.get_cash()
    bench = store.get_benchmark()
    store.upsert_nav(NavPoint(
        date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        portfolio_nav_sek=round(nav, 2),
        benchmark_value=bench[-1]["close"] if bench else 0.0,
        cash_sek=store.get_cash(),
    ))


# ── Telegram ──────────────────────────────────────────────────────────────────

_KIND_MARK = {"opened": "🟢 Opened", "added": "➕ Added to", "trimmed": "➖ Trimmed",
              "closed": "⚪ Closed"}


def _esc(text) -> str:
    import html
    return html.escape(str(text or ""), quote=False)


def format_sync_message(name: str, changes: list[dict], cash_note: float | None,
                        warnings: list[str], plans: list[dict],
                        store: Store | None = None) -> str:
    lines = [f"<b>🏦 {_esc(name)} — changes picked up</b>"]
    if not changes:
        lines.append("No trades since the last sync.")
    plan_by = {p["ticker"]: p for p in plans}
    for ch in changes:
        t = ch["ticker"]
        verb = _KIND_MARK[ch["kind"]]
        line = f"\n{verb} <b>{_esc(t)}</b>: {ch['shares']:g} @ {ch['price_sek']:,.2f} SEK"
        if ch["side"] == "sell" and ch.get("avg_cost_sek"):
            realised = (ch["price_sek"] / ch["avg_cost_sek"] - 1) * 100
            line += f" ({realised:+.1f}% on cost)"
        if ch["basis"] and ch["basis"] != "broker cost":
            line += f" <i>· {_esc(ch['basis'])}</i>"
        lines.append(line)
        if t in plan_by:
            lines += ["  " + s for s in plan_lines(plan_by[t])]
        elif ch["kind"] in ("added", "trimmed") and store is not None:
            lines += ["  " + s for s in standing_plan_lines(store, t)]
    if cash_note:
        word = "deposit" if cash_note > 0 else "withdrawal"
        lines.append(f"\n💰 Cash moved {cash_note:+,.0f} SEK beyond the trades — booked as a {word}.")
    for w in warnings:
        lines.append(f"⚠ {_esc(w)}")
    return "\n".join(lines)


_STANCE = {"hold": "🟢 hold", "add": "🔵 add", "trim": "🟡 trim", "exit": "🔴 exit"}


def plan_lines(plan: dict) -> list[str]:
    """A written plan, as Telegram lines."""
    out = []
    if plan.get("stance"):
        out.append(f"{_STANCE.get(plan['stance'], plan['stance'])} — "
                   f"{_esc(plan.get('stance_reason'))}")
    if plan.get("kill_criterion"):
        out.append(f"Kill: {_esc(plan['kill_criterion'])}")
    if plan.get("add_criterion"):
        out.append(f"Add if: {_esc(plan['add_criterion'])}")
    bits = []
    if plan.get("target_price"):
        bits.append(f"🎯 target {plan['target_price']:,.2f} {_esc(plan.get('currency') or '')}"
                    .rstrip())
    if plan.get("max_drawdown_pct"):
        bits.append(f"max drop −{plan['max_drawdown_pct']:g}%")
    if plan.get("horizon_months"):
        bits.append(f"review in {plan['horizon_months']} mo")
    if bits:
        out.append(" · ".join(bits))
    return out


def standing_plan_lines(store: Store, ticker: str) -> list[str]:
    """What the existing plan says about a name you just added to or trimmed."""
    from fundmgr import addsignal, watchplan
    rule = watchplan.get_kill_rules(store).get(ticker) or {}
    target = addsignal.get_plan(store, ticker).get("target_price") or rule.get("price_above")
    out = []
    if target:
        rows = store.get_prices(ticker)
        if rows and rows[-1]["close"]:
            out.append(f"🎯 target {target:,.2f} — {(target / rows[-1]['close'] - 1) * 100:+.1f}% "
                       "from the last close")
        else:
            out.append(f"🎯 target {target:,.2f}")
    if not watchplan.get_kill_text(store).get(ticker) and not target:
        out.append("No monitoring plan yet — run a plan to have it watched.")
    return out
