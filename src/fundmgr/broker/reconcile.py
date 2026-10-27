"""
Montrose phase 1 — refresh and reconcile, read-only.

`refresh` reads the fund's real account (holdings, cash, transactions) and
saves a private snapshot. `reconcile` compares that snapshot with the fund's
ledger and says where they disagree. Neither writes to the fund database:
the ledger is still booked by hand (/fill, OCR), and this is the check that
the hand-booking matches what Montrose actually did. Writing fills from the
broker is a later phase, once this report has been seen to agree for a while.

Montrose names instruments by its own ticker ("TRUE", "KCC") and orderbookId,
never by ISIN, and transactions carry only the orderbookId. So each orderbookId
is resolved once to a ticker/name/currency (from holdings, else the read-only
`search_instruments`), cached in data/montrose/instruments.json, and mapped to
a Yahoo symbol by matching ticker and quote currency against the ledger's own
holdings and the fund's universe. Anything ambiguous stays unresolved and is
reported, never guessed — `fund montrose-map` pins it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fundmgr.broker import montrose
from fundmgr.config import _SUFFIX_CURRENCY, UniverseTicker

TRADE_CATEGORIES = ("Buy", "Sell")
FLOW_CATEGORIES = ("Deposit", "Withdrawal")
MAX_PAGES = 40
DATE_SLACK_DAYS = 3
AMOUNT_TOLERANCE_SEK = 1.0
SHARES_EPS = 1e-6

_CURRENCY_SUFFIX = {cur: f".{sfx}" for sfx, cur in _SUFFIX_CURRENCY.items()
                    if sfx in ("ST", "OL", "CO", "HE")}


def _unwrap(obj: Any) -> Any:
    """Montrose wraps some list replies as {"result": [...]}."""
    if isinstance(obj, dict) and set(obj) == {"result"}:
        return obj["result"]
    return obj


def _num(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# ── Instrument resolution ────────────────────────────────────────────────────

def _instruments_path() -> Path:
    return montrose.state_dir() / "instruments.json"


def load_instruments() -> dict[str, dict]:
    return montrose._read(_instruments_path()) or {}


def save_instruments(instruments: dict[str, dict]) -> None:
    montrose._write_private(_instruments_path(), instruments)


def pin_instrument(orderbook_id: int, yahoo: str) -> dict:
    """Pin an orderbookId to a Yahoo symbol (`fund montrose-map`)."""
    instruments = load_instruments()
    entry = instruments.setdefault(str(int(orderbook_id)), {})
    entry["yahoo"] = yahoo.strip().upper()
    save_instruments(instruments)
    return entry


def _share_class(name: str) -> str | None:
    m = re.search(r"\b(?:ser\.?\s*)?([A-D])$", (name or "").strip(), re.IGNORECASE)
    return m.group(1).upper() if m else None


def resolve_yahoo(ticker: str, name: str, currency: str, known: dict[str, str]) -> str | None:
    """Montrose ticker + quote currency → one Yahoo symbol from `known`, or None.

    `known` maps Yahoo symbol → quote currency for everything the fund could
    hold (its ledger and its universe). "TRUE" in SEK finds TRUE-B.ST; if a
    base has several share classes, the class letter in the name decides, and
    if it still can't, nothing is returned rather than a guess.
    """
    base = (ticker or "").strip().upper().replace(" ", "-")
    cur = (currency or "").strip().upper()
    if not base:
        return None
    same_ccy = {y: c for y, c in known.items() if not cur or c == cur}

    def stem(y: str) -> str:
        return y.rsplit(".", 1)[0] if "." in y else y

    exact = [y for y in same_ccy if stem(y) == base]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        suffix = _CURRENCY_SUFFIX.get(cur)
        hit = [y for y in exact if suffix and y.endswith(suffix)]
        return hit[0] if len(hit) == 1 else None
    classes = [y for y in same_ccy if stem(y).startswith(base + "-")]
    if len(classes) == 1:
        return classes[0]
    cls = _share_class(name)
    if cls:
        hit = [y for y in classes if stem(y) == f"{base}-{cls}"]
        if len(hit) == 1:
            return hit[0]
    return None


def known_symbols(ledger_tickers: list[str], universe: list[UniverseTicker]) -> dict[str, str]:
    known = {u.yahoo_ticker.upper(): u.currency for u in universe}
    for t in ledger_tickers:
        t = t.upper()
        if t not in known:
            sfx = t.rsplit(".", 1)[-1] if "." in t else ""
            known[t] = _SUFFIX_CURRENCY.get(sfx, "USD" if not sfx else "")
    return known


# ── Refresh ──────────────────────────────────────────────────────────────────

def _find_account(accounts: list[dict], account_name: str) -> dict:
    want = account_name.strip().casefold()
    hits = [a for a in accounts if str(a.get("accountName", "")).strip().casefold() == want]
    if len(hits) != 1:
        names = ", ".join(repr(str(a.get("accountName", "")).strip()) for a in accounts)
        raise montrose.MontroseError(
            f"{'No' if not hits else 'More than one'} Montrose account named {account_name!r} "
            f"(accounts: {names})")
    return hits[0]


def _fetch_transactions(s: montrose.Session, account_id: str, since: date) -> list[dict]:
    rows: list[dict] = []
    cursor: str | None = None
    for _ in range(MAX_PAGES):
        args: dict = {"accountIds": [account_id], "from": since.isoformat(), "pageSize": 50}
        if cursor:
            args["cursor"] = cursor
        page = s.call_tool("get_transactions", args)
        rows += [t for t in page.get("transactions", []) if t.get("accountId") == account_id]
        cursor = (page.get("pagination") or {}).get("nextCursor")
        if not cursor:
            return rows
    raise montrose.MontroseError(f"More than {MAX_PAGES} pages of transactions since {since}; "
                                 "use a later --since")


def refresh(account_name: str, since: date) -> tuple[Path, dict]:
    """Read the account from Montrose and save a private snapshot. Read-only."""
    instruments = load_instruments()
    with montrose.Session() as s:
        account = _find_account(_unwrap(s.call_tool("get_user_accounts")), account_name)
        account_id = account["accountId"]
        holdings = [h for h in _unwrap(s.call_tool("get_holdings", {"accountId": account_id}))
                    if h.get("accountId") == account_id]
        if len(holdings) != 1:
            raise montrose.MontroseError(f"get_holdings returned {len(holdings)} entries for the account")
        h = holdings[0]
        if h.get("error"):
            raise montrose.MontroseError(f"get_holdings: {h['error']}")
        txns = _fetch_transactions(s, account_id, since)

        for p in h.get("positions") or []:
            entry = instruments.setdefault(str(p["orderbookId"]), {})
            entry.update(ticker=p.get("ticker"), name=p.get("instrumentName"),
                         currency=p.get("instrumentCurrency"))
        for t in txns:
            obid = t.get("orderbookId")
            if not obid:
                continue
            entry = instruments.setdefault(str(obid), {})
            if t.get("priceCurrency") and not entry.get("currency"):
                entry["currency"] = t["priceCurrency"]
            if not entry.get("ticker"):
                found = _unwrap(s.call_tool("search_instruments", {"orderbookId": int(obid)}))
                match = [i for i in found or [] if i.get("orderbookId") == obid]
                if match:
                    entry.update(ticker=match[0].get("ticker"), name=match[0].get("name"))
    save_instruments(instruments)

    snapshot = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "since": since.isoformat(),
        "account": {k: account.get(k) for k in ("accountId", "accountName", "accountType")},
        "currency": h.get("currency"),
        "summary": h.get("summary") or {},
        "currency_positions": h.get("currencyPositions") or [],
        "positions": h.get("positions") or [],
        "transactions": txns,
    }
    path = montrose.state_dir() / "snapshot-latest.json"
    montrose._write_private(path, snapshot)
    return path, snapshot


def load_snapshot(path: Path | None = None) -> dict:
    snap = montrose._read(path or montrose.state_dir() / "snapshot-latest.json")
    if not snap:
        raise montrose.MontroseError("No Montrose snapshot yet — run fund montrose-refresh.")
    return snap


# ── Reconcile ────────────────────────────────────────────────────────────────

@dataclass
class LedgerTrade:
    date: date
    ticker: str
    side: str
    shares: float
    cash_effect: float          # SEK; buy negative, sell positive, fee included


@dataclass
class Report:
    account: str
    fetched_at: str
    since: str
    broker_cash: float
    ledger_cash: float
    broker_total: float
    cash_tolerance: float
    foreign_cash: list[dict] = field(default_factory=list)
    positions: list[dict] = field(default_factory=list)       # every line, matched or not
    unresolved: list[dict] = field(default_factory=list)
    missing_in_ledger: list[dict] = field(default_factory=list)
    not_at_broker: list[dict] = field(default_factory=list)
    amount_diffs: list[dict] = field(default_factory=list)
    date_diffs: list[dict] = field(default_factory=list)
    flows_missing: list[dict] = field(default_factory=list)
    flows_extra: list[dict] = field(default_factory=list)
    cash_events: list[dict] = field(default_factory=list)
    explained_cash: float = 0.0

    @property
    def cash_diff(self) -> float:
        return self.broker_cash - self.ledger_cash

    @property
    def position_mismatches(self) -> list[dict]:
        return [p for p in self.positions if p["status"] != "ok"]

    @property
    def in_sync(self) -> bool:
        return (not self.position_mismatches and not self.unresolved
                and not self.foreign_cash and abs(self.cash_diff) <= self.cash_tolerance)


def reconcile(snapshot: dict, *, ledger_cash: float, ledger_positions: dict[str, float],
              ledger_costs: dict[str, float], ledger_trades: list[LedgerTrade],
              ledger_flows: list[dict], universe: list[UniverseTicker],
              instruments: dict[str, dict], cash_tolerance: float = 25.0,
              cost_tolerance_pct: float = 1.0) -> Report:
    """Compare a broker snapshot with the ledger. Pure: no I/O, no writes."""
    since = date.fromisoformat(snapshot["since"])
    known = known_symbols(list(ledger_positions), universe)
    account_ccy = (snapshot.get("currency") or "SEK").upper()

    def yahoo_for(obid: Any, ticker: str | None = None, name: str | None = None,
                  currency: str | None = None) -> str | None:
        entry = instruments.get(str(obid), {})
        if entry.get("yahoo"):
            return entry["yahoo"]
        return resolve_yahoo(ticker or entry.get("ticker") or "", name or entry.get("name") or "",
                             currency or entry.get("currency") or "", known)

    cash = 0.0
    foreign: list[dict] = []
    for cp in snapshot.get("currency_positions") or []:
        code = str(cp.get("currencyCode", "")).upper()
        if code == account_ccy:
            cash += _num(cp.get("balance"))
        elif abs(_num(cp.get("balance"))) > 0.005:
            foreign.append({"currency": code, "balance": _num(cp.get("balance"))})
    if not snapshot.get("currency_positions"):
        cash = _num((snapshot.get("summary") or {}).get("availableForPurchase"))

    report = Report(
        account=str((snapshot.get("account") or {}).get("accountName", "")).strip(),
        fetched_at=snapshot.get("fetched_at", ""), since=snapshot["since"],
        broker_cash=cash, ledger_cash=ledger_cash,
        broker_total=_num((snapshot.get("summary") or {}).get("totalValue")),
        cash_tolerance=cash_tolerance, foreign_cash=foreign,
    )

    broker_pos: dict[str, dict] = {}
    for p in snapshot.get("positions") or []:
        y = yahoo_for(p.get("orderbookId"), p.get("ticker"), p.get("instrumentName"),
                      p.get("instrumentCurrency"))
        qty = _num(p.get("quantity"))
        mv = _num((p.get("marketValue") or {}).get("accountCurrency"))
        pnl = _num((p.get("unrealizedResult") or {}).get("accountCurrency"))
        line = {"orderbookId": p.get("orderbookId"), "montrose": p.get("ticker"),
                "name": p.get("instrumentName"), "shares": qty, "value": mv,
                "avg_cost": (mv - pnl) / qty if qty else None}
        if y is None:
            report.unresolved.append(line)
        elif y in broker_pos:
            broker_pos[y]["shares"] += qty
        else:
            broker_pos[y] = line

    for y in sorted(set(broker_pos) | {t for t, s in ledger_positions.items() if s > SHARES_EPS}):
        b = broker_pos.get(y)
        bs = b["shares"] if b else 0.0
        ls = ledger_positions.get(y, 0.0)
        status = ("ok" if abs(bs - ls) <= SHARES_EPS
                  else "missing_in_ledger" if ls <= SHARES_EPS
                  else "not_at_broker" if bs <= SHARES_EPS else "shares_differ")
        line = {"ticker": y, "broker_shares": bs, "ledger_shares": ls, "status": status,
                "montrose": b["montrose"] if b else None, "value": b["value"] if b else None}
        lc, bc = ledger_costs.get(y), b and b["avg_cost"]
        if status == "ok" and lc and bc and abs(bc - lc) / lc * 100 > cost_tolerance_pct:
            line["cost_diff"] = {"broker": bc, "ledger": lc}
        report.positions.append(line)

    # Trades: exact date first, then within a few days (fills booked on the wrong day).
    explained = 0.0
    open_ledger = [t for t in ledger_trades if t.date >= since]
    broker_trades = []
    for t in snapshot.get("transactions") or []:
        if t.get("isCanceled"):
            continue
        cat = t.get("category")
        amount = _num(t.get("amount"))
        settles_in_account_ccy = str(t.get("currency", account_ccy)).upper() == account_ccy
        if cat in TRADE_CATEGORIES:
            broker_trades.append(t)
        elif cat in FLOW_CATEGORIES:
            continue
        elif settles_in_account_ccy and abs(amount) > 0.005:
            report.cash_events.append({"date": t.get("tradeDate"), "category": cat,
                                       "amount": amount, "text": t.get("text") or ""})
            explained += amount

    for exact in (True, False):
        for t in list(broker_trades):
            side = "buy" if t["category"] == "Buy" else "sell"
            qty = abs(_num(t.get("quantity")))
            d = date.fromisoformat(t["tradeDate"])
            y = yahoo_for(t.get("orderbookId"), currency=t.get("priceCurrency"))
            match = next((lt for lt in open_ledger
                          if lt.ticker == y and lt.side == side and abs(lt.shares - qty) <= SHARES_EPS
                          and (lt.date == d if exact else abs((lt.date - d).days) <= DATE_SLACK_DAYS)),
                         None)
            if match is None:
                continue
            broker_trades.remove(t)
            open_ledger.remove(match)
            if not exact:
                report.date_diffs.append({"ticker": y, "side": side, "shares": qty,
                                          "broker_date": d.isoformat(),
                                          "ledger_date": match.date.isoformat()})
            amount = _num(t.get("amount"))
            if str(t.get("currency", account_ccy)).upper() == account_ccy:
                delta = amount - match.cash_effect
                explained += delta
                if abs(delta) > AMOUNT_TOLERANCE_SEK:
                    report.amount_diffs.append({"date": d.isoformat(), "ticker": y, "side": side,
                                                "shares": qty, "broker": amount,
                                                "ledger": match.cash_effect, "diff": delta})

    for t in broker_trades:
        amount = _num(t.get("amount"))
        entry = instruments.get(str(t.get("orderbookId")), {})
        report.missing_in_ledger.append({
            "date": t.get("tradeDate"), "side": "buy" if t["category"] == "Buy" else "sell",
            "ticker": yahoo_for(t.get("orderbookId"), currency=t.get("priceCurrency")),
            "montrose": entry.get("ticker"), "orderbookId": t.get("orderbookId"),
            "shares": abs(_num(t.get("quantity"))), "price": _num(t.get("price")),
            "price_currency": t.get("priceCurrency"), "amount": amount})
        if str(t.get("currency", account_ccy)).upper() == account_ccy:
            explained += amount
    for lt in open_ledger:
        report.not_at_broker.append({"date": lt.date.isoformat(), "side": lt.side,
                                     "ticker": lt.ticker, "shares": lt.shares,
                                     "amount": lt.cash_effect})
        explained -= lt.cash_effect

    flows = [{"date": f["date"], "amount": _num(f["amount_sek"])}
             for f in ledger_flows if date.fromisoformat(f["date"][:10]) >= since]
    for t in snapshot.get("transactions") or []:
        if t.get("isCanceled") or t.get("category") not in FLOW_CATEGORIES:
            continue
        amount = _num(t.get("amount"))
        d = date.fromisoformat(t["tradeDate"])
        match = next((f for f in flows if abs(f["amount"] - amount) <= AMOUNT_TOLERANCE_SEK
                      and abs((date.fromisoformat(f["date"][:10]) - d).days) <= DATE_SLACK_DAYS), None)
        if match:
            flows.remove(match)
            explained += amount - match["amount"]
        else:
            report.flows_missing.append({"date": d.isoformat(), "category": t["category"],
                                         "amount": amount})
            explained += amount
    for f in flows:
        report.flows_extra.append(f)
        explained -= f["amount"]

    report.explained_cash = explained
    return report


def ledger_inputs(store: Any, since: date) -> dict:
    """Read what `reconcile` needs from the fund's Store. Reads only."""
    positions = store.get_positions()
    trades = []
    for t in store.get_transactions(limit=100_000):
        if t.side not in ("buy", "sell") or t.timestamp.date() < since:
            continue
        gross = t.shares * t.price_sek
        effect = -(gross + t.fee_sek) if t.side == "buy" else gross - t.fee_sek
        trades.append(LedgerTrade(t.timestamp.date(), t.ticker.upper(), t.side, t.shares, effect))
    return {
        "ledger_cash": store.get_cash(),
        "ledger_positions": {p.ticker.upper(): p.shares for p in positions},
        "ledger_costs": {p.ticker.upper(): p.avg_cost_sek for p in positions},
        "ledger_trades": trades,
        "ledger_flows": store.get_cash_flows(),
    }


def default_since(lookback_days: int, today: date | None = None) -> date:
    return (today or date.today()) - timedelta(days=lookback_days)


# ── Formatting ───────────────────────────────────────────────────────────────

def _sek(v: float) -> str:
    return f"{v:,.2f}".replace(",", " ")


def _sh(v: float) -> str:
    return f"{v:g}"


def format_report(r: Report) -> str:
    lines = [
        f"{'✅' if r.in_sync else '⚠️'} Montrose ↔ ledger — {r.account} "
        f"({r.fetched_at[:16].replace('T', ' ')} UTC)",
        f"Cash: broker {_sek(r.broker_cash)} · ledger {_sek(r.ledger_cash)} · "
        f"diff {r.cash_diff:+,.2f} SEK".replace(",", " "),
    ]
    residual = r.cash_diff - r.explained_cash
    if abs(r.cash_diff) > 0.005:
        # /setcash corrections leave no ledger row, so a dividend fixed that way
        # shows up here as an item below plus an equal and opposite remainder.
        lines.append(f"  items below net to {r.explained_cash:+.2f}; the other {residual:+.2f} "
                     f"is from before {r.since} or a /setcash correction")
    for fc in r.foreign_cash:
        lines.append(f"  ⚠️ foreign cash not in the ledger: {fc['balance']:,.2f} {fc['currency']}")
    ok = [p for p in r.positions if p["status"] == "ok"]
    lines.append(f"Positions: {len(ok)}/{len(r.positions)} match"
                 + (f" · broker total value {_sek(r.broker_total)} SEK" if r.broker_total else ""))
    labels = {"missing_in_ledger": "held at Montrose, not in ledger",
              "not_at_broker": "in ledger, not held at Montrose",
              "shares_differ": "shares differ"}
    for p in r.position_mismatches:
        lines.append(f"  ❗ {p['ticker']}: broker {_sh(p['broker_shares'])} · ledger "
                     f"{_sh(p['ledger_shares'])} — {labels[p['status']]}")
    for u in r.unresolved:
        lines.append(f"  ❓ {u['montrose']} ({u['name']}, orderbook {u['orderbookId']}) "
                     f"{_sh(u['shares'])} sh — no ledger/universe symbol; "
                     f"pin with fund montrose-map {u['orderbookId']} SYMBOL")
    for p in r.positions:
        if "cost_diff" in p:
            c = p["cost_diff"]
            lines.append(f"  · {p['ticker']} avg cost: broker {c['broker']:.2f} · ledger "
                         f"{c['ledger']:.2f} (/psetcost-style fix is manual)")
    if r.missing_in_ledger:
        lines.append(f"Broker trades not in the ledger (since {r.since}):")
        for t in r.missing_in_ledger:
            lines.append(f"  ❗ {t['date']} {t['side']} {_sh(t['shares'])} "
                         f"{t['ticker'] or t['montrose'] or t['orderbookId']} @ {t['price']:g} "
                         f"{t['price_currency']} = {t['amount']:+.2f}")
    if r.not_at_broker:
        lines.append(f"Ledger trades Montrose doesn't have (since {r.since}):")
        for t in r.not_at_broker:
            lines.append(f"  ❗ {t['date']} {t['side']} {_sh(t['shares'])} {t['ticker']} "
                         f"= {t['amount']:+.2f}")
    if r.date_diffs:
        lines.append("Booked on a different day:")
        for t in r.date_diffs:
            lines.append(f"  · {t['ticker']} {t['side']} {_sh(t['shares'])}: broker "
                         f"{t['broker_date']} · ledger {t['ledger_date']}")
    if r.amount_diffs:
        lines.append(f"Trade amounts that differ by more than {AMOUNT_TOLERANCE_SEK:g} SEK:")
        for t in r.amount_diffs:
            lines.append(f"  · {t['date']} {t['ticker']} {t['side']} {_sh(t['shares'])}: broker "
                         f"{t['broker']:+.2f} · ledger {t['ledger']:+.2f} ({t['diff']:+.2f})")
    for f in r.flows_missing:
        lines.append(f"  ❗ {f['date']} {f['category']} {f['amount']:+,.2f} not in the ledger "
                     "(fund deposit/withdraw --record-only)")
    for f in r.flows_extra:
        lines.append(f"  ❗ ledger cash flow {f['date']} {f['amount']:+,.2f} not at Montrose")
    if r.cash_events:
        lines.append("Cash events the ledger doesn't book:")
        for e in r.cash_events:
            lines.append(f"  · {e['date']} {e['category']} {e['amount']:+.2f}"
                         + (f" ({e['text']})" if e["text"] else ""))
    if r.in_sync and not (r.amount_diffs or r.date_diffs or r.cash_events):
        lines.append("Nothing to fix.")
    lines.append("Read-only: nothing was written to the fund.")
    return "\n".join(lines)
