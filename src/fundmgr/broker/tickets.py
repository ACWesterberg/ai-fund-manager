"""
Montrose phase 3 — a pre-filled order ticket for each trade the weekly run approved.

A ticket is not an order: it opens the Montrose app on a filled-in order
screen (https://app.montrose.io/trade?ticketId=…) and the owner sets the limit
and approves it there. So the run still never places anything; this saves the
typing and, more to the point, the chance of typing the wrong instrument.

Instruments are where a ticket could go wrong, so they resolve only on
certainty, never on a best guess:
  1. an orderbookId pinned with `fund montrose-map`;
  2. one already seen in this account (holdings or transactions), whose
     ticker and quote currency match the Yahoo symbol;
  3. Montrose's `search_instruments`, accepted only if exactly one result has
     the right ticker. Search results carry no currency, so a name listed
     twice (ASML in Amsterdam and New York) gets no ticket — it is named, and
     picked in the app by hand.

The plan (which trades, how many whole shares) is fixed when the run decides
and saved privately, so /montrose_tickets can re-issue the same tickets later
without re-deciding anything.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from fundmgr.broker import montrose
from fundmgr.broker import reconcile as rec


@dataclass
class TicketLine:
    ticker: str
    side: str                       # buy | sell
    shares: int
    kind: str = ""                  # NEW / ADD / TRIM / EXIT
    ref_price: float | None = None  # last close, native currency
    currency: str = ""
    orderbook_id: int | None = None
    montrose: str | None = None
    url: str | None = None
    problem: str | None = None

    def label(self) -> str:
        return f"{self.side.upper()} {self.shares} {self.ticker}"


# ── Plan ─────────────────────────────────────────────────────────────────────

def plan_from_run(run_id: str, actions: list, snap: Any, features: dict, cfg: Any) -> dict:
    """Whole-share trades from a run's approved actions, sells first."""
    from fundmgr.guardrails.rules import shares_for_action
    from fundmgr.reporting.actions import trade_kind

    lines: list[TicketLine] = []
    for a in actions:
        if a.side not in ("buy", "sell"):
            continue
        kind = trade_kind(a, snap)
        held = next((p.shares for p in snap.positions if p.ticker == a.ticker), 0.0)
        # An exit sells what is held, not what the SEK estimate rounds to.
        shares = (math.floor(held) if kind == "EXIT"
                  else shares_for_action(a, snap, features, cfg))
        feat = features.get(a.ticker)
        line = TicketLine(ticker=a.ticker, side=a.side, shares=int(shares or 0), kind=kind,
                          ref_price=feat.last_price if feat else None,
                          currency=feat.currency if feat else "")
        if not line.shares:
            line.problem = "no price to size it, or it rounds to 0 shares"
        lines.append(line)
    lines.sort(key=lambda t: 0 if t.side == "sell" else 1)
    return {"run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "lines": [asdict(t) for t in lines]}


def _plan_path():
    return montrose.state_dir() / "ticket-plan.json"


def save_plan(plan: dict) -> None:
    montrose._write_private(_plan_path(), plan)


def load_plan() -> dict:
    plan = montrose._read(_plan_path())
    if not plan:
        raise montrose.MontroseError("No ticket plan yet — it is saved by the next weekly run.")
    return plan


def plan_lines(plan: dict) -> list[TicketLine]:
    return [TicketLine(**{k: v for k, v in t.items() if k in TicketLine.__dataclass_fields__})
            for t in plan.get("lines", [])]


def mark_done(lines: list[TicketLine], ledger_trades: list[rec.LedgerTrade], since: str) -> list[str]:
    """Drop lines already filled in the ledger since the plan was made; name them."""
    start = datetime.fromisoformat(since).date()
    done = []
    for t in list(lines):
        if any(lt.ticker == t.ticker.upper() and lt.side == t.side and lt.date >= start
               for lt in ledger_trades):
            lines.remove(t)
            done.append(t.label())
    return done


# ── Resolution ───────────────────────────────────────────────────────────────

def _norm(ticker: str) -> str:
    return (ticker or "").strip().upper().replace(" ", "-")


def resolve_orderbook(yahoo: str, instruments: dict[str, dict], known: dict[str, str],
                      search: Any) -> tuple[int | None, str | None, str | None]:
    """(orderbookId, Montrose ticker, problem). `search(query)` → search_instruments rows."""
    y = yahoo.upper()
    pinned = [k for k, e in instruments.items() if (e.get("yahoo") or "").upper() == y]
    if len(pinned) == 1:
        return int(pinned[0]), instruments[pinned[0]].get("ticker"), None
    if len(pinned) > 1:
        return None, None, f"{y} is pinned to {len(pinned)} orderbooks — fix with fund montrose-map"

    seen = [k for k, e in instruments.items()
            if not e.get("yahoo") and e.get("ticker")
            and ((e.get("resolved_yahoo") or "").upper() == y
                 or rec.resolve_yahoo(e["ticker"], e.get("name") or "", e.get("currency") or "",
                                      known) == y)]
    if len(seen) == 1:
        return int(seen[0]), instruments[seen[0]].get("ticker"), None

    stem = y.rsplit(".", 1)[0] if "." in y else y
    rows = search(stem.replace("-", " ")) or []
    exact = [r for r in rows if _norm(r.get("ticker")) == stem]
    # Montrose drops the class for single-class listings: TRUE for TRUE-B.ST.
    prefix = [r for r in rows if stem.startswith(_norm(r.get("ticker")) + "-")]
    hits = exact or prefix
    if len(hits) == 1 and hits[0].get("orderbookId"):
        r = hits[0]
        entry = instruments.setdefault(str(r["orderbookId"]), {})
        entry.setdefault("ticker", r.get("ticker"))
        entry.setdefault("name", r.get("name"))
        entry["resolved_yahoo"] = y
        return int(r["orderbookId"]), r.get("ticker"), None
    if hits:
        names = ", ".join(f"{r.get('ticker')} ({r.get('name')}, {r.get('orderbookId')})" for r in hits)
        return None, None, (f"{stem} matches {len(hits)} Montrose listings: {names} — pick it in "
                            f"the app, or fund montrose-map ORDERBOOKID {y}")
    return None, None, f"Montrose has no listing with ticker {stem} — pick it in the app"


def create_tickets(account_name: str, lines: list[TicketLine], known: dict[str, str]) -> list[TicketLine]:
    """Resolve and create a ticket for every line without a problem. One session."""
    instruments = rec.load_instruments()
    with montrose.Session() as s:
        account = rec._find_account(rec._unwrap(s.call_tool("get_user_accounts")), account_name)

        def search(query: str) -> list[dict]:
            return rec._unwrap(s.call_tool("search_instruments", {"query": query})) or []

        for t in lines:
            if t.problem:
                continue
            t.orderbook_id, t.montrose, t.problem = resolve_orderbook(t.ticker, instruments, known, search)
            if t.problem:
                continue
            try:
                t.url = s.create_trade_ticket("Buy" if t.side == "buy" else "Sell",
                                              t.orderbook_id, t.shares, account["accountId"])
            except montrose.MontroseError as exc:
                t.problem = str(exc)
    rec.save_instruments(instruments)
    return lines


# ── Telegram ─────────────────────────────────────────────────────────────────

def message(lines: list[TicketLine], run_id: str, done: list[str] | None = None) -> tuple[str, list]:
    """(HTML text, URL button rows) — one button per ticket."""
    import html

    out = ["🎫 <b>Montrose ticket</b>" if run_id == "manual"
           else f"🎫 <b>Montrose tickets</b> — run <code>{html.escape(run_id)}</code>"]
    for t in lines:
        ref = f" · last {t.ref_price:g} {t.currency}" if t.ref_price else ""
        mtr = f" ({html.escape(t.montrose)})" if t.montrose and t.url else ""
        line = f"{'✅' if t.url else '⚠️'} {t.kind} {html.escape(t.label())}{mtr}{ref}"
        if t.problem:
            line += f"\n    {html.escape(t.problem)}"
        out.append(line)
    if done:
        out.append("Already in the ledger, no ticket: " + html.escape(", ".join(done)))
    if any(t.url for t in lines):
        out.append("\nEach button opens a filled-in order in Montrose. Set the limit price and "
                   "approve it there — nothing is placed until you do.")
    buttons = [[(t.label(), t.url)] for t in lines if t.url]
    return "\n".join(out), buttons


def to_json(lines: list[TicketLine]) -> str:
    return json.dumps([asdict(t) for t in lines], indent=2, ensure_ascii=False)
