"""
Montrose phase 2 — book broker rows into the ledger, one confirmed row at a time.

The reconcile (phase 1) already knows which Montrose transactions the ledger
lacks. This turns each into a proposal — a fill, a deposit/withdrawal, or a
dividend/tax cash row — that the owner books or skips with a Telegram button.
Nothing is written without that tap.

Every decision goes through `decide`, which re-reads Montrose and re-runs the
reconcile first: the row must still exist, still be uncancelled and still be
missing from the ledger. A fill booked by hand (/fill, a screenshot) between
the offer and the tap therefore can't be booked twice, and a stale button
does nothing. The decision is stored under Montrose's own id
(`broker_imports`), so a row is offered and booked at most once.

Rows dated before `montrose.import_from` are never proposed: history before
phase 2 was booked by hand, and some of it (the September KCC dividend) was
folded into cash with /setcash, so booking it again would count it twice.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

from fundmgr.broker import montrose
from fundmgr.broker import reconcile as rec
from fundmgr.state.models import Transaction

# Return on the holdings: booked as cash, not as a flow.
CASH_CATEGORIES = ("Dividend", "ForeignTax", "WithholdingTax", "Interest")
# A split, transfer or FX exchange changes more than cash; the ledger has no
# model for it, so it is named and left for a hand fix.
MANUAL_CATEGORIES = ("Other", "CurrencyExchange", "CorporateAction",
                     "SecuritiesTransferIn", "SecuritiesTransferOut")
MAX_FEE_FRACTION = 0.02


@dataclass
class Proposal:
    broker_id: int
    kind: str                     # fill | flow | cash | manual
    category: str
    trade_date: str
    amount_sek: float             # the broker's cash effect, fee included
    ticker: str | None = None
    montrose: str | None = None
    side: str | None = None
    shares: float | None = None
    price_sek: float | None = None
    fee_sek: float | None = None
    price_native: float | None = None
    price_currency: str | None = None
    fee_estimated: bool = False
    text: str = ""
    problem: str | None = None    # set → no Book button; says what to do instead

    @property
    def bookable(self) -> bool:
        return self.problem is None and self.kind != "manual"

    def to_dict(self) -> dict:
        return asdict(self)


def split_fee(side: str, shares: float, price: float, price_currency: str, amount: float,
              fee_calc: Any) -> tuple[float, float, bool]:
    """(price_sek, fee_sek, estimated) whose cash effect is exactly the broker's amount.

    Montrose reports the settled SEK amount, fee included, but not the fee.
    For a SEK trade the fee is the gap between amount and shares × price. For a
    foreign one that gap also holds the FX conversion, so the fee is the fee
    schedule's estimate and the per-share SEK price absorbs the rest — the
    ledger's cash then moves by exactly what Montrose moved.
    """
    paid = abs(amount)
    if price_currency.upper() == "SEK" and shares > 0 and price > 0:
        gross = shares * price
        fee = paid - gross if side == "buy" else gross - paid
        if 0 <= fee <= MAX_FEE_FRACTION * gross + 1:
            return price, round(fee, 2), False
    fee = round(float(fee_calc(paid)), 2)
    if side == "buy":
        return (paid - fee) / shares, fee, True
    return (paid + fee) / shares, fee, True


def proposals(report: rec.Report, *, import_from: date, imports: dict[int, dict],
              fee_calc: Any, account_ccy: str = "SEK") -> list[Proposal]:
    """Bookable (or explicitly not-bookable) proposals from a reconcile report."""
    out: list[Proposal] = []

    def fresh(item: dict) -> bool:
        return (item.get("id") is not None and int(item["id"]) not in imports
                and date.fromisoformat(item["date"]) >= import_from)

    for t in report.missing_in_ledger:
        if not fresh(t):
            continue
        p = Proposal(broker_id=int(t["id"]), kind="fill", category=t["side"].title(),
                     trade_date=t["date"], amount_sek=t["amount"], ticker=t["ticker"],
                     montrose=t.get("montrose"), side=t["side"], shares=t["shares"],
                     price_native=t["price"], price_currency=t.get("price_currency") or "")
        clash = [lt for lt in report.not_at_broker
                 if lt["ticker"] == t["ticker"] and lt["side"] == t["side"]
                 and abs((date.fromisoformat(lt["date"]) - date.fromisoformat(t["date"])).days)
                 <= rec.DATE_SLACK_DAYS]
        if not t["ticker"]:
            p.problem = (f"no ledger symbol for {t.get('montrose') or t['orderbookId']} — "
                         f"fund montrose-map {t['orderbookId']} SYMBOL, then it is offered again")
        elif str(t.get("currency") or account_ccy).upper() != account_ccy:
            p.problem = f"settled in {t.get('currency')}, not {account_ccy} — book by hand"
        elif clash:
            p.problem = (f"the ledger has a {t['side']} of {rec._sh(clash[0]['shares'])} "
                         f"{t['ticker']} on {clash[0]['date']} that Montrose doesn't — fix that "
                         "fill by hand first (/undo or /fill), or this would count it twice")
        else:
            p.price_sek, p.fee_sek, p.fee_estimated = split_fee(
                t["side"], t["shares"], t["price"], p.price_currency, t["amount"], fee_calc)
        out.append(p)

    for f in report.flows_missing:
        if not fresh(f):
            continue
        p = Proposal(broker_id=int(f["id"]), kind="flow", category=f["category"],
                     trade_date=f["date"], amount_sek=f["amount"])
        if report.flows_extra:
            p.problem = ("the ledger has a deposit/withdrawal Montrose doesn't — "
                         "check /cash_flows before booking this one")
        out.append(p)

    for e in report.cash_events:
        if not fresh(e):
            continue
        kind = "cash" if e["category"] in CASH_CATEGORIES else "manual"
        p = Proposal(broker_id=int(e["id"]), kind=kind, category=e["category"],
                     trade_date=e["date"], amount_sek=e["amount"], text=e.get("text") or "")
        if kind == "manual":
            p.problem = f"{e['category']} changes more than cash — book by hand"
        out.append(p)

    order = {"flow": 0, "cash": 1, "fill": 2, "manual": 3}
    # Same day: money in and sells before buys, so a buy funded by them books.
    return sorted(out, key=lambda p: (p.trade_date, order[p.kind],
                                      0 if (p.side or "sell") == "sell" else 1, p.broker_id))


def describe(p: Proposal) -> str:
    if p.kind == "fill":
        fee = f"fee {p.fee_sek:.2f}{' (est.)' if p.fee_estimated else ''}" if p.fee_sek is not None else ""
        native = (f" ({p.price_native:g} {p.price_currency})"
                  if p.price_currency and p.price_currency.upper() != "SEK" else "")
        price = f" @ {p.price_sek:.4f} SEK{native}, {fee}" if p.price_sek is not None else ""
        line = (f"{p.trade_date} {p.side.upper()} {rec._sh(p.shares)} "
                f"{p.ticker or p.montrose}{price} = {p.amount_sek:+,.2f} SEK")
    elif p.kind == "flow":
        line = f"{p.trade_date} {p.category} {p.amount_sek:+,.2f} SEK (money in/out — not return)"
    else:
        line = (f"{p.trade_date} {p.category} {p.amount_sek:+,.2f} SEK"
                + (f" — {p.text}" if p.text else ""))
    return line


# ── Offers (which proposals Telegram has already shown) ──────────────────────

def _offered_path():
    return montrose.state_dir() / "offered.json"


def unoffered(props: list[Proposal]) -> list[Proposal]:
    """Proposals not yet sent with buttons — the daily job offers each once;
    /montrose_import sends them all again on demand."""
    seen = montrose._read(_offered_path()) or {}
    return [p for p in props if str(p.broker_id) not in seen]


def mark_offered(props: list[Proposal]) -> None:
    seen = montrose._read(_offered_path()) or {}
    now = datetime.now().isoformat(timespec="seconds")
    seen.update({str(p.broker_id): now for p in props})
    montrose._write_private(_offered_path(), seen)


def _note(p: Proposal) -> str:
    return f"Montrose #{p.broker_id} {p.category} {p.trade_date}" + (f" {p.text}" if p.text else "")


def book(store: Any, p: Proposal) -> str:
    """Write one proposal to the ledger, atomically with its import record."""
    if not p.bookable:
        raise ValueError(p.problem or f"{p.category} is booked by hand")
    if p.kind == "fill":
        txn = Transaction(ticker=p.ticker, side=p.side, shares=p.shares, price_sek=p.price_sek,
                          fee_sek=p.fee_sek, source="montrose",
                          timestamp=datetime.fromisoformat(p.trade_date + "T12:00:00"))
        store.book_broker_fill(p.broker_id, txn, p.category, p.trade_date, p.amount_sek, _note(p))
        return (f"✓ Booked {p.side} {rec._sh(p.shares)} × {p.ticker} @ {p.price_sek:.4f} SEK "
                f"(fee {p.fee_sek:.2f}) — cash now {store.get_cash():,.2f} SEK")
    cash = store.book_broker_cash(p.broker_id, p.category, p.trade_date, p.amount_sek,
                                  _note(p), flow=p.kind == "flow")
    return f"✓ Booked {p.category} {p.amount_sek:+,.2f} SEK — cash now {cash:,.2f} SEK"


def skip(store: Any, p: Proposal) -> str:
    store.skip_broker_row(p.broker_id, p.category, p.trade_date, p.amount_sek, _note(p))
    return f"Skipped Montrose #{p.broker_id} ({p.category} {p.trade_date}); it won't be offered again."


def find(props: list[Proposal], broker_id: int) -> Proposal:
    for p in props:
        if p.broker_id == broker_id:
            return p
    raise montrose.MontroseError(
        f"Montrose #{broker_id} is not waiting to be booked any more — already in the ledger, "
        "already decided, cancelled at Montrose, or dated before montrose.import_from.")
