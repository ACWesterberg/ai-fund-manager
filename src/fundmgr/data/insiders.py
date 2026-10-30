"""Insider trades from Finansinspektionen's PDMR register (insynshandel).

Managers and board members of Swedish issuers report their own trades to FI
within three business days, and FI's search page exports them as CSV. That is
the source here, matched to the universe by ISIN.

The reader in financedata cannot do this job: its fixed `insynsregistret.csv`
link now returns 404, and it matched a ticker's text inside the issuer's name,
which fails for every share-class ticker (`VOLV-B` is not in "AB Volvo").

FI covers Swedish issuers only; Oslo, Copenhagen and Helsinki names report to
their own regulators and never appear.
"""
from __future__ import annotations

import csv
import io
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import httpx

EXPORT_URL = "https://marknadssok.fi.se/publiceringsklient/sv-SE/Search/Search"
CHUNK_DAYS = 14

BUY, SELL = "Förvärv", "Avyttring"
REQUIRED = ("Publiceringsdatum", "Emittent", "Person i ledande ställning", "Befattning",
            "Närstående", "Är kopplad till aktieprogram", "Karaktär", "Instrumenttyp",
            "ISIN", "Transaktionsdatum", "Volym", "Pris", "Valuta", "Status")


class InsiderDataError(RuntimeError):
    """FI's export could not be read as the register it used to be."""


@dataclass(frozen=True)
class InsiderTrade:
    published: datetime          # when the market could know
    traded: date
    issuer: str
    isin: str
    person: str                  # the manager the trade is reported under
    role: str
    side: str                    # buy | sell
    shares: float
    price: float
    currency: str
    closely_associated: bool     # made by a relative or a company the manager controls

    @property
    def value(self) -> float:
        return self.shares * self.price


def _num(text: str) -> float | None:
    try:
        return float((text or "").replace("\xa0", "").replace(" ", "").replace(",", "."))
    except ValueError:
        return None


def parse_export(raw: bytes) -> list[InsiderTrade]:
    """Open-market share buys and sells, current filings only.

    Dropped: grants and option exercises tied to a share programme (pay, not a
    view), subscriptions, loans and internal transfers, anything but shares,
    and filings FI marks revised or cancelled — a correction arrives as its
    own current row.
    """
    text = None
    for enc in ("utf-16", "utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            if "Publiceringsdatum" in text[:200]:
                break
        except UnicodeDecodeError:
            continue
    if not text or "Publiceringsdatum" not in text[:200]:
        raise InsiderDataError("FI export has no header row — the format has changed")
    rows = list(csv.DictReader(io.StringIO(text), delimiter=";"))
    missing = [c for c in REQUIRED if rows and c not in rows[0]]
    if missing:
        raise InsiderDataError(f"FI export lacks columns: {', '.join(missing)}")

    trades = []
    for r in rows:
        side = {BUY: "buy", SELL: "sell"}.get((r["Karaktär"] or "").strip())
        if (side is None or r["Instrumenttyp"].strip() != "Aktie"
                or r["Status"].strip() != "Aktuell"
                or r["Är kopplad till aktieprogram"].strip() == "Ja"
                or not r["ISIN"].strip()):
            continue
        shares, price = _num(r["Volym"]), _num(r["Pris"])
        if not shares or shares <= 0 or price is None or price <= 0:
            continue
        try:
            published = datetime.strptime(r["Publiceringsdatum"].strip()[:19], "%Y-%m-%d %H:%M:%S")
            traded = datetime.strptime(r["Transaktionsdatum"].strip()[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        trades.append(InsiderTrade(
            published=published, traded=traded, issuer=r["Emittent"].strip(),
            isin=r["ISIN"].strip().upper(),
            person=" ".join(r["Person i ledande ställning"].split()),
            role=" ".join(r["Befattning"].replace("\xa0", " ").split()),
            side=side, shares=shares, price=price, currency=r["Valuta"].strip().upper() or "SEK",
            closely_associated=r["Närstående"].strip() == "Ja",
        ))
    return trades


def fetch_trades(start: date, end: date, *, client: httpx.Client | None = None,
                 pause: float = 0.5) -> list[InsiderTrade]:
    """Every filing published from `start` to `end`, fetched two weeks at a time."""
    own = client is None
    client = client or httpx.Client(timeout=60, follow_redirects=True,
                                    headers={"User-Agent": "fundmgr insider study"})
    trades: list[InsiderTrade] = []
    try:
        lo = start
        while lo <= end:
            hi = min(end, lo + timedelta(days=CHUNK_DAYS - 1))
            resp = client.get(EXPORT_URL, params={
                "SearchFunctionType": "Insyn", "Utgivare": "", "PersonILedandeStallningNamn": "",
                "Transaktionsdatum.From": "", "Transaktionsdatum.To": "",
                "Publiceringsdatum.From": lo.isoformat(), "Publiceringsdatum.To": hi.isoformat(),
                "button": "export",
            })
            resp.raise_for_status()
            trades += parse_export(resp.content)
            lo = hi + timedelta(days=1)
            if lo <= end and pause:
                time.sleep(pause)
    finally:
        if own:
            client.close()
    return trades
