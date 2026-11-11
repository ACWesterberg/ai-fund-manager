"""Do insider purchases here precede outperformance? Free, and honest to backtest.

Research finds insider *buying* informative, especially when several insiders
buy at once, and insider selling much less so (people sell for tax, a house,
diversification). This measures it on this universe from FI's register: each
event's abnormal return (stock less the index) over the next 5, 20 and 28
trading days, for any buy, a cluster buy and any sell.

Unlike the report-reaction experiment, this rule is mechanical, so history can
be used: no model sees anything, and nothing a model remembers can leak in.
Entry is the first close on a day after FI published the filing — the market
could not act on it earlier — never the trade date, which precedes publication
by up to three business days. No model calls, nothing written but prices.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import Callable

from fundmgr.data.insiders import InsiderTrade, fetch_trades
from fundmgr.engine.report_events import (
    HORIZONS, _ensure_benchmark, _fetch_history, _fmt_row, forward, summarize,
)

CLUSTER_DAYS = 30
CLUSTER_PEOPLE = 2
EVENT_GAP_DAYS = 30          # one event of a kind per name per month: fills and follow-ups
MIN_VALUE_SEK = 100_000      # below this a purchase is a gesture, not a bet
KINDS = (("buy", "insider buy"), ("cluster", "cluster buy"), ("sell", "insider sell"))


def _sek(trade: InsiderTrade, rates: dict[str, float | None]) -> float | None:
    if trade.currency == "SEK":
        return trade.value
    if trade.currency not in rates:
        from fundmgr.data.fx import rate_to_sek
        rates[trade.currency] = rate_to_sek(trade.currency)
    rate = rates[trade.currency]
    return trade.value * rate if rate else None


def find_events(trades: list[InsiderTrade], isin_to_ticker: dict[str, str],
                min_value_sek: float = MIN_VALUE_SEK) -> list[dict]:
    """Buy, cluster-buy and sell events per name, dated by publication.

    A manager's fills on one day are one decision, so they are summed first and
    the threshold applies to the total. A cluster fires when the second distinct
    manager's buy is published within CLUSTER_DAYS of another's.
    """
    rates: dict[str, float | None] = {}
    daily: dict[tuple, float] = defaultdict(float)
    for t in trades:
        ticker = isin_to_ticker.get(t.isin)
        sek = _sek(t, rates) if ticker else None
        if sek is not None:
            daily[(ticker, t.side, t.person, t.published.date())] += sek

    by_name: dict[tuple, list[tuple[date, str]]] = defaultdict(list)
    for (ticker, side, person, day), sek in daily.items():
        if sek >= min_value_sek:
            by_name[(ticker, side)].append((day, person))

    events = []
    for (ticker, side), filings in by_name.items():
        filings.sort()
        last: dict[str, date] = {}
        for i, (day, _person) in enumerate(filings):
            kinds = [side]
            recent = {p for d, p in filings[:i + 1] if (day - d).days <= CLUSTER_DAYS}
            if side == "buy" and len(recent) >= CLUSTER_PEOPLE:
                kinds.append("cluster")
            for kind in kinds:
                if kind in last and (day - last[kind]).days < EVENT_GAP_DAYS:
                    continue
                last[kind] = day
                events.append({"ticker": ticker, "kind": kind, "published": day.isoformat()})
    return sorted(events, key=lambda e: (e["published"], e["ticker"], e["kind"]))


def run_study(store, cfg, universe: list, years: int = 1,
              fetch: Callable[[date, date], list[InsiderTrade]] = fetch_trades,
              echo: Callable[[str], None] = print) -> dict:
    end = date.today()
    start = end - timedelta(days=365 * years)
    echo(f"  Reading FI's insider register, {start} → {end}, two weeks at a time…")
    trades = fetch(start, end)
    isin_map = {t.isin.strip().upper(): t.yahoo_ticker for t in universe if t.isin}
    matched = [t for t in trades if t.isin in isin_map]
    events = find_events(matched, isin_map)
    names = {e["ticker"] for e in events}
    echo(f"  {len(trades)} open-market share trades; {len(matched)} in this universe; "
         f"{len(events)} events on {len(names)} names.")

    days = 365 * years + 60
    _fetch_history(store, [t for t in universe if t.yahoo_ticker in names], days, echo)
    bench = _ensure_benchmark(store, cfg, days)
    for e in events:
        series = store.get_prices(e["ticker"])
        entry = next((i for i, r in enumerate(series) if r["date"] > e["published"]), None)
        e["fwd"] = forward(series, bench, entry) if entry is not None else {n: None for n in HORIZONS}
    return {"start": start.isoformat(), "end": end.isoformat(), "trades": len(trades),
            "matched": len(matched), "names": len(names), "events": events}


def format_study(result: dict) -> str:
    lines = [
        f"\n─── Insider trades — base rates ({result['start']} → {result['end']}) ───",
        f"  {result['trades']} open-market share trades in FI's register, {result['matched']} "
        f"in this universe ({result['names']} names with an event). FI covers Swedish",
        f"  issuers only. A trade counts at {MIN_VALUE_SEK:,} SEK or more per manager per day;",
        f"  programme grants are excluded. A cluster is {CLUSTER_PEOPLE}+ managers buying within "
        f"{CLUSTER_DAYS} days.",
        "  Entry: the first close after FI published the filing. Then: abnormal return",
        "  (stock less the index) from that close, N trading days on.",
    ]
    for n in HORIZONS:
        lines.append(f"\n  +{n} trading days:")
        for kind, label in KINDS:
            vals = [e["fwd"][n] for e in result["events"]
                    if e["kind"] == kind and e["fwd"][n] is not None]
            lines.append(_fmt_row(label, summarize(vals)))
    lines.append(
        "\n  Read: a positive mean on the buy rows, with an interval clear of zero, is the"
        "\n  edge the research describes. One year is a short window; --years 3 widens it."
    )
    return "\n".join(lines)
