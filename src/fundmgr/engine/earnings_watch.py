"""
Earnings watch: the evening before a holding reports, say so — with its thesis.

The report itself is read by `report-check` (engine/report_check.py) at 08:45,
before the open, and again at 18:30. This is the part that comes first: a
heads-up the evening before, so the owner knows what is due and what the
position is held on before the morning verdict arrives.

It used to read the report as well, with one call of its own; that duplicated
report-check (built separately on 21 Oct), so the verdict was handed to it —
one reader per report, filed as a scored review.

Report dates come from Yahoo's earnings calendar, which misses many small Nordic
names: those get no heads-up, but report-check still finds their report from
its headlines once it is out.
"""
from __future__ import annotations

import html
from datetime import date, datetime, timedelta, timezone
from typing import Callable


def next_weekday(d: date) -> date:
    d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _calendar_date(ticker: str) -> date | None:
    from fundmgr.paper import _next_earnings_date
    return _next_earnings_date(ticker)


def heads_up_html(fund: str, ticker: str, name: str, report_date: str, theses: dict) -> str:
    when = datetime.fromisoformat(report_date).strftime("%a %d %b")
    lines = [f"📅 <b>{html.escape(fund)} — {html.escape(name)} ({html.escape(ticker)}) reports "
             f"{when}</b>"]
    for key, label in (("buy", "Bought on"), ("hold", "Held on")):
        if key in theses:
            lines.append(f"<i>{label} ({theses[key]['date']}):</i> {html.escape(theses[key]['thesis'])}")
    if not theses:
        lines.append("No thesis recorded for this position.")
    lines.append("The report check reads it against this before the next open (08:45), "
                 "or at 18:30 if it comes out during the day.")
    return "\n".join(lines)


def run(store, cfg, today: date | None = None, *, send: Callable[[str], bool],
        names: dict[str, str] | None = None, dry_run: bool = False,
        calendar_fn: Callable = _calendar_date,
        echo: Callable[[str], None] = print) -> list[dict]:
    """Heads-up for every holding that reports on the next weekday. Once each."""
    today = today or datetime.now(timezone.utc).date()
    names = names or {}
    due = next_weekday(today)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sent: list[dict] = []
    for p in store.get_positions():
        if p.shares <= 0:
            continue
        when = calendar_fn(p.ticker)
        if when != due:
            continue
        rd = when.isoformat()
        row = store.get_earnings_check(p.ticker, rd)
        if row and row["heads_up_at"]:
            continue
        echo(f"  📅 {p.ticker} reports {rd} — heads-up")
        if not dry_run:
            send(heads_up_html(cfg.display_name, p.ticker, names.get(p.ticker, p.ticker), rd,
                               store.get_position_theses(p.ticker)))
            store.upsert_earnings_check(p.ticker, rd, found_by="calendar", heads_up_at=now)
        sent.append({"ticker": p.ticker, "kind": "heads_up", "report_date": rd})
    return sent
