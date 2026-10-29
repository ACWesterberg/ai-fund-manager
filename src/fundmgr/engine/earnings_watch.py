"""
Earnings watch: read each held name's report against the thesis it was bought on.

The real book had no-one reading its own holdings' reports. `check-news` scores
headlines with FinBERT from 09:00 — after a Nordic report has landed at 07:30
and the open has priced it — `report-watch` asks a different question (was a
drop an overreaction?) about the whole universe after the close, and only the
live sleeves had an earnings calendar. So a report that broke a holding's
thesis reached the owner, if at all, as a price move.

`fund earnings-watch` runs at 08:45 and 18:30 on weekdays:

  • The evening before a holding reports: a heads-up with the thesis it is held on.
  • Once the report is out: one model call reads the coverage against that
    thesis — held / broken / unclear — with quotes checked against the text,
    as in report_events. The morning slot lands before the 09:00 open for the
    usual 07:00–08:00 Nordic release; the evening slot catches midday reports
    before the next open.

Rules that keep it honest:
  • Only coverage published on or after the report date is read, so the week's
    previews can't stand in for the results. No coverage, no call.
  • An `unclear` on the report day is not final — the results may not be out
    yet — so it is asked again at the next slot, up to `max_checks_per_report`
    calls; after that the owner is told to read it themselves.
  • The price reaction is not in the prompt: the verdict is on the report, not
    on what the market made of it.
  • It never trades. A broken thesis comes with `/review TICKER` and a sell
    ticket command, and the decision stays with the owner.

Reports are found from Yahoo's earnings calendar, which misses many small
Nordic names; for those a report headline on a held name in the last two days
is taken as the report (`report_events.REPORT_WORDS`). A headline that turns
out to be a preview ends as `unclear` and costs at most the capped calls.
"""
from __future__ import annotations

import copy
import html
import json
from datetime import date, datetime, timedelta, timezone
from typing import Callable

MAX_SOURCE_CHARS = 30_000
HEADLINE_LOOKBACK_DAYS = 2
# A report older than this with no verdict is given up on, not retried forever.
GIVE_UP_AFTER_DAYS = 3


def next_weekday(d: date) -> date:
    d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


# ── Finding reports ──────────────────────────────────────────────────────────

def _calendar_date(ticker: str) -> date | None:
    from fundmgr.paper import _next_earnings_date
    return _next_earnings_date(ticker)


def _headline_date(ticker: str, today: date, news_fn: Callable) -> date | None:
    """The newest report headline on `ticker` dated within the lookback, else None."""
    from fundmgr.data.release import _parse_date
    from fundmgr.engine.report_events import REPORT_WORDS

    found = None
    for item in news_fn(ticker) or []:
        when = _parse_date(item.get("published", ""))
        if when is None or not (today - timedelta(days=HEADLINE_LOOKBACK_DAYS) <= when <= today):
            continue
        if REPORT_WORDS.search(item.get("headline", "")):
            found = max(found, when) if found else when
    return found


def discover(store, tickers: list[str], today: date, calendar_fn: Callable = _calendar_date,
             news_fn: Callable | None = None) -> None:
    """Record each held name's next or just-published report in earnings_checks."""
    from fundmgr import evidence
    news_fn = news_fn or evidence.news_items
    for t in tickers:
        when = calendar_fn(t)
        if when is not None and when <= next_weekday(today):
            store.upsert_earnings_check(t, when.isoformat(), found_by="calendar")
            continue
        open_rows = [r for r in store.get_earnings_checks(open_only=True) if r["ticker"] == t]
        if open_rows:
            continue
        hd = _headline_date(t, today, news_fn)
        if hd is not None and store.get_earnings_check(t, hd.isoformat()) is None:
            # Within a window of a calendar row already judged: same report.
            judged = [r for r in store.get_earnings_checks() if r["ticker"] == t and r["verdict"]
                      and abs((date.fromisoformat(r["report_date"]) - hd).days) <= HEADLINE_LOOKBACK_DAYS]
            if not judged:
                store.upsert_earnings_check(t, hd.isoformat(), found_by="headline")


# ── Judging ──────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are the analyst for a fund that owns this company. It has just published a \
report. Decide whether the report confirms the reason the fund holds the \
shares, breaks it, or cannot be judged from the coverage given.

- held: the report supports the thesis, or leaves its claims intact. A miss on \
timing, currency, mix or one-off costs that the thesis does not depend on is \
still held.
- broken: the report contradicts something the thesis rests on — guidance cut \
for real reasons, the growth, margin, order or cash-flow path it relied on not \
happening, lost share, balance-sheet stress.
- unclear: the coverage does not contain the results (a preview, an invitation \
to a presentation, a headline only), or is too thin to tell. That is a correct \
answer, not a failure.

Rules:
- Use only the coverage below. Do not rely on anything you remember about this \
company or how its share price moved after this report.
- Quote sentences verbatim from the coverage. Quotes are checked; one that is \
not in the text is discarded, and a held or broken verdict with no surviving \
quote counts as unclear.
- Judge the thesis as written. If no thesis is recorded, judge whether the \
report leaves the business stronger or weaker than before."""


def build_prompt(ticker: str, name: str, report_date: str, theses: dict, source: str) -> tuple[str, str]:
    parts = []
    if "buy" in theses:
        parts.append(f"Bought ({theses['buy']['date']}) because: {theses['buy']['thesis']}")
    if "hold" in theses:
        parts.append(f"Latest reason to keep holding ({theses['hold']['date']}): {theses['hold']['thesis']}")
    thesis = "\n".join(parts) or "No thesis is recorded for this position."
    user = f"""# {name} ({ticker}) — report {report_date}

## Why the fund holds it
{thesis}

## Coverage of the report
{source}

## Your task
Does this report leave the thesis held, broken, or is it unclear from this coverage?
Return the ReportThesisCheck JSON."""
    return _SYSTEM, user


def coverage_since(ticker: str, report_date: str, coverage_fn: Callable | None = None) -> list[dict]:
    """Coverage published on/after the report date — undated items are dropped
    here (unlike release.find_coverage) because a dateless preview would be read
    as the results."""
    from fundmgr.data.release import _parse_date, find_coverage
    coverage_fn = coverage_fn or find_coverage
    start = date.fromisoformat(report_date)
    out = []
    for item in coverage_fn(ticker, report_date) or []:
        when = _parse_date(item.get("published", ""))
        if when is not None and when >= start:
            out.append(item)
    return out


def judge(ticker: str, name: str, report_date: str, theses: dict, cfg,
          coverage_fn: Callable | None = None, llm: Callable | None = None) -> dict | None:
    """One model call, quotes checked. None when there is no coverage yet (no call)."""
    from fundmgr.data.release import _source_text
    from fundmgr.engine.client import call_llm
    from fundmgr.engine.report_events import quote_in_source
    from fundmgr.engine.schema import ReportThesisCheck

    items = coverage_since(ticker, report_date, coverage_fn)
    if not items:
        return None
    source = _source_text(items)[:MAX_SOURCE_CHARS]
    jcfg = copy.deepcopy(cfg)
    jcfg.llm.reasoning_effort = cfg.earnings_watch.reasoning_effort
    jcfg.llm.max_tokens = cfg.earnings_watch.max_tokens
    jcfg.llm.n_samples = 1
    system, user = build_prompt(ticker, name, report_date, theses, source)
    parsed, _raw = (llm or call_llm)(system, user, jcfg, ReportThesisCheck)

    quotes = [{"quote": q.quote, "shows": q.shows} for q in parsed.quotes
              if quote_in_source(q.quote, source)]
    verdict, what = parsed.verdict, parsed.what_changed
    if verdict != "unclear" and not quotes:
        verdict = "unclear"
        what = f"[No quote survived the source check; verdict was {parsed.verdict}.] {what}"
    return {"verdict": verdict, "confidence": parsed.confidence, "what_changed": what,
            "thesis_impact": parsed.thesis_impact, "watch_next": parsed.watch_next,
            "quotes": quotes, "n_sources": len(items), "model": jcfg.llm.model_id,
            "sources": [i.get("url") for i in items if i.get("url")][:5]}


# ── Messages ─────────────────────────────────────────────────────────────────

def heads_up_html(fund: str, ticker: str, name: str, report_date: str, theses: dict) -> str:
    when = datetime.fromisoformat(report_date).strftime("%a %d %b")
    lines = [f"📅 <b>{html.escape(fund)} — {html.escape(name)} ({html.escape(ticker)}) reports "
             f"{when}</b>"]
    for key, label in (("buy", "Bought on"), ("hold", "Held on")):
        if key in theses:
            lines.append(f"<i>{label} ({theses[key]['date']}):</i> {html.escape(theses[key]['thesis'])}")
    if not theses:
        lines.append("No thesis recorded for this position.")
    lines.append("A read of the report against this follows once it is out — before the next open.")
    return "\n".join(lines)


_ICON = {"held": "✅", "broken": "🔴", "unclear": "❔"}


def verdict_html(fund: str, ticker: str, name: str, report_date: str, result: dict,
                 shares: float | None = None) -> str:
    v = result["verdict"]
    lines = [f"{_ICON[v]} <b>{html.escape(name)} ({html.escape(ticker)}) — report {report_date}: "
             f"thesis {v.upper()}</b> (confidence {result.get('confidence') or 0:.0%})",
             f"<i>{html.escape(fund)}</i>"]
    if result.get("what_changed"):
        lines.append(f"\n<b>What it said:</b> {html.escape(result['what_changed'])}")
    if result.get("thesis_impact"):
        lines.append(f"<b>Against the thesis:</b> {html.escape(result['thesis_impact'])}")
    for q in result.get("quotes", [])[:3]:
        lines.append(f"  “{html.escape(q['quote'])}”")
    if result.get("watch_next"):
        lines.append(f"<b>Watch:</b> {html.escape(result['watch_next'])}")
    if v == "broken":
        lines.append(f"\nNothing is traded. To re-decide: /review {html.escape(ticker)}")
        if shares:
            lines.append(f"To sell it all: /montrose_ticket {html.escape(ticker)} sell {int(shares)}")
    return "\n".join(lines)


def gave_up_html(fund: str, ticker: str, name: str, report_date: str, attempts: int) -> str:
    return (f"❔ <b>{html.escape(name)} ({html.escape(ticker)}) — report {report_date}</b>\n"
            f"<i>{html.escape(fund)}</i>\nNo verdict after {attempts} attempt(s): the coverage "
            "found didn't contain the results, or there was none. Read the report yourself.")


# ── The job ──────────────────────────────────────────────────────────────────

def run(store, cfg, today: date | None = None, *, send: Callable[[str], bool],
        names: dict[str, str] | None = None, dry_run: bool = False,
        calendar_fn: Callable = _calendar_date, news_fn: Callable | None = None,
        coverage_fn: Callable | None = None, llm: Callable | None = None,
        echo: Callable[[str], None] = print) -> list[dict]:
    """One pass: discover, heads-up, judge. Returns what was sent."""
    today = today or datetime.now(timezone.utc).date()
    names = names or {}
    positions = {p.ticker: p for p in store.get_positions() if p.shares > 0}
    discover(store, sorted(positions), today, calendar_fn, news_fn)
    sent: list[dict] = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tomorrow = next_weekday(today)

    for row in store.get_earnings_checks(open_only=True):
        t, rd = row["ticker"], row["report_date"]
        rdate = date.fromisoformat(rd)
        name = names.get(t, t)
        if t not in positions:
            continue                     # sold since; nothing to read it against
        theses = store.get_position_theses(t)

        if rdate == tomorrow and not row["heads_up_at"]:
            echo(f"  📅 {t} reports {rd} — heads-up")
            if not dry_run:
                send(heads_up_html(cfg.display_name, t, name, rd, theses))
                store.upsert_earnings_check(t, rd, heads_up_at=now)
            sent.append({"ticker": t, "kind": "heads_up"})
            continue
        if rdate > today:
            continue

        attempts = row["attempts"] or 0
        too_old = (today - rdate).days > GIVE_UP_AFTER_DAYS
        if attempts >= cfg.earnings_watch.max_checks_per_report or too_old:
            echo(f"  ❔ {t} {rd}: no verdict after {attempts} attempt(s) — giving up")
            if not dry_run:
                send(gave_up_html(cfg.display_name, t, name, rd, attempts))
                store.upsert_earnings_check(t, rd, verdict="unclear", checked_at=now)
            sent.append({"ticker": t, "kind": "gave_up"})
            continue
        if dry_run:
            echo(f"  🔎 {t} {rd}: would read the coverage (attempt {attempts + 1})")
            continue

        try:
            result = judge(t, name, rd, theses, cfg, coverage_fn, llm)
        except Exception as exc:          # advisory: one failed read never stops the rest
            echo(f"  ⚠ {t} {rd}: {exc}")
            continue
        if result is None:
            echo(f"  … {t} {rd}: no coverage dated on/after the report yet")
            if (today - rdate).days >= 1:
                store.upsert_earnings_check(t, rd, attempts=attempts + 1)
            continue
        store.upsert_earnings_check(t, rd, attempts=attempts + 1)
        final = result["verdict"] != "unclear" or attempts + 1 >= cfg.earnings_watch.max_checks_per_report
        echo(f"  {_ICON[result['verdict']]} {t} {rd}: {result['verdict']}"
             + ("" if final else " — not final, asking again next slot"))
        if not final:
            continue
        # Last cached close (native currency): the reference a later scoring of
        # the verdict measures from.
        closes = store.get_prices(t, since_date=(today - timedelta(days=10)).isoformat())
        price = closes[-1]["close"] if closes else None
        send(verdict_html(cfg.display_name, t, name, rd, result, positions[t].shares))
        store.upsert_earnings_check(t, rd, verdict=result["verdict"],
                                    confidence=result.get("confidence"),
                                    detail=json.dumps(result, ensure_ascii=False),
                                    price_at_check=price, checked_at=now)
        sent.append({"ticker": t, "kind": "verdict", "verdict": result["verdict"]})
    return sent
