"""
Report-reaction experiment: are some report-day drops overreactions?

The idea (EFN, Gabriel Isskander, Oct 2026): the market weighs the next quarter
too heavily, so a drop the report itself does not justify can be mispriced. The
research is less kind to the naive version. On average a report reaction keeps
going in the same direction for weeks (post-earnings-announcement drift), more so
in small, thinly covered markets like the Nordics, so buying every drop buys into
the drift. What evidence there is for reversal is about moves the news does not
support. The edge, if there is one, lives in the filter.

That filter cannot be backtested honestly — the model may know how a past
report played out — so this module splits the question in two:

  • `run_study`: free. Over past reports, what did report-day reactions of each
    size go on to do? (Do Nordic drops revert or drift?)
  • `run_watch`: forward only. Log each report that moves a stock, have the model
    judge the biggest drops against the coverage, then score every event at
    +5/+20/+28 trading days. `format_results` compares the verdicts.

Nothing here trades, sizes, or reaches the weekly prompt.
"""
from __future__ import annotations

import copy
import html
import json
import logging
import math
import random
import re
import time
from bisect import bisect_right
from datetime import datetime, timedelta, timezone
from statistics import mean, median
from typing import Callable

HORIZONS = (5, 20, 28)
FETCH_CHUNK = 100
MIN_TO_CONCLUDE = 20
MAX_SOURCE_CHARS = 30_000

BUCKETS = (
    (-math.inf, -10.0, "≤ −10%"),
    (-10.0, -5.0, "−10 … −5%"),
    (-5.0, 5.0, "−5 … +5%"),
    (5.0, 10.0, "+5 … +10%"),
    (10.0, math.inf, "≥ +10%"),
)

# Headline words that mark a report, for names Yahoo has no earnings date for.
REPORT_WORDS = re.compile(
    r"\b(delårsrapport|kvartalsrapport|bokslutskommuniké|halvårsrapport|"
    r"osavuosikatsaus|tilinpäätöstiedote|interim report|quarterly report|"
    r"half-year report|year-end report|full-year report|q[1-4]\b|"
    r"(first|second|third|fourth) quarter)",
    re.I,
)


# ── Price arithmetic ──────────────────────────────────────────────────────────

class _Bench:
    """Benchmark close on or before a date — the index trades on days a name may not."""

    def __init__(self, rows: list[dict]):
        self.dates = [r["date"] for r in rows]
        self.closes = [r["close"] for r in rows]

    def at(self, day: str) -> float | None:
        i = bisect_right(self.dates, day) - 1
        return self.closes[i] if i >= 0 and self.closes[i] else None


def _pct(a: float, b: float) -> float:
    return (b / a - 1) * 100


def abnormal(series: list[dict], bench: _Bench, i0: int, i1: int) -> tuple[float, float] | None:
    """(raw %, abnormal %) from close i0 to close i1: the stock less the index."""
    a, b = series[i0]["close"], series[i1]["close"]
    ba, bb = bench.at(series[i0]["date"]), bench.at(series[i1]["date"])
    if not a or not b or not ba or not bb:
        return None
    raw = _pct(a, b)
    return raw, raw - _pct(ba, bb)


def reaction_window(series: list[dict], report_date: str) -> tuple[int, int] | None:
    """Last close before the report and first close after it.

    Nordic reports land before the open and some elsewhere after the close, so
    "day −1 to day +1" catches the reaction either way, at the cost of one
    extra day of noise.
    """
    before = [i for i, r in enumerate(series) if r["date"] < report_date]
    after = [i for i, r in enumerate(series) if r["date"] > report_date]
    if not before or not after:
        return None
    return before[-1], after[0]


def forward(series: list[dict], bench: _Bench, entry: int) -> dict[int, float | None]:
    """Abnormal return from the entry close to each horizon, None where not yet there."""
    out: dict[int, float | None] = {}
    for n in HORIZONS:
        r = abnormal(series, bench, entry, entry + n) if entry + n < len(series) else None
        out[n] = round(r[1], 3) if r else None
    return out


def volume_ratio(series: list[dict], i0: int, i1: int) -> float | None:
    """Volume over the reaction against the 20 sessions before it."""
    during = [r["volume"] for r in series[i0 + 1:i1 + 1] if r.get("volume")]
    before = [r["volume"] for r in series[max(0, i0 - 19):i0 + 1] if r.get("volume")]
    if not during or len(before) < 5 or not mean(before):
        return None
    return round(mean(during) / mean(before), 2)


def bucket_of(abnormal_pct: float) -> str:
    for lo, hi, label in BUCKETS:
        if lo <= abnormal_pct < hi:
            return label
    return BUCKETS[-1][2]


def summarize(values: list[float], seed: int = 0) -> dict:
    """n, mean, median, share above zero and a 90% bootstrap interval of the mean."""
    if not values:
        return {"n": 0}
    rng = random.Random(seed)
    boots = sorted(mean(rng.choices(values, k=len(values))) for _ in range(2000))
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "pos_share": sum(v > 0 for v in values) / len(values),
        "ci": (boots[100], boots[1899]),
    }


# ── Report dates ──────────────────────────────────────────────────────────────

def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


class CalendarUnavailable(RuntimeError):
    """Yahoo's report history can't be read at all here — a setup problem, not coverage."""


def _quiet_yfinance() -> None:
    # One "possibly delisted" line per dead ticker buried the run's own output.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def earnings_history(ticker: str) -> list[dict]:
    """Past report dates with EPS estimate, reported EPS and surprise, from Yahoo.

    Coverage of Nordic small caps is patchy; a name Yahoo knows nothing about
    returns [] and is caught, if at all, by the headline fallback.

    A missing library is not "no coverage". yfinance needs lxml to read this
    table and raises ImportError per call; swallowing it made the first study
    on the Pi report 0/100 names covered when Yahoo had the dates all along.
    """
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).get_earnings_dates(limit=16)
    except ImportError as exc:
        raise CalendarUnavailable(
            f"Yahoo's report history can't be read: {exc} "
            "Redeploy (lxml is now a dependency) or run: .venv/bin/pip install lxml"
        ) from exc
    except Exception:
        return []
    if df is None or getattr(df, "empty", True):
        return []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = []
    for ts, row in df.iterrows():
        try:
            day = ts.strftime("%Y-%m-%d")
        except Exception:
            continue
        if day > today:
            continue
        out.append({
            "date": day,
            "eps_estimate": _num(row.get("EPS Estimate")),
            "eps_reported": _num(row.get("Reported EPS")),
            "eps_surprise_pct": _num(row.get("Surprise(%)")),
        })
    return sorted(out, key=lambda r: r["date"])


def _ensure_benchmark(store, cfg, days: int) -> _Bench:
    from fundmgr.data.benchmark import fetch_and_cache_benchmark
    since = (datetime.utcnow() - timedelta(days=days - 7)).strftime("%Y-%m-%d")
    rows = store.get_benchmark()
    force = not rows or rows[0]["date"] > since
    fetch_and_cache_benchmark(store, cfg.benchmark, days, force_refresh=force)
    return _Bench(store.get_benchmark())


# ── Step 1: the free base-rate study ──────────────────────────────────────────

def _elapsed(started: float) -> str:
    secs = int(time.monotonic() - started)
    return f"{secs // 60}m{secs % 60:02d}s"


def _fetch_history(store, tickers: list, days: int, echo: Callable[[str], None]) -> None:
    """Prices back `days` for every name, in chunks, saying how far it has got.

    One 1,605-ticker download printed nothing for as long as it took, which
    from the outside is indistinguishable from a hang, and is the request
    Yahoo is likeliest to throttle. The cache also skips any name already
    updated today however short its history, so a name the 90-day watch has
    touched would give the study one year instead of two: those are fetched
    again, forced.
    """
    from fundmgr.data.prices import fetch_and_cache_prices

    since = (datetime.utcnow() - timedelta(days=days - 14)).strftime("%Y-%m-%d")
    started = time.monotonic()
    for i in range(0, len(tickers), FETCH_CHUNK):
        chunk = tickers[i:i + FETCH_CHUNK]
        try:
            fetch_and_cache_prices(chunk, store, lookback_days=days)
            short = [t for t in chunk
                     if not (rows := store.get_prices(t.yahoo_ticker)) or rows[0]["date"] > since]
            if short:
                fetch_and_cache_prices(short, store, lookback_days=days, force_refresh=True)
        except Exception as exc:
            echo(f"  ⚠ prices for names {i + 1}–{i + len(chunk)} failed: {exc}")
        echo(f"  prices {min(i + FETCH_CHUNK, len(tickers))}/{len(tickers)} · {_elapsed(started)}")


def run_study(store, cfg, tickers: list, years: int = 2,
              history: Callable[[str], list[dict]] | None = None,
              echo: Callable[[str], None] = print) -> dict:
    """Every past report Yahoo dates for these names: reaction and what followed."""
    history = history or earnings_history
    _quiet_yfinance()

    days = 365 * years + 30
    echo(f"  Fetching {years}y of prices for {len(tickers)} names, {FETCH_CHUNK} at a time…")
    _fetch_history(store, tickers, days, echo)
    bench = _ensure_benchmark(store, cfg, days)

    echo("  Looking up report dates, one name at a time…")
    events, covered = [], 0
    started = time.monotonic()
    for n, t in enumerate(tickers, 1):
        reports = history(t.yahoo_ticker)
        if reports:
            covered += 1
        series = store.get_prices(t.yahoo_ticker)
        for rep in reports:
            win = reaction_window(series, rep["date"])
            if win is None:
                continue
            r = abnormal(series, bench, *win)
            if r is None:
                continue
            events.append({"ticker": t.yahoo_ticker, "report_date": rep["date"],
                           "abnormal_pct": r[1], "fwd": forward(series, bench, win[1]),
                           "eps_surprise_pct": rep.get("eps_surprise_pct")})
        if n % 50 == 0 or n == len(tickers):
            spent = time.monotonic() - started
            left = int(spent / n * (len(tickers) - n))
            echo(f"  reports {n}/{len(tickers)} names · {len(events)} found · "
                 f"{_elapsed(started)} elapsed, ~{left // 60}m left")
    return {"n_tickers": len(tickers), "covered": covered, "events": events}


def _fmt_row(label: str, s: dict) -> str:
    if not s.get("n"):
        return f"    {label:<14} n=0"
    lo, hi = s["ci"]
    return (f"    {label:<14} n={s['n']:<4} mean {s['mean']:+6.2f}%  median {s['median']:+6.2f}%  "
            f"up {s['pos_share'] * 100:3.0f}%  90% CI [{lo:+.2f}, {hi:+.2f}]")


def format_study(result: dict) -> str:
    cov = result["covered"] / result["n_tickers"] * 100 if result["n_tickers"] else 0.0
    lines = [
        f"\n─── Report reactions — base rates ({len(result['events'])} reports) ─────",
        f"  Yahoo dates reports for {result['covered']}/{result['n_tickers']} names "
        f"({cov:.0f}%). The rest are invisible to the calendar; the daily watch",
        "  falls back to report headlines for them.",
        "  Reaction = close before → close after the report, minus the index.",
        "  Then: abnormal return from that close, N trading days on.",
    ]
    for n in HORIZONS:
        lines.append(f"\n  +{n} trading days after the reaction:")
        for lo, hi, label in BUCKETS:
            vals = [e["fwd"][n] for e in result["events"]
                    if e["fwd"][n] is not None and lo <= e["abnormal_pct"] < hi]
            lines.append(_fmt_row(label, summarize(vals)))
    lines.append(
        "\n  Read: a negative mean after a drop is drift (it kept falling); positive is"
        "\n  reversal. An interval spanning zero means this history cannot tell."
    )
    return "\n".join(lines)


# ── Step 2: detect today's reactions ──────────────────────────────────────────

def _headline_report(ticker: str, after: str, upto: str, news_fn) -> bool:
    from fundmgr.data.release import _parse_date
    for item in news_fn(ticker) or []:
        when = _parse_date(item.get("published", ""))
        if when is not None and not (after < when.isoformat() <= upto):
            continue
        if REPORT_WORDS.search(item.get("headline", "")):
            return True
    return False


def detect_events(store, cfg, tickers: list, bench: _Bench, today: str | None = None,
                  history: Callable[[str], list[dict]] | None = None,
                  news_fn: Callable | None = None,
                  warn: Callable[[str], None] = print) -> list[dict]:
    """Reports from the previous session that moved a stock by the threshold.

    Price first, calendar second: only the handful of movers get a Yahoo lookup,
    which is what makes a 1,600-name daily sweep affordable.
    """
    from fundmgr import evidence
    news_fn = news_fn or evidence.news_items
    history = history or earnings_history
    today = today or datetime.utcnow().strftime("%Y-%m-%d")
    stale = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=4)).strftime("%Y-%m-%d")
    since = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")
    threshold = cfg.report_watch.move_threshold_pct
    calendar_down = False

    events = []
    for t in tickers:
        series = store.get_prices(t.yahoo_ticker, since_date=since)
        if len(series) < 3 or series[-1]["date"] < stale:
            continue
        i0, i1 = len(series) - 3, len(series) - 1
        report_day = series[-2]["date"]
        r = abnormal(series, bench, i0, i1)
        if r is None or abs(r[1]) < threshold:
            continue
        before = series[i0]["date"]
        reports: list[dict] = []
        if not calendar_down:
            try:
                reports = history(t.yahoo_ticker)
            except CalendarUnavailable as exc:
                calendar_down = True
                warn(f"  ⚠ {exc}\n    Falling back to report headlines for every name this run.")
        rep = next((h for h in reports if before < h["date"] <= report_day), None)
        if rep is not None:
            source = "calendar"
        elif _headline_report(t.yahoo_ticker, before, series[i1]["date"], news_fn):
            source, rep = "news", {}
        else:
            continue
        events.append({
            "ticker": t.yahoo_ticker, "name": t.name, "report_date": report_day,
            "event_date": series[i1]["date"], "reaction_pct": round(r[0], 3),
            "abnormal_pct": round(r[1], 3), "volume_ratio": volume_ratio(series, i0, i1),
            "source": source, "eps_estimate": rep.get("eps_estimate"),
            "eps_reported": rep.get("eps_reported"),
            "eps_surprise_pct": rep.get("eps_surprise_pct"),
            "run_up_60d": round(_pct(series[0]["close"], series[i0]["close"]), 1)
            if series[0]["close"] else None,
        })
    return events


# ── Step 2: judge the drops ───────────────────────────────────────────────────

_SYSTEM = """\
You are an equity analyst reviewing a Nordic company whose share price fell \
sharply on its quarterly report. Decide whether the report justifies the drop.

The market often weighs the next quarter too heavily and the next five years \
too little. A miss driven by currency, product mix, delayed deliveries, timing \
or one-off costs, at a company whose competitive position, capital allocation \
and earnings power are intact, may be an overreaction. A drop is justified when \
the report changes the long-term picture: guidance cut for real reasons, lost \
market share, structurally weaker margins, cash flow not following profit, \
balance-sheet stress, or a business model under threat (including from AI).

Ask, from the coverage given and nothing else:
- Does the company take or lose market share?
- Are earnings and cash flow growing over time — does profit become cash?
- Are margins improving or structurally weakening?
- Is management allocating capital well?
- Is the business model stronger or weaker than a year ago?
- Did the business disappoint, or did results merely miss high expectations?
- Where does the company stand in three to five years on what is known now?

Rules:
- Use only the coverage below. Do not rely on anything you remember about how \
this company or its share price did after this report.
- Quote sentences verbatim from the coverage. Quotes are checked; one that is \
not in the text is discarded, and a verdict with no surviving quote counts as \
unclear.
- If the coverage is thin, headline-only, or does not say what changed, answer \
unclear. That is a correct answer, not a failure."""


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def quote_in_source(quote: str, source: str) -> bool:
    q = _normalise(quote).strip(" .\"'“”‘’")
    return len(q) >= 20 and q in _normalise(source)


def _fmt(v, spec: str, unit: str = "") -> str:
    return "n/a" if v is None else f"{v:{spec}}{unit}"


def build_judge_prompt(event: dict, source: str) -> tuple[str, str]:
    user = f"""# {event.get('name') or event['ticker']} ({event['ticker']}) — report {event['report_date']}

## The reaction
Share price: {event['reaction_pct']:+.1f}% from the close before the report to the close after it.
Against the index: {event['abnormal_pct']:+.1f}%.
Volume over the reaction vs the 20 sessions before: {_fmt(event.get('volume_ratio'), '.1f', '×')}.
Move over the ~60 days before the report: {_fmt(event.get('run_up_60d'), '+.1f', '%')}.
EPS: estimate {_fmt(event.get('eps_estimate'), '.2f')}, reported {_fmt(event.get('eps_reported'), '.2f')}, surprise {_fmt(event.get('eps_surprise_pct'), '+.1f', '%')} (where the data provider carries it).

## Coverage of the report
{source}

## Your task
Is this drop justified by the report, an overreaction, or unclear from this coverage?
Return the ReportReaction JSON."""
    return _SYSTEM, user


def judge_event(event: dict, cfg, coverage_fn: Callable | None = None,
                llm: Callable | None = None) -> dict:
    """One model call per drop, its quotes checked against the coverage it was shown.

    No coverage, no call: the verdict is recorded as unclear for free, and does
    not count against the daily cap.
    """
    from fundmgr.data.release import _source_text, find_coverage
    from fundmgr.engine.client import call_llm
    from fundmgr.engine.schema import ReportReaction

    coverage_fn = coverage_fn or find_coverage
    llm = llm or call_llm
    items = coverage_fn(event["ticker"], event["report_date"])
    if not items:
        return {"verdict": "unclear", "what_changed": "No coverage of the report was found.",
                "n_sources": 0, "judge_model": ""}
    source = _source_text(items)[:MAX_SOURCE_CHARS]

    jcfg = copy.deepcopy(cfg)
    jcfg.llm.reasoning_effort = cfg.report_watch.reasoning_effort
    jcfg.llm.max_tokens = cfg.report_watch.max_tokens
    jcfg.llm.n_samples = 1
    system, user = build_judge_prompt(event, source)
    parsed, _raw = llm(system, user, jcfg, ReportReaction)

    quotes = [{"quote": q.quote, "shows": q.shows} for q in parsed.quotes
              if quote_in_source(q.quote, source)]
    verdict = parsed.verdict
    what = parsed.what_changed
    if verdict != "unclear" and not quotes:
        verdict = "unclear"
        what = f"[No quote survived the source check; verdict was {parsed.verdict}.] {what}"
    return {
        "verdict": verdict, "confidence": parsed.confidence, "what_changed": what,
        "temporary_factors": parsed.temporary_factors,
        "structural_factors": parsed.structural_factors,
        "expectations": parsed.expectations_vs_results, "quotes": quotes,
        "n_sources": len(items), "judge_model": jcfg.llm.model_id,
    }


# ── Step 3: mature and compare ────────────────────────────────────────────────

def mature(store, bench: _Bench) -> int:
    """Fill in the +5/+20/+28 abnormal returns that now exist. Free."""
    updated = 0
    for ev in store.get_report_events():
        if ev["matured_at"]:
            continue
        since = (datetime.strptime(ev["event_date"], "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
        series = store.get_prices(ev["ticker"], since_date=since)
        entry = next((i for i, r in enumerate(series) if r["date"] == ev["event_date"]), None)
        if entry is None:
            continue
        fwd = forward(series, bench, entry)
        if any(fwd[n] is not None and ev[f"fwd_{n}"] is None for n in HORIZONS):
            store.set_report_forward(ev["id"], fwd)
            updated += 1
    return updated


def format_results(events: list[dict], judge_threshold: float) -> str:
    drops = [e for e in events if e["abnormal_pct"] <= -judge_threshold]
    groups = [
        ("all drops", drops),
        ("overreaction", [e for e in drops if e["verdict"] == "overreaction"]),
        ("justified", [e for e in drops if e["verdict"] == "justified"]),
        ("unclear", [e for e in drops if e["verdict"] == "unclear"]),
        ("not judged", [e for e in drops if e["verdict"] is None]),
        ("rises", [e for e in events if e["abnormal_pct"] >= judge_threshold]),
    ]
    lines = [f"\n─── Report-reaction experiment — {len(events)} events logged ─────────"]
    for n in HORIZONS:
        lines.append(f"\n  +{n} trading days, abnormal vs the index:")
        for label, evs in groups:
            lines.append(_fmt_row(label, summarize([e[f"fwd_{n}"] for e in evs
                                                    if e[f"fwd_{n}"] is not None])))
    over = [e["fwd_28"] for e in groups[1][1] if e["fwd_28"] is not None]
    just = [e["fwd_28"] for e in groups[2][1] if e["fwd_28"] is not None]
    if min(len(over), len(just)) < MIN_TO_CONCLUDE:
        lines.append(f"\n  Too few matured verdicts to conclude anything yet "
                     f"({len(over)} overreaction, {len(just)} justified at +28; "
                     f"want {MIN_TO_CONCLUDE}+ each).")
    else:
        lines.append(f"\n  At +28: overreaction minus justified = "
                     f"{mean(over) - mean(just):+.2f} pp. The filter earns its keep only if "
                     "this is positive and the intervals above separate.")
    return "\n".join(lines)


def digest_html(fund_name: str, judged: list[dict]) -> str | None:
    """The day's overreaction calls for Telegram, or None when there are none."""
    flagged = [e for e in judged if e.get("verdict") == "overreaction"]
    if not flagged:
        return None
    others = len(judged) - len(flagged)
    lines = [f"<b>{html.escape(fund_name)}</b>",
             "🔬 <b>Report reactions — experiment, no action asked</b>",
             f"{len(flagged)} drop(s) judged an overreaction"
             + (f"; {others} other verdict(s) logged." if others else ".")]
    for e in flagged:
        lines.append("")
        lines.append(f"<b>{html.escape(e['ticker'])}</b> {e['abnormal_pct']:+.1f}% vs index "
                     f"(report {html.escape(e['report_date'])}) · conf {e.get('confidence') or 0:.2f}")
        if e.get("what_changed"):
            lines.append(html.escape(e["what_changed"]))
        quotes = e.get("quotes") or json.loads(e.get("quotes_json") or "[]")
        if quotes:
            lines.append(f"<i>“{html.escape(quotes[0]['quote'])}”</i>")
    lines.append("\nScored at +28 trading days; see <code>fund report-results</code>.")
    return "\n".join(lines)


# ── The daily job ─────────────────────────────────────────────────────────────

def run_watch(store, cfg, tickers: list, dry_run: bool = False,
              echo: Callable[[str], None] = print,
              judge: Callable[[dict], dict] | None = None) -> list[dict]:
    """Detect, log, judge (capped), mature. Returns the events judged this run."""
    from fundmgr.data.prices import fetch_and_cache_prices
    from fundmgr.engine.client import LLMError

    _quiet_yfinance()
    fetch_and_cache_prices(tickers, store, lookback_days=90)
    bench = _ensure_benchmark(store, cfg, 120)
    found = detect_events(store, cfg, tickers, bench, warn=echo)
    echo(f"  {len(found)} report reaction(s) of ±{cfg.report_watch.move_threshold_pct:.0f}% "
         "or more vs the index.")

    rw = cfg.report_watch
    today = datetime.utcnow().strftime("%Y-%m-%d")
    budget = max(0, rw.max_judged_per_day - store.count_report_verdicts_on(today))
    if dry_run:
        drops = sorted((e for e in found if e["abnormal_pct"] <= -rw.judge_threshold_pct),
                       key=lambda e: e["abnormal_pct"])
        for e in found:
            mark = " ← would judge" if e in drops[:budget] else ""
            echo(f"    {e['ticker']:<14} {e['abnormal_pct']:+6.1f}%  report {e['report_date']}"
                 f"  [{e['source']}]{mark}")
        return []

    by_key = {(e["ticker"], e["report_date"]): e for e in found}
    for e in found:
        store.add_report_event(e)
    recent = (datetime.utcnow() - timedelta(days=4)).strftime("%Y-%m-%d")
    pending = sorted(
        (ev for ev in store.get_report_events()
         if ev["verdict"] is None and ev["event_date"] >= recent
         and ev["abnormal_pct"] <= -rw.judge_threshold_pct),
        key=lambda ev: ev["abnormal_pct"],
    )
    judge = judge or (lambda ev: judge_event(ev, cfg))
    judged = []
    for ev in pending:
        if budget <= 0:
            break
        ev = {**by_key.get((ev["ticker"], ev["report_date"]), {}), **ev}
        try:
            verdict = judge(ev)
        except LLMError as exc:
            # Recorded, and counted against the cap: a failing call is not retried
            # tomorrow at full price.
            verdict = {"verdict": "unclear", "what_changed": f"Judge failed: {exc}",
                       "n_sources": 0, "judge_model": "error"}
        store.set_report_verdict(ev["id"], verdict)
        if verdict.get("judge_model"):
            budget -= 1
        judged.append({**ev, **verdict})
        echo(f"    {ev['ticker']:<14} {ev['abnormal_pct']:+6.1f}%  → {verdict['verdict']}")

    matured = mature(store, bench)
    if matured:
        echo(f"  Scored {matured} earlier event(s) at their new horizons.")
    return judged
