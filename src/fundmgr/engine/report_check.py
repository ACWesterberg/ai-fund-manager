"""
Report check — read a held company's quarterly report against the thesis it is
held on, and say whether to act.

Before this, a holding's report reached the fund only sideways: `check-news`
scores headlines (a flat "X reports Q3 sales of SEK 1.2bn" over weak numbers
reads neutral), `check-stops` waits for the price to break the stop, and the
weekly run sees it the following Monday. `report-watch` reads reports but is a
universe-wide experiment that never asks about the book. Nothing read the report
of a company the real fund owns, on the day, and asked whether it broke the
reason it was bought.

The flow, per holding that has just reported:

  • find the report: Yahoo's earnings calendar, falling back to report headlines;
  • fetch the coverage (`release.find_coverage`) — no coverage, no call;
  • N-sample consensus at the fund's own model and effort, because the answer
    can move real money, against the logged theses on the name;
  • every quote is checked against the coverage; a sample whose verdict rests on
    no surviving sentence is counted as unclear/hold, so an invented quote cannot
    vote for an exit;
  • the verdict is filed like a stop or target review (dashboard + scoring), and
    each report is reviewed once.

Advisory only, like every review: nothing here trades.
"""
from __future__ import annotations

import html
from datetime import datetime, timedelta, timezone
from typing import Callable

from fundmgr.config import AppConfig, load_universe
from fundmgr.engine.review_common import (
    context_blocks,
    live_price_sek,
    log_review,
    majority,
    run_consensus,
    technicals_block,
    todo_line,
    votes_str,
)
from fundmgr.engine.schema import ReportCheck
from fundmgr.state.store import Store

SOURCE = "report_review"

# A report this recent counts as new. Covers a weekend plus a missed run.
LOOKBACK_DAYS = 4
# Coverage of a Nordic small cap can lag the release by a day. Until this many
# days have passed, a report with no coverage is simply tried again next run.
COVERAGE_GRACE_DAYS = 2
# `report-check TICKER` looks this far back for the report to read.
ON_DEMAND_DAYS = 30


def _meta_key(ticker: str, report_date: str) -> str:
    return f"report_check:{ticker}:{report_date}"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def latest_report(ticker: str, today: str, days: int,
                  history: Callable[[str], list[dict]] | None = None,
                  news_fn: Callable | None = None,
                  warn: Callable[[str], None] = print) -> dict | None:
    """The most recent report within `days` of `today`, or None.

    Calendar first (it carries EPS against the estimate); report headlines when
    Yahoo has no date, which is common for Nordic small caps.
    """
    from fundmgr import evidence
    from fundmgr.data.release import _parse_date
    from fundmgr.engine.report_events import (
        REPORT_WORDS,
        CalendarUnavailable,
        earnings_history,
    )

    history = history or earnings_history
    news_fn = news_fn or evidence.news_items
    since = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")

    try:
        reports = history(ticker)
    except CalendarUnavailable as exc:
        warn(f"  ⚠ {exc}")
        reports = []
    dated = [r for r in reports if since < r["date"] <= today]
    if dated:
        return {**dated[-1], "source": "calendar"}

    newest = None
    for item in news_fn(ticker) or []:
        when = _parse_date(item.get("published", ""))
        if when is None or not REPORT_WORDS.search(item.get("headline", "")):
            continue
        day = when.isoformat()
        if since < day <= today and (newest is None or day > newest):
            newest = day
    return {"date": newest, "source": "news"} if newest else None


def build_prompt(ticker: str, report: dict, source_text: str, store: Store,
                 cfg: AppConfig) -> tuple[str, str, float | None] | None:
    """(system, user, native price), or None when the ticker is not held."""
    pos = next((p for p in store.get_positions() if p.ticker == ticker), None)
    if pos is None:
        return None
    uni = {t.yahoo_ticker: t for t in load_universe(cfg.universe_path)}
    meta = uni.get(ticker)
    name = meta.name if meta else ticker
    currency = meta.currency if meta else "SEK"

    technicals, tech_live = technicals_block(ticker)
    live = live_price_sek(ticker, pos, cfg, store, currency, None, tech_live)
    chg = (live / pos.avg_cost_sek - 1) * 100 if pos.avg_cost_sek else 0.0
    decisions_block, news_block = context_blocks(ticker, store)

    def fmt(v, spec: str, unit: str = "") -> str:
        return "n/a" if v is None else f"{v:{spec}}{unit}"

    system = (
        "You are the risk manager for a real-money Nordic equity fund. A company the "
        "fund holds has just published its quarterly report. Judge it against the "
        "thesis the position is held on: does the report leave that thesis standing, "
        "dent it, or break it? A miss on currency, timing, mix or one-off costs at a "
        "business whose position and earnings power are intact is not a broken thesis; "
        "a real guidance cut, lost market share, structurally weaker margins, profit "
        "not turning into cash or balance-sheet stress may be. Be decisive, but do not "
        "act on noise. This is advisory — a human executes.\n\n"
        "Rules:\n"
        "- Use only the coverage below for what the report said. Do not rely on "
        "anything you remember about this company's results or share price.\n"
        "- Quote sentences verbatim from the coverage. Quotes are checked; one that is "
        "not in the text is discarded, and a verdict with no surviving quote counts as "
        "unclear and hold.\n"
        "- If the coverage is thin or headline-only, answer unclear and hold. That is "
        "a correct answer, not a failure."
    )

    user = f"""# Report Check — {ticker} ({name}), report {report['date']}

## Position
Shares: {pos.shares:.0f}  |  Avg cost: {pos.avg_cost_sek:.2f}  |  Live: {live:.2f}
Change since entry: {chg:+.1f}%   |  Market value: {pos.shares * live:,.0f} SEK

## The thesis — most recent decisions on this name
{decisions_block}

## The report
EPS: estimate {fmt(report.get('eps_estimate'), '.2f')}, reported {fmt(report.get('eps_reported'), '.2f')}, surprise {fmt(report.get('eps_surprise_pct'), '+.1f', '%')} (where the data provider carries it).

### Coverage
{source_text}

## Current technicals
{technicals}

## Recent news / sentiment (14d)
{news_block}

## Your task
Is the thesis intact, strengthened, weakened, broken, or unclear from this coverage?
Then choose EXIT (sell all), TRIM (reduce), HOLD, or ADD. Return the ReportCheck JSON."""
    return system, user, tech_live


def _verified(sample: ReportCheck, source_text: str) -> ReportCheck:
    """Keep only quotes that are in the coverage; with none, the sample is unclear/hold."""
    from fundmgr.engine.report_events import quote_in_source

    quotes = [q for q in sample.quotes if quote_in_source(q.quote, source_text)]
    if quotes:
        return sample.model_copy(update={"quotes": quotes})
    if sample.thesis_status == "unclear" and sample.recommendation == "hold":
        return sample.model_copy(update={"quotes": []})
    return sample.model_copy(update={
        "quotes": [], "thesis_status": "unclear", "recommendation": "hold", "trim_pct": None,
        "what_changed": f"[No quote survived the source check; it said {sample.thesis_status}/"
                        f"{sample.recommendation}.] {sample.what_changed}"[:400],
    })


def vote(samples: list[ReportCheck]) -> tuple[ReportCheck, dict[str, int]]:
    """Majority recommendation; status, confidence and quotes from the samples behind it."""
    winner, agreeing, counts = majority(samples)
    best = max(agreeing, key=lambda s: s.confidence)
    status, _ = max(((s, sum(a.thesis_status == s for a in agreeing))
                     for s in {a.thesis_status for a in agreeing}), key=lambda kv: kv[1])
    return best.model_copy(update={
        "recommendation": winner,
        "thesis_status": status,
        "confidence": round(sum(s.confidence for s in agreeing) / len(agreeing), 3),
    }), counts


def dated_since(items: list[dict], report_date: str) -> list[dict]:
    """Coverage published on or after the report day.

    `find_coverage` keeps ±10 days and undated items so a report's own release
    isn't dropped by a feed without timestamps — but on the morning of the report
    that window is mostly the week's "X will report on Friday" previews, which
    would be read as the results. Undated items go too, for the same reason.
    """
    from fundmgr.data.release import _parse_date

    start = datetime.strptime(report_date, "%Y-%m-%d").date()
    return [i for i in items or []
            if (when := _parse_date(i.get("published", ""))) is not None and when >= start]


def review_report(ticker: str, report: dict, store: Store, cfg: AppConfig,
                  coverage_fn: Callable | None = None,
                  consensus: Callable | None = None) -> dict | None:
    """Review one report. None when there is no coverage to read (nothing is called)."""
    from fundmgr.data.release import _source_text, find_coverage
    from fundmgr.engine.report_events import MAX_SOURCE_CHARS

    coverage_fn = coverage_fn or find_coverage
    consensus = consensus or run_consensus
    items = coverage_fn(ticker, report["date"])
    if report.get("source") != "on demand":
        items = dated_since(items, report["date"])
    if not items:
        return None
    source_text = _source_text(items)[:MAX_SOURCE_CHARS]
    built = build_prompt(ticker, report, source_text, store, cfg)
    if built is None:
        return None
    system, user, native_price = built

    samples = [_verified(s, source_text)
               for s in consensus(system, user, cfg, ReportCheck, "report-check")]
    verdict, votes = vote(samples)
    n = max(1, cfg.llm.n_samples)
    review_id = log_review(verdict, store, SOURCE, native_price, votes=votes, n_samples=n)
    store.set_meta(_meta_key(ticker, report["date"]), review_id)
    return {"review": verdict, "votes": votes, "n": n, "report": report,
            "n_sources": len(items), "review_id": review_id}


def run_check(store: Store, cfg: AppConfig, only: str | None = None,
              today: str | None = None, dry_run: bool = False,
              history: Callable | None = None, news_fn: Callable | None = None,
              coverage_fn: Callable | None = None, consensus: Callable | None = None,
              echo: Callable[[str], None] = print) -> dict:
    """Check every holding (or just `only`) for a report not yet reviewed.

    Returns {"reviewed": [...], "no_coverage": [...], "failed": [...]}.
    `only` re-reads that holding's latest report even if it was reviewed before.
    """
    from fundmgr.engine.client import LLMError

    today = today or _today()
    held = [p.ticker for p in store.get_positions()]
    tickers = [only] if only else held
    days = ON_DEMAND_DAYS if only else LOOKBACK_DAYS
    out: dict[str, list] = {"reviewed": [], "no_coverage": [], "failed": []}

    for tk in tickers:
        report = latest_report(tk, today, days, history=history, news_fn=news_fn, warn=echo)
        if report is None:
            if not only:
                continue
            # Asked about by name with no dated report: read whatever is recent.
            report = {"date": today, "source": "on demand"}
        key = _meta_key(tk, report["date"])
        if not only and store.get_meta(key):
            continue
        echo(f"  {tk}: report {report['date']} [{report['source']}]")
        if dry_run:
            continue
        try:
            result = review_report(tk, report, store, cfg,
                                   coverage_fn=coverage_fn, consensus=consensus)
        except LLMError as exc:
            # Not marked as done: the next run tries again rather than losing it.
            echo(f"    review failed: {exc}")
            out["failed"].append({"ticker": tk, "report": report, "error": str(exc)})
            continue
        if result is None:
            age = (datetime.strptime(today, "%Y-%m-%d")
                   - datetime.strptime(report["date"], "%Y-%m-%d")).days
            if only or age >= COVERAGE_GRACE_DAYS:
                if not only:
                    store.set_meta(key, "no-coverage")
                out["no_coverage"].append({"ticker": tk, "report": report})
                echo("    no coverage found — read it yourself")
            else:
                echo("    no coverage yet — will try again next run")
            continue
        r = result["review"]
        echo(f"    thesis {r.thesis_status} → {r.recommendation.upper()} "
             f"({votes_str(result['votes'], result['n'])})")
        out["reviewed"].append(result)
    return out


_STATUS_EMOJI = {"broken": "🔴", "weakened": "🟠", "intact": "🟢",
                 "strengthened": "🟢", "unclear": "⚪"}


def format_text(result: dict) -> str:
    r = result["review"]
    trim = f" {r.trim_pct:.0f}%" if r.recommendation == "trim" and r.trim_pct else ""
    lines = [
        f"{_STATUS_EMOJI.get(r.thesis_status, '')} REPORT CHECK {r.ticker} "
        f"(report {result['report']['date']}): thesis {r.thesis_status.upper()} → "
        f"{r.recommendation.upper()}{trim}  (conf {r.confidence:.2f}; "
        f"consensus {votes_str(result['votes'], result['n'])})",
        f"  Do: {todo_line(r.recommendation, r.trim_pct)}",
        f"  What changed: {r.what_changed}",
        f"  Rationale: {r.rationale}",
    ]
    lines += [f"  “{q.quote}”" for q in r.quotes[:2]]
    return "\n".join(lines)


def format_html(out: dict) -> str | None:
    """One Telegram message for the run, or None when there is nothing to say."""
    blocks = []
    for result in out["reviewed"]:
        r = result["review"]
        trim = f" {r.trim_pct:.0f}%" if r.recommendation == "trim" and r.trim_pct else ""
        block = [
            f"{_STATUS_EMOJI.get(r.thesis_status, '')} <b>{html.escape(r.ticker)}</b> "
            f"(report {html.escape(result['report']['date'])}): thesis "
            f"<b>{html.escape(r.thesis_status)}</b> → <b>{r.recommendation.upper()}{trim}</b> "
            f"<i>(conf {r.confidence:.2f}; {votes_str(result['votes'], result['n'])})</i>",
            f"<i>Do:</i> {html.escape(todo_line(r.recommendation, r.trim_pct))}",
            f"<i>Changed:</i> {html.escape(r.what_changed)}",
            f"<i>Why:</i> {html.escape(r.rationale)}",
        ]
        if r.quotes:
            block.append(f"<i>“{html.escape(r.quotes[0].quote)}”</i>")
        blocks.append("\n".join(block))
    for miss in out["no_coverage"]:
        blocks.append(f"⚪ <b>{html.escape(miss['ticker'])}</b> reported "
                      f"{html.escape(miss['report']['date'])}, but no coverage was found to "
                      "read — check the report yourself.")
    for fail in out["failed"]:
        blocks.append(f"⚠ <b>{html.escape(fail['ticker'])}</b>: report check failed "
                      f"({html.escape(fail['error'][:200])}); it will be retried.")
    if not blocks:
        return None
    return ("<b>📑 Report check — held positions</b>\n"
            "Advisory; you place any order yourself.\n\n" + "\n\n".join(blocks))
