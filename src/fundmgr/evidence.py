"""
Evidence packs for the kill-criterion judge.

The judge used to see eight headlines and nothing else, which is fine for an
event criterion ("loses the Apple socket") and useless for a financial one
("recurring growth falls below 8% while EBITA/FCF deteriorates") — those facts
live in the report, not in a headline. Worse, it failed *silently*: the judge
answered NO because the numbers weren't there, which is indistinguishable from
NO because the thesis is intact.

This module assembles what the judge actually needs:

  • **News with substance** — headline, publisher, date and the article
    *summary* the yfinance payload already carries (the main fund's prompt has
    fed summaries to the LLM all along; only this path threw them away), plus a
    best-effort fetch of the article body for the top few items.
  • **Fundamentals with a trend** — the cached valuation/growth/margin figures
    and how they have moved since the oldest snapshot on file, so
    "deteriorates" is a measurable claim rather than a vibe.
  • **Position context** — cost basis and distance from it.

Every part degrades quietly: no network, no fundamentals cache and no article
host are all normal, and the pack simply says so. What it never does is let a
missing input look like a clean bill of health — `coverage` reports which parts
of the pack are populated so the judge can answer INSUFFICIENT.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone

from fundmgr.state.store import Store

# app_meta key holding the rolling fundamentals series per ticker.
SNAPSHOT_KEY = "paper_fundsnap"

# How many dated fundamentals snapshots to keep per ticker. Weekly-ish cadence
# from the daily track (a snapshot is only taken when the values actually
# change), so ~12 covers a year of trend — enough to see a margin roll over.
MAX_SNAPSHOTS = 12

# Article-body fetching is the one genuinely fragile part of the pack: third-
# party hosts, paywalls, redirects. Kept small, capped and off the critical
# path — set FUND_KILL_FETCH_ARTICLES=0 to disable it entirely.
MAX_ARTICLES = 4
ARTICLE_TIMEOUT = 6           # seconds per article
ARTICLE_CHARS = 2200          # per-article excerpt cap, keeps the prompt bounded
MAX_NEWS_ITEMS = 8

# Fundamentals worth putting in front of the judge, in reading order. Fractions
# from the provider are rendered as percentages.
_FUND_FIELDS: list[tuple[str, str, bool]] = [
    # (key, label, is_fraction)
    ("revenue_growth",   "Revenue growth (yoy)",   True),
    ("earnings_growth",  "Earnings growth (yoy)",  True),
    ("gross_margin",     "Gross margin",           True),
    ("operating_margin", "Operating margin",       True),
    ("ebitda_margin",    "EBITDA margin",          True),
    ("profit_margin",    "Profit margin",          True),
    ("roe",              "Return on equity",       True),
    ("roa",              "Return on assets",       True),
    ("free_cash_flow",   "Free cash flow",         False),
    ("operating_cash_flow", "Operating cash flow", False),
    ("total_debt",       "Total debt",             False),
    ("ev_to_ebitda",     "EV/EBITDA",              False),
    ("pe_ratio",         "P/E",                    False),
    ("forward_pe",       "Forward P/E",            False),
    ("price_to_sales",   "Price/sales",            False),
]

# A move smaller than this is noise, not a trend, and is rendered as "flat".
_TREND_EPSILON = 0.005

# Public view of the field table: {key: {label, is_fraction}}. The criterion
# analyser offers exactly these to the model, so a rule it proposes can only
# name a metric this codebase can actually read.
FUND_FIELD_META: dict[str, dict] = {
    key: {"label": label, "is_fraction": is_fraction}
    for key, label, is_fraction in _FUND_FIELDS
}


def current_metric(store: Store, ticker: str, metric: str) -> float | None:
    """Current value of one fundamentals metric, in the unit a rule compares in.

    Fractions from the provider (0.071) are returned as percent (7.1) so a rule
    written as "below 8" means 8%, the way it reads on the screen.
    """
    meta = FUND_FIELD_META.get(metric)
    if meta is None:
        return None
    value = _num((store.get_fundamentals(ticker) or {}).get(metric))
    if value is None:
        return None
    return value * 100 if meta["is_fraction"] else value


# ── News ──────────────────────────────────────────────────────────────────────

def news_items(ticker: str, max_items: int = MAX_NEWS_ITEMS) -> list[dict]:
    """Recent news for a ticker as {headline, publisher, published, summary, url}.

    Keeps the summary and link that `_recent_headlines` discarded. Tolerant of
    both yfinance news shapes (flat, and the newer nested `content` object).
    Returns [] on any failure — news being unavailable is not an error.
    """
    try:
        import yfinance as yf
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []

    out: list[dict] = []
    for item in raw[: max_items * 2]:
        if not isinstance(item, dict):
            continue
        content = item.get("content") if isinstance(item.get("content"), dict) else item
        headline = (content.get("title") or "").strip()
        if not headline:
            continue

        publisher = ""
        provider = content.get("provider")
        if isinstance(provider, dict):
            publisher = provider.get("displayName") or ""
        publisher = publisher or item.get("publisher") or ""

        summary = (content.get("summary") or content.get("description")
                   or item.get("summary") or "")

        url = ""
        for candidate in (content.get("canonicalUrl"), content.get("clickThroughUrl")):
            if isinstance(candidate, dict) and candidate.get("url"):
                url = candidate["url"]
                break
        url = url or item.get("link") or ""

        out.append({
            "headline": headline,
            "publisher": str(publisher).strip(),
            "published": str(content.get("pubDate") or item.get("providerPublishTime") or ""),
            "summary": re.sub(r"\s+", " ", str(summary)).strip(),
            "url": str(url),
            "body": "",
        })
        if len(out) >= max_items:
            break
    return out


_TAG_RE = re.compile(r"<[^>]+>")
_DROP_RE = re.compile(r"<(script|style|noscript)\b.*?</\1>", re.S | re.I)
_BLOCK_RE = re.compile(r"</(p|div|h[1-6]|li|br)\s*>", re.I)


def _html_to_text(html: str) -> str:
    """Crude but dependency-free HTML → text. Good enough for an article body."""
    html = _DROP_RE.sub(" ", html)
    html = _BLOCK_RE.sub("\n", html)
    text = _TAG_RE.sub(" ", html)
    for entity, char in (("&nbsp;", " "), ("&amp;", "&"), ("&quot;", '"'),
                         ("&#39;", "'"), ("&lt;", "<"), ("&gt;", ">")):
        text = text.replace(entity, char)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
    # Drop nav/boilerplate fragments; real article prose comes in sentences.
    return "\n".join(line for line in lines if len(line) > 60)


def fetch_article(url: str, timeout: int = ARTICLE_TIMEOUT) -> str:
    """Best-effort article body text. '' on anything less than a clean HTML 200."""
    if not url or not url.startswith(("http://", "https://")):
        return ""
    try:
        import requests
        resp = requests.get(
            url, timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ai-fund-manager/1.0)"},
        )
        if resp.status_code != 200:
            return ""
        if "html" not in resp.headers.get("Content-Type", "").lower():
            return ""
        return _html_to_text(resp.text)[:ARTICLE_CHARS]
    except Exception:
        return ""


def _articles_enabled() -> bool:
    return (os.getenv("FUND_KILL_FETCH_ARTICLES", "1") or "1").strip().lower() not in (
        "0", "false", "no", "off")


def enrich_with_bodies(items: list[dict], max_articles: int = MAX_ARTICLES) -> int:
    """Fetch article bodies in place for the items that need them most.

    Prioritises items whose summary is thin — a full summary already gives the
    judge something to read, so the network budget goes where it's missing.
    Returns how many bodies were actually filled.
    """
    if not items or not _articles_enabled():
        return 0
    ranked = sorted(items, key=lambda i: len(i.get("summary") or ""))
    filled = 0
    for item in ranked[:max_articles]:
        body = fetch_article(item.get("url", ""))
        if body:
            item["body"] = body
            filled += 1
    return filled


# ── Fundamentals + trend ──────────────────────────────────────────────────────

def _num(value):
    try:
        if value is None or isinstance(value, bool):
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out or out in (float("inf"), float("-inf")) else out


def snapshot_fundamentals(store: Store, tickers: list[str],
                          today: str | None = None) -> int:
    """Record today's fundamentals for each ticker so trends can be measured.

    Only writes when a value actually changed since the last snapshot, so a
    weekly-refreshed cache read daily doesn't fill the series with duplicates.
    Returns the number of tickers whose series grew.
    """
    import json

    today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    written = 0
    for ticker in tickers:
        data = store.get_fundamentals(ticker)
        if not data:
            continue
        current = {key: _num(data.get(key)) for key, _label, _frac in _FUND_FIELDS}
        current = {k: v for k, v in current.items() if v is not None}
        if not current:
            continue

        try:
            series = json.loads(store.get_meta(f"{SNAPSHOT_KEY}:{ticker}") or "[]")
        except json.JSONDecodeError:
            series = []
        if not isinstance(series, list):
            series = []

        if series and series[-1].get("values") == current:
            continue  # unchanged since the last snapshot — nothing new to record
        series.append({"date": today, "values": current})
        store.set_meta(f"{SNAPSHOT_KEY}:{ticker}", json.dumps(series[-MAX_SNAPSHOTS:]))
        written += 1
    return written


def _fmt(value: float, is_fraction: bool) -> str:
    if is_fraction:
        return f"{value * 100:+.1f}%" if abs(value) < 10 else f"{value:+.1f}%"
    if abs(value) >= 1e9:
        return f"{value / 1e9:.2f}bn"
    if abs(value) >= 1e6:
        return f"{value / 1e6:.1f}m"
    return f"{value:,.2f}"


def fundamentals_trend(store: Store, ticker: str,
                       today: str | None = None) -> list[dict]:
    """Current fundamentals with their move since the oldest snapshot on file.

    Returns [{key, label, current, earlier, earlier_date, direction, text}].
    Empty when nothing is cached — which the pack reports as missing coverage
    rather than as "nothing has deteriorated".

    A snapshot taken today is not a reference point: the daily track snapshots
    before the judge reads, so on the first run the only snapshot *is* today's.
    Comparing against it would report a confident "flat" where there is in fact
    no history, which is exactly the false reassurance this module exists to
    remove.
    """
    import json

    today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    data = store.get_fundamentals(ticker) or {}
    try:
        series = json.loads(store.get_meta(f"{SNAPSHOT_KEY}:{ticker}") or "[]")
    except json.JSONDecodeError:
        series = []
    oldest = series[0] if isinstance(series, list) and series else None
    if oldest and str(oldest.get("date", "")) >= today:
        oldest = None

    rows: list[dict] = []
    for key, label, is_fraction in _FUND_FIELDS:
        current = _num(data.get(key))
        if current is None:
            continue
        earlier = _num((oldest or {}).get("values", {}).get(key))
        direction, text = "", _fmt(current, is_fraction)
        if earlier is not None:
            delta = current - earlier
            scale = max(abs(earlier), _TREND_EPSILON)
            if abs(delta) / scale < 0.02:
                direction = "flat"
            else:
                direction = "up" if delta > 0 else "down"
            text += (f"  (was {_fmt(earlier, is_fraction)} on {oldest['date']} — "
                     f"{direction})")
        rows.append({
            "key": key, "label": label, "current": current,
            "earlier": earlier, "earlier_date": (oldest or {}).get("date"),
            "direction": direction, "text": text,
        })
    return rows


# ── Pack assembly ─────────────────────────────────────────────────────────────

def build_pack(store: Store, ticker: str, *, with_articles: bool = True,
               news_days: int = 21) -> dict:
    """Assemble everything the judge should read about one position.

    `coverage` is the honest part: it says which evidence classes are actually
    present, so a criterion that cannot be checked from what we have is
    reported as unverifiable instead of quietly passing.
    """
    items = news_items(ticker)
    bodies = enrich_with_bodies(items) if with_articles else 0

    # Cached, already-scored news from the store adds anything the live yfinance
    # call missed (and survives an outage on it).
    since = (datetime.now(timezone.utc) - timedelta(days=news_days)).strftime("%Y-%m-%d")
    try:
        cached = store.get_recent_news(ticker, since)
    except Exception:
        cached = []
    seen = {i["headline"] for i in items}
    for row in cached:
        headline = (row.get("headline") or "").strip()
        if not headline or headline in seen:
            continue
        seen.add(headline)
        items.append({
            "headline": headline,
            "publisher": "",
            "published": row.get("published_at") or "",
            "summary": re.sub(r"\s+", " ", str(row.get("summary") or "")).strip(),
            "url": "",
            "body": "",
            "sentiment": row.get("sentiment_label") or "",
        })

    fundamentals = fundamentals_trend(store, ticker)
    position = next((p for p in store.get_positions() if p.ticker == ticker), None)
    pos_block = None
    if position and position.avg_cost_sek:
        pos_block = {
            "shares": position.shares,
            "avg_cost_sek": position.avg_cost_sek,
            "current_price_sek": position.current_price_sek or None,
        }

    has_text = any(i.get("summary") or i.get("body") for i in items)
    has_trend = any(f["earlier"] is not None for f in fundamentals)
    return {
        "ticker": ticker,
        "news": items,
        "fundamentals": fundamentals,
        "position": pos_block,
        "coverage": {
            "headlines": bool(items),
            "article_text": has_text,
            "article_bodies": bodies,
            "fundamentals": bool(fundamentals),
            "fundamentals_trend": has_trend,
        },
    }


def render_pack(pack: dict) -> str:
    """Render an evidence pack as the prompt block the judge reads."""
    lines: list[str] = []

    lines.append("## Recent news")
    if pack["news"]:
        for item in pack["news"]:
            head = f"- {item['headline']}"
            meta = " · ".join(x for x in (item.get("publisher"), item.get("published")) if x)
            if meta:
                head += f"  [{meta}]"
            lines.append(head)
            if item.get("summary"):
                lines.append(f"    Summary: {item['summary'][:600]}")
            if item.get("body"):
                lines.append(f"    Article text: {item['body']}")
    else:
        lines.append("  (no recent news found)")

    lines.append("\n## Fundamentals" + (
        "" if pack["fundamentals"] else " — none cached for this ticker"))
    for row in pack["fundamentals"]:
        lines.append(f"- {row['label']}: {row['text']}")
    if pack["fundamentals"] and not pack["coverage"]["fundamentals_trend"]:
        lines.append("  (no earlier snapshot yet — direction of travel is unknown)")

    if pack["position"]:
        p = pack["position"]
        lines.append(f"\n## Position\n- {p['shares']:g} shares at "
                     f"{p['avg_cost_sek']:,.2f} SEK average cost")

    cov = pack["coverage"]
    lines.append(
        "\n## What this evidence covers\n"
        f"- Headlines: {'yes' if cov['headlines'] else 'no'}"
        f" · Article text: {'yes' if cov['article_text'] else 'no'}"
        f" · Fundamentals: {'yes' if cov['fundamentals'] else 'no'}"
        f" · Fundamentals trend: {'yes' if cov['fundamentals_trend'] else 'no'}"
    )
    return "\n".join(lines)
