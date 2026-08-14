"""
Reading a company's reported figures out of its own release coverage.

Half the ADD criteria in a real policy are written on measures no provider
carries — organic growth, ARR, adjusted EBITA, cash conversion, order backlog —
because they are company-defined rather than accounting standards. `evidence`
can store them once someone types them in; this module finds them.

The pieces already existed: the earnings calendar knows *when* a report lands,
`evidence.news_items` pulls what was published around it, and
`evidence.fetch_article` fetches the body. What was missing was the step that
turns that prose into the figures a criterion names.

Two rules make the difference between a useful assistant and a plausible liar:

  • **A figure not stated is absent, never inferred.** No computing one metric
    from another, no converting, no "approximately". A missing figure is a
    correct answer and leaves the criterion honestly unread.
  • **Every figure carries the sentence it came from, and the number must
    actually appear in the source text.** That check runs here, not in the
    prompt, because a model asked not to hallucinate can still hallucinate; a
    value absent from the text it claims to quote is dropped outright.

Nothing is written to the book from here. The output is a review table, and the
investor confirms before `evidence.record_reported` stores anything — a number
that gates money should not enter on a model's say-so.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone

from fundmgr.state.store import Store

DEFAULT_READ_MODEL = "gpt-4o-mini"

# How far either side of the reporting date coverage is still about that report.
# Wide enough for a weekend and a slow follow-up, narrow enough that the
# previous quarter's write-ups don't drift in.
WINDOW_DAYS = 10

# Articles fetched per read. The release itself is usually first; beyond a
# handful it is commentary re-quoting the same numbers.
MAX_ARTICLES = 5

_SYSTEM = """\
You read coverage of a listed company's quarterly report and extract ONLY the \
figures explicitly stated in the text.

You are given a list of metrics the investor tracks. For each one, find the \
number the company reported for the period in question — and if it is not \
stated, leave it out. A missing figure is a correct and useful answer.

Rules, in order of importance:
- NEVER infer, compute, derive or convert a figure. If the text gives revenue
  and you could calculate a margin, do not: that margin is not a reported
  figure. If the text gives a figure in EUR and the metric is in SEK, leave it
  out rather than converting.
- NEVER estimate or round to a neat number. Copy the digits as printed.
- Quote, verbatim, the sentence the number came from. The quote is checked
  against the source text, and a figure whose number does not appear there is
  discarded — so a guess is worse than an omission.
- If a figure is stated for a different period than the one asked about (a
  full-year number when the question is about a quarter, or a year-ago
  comparative), leave it out and say so in `notes`.
- Percentages as printed: "6.4 per cent" is 6.4, not 0.064.

Reply with a JSON object only:
{"figures": [
   {"metric": "<key from the list>",
    "value": <number as printed>,
    "period": "<the period this figure is for, as the text describes it>",
    "quote": "<the sentence, verbatim from the source>",
    "confidence": 0.0-1.0}
 ],
 "notes": "<anything ambiguous, or ''>"}\
"""


def find_coverage(ticker: str, period_end: str = "",
                  max_articles: int = MAX_ARTICLES) -> list[dict]:
    """Published items about a ticker's report, with bodies fetched.

    Filtered to a window around `period_end` when the item carries a usable
    date; undated items are kept, since dropping them would silently discard
    the release itself on feeds that omit timestamps.
    """
    from fundmgr import evidence

    items = evidence.news_items(ticker, max_items=evidence.MAX_NEWS_ITEMS)
    if not items:
        return []

    target = _parse_date(period_end)
    if target:
        items = [i for i in items if _within_window(i.get("published"), target)]
    items = items[:max_articles]
    try:
        evidence.enrich_with_bodies(items, max_articles=max_articles)
    except Exception:
        pass
    return [i for i in items if (i.get("body") or i.get("summary"))]


def _parse_date(value: str):
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime((value or "").strip()[:len(fmt) + 6], fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def _within_window(published: str, target) -> bool:
    when = _parse_date(published)
    if when is None:
        return True          # undated: keep rather than silently drop
    return abs((when - target).days) <= WINDOW_DAYS


def _source_text(items: list[dict]) -> str:
    parts = []
    for item in items:
        head = item.get("headline", "")
        body = (item.get("body") or item.get("summary") or "").strip()
        parts.append(f"[{item.get('publisher', '?')}, {item.get('published', '?')}] "
                     f"{head}\n{body}")
    return "\n\n---\n\n".join(parts)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).lower()


def _number_appears(value: float, text: str) -> bool:
    """Is this number actually in the source, in any of its printed forms?

    The anti-fabrication check that matters. A model told not to invent figures
    can still invent one; a figure whose value is nowhere in the text it claims
    to quote did not come from that text.
    """
    haystack = _normalise(text).replace(" ", " ")
    candidates = set()
    for raw in (f"{value:g}", f"{value:.1f}", f"{value:.2f}"):
        candidates.add(raw)
        candidates.add(raw.replace(".", ","))          # Nordic decimal comma
    if float(value).is_integer():
        candidates.add(f"{int(value)}")
    # Thousands separators, for the big ones (ARR, backlog).
    if abs(value) >= 1000:
        whole = f"{int(round(value)):,}"
        candidates.update({whole, whole.replace(",", " "), whole.replace(",", ".")})
    return any(c.lower() in haystack for c in candidates if c)


def extract(store: Store, ticker: str, items: list[dict], period_end: str = "",
            model: str | None = None) -> dict:
    """Pull the book's tracked metrics out of fetched coverage.

    Returns {figures, notes, rejected, sources}. `figures` are the ones whose
    number was found in the source text; `rejected` are those the model
    returned that failed that check, kept so the failure is visible rather than
    silent.
    """
    from fundmgr.evidence import field_meta

    known = {k: v for k, v in field_meta(store).items() if v.get("custom")}
    if not known:
        return {"figures": [], "notes": "no book-specific metrics defined",
                "rejected": [], "sources": []}
    if not items or not os.getenv("OPENAI_API_KEY"):
        return {"figures": [], "notes": "", "rejected": [], "sources": []}

    text = _source_text(items)
    catalogue = "\n".join(
        f"  {key} — {meta['label']}"
        + (f" ({meta['unit']})" if meta.get("unit") else "")
        + (f" — {meta['note']}" if meta.get("note") else "")
        for key, meta in known.items())

    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        kwargs = {
            "model": model or os.getenv("FUND_READ_MODEL", DEFAULT_READ_MODEL),
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": (
                    f"Company: {ticker}\n"
                    f"Reporting period ending: {period_end or 'the latest quarter'}\n\n"
                    f"Metrics tracked:\n{catalogue}\n\n"
                    f"Coverage:\n{text}")},
            ],
            "max_tokens": 900,
            "temperature": 0.0,
        }
        try:
            resp = client.chat.completions.create(
                response_format={"type": "json_object"}, **kwargs)
        except Exception:
            resp = client.chat.completions.create(**kwargs)
        raw = (resp.choices[0].message.content or "").strip()
    except Exception:
        return {"figures": [], "notes": "extraction call failed",
                "rejected": [], "sources": []}

    return _parse(raw, known, text, items)


def _parse(raw: str, known: dict, text: str, items: list[dict]) -> dict:
    from fundmgr.paper import extract_json_object

    data = None
    for candidate in (raw, extract_json_object(raw or "")):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(data, dict) or not isinstance(data.get("figures"), list):
        return {"figures": [], "notes": "unreadable reply", "rejected": [],
                "sources": _sources(items)}

    figures, rejected = [], []
    for row in data["figures"]:
        if not isinstance(row, dict):
            continue
        metric = str(row.get("metric") or "").strip()
        if metric not in known:
            continue
        try:
            value = float(str(row.get("value")).replace(",", "."))
        except (TypeError, ValueError):
            continue
        entry = {
            "metric": metric,
            "label": known[metric]["label"],
            "unit": known[metric].get("unit", ""),
            "value": value,
            "period": str(row.get("period") or "").strip(),
            "quote": str(row.get("quote") or "").strip(),
            "confidence": _confidence(row.get("confidence")),
        }
        if not _number_appears(value, text):
            entry["reason"] = "the figure does not appear in the source text"
            rejected.append(entry)
            continue
        entry["quote_verbatim"] = _normalise(entry["quote"]) in _normalise(text)
        figures.append(entry)

    return {"figures": figures, "rejected": rejected,
            "notes": str(data.get("notes") or "").strip(),
            "sources": _sources(items)}


def _confidence(value) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _sources(items: list[dict]) -> list[dict]:
    return [{"headline": i.get("headline", ""), "publisher": i.get("publisher", ""),
             "published": i.get("published", ""), "url": i.get("url", "")}
            for i in items]


def read_report(store: Store, ticker: str, period_end: str = "",
                model: str | None = None) -> dict:
    """Find coverage of a ticker's report and extract the tracked figures.

    The whole Route-A path in one call: locate, fetch, extract, verify. Writes
    nothing — the caller shows the result and asks for confirmation.
    """
    items = find_coverage(ticker, period_end)
    if not items:
        return {"figures": [], "rejected": [], "sources": [],
                "notes": "no coverage found in the window around that date"}
    return extract(store, ticker, items, period_end, model)
