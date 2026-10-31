"""A local model reads the screener's candidates' news; a fixed rule re-ranks them.

The mechanical screener ranks on numbers and is blind to why a stock moved: a
report, a large order or a share placement all look like the same volume
spike. A local model (on a machine on the tailnet, behind an OpenAI-compatible
server) reads each candidate's recent headlines and MFN releases and answers
in tags — is there material news, what kind, which way — and does not rank or
recommend. The rule that turns tags into a selection is fixed and visible
(`prescreen.weights` added to the mechanical score), so every week's choice
can be recomputed from its record, and each tag's worth can be measured
separately rather than trusting a black-box filter.

It runs in shadow first: the decision model keeps getting the mechanical list
and the record keeps both, so the two sets can be scored before the
pre-screen is trusted. It can't be backtested honestly — a model may know how
past news played out — so the evidence has to come forward.

Held and pinned names are always kept: the fund must be able to sell what it
owns. Anything that fails falls back to the mechanical list (advisory, fail
open), and the record says so.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator

from fundmgr.data.screener import _score
from fundmgr.engine.client import LLMError, call_local_json

RECORD_VERSION = 1
MAX_HEADLINES = 6
MAX_RELEASES = 5

NewsType = Literal["report", "guidance", "deal", "order", "capital_raise", "management",
                   "legal", "index_or_flow", "other", "none"]


class PrescreenTag(BaseModel):
    ticker: str
    material_news: bool = Field(description="Something happened that could change what the company is worth")
    news_type: NewsType = Field(description="Kind of the most material item; none when nothing is material")
    direction: Literal["positive", "negative", "mixed", "none"] = Field(
        description="For the company's value; none when nothing is material")
    note: str = Field(default="", max_length=200, description="One short sentence naming the item")

    @field_validator("ticker")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()


class PrescreenBatch(BaseModel):
    tags: list[PrescreenTag]


SYSTEM = """\
You tag news for a Nordic equity fund's candidate list. You do not rank stocks \
and you do not recommend trades.

For each ticker, read only the headlines and press releases given under it and answer:
- material_news: did something happen that could change what the company is worth \
— a report, guidance, a deal or large order, a share issue, a management change, \
a legal ruling? Routine notices (AGM and nomination-committee notices, buyback \
updates, financial calendars) are not material.
- news_type: the kind of the most material item. Use "index_or_flow" for index \
changes, block sales or placements by holders and short-selling news — trading \
that says nothing about the business. Use "none" when nothing is material.
- direction: for the company's value; "none" when nothing is material.
- note: one short sentence in English naming the item.

Use only the text given. A ticker with no text gets material_news false, \
news_type "none", direction "none". Return exactly one entry per ticker given.\
"""


@dataclass
class PrescreenResult:
    status: str                              # ok | partial | unavailable
    selected: list[str]
    tags: dict[str, PrescreenTag] = field(default_factory=dict)
    calls: list[dict] = field(default_factory=list)
    error: str = ""


def _fmt_date(value) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    return str(value or "")[:10]


def candidate_block(feat, releases: list, since: datetime) -> str:
    """One candidate's text for the local model: headlines and MFN releases, dated."""
    lines = [f"## {feat.ticker} — {feat.name}" + (f" ({feat.sector})" if feat.sector else "")]
    cutoff = since.date().isoformat()
    for h in (feat.headlines or [])[:MAX_HEADLINES]:
        if _fmt_date(h.get("published_at")) and _fmt_date(h.get("published_at")) < cutoff:
            continue
        summary = " ".join((h.get("summary") or "").split())[:200]
        lines.append(f"- {_fmt_date(h.get('published_at'))} news: {h['headline']}"
                     + (f" — {summary}" if summary else ""))
    for r in releases[:MAX_RELEASES]:
        kind = ", ".join(k for k, on in (("regulatory", r.regulatory), ("report", r.report)) if on)
        lines.append(f"- {_fmt_date(r.published)} press release{f' ({kind})' if kind else ''}: "
                     f"{r.title} — {r.summary[:300]}")
    if len(lines) == 1:
        lines.append("(no news in the window)")
    return "\n".join(lines)


def tag_weight(tag: PrescreenTag | None, weights: dict[str, float]) -> float:
    if tag is None or not tag.material_news:
        return 0.0
    if tag.news_type == "index_or_flow":
        return weights.get("index_or_flow", 0.0)
    return weights.get(f"material_{tag.direction}", 0.0)


def select(scores: dict[str, float], tags: dict[str, PrescreenTag], weights: dict[str, float],
           top_n: int, forced: set[str]) -> list[str]:
    """Held and pinned first, then the pool by score plus tag weight; ties by ticker."""
    adjusted = {t: s + tag_weight(tags.get(t), weights) for t, s in scores.items()}
    chosen = sorted(t for t in forced if t in scores)
    for t in sorted(adjusted, key=lambda t: (-adjusted[t], t)):
        if len(chosen) >= top_n:
            break
        if t not in chosen:
            chosen.append(t)
    return chosen


def run_prescreen(pool: dict, mechanical: list[str], forced: set[str], pcfg, top_n: int,
                  releases: dict[str, list] | None = None, *,
                  call: Callable | None = None, now: datetime | None = None) -> PrescreenResult:
    """Tag the pool in batches and re-rank it; the mechanical list on total failure."""
    call = call or call_local_json
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(days=pcfg.news_days)
    releases = releases or {}
    scores = {t: _score(f) for t, f in pool.items()}
    order = sorted(pool, key=lambda t: (-scores[t], t))
    tags: dict[str, PrescreenTag] = {}
    calls: list[dict] = []
    for i in range(0, len(order), max(1, pcfg.batch_size)):
        batch = order[i:i + pcfg.batch_size]
        user = "\n\n".join(candidate_block(pool[t], releases.get(t, []), since) for t in batch)
        started = time.monotonic()
        entry = {"tickers": batch, "user": user}
        try:
            parsed, raw = call(SYSTEM, user, pcfg, PrescreenBatch)
            wanted = set(batch)
            got = {tag.ticker: tag for tag in parsed.tags if tag.ticker in wanted}
            tags.update(got)
            entry.update(raw=raw, missing=sorted(wanted - got.keys()))
        except LLMError as exc:
            entry["error"] = str(exc)[:2000]
        entry["latency_s"] = round(time.monotonic() - started, 2)
        calls.append(entry)

    failed = [c for c in calls if c.get("error")]
    if calls and len(failed) == len(calls):
        return PrescreenResult("unavailable", list(mechanical), {}, calls,
                               error=failed[0]["error"].split("\n")[0])
    selected = select(scores, tags, pcfg.weights, top_n, forced)
    return PrescreenResult("partial" if failed else "ok", selected, tags, calls)


def record_dir(cfg) -> Path:
    return cfg.db_path.parent / "prescreen" / cfg.db_path.stem


def write_record(cfg, run_id: str, mode: str, result: PrescreenResult, pool: dict,
                 mechanical: list[str], forced: set[str], note: str = "") -> Path:
    """Everything needed to score this week's two lists later and replay the selection."""
    pcfg = cfg.prescreen
    payload = {
        "version": RECORD_VERSION, "run_id": run_id, "fund": cfg.display_name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": mode, "status": result.status, "error": result.error, "note": note,
        "model": {"host": urlsplit(pcfg.base_url).netloc, "model_id": pcfg.model_id,
                  "temperature": pcfg.temperature, "seed": pcfg.seed, "json_mode": pcfg.json_mode},
        "weights": pcfg.weights, "top_n": cfg.screener.top_n, "news_days": pcfg.news_days,
        "pool": [{"ticker": t, "score": _score(f), "last_price": f.last_price, "last_date": f.last_date}
                 for t, f in pool.items()],
        "forced": sorted(forced), "mechanical": list(mechanical), "selected": result.selected,
        "tags": {t: tag.model_dump() for t, tag in sorted(result.tags.items())},
        "system": SYSTEM, "calls": result.calls,
    }
    path = record_dir(cfg) / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    return path
