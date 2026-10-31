"""Press releases from MFN, read from each company's own feed, matched by ISIN.

Yahoo's news feed had nothing on Invisio's 30 Oct report while MFN carried the
release itself, tagged `sub:report:interim`, so this is the text a local
pre-screen reads. Each company has a feed at `mfn.se/a/<slug>.rss`; the slug
comes from MFN's company search, which lists every ISIN a company has, so a
universe row resolves only on an exact ISIN match and is never guessed from
its name. MFN answers an unknown slug with an empty feed rather than an error,
which is why a guessed slug cannot be checked.

The all-companies feed only holds the newest 48 releases market-wide. The one
in config.yaml (`mfn.se/a/all.rss`) returns no items at all; `mfn.se/all/a.rss`
is the live one.

Advisory throughout: a name that cannot be resolved or read is left out.
"""
from __future__ import annotations

import html
import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx

SEARCH_URL = "https://mfn.se/search/companies"
FEED_URL = "https://mfn.se/a/{slug}.rss"
FEED_ITEMS = 15
RETRY_MISS_DAYS = 30
SUMMARY_CHARS = 600
_NS = {"x": "https://mfn.se/schemas/rss-ns-x/"}
_SUFFIXES = re.compile(
    r"\b(ab|asa|oyj|abp|a/s|as|plc|ltd|inc|se|nv|sa|holding|group|publ|series [a-z]|"
    r"class [a-z]|ser\. [a-z])\b|[().,]", re.I)


@dataclass(frozen=True)
class Release:
    title: str
    link: str
    published: datetime          # UTC
    language: str
    tags: tuple[str, ...]
    summary: str

    @property
    def regulatory(self) -> bool:
        return ":regulatory" in self.tags

    @property
    def report(self) -> bool:
        return any(t.startswith("sub:report") for t in self.tags)


def _client() -> httpx.Client:
    return httpx.Client(timeout=30, follow_redirects=True,
                        headers={"User-Agent": "fundmgr prescreen"})


def _queries(name: str) -> list[str]:
    """Search terms for a universe name, most specific first.

    MFN matches the start of its own short name, so "H & M Hennes & Mauritz AB"
    finds nothing until it is "H&M", and "Investment AB Latour" only as
    "Latour". Each answer is checked by ISIN, so a loose term cannot mismatch.
    """
    cleaned = " ".join(_SUFFIXES.sub(" ", name).split())
    words = cleaned.split()
    squeezed = cleaned.replace(" & ", "&").split(" ")[0] if cleaned else ""
    distinctive = sorted({w for w in words[1:] if len(w) >= 4 and w.isalpha()},
                         key=len, reverse=True)[:2]
    candidates = [cleaned, squeezed, words[0] if words else "", *distinctive]
    return [q for q in dict.fromkeys(candidates) if len(q) >= 2][:5]


def resolve_slug(name: str, isin: str, client: httpx.Client) -> str | None:
    """MFN's slug for the company whose ISINs include `isin`, else None."""
    isin = isin.strip().upper()
    for query in _queries(name):
        resp = client.get(SEARCH_URL, params={"query": query, "limit": 10})
        resp.raise_for_status()
        for hit in resp.json() or []:
            if isin in {i.upper() for i in hit.get("isins") or []} and hit.get("slug"):
                return hit["slug"]
    return None


class SlugCache:
    """ISIN → slug, with misses retried after a month (a newly listed name
    joins MFN later than the universe)."""

    def __init__(self, path: Path):
        self.path = path
        try:
            self.data = json.loads(path.read_text())
        except (OSError, ValueError):
            self.data = {}

    def get(self, isin: str) -> tuple[bool, str | None]:
        entry = self.data.get(isin)
        if not entry:
            return False, None
        if entry.get("slug"):
            return True, entry["slug"]
        checked = datetime.fromisoformat(entry["checked"])
        return datetime.now(timezone.utc) - checked < timedelta(days=RETRY_MISS_DAYS), None

    def put(self, isin: str, slug: str | None) -> None:
        self.data[isin] = {"slug": slug, "checked": datetime.now(timezone.utc).isoformat()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1, sort_keys=True))


def _text(markup: str | None) -> str:
    plain = html.unescape(re.sub(r"<[^>]+>", " ", markup or ""))
    return " ".join(plain.split())[:SUMMARY_CHARS]


def parse_feed(raw: bytes) -> list[Release]:
    """Releases in a feed, English kept where a release comes in two languages."""
    root = ET.fromstring(raw)
    releases = []
    for item in root.findall("./channel/item"):
        try:
            published = parsedate_to_datetime(item.findtext("pubDate") or "").astimezone(timezone.utc)
        except (TypeError, ValueError):
            continue
        releases.append(Release(
            title=" ".join((item.findtext("title") or "").split()),
            link=(item.findtext("link") or "").strip(),
            published=published,
            language=(item.findtext("x:language", namespaces=_NS) or "").strip(),
            tags=tuple(t.text.strip() for t in item.findall("x:tag", _NS) if t.text),
            summary=_text(item.findtext("description")),
        ))
    english = {(r.published, r.tags) for r in releases if r.language == "en"}
    return [r for r in releases if r.language == "en" or (r.published, r.tags) not in english]


def releases(rows: list, since: datetime, cache_path: Path, *,
             client: httpx.Client | None = None, pause: float = 0.3) -> dict[str, list[Release]]:
    """{yahoo ticker: releases published since `since`, newest first} for rows with an ISIN."""
    own = client is None
    client = client or _client()
    cache = SlugCache(cache_path)
    out: dict[str, list[Release]] = {}
    try:
        for row in rows:
            isin = (row.isin or "").strip().upper()
            if not isin:
                continue
            try:
                known, slug = cache.get(isin)
                if not known:
                    slug = resolve_slug(row.name, isin, client)
                    cache.put(isin, slug)
                if not slug:
                    continue
                resp = client.get(FEED_URL.format(slug=slug),
                                  params={"limit": FEED_ITEMS, "compact": "true"})
                resp.raise_for_status()
                recent = [r for r in parse_feed(resp.content) if r.published >= since]
                if recent:
                    out[row.yahoo_ticker] = sorted(recent, key=lambda r: r.published, reverse=True)
            except (httpx.HTTPError, ET.ParseError, ValueError, KeyError):
                continue
            if pause:
                time.sleep(pause)
    finally:
        cache.save()
        if own:
            client.close()
    return out
