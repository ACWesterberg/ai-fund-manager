"""
Global macro context for the decision prompt.

Indicator fetch and caching is delegated to financedata (6-hour SQLite TTL).
RSS headline fetching remains local (ephemeral, no shared-cache benefit).
"""
from __future__ import annotations

import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone

from financedata import (  # noqa: F401 — re-export for existing callers
    get_macro_indicators_cached as fetch_macro_indicators,
    build_macro_block,
    Indicator,
)


# ── Global news headlines (local RSS fetch, not cached in shared DB) ──────────

@dataclass
class Headline:
    source: str
    title: str
    published: str = ""


def _fetch_rss_headlines(url: str, max_age_hours: int = 48, max_items: int = 5) -> list[Headline]:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; fundmgr/1.0)"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read()
        root = ET.fromstring(raw)
    except Exception:
        return []

    channel = root.find("channel")
    source = "News"
    if channel is not None:
        title_el = channel.find("title")
        if title_el is not None and title_el.text:
            source = title_el.text.strip()[:30]

    items = root.findall(".//item")
    headlines: list[Headline] = []
    now = datetime.now(timezone.utc)

    for item in items:
        if len(headlines) >= max_items:
            break
        title_el = item.find("title")
        if title_el is None or not title_el.text:
            continue
        title = title_el.text.strip()

        pub_str = ""
        pub_el = item.find("pubDate")
        if pub_el is not None and pub_el.text:
            pub_str = pub_el.text.strip()
            try:
                from email.utils import parsedate_to_datetime
                pub_dt = parsedate_to_datetime(pub_str).astimezone(timezone.utc)
                age_h = (now - pub_dt).total_seconds() / 3600
                if age_h > max_age_hours:
                    continue
            except Exception:
                pass

        headlines.append(Headline(source=source, title=title, published=pub_str))

    return headlines


def fetch_macro_headlines(
    feeds: list[str],
    max_age_hours: int = 48,
    max_per_feed: int = 4,
) -> list[Headline]:
    all_headlines: list[Headline] = []
    for url in feeds:
        try:
            hl = _fetch_rss_headlines(url, max_age_hours=max_age_hours, max_items=max_per_feed)
            all_headlines.extend(hl)
        except Exception:
            pass
    return all_headlines
