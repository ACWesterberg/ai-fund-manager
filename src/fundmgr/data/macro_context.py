"""
Global macro context for the decision prompt.

Fetches live market data (indices, commodities, FX, rates) via yfinance
and recent global news headlines via RSS. No extra API keys required.
"""
from __future__ import annotations

import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import yfinance as yf


# ── Macro indicators ──────────────────────────────────────────────────────────

@dataclass
class Indicator:
    label: str
    category: str          # index | commodity | fx | rate
    unit: str = ""
    price: Optional[float] = None
    change_5d_pct: Optional[float] = None


_INDICATOR_DEFS: list[tuple[str, str, str, str]] = [
    # (yahoo_symbol, label, category, unit)
    ("^GSPC",    "S&P 500",        "index",     ""),
    ("^IXIC",    "Nasdaq",         "index",     ""),
    ("^GDAXI",   "DAX",            "index",     ""),
    ("^OMXSPI",  "OMXSPI",         "index",     ""),
    ("BZ=F",     "Brent crude",    "commodity", "$/bbl"),
    ("GC=F",     "Gold",           "commodity", "$/oz"),
    ("HG=F",     "Copper",         "commodity", "$/lb"),
    ("EURSEK=X", "EUR/SEK",        "fx",        ""),
    ("USDSEK=X", "USD/SEK",        "fx",        ""),
    ("NOKSEK=X", "NOK/SEK",        "fx",        ""),
    ("^TNX",     "US 10Y yield",   "rate",      "%"),
]


def fetch_macro_indicators(timeout: int = 10) -> list[Indicator]:
    """Fetch current prices and 5-day changes for key macro indicators."""
    symbols = [s for s, *_ in _INDICATOR_DEFS]
    result: list[Indicator] = []

    try:
        data = yf.download(
            symbols,
            period="10d",      # a bit extra in case of market holidays
            progress=False,
            auto_adjust=True,
            timeout=timeout,
        )
        closes = data["Close"] if "Close" in data.columns else data
    except Exception:
        return result

    for symbol, label, category, unit in _INDICATOR_DEFS:
        ind = Indicator(label=label, category=category, unit=unit)
        try:
            series = closes[symbol].dropna()
            if len(series) >= 2:
                ind.price = float(series.iloc[-1])
                # 5-day or as many days as available
                lookback = min(5, len(series) - 1)
                base = float(series.iloc[-1 - lookback])
                if base and base != 0:
                    ind.change_5d_pct = (ind.price - base) / base * 100
        except Exception:
            pass
        result.append(ind)

    return result


# ── Global news headlines ─────────────────────────────────────────────────────

@dataclass
class Headline:
    source: str
    title: str
    published: str = ""


def _fetch_rss_headlines(url: str, max_age_hours: int = 48, max_items: int = 5) -> list[Headline]:
    """Fetch up to max_items recent headlines from an RSS feed."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; fundmgr/1.0)"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read()
        root = ET.fromstring(raw)
    except Exception:
        return []

    # Derive source name from feed title or URL
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
            # Basic age filter
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
    """Fetch headlines from all configured global macro news feeds."""
    all_headlines: list[Headline] = []
    for url in feeds:
        try:
            hl = _fetch_rss_headlines(url, max_age_hours=max_age_hours, max_items=max_per_feed)
            all_headlines.extend(hl)
        except Exception:
            pass
    return all_headlines


# ── Prompt block ──────────────────────────────────────────────────────────────

def build_macro_block(
    indicators: list[Indicator],
    headlines: list[Headline],
) -> str:
    if not indicators and not headlines:
        return ""

    lines = [f"## Global Macro Context  (fetched {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})"]

    # Group indicators by category
    categories = [
        ("index",     "Equity indices"),
        ("commodity", "Commodities"),
        ("fx",        "FX vs SEK"),
        ("rate",      "Rates"),
    ]

    for cat_key, cat_label in categories:
        cat_inds = [i for i in indicators if i.category == cat_key and i.price is not None]
        if not cat_inds:
            continue
        parts = []
        for ind in cat_inds:
            chg = f" ({ind.change_5d_pct:+.1f}% 5d)" if ind.change_5d_pct is not None else ""
            unit = f" {ind.unit}" if ind.unit else ""
            parts.append(f"{ind.label} {ind.price:.2f}{unit}{chg}")
        lines.append(f"  {cat_label}: {' | '.join(parts)}")

    if headlines:
        lines.append("")
        lines.append("  Recent global headlines:")
        for h in headlines[:12]:
            lines.append(f"    [{h.source[:20]}] {h.title}")

    lines.append(
        "\n  Use this context to inform your macro overlay. "
        "Flag relevant macro risks or tailwinds in market_summary."
    )

    return "\n".join(lines)
