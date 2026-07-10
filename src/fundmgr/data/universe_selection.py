"""Select which universe tickers to actively work each run.

Large global universes (10k+ names) cannot refresh every ticker every week.
We always refresh held + pinned names, rotate through the rest in stable
weekly buckets, and still build features from anything already in the price
cache so prior weeks' discoveries remain tradable.
"""
from __future__ import annotations

import hashlib
from datetime import datetime

from fundmgr.config import ScreenerConfig, UniverseTicker
from fundmgr.state.store import Store


def _stable_bucket(ticker: str, n: int) -> int:
    digest = hashlib.md5(ticker.encode()).hexdigest()
    return int(digest, 16) % n


def select_tickers_for_price_fetch(
    tickers: list[UniverseTicker],
    held: set[str],
    cfg: ScreenerConfig,
) -> tuple[list[UniverseTicker], str]:
    """Pick tickers whose prices should be refreshed this run."""
    if not tickers:
        return [], "empty"

    limit = cfg.price_fetch_limit
    if limit is None or len(tickers) <= limit:
        return tickers, "full"

    by_yahoo = {t.yahoo_ticker: t for t in tickers}
    must = (held | set(cfg.pinned_tickers)) & by_yahoo.keys()
    selected: list[UniverseTicker] = [by_yahoo[y] for y in sorted(must)]

    if len(selected) >= limit:
        return selected[:limit], f"capped at {limit} (held+pinned)"

    rotate_weeks = max(1, cfg.rotate_weeks)
    bucket = datetime.utcnow().isocalendar().week % rotate_weeks

    for t in tickers:
        if t.yahoo_ticker in must:
            continue
        if _stable_bucket(t.yahoo_ticker, rotate_weeks) != bucket:
            continue
        selected.append(t)
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        for t in tickers:
            if t.yahoo_ticker in must:
                continue
            if t in selected:
                continue
            selected.append(t)
            if len(selected) >= limit:
                break

    return selected, f"bucket {bucket + 1}/{rotate_weeks}, limit {limit}"


def tickers_for_feature_build(
    universe: list[UniverseTicker],
    fetch_batch: list[UniverseTicker],
    store: Store,
) -> list[UniverseTicker]:
    """Universe rows that have (or just received) price history in the fund store."""
    symbols = [t.yahoo_ticker for t in universe]
    cached = store.tickers_with_price_cache(symbols)
    cached.update(t.yahoo_ticker for t in fetch_batch)
    by_yahoo = {t.yahoo_ticker: t for t in universe}
    return [by_yahoo[y] for y in sorted(cached) if y in by_yahoo]


def news_watch_tickers(
    universe: list[UniverseTicker],
    held: set[str],
    cfg: ScreenerConfig,
) -> list[UniverseTicker]:
    """Tickers scanned by check-news — held positions plus pinned watch names."""
    watch = held | set(cfg.pinned_tickers)
    return [t for t in universe if t.yahoo_ticker in watch]
