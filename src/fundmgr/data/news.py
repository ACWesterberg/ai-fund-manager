from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from financedata import get_news_cached, score_sentiment, score_and_save, get_cache as fd_cache_getter

from fundmgr.config import SentimentConfig, UniverseTicker
from fundmgr.state.store import Store

if TYPE_CHECKING:
    from fundmgr.data.prices import TickerFeatures


def fetch_news(
    feeds: list[str],
    tickers: list[UniverseTicker],
    max_age_hours: int = 72,
    ttl_hours: float = 6.0,
    use_fallback: bool = True,
    force_refresh: bool = False,
) -> dict[str, list[dict]]:
    """Fetch headlines matched to tickers via financedata's shared read-through cache.

    Only stale/missing tickers hit the network; the rest are served from the shared
    SQLite cache (so this fetch also warms the cache for DeepSwing's intraday scans,
    and vice-versa). When RSS/NewsAPI find nothing for a ticker, the fallback fills
    in per-ticker: Finnhub for US symbols (when FINNHUB_API_KEY is set) and yfinance
    for everything else — the universe is global, not Nordic-only. Set
    force_refresh=True for freshness-sensitive callers such as the news-trigger scan."""
    symbols = [t.yahoo_ticker for t in tickers]
    names = {t.yahoo_ticker: t.name for t in tickers}
    return get_news_cached(
        symbols,
        feeds=feeds,
        names=names,
        max_age_hours=max_age_hours,
        ttl_hours=ttl_hours,
        market=None,  # global universe → infer US (Finnhub) vs non-US per ticker
        use_fallback=use_fallback,
        force_refresh=force_refresh,
    )


def score_and_cache_sentiment(
    ticker_news: dict[str, list[dict]],
    store: Store,
    model: str = "ProsusAI/finbert",
    device: str = "cpu",
) -> None:
    """Score headlines via financedata FinBERT and mirror to fund's store."""
    score_and_save(ticker_news, model=model, device=device)

    # Mirror scored articles to fund's store for attach_sentiment_to_features
    fd_cache = fd_cache_getter()
    now = datetime.utcnow().strftime("%Y-%m-%d")
    for ticker in ticker_news:
        rows = fd_cache.get_news(ticker, since_date=now)
        if rows:
            store.save_news_sentiment(ticker, rows)


def check_news_triggers(
    feeds: list[str],
    tickers: list[UniverseTicker],
    held_tickers: set[str],
    store: Store,
    cfg: SentimentConfig,
    max_age_hours: int = 8,
) -> list[dict]:
    """
    Fetch recent headlines, score with FinBERT, and return trigger events.

    A trigger fires when:
      - A held ticker gets a negative score >= trigger_threshold_negative, OR
      - Any ticker gets a positive score >= trigger_threshold_positive
    and the same article hasn't triggered before (deduplicated by hash)
    and the cooldown window hasn't elapsed for that ticker.
    """
    if not cfg.enabled:
        return []

    # Triggers must reflect the very latest headlines, so bypass the cache and skip
    # the per-ticker yfinance fallback (too costly across the whole universe here).
    ticker_news = fetch_news(
        feeds, tickers, max_age_hours=max_age_hours, use_fallback=False, force_refresh=True
    )
    if not ticker_news:
        return []

    all_items: list[tuple[str, dict]] = [
        (ticker, item)
        for ticker, items in ticker_news.items()
        for item in items
    ]
    if not all_items:
        return []

    headlines = [item["headline"] for _, item in all_items]
    scores = score_sentiment(headlines, model=cfg.model, device=cfg.device)
    scored = [(ticker, item, s) for (ticker, item), s in zip(all_items, scores)]

    triggers: list[dict] = []
    now = datetime.utcnow()
    cooldown_cutoff = (now - timedelta(hours=cfg.trigger_cooldown_hours)).isoformat()

    for ticker, item, score in scored:
        label = score["label"]
        val = score["score"]
        is_held = ticker in held_tickers

        if is_held and label == "negative" and val >= cfg.trigger_threshold_negative:
            pass
        elif label == "positive" and val >= cfg.trigger_threshold_positive:
            pass
        else:
            continue

        article_hash = hashlib.sha1(f"{ticker}:{item['headline']}".encode()).hexdigest()
        if store.has_triggered(article_hash):
            continue

        last = store.last_trigger_at(ticker)
        if last and last >= cooldown_cutoff:
            continue

        store.record_trigger(ticker, item["headline"], label, val, article_hash)
        triggers.append({
            "ticker": ticker,
            "headline": item["headline"],
            "sentiment_label": label,
            "sentiment_score": val,
            "is_held": is_held,
        })

    return triggers


def attach_sentiment_to_features(
    features: dict[str, "TickerFeatures"],
    store: Store,
    since_date: str,
) -> None:
    """Pull cached sentiment for each ticker and attach to its TickerFeatures in-place."""
    for ticker, feat in features.items():
        rows = store.get_recent_news(ticker, since_date=since_date)
        if not rows:
            continue
        feat.news_count = len(rows)
        score_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
        scores = [
            r["sentiment_score"] * score_map.get(r["sentiment_label"], 0.0)
            for r in rows
            if r["sentiment_score"] is not None
        ]
        if scores:
            avg = sum(scores) / len(scores)
            feat.sentiment_label = "positive" if avg > 0.1 else "negative" if avg < -0.1 else "neutral"
            feat.sentiment_score = round(abs(avg), 3)
