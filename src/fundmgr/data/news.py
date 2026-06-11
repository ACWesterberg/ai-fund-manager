from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import feedparser

from fundmgr.config import SentimentConfig, UniverseTicker
from fundmgr.state.store import Store

if TYPE_CHECKING:
    from fundmgr.data.prices import TickerFeatures

# ── FinBERT (lazy-loaded so the model isn't required to start the app) ────────

_finbert_pipeline = None


def _get_finbert(model: str = "ProsusAI/finbert", device: str = "cpu"):
    global _finbert_pipeline
    if _finbert_pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline
            _finbert_pipeline = hf_pipeline(
                "sentiment-analysis",
                model=model,
                device=device,
                truncation=True,
                max_length=512,
            )
        except Exception as e:
            print(f"  FinBERT unavailable ({e}), falling back to neutral sentiment.")
            _finbert_pipeline = None
    return _finbert_pipeline


def _score_sentiment(texts: list[str], model: str, device: str) -> list[dict]:
    """Returns list of {label, score} for each text. Falls back to neutral on failure."""
    if not texts:
        return []
    pipe = _get_finbert(model, device)
    if pipe is None:
        return [{"label": "neutral", "score": 0.5}] * len(texts)
    try:
        results = pipe(texts, batch_size=16)
        # FinBERT labels are "positive", "negative", "neutral" (lowercase)
        return [{"label": r["label"].lower(), "score": round(float(r["score"]), 4)} for r in results]
    except Exception as e:
        print(f"  Sentiment scoring error: {e}")
        return [{"label": "neutral", "score": 0.5}] * len(texts)


def _build_keyword_map(tickers: list[UniverseTicker]) -> dict[str, list[str]]:
    """Map yahoo_ticker -> list of keywords (name parts and ticker stem) to match in headlines."""
    kmap: dict[str, list[str]] = {}
    for t in tickers:
        kws = []
        # Strip exchange suffix for stem matching (e.g. VOLV-B.ST -> VOLV-B, VOLV)
        stem = t.yahoo_ticker.rsplit(".", 1)[0]
        kws.append(stem.lower())
        kws.append(stem.split("-")[0].lower())
        # Add significant name words (≥5 chars)
        for word in t.name.split():
            word = re.sub(r"[^a-z0-9]", "", word.lower())
            if len(word) >= 5:
                kws.append(word)
        kmap[t.yahoo_ticker] = list(set(kws))
    return kmap


def _match_ticker(text: str, keyword_map: dict[str, list[str]]) -> list[str]:
    """Return list of tickers whose keywords appear in the text."""
    text_lower = text.lower()
    matched = []
    for ticker, kws in keyword_map.items():
        if any(kw in text_lower for kw in kws):
            matched.append(ticker)
    return matched


def fetch_news(
    feeds: list[str],
    tickers: list[UniverseTicker],
    max_age_hours: int = 72,
) -> dict[str, list[dict]]:
    """
    Pull RSS feeds and match headlines to tickers.
    Returns dict: ticker -> list of {headline, source_url, published_at}.
    """
    keyword_map = _build_keyword_map(tickers)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

    ticker_news: dict[str, list[dict]] = {t.yahoo_ticker: [] for t in tickers}

    for feed_url in feeds:
        try:
            feed = feedparser.parse(feed_url)
        except Exception as e:
            print(f"  Feed error ({feed_url}): {e}")
            continue

        for entry in feed.entries:
            title = getattr(entry, "title", "") or ""
            summary = getattr(entry, "summary", "") or ""
            text = f"{title} {summary}"
            link = getattr(entry, "link", "") or ""

            # Parse publish date
            published_str = None
            published_dt = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                published_str = published_dt.strftime("%Y-%m-%d %H:%M")
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                published_dt = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
                published_str = published_dt.strftime("%Y-%m-%d %H:%M")

            if published_dt and published_dt < cutoff:
                continue  # too old

            matched = _match_ticker(text, keyword_map)
            for ticker in matched:
                ticker_news[ticker].append({
                    "headline": title[:500],
                    "source_url": link[:500],
                    "published_at": published_str,
                })

    return {k: v for k, v in ticker_news.items() if v}


def score_and_cache_sentiment(
    ticker_news: dict[str, list[dict]],
    store: Store,
    model: str = "ProsusAI/finbert",
    device: str = "cpu",
) -> None:
    """Score all fetched headlines with FinBERT and persist to news_cache."""
    for ticker, items in ticker_news.items():
        if not items:
            continue
        headlines = [item["headline"] for item in items]
        scores = _score_sentiment(headlines, model, device)
        enriched = [
            {**item, "sentiment_label": s["label"], "sentiment_score": s["score"]}
            for item, s in zip(items, scores)
        ]
        store.save_news_sentiment(ticker, enriched)


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

    Returns list of dicts: {ticker, headline, sentiment_label, sentiment_score, is_held}.
    """
    if not cfg.enabled:
        return []

    ticker_news = fetch_news(feeds, tickers, max_age_hours=max_age_hours)
    if not ticker_news:
        return []

    # Score all fetched headlines
    all_items: list[tuple[str, dict]] = []
    for ticker, items in ticker_news.items():
        for item in items:
            all_items.append((ticker, item))

    if not all_items:
        return []

    headlines = [item["headline"] for _, item in all_items]
    scores = _score_sentiment(headlines, cfg.model, cfg.device)
    scored = [(ticker, item, s) for (ticker, item), s in zip(all_items, scores)]

    triggers: list[dict] = []
    now = datetime.utcnow()
    cooldown_cutoff = (now - timedelta(hours=cfg.trigger_cooldown_hours)).isoformat()

    for ticker, item, score in scored:
        label = score["label"]
        val = score["score"]
        is_held = ticker in held_tickers

        # Decide whether this article clears the threshold
        if is_held and label == "negative" and val >= cfg.trigger_threshold_negative:
            pass  # qualifies
        elif label == "positive" and val >= cfg.trigger_threshold_positive:
            pass  # qualifies
        else:
            continue

        # Deduplication: skip if this exact article already fired a trigger
        article_hash = hashlib.sha1(f"{ticker}:{item['headline']}".encode()).hexdigest()
        if store.has_triggered(article_hash):
            continue

        # Cooldown: skip if this ticker triggered recently
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
        # Weighted average score; positive=+1, negative=-1, neutral=0
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
