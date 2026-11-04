"""MFN releases join the weekly run's news per company, matched by ISIN, and
never by words in the text the way the all-companies feed would be."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from fundmgr import cli
from fundmgr.config import AppConfig, UniverseTicker
from fundmgr.data import mfn, news


def _row(ticker: str, isin: str = "SE0001200015") -> UniverseTicker:
    return UniverseTicker(name=ticker, yahoo_ticker=ticker, isin=isin, country="SE",
                          exchange="OMXS", sector="", enabled=True)


def _release(title: str, hours_ago: float = 5, summary: str = "") -> mfn.Release:
    return mfn.Release(title=title, link=f"https://mfn.se/a/x/{len(title)}",
                       published=datetime(2026, 10, 30, 10, tzinfo=timezone.utc) - timedelta(hours=hours_ago),
                       language="en", tags=(":regulatory",), summary=summary)


def test_releases_become_news_items_over_a_week(monkeypatch, tmp_path):
    asked = {}

    def fake(rows, since, cache_path, **_k):
        asked.update(since=since, cache=cache_path, tickers=[r.yahoo_ticker for r in rows])
        return {"IVSO.ST": [_release("INVISIO Interim Report January–September 2026",
                                     summary="word " * 200)]}
    monkeypatch.setattr(mfn, "releases", fake)
    out = news.fetch_mfn_news([_row("IVSO.ST")], tmp_path / "slugs.json")
    [item] = out["IVSO.ST"]
    assert item["source"] == "MFN" and item["published_at"] == "2026-10-30 05:00"
    assert item["headline"].startswith("INVISIO Interim Report")
    assert len(item["summary"]) <= news.MFN_SUMMARY_CHARS + 1 and item["summary"].endswith("…")
    age = datetime.now(timezone.utc) - asked["since"]
    assert timedelta(days=7) - timedelta(minutes=1) < age < timedelta(days=7, minutes=1)
    assert asked["cache"] == tmp_path / "slugs.json" and asked["tickers"] == ["IVSO.ST"]


def test_merging_drops_a_headline_the_ticker_already_has():
    base = {"A.ST": [{"headline": "A wins order", "source": "Di"}]}
    extra = {"A.ST": [{"headline": "a wins ORDER "}, {"headline": "A reports Q3"}],
             "B.ST": [{"headline": "B buys back shares"}], "C.ST": []}
    merged = news.merge_news(base, extra)
    assert [i["headline"] for i in merged["A.ST"]] == ["A wins order", "A reports Q3"]
    assert list(merged) == ["A.ST", "B.ST"] and base["A.ST"] == [{"headline": "A wins order", "source": "Di"}]


@pytest.fixture
def run_cfg(tmp_path, monkeypatch):
    cfg = AppConfig(db_path=tmp_path / "fund.db")
    cfg.data.news_feeds = ["https://example.com/rss"]
    monkeypatch.setattr(cli, "fetch_news", lambda feeds, rows, **_k: {
        "A.ST": [{"headline": "A wins order"}]})
    return cfg


def test_the_run_reads_mfn_only_when_switched_on(run_cfg, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "fetch_mfn_news", lambda rows, path: calls.append(path) or {
        "A.ST": [{"headline": "A reports Q3"}]})
    assert cli._news_for_run(run_cfg, [_row("A.ST")]) == {"A.ST": [{"headline": "A wins order"}]}
    assert calls == []

    run_cfg.data.mfn_news = True
    got = cli._news_for_run(run_cfg, [_row("A.ST")])
    assert [i["headline"] for i in got["A.ST"]] == ["A wins order", "A reports Q3"]
    assert calls == [Path(run_cfg.db_path).parent / "mfn_slugs.json"]


def test_an_mfn_failure_leaves_the_feeds_news(run_cfg, monkeypatch):
    run_cfg.data.mfn_news = True

    def boom(rows, path):
        raise RuntimeError("mfn.se down")
    monkeypatch.setattr(cli, "fetch_mfn_news", boom)
    assert cli._news_for_run(run_cfg, [_row("A.ST")]) == {"A.ST": [{"headline": "A wins order"}]}


def test_the_dead_all_companies_feed_is_gone_and_the_real_fund_reads_mfn():
    from fundmgr.config import load_config
    root = Path(__file__).resolve().parents[1] / "config"
    for path in sorted(root.glob("config*.yaml")):
        assert not any("mfn.se" in url for url in load_config(path).data.news_feeds), path.name
    assert load_config(root / "config.yaml").data.mfn_news is True
