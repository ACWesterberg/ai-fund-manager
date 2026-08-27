"""Evidence packs for the kill judge: news text, fundamentals trend, coverage."""
import json
from datetime import datetime, timezone

import pytest

from fundmgr import evidence, paper
from fundmgr.state.models import Transaction
from fundmgr.state.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "book.db")
    s.initialise(100_000)
    s.set_meta("paper_name", "Evidence Sleeve")
    return s


# ── News extraction ───────────────────────────────────────────────────────────

NESTED_ITEM = {
    "content": {
        "title": "Vitec Q3: recurring revenue up 7%, EBITA margin down 200bp",
        "summary": "Recurring revenue growth slowed to 7% from 12%, while the EBITA "
                   "margin fell to 22% and cash conversion weakened.",
        "pubDate": "2026-08-10T06:00:00Z",
        "provider": {"displayName": "Placera"},
        "canonicalUrl": {"url": "https://example.test/vitec-q3"},
    }
}
FLAT_ITEM = {
    "title": "Board proposes dividend",
    "summary": "The board proposes an unchanged dividend.",
    "publisher": "Reuters",
    "link": "https://example.test/dividend",
}


def _mock_yf(monkeypatch, items):
    class _T:
        def __init__(self, ticker): self.news = items
    import yfinance as yf
    monkeypatch.setattr(yf, "Ticker", _T)


def test_news_items_keeps_summaries_and_urls(monkeypatch):
    """The summary was in the payload all along — the old path dropped it."""
    _mock_yf(monkeypatch, [NESTED_ITEM, FLAT_ITEM])
    items = evidence.news_items("VIT-B.ST")

    assert items[0]["headline"].startswith("Vitec Q3")
    assert "EBITA margin fell to 22%" in items[0]["summary"]
    assert items[0]["publisher"] == "Placera"
    assert items[0]["url"] == "https://example.test/vitec-q3"
    assert items[1]["publisher"] == "Reuters"
    assert items[1]["url"] == "https://example.test/dividend"


def test_news_items_skips_untitled_and_malformed(monkeypatch):
    _mock_yf(monkeypatch, [{"content": {"title": "  "}}, "not a dict", FLAT_ITEM])
    assert [i["headline"] for i in evidence.news_items("X")] == ["Board proposes dividend"]


def test_news_items_respects_the_cap(monkeypatch):
    _mock_yf(monkeypatch, [FLAT_ITEM] * 30)
    assert len(evidence.news_items("X", max_items=3)) == 3


def test_news_items_survives_a_dead_feed(monkeypatch):
    import yfinance as yf

    def boom(ticker):
        raise RuntimeError("feed down")
    monkeypatch.setattr(yf, "Ticker", boom)
    assert evidence.news_items("X") == []


# ── Article fetching ──────────────────────────────────────────────────────────

def test_html_to_text_strips_markup_and_boilerplate():
    html = """
    <html><head><style>.a{color:red}</style><script>var x=1;</script></head>
    <body><nav>Home</nav>
    <p>Recurring revenue growth slowed to 7 percent from 12 percent in the quarter,
    the company said, while the EBITA margin contracted by 200 basis points.</p>
    <p>Short.</p></body></html>
    """
    text = evidence._html_to_text(html)
    assert "Recurring revenue growth slowed to 7 percent" in text
    assert "var x=1" not in text and "color:red" not in text
    assert "Home" not in text        # nav fragment is below the length floor
    assert "Short." not in text


def test_html_to_text_decodes_entities():
    html = "<p>" + "Margins fell &amp; cash conversion weakened materially " * 2 + "</p>"
    assert "&" in evidence._html_to_text(html)
    assert "&amp;" not in evidence._html_to_text(html)


class _Resp:
    def __init__(self, text="", status=200, ctype="text/html"):
        self.text, self.status_code = text, status
        self.headers = {"Content-Type": ctype}


def _mock_requests(monkeypatch, resp):
    import requests
    monkeypatch.setattr(requests, "get", lambda url, **kw: resp
                        if not isinstance(resp, Exception) else (_ for _ in ()).throw(resp))


def test_fetch_article_returns_body(monkeypatch):
    body = "<p>" + ("Recurring revenue growth slowed to seven percent this quarter. " * 3) + "</p>"
    _mock_requests(monkeypatch, _Resp(body))
    assert "Recurring revenue growth slowed" in evidence.fetch_article("https://example.test/a")


def test_fetch_article_ignores_non_html_and_errors(monkeypatch):
    _mock_requests(monkeypatch, _Resp("{}", ctype="application/json"))
    assert evidence.fetch_article("https://example.test/a") == ""

    _mock_requests(monkeypatch, _Resp("<p>x</p>", status=403))
    assert evidence.fetch_article("https://example.test/a") == ""

    _mock_requests(monkeypatch, RuntimeError("timeout"))
    assert evidence.fetch_article("https://example.test/a") == ""


def test_fetch_article_rejects_non_http_urls():
    assert evidence.fetch_article("") == ""
    assert evidence.fetch_article("file:///etc/passwd") == ""
    assert evidence.fetch_article("javascript:alert(1)") == ""


def test_fetch_article_truncates(monkeypatch):
    long_para = "<p>" + ("Margins deteriorated materially across every segment. " * 400) + "</p>"
    _mock_requests(monkeypatch, _Resp(long_para))
    assert len(evidence.fetch_article("https://example.test/a")) <= evidence.ARTICLE_CHARS


def test_enrich_prioritises_thin_summaries(monkeypatch):
    fetched = []

    def fake_fetch(url, timeout=6):
        fetched.append(url)
        return "body text"
    monkeypatch.setattr(evidence, "fetch_article", fake_fetch)

    items = [
        {"summary": "a very long summary " * 20, "url": "https://example.test/long"},
        {"summary": "", "url": "https://example.test/empty"},
    ]
    filled = evidence.enrich_with_bodies(items, max_articles=1)
    assert filled == 1
    assert fetched == ["https://example.test/empty"]   # thin summary gets the budget


def test_article_fetching_can_be_disabled(monkeypatch):
    monkeypatch.setenv("FUND_KILL_FETCH_ARTICLES", "0")
    monkeypatch.setattr(evidence, "fetch_article",
                        lambda *a, **kw: pytest.fail("should not fetch"))
    assert evidence.enrich_with_bodies([{"summary": "", "url": "https://x.test"}]) == 0


# ── Fundamentals snapshots and trend ──────────────────────────────────────────

def test_snapshot_records_and_dedupes(store):
    store.save_fundamentals("VIT-B.ST", {"revenue_growth": 0.12, "profit_margin": 0.18})
    assert evidence.snapshot_fundamentals(store, ["VIT-B.ST"], today="2026-01-01") == 1
    # Unchanged values on a later day add no new snapshot.
    assert evidence.snapshot_fundamentals(store, ["VIT-B.ST"], today="2026-01-02") == 0

    store.save_fundamentals("VIT-B.ST", {"revenue_growth": 0.07, "profit_margin": 0.15})
    assert evidence.snapshot_fundamentals(store, ["VIT-B.ST"], today="2026-08-01") == 1

    series = json.loads(store.get_meta("paper_fundsnap:VIT-B.ST"))
    assert [s["date"] for s in series] == ["2026-01-01", "2026-08-01"]


def test_snapshot_skips_tickers_without_data(store):
    assert evidence.snapshot_fundamentals(store, ["NOPE"]) == 0


def test_snapshot_series_is_capped(store):
    for i in range(evidence.MAX_SNAPSHOTS + 5):
        store.save_fundamentals("X", {"revenue_growth": 0.10 + i / 100})
        evidence.snapshot_fundamentals(store, ["X"], today=f"2026-01-{i + 1:02d}")
    series = json.loads(store.get_meta("paper_fundsnap:X"))
    assert len(series) == evidence.MAX_SNAPSHOTS


# ── Fiscal-period bucketing ───────────────────────────────────────────────────

def _quarter(store, ticker, period, **values):
    """Record one reported quarter for a ticker."""
    store.save_fundamentals(ticker, {"fiscal_period_end": period, **values})
    evidence.snapshot_fundamentals(store, [ticker], today=period)


def test_a_revised_figure_replaces_its_quarter_rather_than_adding_one(store):
    """Otherwise a restatement looks like a second quarter of the same result —
    and a 'two consecutive quarters' rule fires on one quarter read twice."""
    _quarter(store, "VIT-B.ST", "2026-03-31", revenue_growth=0.07)
    store.save_fundamentals("VIT-B.ST", {"fiscal_period_end": "2026-03-31",
                                         "revenue_growth": 0.065})   # restated
    evidence.snapshot_fundamentals(store, ["VIT-B.ST"], today="2026-05-20")

    series = json.loads(store.get_meta("paper_fundsnap:VIT-B.ST"))
    assert len(series) == 1
    assert series[0]["period"] == "2026-03-31"
    assert series[0]["values"]["revenue_growth"] == 0.065
    assert series[0]["date"] == "2026-05-20"      # when we read it


def test_metric_history_is_one_row_per_period_newest_first(store):
    _quarter(store, "VIT-B.ST", "2025-12-31", revenue_growth=0.11)
    _quarter(store, "VIT-B.ST", "2026-03-31", revenue_growth=0.07)
    _quarter(store, "VIT-B.ST", "2026-06-30", revenue_growth=0.05)

    history = evidence.metric_history(store, "VIT-B.ST", "revenue_growth")
    assert [r["period"] for r in history] == ["2026-06-30", "2026-03-31", "2025-12-31"]
    # Percent, the unit a rule is written in — not the provider's 0-1 fraction.
    assert [round(r["value"], 1) for r in history] == [5.0, 7.0, 11.0]


def test_readings_without_a_period_are_not_counted_as_quarters(store):
    """A series recorded before periods were tracked is unknown, not passing."""
    store.save_fundamentals("VIT-B.ST", {"revenue_growth": 0.02})
    evidence.snapshot_fundamentals(store, ["VIT-B.ST"], today="2026-01-01")
    assert evidence.metric_history(store, "VIT-B.ST", "revenue_growth") == []

    run = evidence.sustained_breach(store, "VIT-B.ST", "revenue_growth", "below", 8, 2)
    assert run["hit"] is False and run["periods_seen"] == 0


def test_sustained_breach_counts_the_run_from_the_newest_period(store):
    _quarter(store, "X", "2025-12-31", revenue_growth=0.11)
    _quarter(store, "X", "2026-03-31", revenue_growth=0.07)
    _quarter(store, "X", "2026-06-30", revenue_growth=0.05)

    two = evidence.sustained_breach(store, "X", "revenue_growth", "below", 8, 2)
    assert two["hit"] is True and two["streak"] == 2

    three = evidence.sustained_breach(store, "X", "revenue_growth", "below", 8, 3)
    assert three["hit"] is False          # 2025-12-31 was above the line
    assert three["streak"] == 2 and three["periods_seen"] == 3


def test_one_good_quarter_resets_the_run(store):
    _quarter(store, "X", "2025-09-30", revenue_growth=0.04)
    _quarter(store, "X", "2025-12-31", revenue_growth=0.03)
    _quarter(store, "X", "2026-03-31", revenue_growth=0.12)   # recovered
    _quarter(store, "X", "2026-06-30", revenue_growth=0.05)   # falls again

    run = evidence.sustained_breach(store, "X", "revenue_growth", "below", 8, 2)
    assert run["hit"] is False and run["streak"] == 1


def test_thin_history_never_passes_a_multi_quarter_rule(store):
    """One quarter on file cannot satisfy 'two consecutive quarters', however
    badly that quarter reads."""
    _quarter(store, "X", "2026-06-30", revenue_growth=0.01)
    run = evidence.sustained_breach(store, "X", "revenue_growth", "below", 8, 2)
    assert run["hit"] is False
    assert run["streak"] == 1 and run["periods_seen"] == 1


def test_backfill_seeds_past_periods_and_arms_a_multi_quarter_rule(store):
    """Without this a two-quarter leverage rule is inert until we have watched
    the company report twice — while the quarters sit in the statements."""
    _quarter(store, "BALD-B.ST", "2026-06-30", net_debt_to_assets=0.528)
    assert evidence.sustained_breach(
        store, "BALD-B.ST", "net_debt_to_assets", "above", 50, 2)["hit"] is False

    added = evidence.backfill_periods(store, "BALD-B.ST", {
        "2026-03-31": {"net_debt_to_assets": 0.51},
        "2025-12-31": {"net_debt_to_assets": 0.48},
    })
    assert added == 2

    run = evidence.sustained_breach(store, "BALD-B.ST", "net_debt_to_assets",
                                    "above", 50, 2)
    assert run["hit"] is True and run["streak"] == 2 and run["periods_seen"] == 3


def test_backfill_never_overwrites_an_observed_reading(store):
    """What we actually saw beats a recomputation of the same period."""
    _quarter(store, "X", "2026-06-30", net_debt_to_assets=0.528)
    assert evidence.backfill_periods(store, "X", {
        "2026-06-30": {"net_debt_to_assets": 0.99}}) == 0

    history = evidence.metric_history(store, "X", "net_debt_to_assets")
    assert [round(r["value"], 1) for r in history] == [52.8]


def test_backfill_keeps_the_series_oldest_first(store):
    """Trend reads the first row as the oldest, so order is load-bearing."""
    store.save_fundamentals("X", {"revenue_growth": 0.05})          # legacy, no period
    evidence.snapshot_fundamentals(store, ["X"], today="2024-01-01")
    _quarter(store, "X", "2026-06-30", net_debt_to_assets=0.528)
    evidence.backfill_periods(store, "X", {
        "2026-03-31": {"net_debt_to_assets": 0.51},
        "2025-12-31": {"net_debt_to_assets": 0.48},
    })

    series = json.loads(store.get_meta("paper_fundsnap:X"))
    assert [r.get("period", "") for r in series] == [
        "", "2025-12-31", "2026-03-31", "2026-06-30"]


def test_backfill_is_idempotent(store):
    history = {"2026-03-31": {"net_debt_to_assets": 0.51}}
    assert evidence.backfill_periods(store, "X", history) == 1
    assert evidence.backfill_periods(store, "X", history) == 0
    assert evidence.backfill_periods(store, "X", {}) == 0


def test_trend_reports_direction_against_oldest_snapshot(store):
    store.save_fundamentals("VIT-B.ST", {"revenue_growth": 0.12, "profit_margin": 0.20})
    evidence.snapshot_fundamentals(store, ["VIT-B.ST"], today="2026-01-01")
    store.save_fundamentals("VIT-B.ST", {"revenue_growth": 0.07, "profit_margin": 0.15})
    evidence.snapshot_fundamentals(store, ["VIT-B.ST"], today="2026-08-01")

    rows = {r["key"]: r for r in evidence.fundamentals_trend(store, "VIT-B.ST")}
    assert rows["revenue_growth"]["current"] == pytest.approx(0.07)
    assert rows["revenue_growth"]["direction"] == "down"
    assert "was +12.0%" in rows["revenue_growth"]["text"]
    assert rows["profit_margin"]["direction"] == "down"


def test_trend_calls_a_tiny_move_flat(store):
    store.save_fundamentals("X", {"revenue_growth": 0.1000})
    evidence.snapshot_fundamentals(store, ["X"], today="2026-01-01")
    store.save_fundamentals("X", {"revenue_growth": 0.1001})
    rows = {r["key"]: r for r in evidence.fundamentals_trend(store, "X")}
    assert rows["revenue_growth"]["direction"] == "flat"


def test_trend_without_history_has_no_direction(store):
    store.save_fundamentals("X", {"revenue_growth": 0.09})
    rows = evidence.fundamentals_trend(store, "X")
    assert rows[0]["direction"] == ""
    assert rows[0]["earlier"] is None


def test_trend_ignores_junk_values(store):
    store.save_fundamentals("X", {"revenue_growth": None, "profit_margin": "n/a",
                                  "roe": float("nan"), "pe_ratio": 21.4})
    rows = {r["key"] for r in evidence.fundamentals_trend(store, "X")}
    assert rows == {"pe_ratio"}


def test_trend_is_empty_without_a_cache(store):
    assert evidence.fundamentals_trend(store, "UNKNOWN") == []


# ── Pack assembly + rendering ─────────────────────────────────────────────────

def test_pack_combines_news_fundamentals_and_position(store, monkeypatch):
    _mock_yf(monkeypatch, [NESTED_ITEM])
    monkeypatch.setattr(evidence, "enrich_with_bodies", lambda items, **kw: 0)
    store.save_fundamentals("VIT-B.ST", {"revenue_growth": 0.07})
    evidence.snapshot_fundamentals(store, ["VIT-B.ST"], today="2026-01-01")
    store.apply_fill(Transaction(ticker="VIT-B.ST", side="buy", shares=100,
                                 price_sek=300.0, fee_sek=0.0, source="fill"))

    pack = evidence.build_pack(store, "VIT-B.ST")
    assert pack["news"][0]["summary"]
    assert pack["fundamentals"][0]["key"] == "revenue_growth"
    assert pack["position"]["shares"] == 100
    assert pack["coverage"] == {
        "headlines": True, "article_text": True, "article_bodies": 0,
        "fundamentals": True, "fundamentals_trend": True,
    }


def test_a_snapshot_taken_today_is_not_a_trend(store):
    """Day one must not read as 'flat' — there is no history to be flat against."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    store.save_fundamentals("X", {"revenue_growth": 0.09})
    evidence.snapshot_fundamentals(store, ["X"], today=today)

    rows = evidence.fundamentals_trend(store, "X", today=today)
    assert rows[0]["earlier"] is None
    assert rows[0]["direction"] == ""
    assert "was" not in rows[0]["text"]


def test_pack_folds_in_cached_store_news(store, monkeypatch):
    _mock_yf(monkeypatch, [NESTED_ITEM])
    monkeypatch.setattr(evidence, "enrich_with_bodies", lambda items, **kw: 0)
    store.save_news_sentiment("VIT-B.ST", [{
        "headline": "Acquisition multiples creeping up", "summary": "Deal prices rose.",
        "published_at": "2026-08-09", "sentiment_label": "negative", "sentiment_score": 0.7,
    }])

    pack = evidence.build_pack(store, "VIT-B.ST")
    headlines = [i["headline"] for i in pack["news"]]
    assert "Acquisition multiples creeping up" in headlines
    assert len(headlines) == len(set(headlines))     # no duplicates across sources


def test_pack_reports_empty_coverage_honestly(store, monkeypatch):
    import yfinance as yf
    monkeypatch.setattr(yf, "Ticker", lambda t: type("T", (), {"news": []})())
    pack = evidence.build_pack(store, "NOTHING")
    assert pack["coverage"] == {
        "headlines": False, "article_text": False, "article_bodies": 0,
        "fundamentals": False, "fundamentals_trend": False,
    }


def test_render_pack_includes_every_section(store, monkeypatch):
    _mock_yf(monkeypatch, [NESTED_ITEM])
    monkeypatch.setattr(evidence, "enrich_with_bodies", lambda items, **kw: 0)
    store.save_fundamentals("VIT-B.ST", {"revenue_growth": 0.07})
    text = evidence.render_pack(evidence.build_pack(store, "VIT-B.ST"))

    assert "## Recent news" in text
    assert "Summary: Recurring revenue growth slowed" in text
    assert "Revenue growth (yoy): +7.0%" in text
    assert "no earlier snapshot yet" in text
    assert "What this evidence covers" in text


def test_render_pack_says_when_there_is_nothing(store, monkeypatch):
    import yfinance as yf
    monkeypatch.setattr(yf, "Ticker", lambda t: type("T", (), {"news": []})())
    text = evidence.render_pack(evidence.build_pack(store, "X"))
    assert "(no recent news found)" in text
    assert "none cached for this ticker" in text


# ── Verdict parsing ───────────────────────────────────────────────────────────

def test_parse_verdict_json():
    out = paper._parse_verdict(json.dumps({
        "verdict": "INSUFFICIENT",
        "reason": "Acquisition returns are not disclosed in any source here.",
        "unchecked": ["acquisitions generating weaker returns"],
    }))
    assert out["verdict"] == "insufficient"
    assert out["unchecked"] == ["acquisitions generating weaker returns"]


def test_parse_verdict_json_in_prose():
    raw = 'Here is my assessment:\n{"verdict": "YES", "reason": "growth fell to 6%"}\nDone.'
    assert paper._parse_verdict(raw)["verdict"] == "yes"


def test_parse_verdict_plain_text_fallback():
    assert paper._parse_verdict("NO")["verdict"] == "no"
    assert paper._parse_verdict("YES: growth fell below 8%")["verdict"] == "yes"
    assert paper._parse_verdict("YES: growth fell below 8%")["reason"] == "growth fell below 8%"
    assert paper._parse_verdict("INSUFFICIENT: no report data")["verdict"] == "insufficient"


def test_parse_verdict_rejects_noise():
    assert paper._parse_verdict("") is None
    assert paper._parse_verdict("I'm not sure what you mean") is None


def test_parse_verdict_defaults_unknown_verdict_to_no():
    assert paper._parse_verdict('{"verdict": "maybe", "reason": "x"}')["verdict"] == "no"


# ── check_kill_criteria with the tri-state verdict ────────────────────────────

@pytest.fixture
def watched_book(tmp_path, monkeypatch):
    """A live book with one text kill criterion, isolated from the network."""
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path / "paper")
    monkeypatch.setattr(paper, "detect_currency", lambda t: "SEK")
    monkeypatch.setattr(paper, "_cache_price_history", lambda store, tickers, **kw: None)
    monkeypatch.setattr(paper, "_search_symbol", lambda name: None)
    import fundmgr.data.benchmark as benchmark
    monkeypatch.setattr(benchmark, "fetch_and_cache_benchmark", lambda store, **kw: True)
    import fundmgr.data.quotes as quotes
    monkeypatch.setattr(quotes, "live_prices", lambda tickers: {t: 300.0 for t in tickers})
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    slug, _ = paper.create_portfolio(
        "Serial Acquirer", 30_000, "",
        holdings_override=[{
            "ticker": "VIT-B.ST", "name": "Vitec", "weight_pct": 100,
            "shares": 100, "avg_cost_sek": 300.0,
            "kill_criterion": ("Recurring growth falls below 8% while EBITA/FCF "
                               "deteriorates, or acquisitions begin generating "
                               "clearly weaker returns"),
        }],
        kind="live", seed_holdings=True)
    return slug


@pytest.fixture
def telegram(monkeypatch):
    sent = []
    import fundmgr.notify.send as send_mod
    monkeypatch.setattr(send_mod, "send_telegram", lambda text, **kw: sent.append(text) or True)
    return sent


def _pack_stub(monkeypatch, **coverage):
    base = {"headlines": True, "article_text": True, "article_bodies": 1,
            "fundamentals": True, "fundamentals_trend": True}
    base.update(coverage)
    monkeypatch.setattr("fundmgr.evidence.build_pack", lambda store, ticker, **kw: {
        "ticker": ticker,
        "news": [{"headline": "Q3 out", "publisher": "", "published": "",
                  "summary": "Recurring growth 7%.", "url": "", "body": ""}],
        "fundamentals": [{"key": "revenue_growth", "label": "Revenue growth (yoy)",
                          "current": 0.07, "earlier": 0.12, "earlier_date": "2026-01-01",
                          "direction": "down", "text": "+7.0% (was +12.0% — down)"}],
        "position": None,
        "coverage": base,
    })


def test_insufficient_verdict_is_reported_not_silently_passed(
        watched_book, monkeypatch, telegram):
    """The core fix: a criterion we cannot check must not look like 'intact'."""
    _pack_stub(monkeypatch)
    monkeypatch.setattr(paper, "_judge_kill_hit", lambda ticker, criterion, pack: {
        "verdict": "insufficient",
        "reason": "Acquisition returns are not disclosed in the available evidence.",
        "unchecked": ["acquisitions begin generating clearly weaker returns"],
    })

    log = paper.check_kill_criteria(watched_book)
    assert any("not verifiable" in line for line in log)
    assert len(telegram) == 1
    assert "cannot be checked automatically" in telegram[0]
    assert "acquisitions begin generating clearly weaker returns" in telegram[0]

    _meta, store = paper.open_portfolio(watched_book)
    saved = json.loads(store.get_meta("paper_killverdict:VIT-B.ST"))
    assert saved["verdict"] == "insufficient"
    # No kill hit was recorded — unverifiable is not a trigger.
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert store.get_meta(f"paper_killhit:VIT-B.ST:{today}") is None


def test_insufficient_notifies_once_per_wording(watched_book, monkeypatch, telegram):
    _pack_stub(monkeypatch)
    monkeypatch.setattr(paper, "_judge_kill_hit", lambda ticker, criterion, pack: {
        "verdict": "insufficient", "reason": "no data", "unchecked": ["x"]})

    _meta, store = paper.open_portfolio(watched_book)
    paper.check_kill_criteria(watched_book)
    assert len(telegram) == 1

    # A new day, same criterion → logged again but no repeat push.
    store.set_meta("paper_killwatch:VIT-B.ST", "1999-01-01")
    paper.check_kill_criteria(watched_book)
    assert len(telegram) == 1

    # Rewording it re-notifies, since the new phrasing may be checkable.
    from fundmgr import watchplan
    watchplan.set_position_plan(store, "VIT-B.ST",
                                kill_criterion="Recurring growth below 8% for two quarters")
    store.set_meta("paper_killwatch:VIT-B.ST", "1999-01-01")
    paper.check_kill_criteria(watched_book)
    assert len(telegram) == 2


def test_becoming_checkable_clears_the_gap_flag(watched_book, monkeypatch, telegram):
    _pack_stub(monkeypatch)
    monkeypatch.setattr(paper, "_judge_kill_hit", lambda ticker, criterion, pack: {
        "verdict": "insufficient", "reason": "no data", "unchecked": []})
    paper.check_kill_criteria(watched_book)

    _meta, store = paper.open_portfolio(watched_book)
    assert store.get_meta("paper_killgap:VIT-B.ST")

    monkeypatch.setattr(paper, "_judge_kill_hit", lambda ticker, criterion, pack: {
        "verdict": "no", "reason": "", "unchecked": []})
    store.set_meta("paper_killwatch:VIT-B.ST", "1999-01-01")
    paper.check_kill_criteria(watched_book)
    assert store.get_meta("paper_killgap:VIT-B.ST") == ""


def test_hit_still_alerts_and_records(watched_book, monkeypatch, telegram):
    _pack_stub(monkeypatch)
    monkeypatch.setattr(paper, "_judge_kill_hit", lambda ticker, criterion, pack: {
        "verdict": "yes",
        "reason": "Recurring growth 7% with EBITA margin down 200bp — both legs met.",
        "unchecked": []})

    log = paper.check_kill_criteria(watched_book)
    assert any("may be triggering" in line for line in log)
    assert "may falsify a thesis" in telegram[0]

    _meta, store = paper.open_portfolio(watched_book)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert "both legs met" in store.get_meta(f"paper_killhit:VIT-B.ST:{today}")


def test_no_evidence_at_all_skips_the_judge(watched_book, monkeypatch, telegram):
    monkeypatch.setattr("fundmgr.evidence.build_pack", lambda store, ticker, **kw: {
        "ticker": ticker, "news": [], "fundamentals": [], "position": None,
        "coverage": {"headlines": False, "article_text": False, "article_bodies": 0,
                     "fundamentals": False, "fundamentals_trend": False},
    })
    monkeypatch.setattr(paper, "_judge_kill_hit",
                        lambda *a: pytest.fail("judge should not run without evidence"))
    assert paper.check_kill_criteria(watched_book) == []
    assert telegram == []


def test_judge_failure_is_not_a_verdict(watched_book, monkeypatch, telegram):
    """An API failure must not be recorded as 'intact'."""
    _pack_stub(monkeypatch)
    monkeypatch.setattr(paper, "_judge_kill_hit", lambda ticker, criterion, pack: None)
    assert paper.check_kill_criteria(watched_book) == []
    _meta, store = paper.open_portfolio(watched_book)
    assert store.get_meta("paper_killverdict:VIT-B.ST") is None


def test_dashboard_flags_an_unverifiable_criterion(watched_book, monkeypatch):
    from fastapi.testclient import TestClient
    _pack_stub(monkeypatch)
    monkeypatch.setattr(paper, "_judge_kill_hit", lambda ticker, criterion, pack: {
        "verdict": "insufficient", "reason": "Acquisition returns not disclosed.",
        "unchecked": ["acquisitions begin generating clearly weaker returns"]})
    import fundmgr.notify.send as send_mod
    monkeypatch.setattr(send_mod, "send_telegram", lambda *a, **kw: True)
    paper.check_kill_criteria(watched_book)

    from fundmgr.web.app import app
    r = TestClient(app).get(f"/live/{watched_book}")
    assert r.status_code == 200
    assert "not auto-checkable" in r.text
    assert "acquisitions begin generating clearly weaker returns" in r.text


# ── JSON extraction (the bug that ate verdicts with arrays) ───────────────────

def test_extract_json_object_prefers_the_outer_object():
    """_extract_json_block looks for '[' first and would return the inner array."""
    raw = 'Result:\n{"verdict": "NO", "unchecked": ["a", "b"], "reason": "x"}\nEnd.'
    assert json.loads(paper.extract_json_object(raw))["verdict"] == "NO"
    # The old helper grabs the nested array instead — the regression guarded here.
    assert json.loads(paper._extract_json_block(raw)) == ["a", "b"]


def test_extract_json_object_handles_fences_and_braces_in_strings():
    raw = '```json\n{"reason": "the filing said {redacted}", "unchecked": []}\n```'
    assert json.loads(paper.extract_json_object(raw))["reason"] == "the filing said {redacted}"


def test_extract_json_object_handles_escaped_quotes():
    raw = r'{"reason": "he said \"below 8%\" clearly", "unchecked": []}'
    assert json.loads(paper.extract_json_object(raw))["unchecked"] == []


def test_extract_json_object_gives_up_cleanly():
    assert paper.extract_json_object("no object here [1,2]") is None
    assert paper.extract_json_object("") is None


def test_verdict_with_unchecked_array_parses():
    """End-to-end guard: the real judge reply shape survives extraction."""
    raw = ('Assessment follows.\n{"verdict": "INSUFFICIENT", "reason": "no deal data", '
           '"unchecked": ["acquisition returns"]}')
    out = paper._parse_verdict(raw)
    assert out["verdict"] == "insufficient"
    assert out["unchecked"] == ["acquisition returns"]


# ── The metric catalogue must match what the data layer actually fetches ──────

# financedata's `_FIELD_MAP` (src/financedata/fundamentals.py) — the complete
# set of keys `get_fundamentals` can return. Copied here as a guard: offering
# the analyser a metric with no data behind it invites a rule that can never
# evaluate, so a new entry in _FUND_FIELDS must be one of these.
FINANCEDATA_KEYS = {
    "pe_ratio", "forward_pe", "pb_ratio", "ev_to_ebitda", "price_to_sales",
    "market_cap", "profit_margin", "gross_margin", "operating_margin",
    "ebitda_margin", "roe", "roa", "free_cash_flow", "operating_cash_flow",
    "total_cash", "total_debt", "debt_to_equity",
    "revenue_growth", "earnings_growth", "beta", "fifty_two_week_high",
    "fifty_two_week_low", "dividend_yield", "analyst_target_price",
    "analyst_count", "currency", "earnings_timestamp", "ex_div_timestamp",
    "website", "sector",
}


def test_every_offered_metric_is_actually_fetched():
    from fundmgr.data.statements import STATEMENT_METRICS
    # Two sources feed the cache: financedata's `.info` map, and the ratios
    # derived from the statements. An offered metric must come from one of them.
    unknown = set(evidence.FUND_FIELD_META) - FINANCEDATA_KEYS - set(STATEMENT_METRICS)
    assert not unknown, (
        f"{unknown} are offered to the criterion analyser but financedata never "
        "fetches them — a rule naming one could never evaluate.")


def test_profitability_and_cash_flow_metrics_are_offered():
    """These were absent until financedata mapped them; verified live on the Pi
    (`fund fundamentals-check VOLV-B.ST`) before being offered here."""
    for key in ("ebitda_margin", "operating_margin", "roa",
                "free_cash_flow", "operating_cash_flow", "total_cash", "total_debt"):
        assert key in evidence.FUND_FIELD_META
        assert key in FINANCEDATA_KEYS


def test_price_to_sales_is_mapped_but_not_offered():
    """Yahoo returns null for it (VOLV-B.ST and AAPL), so a rule on it could
    never evaluate — it stays in the data layer but off the metric menu."""
    assert "price_to_sales" in FINANCEDATA_KEYS
    assert "price_to_sales" not in evidence.FUND_FIELD_META


def test_fraction_flags_match_the_providers_units():
    """financedata stores yfinance values raw; these are the 0-1 ones."""
    fractions = {k for k, m in evidence.FUND_FIELD_META.items() if m["is_fraction"]}
    assert fractions == {"profit_margin", "gross_margin", "operating_margin",
                         "ebitda_margin", "roe", "roa", "revenue_growth",
                         "earnings_growth", "dividend_yield",
                         # derived ratios, already fractions by construction
                         "equity_to_assets", "net_debt_to_assets",
                         "cost_to_income", "fcf_to_net_income"}


# ── fundamentals-check CLI ────────────────────────────────────────────────────

def _run_check(monkeypatch, payload):
    from click.testing import CliRunner
    import financedata
    monkeypatch.setattr(financedata, "get_fundamentals",
                        lambda tickers, **kw: {tickers[0]: payload})
    from fundmgr.cli import cli
    return CliRunner().invoke(cli, ["fundamentals-check", "volv-b.st"])


def test_fundamentals_check_flags_a_metric_with_no_data(monkeypatch):
    """The guard against a silently mis-mapped yfinance key."""
    payload = {k: 0.1 for k in evidence.FUND_FIELD_META}
    del payload["profit_margin"]                # simulate a mis-mapped key name
    result = _run_check(monkeypatch, payload)
    assert result.exit_code == 0
    assert "NOT in the payload" in result.output
    assert "profit_margin" in result.output
    assert "can never evaluate" in result.output


def test_fundamentals_check_reports_a_clean_payload(monkeypatch):
    payload = {k: 0.1 for k in evidence.FUND_FIELD_META}
    payload["sector"] = "Industrials"
    payload["beta"] = None
    result = _run_check(monkeypatch, payload)
    assert result.exit_code == 0
    assert "every offered kill-criterion metric is present" in result.output
    assert "Returned but empty for this ticker" in result.output
    assert "beta" in result.output


def test_fundamentals_check_errors_on_no_data(monkeypatch):
    result = _run_check(monkeypatch, None)
    assert result.exit_code != 0
    assert "No fundamentals returned" in result.output


def test_fundamentals_check_surfaces_newly_available_fields(monkeypatch):
    """A field the data layer returns but no criterion can use yet."""
    payload = {k: 0.1 for k in evidence.FUND_FIELD_META}
    payload.update({"market_cap": 6.99e11, "beta": 0.983, "sector": "Industrials"})
    result = _run_check(monkeypatch, payload)
    assert "Available but not offered as a metric" in result.output
    tail = result.output.split("Available but not offered")[1]
    assert "market_cap" in tail and "beta" in tail
    assert "sector" not in tail          # non-numeric fields aren't rule candidates


def test_fundamentals_check_shows_statement_derived_metrics(monkeypatch):
    """A criterion reads the merged cache, so the check must show both sources."""
    from fundmgr.data import statements
    payload = {k: 0.1 for k in evidence.FUND_FIELD_META}
    monkeypatch.setattr(statements, "derive", lambda t: {
        "equity_to_assets": 0.37, "net_debt_to_assets": 0.504,
        "net_debt_to_ebitda": 12.8, "interest_coverage": 2.65,
        "cost_to_income": None, "fcf_to_net_income": 0.9,
        "cash_runway_months": None,
    })
    result = _run_check(monkeypatch, payload)
    assert "Derived from the financial statements (5/7)" in result.output
    assert "equity_to_assets" in result.output and "37.0%" in result.output
    assert "no matching statement row" in result.output
    assert "row label needs adding" in result.output


def test_fundamentals_check_survives_a_statement_failure(monkeypatch):
    from fundmgr.data import statements

    def boom(ticker):
        raise RuntimeError("statements exploded")
    monkeypatch.setattr(statements, "derive", boom)
    result = _run_check(monkeypatch, {k: 0.1 for k in evidence.FUND_FIELD_META})
    assert result.exit_code == 0
    assert "statement metrics unavailable" in result.output


# ── Book-local metrics: the figures no provider carries ──────────────────────
#
# Organic growth, ARR, adjusted EBITA, cash conversion, backlog. These are
# company-defined measures rather than accounting standards, which is exactly
# why no standardised feed has them — the figure has to come from the company's
# own report, entered by hand and stamped with the period it describes.

def test_a_custom_metric_joins_the_menu(store):
    evidence.define_custom_metric(store, "organic_growth", "Organic growth", "%")
    menu = evidence.field_meta(store)
    assert menu["organic_growth"]["label"] == "Organic growth"
    assert menu["organic_growth"]["custom"] is True
    assert menu["organic_growth"]["is_fraction"] is False   # entered as printed
    assert "revenue_growth" in menu                          # built-ins still there


def test_the_built_in_menu_is_untouched_without_a_store():
    assert evidence.field_meta() == evidence.FUND_FIELD_META
    assert "custom" not in evidence.FUND_FIELD_META["revenue_growth"]


def test_a_custom_metric_cannot_shadow_a_provider_field(store):
    """A rule written against revenue_growth must not change meaning because
    someone defined a metric with the same name."""
    with pytest.raises(ValueError, match="already a built-in"):
        evidence.define_custom_metric(store, "revenue_growth", "Mine", "%")


def test_metric_keys_are_normalised(store):
    evidence.define_custom_metric(store, "  Adj. EBIT Margin ", unit="%")
    assert "adj_ebit_margin" in evidence.custom_metrics(store)


def test_a_bad_unit_is_refused(store):
    with pytest.raises(ValueError, match="Unit must be"):
        evidence.define_custom_metric(store, "backlog", unit="furlongs")


def test_forgetting_a_metric(store):
    evidence.define_custom_metric(store, "arr", "ARR", "SEK")
    assert evidence.forget_custom_metric(store, "arr") is True
    assert evidence.forget_custom_metric(store, "arr") is False
    assert "arr" not in evidence.field_meta(store)


def test_a_reported_figure_reads_back_like_any_other_metric(store):
    evidence.define_custom_metric(store, "organic_growth", "Organic growth", "%")
    evidence.record_reported(store, "SYSR.ST", {"organic_growth": 6.4}, "2026-06-30")
    assert evidence.current_metric(store, "SYSR.ST", "organic_growth") == 6.4


def test_a_reported_figure_occupies_its_own_reporting_period(store):
    """Two reports are two quarters; one report entered twice is still one."""
    evidence.define_custom_metric(store, "organic_growth", "Organic growth", "%")
    evidence.record_reported(store, "X", {"organic_growth": 4.1}, "2026-03-31")
    evidence.record_reported(store, "X", {"organic_growth": 6.4}, "2026-06-30")
    evidence.record_reported(store, "X", {"organic_growth": 6.5}, "2026-06-30")  # restated

    history = evidence.metric_history(store, "X", "organic_growth")
    assert [(r["period"], r["value"]) for r in history] == [
        ("2026-06-30", 6.5), ("2026-03-31", 4.1)]

    run = evidence.sustained_breach(store, "X", "organic_growth", "above", 4, 2)
    assert run["hit"] is True and run["streak"] == 2


def test_a_reported_figure_does_not_disturb_provider_fields(store):
    evidence.define_custom_metric(store, "organic_growth", "Organic growth", "%")
    store.save_fundamentals("X", {"revenue_growth": 0.12, "pe_ratio": 20.0})
    evidence.record_reported(store, "X", {"organic_growth": 6.4}, "2026-06-30")
    cached = store.get_fundamentals("X")
    assert cached["revenue_growth"] == 0.12 and cached["pe_ratio"] == 20.0
    assert cached["organic_growth"] == 6.4


def test_a_printed_percentage_in_a_fraction_metric_is_refused(store):
    """The wrong value looks entirely plausible in every listing — 41.2 reads
    back as 4120%, so a "gross margin below 45" rule silently stops matching.
    Nothing downstream would ever flag it, so entry has to."""
    with pytest.raises(ValueError, match="4,120%"):
        evidence.record_reported(store, "X", {"gross_margin": 41.2}, "2026-06-30")
    assert evidence.current_metric(store, "X", "gross_margin") is None


def test_percent_records_printed_figures_in_provider_units(store):
    evidence.record_reported(store, "X", {"gross_margin": 41.2}, "2026-06-30",
                             percent=True)
    assert (store.get_fundamentals("X") or {})["gross_margin"] == pytest.approx(0.412)
    assert evidence.current_metric(store, "X", "gross_margin") == pytest.approx(41.2)


def test_provider_units_still_record_unchanged(store):
    """The guard must not move the goalposts for a caller doing it right."""
    evidence.record_reported(store, "X", {"gross_margin": 0.412}, "2026-06-30")
    assert evidence.current_metric(store, "X", "gross_margin") == pytest.approx(41.2)


def test_an_unbounded_fraction_metric_may_exceed_100_percent(store):
    """Growth, ROE and FCF conversion genuinely pass 100%. A guard firing on a
    real 200% growth figure would only teach the operator to bypass it."""
    evidence.record_reported(store, "X", {"revenue_growth": 2.0}, "2026-06-30")
    assert evidence.current_metric(store, "X", "revenue_growth") == pytest.approx(200.0)


def test_percent_leaves_non_fraction_metrics_alone(store):
    """--percent is about the fraction convention. A P/E of 20 is 20 either way."""
    evidence.record_reported(store, "X", {"pe_ratio": 20.0}, "2026-06-30", percent=True)
    assert (store.get_fundamentals("X") or {})["pe_ratio"] == 20.0


def test_a_custom_metric_is_unaffected_by_percent(store):
    """Custom metrics are already entered as printed, so the flag must not
    convert them twice."""
    evidence.define_custom_metric(store, "organic_growth", "Organic growth", "%")
    evidence.record_reported(store, "X", {"organic_growth": 6.4}, "2026-06-30",
                             percent=True)
    assert evidence.current_metric(store, "X", "organic_growth") == 6.4


def test_a_refused_figure_records_nothing_at_all(store):
    """One bad figure must not half-write the batch it arrived in."""
    with pytest.raises(ValueError):
        evidence.record_reported(
            store, "X", {"pe_ratio": 20.0, "gross_margin": 41.2}, "2026-06-30")
    assert store.get_fundamentals("X") in (None, {})


def test_an_undefined_metric_is_not_recorded(store):
    """Silently accepting an unknown key would create a figure no criterion can
    ever name, and no error to say why."""
    assert evidence.record_reported(store, "X", {"made_up": 1.0}, "2026-06-30") == {}
