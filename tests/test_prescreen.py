"""The local pre-screen: its client, the MFN reader, the selection rule, and the run."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest
from click.testing import CliRunner

from fundmgr import cli as commands
from fundmgr.config import AppConfig, PrescreenConfig, UniverseTicker
from fundmgr.data import mfn
from fundmgr.data.prices import TickerFeatures
from fundmgr.engine import prescreen
from fundmgr.engine.client import LLMError, _clean_local_json, call_local_json
from fundmgr.engine.prescreen import PrescreenBatch, PrescreenTag
from fundmgr.state.store import Store

NOW = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)


def feat(ticker: str, headlines: list[dict] | None = None, ret20: float = 0.0) -> TickerFeatures:
    f = TickerFeatures(ticker, f"{ticker} AB", 100.0, "2026-10-29", 0, sector="Tech", country="SE",
                       return_20d_pct=ret20)
    f.headlines = headlines or []
    return f


def tag(ticker, material=True, kind="report", direction="positive"):
    return PrescreenTag(ticker=ticker, material_news=material, news_type=kind, direction=direction)


# ── The local client ──────────────────────────────────────────────────────────

def test_a_reasoning_models_think_block_and_fence_are_stripped():
    raw = '<think>The user wants JSON {not this}</think>\n```json\n{"tags": []}\n```'
    assert json.loads(_clean_local_json(raw)) == {"tags": []}


@pytest.fixture
def fake_openai(monkeypatch):
    seen = {}

    class FakeOpenAI:
        def __init__(self, **kw):
            seen["client"] = kw
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kw):
            seen["request"] = kw
            return SimpleNamespace(choices=[SimpleNamespace(
                finish_reason=seen.get("finish", "stop"),
                message=SimpleNamespace(content=seen.get("reply", '{"tags": []}')))])

    import openai
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    return seen


def test_the_local_client_asks_for_deterministic_json(fake_openai):
    pcfg = PrescreenConfig(base_url="http://studio:11434/v1", model_id="qwen3:32b")
    fake_openai["reply"] = '<think>…</think>{"tags": [{"ticker": "alfa.st", "material_news": true, ' \
                           '"news_type": "report", "direction": "positive"}]}'
    parsed, _raw = call_local_json("sys", "user", pcfg, PrescreenBatch)
    assert parsed.tags[0].ticker == "ALFA.ST"
    req = fake_openai["request"]
    assert req["temperature"] == 0.0 and req["seed"] == 0
    assert req["response_format"] == {"type": "json_object"}
    assert fake_openai["client"]["base_url"] == "http://studio:11434/v1"
    assert fake_openai["client"]["max_retries"] == 0


def test_the_local_client_refuses_a_truncated_or_unconfigured_call(fake_openai):
    with pytest.raises(LLMError, match="must be set"):
        call_local_json("s", "u", PrescreenConfig(), PrescreenBatch)
    fake_openai["finish"] = "length"
    with pytest.raises(LLMError, match="max_tokens"):
        call_local_json("s", "u", PrescreenConfig(base_url="http://x/v1", model_id="m"),
                        PrescreenBatch)


# ── Selection ─────────────────────────────────────────────────────────────────

def test_selection_keeps_holdings_and_adds_tag_weights_to_the_score():
    scores = {"A": 4.5, "B": 4.0, "C": 3.0, "HELD": -9.0}   # A's placement news costs it 1
    weights = {"material_positive": 3.0, "index_or_flow": -1.0}
    tags = {"C": tag("C"), "A": tag("A", kind="index_or_flow", direction="negative")}
    assert prescreen.select(scores, tags, weights, top_n=3, forced={"HELD"}) == ["HELD", "C", "B"]


def test_a_material_tag_with_no_direction_or_no_news_adds_nothing():
    weights = {"material_positive": 3.0}
    assert prescreen.tag_weight(tag("A", material=False), weights) == 0.0
    assert prescreen.tag_weight(tag("A", direction="none"), weights) == 0.0
    assert prescreen.tag_weight(None, weights) == 0.0


# ── The batch run ─────────────────────────────────────────────────────────────

def test_a_failed_batch_leaves_its_names_untagged_and_the_run_partial():
    pool = {t: feat(t, ret20=s) for t, s in (("A", 30), ("B", 20), ("C", 10), ("D", 0))}
    pcfg = PrescreenConfig(batch_size=2)

    def call(system, user, cfg, schema):
        if "## A" in user:
            return schema(tags=[tag("A"), tag("B", material=False, kind="none", direction="none"),
                                tag("ZZZ")]), "{}"
        raise LLMError("server busy")

    result = prescreen.run_prescreen(pool, ["A", "B"], set(), pcfg, top_n=2, call=call, now=NOW)
    assert result.status == "partial"
    assert set(result.tags) == {"A", "B"}             # ZZZ wasn't asked about
    assert result.calls[1]["error"] == "server busy"


def test_every_batch_failing_falls_back_to_the_mechanical_list():
    pool = {t: feat(t) for t in "ABC"}

    def call(*_a):
        raise LLMError("Connection refused\nmore detail")

    result = prescreen.run_prescreen(pool, ["B", "A"], set(), PrescreenConfig(), top_n=2,
                                     call=call, now=NOW)
    assert (result.status, result.selected, result.error) == ("unavailable", ["B", "A"],
                                                               "Connection refused")


def test_a_candidate_block_is_dated_and_keeps_to_the_window():
    since = NOW - timedelta(days=14)
    f = feat("A", headlines=[{"headline": "Fresh order", "published_at": "2026-10-28T08:00:00"},
                             {"headline": "Old news", "published_at": "2026-09-01T08:00:00"}])
    release = mfn.Release("A AB Q3 report", "https://mfn.se/a/a/q3", NOW - timedelta(days=1), "en",
                          (":regulatory", "sub:report:interim"), "Sales up 12%")
    block = prescreen.candidate_block(f, [release], since)
    assert "2026-10-28 news: Fresh order" in block and "Old news" not in block
    assert "press release (regulatory, report): A AB Q3 report — Sales up 12%" in block
    assert "(no news in the window)" in prescreen.candidate_block(feat("B"), [], since)


def test_the_record_keeps_both_lists_and_the_rule(tmp_path):
    cfg = AppConfig(db_path=tmp_path / "fund.db")
    cfg.prescreen = PrescreenConfig(base_url="http://studio:11434/v1", model_id="qwen")
    pool = {t: feat(t) for t in "AB"}
    result = prescreen.PrescreenResult("ok", ["B", "A"], {"B": tag("B")}, [{"tickers": ["A", "B"]}])
    path = prescreen.write_record(cfg, "r1", "shadow", result, pool, ["A", "B"], {"A"})
    rec = json.loads(path.read_text())
    assert path == tmp_path / "prescreen" / "fund" / "r1.json"
    assert rec["mechanical"] == ["A", "B"] and rec["selected"] == ["B", "A"]
    assert rec["model"]["host"] == "studio:11434" and rec["weights"]["material_positive"] == 3.0
    assert rec["tags"]["B"]["news_type"] == "report" and rec["forced"] == ["A"]


# ── MFN ───────────────────────────────────────────────────────────────────────

FEED = """<?xml version="1.0" encoding="UTF-8"?><rss version="2.0">
<channel xmlns:x="https://mfn.se/schemas/rss-ns-x/">
<item><title>Alfa Interim Report Q3</title><link>https://mfn.se/a/alfa/q3-en</link>
<description><![CDATA[<p>Sales <b>up</b> 12%&nbsp;&amp; margin up</p>]]></description>
<pubDate>Fri, 30 Oct 2026 07:30:00 +0000</pubDate><x:language>en</x:language>
<x:tag>:regulatory</x:tag><x:tag>sub:report:interim</x:tag></item>
<item><title>Alfa Delårsrapport Q3</title><link>https://mfn.se/a/alfa/q3-sv</link>
<description>Försäljningen ökade</description>
<pubDate>Fri, 30 Oct 2026 07:30:00 +0000</pubDate><x:language>sv</x:language>
<x:tag>:regulatory</x:tag><x:tag>sub:report:interim</x:tag></item>
<item><title>Alfa kallar till årsstämma</title><link>https://mfn.se/a/alfa/agm</link>
<description>Kallelse</description>
<pubDate>Mon, 01 Sep 2026 07:30:00 +0000</pubDate><x:language>sv</x:language>
<x:tag>sub:ci</x:tag></item>
</channel></rss>"""


def test_a_feed_keeps_english_where_a_release_comes_in_two_languages():
    releases = mfn.parse_feed(FEED.encode())
    assert [r.title for r in releases] == ["Alfa Interim Report Q3", "Alfa kallar till årsstämma"]
    first = releases[0]
    assert first.report and first.regulatory and first.summary == "Sales up 12% & margin up"


def mfn_client(log: list, hits: dict, feeds: dict) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        log.append(str(request.url))
        if request.url.path == "/search/companies":
            return httpx.Response(200, json=hits.get(request.url.params["query"], []))
        slug = request.url.path.split("/")[-1].removesuffix(".rss")
        return httpx.Response(200, content=feeds.get(slug, b"<rss><channel/></rss>"))
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_a_company_resolves_only_on_an_exact_isin_match():
    hits = {"H&M": [{"slug": "hynion", "isins": ["NO0010920945"]},
                    {"slug": "handm", "isins": ["SE0000106270", "SE0000106205"]}],
            "Volvo": [{"slug": "volvo-cars", "isins": ["SE0021628898"]}]}
    with mfn_client([], hits, {}) as client:
        assert mfn.resolve_slug("H & M Hennes & Mauritz AB (publ)", "se0000106270", client) == "handm"
        assert mfn.resolve_slug("AB Volvo (publ)", "SE0000115446", client) is None


def test_releases_resolve_once_then_read_the_cached_slug(tmp_path):
    row = UniverseTicker("Alfa AB", "ALFA-B.ST", "SE0000000001", "SE", "OMXS", "Tech", True)
    plain = UniverseTicker("Beta AB", "BETA.ST", "", "SE", "OMXS", "Tech", True)
    hits = {"Alfa": [{"slug": "alfa", "isins": ["SE0000000001"]}]}
    since = NOW - timedelta(days=14)
    log: list = []
    with mfn_client(log, hits, {"alfa": FEED.encode()}) as client:
        out = mfn.releases([row, plain], since, tmp_path / "slugs.json", client=client, pause=0)
        assert [r.title for r in out["ALFA-B.ST"]] == ["Alfa Interim Report Q3"]   # AGM is too old
        searches = sum("/search/" in u for u in log)
        mfn.releases([row], since, tmp_path / "slugs.json", client=client, pause=0)
    assert searches == 1 and sum("/search/" in u for u in log) == 1
    assert json.loads((tmp_path / "slugs.json").read_text())["SE0000000001"]["slug"] == "alfa"


# ── In the weekly run ─────────────────────────────────────────────────────────

@pytest.fixture
def weekly(tmp_path, monkeypatch):
    """A run of four equal names, two slots, and a pre-screen that likes DDD."""
    path = tmp_path / "universe.csv"
    path.write_text("name,yahoo_ticker,isin,country,exchange,sector,enabled\n"
                    + "".join(f"{t} AB,{t}.ST,,SE,OMXS,Tech,true\n" for t in ("AAA", "BBB", "CCC", "DDD")))
    cfg = AppConfig(name="Test", db_path=tmp_path / "fund.db", universe_path=path)
    cfg.screener.top_n = 2
    cfg.prescreen = PrescreenConfig(enabled=True, base_url="http://studio:11434/v1",
                                    model_id="qwen", pool_size=4, mfn=False)
    store = Store(cfg.db_path)
    store.initialise(100000)
    seen: dict = {}
    today = datetime.utcnow().strftime("%Y-%m-%d")

    def features(tickers, *_a):
        return {t.yahoo_ticker: TickerFeatures(t.yahoo_ticker, t.name, 100.0, today, 0,
                                               sector="Tech", country="SE") for t in tickers}

    def prompt(cfg_, snap, screened, *_a, **_k):
        seen["shown"] = sorted(screened)
        return "system", "user", {}

    def local(system, user, pcfg, schema):
        return schema(tags=[tag("DDD.ST")]), "{}"

    def model(*_a):
        raise LLMError("stopped by the test")

    monkeypatch.setattr(commands, "_get_store", lambda cfg_=None: (cfg, store))
    monkeypatch.setattr(commands, "fetch_and_cache_prices",
                        lambda tickers, *_a: {t.yahoo_ticker: True for t in tickers})
    monkeypatch.setattr(commands, "fetch_and_cache_benchmark", lambda *_a, **_k: True)
    monkeypatch.setattr(commands, "build_all_features", features)
    monkeypatch.setattr(commands, "apply_to_features", lambda *_a: None)
    monkeypatch.setattr(commands, "attach_sentiment_to_features", lambda *_a, **_k: None)
    monkeypatch.setattr(commands, "build_prompt", prompt)
    monkeypatch.setattr(commands, "call_llm_consensus", model)
    monkeypatch.setattr("fundmgr.engine.prescreen.call_local_json", local)
    monkeypatch.setattr("fundmgr.notify.send.send_telegram", lambda *a, **k: True)

    def go():
        return CliRunner().invoke(commands.cli, ["run", "--force", "--skip-news", "--skip-macro",
                                                 "--skip-fundamentals"])
    return cfg, seen, go


def test_in_shadow_the_model_still_gets_the_screeners_list_and_both_are_recorded(weekly):
    cfg, seen, go = weekly
    go()
    assert seen["shown"] == ["AAA.ST", "BBB.ST"]
    [record] = (cfg.db_path.parent / "prescreen" / "fund").glob("*.json")
    rec = json.loads(record.read_text())
    assert rec["mode"] == "shadow" and rec["status"] == "ok"
    assert rec["mechanical"] == ["AAA.ST", "BBB.ST"] and rec["selected"][0] == "DDD.ST"


def test_live_the_model_gets_the_prescreens_list(weekly):
    cfg, seen, go = weekly
    cfg.prescreen.shadow = False
    go()
    assert "DDD.ST" in seen["shown"] and len(seen["shown"]) == 2


def test_live_under_a_regional_mix_stays_shadow(weekly):
    cfg, seen, go = weekly
    cfg.prescreen.shadow = False
    cfg.risk.region_targets = {"nordics": 100}
    go()
    assert "DDD.ST" not in seen["shown"]
    [record] = (cfg.db_path.parent / "prescreen" / "fund").glob("*.json")
    assert "mix" in json.loads(record.read_text())["note"]


def test_switched_off_the_run_is_untouched(weekly):
    cfg, seen, go = weekly
    cfg.prescreen.enabled = False
    go()
    assert seen["shown"] == ["AAA.ST", "BBB.ST"]
    assert not (cfg.db_path.parent / "prescreen").exists()
