"""Reading company-reported figures out of release coverage.

The figures that matter most to an ADD criterion — organic growth, adjusted
EBIT margin, ARR — are not in any provider feed, so they get read out of what
the company published. That makes the extraction a place where a plausible
invention would be indistinguishable from a fact, which is what most of these
tests are about.
"""
import json

import pytest

from fundmgr import evidence
from fundmgr.data import release
from fundmgr.state.store import Store

# A Systemair-shaped Q2 write-up: two figures stated plainly, one absent, and
# a revenue number that a careless reader could turn into a margin.
COVERAGE = (
    "Systemair reported its second quarter on 30 June 2026. "
    "Organic growth was 6.4 per cent in the quarter, up from 4.1 per cent a year "
    "earlier. The adjusted EBIT margin came in at 9.3 per cent. "
    "Net sales reached 3 145 MSEK. The company did not disclose order intake."
)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "book.db")
    s.initialise(100_000)
    evidence.define_custom_metric(s, "organic_growth", "Organic growth", "%")
    evidence.define_custom_metric(s, "adj_ebit_margin", "Adj. EBIT margin", "%")
    evidence.define_custom_metric(s, "order_intake", "Order intake", "SEK")
    return s


@pytest.fixture
def items():
    return [{"headline": "Systemair Q2", "publisher": "Placera",
             "published": "2026-07-02", "url": "https://example.test/q2",
             "summary": "", "body": COVERAGE}]


@pytest.fixture
def fake_llm(monkeypatch):
    seen = {}

    class _Msg:
        def __init__(self, content): self.content = content

    class _Choice:
        def __init__(self, content): self.message = _Msg(content)

    class _Resp:
        def __init__(self, content): self.choices = [_Choice(content)]

    class _Completions:
        def create(self, **kw):
            seen["kwargs"] = kw
            return _Resp(seen["answer"])

    class _Chat:
        completions = _Completions()

    class _Client:
        def __init__(self, api_key=None): self.chat = _Chat()

    import openai
    monkeypatch.setattr(openai, "OpenAI", _Client)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen["answer"] = json.dumps({"figures": [
        {"metric": "organic_growth", "value": 6.4, "period": "Q2 2026",
         "quote": "Organic growth was 6.4 per cent in the quarter",
         "confidence": 0.95},
        {"metric": "adj_ebit_margin", "value": 9.3, "period": "Q2 2026",
         "quote": "The adjusted EBIT margin came in at 9.3 per cent",
         "confidence": 0.9},
    ], "notes": ""})
    return seen


# ── Extraction ────────────────────────────────────────────────────────────────

def test_stated_figures_are_extracted_with_their_quotes(store, items, fake_llm):
    out = release.extract(store, "SYSR.ST", items, "2026-06-30")
    by_metric = {f["metric"]: f for f in out["figures"]}

    assert by_metric["organic_growth"]["value"] == 6.4
    assert by_metric["organic_growth"]["unit"] == "%"
    assert "6.4 per cent" in by_metric["organic_growth"]["quote"]
    assert by_metric["organic_growth"]["quote_verbatim"] is True
    assert by_metric["adj_ebit_margin"]["value"] == 9.3


def test_a_metric_the_text_does_not_state_is_simply_absent(store, items, fake_llm):
    """"The company did not disclose order intake" must leave the criterion
    honestly unread, not filled with a guess."""
    out = release.extract(store, "SYSR.ST", items, "2026-06-30")
    assert "order_intake" not in {f["metric"] for f in out["figures"]}


def test_a_figure_not_in_the_source_is_discarded(store, items, fake_llm):
    """The check that matters: a model told not to invent can still invent."""
    fake_llm["answer"] = json.dumps({"figures": [
        {"metric": "organic_growth", "value": 8.8, "period": "Q2 2026",
         "quote": "Organic growth was 8.8 per cent", "confidence": 0.95}], "notes": ""})

    out = release.extract(store, "SYSR.ST", items, "2026-06-30")
    assert out["figures"] == []
    assert out["rejected"][0]["value"] == 8.8
    assert "does not appear in the source" in out["rejected"][0]["reason"]


def test_a_rejected_figure_is_visible_rather_than_silent(store, items, fake_llm):
    """Dropping it quietly would leave the investor thinking nothing was found."""
    fake_llm["answer"] = json.dumps({"figures": [
        {"metric": "organic_growth", "value": 6.4, "quote": "…", "confidence": 0.9},
        {"metric": "adj_ebit_margin", "value": 99.9, "quote": "…", "confidence": 0.9},
    ], "notes": ""})
    out = release.extract(store, "SYSR.ST", items, "2026-06-30")
    assert [f["value"] for f in out["figures"]] == [6.4]
    assert [f["value"] for f in out["rejected"]] == [99.9]


def test_a_nordic_decimal_comma_still_counts_as_present(store, items, fake_llm):
    """Swedish releases print 6,4 — the same number, and it must not be
    discarded as a fabrication."""
    items[0]["body"] = "Den organiska tillväxten var 6,4 procent i kvartalet."
    fake_llm["answer"] = json.dumps({"figures": [
        {"metric": "organic_growth", "value": 6.4, "quote": "6,4 procent",
         "confidence": 0.9}], "notes": ""})
    out = release.extract(store, "SYSR.ST", items, "2026-06-30")
    assert [f["value"] for f in out["figures"]] == [6.4]


def test_a_thousands_separated_figure_counts_as_present(store, items, fake_llm):
    evidence.define_custom_metric(store, "arr", "ARR", "SEK")
    items[0]["body"] = "ARR amounted to 1 240 000 SEK at the end of the quarter."
    fake_llm["answer"] = json.dumps({"figures": [
        {"metric": "arr", "value": 1240000, "quote": "ARR amounted to 1 240 000 SEK",
         "confidence": 0.9}], "notes": ""})
    out = release.extract(store, "SYSR.ST", items, "2026-06-30")
    assert [f["value"] for f in out["figures"]] == [1240000]


def test_a_paraphrased_quote_is_kept_but_flagged(store, items, fake_llm):
    """The number is in the text, the sentence was reworded — usable, but the
    investor should see that the quote is not verbatim."""
    fake_llm["answer"] = json.dumps({"figures": [
        {"metric": "organic_growth", "value": 6.4,
         "quote": "organic growth came to 6.4%", "confidence": 0.9}], "notes": ""})
    out = release.extract(store, "SYSR.ST", items, "2026-06-30")
    assert out["figures"][0]["quote_verbatim"] is False


def test_an_unknown_metric_key_is_ignored(store, items, fake_llm):
    fake_llm["answer"] = json.dumps({"figures": [
        {"metric": "invented_metric", "value": 6.4, "quote": "6.4", "confidence": 1.0}],
        "notes": ""})
    assert release.extract(store, "SYSR.ST", items, "2026-06-30")["figures"] == []


def test_a_provider_metric_is_not_extracted(store, items, fake_llm):
    """This path exists for the figures no feed carries. Letting it overwrite
    revenue_growth would put a journalist's paraphrase where a provider's
    number belongs."""
    fake_llm["answer"] = json.dumps({"figures": [
        {"metric": "revenue_growth", "value": 6.4, "quote": "6.4 per cent",
         "confidence": 1.0}], "notes": ""})
    out = release.extract(store, "SYSR.ST", items, "2026-06-30")
    assert out["figures"] == []
    prompt = fake_llm["kwargs"]["messages"][1]["content"]
    assert "revenue_growth" not in prompt
    assert "organic_growth" in prompt


def test_no_custom_metrics_means_nothing_to_look_for(tmp_path, items, fake_llm):
    bare = Store(tmp_path / "bare.db")
    bare.initialise(100)
    out = release.extract(bare, "SYSR.ST", items, "2026-06-30")
    assert out["figures"] == [] and "no book-specific metrics" in out["notes"]


def test_a_junk_reply_yields_nothing(store, items, fake_llm):
    for junk in ("not json", "{}", '{"figures": "nope"}'):
        fake_llm["answer"] = junk
        assert release.extract(store, "SYSR.ST", items, "2026-06-30")["figures"] == []


def test_a_failed_call_degrades_quietly(store, items, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    import openai

    def boom(api_key=None):
        raise RuntimeError("network down")
    monkeypatch.setattr(openai, "OpenAI", boom)
    out = release.extract(store, "SYSR.ST", items, "2026-06-30")
    assert out["figures"] == [] and "failed" in out["notes"]


def test_without_an_api_key_nothing_is_attempted(store, items, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert release.extract(store, "SYSR.ST", items, "2026-06-30")["figures"] == []


# ── Finding the coverage ──────────────────────────────────────────────────────

def test_coverage_is_filtered_to_the_window_around_the_report(monkeypatch):
    monkeypatch.setattr(evidence, "news_items", lambda t, max_items=8: [
        {"headline": "Q2 report", "published": "2026-07-02", "summary": "…"},
        {"headline": "Last quarter's write-up", "published": "2026-04-02", "summary": "…"},
        {"headline": "Undated wire copy", "published": "", "summary": "…"},
    ])
    monkeypatch.setattr(evidence, "enrich_with_bodies", lambda items, max_articles=5: 0)

    heads = [i["headline"] for i in release.find_coverage("SYSR.ST", "2026-06-30")]
    assert "Q2 report" in heads
    assert "Last quarter's write-up" not in heads
    # Undated items are kept: dropping them would discard the release itself on
    # feeds that omit timestamps.
    assert "Undated wire copy" in heads


def test_no_coverage_is_reported_as_such(store, monkeypatch):
    monkeypatch.setattr(evidence, "news_items", lambda t, max_items=8: [])
    out = release.read_report(store, "SYSR.ST", "2026-06-30")
    assert out["figures"] == [] and "no coverage found" in out["notes"]


# ── Nothing is written without confirmation ───────────────────────────────────

def test_extraction_stores_nothing_on_its_own(store, items, fake_llm):
    release.extract(store, "SYSR.ST", items, "2026-06-30")
    assert store.get_fundamentals("SYSR.ST") in (None, {})
    assert evidence.current_metric(store, "SYSR.ST", "organic_growth") is None


def test_confirmed_figures_become_a_reported_quarter(store, items, fake_llm):
    """The path end to end: extract, confirm, and the ADD criterion can see it."""
    out = release.extract(store, "SYSR.ST", items, "2026-06-30")
    evidence.record_reported(store, "SYSR.ST",
                             {f["metric"]: f["value"] for f in out["figures"]},
                             "2026-06-30")

    assert evidence.current_metric(store, "SYSR.ST", "organic_growth") == 6.4
    history = evidence.metric_history(store, "SYSR.ST", "adj_ebit_margin")
    assert [(r["period"], r["value"]) for r in history] == [("2026-06-30", 9.3)]
