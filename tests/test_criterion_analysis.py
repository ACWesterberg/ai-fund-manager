"""Reading a kill criterion back at save time, and the rules lifted from it."""
import json

import pytest

from fundmgr import evidence, paper, watchplan
from fundmgr.state.models import Transaction
from fundmgr.state.store import Store

# The criterion that motivated all of this: two legs that need figures, one
# that no feed will ever settle.
SERIAL_ACQUIRER = ("Recurring growth falls below 8% while EBITA/FCF deteriorates, "
                   "or acquisitions begin generating clearly weaker returns")

ANALYSIS_ANSWER = json.dumps({
    "logic": "(A and B) or C",
    "conditions": [
        {"id": "A", "text": "recurring growth falls below 8%",
         "checkable": "fundamentals", "metric": "revenue_growth", "op": "below",
         "value": 8, "note": "total revenue growth, not recurring specifically"},
        {"id": "B", "text": "EBITA/FCF deteriorates",
         "checkable": "fundamentals", "metric": "profit_margin", "op": "below",
         "value": 9, "note": "net margin — EBITA and FCF are not in the data set"},
        {"id": "C", "text": "acquisitions begin generating clearly weaker returns",
         "checkable": "manual", "metric": None, "op": None, "value": None,
         "note": "deal-level returns are not disclosed in any feed"},
    ],
})


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "book.db")
    s.initialise(100_000)
    s.set_meta("paper_name", "Analysis Sleeve")
    return s


@pytest.fixture
def fake_llm(monkeypatch):
    """Stub the analyser's model call; record what it was asked."""
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
            return _Resp(seen.get("answer", ANALYSIS_ANSWER))

    class _Chat:
        completions = _Completions()

    class _Client:
        def __init__(self, api_key=None): self.chat = _Chat()

    import openai
    monkeypatch.setattr(openai, "OpenAI", _Client)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    return seen


# ── Analysis ──────────────────────────────────────────────────────────────────

def test_analyse_decomposes_a_compound_criterion(fake_llm):
    out = watchplan.analyse_criterion(SERIAL_ACQUIRER)
    assert out["logic"] == "(A and B) or C"
    assert [c["checkable"] for c in out["conditions"]] == [
        "fundamentals", "fundamentals", "manual"]
    assert out["conditions"][0]["metric"] == "revenue_growth"
    assert out["conditions"][0]["value"] == 8
    assert "not recurring specifically" in out["conditions"][0]["note"]
    assert out["criterion"] == SERIAL_ACQUIRER


def test_analyse_offers_only_real_metrics(fake_llm):
    """The model is shown the metric catalogue so it can't invent a key."""
    watchplan.analyse_criterion(SERIAL_ACQUIRER)
    prompt = fake_llm["kwargs"]["messages"][1]["content"]
    for key in ("revenue_growth", "profit_margin", "ev_to_ebitda"):
        assert key in prompt
    assert SERIAL_ACQUIRER in prompt


def test_analyse_rejects_an_unknown_metric(fake_llm):
    """A hallucinated metric key must not become a rule — it degrades to news."""
    fake_llm["answer"] = json.dumps({"logic": "A", "conditions": [
        {"id": "A", "text": "churn rises above 5%", "checkable": "fundamentals",
         "metric": "customer_churn", "op": "above", "value": 5},
    ]})
    out = watchplan.analyse_criterion("churn rises above 5%")
    assert out["conditions"][0]["checkable"] == "news"
    assert out["conditions"][0]["metric"] is None
    assert watchplan.suggested_rules(out) == []


def test_analyse_rejects_an_incomplete_comparison(fake_llm):
    fake_llm["answer"] = json.dumps({"logic": "A", "conditions": [
        {"id": "A", "text": "margins fall", "checkable": "fundamentals",
         "metric": "gross_margin", "op": None, "value": None},
    ]})
    out = watchplan.analyse_criterion("margins fall")
    assert out["conditions"][0]["checkable"] == "news"


def test_analyse_keeps_a_zero_threshold(fake_llm):
    """'FCF below 0' is a real line — the positive-only coercion must not eat it."""
    fake_llm["answer"] = json.dumps({"logic": "A", "conditions": [
        {"id": "A", "text": "revenue growth turns negative", "checkable": "fundamentals",
         "metric": "revenue_growth", "op": "below", "value": 0},
    ]})
    out = watchplan.analyse_criterion("revenue growth turns negative")
    assert out["conditions"][0]["value"] == 0
    assert watchplan.suggested_rules(out)[0]["value"] == 0


def test_analyse_without_an_api_key_is_none(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert watchplan.analyse_criterion(SERIAL_ACQUIRER) is None


def test_analyse_survives_a_failed_call(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    import openai

    def boom(api_key=None):
        raise RuntimeError("network down")
    monkeypatch.setattr(openai, "OpenAI", boom)
    assert watchplan.analyse_criterion(SERIAL_ACQUIRER) is None


def test_analyse_handles_junk_replies(fake_llm):
    for junk in ("not json", "{}", '{"conditions": []}', '{"conditions": "nope"}'):
        fake_llm["answer"] = junk
        assert watchplan.analyse_criterion("x") is None


def test_analyse_reads_json_out_of_prose(fake_llm):
    fake_llm["answer"] = f"Sure, here it is:\n```json\n{ANALYSIS_ANSWER}\n```"
    assert watchplan.analyse_criterion(SERIAL_ACQUIRER)["logic"] == "(A and B) or C"


# ── Storage ───────────────────────────────────────────────────────────────────

def test_analysis_round_trips(store, fake_llm):
    watchplan.set_position_plan(store, "VIT-B.ST", kill_criterion=SERIAL_ACQUIRER)
    watchplan.save_analysis(store, "VIT-B.ST", watchplan.analyse_criterion(SERIAL_ACQUIRER))
    assert len(watchplan.get_analysis(store, "VIT-B.ST")["conditions"]) == 3


def test_stale_analysis_is_discarded_after_a_reword(store, fake_llm):
    """An analysis of the old wording must never describe the new one."""
    watchplan.set_position_plan(store, "VIT-B.ST", kill_criterion=SERIAL_ACQUIRER)
    watchplan.save_analysis(store, "VIT-B.ST", watchplan.analyse_criterion(SERIAL_ACQUIRER))
    watchplan.set_position_plan(store, "VIT-B.ST",
                                kill_criterion="Recurring growth below 8% for two quarters")
    assert watchplan.get_analysis(store, "VIT-B.ST") is None


def test_clearing_a_plan_drops_the_analysis(store, fake_llm):
    watchplan.set_position_plan(store, "VIT-B.ST", kill_criterion=SERIAL_ACQUIRER)
    watchplan.save_analysis(store, "VIT-B.ST", watchplan.analyse_criterion(SERIAL_ACQUIRER))
    watchplan.clear_position_plan(store, "VIT-B.ST")
    assert watchplan.get_analysis(store, "VIT-B.ST") is None


def test_suggested_rules_only_offers_the_checkable_legs(fake_llm):
    rules = watchplan.suggested_rules(watchplan.analyse_criterion(SERIAL_ACQUIRER))
    assert [r["metric"] for r in rules] == ["revenue_growth", "profit_margin"]
    assert rules[0]["label"] == "Revenue growth (yoy)"
    assert watchplan.suggested_rules(None) == []


# ── Fundamentals rules ────────────────────────────────────────────────────────

def _hold(store, ticker="VIT-B.ST", shares=100, cost=300.0):
    store.apply_fill(Transaction(ticker=ticker, side="buy", shares=shares,
                                 price_sek=cost, fee_sek=0.0, source="fill"))


@pytest.fixture
def sek_only(monkeypatch):
    monkeypatch.setattr(paper, "sek_prices_for",
                        lambda store, tickers, currency_map, native: dict(native))


def test_current_metric_converts_fractions_to_percent(store):
    store.save_fundamentals("VIT-B.ST", {"revenue_growth": 0.071, "pe_ratio": 21.4})
    assert evidence.current_metric(store, "VIT-B.ST", "revenue_growth") == pytest.approx(7.1)
    assert evidence.current_metric(store, "VIT-B.ST", "pe_ratio") == pytest.approx(21.4)
    assert evidence.current_metric(store, "VIT-B.ST", "nonsense") is None


def test_fundamentals_rule_fires_below_threshold(store, sek_only):
    _hold(store)
    store.save_fundamentals("VIT-B.ST", {"revenue_growth": 0.071})
    watchplan.set_position_plan(store, "VIT-B.ST", fundamentals=[
        {"metric": "revenue_growth", "op": "below", "value": 8}])

    row = watchplan.evaluate_kill_rules(store)[0]
    assert row["hit"]
    assert "Revenue growth (yoy) 7.1% is below the 8.0% line" in row["reasons"][0]


def test_fundamentals_rule_stays_quiet_above_threshold(store, sek_only):
    _hold(store)
    store.save_fundamentals("VIT-B.ST", {"revenue_growth": 0.121})
    watchplan.set_position_plan(store, "VIT-B.ST", fundamentals=[
        {"metric": "revenue_growth", "op": "below", "value": 8}])
    assert watchplan.evaluate_kill_rules(store)[0]["hit"] is False


def test_fundamentals_rule_supports_above(store, sek_only):
    _hold(store)
    store.save_fundamentals("VIT-B.ST", {"ev_to_ebitda": 31.0})
    watchplan.set_position_plan(store, "VIT-B.ST", fundamentals=[
        {"metric": "ev_to_ebitda", "op": "above", "value": 30}])
    assert watchplan.evaluate_kill_rules(store)[0]["hit"]


def test_uncached_metric_is_unread_not_passing(store, sek_only):
    """No data must never read as 'the line is fine'."""
    _hold(store)
    watchplan.set_position_plan(store, "VIT-B.ST", fundamentals=[
        {"metric": "revenue_growth", "op": "below", "value": 8}])
    row = watchplan.evaluate_kill_rules(store)[0]
    assert row["hit"] is False
    assert row["fundamentals"][0]["unread"] is True


def test_malformed_fundamentals_rules_are_dropped(store):
    watchplan.set_position_plan(store, "VIT-B.ST", fundamentals=[
        {"metric": "made_up", "op": "below", "value": 8},
        {"metric": "revenue_growth", "op": "sideways", "value": 8},
        {"metric": "revenue_growth", "op": "below", "value": "abc"},
        "not a dict",
        {"metric": "revenue_growth", "op": "below", "value": 8},
        {"metric": "revenue_growth", "op": "below", "value": 9},   # duplicate key
    ])
    rules = watchplan.get_kill_rules(store)["VIT-B.ST"]["fundamentals"]
    assert rules == [{"metric": "revenue_growth", "op": "below", "value": 8.0,
                      "quarters": 1, "note": ""}]


# ── quarters: a duration in the criterion is a duration in the rule ───────────

def _quarter(store, period, **values):
    from fundmgr import evidence
    store.save_fundamentals("VIT-B.ST", {"fiscal_period_end": period, **values})
    evidence.snapshot_fundamentals(store, ["VIT-B.ST"], today=period)


def test_one_bad_quarter_does_not_fire_a_two_quarter_rule(store, sek_only):
    """The false positive this exists to remove: a single weak print firing a
    line the investor wrote as 'two quarters running'."""
    _hold(store)
    _quarter(store, "2026-03-31", revenue_growth=0.12)
    _quarter(store, "2026-06-30", revenue_growth=0.071)
    watchplan.set_position_plan(store, "VIT-B.ST", fundamentals=[
        {"metric": "revenue_growth", "op": "below", "value": 8, "quarters": 2}])

    row = watchplan.evaluate_kill_rules(store)[0]
    assert row["hit"] is False
    f = row["fundamentals"][0]
    assert f["pending"] is True        # breaching now, one quarter short
    assert f["streak"] == 1 and f["quarters"] == 2


def test_a_second_bad_quarter_fires_it(store, sek_only):
    _hold(store)
    _quarter(store, "2026-03-31", revenue_growth=0.075)
    _quarter(store, "2026-06-30", revenue_growth=0.071)
    watchplan.set_position_plan(store, "VIT-B.ST", fundamentals=[
        {"metric": "revenue_growth", "op": "below", "value": 8, "quarters": 2}])

    row = watchplan.evaluate_kill_rules(store)[0]
    assert row["hit"] is True
    assert "for 2 straight quarters" in row["reasons"][0]
    assert row["fundamentals"][0]["pending"] is False


def test_a_multi_quarter_rule_on_thin_history_is_not_a_pass(store, sek_only):
    _hold(store)
    _quarter(store, "2026-06-30", revenue_growth=0.01)
    watchplan.set_position_plan(store, "VIT-B.ST", fundamentals=[
        {"metric": "revenue_growth", "op": "below", "value": 8, "quarters": 2}])

    f = watchplan.evaluate_kill_rules(store)[0]["fundamentals"][0]
    assert f["hit"] is False
    assert f["thin_history"] is True and f["periods_seen"] == 1


def test_quarters_defaults_to_one_and_is_bounded(store):
    watchplan.set_position_plan(store, "VIT-B.ST", fundamentals=[
        {"metric": "revenue_growth", "op": "below", "value": 8},
        {"metric": "profit_margin", "op": "below", "value": 5, "quarters": 999},
        {"metric": "roe", "op": "below", "value": 5, "quarters": 0},
    ])
    rules = {r["metric"]: r["quarters"]
             for r in watchplan.get_kill_rules(store)["VIT-B.ST"]["fundamentals"]}
    assert rules == {"revenue_growth": 1, "profit_margin": watchplan._MAX_QUARTERS,
                     "roe": 1}


def test_the_analyser_carries_a_duration_into_the_rule(store):
    """A criterion that says 'two quarters running' must not become a rule that
    fires on one."""
    analysis = watchplan._parse_analysis(json.dumps({
        "logic": "A",
        "conditions": [{"id": "A", "text": "growth below 8% two quarters running",
                        "checkable": "fundamentals", "metric": "revenue_growth",
                        "op": "below", "value": 8, "quarters": 2, "note": ""}],
    }), "growth below 8% two quarters running")
    assert analysis["conditions"][0]["quarters"] == 2
    assert watchplan.suggested_rules(analysis)[0]["quarters"] == 2


def test_a_criterion_with_no_duration_stays_single_quarter(store):
    analysis = watchplan._parse_analysis(json.dumps({
        "logic": "A",
        "conditions": [{"id": "A", "text": "growth below 8%",
                        "checkable": "fundamentals", "metric": "revenue_growth",
                        "op": "below", "value": 8, "note": ""}],
    }), "growth below 8%")
    assert analysis["conditions"][0]["quarters"] == 1


def test_fundamentals_breach_keeps_the_watch_armed(store, sek_only, monkeypatch):
    """Price recovering must not re-arm a watch whose metric is still breached."""
    sent = []
    import fundmgr.notify.send as send_mod
    monkeypatch.setattr(send_mod, "send_telegram", lambda t, **k: sent.append(t) or True)

    _hold(store)
    store.save_fundamentals("VIT-B.ST", {"revenue_growth": 0.071})
    watchplan.set_position_plan(store, "VIT-B.ST", max_drawdown_pct="25",
                                fundamentals=[{"metric": "revenue_growth",
                                               "op": "below", "value": 8}])
    watchplan.check_kill_rules("slug", store=store)
    assert len(sent) == 1

    # Growth still below the line → no "re-armed" message, no repeat alert.
    assert watchplan.check_kill_rules("slug", store=store) == []

    store.save_fundamentals("VIT-B.ST", {"revenue_growth": 0.14})
    log = watchplan.check_kill_rules("slug", store=store)
    assert any("re-armed" in line for line in log)


def test_fundamentals_rules_survive_clearing_price_rules(store, sek_only):
    watchplan.set_position_plan(store, "VIT-B.ST", max_drawdown_pct="25",
                                fundamentals=[{"metric": "revenue_growth",
                                               "op": "below", "value": 8}])
    watchplan.set_position_plan(store, "VIT-B.ST", max_drawdown_pct="")
    rules = watchplan.get_kill_rules(store)["VIT-B.ST"]
    assert "max_drawdown_pct" not in rules
    assert rules["fundamentals"]


# ── Web flow ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch, fake_llm):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path / "paper")
    monkeypatch.setattr(paper, "detect_currency", lambda t: "SEK")
    monkeypatch.setattr(paper, "_cache_price_history", lambda store, tickers, **kw: None)
    monkeypatch.setattr(paper, "_search_symbol", lambda name: None)
    import fundmgr.data.benchmark as benchmark
    monkeypatch.setattr(benchmark, "fetch_and_cache_benchmark", lambda store, **kw: True)
    import fundmgr.data.quotes as quotes
    monkeypatch.setattr(quotes, "live_prices", lambda tickers: {t: 300.0 for t in tickers})

    from fundmgr.web.app import app
    paper.create_portfolio(
        "Acquirer Sleeve", 30_000, "",
        holdings_override=[{"ticker": "VIT-B.ST", "name": "Vitec", "weight_pct": 100,
                            "shares": 100, "avg_cost_sek": 300.0}],
        kind="live", seed_holdings=True)
    return TestClient(app)


def test_saving_a_criterion_analyses_it(client):
    client.post("/live/acquirer-sleeve/watchplan",
                data={"ticker": "VIT-B.ST", "kill_criterion": SERIAL_ACQUIRER},
                follow_redirects=False)

    _meta, store = paper.open_portfolio("acquirer-sleeve")
    analysis = watchplan.get_analysis(store, "VIT-B.ST")
    assert analysis["logic"] == "(A and B) or C"


def test_dashboard_shows_how_the_criterion_was_read(client):
    client.post("/live/acquirer-sleeve/watchplan",
                data={"ticker": "VIT-B.ST", "kill_criterion": SERIAL_ACQUIRER},
                follow_redirects=False)

    body = client.get("/live/acquirer-sleeve").text
    assert "read as (A and B) or C" in body
    assert "recurring growth falls below 8%" in body
    assert "acquisitions begin generating clearly weaker returns" in body
    assert "total revenue growth, not recurring specifically" in body   # the proxy warning
    assert "Also check 2 of these as a hard rule" in body


def test_applying_suggested_rules_creates_real_checks(client):
    client.post("/live/acquirer-sleeve/watchplan",
                data={"ticker": "VIT-B.ST", "kill_criterion": SERIAL_ACQUIRER},
                follow_redirects=False)
    r = client.post("/live/acquirer-sleeve/watchplan/apply-rules",
                    data={"ticker": "VIT-B.ST"}, follow_redirects=False)
    assert r.status_code == 303
    assert "ok=1" in r.headers["location"]

    _meta, store = paper.open_portfolio("acquirer-sleeve")
    rules = watchplan.get_kill_rules(store)["VIT-B.ST"]["fundamentals"]
    assert {r["metric"] for r in rules} == {"revenue_growth", "profit_margin"}
    # The text criterion is untouched — both halves now watch the same line.
    assert watchplan.get_kill_text(store)["VIT-B.ST"] == SERIAL_ACQUIRER


def test_applying_twice_does_not_duplicate(client):
    client.post("/live/acquirer-sleeve/watchplan",
                data={"ticker": "VIT-B.ST", "kill_criterion": SERIAL_ACQUIRER},
                follow_redirects=False)
    for _ in range(2):
        client.post("/live/acquirer-sleeve/watchplan/apply-rules",
                    data={"ticker": "VIT-B.ST"}, follow_redirects=False)

    _meta, store = paper.open_portfolio("acquirer-sleeve")
    assert len(watchplan.get_kill_rules(store)["VIT-B.ST"]["fundamentals"]) == 2
    # …and the button is gone once everything is applied.
    assert "as a hard rule" not in client.get("/live/acquirer-sleeve").text


def test_apply_rules_with_nothing_to_apply(client):
    r = client.post("/live/acquirer-sleeve/watchplan/apply-rules",
                    data={"ticker": "VIT-B.ST"}, follow_redirects=False)
    assert "ok=0" in r.headers["location"]


def test_apply_rules_404s_on_unknown_sleeve(client):
    r = client.post("/live/nope/watchplan/apply-rules", data={"ticker": "X"})
    assert r.status_code == 404


def test_clearing_the_criterion_clears_the_analysis(client):
    client.post("/live/acquirer-sleeve/watchplan",
                data={"ticker": "VIT-B.ST", "kill_criterion": SERIAL_ACQUIRER},
                follow_redirects=False)
    client.post("/live/acquirer-sleeve/watchplan",
                data={"ticker": "VIT-B.ST", "kill_criterion": ""},
                follow_redirects=False)

    _meta, store = paper.open_portfolio("acquirer-sleeve")
    assert watchplan.get_analysis(store, "VIT-B.ST") is None


# ── The ADD criterion: written like a kill criterion, read as its mirror ──────

def test_saving_an_add_criterion_analyses_it_with_the_add_framing(client, monkeypatch):
    seen = {}
    import fundmgr.watchplan as wp
    real = wp.analyse_criterion
    monkeypatch.setattr(wp, "analyse_criterion",
                        lambda c, model=None, *, kind="kill", store=None:
                        seen.update(kind=kind) or real(c, model, kind=kind, store=store))
    client.post("/live/acquirer-sleeve/watchplan",
                data={"ticker": "VIT-B.ST",
                      "add_criterion": "Recurring revenue growth at least 10%"},
                follow_redirects=False)
    assert seen["kind"] == "add"

    _meta, store = paper.open_portfolio("acquirer-sleeve")
    assert watchplan.get_add_text(store)["VIT-B.ST"].startswith("Recurring revenue")


def test_the_add_criterion_appears_on_the_dashboard(client):
    client.post("/live/acquirer-sleeve/watchplan",
                data={"ticker": "VIT-B.ST",
                      "add_criterion": "Recurring revenue growth at least 10%"},
                follow_redirects=False)
    body = client.get("/live/acquirer-sleeve").text
    assert "Recurring revenue growth at least 10%" in body
    assert "ADD criterion (what must still be true to add)" in body


def test_clearing_the_add_criterion_clears_its_analysis(client):
    client.post("/live/acquirer-sleeve/watchplan",
                data={"ticker": "VIT-B.ST", "add_criterion": "Growth at least 10%"},
                follow_redirects=False)
    client.post("/live/acquirer-sleeve/watchplan",
                data={"ticker": "VIT-B.ST", "add_criterion": ""},
                follow_redirects=False)
    _meta, store = paper.open_portfolio("acquirer-sleeve")
    assert watchplan.get_add_text(store) == {}
    assert watchplan.get_add_analysis(store, "VIT-B.ST") is None


def test_an_add_condition_passes_when_the_metric_holds(store):
    """The mirror of a kill rule: it passes on the named side of the line."""
    watchplan.set_add_text(store, "VIT-B.ST", "Revenue growth at least 10%")
    watchplan.save_add_analysis(store, "VIT-B.ST", {
        "criterion": "Revenue growth at least 10%", "logic": "A",
        "conditions": [{"id": "A", "text": "growth >= 10", "checkable": "fundamentals",
                        "metric": "revenue_growth", "op": "above", "value": 10,
                        "quarters": 1, "note": ""}]})

    store.save_fundamentals("VIT-B.ST", {"revenue_growth": 0.12})
    out = watchplan.evaluate_add_criterion(store, "VIT-B.ST")
    assert out["satisfied"] is True and out["checked"] and not out["failed"]

    store.save_fundamentals("VIT-B.ST", {"revenue_growth": 0.07})
    out = watchplan.evaluate_add_criterion(store, "VIT-B.ST")
    assert out["satisfied"] is False and out["failed"]


def test_an_uncached_add_metric_is_unread_never_satisfied(store):
    """Unknown must not open a gate — the same rule the kill side follows."""
    watchplan.set_add_text(store, "VIT-B.ST", "Revenue growth at least 10%")
    watchplan.save_add_analysis(store, "VIT-B.ST", {
        "criterion": "Revenue growth at least 10%", "logic": "A",
        "conditions": [{"id": "A", "text": "growth >= 10", "checkable": "fundamentals",
                        "metric": "revenue_growth", "op": "above", "value": 10,
                        "quarters": 1, "note": ""}]})
    out = watchplan.evaluate_add_criterion(store, "VIT-B.ST")
    assert out["satisfied"] is False and out["unread"]


def test_a_manual_add_condition_still_needs_the_human(store):
    """"Cash generation intact" is not in any feed, so it cannot auto-satisfy."""
    watchplan.set_add_text(store, "VIT-B.ST", "Growth at least 10%, cash generation intact")
    watchplan.save_add_analysis(store, "VIT-B.ST", {
        "criterion": "Growth at least 10%, cash generation intact", "logic": "A and B",
        "conditions": [
            {"id": "A", "text": "growth >= 10", "checkable": "fundamentals",
             "metric": "revenue_growth", "op": "above", "value": 10, "quarters": 1, "note": ""},
            {"id": "B", "text": "cash generation intact", "checkable": "manual",
             "metric": None, "op": None, "value": None, "quarters": None, "note": ""}]})
    store.save_fundamentals("VIT-B.ST", {"revenue_growth": 0.12})

    out = watchplan.evaluate_add_criterion(store, "VIT-B.ST")
    assert out["checked"] and out["manual"]
    assert out["satisfied"] is False


def test_a_reworded_add_criterion_drops_its_stale_analysis(store):
    watchplan.set_add_text(store, "VIT-B.ST", "Growth at least 10%")
    watchplan.save_add_analysis(store, "VIT-B.ST", {
        "criterion": "Growth at least 10%", "logic": "A", "conditions": []})
    assert watchplan.get_add_analysis(store, "VIT-B.ST") is not None
    watchplan.set_add_text(store, "VIT-B.ST", "Growth at least 12%")
    assert watchplan.get_add_analysis(store, "VIT-B.ST") is None


def test_no_add_criterion_evaluates_to_none(store):
    assert watchplan.evaluate_add_criterion(store, "VIT-B.ST") is None


def test_the_cli_can_set_an_add_criterion(store, fake_llm, monkeypatch, tmp_path):
    """It was reachable only from the web form, which is no use when you are
    fixing a book over ssh."""
    from click.testing import CliRunner

    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path / "paper")
    monkeypatch.setattr(paper, "detect_currency", lambda t: "SEK")
    monkeypatch.setattr(paper, "_cache_price_history", lambda store, tickers, **kw: None)
    monkeypatch.setattr(paper, "_search_symbol", lambda name: None)
    import fundmgr.data.benchmark as benchmark
    monkeypatch.setattr(benchmark, "fetch_and_cache_benchmark", lambda store, **kw: True)
    paper.create_portfolio("Svenska Aktier", 10_000, "", kind="live",
                           holdings_override=[{"ticker": "DYVOX.ST", "name": "Dynavox",
                                               "weight_pct": 100, "shares": 10,
                                               "avg_cost_sek": 77.0}],
                           seed_holdings=True)

    from fundmgr.cli import cli
    result = CliRunner().invoke(cli, [
        "paper-plan", "svenska-aktier", "DYVOX.ST",
        "--add", "CC growth at least 15% and EBIT margin at least 15%"])
    assert result.exit_code == 0, result.output
    assert "ADD criterion:  CC growth at least 15%" in result.output

    _meta, s2 = paper.open_portfolio("svenska-aktier")
    assert watchplan.get_add_text(s2)["DYVOX.ST"].startswith("CC growth")
    assert watchplan.get_add_analysis(s2, "DYVOX.ST") is not None


# ── Horizon: months must not be swallowed by the pre-filled date ─────────────

def test_months_win_over_a_date_that_was_not_changed(client):
    """The editor pre-fills the existing date and the server prefers a date, so
    typing 15 months into an existing horizon silently did nothing."""
    client.post("/live/acquirer-sleeve/watchplan",
                data={"ticker": "VIT-B.ST", "horizon_months": "24"},
                follow_redirects=False)
    _meta, store = paper.open_portfolio("acquirer-sleeve")
    original = watchplan.get_horizons(store)["VIT-B.ST"]["review_date"]

    # Exactly what the pencil submits: the stored date, plus the new months.
    client.post("/live/acquirer-sleeve/watchplan",
                data={"ticker": "VIT-B.ST", "horizon_date": original,
                      "horizon_months": "15"},
                follow_redirects=False)

    horizon = watchplan.get_horizons(store)["VIT-B.ST"]
    assert horizon["review_date"] != original
    assert horizon["label"] == "15 months"


def test_a_changed_date_still_wins_over_months(client):
    """Editing the date itself must not be overridden by a stale months box."""
    client.post("/live/acquirer-sleeve/watchplan",
                data={"ticker": "VIT-B.ST", "horizon_months": "24"},
                follow_redirects=False)
    client.post("/live/acquirer-sleeve/watchplan",
                data={"ticker": "VIT-B.ST", "horizon_date": "2027-03-01",
                      "horizon_months": "15"},
                follow_redirects=False)

    _meta, store = paper.open_portfolio("acquirer-sleeve")
    assert watchplan.get_horizons(store)["VIT-B.ST"]["review_date"] == "2027-03-01"


def test_a_date_alone_still_works(client):
    client.post("/live/acquirer-sleeve/watchplan",
                data={"ticker": "VIT-B.ST", "horizon_date": "2027-06-30"},
                follow_redirects=False)
    _meta, store = paper.open_portfolio("acquirer-sleeve")
    assert watchplan.get_horizons(store)["VIT-B.ST"]["review_date"] == "2027-06-30"
