"""The report-reaction experiment: detect, judge within budget, score honestly.

Nothing here may trade or spend beyond its cap, and a verdict must rest on
sentences that are actually in the coverage the model was shown.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from click.testing import CliRunner

from fundmgr import cli as commands
from fundmgr.config import AppConfig, ReportWatchConfig, UniverseTicker
from fundmgr.engine import report_events as rx
from fundmgr.engine.client import LLMError
from fundmgr.engine.schema import ReportQuote, ReportReaction
from fundmgr.state.store import Store


def _days(n: int, end: datetime | None = None) -> list[str]:
    """n consecutive weekdays ending today (or `end`)."""
    day, out = end or datetime.utcnow(), []
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day.strftime("%Y-%m-%d"))
        day -= timedelta(days=1)
    return out[::-1]


def _series(closes: list[float], days: list[str]) -> list[dict]:
    return [{"date": d, "open": c, "high": c, "low": c, "close": c, "volume": 1000.0}
            for d, c in zip(days, closes)]


def _ticker(sym: str) -> UniverseTicker:
    return UniverseTicker(name=sym.split(".")[0].title(), yahoo_ticker=sym, isin="", country="SE",
                          exchange="XSTO", sector="Industrials", enabled=True)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "fund.db")


@pytest.fixture
def cfg(tmp_path):
    return AppConfig(db_path=tmp_path / "fund.db", report_watch=ReportWatchConfig(enabled=True))


def _flat_bench(store, days):
    store.save_benchmark([{"date": d, "close": 100.0} for d in days])
    return rx._Bench(store.get_benchmark())


# ── The arithmetic ────────────────────────────────────────────────────────────

def test_the_reaction_is_measured_across_the_report_and_net_of_the_index(store):
    days = _days(5)
    series = _series([100, 100, 100, 90, 91], days)
    store.save_benchmark([{"date": d, "close": c} for d, c in zip(days, [100, 100, 100, 102, 102])])
    bench = rx._Bench(store.get_benchmark())
    i0, i1 = rx.reaction_window(series, days[3])
    assert (i0, i1) == (2, 4)
    raw, abn = rx.abnormal(series, bench, i0, i1)
    assert raw == pytest.approx(-9.0)
    assert abn == pytest.approx(-11.0)


def test_forward_returns_appear_only_once_the_horizon_has_passed(store):
    days = _days(30)
    bench = _flat_bench(store, days)
    series = _series([100.0] * 10 + [110.0] * 20, days)
    fwd = rx.forward(series, bench, 0)
    assert fwd[5] == pytest.approx(0.0)
    assert fwd[20] == pytest.approx(10.0)
    assert fwd[28] == pytest.approx(10.0)
    assert rx.forward(series, bench, 5)[28] is None


def test_summary_says_nothing_about_an_empty_group():
    assert rx.summarize([]) == {"n": 0}
    s = rx.summarize([1.0, -1.0, 3.0])
    assert s["n"] == 3 and s["pos_share"] == pytest.approx(2 / 3)
    assert s["ci"][0] <= s["mean"] <= s["ci"][1]


# ── Detection ─────────────────────────────────────────────────────────────────

def _seed(store, sym, closes, days):
    store.save_prices(sym, _series(closes, days))


def test_only_moves_on_a_report_become_events(store, cfg):
    days = _days(25)
    bench = _flat_bench(store, days)
    report_day = days[-2]
    base = [100.0] * 23
    _seed(store, "CAL.ST", base + [92, 91], days)      # Yahoo dates the report
    _seed(store, "NEWS.ST", base + [92, 90], days)     # only a headline does
    _seed(store, "NOREP.ST", base + [92, 90], days)    # a drop with no report
    _seed(store, "CALM.ST", base + [99, 99], days)     # a report that moved nothing

    history = {"CAL.ST": [{"date": report_day, "eps_estimate": 1.2, "eps_reported": 1.0,
                           "eps_surprise_pct": -16.7}],
               "CALM.ST": [{"date": report_day}]}
    news = {"NEWS.ST": [{"headline": "Delårsrapport januari–september", "published": report_day}],
            "NOREP.ST": [{"headline": "CEO interviewed on strategy", "published": report_day}]}
    found = rx.detect_events(store, cfg, [_ticker(s) for s in ("CAL.ST", "NEWS.ST", "NOREP.ST", "CALM.ST")],
                             bench, history=lambda t: history.get(t, []),
                             news_fn=lambda t: news.get(t, []))
    by = {e["ticker"]: e for e in found}
    assert set(by) == {"CAL.ST", "NEWS.ST"}
    assert by["CAL.ST"]["source"] == "calendar" and by["CAL.ST"]["eps_surprise_pct"] == -16.7
    assert by["CAL.ST"]["abnormal_pct"] == pytest.approx(-9.0)
    assert by["NEWS.ST"]["source"] == "news"


# ── Judging ───────────────────────────────────────────────────────────────────

SOURCE = [{"headline": "Q3 report", "publisher": "MFN", "published": "2026-10-20",
           "body": "Order intake fell 4 percent, mainly due to delayed deliveries in Germany. "
                   "The company reiterates its outlook for the full year."}]


def _reaction(quote: str, verdict: str = "overreaction") -> ReportReaction:
    return ReportReaction(ticker="CAL.ST", verdict=verdict, confidence=0.7,
                          what_changed="Orders slipped on timing.", temporary_factors="Delays.",
                          structural_factors="", expectations_vs_results="Missed high hopes.",
                          quotes=[ReportQuote(quote=quote, shows="timing, not demand")])


EVENT = {"ticker": "CAL.ST", "name": "Cal", "report_date": "2026-10-20", "event_date": "2026-10-21",
         "reaction_pct": -9.0, "abnormal_pct": -9.5, "volume_ratio": 3.1}


def test_a_verdict_resting_on_a_real_sentence_stands(cfg):
    real = "The company reiterates its outlook for the full year."
    out = rx.judge_event(EVENT, cfg, coverage_fn=lambda *a: SOURCE,
                         llm=lambda *a: (_reaction(real), ""))
    assert out["verdict"] == "overreaction" and len(out["quotes"]) == 1


def test_a_verdict_resting_on_an_invented_quote_is_unclear(cfg):
    out = rx.judge_event(EVENT, cfg, coverage_fn=lambda *a: SOURCE,
                         llm=lambda *a: (_reaction("Management expects record margins next year."), ""))
    assert out["verdict"] == "unclear" and out["quotes"] == []
    assert "No quote survived" in out["what_changed"]


def test_no_coverage_means_no_call(cfg):
    out = rx.judge_event(EVENT, cfg, coverage_fn=lambda *a: [],
                         llm=lambda *a: pytest.fail("paid for a verdict with nothing to read"))
    assert out["verdict"] == "unclear" and out["judge_model"] == ""


def test_the_judge_runs_at_the_experiments_effort_not_the_funds(cfg):
    cfg.llm.reasoning_effort, cfg.llm.n_samples = "high", 3
    seen = {}

    def llm(system, user, jcfg, schema):
        seen.update(effort=jcfg.llm.reasoning_effort, n=jcfg.llm.n_samples)
        return _reaction("The company reiterates its outlook for the full year."), ""
    rx.judge_event(EVENT, cfg, coverage_fn=lambda *a: SOURCE, llm=llm)
    assert seen == {"effort": "medium", "n": 1}
    assert cfg.llm.reasoning_effort == "high"


# ── The daily job ─────────────────────────────────────────────────────────────

@pytest.fixture
def watch(store, cfg, monkeypatch):
    days = _days(25)
    _flat_bench(store, days)
    monkeypatch.setattr("fundmgr.data.prices.fetch_and_cache_prices", lambda *a, **k: {})
    monkeypatch.setattr(rx, "_ensure_benchmark", lambda s, c, d: rx._Bench(s.get_benchmark()))
    for sym, last in (("BIG.ST", 80.0), ("MID.ST", 88.0), ("SMALL.ST", 93.0)):
        _seed(store, sym, [100.0] * 23 + [100.0, last], days)
    monkeypatch.setattr(rx, "earnings_history", lambda t: [{"date": days[-2]}])
    return [_ticker(s) for s in ("BIG.ST", "MID.ST", "SMALL.ST")]


def _paid(ev):
    return {"verdict": "justified", "judge_model": "m", "n_sources": 1}


def test_the_cap_judges_the_largest_drops_first(store, cfg, watch):
    cfg.report_watch.max_judged_per_day = 2
    judged = rx.run_watch(store, cfg, watch, judge=_paid, echo=lambda *_: None)
    assert [e["ticker"] for e in judged] == ["BIG.ST", "MID.ST"]
    left = {e["ticker"]: e["verdict"] for e in store.get_report_events()}
    assert left == {"BIG.ST": "justified", "MID.ST": "justified", "SMALL.ST": None}


def test_a_second_run_the_same_day_spends_nothing_more(store, cfg, watch):
    cfg.report_watch.max_judged_per_day = 2
    rx.run_watch(store, cfg, watch, judge=_paid, echo=lambda *_: None)
    again = rx.run_watch(store, cfg, watch, judge=lambda ev: pytest.fail("over budget"),
                         echo=lambda *_: None)
    assert again == []


def test_a_failed_call_is_recorded_and_counted_not_retried(store, cfg, watch):
    cfg.report_watch.max_judged_per_day = 1

    def boom(ev):
        raise LLMError("timeout")
    judged = rx.run_watch(store, cfg, watch, judge=boom, echo=lambda *_: None)
    assert judged[0]["verdict"] == "unclear" and judged[0]["judge_model"] == "error"
    assert store.count_report_verdicts_on(datetime.utcnow().strftime("%Y-%m-%d")) == 1


def test_a_dry_run_writes_and_spends_nothing(store, cfg, watch):
    out = rx.run_watch(store, cfg, watch, dry_run=True,
                       judge=lambda ev: pytest.fail("paid on a dry run"), echo=lambda *_: None)
    assert out == [] and store.get_report_events() == []


def test_events_are_scored_as_their_horizons_arrive(store):
    days = _days(40)
    bench = _flat_bench(store, days)
    _seed(store, "X.ST", [100.0] * 10 + [110.0] * 30, days)
    store.add_report_event({"ticker": "X.ST", "report_date": days[1], "event_date": days[2],
                            "reaction_pct": -6.0, "abnormal_pct": -6.0, "source": "calendar"})
    assert rx.mature(store, bench) == 1
    ev = store.get_report_events()[0]
    assert ev["fwd_28"] == pytest.approx(10.0) and ev["matured_at"]
    assert rx.mature(store, bench) == 0


# ── What reaches people ───────────────────────────────────────────────────────

def test_the_digest_lists_only_overreactions_and_escapes_model_text():
    judged = [{**EVENT, "verdict": "overreaction", "confidence": 0.7,
               "what_changed": "Margins <fell> on mix", "quotes": [{"quote": "A & B", "shows": ""}]},
              {**EVENT, "ticker": "J.ST", "verdict": "justified"}]
    msg = rx.digest_html("Nordic", judged)
    assert "no action asked" in msg and "CAL.ST" in msg and "J.ST" not in msg
    assert "&lt;fell&gt;" in msg and "A &amp; B" in msg
    assert rx.digest_html("Nordic", judged[1:]) is None


def test_results_refuse_to_conclude_on_a_handful():
    events = [{"abnormal_pct": -8.0, "verdict": "overreaction", "fwd_5": 1.0, "fwd_20": 2.0,
               "fwd_28": 3.0}]
    assert "Too few matured verdicts" in rx.format_results(events, 5.0)


def test_a_paused_or_unenrolled_fund_runs_no_watch(store, monkeypatch):
    monkeypatch.setattr(rx, "run_watch", lambda *a, **k: pytest.fail("watch ran"))
    for c in (AppConfig(paused=True, report_watch=ReportWatchConfig(enabled=True)),
              AppConfig(report_watch=ReportWatchConfig(enabled=False))):
        monkeypatch.setattr(commands, "_get_store", lambda cfg_=None, c=c: (c, store))
        result = CliRunner().invoke(commands.cli, ["report-watch"])
        assert result.exit_code == 0, result.output


def test_the_study_buckets_past_reports_and_reports_its_coverage(store, cfg, monkeypatch):
    days = _days(60)
    _flat_bench(store, days)
    monkeypatch.setattr("fundmgr.data.prices.fetch_and_cache_prices", lambda *a, **k: {})
    monkeypatch.setattr(rx, "_ensure_benchmark", lambda s, c, d: rx._Bench(s.get_benchmark()))
    _seed(store, "DROP.ST", [100.0] * 10 + [85.0] * 50, days)   # −15% on the report, then flat
    _seed(store, "NODATE.ST", [100.0] * 60, days)
    result = rx.run_study(store, cfg, [_ticker("DROP.ST"), _ticker("NODATE.ST")],
                          history=lambda t: [{"date": days[9]}] if t == "DROP.ST" else [],
                          echo=lambda *_: None)
    assert result["covered"] == 1 and len(result["events"]) == 1
    assert result["events"][0]["abnormal_pct"] == pytest.approx(-15.0)
    text = rx.format_study(result)
    assert "1/2 names (50%)" in text and "≤ −10%" in text


def test_a_missing_library_is_said_out_loud_not_counted_as_no_coverage(monkeypatch):
    """The first study on the Pi read 0/100 covered: lxml was missing, not the dates."""
    class FakeTicker:
        def __init__(self, t):
            pass

        def get_earnings_dates(self, limit):
            raise ImportError("`Import lxml` failed.")
    monkeypatch.setattr("yfinance.Ticker", FakeTicker)
    with pytest.raises(rx.CalendarUnavailable, match="lxml"):
        rx.earnings_history("VOLV-B.ST")


def test_the_watch_falls_back_to_headlines_once_and_says_why(store, cfg):
    days = _days(25)
    bench = _flat_bench(store, days)
    for sym in ("A.ST", "B.ST"):
        _seed(store, sym, [100.0] * 23 + [92, 90], days)
    calls, warned = [], []

    def broken(t):
        calls.append(t)
        raise rx.CalendarUnavailable("lxml missing")
    news = {"B.ST": [{"headline": "Interim report Q3", "published": days[-2]}]}
    found = rx.detect_events(store, cfg, [_ticker("A.ST"), _ticker("B.ST")], bench,
                             history=broken, news_fn=lambda t: news.get(t, []),
                             warn=warned.append)
    assert calls == ["A.ST"]                      # not retried for every name
    assert len(warned) == 1 and "lxml missing" in warned[0]
    assert [e["ticker"] for e in found] == ["B.ST"]


def test_the_study_stops_with_the_reason_when_the_calendar_cannot_be_read(store, cfg, monkeypatch):
    monkeypatch.setattr(commands, "_get_store", lambda cfg_=None: (cfg, store))

    def broken(*a, **k):
        raise rx.CalendarUnavailable("Yahoo's report history can't be read: lxml")
    monkeypatch.setattr(rx, "run_study", broken)
    result = CliRunner().invoke(commands.cli, ["report-study", "--limit", "5"])
    assert result.exit_code != 0 and "lxml" in result.output


def test_the_study_fetches_in_chunks_and_says_how_far_it_has_got(store, cfg, monkeypatch):
    """One silent 1,605-name download was indistinguishable from a hang."""
    calls, lines = [], []

    def fetch(chunk, store_, lookback_days, force_refresh=False):
        calls.append(([t.yahoo_ticker for t in chunk], force_refresh))
        if any(t.yahoo_ticker == "N120.ST" for t in chunk) and not force_refresh:
            raise RuntimeError("Yahoo throttled")
        return {}
    monkeypatch.setattr("fundmgr.data.prices.fetch_and_cache_prices", fetch)
    monkeypatch.setattr(rx, "_ensure_benchmark", lambda s, c, d: rx._Bench([]))
    tickers = [_ticker(f"N{i}.ST") for i in range(250)]
    rx.run_study(store, cfg, tickers, history=lambda t: [], echo=lines.append)

    plain = [names for names, forced in calls if not forced]
    assert [len(c) for c in plain] == [100, 100, 50]
    assert sorted(sum(plain, [])) == sorted(t.yahoo_ticker for t in tickers)
    assert any("failed: Yahoo throttled" in ln for ln in lines)
    assert [ln.split(" · ")[0].strip() for ln in lines if ln.strip().startswith("prices ")] == [
        "prices 100/250", "prices 200/250", "prices 250/250"]
    assert sum(ln.strip().startswith("reports ") for ln in lines) == 5     # every 50 names


def test_a_name_with_too_short_a_history_is_fetched_again_forced(store, cfg, monkeypatch):
    """The cache skips anything updated today, however little of it there is."""
    days = _days(60)
    _seed(store, "SHORT.ST", [100.0] * 60, days)
    forced = []

    def fetch(chunk, store_, lookback_days, force_refresh=False):
        if force_refresh:
            forced.extend(t.yahoo_ticker for t in chunk)
        return {}
    monkeypatch.setattr("fundmgr.data.prices.fetch_and_cache_prices", fetch)
    monkeypatch.setattr(rx, "_ensure_benchmark", lambda s, c, d: rx._Bench([]))
    rx.run_study(store, cfg, [_ticker("SHORT.ST")], history=lambda t: [], echo=lambda *_: None)
    assert forced == ["SHORT.ST"]
