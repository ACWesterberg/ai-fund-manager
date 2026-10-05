"""Forward registration, future execution and complete outcome evidence."""
import copy
import json
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from fundmgr.engine import experiments as ex, forward as fw
from fundmgr.engine.prompt import build_prompt, snapshot_to_dict
from fundmgr.engine.schema import Action, DecisionRun
from fundmgr.config import AppConfig, ShadowConfig
from fundmgr.data.prices import TickerFeatures
from fundmgr.state.models import PortfolioSnapshot
from fundmgr.state.store import Store


class Clock(datetime):
    current = datetime(2026, 1, 5, 12, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls.current if tz else cls.current.replace(tzinfo=None)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    Clock.current = datetime(2026, 1, 5, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(fw, "datetime", Clock)
    monkeypatch.setattr(ex, "datetime", Clock)
    monkeypatch.setattr(fw, "_session", lambda mic, day: day.weekday() < 5)
    cfg = AppConfig(db_path=tmp_path / "test.db", mandate_path=tmp_path / "mandate.md")
    cfg.mandate_path.write_text("Allocate sensibly")
    cfg.optimizer.compiled_dir = tmp_path / "compiled"
    cfg.llm.n_samples = 1
    cfg.evaluation_horizon_days = 2
    features = {t: TickerFeatures(t, t, 100, "2026-01-02", 0, sector=sector)
                for t, sector in (("A", "Technology"), ("B", "Industrials"))}
    snap = PortfolioSnapshot([], 100000, timestamp=Clock.current)
    system, user, fields = build_prompt(cfg, snap, features, Store(cfg.db_path), "weekly-1")
    snapshot = json.loads(snapshot_to_dict(snap, system, user, fields, cfg, features=features))
    candidate = {"fund_id": "test", "task_model": cfg.llm.model_id, "provider": cfg.llm.provider,
                 "horizon_days": 2, "mandate": fields["mandate"], "incumbent_guidance_hash": None,
                 "status": "pending_evaluation", "instructions": "Alternative guidance",
                 "created_at": "2026-01-01T12:00:00+00:00"}
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(candidate))
    cfg.shadow = ShadowConfig(str(candidate_path), "SEK", "XSTO")
    tickers = [SimpleNamespace(yahoo_ticker=t, exchange="OMXS") for t in features]
    calls = []

    def model(*args):
        # Registration is already durable BEFORE either model call.
        assert list(fw.experiment_root(cfg).glob("*/registration.json"))
        ticker = "A" if not calls else "B"
        calls.append(args)
        run = DecisionRun(run_id="test", market_summary="Test", cash_target_pct=90,
            actions=[Action(ticker=ticker, side="buy", target_weight_pct=10,
                            sek_estimate=10000, confidence=.9, thesis="Test")])
        return run, run.model_dump_json(), {}, {"requested": 1, "succeeded": 1, "failed": 0}
    monkeypatch.setattr(ex, "call_llm_consensus", model)
    return cfg, snapshot, tickers, calls


def record(setup):
    cfg, snapshot, tickers, _ = setup
    directory = fw.record_forward(cfg, snapshot, "weekly-1", tickers)
    return directory, json.loads((directory / "registration.json").read_text()), json.loads((directory / "comparison.json").read_text())


def bundle():
    values = {"A": [100, 105, 110, 115, 121], "B": [100, 100, 100, 110, 120],
              "^OMXSPI": [100, 100, 100, 101, 102]}
    days = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
    return {symbol: {"symbol": symbol, "currency": "SEK", "auto_adjust": True,
                     "retrieved_at": "2026-01-09T12:00:00+00:00",
                     "rows": [{"date": d, "close": p, "dividends": 0, "splits": 0} for d, p in zip(days, prices)]}
            for symbol, prices in values.items()}


def mature():
    Clock.current = datetime(2026, 1, 9, 12, tzinfo=timezone.utc)


def test_reserves_before_calls_and_never_retries_same_run(setup):
    cfg, snapshot, tickers, calls = setup
    directory, reg, pair = record(setup)
    assert len(calls) == 2 and pair["mode"] == "registered_forward"
    assert fw.execution_date(reg, pair) == date(2026, 1, 6)
    assert fw.record_forward(cfg, snapshot, "weekly-1", tickers) == directory
    assert len(calls) == 2


@pytest.mark.parametrize("change", ["disabled", "old_case", "new_candidate", "unknown_venue", "benchmark"])
def test_preflight_never_pays_for_ineligible_cases(setup, change):
    cfg, snapshot, tickers, calls = setup
    if change == "disabled":
        cfg.shadow.candidate = None
        assert fw.record_forward(cfg, snapshot, "weekly-1", tickers) is None
    else:
        if change == "old_case":
            snapshot["evaluation_case"]["decision_time"] = "2026-01-02T12:00:00+00:00"
        elif change == "new_candidate":
            path = cfg.shadow.candidate
            candidate = json.loads(open(path).read())
            candidate["created_at"] = "2026-01-06T12:00:00+00:00"
            open(path, "w").write(json.dumps(candidate))
        elif change == "unknown_venue":
            tickers[0].exchange = "UNKNOWN"
        else:
            cfg.shadow.benchmark_currency = ""
        with pytest.raises(ValueError):
            fw.record_forward(cfg, snapshot, "weekly-1", tickers)
    assert not calls
    assert not fw.experiment_root(cfg).exists()


def test_future_entry_excludes_return_before_decisions_and_entry(setup):
    _, reg, pair = record(setup)
    original = copy.deepcopy(pair)
    mature()
    derived, outcomes = fw.derive_forward(reg, pair, bundle())
    score = ex.score_comparison(derived, outcomes)
    # A rises 21% from the frozen quote but only 10% AFTER future entry.
    assert score["scores"]["incumbent"]["net_return_pct"] == pytest.approx(.99)
    assert score["candidate_advantage_pp"] == pytest.approx(1)
    assert derived["arms"]["incumbent"]["allocation"]["shares"]["A"] == pytest.approx(10000 / 110)
    assert pair == original and score["promotion_eligible"] is False
    assert outcomes.observations[0].date == date(2026, 1, 6)


def test_adjusted_history_revisions_and_split_do_not_create_fake_loss(setup):
    _, reg, pair = record(setup)
    mature()
    data = bundle()
    for row in data["A"]["rows"]:
        row["close"] /= 2
    data["A"]["rows"][2]["splits"] = 2
    derived, outcomes = fw.derive_forward(reg, pair, data)
    assert ex.score_comparison(derived, outcomes)["scores"]["incumbent"]["net_return_pct"] == pytest.approx(.99)


@pytest.mark.parametrize("change,match", [("missing", "Missing"), ("currency", "currency"),
    ("adjustment", "adjustment"), ("duplicate", "duplicate"), ("nan", "Invalid")])
def test_bad_provider_evidence_fails_closed_including_unchosen_names(setup, change, match):
    _, reg, pair = record(setup)
    mature()
    data = bundle()
    if change == "missing":
        data["B"]["rows"].pop(3)
    elif change == "currency":
        data["A"]["currency"] = "GBp"
    elif change == "adjustment":
        data["A"]["auto_adjust"] = False
    elif change == "duplicate":
        data["A"]["rows"].append(data["A"]["rows"][0])
    else:
        data["A"]["rows"][0]["close"] = float("nan")
    with pytest.raises(ValueError, match=match):
        fw.derive_forward(reg, pair, data)


def test_fx_revalues_both_security_and_foreign_benchmark(setup):
    cfg, snapshot, _, _ = setup
    cfg.fx_to_sek = True
    cfg.shadow.benchmark_currency = "USD"
    snapshot["evaluation_case"].update(basis="SEK", fx_rates={"USD": 10})
    snapshot["evaluation_case"]["features"]["A"]["currency"] = "USD"
    _, reg, pair = record(setup)
    mature()
    data = bundle()
    data["A"]["currency"] = data["^OMXSPI"]["currency"] = "USD"
    data["USDSEK=X"] = copy.deepcopy(data["A"])
    data["USDSEK=X"].update(symbol="USDSEK=X", currency="SEK")
    for row, rate in zip(data["USDSEK=X"]["rows"], [10, 10, 11, 11.5, 12]):
        row["close"] = rate
    derived, outcomes = fw.derive_forward(reg, pair, data)
    score = ex.score_comparison(derived, outcomes)
    assert score["scores"]["incumbent"]["net_return_pct"] == pytest.approx(1.99)
    assert score["benchmark_return_pct"] == pytest.approx((102 * 12 / (100 * 11) - 1) * 100)


def test_collection_waits_then_scores_once_and_preserves_evidence(setup):
    cfg, _, _, _ = setup
    directory, _, _ = record(setup)
    calls = []
    def fetch(symbol, start, end):
        calls.append(symbol)
        assert start <= date(2026, 1, 2) and end == date(2026, 1, 8)
        return bundle()[symbol]
    assert fw.collect_forward(cfg, fetch)[0]["status"] == "waiting"
    assert not calls
    mature()
    assert fw.collect_forward(cfg, fetch)[0]["status"] == "scored"
    assert set(calls) == {"A", "B", "^OMXSPI"}
    assert list(directory.glob("collections/*/history.json"))
    assert list(directory.glob("collections/*/outcomes.json"))
    saved = (directory / "result.json").read_bytes()
    assert fw.collect_forward(cfg, fetch)[0]["status"] == "complete"
    assert len(calls) == 3 and (directory / "result.json").read_bytes() == saved


def test_missing_history_pending_retry_keeps_failed_attempt(setup):
    cfg, _, _, _ = setup
    directory, _, _ = record(setup)
    mature()
    data = bundle()
    data["B"]["rows"].pop(3)
    assert fw.collect_forward(cfg, lambda s, *a: data[s])[0]["status"] == "pending"
    assert not (directory / "result.json").exists()
    assert list(directory.glob("collections/*/error.json"))
    assert fw.collect_forward(cfg, lambda s, *a: bundle()[s])[0]["status"] == "scored"
    assert len(list(directory.glob("collections/*/history.json"))) == 2


def test_collect_cli_never_calls_model(setup, monkeypatch):
    import fundmgr.cli as commands
    cfg, _, _, calls = setup
    record(setup)
    mature()
    monkeypatch.setattr(commands, "load_config", lambda: cfg)
    monkeypatch.setattr(fw, "fetch_history", lambda s, *a: bundle()[s])
    monkeypatch.setattr(ex, "call_llm_consensus", lambda *a: pytest.fail("Collector must not call models"))
    result = CliRunner().invoke(commands.cli, ["collect-guidance"])
    assert result.exit_code == 0, result.output
    assert "scored" in result.output and len(calls) == 2


def test_unknown_calendar_does_not_guess_a_closed_day(monkeypatch):
    monkeypatch.setattr(fw, "is_trading_day", lambda *a: None)
    with pytest.raises(ValueError, match="calendar"):
        fw._session("UNKNOWN", date(2026, 1, 6))


def test_weekends_carry_but_missing_open_day_never_does(setup):
    friday = date(2026, 1, 2)
    assert fw._mark({friday: 100}, date(2026, 1, 4), "XSTO") == 100
    with pytest.raises(ValueError, match="Missing"):
        fw._mark({friday: 100}, date(2026, 1, 5), "XSTO")


def test_all_shown_alternatives_required_even_when_neither_arm_buys_them(setup):
    _, snapshot, tickers, _ = setup
    snapshot["evaluation_case"]["features"]["C"] = dict(snapshot["evaluation_case"]["features"]["A"], ticker="C")
    snapshot["evaluation_case"]["universe"].append("C")
    tickers.append(SimpleNamespace(yahoo_ticker="C", exchange="OMXS"))
    cfg = setup[0]
    directory, _, _ = record(setup)
    mature()
    result = fw.collect_forward(cfg, lambda s, *a: bundle()[s])[0]
    assert result["status"] == "pending" and "C" in result["error"]
    assert not (directory / "result.json").exists()


def test_failed_model_samples_are_recorded_without_retries(setup, monkeypatch):
    cfg, snapshot, tickers, _ = setup
    def fail(*a):
        raise RuntimeError("Model unavailable")
    monkeypatch.setattr(ex, "call_llm_consensus", fail)
    directory, _, pair = record(setup)
    assert all(a["status"] == "invalid" for a in pair["arms"].values())
    monkeypatch.setattr(ex, "call_llm_consensus", lambda *a: pytest.fail("Never retry failed model experiments"))
    assert fw.record_forward(cfg, snapshot, "weekly-1", tickers) == directory
    mature()
    assert fw.collect_forward(cfg, lambda *a: pytest.fail("Invalid arms need no outcomes"))[0]["status"] == "pending"


def test_registration_tampering_is_detected(setup):
    _, reg, pair = record(setup)
    mature()
    reg["benchmark_currency"] = "USD"
    with pytest.raises(ValueError, match="Modified artifact"):
        fw.derive_forward(reg, pair, bundle())


def test_provider_adapter_requests_adjusted_actions_and_preserves_rows(monkeypatch):
    import pandas as pd
    import yfinance as yf
    calls = []
    class Ticker:
        def history(self, **kwargs):
            calls.append(kwargs)
            return pd.DataFrame({"Close": [50., 51.], "Dividends": [1., 0.], "Stock Splits": [0., 2.]},
                                index=pd.to_datetime(["2026-01-02", "2026-01-05"]))
        def get_history_metadata(self):
            return {"currency": "SEK", "exchangeTimezoneName": "Europe/Stockholm"}
    monkeypatch.setattr(yf, "Ticker", lambda symbol: Ticker())
    history = fw.fetch_history("A", date(2026, 1, 2), date(2026, 1, 5))
    assert calls[0]["auto_adjust"] is True and calls[0]["actions"] is True
    assert calls[0]["end"] == "2026-01-06"
    assert history["rows"][0]["dividends"] == 1
    assert history["rows"][1]["splits"] == 2
    assert history["currency"] == "SEK" and history["provider_version"]


def test_shadow_config_loads_explicit_candidate_path(tmp_path, monkeypatch):
    from fundmgr.config import load_config, ROOT
    monkeypatch.delenv("FUND_CONFIG", raising=False)
    path = tmp_path / "config.yaml"
    path.write_text("shadow:\n  candidate: config/compiled/candidate.json\n  benchmark_currency: SEK\n  benchmark_calendar: XSTO\n")
    cfg = load_config(path)
    assert cfg.shadow.candidate == str(ROOT / "config/compiled/candidate.json")
    assert cfg.shadow.benchmark_calendar == "XSTO"
    assert AppConfig().shadow.candidate is None


def test_held_book_is_revalued_at_entry_before_scoring(setup, monkeypatch):
    _, snapshot, _, _ = setup
    snapshot["evaluation_case"]["cash"] = 90000
    snapshot["evaluation_case"]["positions"] = [{
        "ticker": "A", "shares": 100, "avg_cost_sek": 100,
        "current_price_sek": 100, "updated_at": "2026-01-05T12:00:00+00:00"}]
    def hold(*args):
        run = DecisionRun(run_id="test", market_summary="Hold", cash_target_pct=90, actions=[])
        return run, run.model_dump_json(), {}, {"requested": 1, "succeeded": 1, "failed": 0}
    monkeypatch.setattr(ex, "call_llm_consensus", hold)
    _, reg, pair = record(setup)
    mature()
    derived, outcomes = fw.derive_forward(reg, pair, bundle())
    plan = derived["arms"]["incumbent"]["allocation"]
    assert plan["initial_nav"] == 101000  # 90k cash + 100 shares at the entry mark 110.
    assert plan["shares"] == {"A": 100} and plan["fees"] == 0
    score = ex.score_comparison(derived, outcomes)
    # Only 110 -> 121 accrues to this period; the earlier 100 -> 110 gain is excluded.
    assert score["scores"]["incumbent"]["net_return_pct"] == pytest.approx(1100 / 101000 * 100)


def test_horizon_day_itself_cannot_be_scored(setup):
    cfg, _, _, _ = setup
    record(setup)
    Clock.current = datetime(2026, 1, 8, 23, tzinfo=timezone.utc)
    result = fw.collect_forward(cfg, lambda *a: pytest.fail("Do not fetch incomplete closing days"))
    assert result[0]["status"] == "waiting"
