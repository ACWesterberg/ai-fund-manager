"""Tests for the What-If Lab: profile discovery, synthetic clean-slate snapshot,
consensus plumbing and the web routes. The LLM is always stubbed."""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from fundmgr.engine import whatif
from fundmgr.engine.schema import Action, DecisionRun
from fundmgr.state.store import Store


# ── Fixtures ──────────────────────────────────────────────────────────────────

UNIVERSE_CSV = """name,yahoo_ticker,isin,country,exchange,sector,enabled
Alfa AB,ALFA.ST,SE0001,SE,OMXS,Industrials,true
Beta AB,BETA.ST,SE0002,SE,OMXS,Technology,true
Gamma AB,GAMMA.ST,SE0003,SE,OMXS,Healthcare,true
Delta AB,DELTA.ST,SE0004,SE,OMXS,Financials,false
"""

MANDATE = "You are a test fund manager. Buy good things."


def _price_rows(n: int = 260, start: float = 100.0) -> list[dict]:
    """Daily closes ending today so features are fresh (not stale)."""
    rows = []
    today = datetime.utcnow().date()
    for i in range(n):
        d = today - timedelta(days=n - 1 - i)
        px = start + i * 0.1
        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "open": px, "high": px * 1.01, "low": px * 0.99,
            "close": px, "volume": 10_000 + i,
        })
    return rows


@pytest.fixture
def profile(tmp_path, monkeypatch):
    """A complete fund profile on disk: config + mandate + universe + warm store."""
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    config_dir.mkdir()
    data_dir.mkdir()

    (config_dir / "universe_test.csv").write_text(UNIVERSE_CSV)
    (config_dir / "mandate_test.md").write_text(MANDATE)

    db_path = data_dir / "fund_test.db"
    cfg_yaml = f"""
name: "🧪 Test Fund"
capital_sek: 100000
benchmark: "^OMXSPI"
db_path: "{db_path}"
mandate_path: "{config_dir / 'mandate_test.md'}"
universe_path: "{config_dir / 'universe_test.csv'}"
llm:
  provider: openai
  model_id: "gpt-5.6-sol"
  n_samples: 3
risk:
  max_position_pct: 20
  max_positions: 5
  min_cash_pct: 5
  max_cash_pct: 10
  min_trade_sek: 1000
  max_turnover_pct: 25
  cold_start_cash_threshold: 80
  cold_start_turnover_pct: 100
screener:
  top_n: 10
"""
    (config_dir / "config_test.yaml").write_text(cfg_yaml)

    # Warm the store's price cache — this is what the what-if run reads.
    store = Store(db_path)
    store.initialise(100000)
    for ticker in ("ALFA.ST", "BETA.ST", "GAMMA.ST"):
        store.save_prices(ticker, _price_rows())

    monkeypatch.setattr(whatif, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(whatif, "WHATIF_DIR", data_dir / "whatif")
    return {"config_dir": config_dir, "data_dir": data_dir, "db_path": db_path}


def _stub_consensus(actions: list[Action], vote_counts: dict | None = None, n: int = 3):
    """Replace call_llm_consensus with a deterministic decision."""
    def _fake(system, user, cfg):
        decision = DecisionRun(
            run_id="stub",
            market_summary="Stubbed market read.",
            actions=actions,
            cash_target_pct=8.0,
            notes="stub notes",
        )
        sampling = {"requested": n, "succeeded": n, "failed": 0, "errors": []}
        return decision, decision.model_dump_json(), vote_counts, sampling
    return _fake


# ── Profile discovery ─────────────────────────────────────────────────────────

def test_list_profiles_bundles_mandate_and_universe(profile):
    profiles = whatif.list_profiles()
    assert len(profiles) == 1
    p = profiles[0]
    assert p["config"] == "config_test.yaml"
    assert p["name"] == "🧪 Test Fund"
    assert p["universe"] == "universe_test.csv"
    assert p["mandate"] == "mandate_test.md"
    assert p["default_n_samples"] == 3


def test_real_repo_profiles_pair_each_mandate_with_its_own_universe():
    """The Buffett profile must not be generatable against the Nordic universe."""
    profiles = whatif.list_profiles()
    by_config = {p["config"]: p for p in profiles}
    buffett = by_config["config_buffett_gpt.yaml"]
    assert buffett["mandate"] == "mandate_buffett.md"
    assert buffett["universe"] == "universe_buffett.csv"
    main = by_config["config.yaml"]
    assert main["universe"] == "universe.csv"


@pytest.mark.parametrize("bad", ["../secrets.yaml", "/etc/passwd", "config.txt", "sub/config.yaml"])
def test_profile_path_traversal_rejected(profile, bad):
    with pytest.raises(ValueError):
        whatif.load_profile_config(bad)


# ── Generation ────────────────────────────────────────────────────────────────

def test_generate_uses_clean_slate_snapshot(profile, monkeypatch):
    """The prompt must describe an all-cash book, never the fund's real positions."""
    captured = {}

    def _capture(system, user, cfg):
        captured["user"] = user
        captured["cfg"] = cfg
        decision = DecisionRun(
            run_id="stub", market_summary="ok",
            actions=[Action(ticker="ALFA.ST", side="buy", target_weight_pct=15,
                            sek_estimate=15000, confidence=0.8, thesis="t")],
            cash_target_pct=8.0,
        )
        return decision, "{}", {"ALFA.ST": 3}, {"requested": 3, "succeeded": 3, "failed": 0, "errors": []}

    monkeypatch.setattr(whatif, "call_llm_consensus", _capture)
    whatif.generate_whatif("config_test.yaml", refresh_prices=False, n_runs=3, include_macro=False)

    assert "No open positions — fully in cash." in captured["user"]
    assert "NAV: 100,000 SEK  |  Cash: 100,000 SEK (100.0%)" in captured["user"]
    # 100% cash trips the cold-start turnover lift
    assert captured["cfg"].risk.max_turnover_pct == 100


def test_generate_applies_model_override_and_run_count(profile, monkeypatch):
    seen = {}

    def _capture(system, user, cfg):
        seen["provider"] = cfg.llm.provider
        seen["model_id"] = cfg.llm.model_id
        seen["n_samples"] = cfg.llm.n_samples
        decision = DecisionRun(
            run_id="stub", market_summary="ok",
            actions=[Action(ticker="ALFA.ST", side="buy", target_weight_pct=10,
                            sek_estimate=10000, confidence=0.7, thesis="t")],
            cash_target_pct=8.0,
        )
        return decision, "{}", None, {"requested": 5, "succeeded": 5, "failed": 0, "errors": []}

    monkeypatch.setattr(whatif, "call_llm_consensus", _capture)
    result = whatif.generate_whatif(
        "config_test.yaml", refresh_prices=False, provider="anthropic", model_id="claude-opus-4-8",
        n_runs=5, include_macro=False,
    )

    assert seen == {"provider": "anthropic", "model_id": "claude-opus-4-8", "n_samples": 5}
    assert result["model"]["provider"] == "anthropic"
    assert result["model"]["n_runs"] == 5


def test_n_runs_is_clamped(profile, monkeypatch):
    seen = {}

    def _capture(system, user, cfg):
        seen["n"] = cfg.llm.n_samples
        decision = DecisionRun(
            run_id="stub", market_summary="ok",
            actions=[Action(ticker="ALFA.ST", side="buy", target_weight_pct=10,
                            sek_estimate=10000, confidence=0.7, thesis="t")],
            cash_target_pct=8.0,
        )
        return decision, "{}", None, {"requested": 1, "succeeded": 1, "failed": 0, "errors": []}

    monkeypatch.setattr(whatif, "call_llm_consensus", _capture)
    whatif.generate_whatif("config_test.yaml", refresh_prices=False, n_runs=99, include_macro=False)
    assert seen["n"] == whatif.MAX_RUNS


def test_generate_records_votes_and_guardrail_verdicts(profile, monkeypatch):
    actions = [
        # Approved
        Action(ticker="ALFA.ST", side="buy", target_weight_pct=15,
               sek_estimate=15000, confidence=0.85, thesis="good"),
        # Rejected — not in universe
        Action(ticker="NOPE.ST", side="buy", target_weight_pct=10,
               sek_estimate=10000, confidence=0.6, thesis="bad"),
        # Clipped — above max_position_pct of 20
        Action(ticker="BETA.ST", side="buy", target_weight_pct=35,
               sek_estimate=35000, confidence=0.75, thesis="big"),
    ]
    votes = {"ALFA.ST": 3, "NOPE.ST": 2, "BETA.ST": 2}
    monkeypatch.setattr(whatif, "call_llm_consensus", _stub_consensus(actions, votes))

    result = whatif.generate_whatif("config_test.yaml", refresh_prices=False, n_runs=3, include_macro=False)

    by_ticker = {a["ticker"]: a for a in result["actions"]}
    assert by_ticker["ALFA.ST"]["status"] == "APPROVED"
    assert by_ticker["ALFA.ST"]["votes"] == 3
    assert by_ticker["ALFA.ST"]["name"] == "Alfa AB"
    assert by_ticker["NOPE.ST"]["status"] == "REJECTED"
    assert "not in universe" in by_ticker["NOPE.ST"]["reason"]
    assert by_ticker["BETA.ST"]["status"] == "CLIPPED"
    assert by_ticker["BETA.ST"]["target_weight_pct"] == 20
    assert result["consensus"] is True
    assert result["market_summary"] == "Stubbed market read."


def test_generate_never_writes_to_the_fund_book(profile, monkeypatch):
    """A what-if run must leave recommendations, NAV history and positions untouched."""
    store = Store(profile["db_path"])
    before = (
        store.count_recommendations(),
        len(store.get_nav_history()),
        len(store.get_positions()),
        store.get_cash(),
    )

    actions = [Action(ticker="ALFA.ST", side="buy", target_weight_pct=15,
                      sek_estimate=15000, confidence=0.85, thesis="good")]
    monkeypatch.setattr(whatif, "call_llm_consensus", _stub_consensus(actions, {"ALFA.ST": 3}))
    whatif.generate_whatif("config_test.yaml", refresh_prices=False, n_runs=3, include_macro=False)

    after = (
        store.count_recommendations(),
        len(store.get_nav_history()),
        len(store.get_positions()),
        store.get_cash(),
    )
    assert before == after


def test_result_is_persisted_and_listed(profile, monkeypatch):
    actions = [Action(ticker="ALFA.ST", side="buy", target_weight_pct=15,
                      sek_estimate=15000, confidence=0.85, thesis="good")]
    monkeypatch.setattr(whatif, "call_llm_consensus", _stub_consensus(actions, {"ALFA.ST": 3}))

    result = whatif.generate_whatif("config_test.yaml", refresh_prices=False, n_runs=3, include_macro=False)
    path = whatif.WHATIF_DIR / f"{result['id']}.json"
    assert path.exists()
    assert json.loads(path.read_text())["id"] == result["id"]

    listed = whatif.list_results()
    assert [r["id"] for r in listed] == [result["id"]]


# ── Amount to place / deployment mode ─────────────────────────────────────────

def _capture_cfg(monkeypatch, seen: dict):
    """Stub the LLM while recording the config the prompt was built with."""
    def _fake(system, user, cfg):
        seen["cfg"] = cfg
        seen["user"] = user
        decision = DecisionRun(
            run_id="stub", market_summary="ok",
            actions=[Action(ticker="ALFA.ST", side="buy", target_weight_pct=15,
                            sek_estimate=9000, confidence=0.8, thesis="t")],
            cash_target_pct=0.0,
        )
        return decision, "{}", None, {"requested": 1, "succeeded": 1, "failed": 0, "errors": []}
    monkeypatch.setattr(whatif, "call_llm_consensus", _fake)


def test_capital_override_sets_the_placed_amount(profile, monkeypatch):
    seen = {}
    _capture_cfg(monkeypatch, seen)

    result = whatif.generate_whatif("config_test.yaml", refresh_prices=False, capital_sek=60000, include_macro=False)

    assert seen["cfg"].capital_sek == 60000
    # The synthetic book the model sees is the placed amount, not the profile's
    assert "NAV: 60,000 SEK  |  Cash: 60,000 SEK (100.0%)" in seen["user"]
    assert result["deployment"]["placed_sek"] == 60000
    assert result["deployment"]["profile_capital_sek"] == 100000
    assert result["deployment"]["amount_overridden"] is True


def test_amount_defaults_to_profile_capital(profile, monkeypatch):
    seen = {}
    _capture_cfg(monkeypatch, seen)

    result = whatif.generate_whatif("config_test.yaml", refresh_prices=False, include_macro=False)

    assert seen["cfg"].capital_sek == 100000
    assert result["deployment"]["placed_sek"] == 100000
    assert result["deployment"]["amount_overridden"] is False


def test_full_deploy_lifts_turnover_cap_and_cash_floor(profile, monkeypatch):
    seen = {}
    _capture_cfg(monkeypatch, seen)

    result = whatif.generate_whatif("config_test.yaml", refresh_prices=False, deploy_full=True, include_macro=False)

    risk = seen["cfg"].risk
    assert risk.max_turnover_pct == 100.0
    assert risk.min_cash_pct == 0.0
    assert risk.max_cash_pct == 0.0  # or the guardrail would allow cash up to the ceiling
    assert "Deploy the ENTIRE amount in this single run" in seen["user"]
    assert result["deployment"]["full_deploy"] is True
    assert result["cash_target_pct"] == 0.0


def test_staged_deploy_keeps_the_cold_start_cap(profile, monkeypatch):
    seen = {}
    _capture_cfg(monkeypatch, seen)

    result = whatif.generate_whatif("config_test.yaml", refresh_prices=False, deploy_full=False, include_macro=False)

    risk = seen["cfg"].risk
    assert risk.max_turnover_pct == 100.0  # fixture's cold_start_turnover_pct
    assert risk.min_cash_pct == 5.0        # mandate floor still applies
    assert "Deploy the ENTIRE amount" not in seen["user"]
    assert result["deployment"]["full_deploy"] is False


def test_full_deploy_does_not_mutate_the_profile_defaults(profile, monkeypatch):
    """The risk overrides are per-run — a later staged run must be unaffected."""
    seen = {}
    _capture_cfg(monkeypatch, seen)

    whatif.generate_whatif("config_test.yaml", refresh_prices=False, deploy_full=True, include_macro=False)
    whatif.generate_whatif("config_test.yaml", refresh_prices=False, deploy_full=False, include_macro=False)

    assert seen["cfg"].risk.min_cash_pct == 5.0  # restored from the yaml, not leaked as 0


@pytest.mark.parametrize("amount", [0, -5000])
def test_non_positive_amount_rejected(profile, monkeypatch, amount):
    _capture_cfg(monkeypatch, {})
    with pytest.raises(ValueError, match="greater than 0"):
        whatif.generate_whatif("config_test.yaml", refresh_prices=False, capital_sek=amount, include_macro=False)


def test_amount_below_min_trade_size_rejected_before_spending_calls(profile, monkeypatch):
    """min_trade_sek is 1000 in the fixture — 500 SEK can never trade."""
    called = {"n": 0}

    def _should_not_run(system, user, cfg):
        called["n"] += 1
        raise AssertionError("LLM must not be called for an unusable amount")

    monkeypatch.setattr(whatif, "call_llm_consensus", _should_not_run)
    with pytest.raises(ValueError, match="below this profile's minimum trade size"):
        whatif.generate_whatif("config_test.yaml", refresh_prices=False, capital_sek=500, include_macro=False)
    assert called["n"] == 0


def test_undersized_amount_is_flagged_but_still_runs(profile, monkeypatch):
    """Between the hard floor and the comfortable floor, run but warn.

    Fixture: min_trade 1000, max_position_pct 20 → comfortable floor 5000 SEK.
    """
    seen = {}
    _capture_cfg(monkeypatch, seen)

    result = whatif.generate_whatif("config_test.yaml", refresh_prices=False, capital_sek=3000, include_macro=False)

    assert result["deployment"]["undersized"] is True
    assert result["deployment"]["comfortable_floor_sek"] == 5000


def test_deployment_floors_derive_from_risk_config(profile):
    cfg = whatif.load_profile_config("config_test.yaml")
    hard, comfortable = whatif.deployment_floors(cfg)
    assert hard == 1000              # min_trade_sek
    assert comfortable == 5000       # 1000 / 0.20


def test_profiles_expose_the_minimum_sensible_amount(profile):
    p = whatif.list_profiles()[0]
    assert p["min_amount_sek"] == 5000
    assert p["staged_turnover_pct"] == 100


def test_generate_errors_when_no_cached_data(tmp_path, monkeypatch):
    """A profile whose fund has never run has no cached prices to work from."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "universe_cold.csv").write_text(UNIVERSE_CSV)
    (config_dir / "mandate_cold.md").write_text(MANDATE)
    (config_dir / "config_cold.yaml").write_text(f"""
name: "Cold Fund"
capital_sek: 50000
db_path: "{tmp_path / 'cold.db'}"
mandate_path: "{config_dir / 'mandate_cold.md'}"
universe_path: "{config_dir / 'universe_cold.csv'}"
""")
    monkeypatch.setattr(whatif, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(whatif, "WHATIF_DIR", tmp_path / "whatif")

    with pytest.raises(RuntimeError, match="No cached price data"):
        whatif.generate_whatif("config_cold.yaml", include_macro=False)


# ── Web routes ────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    from fundmgr.web.app import app
    return TestClient(app)


def test_whatif_page_renders(client):
    res = client.get("/whatif/")
    assert res.status_code == 200
    assert "What-If Lab" in res.text
    # Every real profile is offered
    assert "config_buffett_gpt.yaml" in res.text


def test_amount_field_accepts_realistic_amounts(client):
    """Guards a subtle HTML5 trap: with a numeric `step`, browsers only accept
    values matching min + n*step, so step=1000 off min=1 rejects every round
    amount ("Ange ett giltigt värde" / "Enter a valid value") — including the
    prefilled profile defaults.
    """
    html = client.get("/whatif/").text
    tag = re.search(r'<input id="f-amount"[^>]*>', html)
    assert tag, "amount input not found"
    attrs = dict(re.findall(r'(\w+)="([^"]*)"', tag.group(0)))

    step = attrs.get("step", "1")
    minimum = float(attrs.get("min", 0))

    if step != "any":
        step_val = float(step)
        # Every amount a user could plausibly type must satisfy the step rule,
        # as must each profile's own capital (the prefilled value).
        amounts = [1000, 13889, 25000, 50000, 90000, 150000, 1_000_000]
        amounts += [float(p["capital_sek"]) for p in whatif.list_profiles()]
        bad = [a for a in amounts if abs((a - minimum) % step_val) > 1e-9]
        assert not bad, f'step="{step}" with min="{minimum}" rejects: {sorted(set(bad))}'


def test_generate_rejects_unknown_profile(client):
    res = client.post("/whatif/api/generate", json={"profile": "config_nope.yaml"})
    assert res.status_code == 400


def test_generate_rejects_unknown_model(client):
    res = client.post("/whatif/api/generate", json={
        "profile": "config.yaml", "provider": "openai", "model_id": "gpt-fake",
    })
    assert res.status_code == 400


def test_generate_rejects_out_of_range_run_count(client):
    res = client.post("/whatif/api/generate", json={
        "profile": "config.yaml", "n_runs": whatif.MAX_RUNS + 1,
    })
    assert res.status_code == 422


@pytest.mark.parametrize("amount", [0, -1, 10_000_000_000])
def test_generate_rejects_invalid_amount(client, amount):
    res = client.post("/whatif/api/generate", json={
        "profile": "config.yaml", "capital_sek": amount,
    })
    assert res.status_code == 422


def test_deployment_options_reach_the_engine(client, monkeypatch):
    from fundmgr.web import whatif as web_whatif

    seen = {}

    def _capture(**kwargs):
        seen.update(kwargs)
        return {"id": "whatif-stub", "actions": [], "profile": {"name": "x"}}

    monkeypatch.setattr(web_whatif, "generate_whatif", _capture)
    monkeypatch.setattr(web_whatif, "_job", None)

    job_id = client.post("/whatif/api/generate", json={
        "profile": "config.yaml", "capital_sek": 25000, "deploy_full": True, "n_runs": 1,
    }).json()["job_id"]

    for _ in range(50):
        job = client.get(f"/whatif/api/jobs/{job_id}").json()
        if job["status"] != "running":
            break
        __import__("time").sleep(0.1)

    assert seen["capital_sek"] == 25000
    assert seen["deploy_full"] is True


def test_job_lifecycle_and_single_slot(client, monkeypatch):
    """A job runs to completion, is pollable, and blocks a concurrent second job."""
    import threading

    from fundmgr.web import whatif as web_whatif

    release = threading.Event()

    def _slow_generate(**kwargs):
        release.wait(timeout=5)
        return {"id": "whatif-stub", "actions": [], "profile": {"name": "x"}}

    monkeypatch.setattr(web_whatif, "generate_whatif", _slow_generate)
    monkeypatch.setattr(web_whatif, "_job", None)

    res = client.post("/whatif/api/generate", json={"profile": "config.yaml", "n_runs": 1})
    assert res.status_code == 200
    job_id = res.json()["job_id"]

    assert client.get(f"/whatif/api/jobs/{job_id}").json()["status"] == "running"
    # Second submission is refused while the first is in flight
    assert client.post("/whatif/api/generate", json={"profile": "config.yaml"}).status_code == 409

    release.set()
    for _ in range(50):
        job = client.get(f"/whatif/api/jobs/{job_id}").json()
        if job["status"] != "running":
            break
        __import__("time").sleep(0.1)
    assert job["status"] == "done"
    assert job["result"]["id"] == "whatif-stub"


def test_job_reports_generation_failure(client, monkeypatch):
    from fundmgr.web import whatif as web_whatif

    def _boom(**kwargs):
        raise RuntimeError("no cached price data")

    monkeypatch.setattr(web_whatif, "generate_whatif", _boom)
    monkeypatch.setattr(web_whatif, "_job", None)

    job_id = client.post("/whatif/api/generate",
                         json={"profile": "config.yaml", "n_runs": 1}).json()["job_id"]
    for _ in range(50):
        job = client.get(f"/whatif/api/jobs/{job_id}").json()
        if job["status"] != "running":
            break
        __import__("time").sleep(0.1)
    assert job["status"] == "error"
    assert "no cached price data" in job["error"]


def test_unknown_job_id_is_404(client):
    assert client.get("/whatif/api/jobs/deadbeef").status_code == 404


# ── Candidate refresh ─────────────────────────────────────────────────────────

def test_refresh_refetches_only_the_screened_candidates(profile, monkeypatch):
    """The global profile is 17k tickers on a 2.5k weekly rotation — refreshing
    the universe for one what-if is hours. Only what the model sees is refetched."""
    seen = {}

    def _fake_prices(tickers, store, lookback_days, force_refresh=False):
        seen["tickers"] = [t.yahoo_ticker for t in tickers]
        seen["force"] = force_refresh
        return {t.yahoo_ticker: True for t in tickers}

    monkeypatch.setattr(whatif, "fetch_and_cache_prices", _fake_prices)
    monkeypatch.setattr(whatif, "fetch_and_cache_fundamentals", lambda *a, **k: 0)
    monkeypatch.setattr(whatif, "fetch_and_cache_benchmark", lambda *a, **k: True)
    monkeypatch.setattr(whatif, "call_llm_consensus", _stub_consensus(
        [Action(ticker="ALFA.ST", side="buy", target_weight_pct=10,
                sek_estimate=10000, confidence=0.7, thesis="t")], {"ALFA.ST": 3}))

    result = whatif.generate_whatif("config_test.yaml", include_macro=False)

    assert seen["force"] is True, "a refresh that honours the cache is not a refresh"
    # Never more than the candidates the model is shown.
    assert 0 < len(seen["tickers"]) <= result["data"]["candidates_to_llm"]
    assert result["data"]["refresh"]["refreshed"] is True
    assert result["data"]["refresh"]["tickers"] == len(seen["tickers"])


def test_refresh_can_be_skipped_for_a_cache_only_run(profile, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("no fetching when refresh_prices=False")

    monkeypatch.setattr(whatif, "fetch_and_cache_prices", _boom)
    monkeypatch.setattr(whatif, "call_llm_consensus", _stub_consensus(
        [Action(ticker="ALFA.ST", side="buy", target_weight_pct=10,
                sek_estimate=10000, confidence=0.7, thesis="t")], {"ALFA.ST": 3}))

    result = whatif.generate_whatif(
        "config_test.yaml", include_macro=False, refresh_prices=False
    )
    assert result["data"]["refresh"] == {"refreshed": False}


def test_a_failed_refresh_does_not_take_the_run_down(profile, monkeypatch):
    """Market data is best-effort here; a fetch outage should degrade to the
    cached run, not lose the whole what-if."""
    monkeypatch.setattr(whatif, "fetch_and_cache_prices",
                        lambda tickers, *a, **k: {t.yahoo_ticker: True for t in tickers})
    monkeypatch.setattr(whatif, "fetch_and_cache_fundamentals",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("yfinance down")))
    monkeypatch.setattr(whatif, "fetch_and_cache_benchmark",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("yfinance down")))
    monkeypatch.setattr(whatif, "call_llm_consensus", _stub_consensus(
        [Action(ticker="ALFA.ST", side="buy", target_weight_pct=10,
                sek_estimate=10000, confidence=0.7, thesis="t")], {"ALFA.ST": 3}))

    result = whatif.generate_whatif("config_test.yaml", include_macro=False)
    assert result["data"]["refresh"]["fundamentals_refreshed"] == 0
    assert result["data"]["refresh"]["benchmark_ok"] is False
    assert result["data"]["candidates_to_llm"] > 0
