"""The no-lessons shadow arm: the same frozen run, asked without its lessons.

The fund changed its prompt every week from its own outcomes and nothing
measured whether that helped. This pair does, without touching the real book.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from fundmgr import cli as commands
from fundmgr.config import AppConfig, ShadowConfig
from fundmgr.data.prices import TickerFeatures
from fundmgr.engine import experiments as ex, forward as fw, prompt as pr
from fundmgr.engine.prompt import build_prompt, snapshot_to_dict
from fundmgr.engine.schema import Action, DecisionRun
from fundmgr.state.models import Learning, PortfolioSnapshot
from fundmgr.state.store import Store


class Clock(datetime):
    current = datetime(2026, 1, 5, 12, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls.current if tz else cls.current.replace(tzinfo=None)

    @classmethod
    def utcnow(cls):
        return cls.current.replace(tzinfo=None)


LESSONS = ["Momentum names that gap up on volume tend to hold the move.",
           "Trim winners past target rather than letting them round-trip."]


def _features():
    return {t: TickerFeatures(t, t, 100, "2026-01-02", 0, sector=sector)
            for t, sector in (("A", "Technology"), ("B", "Industrials"))}


def _prompt(cfg, store):
    snap = PortfolioSnapshot([], 100000, timestamp=Clock.current)
    system, user, fields = build_prompt(cfg, snap, _features(), store, "weekly-1")
    return json.loads(snapshot_to_dict(snap, system, user, fields, cfg, features=_features()))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    for module in (fw, ex, pr):
        monkeypatch.setattr(module, "datetime", Clock)
    monkeypatch.setattr(fw, "_session", lambda mic, day: day.weekday() < 5)
    cfg = AppConfig(db_path=tmp_path / "test.db", mandate_path=tmp_path / "mandate.md")
    cfg.mandate_path.write_text("Allocate sensibly")
    cfg.optimizer.compiled_dir = tmp_path / "compiled"
    cfg.llm.n_samples = 1
    cfg.evaluation_horizon_days = 2
    store = Store(cfg.db_path)
    for body in LESSONS:
        store.save_learning(Learning(category="qualitative", body=body, run_ids=["r1", "r2"]))
    Clock.current = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    variant, _ = fw.create_variant(cfg, "no_lessons")          # created before the run
    Clock.current = datetime(2026, 1, 5, 12, tzinfo=timezone.utc)
    cfg.shadow = ShadowConfig(None, "SEK", "XSTO", variants=["no_lessons"])
    tickers = [SimpleNamespace(yahoo_ticker=t, exchange="OMXS") for t in _features()]
    calls = []

    def model(system, user, cfg_):
        calls.append(user)
        run = DecisionRun(run_id="test", market_summary="Test", cash_target_pct=90,
            actions=[Action(ticker="A", side="buy", target_weight_pct=10,
                            sek_estimate=10000, confidence=.9, thesis="Test")])
        return run, run.model_dump_json(), {}, {"requested": 1, "succeeded": 1, "failed": 0}
    monkeypatch.setattr(ex, "call_llm_consensus", model)
    return cfg, store, variant, tickers, calls


def test_the_variant_arm_is_the_prompt_built_without_lessons(setup, tmp_path):
    cfg, store, variant, _, _ = setup
    snapshot = _prompt(cfg, store)
    case = ex.load_case(snapshot)
    system, user = ex.candidate_prompt(case, snapshot, json.loads(variant.read_text()))
    assert all(lesson in case.user for lesson in LESSONS)
    assert not any(lesson in user for lesson in LESSONS)
    assert system == case.system
    bare = _prompt(cfg, Store(tmp_path / "empty.db"))
    assert user == bare["user_message"]


def test_a_run_records_the_variant_pair_with_lessons_in_one_arm_only(setup):
    cfg, store, variant, tickers, calls = setup
    snapshot = _prompt(cfg, store)
    directory = fw.record_forward(cfg, snapshot, "weekly-1", tickers, candidate_path=str(variant))
    pair = json.loads((directory / "comparison.json").read_text())
    assert pair["arms"]["incumbent"]["status"] == pair["arms"]["candidate"]["status"] == "ready"
    incumbent, without = calls
    assert LESSONS[0] in incumbent and LESSONS[0] not in without
    assert "user" in pair["arms"]["candidate"] and "user" not in pair["arms"]["incumbent"]


def test_the_variant_pair_never_collides_with_the_primary_one(setup):
    cfg, store, variant, tickers, _ = setup
    snapshot = _prompt(cfg, store)
    primary = fw.experiment_root(cfg) / ex.digest({"run_id": "weekly-1"})[:24]
    directory = fw.record_forward(cfg, snapshot, "weekly-1", tickers, candidate_path=str(variant))
    assert directory != primary
    assert fw.record_forward(cfg, snapshot, "weekly-1", tickers, candidate_path=str(variant)) == directory


def test_a_run_without_lessons_pays_for_nothing(setup, tmp_path):
    cfg, _, variant, tickers, calls = setup
    snapshot = _prompt(cfg, Store(tmp_path / "empty.db"))
    with pytest.raises(ValueError, match="no lessons"):
        fw.record_forward(cfg, snapshot, "weekly-1", tickers, candidate_path=str(variant))
    assert not calls and not fw.experiment_root(cfg).exists()


def test_a_variant_must_keep_the_incumbent_guidance(setup):
    cfg, store, variant, _, _ = setup
    case = ex.load_case(_prompt(cfg, store))
    altered = {**json.loads(variant.read_text()), "instructions": "Something else"}
    with pytest.raises(ValueError, match="incumbent's guidance"):
        ex.validate_candidate(case, altered)
    with pytest.raises(ValueError, match="Unknown candidate variant"):
        ex.validate_candidate(case, {**json.loads(variant.read_text()), "variant": "no_mandate"})


def test_creating_a_variant_twice_keeps_the_first(setup, monkeypatch):
    cfg, store, variant, _, _ = setup
    first = variant.read_text()
    monkeypatch.setattr(commands, "_get_store", lambda cfg_=None: (cfg, store))
    result = CliRunner().invoke(commands.cli, ["shadow-variant", "no-lessons"])
    assert result.exit_code == 0 and "already exists" in result.output
    assert variant.read_text() == first
    result = CliRunner().invoke(commands.cli, ["shadow-variant", "no_lessons", "--replace"])
    assert result.exit_code == 0 and "written" in result.output
    assert CliRunner().invoke(commands.cli, ["shadow-variant", "bogus"]).exit_code != 0


# ── The traded decision as the incumbent arm ──────────────────────────────────

def _production(n=1, ticker="B"):
    run = DecisionRun(run_id="weekly-1", market_summary="Prod", cash_target_pct=90,
        actions=[Action(ticker=ticker, side="buy", target_weight_pct=10,
                        sek_estimate=10000, confidence=.8, thesis="Prod")])
    return {"decision": run, "response": run.model_dump_json(), "votes": {ticker: n},
            "sampling": {"requested": n, "succeeded": n, "failed": 0, "errors": []}}


def test_the_traded_decision_is_the_incumbent_and_only_the_candidate_is_asked(setup):
    """The first shadow week re-asked the incumbent twice and got IVSO; the owner traded BONEX."""
    cfg, store, variant, tickers, calls = setup
    snapshot = _prompt(cfg, store)
    directory = fw.record_forward(cfg, snapshot, "weekly-1", tickers,
                                  candidate_path=str(variant), incumbent=_production())
    pair = json.loads((directory / "comparison.json").read_text())
    registration = json.loads((directory / "registration.json").read_text())
    assert len(calls) == 1 and LESSONS[0] not in calls[0]          # only the no-lessons arm paid
    incumbent = pair["arms"]["incumbent"]
    assert incumbent["source"] == "production_run" and incumbent["status"] == "ready"
    assert [a["ticker"] for a in incumbent["approved"]] == ["B"]
    assert registration["incumbent_source"] == "production_run"


def test_an_incomplete_production_vote_cannot_be_scored(setup):
    cfg, store, variant, tickers, _ = setup
    production = _production()
    production["sampling"] = {"requested": 1, "succeeded": 0, "failed": 1, "errors": ["x"]}
    directory = fw.record_forward(cfg, _prompt(cfg, store), "weekly-1", tickers,
                                  candidate_path=str(variant), incumbent=production)
    pair = json.loads((directory / "comparison.json").read_text())
    assert pair["arms"]["incumbent"]["status"] == "invalid"


def test_rerun_and_traded_incumbents_are_separate_evidence_groups(setup, tmp_path):
    from fundmgr.engine import aggregate
    cfg, store, variant, tickers, _ = setup
    snapshot = _prompt(cfg, store)
    traded = fw.record_forward(cfg, snapshot, "weekly-1", tickers,
                               candidate_path=str(variant), incumbent=_production())
    rerun = fw.record_forward(cfg, snapshot, "weekly-2", tickers, candidate_path=str(variant))
    groups = {aggregate.audit_case(d)["identity"]["incumbent_source"]: aggregate.audit_case(d)["group"]
              for d in (traded, rerun)}
    assert set(groups) == {"production_run", "rerun"}
    assert groups["production_run"] != groups["rerun"]
