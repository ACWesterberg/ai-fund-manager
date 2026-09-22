"""No paid calls or portfolio mutations: frozen replay and accounting contracts."""
import copy
import json
from datetime import datetime, timezone

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from fundmgr.config import AppConfig
from fundmgr.data.prices import TickerFeatures
from fundmgr.engine import experiments as ex
from fundmgr.engine.client import _aggregate_decisions
from fundmgr.engine.prompt import assemble_system_prompt, build_prompt, snapshot_to_dict
from fundmgr.engine.schema import Action, DecisionRun
from fundmgr.state.models import PortfolioSnapshot, Position, RecommendationLog
from fundmgr.state.store import Store


def action(ticker, side="buy", weight=10, amount=10000, **kwargs):
    return Action(ticker=ticker, side=side, target_weight_pct=weight,
                  sek_estimate=amount, confidence=.8, thesis="Test", **kwargs)


def decision(*actions):
    return DecisionRun(run_id="r", market_summary="Test", actions=list(actions), cash_target_pct=15)


@pytest.fixture
def frozen(tmp_path):
    cfg = AppConfig()
    cfg.db_path = tmp_path / "fund.db"
    cfg.mandate_path = tmp_path / "mandate.md"
    cfg.mandate_path.write_text("Allocate sensibly.")
    cfg.optimizer.compiled_dir = tmp_path / "compiled"
    cfg.risk.max_turnover_pct = 50
    cfg.llm.n_samples = 3
    cfg.evaluation_horizon_days = 2
    features = {t: TickerFeatures(ticker=t, name=t, last_price=100,
                                 last_date="2026-01-01", data_age_trading_days=0,
                                 sector=sector, country="US")
                for t, sector in (("A", "Technology"), ("B", "Industrials"))}
    snap = PortfolioSnapshot([], 100000, timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc))
    store = Store(cfg.db_path)
    system, user, fields = build_prompt(cfg, snap, features, store, "r")
    snapshot = json.loads(snapshot_to_dict(snap, system, user, fields, cfg,
                                          features=features, universe_tickers=set(features)))
    candidate = {"fund_id": "fund", "provider": cfg.llm.provider,
                 "task_model": cfg.llm.model_id, "horizon_days": 2,
                 "mandate": "Allocate sensibly.", "instructions": "Candidate rule",
                 "incumbent_guidance_hash": None, "status": "pending_evaluation",
                 "created_at": "2025-12-31T00:00:00+00:00"}
    return cfg, snapshot, candidate


def patch_calls(monkeypatch, *decisions, failed=0):
    calls = []
    pending = iter(decisions)
    def call(system, user, cfg):
        calls.append((system, user, cfg))
        d = next(pending)
        return d, d.model_dump_json(), {}, {"requested": 3, "succeeded": 3-failed,
                                           "failed": failed, "errors": []}
    monkeypatch.setattr(ex, "call_llm_consensus", call)
    return calls


def paired(frozen, monkeypatch):
    _, snapshot, candidate = frozen
    patch_calls(monkeypatch, decision(action("A")), decision(action("B")))
    return ex.compare_guidance(snapshot, candidate)


def labels(pair):
    return ex.Outcomes.model_validate({
        "case_hash": pair["case_hash"], "basis": "synthetic_native", "benchmark": "^OMXSPI",
        "source": "Fixed test closing marks",
        "observations": [
            {"date": "2026-01-01", "prices": {"A": 100, "B": 100}, "benchmark": 100},
            {"date": "2026-01-02", "prices": {"A": 95, "B": 80}, "benchmark": 99},
            {"date": "2026-01-03", "prices": {"A": 90, "B": 120}, "benchmark": 101},
        ],
    })


def test_both_arms_use_same_inputs_and_live_consensus_settings(frozen, monkeypatch):
    cfg, snapshot, candidate = frozen
    before = copy.deepcopy(snapshot)
    calls = patch_calls(monkeypatch, decision(action("A")), decision(action("B")))
    pair = ex.compare_guidance(snapshot, candidate)
    assert snapshot == before
    assert calls[0][0] == snapshot["system_message"]
    assert calls[1][0] == assemble_system_prompt(candidate["mandate"], "Candidate rule")
    assert calls[0][1] == calls[1][1] == snapshot["user_message"]
    assert calls[0][2].llm == calls[1][2].llm == cfg.llm
    assert pair["mode"] == "retrospective_diagnostic"
    assert not pair["promotion_eligible"]
    assert all(arm["status"] == "ready" for arm in pair["arms"].values())
    assert Store(cfg.db_path).get_transactions() == []
    assert Store(cfg.db_path).get_cash() == 0


def test_score_uses_weights_cash_fees_benchmark_and_daily_drawdown(frozen, monkeypatch):
    pair = paired(frozen, monkeypatch)
    result = ex.score_comparison(pair, labels(pair))
    assert result["scores"]["incumbent"]["net_return_pct"] == pytest.approx(-1.01)
    assert result["scores"]["candidate"]["net_return_pct"] == pytest.approx(1.99)
    assert result["scores"]["candidate"]["excess_return_pp"] == pytest.approx(.99)
    assert result["scores"]["candidate"]["max_daily_drawdown_pct"] == pytest.approx(2.01)
    assert result["scores"]["candidate"]["fees"] == 10
    assert result["candidate_advantage_pp"] == pytest.approx(3)
    assert not result["promotion_eligible"]


def test_larger_weight_changes_reward(frozen, monkeypatch):
    _, snapshot, candidate = frozen
    patch_calls(monkeypatch, decision(action("B", weight=5, amount=5000)), decision(action("B")))
    pair = ex.compare_guidance(snapshot, candidate)
    result = ex.score_comparison(pair, labels(pair))
    assert result["scores"]["incumbent"]["net_return_pct"] == pytest.approx(.995)
    assert result["candidate_advantage_pp"] == pytest.approx(.995)


def test_cash_and_omitted_holdings_remain_in_portfolio(frozen):
    _, snapshot, _ = frozen
    case = ex.load_case(snapshot)
    case.positions = [Position("A", 100, 90, 100)]
    case.cash = 90000
    plan = ex.project_allocation(case, [])
    assert plan["shares"] == {"A": 100}
    assert plan["cash"] == 90000
    assert plan["initial_nav"] == plan["post_trade_nav"] == 100000


def test_sells_reduce_owned_units_and_deduct_fees(frozen):
    case = ex.load_case(frozen[1])
    case.positions = [Position("A", 100, 90, 100)]
    case.cash = 90000
    plan = ex.project_allocation(case, [action("A", "sell", weight=0).model_dump()])
    assert plan["shares"] == {}
    assert plan["cash"] == 99990
    with pytest.raises(ValueError, match="owned position"):
        ex.project_allocation(ex.load_case(frozen[1]), [action("A", "sell", weight=0).model_dump()])


@pytest.mark.parametrize("mutation,match", [
    (lambda c: setattr(c.risk, "max_positions", 1), "position count"),
    (lambda c: setattr(c.features["B"], "sector", "Technology"), "sector cap"),
    (lambda c: setattr(c.risk, "max_turnover_pct", 20), "turnover"),
])
def test_cumulative_feasibility_is_checked(frozen, mutation, match):
    case = ex.load_case(frozen[1])
    mutation(case)
    with pytest.raises(ValueError, match=match):
        ex.project_allocation(case, [action("A", weight=18).model_dump(), action("B", weight=18).model_dump()])


def test_fees_and_aggregate_buys_cannot_breach_cash_floor(frozen):
    case = ex.load_case(frozen[1])
    case.cash = 30000
    case.positions = [Position("A", 700, 100, 100)]
    case.features["C"] = TickerFeatures(ticker="C", name="C", last_price=100,
                                        last_date="2026-01-01", data_age_trading_days=0, sector="Other")
    with pytest.raises(ValueError, match="cash floor"):
        ex.project_allocation(case, [action("B").model_dump(), action("C").model_dump()])


def test_actual_sizing_does_not_trust_model_estimate(frozen):
    case = ex.load_case(frozen[1])
    plan = ex.project_allocation(case, [action("A", weight=18, amount=2500).model_dump()])
    assert plan["trades"][0]["gross"] == 18000
    assert plan["fees"] == 18


@pytest.mark.parametrize("key,value", [("provider", "other"), ("fund_id", "other"),
                                      ("horizon_days", 90), ("mandate", "changed"),
                                      ("incumbent_guidance_hash", "changed")])
def test_wrong_candidate_rejected_before_model_calls(frozen, monkeypatch, key, value):
    _, snapshot, candidate = frozen
    candidate[key] = value
    calls = patch_calls(monkeypatch)
    with pytest.raises(ValueError):
        ex.compare_guidance(snapshot, candidate)
    assert not calls


def test_missing_context_and_missing_fx_rejected_before_calls(frozen, monkeypatch):
    _, snapshot, candidate = frozen
    calls = patch_calls(monkeypatch)
    with pytest.raises(ValueError, match="lacks frozen"):
        ex.compare_guidance({}, candidate)
    snapshot["evaluation_case"]["basis"] = "SEK"
    snapshot["evaluation_case"]["features"]["A"]["currency"] = "EUR"
    with pytest.raises(ValueError, match="Missing decision-time FX"):
        ex.compare_guidance(snapshot, candidate)
    assert not calls


def test_fx_converts_trade_amount_to_book_currency(frozen):
    case = ex.load_case(frozen[1])
    case.basis = "SEK"
    case.features["A"].currency = "EUR"
    case.fx_rates = {"EUR": 10}
    plan = ex.project_allocation(case, [action("A").model_dump()])
    assert plan["shares"]["A"] == 10  # 10k SEK / (100 EUR * 10 SEK/EUR)


def test_incomplete_consensus_cannot_be_scored_as_a_valid_arm(frozen, monkeypatch):
    _, snapshot, candidate = frozen
    patch_calls(monkeypatch, decision(action("A")), decision(action("B")), failed=1)
    pair = ex.compare_guidance(snapshot, candidate)
    assert all(arm["status"] == "invalid" for arm in pair["arms"].values())
    with pytest.raises(ValueError, match="invalid"):
        ex.score_comparison(pair, labels(pair))


@pytest.mark.parametrize("change,match", [
    (lambda d: d["observations"][-1]["prices"].pop("A"), "candidate marks"),
    (lambda d: d["observations"].pop(1), "every calendar day"),
    (lambda d: d.update(basis="SEK"), "currency convention"),
    (lambda d: d.update(benchmark="other"), "benchmark"),
    (lambda d: d.update(case_hash="other"), "different frozen"),
    (lambda d: d["observations"][0]["prices"].update(A=101), "Starting mark"),
])
def test_incomplete_or_mismatched_labels_never_get_a_score(frozen, monkeypatch, change, match):
    pair = paired(frozen, monkeypatch)
    data = labels(pair).model_dump(mode="json")
    change(data)
    with pytest.raises(ValueError, match=match):
        ex.score_comparison(pair, ex.Outcomes.model_validate(data))


def test_modified_decision_artifact_is_detected(frozen, monkeypatch):
    pair = paired(frozen, monkeypatch)
    pair["arms"]["candidate"]["allocation"]["cash"] += 10000
    with pytest.raises(ValueError, match="modified"):
        ex.score_comparison(pair, labels(pair))


def test_reports_are_not_overwritten(tmp_path):
    path = tmp_path / "result.json"
    ex.write_report(path, {"first": True})
    with pytest.raises(FileExistsError):
        ex.write_report(path, {"first": False})
    assert json.loads(path.read_text()) == {"first": True}


def test_upstream_candidates_are_all_rendered_and_archived(tmp_path, frozen):
    cfg, _, _ = frozen
    features = {f"T{i}": TickerFeatures(ticker=f"T{i}", name="Test", last_price=100,
                                      last_date="2026-01-01", data_age_trading_days=0)
                for i in range(120)}
    store = Store(tmp_path / "screen.db")
    snap = PortfolioSnapshot([], 100000)
    system, user, fields = build_prompt(cfg, snap, features, store, "r")
    assert "Showing 120 of 120" in user
    snapshot = json.loads(snapshot_to_dict(snap, system, user, fields, cfg, features=features))
    assert len(ex.load_case(snapshot).features) == 120


def test_empty_consensus_is_a_valid_no_trade_decision():
    consensus, votes = _aggregate_decisions([decision(action("A")), decision(action("B")), decision(action("C"))])
    assert consensus.actions == [] and votes == {}
    with pytest.raises(ValidationError, match="at most one"):
        decision(action("A"), action("A"))


def test_consensus_preserves_the_selected_thesis_monitoring_plan():
    a = action("A", kill_criterion="Margin below 20%", target_price=150)
    consensus, _ = _aggregate_decisions([decision(a), decision(a), decision(a)])
    assert consensus.actions[0].kill_criterion == a.kill_criterion
    assert consensus.actions[0].target_price == 150


def test_score_cli_is_offline_and_writes_report(frozen, monkeypatch, tmp_path):
    from fundmgr.cli import cli
    pair = paired(frozen, monkeypatch)
    p, o, result = (tmp_path / name for name in ("pair.json", "outcomes.json", "score.json"))
    ex.write_report(p, pair)
    ex.write_report(o, labels(pair).model_dump(mode="json"))
    monkeypatch.setattr(ex, "call_llm_consensus", lambda *a: pytest.fail("Scoring must be offline"))
    run = CliRunner().invoke(cli, ["score-guidance", str(p), "--outcomes", str(o), "--output", str(result)])
    assert run.exit_code == 0, run.output
    assert "Candidate advantage: +3.00pp" in run.output
    assert json.loads(result.read_text())["candidate_advantage_pp"] == pytest.approx(3)


def test_compare_cli_records_saved_case_without_changing_store(frozen, monkeypatch, tmp_path):
    import fundmgr.cli as commands

    cfg, snapshot, candidate = frozen
    store = Store(cfg.db_path)
    store.save_recommendation(RecommendationLog(
        run_id="r", timestamp=datetime(2026, 1, 1), prompt_snapshot=json.dumps(snapshot),
        llm_response="original", guardrail_log="{}", actions_json="[]"))
    monkeypatch.setattr(commands, "_get_store", lambda: (cfg, store))
    calls = patch_calls(monkeypatch, decision(action("A")), decision(action("B")))
    candidate_path, output = tmp_path / "candidate.json", tmp_path / "comparison.json"
    candidate_path.write_text(json.dumps(candidate))
    args = ["compare-guidance", "--run-id", "r", "--candidate", str(candidate_path),
            "--output", str(output)]
    result = CliRunner().invoke(commands.cli, args)
    assert result.exit_code == 0, result.output
    assert len(calls) == 2
    assert json.loads(output.read_text())["arms"]["candidate"]["status"] == "ready"
    assert store.get_recommendation_by_run_id("r").llm_response == "original"
    result = CliRunner().invoke(commands.cli, args)
    assert result.exit_code != 0 and "new file" in result.output
    assert len(calls) == 2  # Reject output collisions before any further paid calls.
