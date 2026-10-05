"""Offline artifact verification, chronological selection and coverage accounting."""
import json
import shutil
from datetime import date, timedelta

import pytest
from click.testing import CliRunner

from fundmgr.engine import aggregate as ag, forward as fw
from tests.test_forward import setup as forward_setup, record, mature, bundle

setup = forward_setup


def row(start, advantage=1, status="scored", group="g", drawdown=1, turnover=10):
    start = date.fromisoformat(start)
    return {"group": group, "identity": {"fund_id": group}, "run_id": start.isoformat(),
            "registration_hash": start.isoformat(), "decision_time": start.isoformat(),
            "start": start.isoformat(), "end": (start + timedelta(days=28)).isoformat(),
            "status": status, "score": {"candidate_advantage_pp": advantage,
                "scores": {"incumbent": {"net_return_pct": 1, "excess_return_pp": 0,
                    "max_daily_drawdown_pct": 2, "turnover_pct": 10, "fees": 10},
                    "candidate": {"net_return_pct": 1+advantage, "excess_return_pp": advantage,
                    "max_daily_drawdown_pct": drawdown, "turnover_pct": turnover, "fees": 10}}}}


def test_overlapping_winners_never_replace_missing_earlier_window():
    rows = [row("2025-01-01", status="missing_outcomes"), row("2025-01-08", advantage=50),
            row("2025-02-01", advantage=-1)]
    result = ag.summarize(rows, 2)[0]
    assert result["selected_registrations"] == ["2025-01-01", "2025-02-01"]
    assert result["overlapping_registrations"] == ["2025-01-08"]
    assert result["coverage"] == .5 and result["status"] == "incomplete_evidence"
    assert result["metrics"]["mean_advantage_pp"] == -1


def test_inclusive_boundary_and_input_order_do_not_change_selection():
    rows = [row("2025-01-01"), row("2025-01-29", advantage=50), row("2025-01-30")]
    a, b = ag.summarize(rows, 2)[0], ag.summarize(list(reversed(rows)), 2)[0]
    assert a["selected_registrations"] == b["selected_registrations"] == ["2025-01-01", "2025-01-30"]
    assert a["status"] == "promising_descriptive" and a["promotion_eligible"] is False


def test_small_positive_sample_is_not_enough():
    result = ag.summarize([row("2025-01-01"), row("2025-02-01")])[0]
    assert result["status"] == "insufficient_periods"
    assert result["metrics"]["win_fraction"] == 1


@pytest.mark.parametrize("change", [{"drawdown": 3}, {"turnover": 20}, {"advantage": -1}])
def test_returns_downside_and_turnover_must_all_support_descriptive_label(change):
    result = ag.summarize([row("2025-01-01", **change), row("2025-02-01", **change)], 2)[0]
    assert result["status"] == "mixed_results"


def test_groups_stay_separate_and_unresolved_recordings_block_claims():
    rows = [row("2025-01-01"), row("2025-02-01", group="other")]
    missing = row("2025-03-01", status="recording_incomplete")
    missing.pop("start")
    rows.append(missing)
    result = ag.summarize(rows, 2)
    assert len(result) == 2
    assert result[0]["status"] == "incomplete_evidence"
    assert result[1]["status"] == "insufficient_periods"


def scored(setup):
    cfg = setup[0]
    directory, _, _ = record(setup)
    mature()
    assert fw.collect_forward(cfg, lambda s, *a: bundle()[s])[0]["status"] == "scored"
    return cfg, directory


def test_real_artifacts_are_recomputed_without_network_or_model_calls(setup, monkeypatch):
    cfg, directory = scored(setup)
    monkeypatch.setattr(fw, "fetch_history", lambda *a: pytest.fail("No network"))
    monkeypatch.setattr(fw, "compare_guidance", lambda *a: pytest.fail("No model calls"))
    result = ag.aggregate_evidence([fw.experiment_root(cfg)])
    assert result["counts"] == {"scored": 1}
    assert result["groups"][0]["metrics"]["mean_advantage_pp"] == pytest.approx(1)
    assert result["cases"][0]["path"] == str(directory.resolve())
    assert result["promotion_eligible"] is False


@pytest.mark.parametrize("artifact", ["result.json", "history.json", "outcomes.json", "valuation_case.json"])
def test_tampered_artifacts_are_rejected(setup, artifact):
    cfg, directory = scored(setup)
    path = directory / artifact if artifact == "result.json" else next(directory.glob(f"collections/*/{artifact}"))
    value = json.loads(path.read_text())
    if artifact == "result.json":
        value["candidate_advantage_pp"] = 999
    elif artifact == "history.json":
        value["A"]["rows"][0]["close"] = 999
    else:
        value["unexpected"] = True
    path.write_text(json.dumps(value))
    report = ag.aggregate_evidence([fw.experiment_root(cfg)])
    assert report["counts"] == {"invalid": 1}
    assert report["groups"][0]["status"] == "incomplete_evidence"


def test_copy_does_not_multiply_evidence(setup):
    cfg, directory = scored(setup)
    shutil.copytree(directory, directory.parent / "copied")
    report = ag.aggregate_evidence([fw.experiment_root(cfg), fw.experiment_root(cfg)])
    assert report["counts"] == {"scored": 1, "duplicate": 1}
    assert report["groups"][0]["scored_selected_periods"] == 1


def test_conflicting_copy_cannot_hide_invalid_evidence(setup):
    cfg, directory = scored(setup)
    target = directory.parent / "copied"
    shutil.copytree(directory, target)
    value = json.loads((target / "result.json").read_text())
    value["candidate_advantage_pp"] = 999
    (target / "result.json").write_text(json.dumps(value))
    report = ag.aggregate_evidence([fw.experiment_root(cfg)])
    assert report["counts"] == {"invalid": 2}


def test_incomplete_reservation_is_visible(setup):
    cfg, directory = scored(setup)
    (directory.parent / "interrupted").mkdir()
    report = ag.aggregate_evidence([fw.experiment_root(cfg)])
    assert report["counts"]["invalid"] == 1


def test_cli_writes_immutable_offline_report(setup, tmp_path):
    from fundmgr.cli import cli
    cfg, _ = scored(setup)
    output = tmp_path / "aggregate.json"
    args = ["evaluate-guidance", "--root", str(fw.experiment_root(cfg)), "--output", str(output)]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert "insufficient_periods" in result.output
    assert json.loads(output.read_text())["report_hash"]
    assert CliRunner().invoke(cli, args).exit_code != 0


def test_waiting_period_not_in_matured_coverage_denominator():
    future = row("2099-01-01", status="waiting")
    result = ag.summarize([row("2025-01-01"), future], 2)[0]
    assert result["selected_periods"] == 2
    assert result["matured_selected_periods"] == 1
    assert result["coverage"] == 1 and result["status"] == "insufficient_periods"


def test_cli_saves_errors_before_nonzero_exit(setup, tmp_path):
    from fundmgr.cli import cli
    cfg, directory = scored(setup)
    (directory / "result.json").write_text("{}")
    output = tmp_path / "audit-errors.json"
    result = CliRunner().invoke(cli, ["evaluate-guidance", "--root", str(fw.experiment_root(cfg)),
                                     "--output", str(output)])
    assert result.exit_code != 0
    assert json.loads(output.read_text())["counts"] == {"invalid": 1}
