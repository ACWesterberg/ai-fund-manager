"""Deposits and withdrawals move the money, not the return.

`set-cash` overwrote the balance, and every return figure divided NAV by where
it started — so a 50k top-up on a 55k book read as +100% profit, in the
dashboard, the weekly Telegram line and the run score the optimizer reads.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from click.testing import CliRunner

from fundmgr import cli as commands
from fundmgr.config import AppConfig
from fundmgr.reporting.dashboard import compute_stats, gain, nav_chart_json, return_index
from fundmgr.state.models import NavPoint
from fundmgr.state.store import Store

TODAY = datetime.utcnow().strftime("%Y-%m-%d")


def _day(n: int) -> str:
    return (datetime.utcnow() - timedelta(days=n)).strftime("%Y-%m-%d")


def _nav(date: str, nav: float, bench: float = 100.0) -> NavPoint:
    return NavPoint(date=date, portfolio_nav_sek=nav, benchmark_value=bench, cash_sek=0.0)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "fund.db")
    s.initialise(50_000)
    return s


# ── The scenario that motivated it ────────────────────────────────────────────

def test_a_top_up_halves_the_gain_on_money_put_in_and_leaves_the_return_alone(store):
    """+5k on 50k is +10%. Add 50k: still +10% managed, +5% on what you put in."""
    store.upsert_nav(_nav(_day(30), 50_000))
    store.upsert_nav(_nav(TODAY, 55_000))
    before = compute_stats(store.get_nav_history(), 50_000, store.get_cash_flows())
    assert before["twr_pct"] == pytest.approx(10.0)
    assert before["gain_pct"] == pytest.approx(10.0)

    store.record_cash_flow(50_000, note="top-up")

    after = compute_stats(store.get_nav_history(), 50_000, store.get_cash_flows())
    assert after["nav_current"] == pytest.approx(105_000)
    assert after["twr_pct"] == pytest.approx(10.0)
    assert after["alpha_pct"] == pytest.approx(10.0)
    assert after["gain_sek"] == pytest.approx(5_000)
    assert after["gain_pct"] == pytest.approx(5.0)
    assert after["invested_sek"] == pytest.approx(100_000)


def test_returns_after_the_deposit_compound_on_the_larger_book():
    """A later +10% on 105k is real money: the index takes it, nothing else."""
    nav = [_nav(_day(3), 50_000), _nav(_day(2), 55_000),
           _nav(_day(1), 105_000), _nav(TODAY, 105_000 * 1.10)]
    flows = [{"date": _day(1), "amount_sek": 50_000}]
    assert return_index(nav, flows) == pytest.approx([1.0, 1.10, 1.10, 1.10 * 1.10])
    assert gain(105_000 * 1.10, 50_000, flows)[0] == pytest.approx(115_500 - 100_000)


# ── The ledger ────────────────────────────────────────────────────────────────

def test_a_deposit_after_todays_point_lands_in_that_point(store):
    """The point on a flow's date must include it, whichever job wrote it first."""
    store.upsert_nav(_nav(TODAY, 55_000))
    cash = store.record_cash_flow(50_000)
    assert cash == pytest.approx(100_000)
    assert store.get_cash() == pytest.approx(100_000)
    point = store.get_nav_history()[-1]
    assert point.portfolio_nav_sek == pytest.approx(105_000)


def test_a_withdrawal_is_not_a_loss_or_a_drawdown(store):
    store.upsert_nav(_nav(_day(1), 55_000))
    store.upsert_nav(_nav(TODAY, 55_000))
    store.record_cash_flow(-20_000)
    stats = compute_stats(store.get_nav_history(), 50_000, store.get_cash_flows())
    assert stats["twr_pct"] == pytest.approx(0.0)
    assert stats["max_drawdown_pct"] == pytest.approx(0.0)
    assert stats["invested_sek"] == pytest.approx(30_000)


@pytest.mark.parametrize("amount", [0.0, float("nan"), float("inf")])
def test_a_meaningless_amount_is_refused(store, amount):
    with pytest.raises(ValueError):
        store.record_cash_flow(amount)
    assert store.get_cash_flows() == []


def test_you_cannot_withdraw_more_than_the_cash(store):
    with pytest.raises(ValueError, match="exceeds cash"):
        store.record_cash_flow(-60_000)
    assert store.get_cash() == pytest.approx(50_000)
    assert store.get_cash_flows() == []


def test_a_backfill_logs_the_flow_and_touches_nothing_else(store):
    """For a top-up already booked with set-cash: cash and NAV already show it."""
    store.upsert_nav(_nav(_day(5), 100_000))
    store.record_cash_flow(50_000, flow_date=_day(5), adjust_cash=False)
    assert store.get_cash() == pytest.approx(50_000)
    assert store.get_nav_history()[0].portfolio_nav_sek == pytest.approx(100_000)
    assert store.get_cash_flows()[0]["date"] == _day(5)


def test_a_flow_that_moves_cash_cannot_be_backdated(store):
    with pytest.raises(ValueError, match="dated today"):
        store.record_cash_flow(1_000, flow_date=_day(3))


def test_reset_clears_the_flows_with_the_rest_of_the_book(store):
    store.record_cash_flow(1_000)
    store.reset(50_000)
    assert store.get_cash_flows() == []


# ── Everything that reads a return ────────────────────────────────────────────

def test_the_chart_draws_no_jump_on_deposit_day(store):
    import json
    store.upsert_nav(_nav(_day(1), 55_000))
    store.upsert_nav(_nav(TODAY, 55_000))
    store.record_cash_flow(50_000)
    chart = json.loads(nav_chart_json(store.get_nav_history(), "OMXSPI", store.get_cash_flows()))
    assert chart["data"][0]["y"] == [100.0, 100.0]


def test_the_run_score_nets_out_a_deposit_in_its_week(store):
    """Otherwise a top-up week is the best decision the fund ever made."""
    import json
    from fundmgr.state.models import RecommendationLog
    start, end = _day(10), _day(3)
    store.save_recommendation(RecommendationLog(
        run_id="r1", timestamp=datetime.strptime(start, "%Y-%m-%d"), prompt_snapshot="{}",
        llm_response="{}", guardrail_log=json.dumps({}), actions_json="[]"))
    store.upsert_nav(_nav(start, 50_000, bench=100.0))
    store.upsert_nav(_nav(end, 100_000, bench=100.0))
    store.record_cash_flow(50_000, flow_date=end, adjust_cash=False)
    scored = store.score_runs(min_days=7)
    assert scored and scored[0]["score"] == pytest.approx(0.0)


# ── The commands ──────────────────────────────────────────────────────────────

@pytest.fixture
def cli_store(store, monkeypatch):
    cfg = AppConfig(capital_sek=50_000, db_path=store.db_path)
    monkeypatch.setattr(commands, "_get_store", lambda cfg_=None: (cfg, store))
    return store


def test_deposit_adds_to_cash_rather_than_overwriting_it(cli_store):
    result = CliRunner().invoke(commands.cli, ["deposit", "50000", "--note", "top-up"])
    assert result.exit_code == 0, result.output
    assert cli_store.get_cash() == pytest.approx(100_000)
    assert "Money put in: 100,000 SEK" in result.output


def test_withdraw_past_the_cash_is_refused_with_the_reason(cli_store):
    result = CliRunner().invoke(commands.cli, ["withdraw", "60000"])
    assert result.exit_code != 0 and "exceeds cash" in result.output
    assert cli_store.get_cash() == pytest.approx(50_000)


@pytest.mark.parametrize("args", [["deposit", "100", "--record-only"],
                                  ["deposit", "100", "--date", "2026-01-01"],
                                  ["deposit", "0"], ["deposit", "-5"]])
def test_bad_deposit_arguments_change_nothing(cli_store, args):
    result = CliRunner().invoke(commands.cli, args)
    assert result.exit_code != 0
    assert cli_store.get_cash() == pytest.approx(50_000)
    assert cli_store.get_cash_flows() == []


def test_set_cash_upwards_points_at_deposit(cli_store):
    result = CliRunner().invoke(commands.cli, ["set-cash", "60000"])
    assert result.exit_code == 0
    assert "fund deposit 10000.00" in result.output
