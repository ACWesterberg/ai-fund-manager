from datetime import datetime

import pytest
from click.testing import CliRunner

from fundmgr.cli import _parse_holdings_snapshot, cli
from fundmgr.state.models import Transaction
from fundmgr.state.store import Store


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A configured fund with two seeded positions, wired to the CLI via FUND_CONFIG."""
    db = tmp_path / "fund.db"
    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(f"capital_sek: 50000\ndb_path: {db}\nbenchmark: OMXSPI\n")
    monkeypatch.setenv("FUND_CONFIG", str(cfg_yaml))
    store = Store(db)
    store.set_cash(50_000)
    store.apply_fill(Transaction("VOLV-B.ST", "buy", 100, 250, 5, "fill", datetime(2026, 6, 1, 12)))
    store.apply_fill(Transaction("SAND.ST", "buy", 50, 200, 4, "fill", datetime(2026, 6, 1, 12)))
    return store, tmp_path


# ── parser ────────────────────────────────────────────────────────────────────

def test_parse_snapshot(tmp_path):
    f = tmp_path / "b.txt"
    f.write_text("# statement\nVOLV-B.ST 200\nLIME.ST 30 195.0\nCASH 15,300.50\n\n")
    holdings, cash = _parse_holdings_snapshot(str(f))
    assert holdings == {"VOLV-B.ST": (200.0, None), "LIME.ST": (30.0, 195.0)}
    assert cash == pytest.approx(15_300.50)


def test_parse_snapshot_bad_line(tmp_path):
    f = tmp_path / "b.txt"
    f.write_text("VOLV-B.ST notanumber\n")
    with pytest.raises(Exception):
        _parse_holdings_snapshot(str(f))


# ── reconcile command ───────────────────────────────────────────────────────────

def _snap(tmp_path, text):
    p = tmp_path / "broker.txt"
    p.write_text(text)
    return str(p)


def test_dry_run_reports_drift_without_writing(env):
    store, tmp_path = env
    snap = _snap(tmp_path, "VOLV-B.ST 200\nSAND.ST 50\nLIME.ST 30 195.0\nCASH 14000\n")
    res = CliRunner().invoke(cli, ["reconcile", "--holdings", snap])
    assert res.exit_code == 0
    assert "VOLV-B.ST" in res.output and "+100.00" in res.output
    assert "Dry run" in res.output
    # nothing written
    pos = {p.ticker: p.shares for p in store.get_positions()}
    assert pos == {"VOLV-B.ST": 100.0, "SAND.ST": 50.0}


def test_apply_syncs_positions_and_cash(env):
    store, tmp_path = env
    snap = _snap(tmp_path, "VOLV-B.ST 200\nSAND.ST 50\nLIME.ST 30 195.0\nCASH 15300.50\n")
    res = CliRunner().invoke(cli, ["reconcile", "--holdings", snap, "--apply", "--yes"])
    assert res.exit_code == 0
    pos = {p.ticker: p for p in store.get_positions()}
    assert pos["VOLV-B.ST"].shares == 200.0
    assert pos["VOLV-B.ST"].avg_cost_sek == 250.0  # cost basis preserved through split
    assert pos["LIME.ST"].shares == 30.0 and pos["LIME.ST"].avg_cost_sek == 195.0
    assert store.get_cash() == pytest.approx(15_300.50)


def test_in_sync_is_a_noop(env):
    store, tmp_path = env
    snap = _snap(tmp_path, "VOLV-B.ST 100\nSAND.ST 50\n")
    res = CliRunner().invoke(cli, ["reconcile", "--holdings", snap])
    assert "in sync" in res.output and "Book matches the broker" in res.output


def test_full_zeroes_missing_book_tickers(env):
    store, tmp_path = env
    snap = _snap(tmp_path, "VOLV-B.ST 100\n")  # SAND dropped
    res = CliRunner().invoke(cli, ["reconcile", "--holdings", snap, "--full", "--apply", "--yes"])
    assert res.exit_code == 0
    assert {p.ticker for p in store.get_positions()} == {"VOLV-B.ST"}


def test_without_full_missing_tickers_untouched(env):
    store, tmp_path = env
    snap = _snap(tmp_path, "VOLV-B.ST 100\n")  # SAND dropped, but no --full
    res = CliRunner().invoke(cli, ["reconcile", "--holdings", snap, "--apply", "--yes"])
    assert "left untouched" in res.output
    assert {p.ticker for p in store.get_positions()} == {"VOLV-B.ST", "SAND.ST"}


def test_new_ticker_without_cost_is_skipped(env):
    store, tmp_path = env
    snap = _snap(tmp_path, "VOLV-B.ST 100\nSAND.ST 50\nNEWCO.ST 10\n")
    res = CliRunner().invoke(cli, ["reconcile", "--holdings", snap, "--apply", "--yes"])
    assert "no AVG_COST" in res.output
    assert not any(p.ticker == "NEWCO.ST" for p in store.get_positions())


def test_cash_only_reconcile(env):
    store, tmp_path = env
    res = CliRunner().invoke(cli, ["reconcile", "--cash", "16000", "--apply", "--yes"])
    assert res.exit_code == 0
    assert store.get_cash() == pytest.approx(16_000.0)


def test_no_inputs_errors(env):
    res = CliRunner().invoke(cli, ["reconcile"])
    assert res.exit_code != 0
    assert "Nothing to reconcile" in res.output
