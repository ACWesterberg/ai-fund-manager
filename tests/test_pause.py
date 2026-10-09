"""A paused fund keeps its book and its valuation, and spends nothing.

The four simulations were paused to save API cost while the real Nordic book
is the focus. Cron still fires every job for them, so the pause has to hold in
the commands themselves.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from fundmgr import cli as commands
from fundmgr.config import AppConfig, load_config
from fundmgr.state.store import Store

CONFIG = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture
def paused(tmp_path, monkeypatch):
    cfg = AppConfig(name="Test SIM", paused=True, db_path=tmp_path / "fund.db",
                    universe_path=CONFIG / "universe_global.csv", benchmark="URTH")
    store = Store(cfg.db_path)
    store.initialise(100_000)
    store.upsert_position("AAPL", 10, 150.0)
    monkeypatch.setattr(commands, "_get_store", lambda cfg_=None: (cfg, store))
    return cfg, store


def test_the_simulations_are_paused_and_the_real_book_is_not():
    assert load_config(CONFIG / "config.yaml").paused is False
    for name in ("config_global", "config_claude", "config_buffett_gpt", "config_buffett_claude"):
        assert load_config(CONFIG / f"{name}.yaml").paused is True, name


@pytest.mark.parametrize("args", [
    ["run"], ["run", "--force"], ["optimize"], ["check-news"],
    ["review-stop"], ["review-target"],
])
def test_paid_commands_stop_before_doing_anything(paused, monkeypatch, args):
    """--force overrides the holiday gate, not the pause."""
    monkeypatch.setattr(commands, "fetch_and_cache_prices",
                        lambda *a, **k: pytest.fail("fetched prices for a paused fund"))
    monkeypatch.setattr(commands, "_skip_on_market_holiday",
                        lambda *a, **k: pytest.fail("reached the holiday gate"))
    result = CliRunner().invoke(commands.cli, args)
    assert result.exit_code == 0, result.output
    assert "is paused" in result.output


def test_check_stops_values_the_book_and_trades_nothing(paused, monkeypatch):
    cfg, store = paused
    monkeypatch.setattr("fundmgr.data.quotes.live_prices", lambda ts: {t: 200.0 for t in ts})
    monkeypatch.setattr(commands, "fetch_and_cache_benchmark", lambda *a, **k: True)
    # Far past any stop or target: a live fund would sell here.
    store.set_position_stop("AAPL", stop_pct=5.0, take_profit_pct=10.0)
    monkeypatch.setattr("fundmgr.engine.auto_fill.execute_paper_fills",
                        lambda *a, **k: pytest.fail("traded a paused fund"))

    result = CliRunner().invoke(commands.cli, ["check-stops"])

    assert result.exit_code == 0, result.output
    nav = store.get_nav_history()
    assert len(nav) == 1 and nav[0].portfolio_nav_sek == pytest.approx(100_000 + 10 * 200.0)
    assert store.get_positions()[0].shares == 10


def test_a_missing_price_marks_nothing_rather_than_a_loss(paused, monkeypatch):
    _, store = paused
    monkeypatch.setattr("fundmgr.data.quotes.live_prices", lambda ts: {t: None for t in ts})
    monkeypatch.setattr(commands, "fetch_and_cache_benchmark", lambda *a, **k: True)

    result = CliRunner().invoke(commands.cli, ["mark"])

    assert result.exit_code == 1
    assert "no price for AAPL" in result.output
    assert store.get_nav_history() == []
