from __future__ import annotations

import pytest

from fundmgr.config import AppConfig
from fundmgr.data.prices import TickerFeatures
from fundmgr.engine.client import _aggregate_decisions
from fundmgr.engine.schema import Action, DecisionRun
from fundmgr.guardrails.rules import apply_guardrails
from fundmgr.state.models import PortfolioSnapshot, Position, Transaction
from fundmgr.state.store import Store


def buy(ticker="NEW", weight=8, estimate=8000, confidence=.8):
    return Action(ticker=ticker, side="buy", target_weight_pct=weight,
                  sek_estimate=estimate, confidence=confidence, thesis="A test thesis",
                  kill_criterion="Margin below 40%", add_criterion="Revenue grows",
                  target_price=150, max_weight_pct=20, tranche_pct=3,
                  next_earnings="2026-11-01")


def decision(actions):
    return DecisionRun(run_id="test", market_summary="Test", actions=actions, cash_target_pct=12)


def check(actions, snap, cfg):
    features = {t: TickerFeatures(t, t, 100, "2026-09-21", 0, sector="Industrials" if t == "OLD" else "Tech", country="SE")
                for t in {a.ticker for a in actions} | {p.ticker for p in snap.positions}}
    return apply_guardrails(decision(actions), snap, features, set(features), cfg)


@pytest.mark.parametrize("max_positions", [2, 10])
def test_batch_reserves_cash_and_position_slots(max_positions):
    cfg = AppConfig()
    cfg.risk.max_positions = max_positions
    snap = PortfolioSnapshot([Position("OLD", 800, 100, 100)], 20000)
    cfg.risk.max_sector_pct = 100
    result = check([buy("A", weight=7.5), buy("B", weight=7.5)], snap, cfg)
    assert len(result.approved_actions) == 1
    assert snap.cash_sek == 20000
    assert len(snap.positions) == 1
    assert result.verdicts[1].approved is False


@pytest.mark.parametrize("limit", ["sector", "region", "style"])
def test_batch_reserves_bucket_exposure(limit):
    cfg = AppConfig()
    cfg.risk.max_turnover_pct = 100
    if limit == "sector":
        cfg.risk.max_sector_pct = 15
    elif limit == "region":
        cfg.risk.region_targets = {"nordics": 15}
        cfg.risk.region_tolerance_pct = 0
    else:
        # Explicit quality features make both names belong to the same bucket.
        from fundmgr import styles
        from unittest.mock import patch
        cfg.risk.style_targets = {"quality": 15}
        cfg.risk.style_tolerance_pct = 0
        with patch.object(styles, "bucket_of", return_value="quality"):
            result = check([buy("A"), buy("B")], PortfolioSnapshot([], 100000), cfg)
        assert len(result.approved_actions) == 1
        return
    result = check([buy("A"), buy("B")], PortfolioSnapshot([], 100000), cfg)
    assert len(result.approved_actions) == 1


def test_amount_is_derived_from_target_and_existing_position():
    snap = PortfolioSnapshot([Position("NEW", 100, 100, 100)], 90000)
    result = check([buy(weight=18, estimate=2500)], snap, AppConfig())
    assert result.approved_actions[0].sek_estimate == 8000


def test_turnover_rejection_is_visible_in_audit():
    cfg = AppConfig()
    cfg.risk.max_turnover_pct = 10
    result = check([buy("A"), buy("B")], PortfolioSnapshot([], 100000), cfg)
    assert len(result.approved_actions) == 1
    assert result.to_log()[1]["status"] == "REJECTED"
    assert "turnover" in result.to_log()[1]["reason"]


def test_fees_cannot_take_cash_below_floor():
    snap = PortfolioSnapshot([Position("OLD", 800, 100, 100)], 20000)
    cfg = AppConfig()
    cfg.risk.max_sector_pct = 100
    # Exactly 8% principal would leave 12% cash, before the fee.
    assert not check([buy()], snap, cfg).approved_actions


def test_duplicate_actions_do_not_spend_twice():
    with pytest.raises(ValueError, match="at most one action per ticker"):
        decision([buy(), buy()])
    # Keep the guardrail's defensive check covered even when callers mutate an
    # already-validated decision (normal model output is rejected by the schema).
    run = decision([buy()])
    run.actions.append(buy())
    feature = TickerFeatures("NEW", "NEW", 100, "2026-09-21", 0, sector="Tech")
    result = apply_guardrails(run, PortfolioSnapshot([], 100000),
                              {"NEW": feature}, {"NEW"}, AppConfig())
    assert not result.approved_actions


def test_consensus_and_clipping_preserve_monitoring_plan():
    action = buy(weight=25)
    aggregate, _ = _aggregate_decisions([decision([action])] * 3)
    result = check(aggregate.actions, PortfolioSnapshot([], 100000), AppConfig())
    clipped = result.approved_actions[0]
    assert clipped.target_weight_pct == 18
    for field in ("kill_criterion", "add_criterion", "target_price", "max_weight_pct", "tranche_pct", "next_earnings"):
        assert getattr(clipped, field) == getattr(action, field)


@pytest.mark.parametrize("side,shares,price,fee", [
    ("sell", 11, 100, 0), ("buy", -1, 100, 0), ("buy", 1, -100, 0),
    ("buy", 1, 100, -1), ("buy", float("nan"), 100, 0),
    ("buy", 1, float("inf"), 0), ("other", 1, 100, 0),
    ("buy", 10, 100, 1),
])
def test_invalid_fills_are_atomic(tmp_path, side, shares, price, fee):
    store = Store(tmp_path / "book.db")
    store.initialise(1000)
    store.upsert_position("NEW", 10, 100)
    with pytest.raises(ValueError):
        store.apply_fill(Transaction("NEW", side, shares, price, fee, "fill"))
    assert store.get_cash() == 1000
    assert store.get_positions()[0].shares == 10
    assert store.get_transactions() == []


def test_sell_unheld_cannot_create_cash(tmp_path):
    store = Store(tmp_path / "book.db")
    store.initialise(1000)
    with pytest.raises(ValueError):
        store.apply_fill(Transaction("MISSING", "sell", 1, 100, 0, "fill"))
    assert store.get_cash() == 1000
    assert store.get_transactions() == []


@pytest.fixture
def filler(tmp_path, monkeypatch):
    from fundmgr.engine import auto_fill
    store = Store(tmp_path / "book.db")
    store.initialise(100000)
    monkeypatch.setattr("fundmgr.config.load_universe", lambda *a: [])
    monkeypatch.setattr("fundmgr.data.market_hours.is_exchange_open", lambda *a: True)
    monkeypatch.setattr(auto_fill, "_fetch_price", lambda t: 200)
    cfg = AppConfig(auto_fill=True, fx_to_sek=False)
    return auto_fill, store, cfg


def test_execution_never_exceeds_approved_budget(filler):
    auto_fill, store, cfg = filler
    auto_fill.execute_paper_fills([buy(weight=18, estimate=2500).model_dump()], store, cfg)
    assert store.get_transactions()[0].gross_sek <= 2500


def test_execution_respects_cash_if_earlier_sell_did_not_fill(filler):
    auto_fill, store, cfg = filler
    store.set_cash(20000)
    store.upsert_position("OLD", 400, 100)
    outcome = auto_fill.execute_paper_fills([buy(weight=18, estimate=18000).model_dump()], store, cfg)
    assert store.get_transactions() == []
    assert outcome.skipped == {"NEW": "risk_limit"}


def test_fills_preserve_unrealised_profit_in_nav(filler):
    auto_fill, store, cfg = filler
    store.set_cash(20000)
    store.upsert_position("OLD", 400, 100)
    auto_fill.execute_paper_fills([buy(weight=5, estimate=5000).model_dump()], store, cfg)
    assert store.get_nav_history()[0].portfolio_nav_sek == pytest.approx(99995)


def test_missing_valuation_does_not_overwrite_nav(filler, monkeypatch):
    from fundmgr.state.models import NavPoint
    auto_fill, store, cfg = filler
    store.upsert_position("OLD", 400, 100)
    store.upsert_nav(NavPoint("2026-09-21", 180000, 100, 100000))
    monkeypatch.setattr(auto_fill, "_fetch_price", lambda t: None if t == "OLD" else 200)
    auto_fill.execute_paper_fills([buy().model_dump()], store, cfg)
    assert store.get_transactions() == []
    assert store.get_nav_history()[0].portfolio_nav_sek == 180000


def test_universe_command_loads_the_selected_profile(monkeypatch, tmp_path):
    from click.testing import CliRunner
    import fundmgr.cli as commands

    path = tmp_path / "universe.csv"
    path.write_text("name,yahoo_ticker,isin,country,exchange,sector,enabled\nTest,TEST.ST,,SE,STO,Tech,true\n")
    monkeypatch.setattr(commands, "load_config", lambda: AppConfig(universe_path=path))
    result = CliRunner().invoke(commands.cli, ["universe"])
    assert result.exit_code == 0, result.output
    assert "TEST.ST" in result.output


def test_concurrent_sells_cannot_credit_the_same_holding_twice(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    store = Store(tmp_path / "book.db")
    store.initialise(1000)
    store.upsert_position("NEW", 10, 100)

    def sell():
        try:
            store.apply_fill(Transaction("NEW", "sell", 10, 100, 0, "fill"))
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: sell(), range(2)))
    assert sum(results) == 1
    assert store.get_cash() == 2000
    assert len(store.get_transactions()) == 1


def test_rejected_sell_cannot_fund_a_buy():
    cfg = AppConfig()
    cfg.risk.max_sector_pct = 100
    snap = PortfolioSnapshot([Position("OLD", 1000, 100, 100)], 0)
    sell = buy("OLD", weight=0, confidence=.9).model_copy(update={"side": "sell"})
    result = check([sell, buy()], snap, cfg)
    assert result.approved_actions == []
    assert "turnover" in result.verdicts[0].rejection_reason
    assert "cash floor" in result.verdicts[1].rejection_reason
