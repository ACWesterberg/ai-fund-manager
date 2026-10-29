"""
Montrose phase 2: book broker rows into the ledger, one confirmed row at a time.

Reuses the phase-1 fixtures: the same account, transactions and fake MCP.
"""
from __future__ import annotations

import json
from datetime import date, datetime

import pytest
from click.testing import CliRunner

from fundmgr.broker import importer as imp
from fundmgr.broker import reconcile as rec
from fundmgr.config import FeeConfig
from fundmgr.state.models import Transaction
from fundmgr.state.store import Store

from tests.test_montrose_reconcile import (  # noqa: F401  (fixtures)
    INSTRUMENTS, UNIVERSE, _ledger, _ledger_trade, _run, _snapshot, _tx, fake,
)

FEES = FeeConfig().calc


# ── Fee split ────────────────────────────────────────────────────────────────

def test_a_sek_trade_splits_its_real_fee_and_moves_cash_by_the_broker_amount():
    price, fee, est = imp.split_fee("buy", 98, 81.5, "SEK", -7_995.0, FEES)
    assert (price, fee, est) == (81.5, 8.0, False)
    price, fee, est = imp.split_fee("sell", 13, 332.4, "SEK", 4_317.0, FEES)
    assert (price, fee, est) == (332.4, 4.2, False)


def test_a_foreign_trade_estimates_the_fee_and_keeps_the_cash_effect_exact():
    price, fee, est = imp.split_fee("sell", 36, 112.4, "NOK", 4_319.79, FEES)
    assert est and fee == pytest.approx(4.32)
    assert 36 * price - fee == pytest.approx(4_319.79)
    price, fee, est = imp.split_fee("buy", 9, 220.0, "DKK", -2_996.71, FEES)
    assert 9 * price + fee == pytest.approx(2_996.71)


def test_an_implausible_sek_fee_falls_back_to_the_estimate():
    # A price that doesn't explain the amount (e.g. a corrected trade) is not
    # trusted to give the fee; the cash effect still matches the broker.
    price, fee, est = imp.split_fee("buy", 10, 100.0, "SEK", -1_500.0, FEES)
    assert est and 10 * price + fee == pytest.approx(1_500.0)


# ── Proposals ────────────────────────────────────────────────────────────────

def _without(*tickers):
    trades = [t for t in _ledger()["ledger_trades"] if t.ticker not in tickers]
    return trades


def test_rows_the_ledger_lacks_become_proposals_in_bookable_order():
    report = _run(ledger_trades=_without("INWI.ST", "BONEX.ST"),
                  ledger_positions={"TRUE-B.ST": 100, "KCC.OL": 35})
    props = imp.proposals(report, import_from=date(2026, 10, 5), imports={}, fee_calc=FEES)
    got = [(p.broker_id, p.kind, p.category) for p in props]
    # The 10-01 deposit and the 10-02 buys predate import_from: never offered.
    assert got == [(4, "fill", "Buy"), (5, "fill", "Buy"), (6, "fill", "Sell"),
                   (7, "cash", "Dividend"), (8, "cash", "ForeignTax")]
    inwi = props[0]
    assert inwi.bookable and inwi.ticker == "INWI.ST" and inwi.side == "buy"
    assert inwi.shares * inwi.price_sek + inwi.fee_sek == pytest.approx(7_287.28)
    # Decided rows are not offered again.
    later = imp.proposals(report, import_from=date(2026, 10, 5),
                          imports={4: {"status": "skipped"}, 7: {"status": "booked"}}, fee_calc=FEES)
    assert [p.broker_id for p in later] == [5, 6, 8]


def test_same_day_sells_and_money_in_come_before_buys():
    txns = [_tx(20, "Buy", "2026-10-10", 610, 5, 180.0, "SEK", -900.9),
            _tx(21, "Sell", "2026-10-10", 1757, -10, 26.0, "SEK", 259.74),
            _tx(22, "Deposit", "2026-10-10", amount=1_000.0)]
    report = _run(_snapshot(txns=txns), ledger_trades=[], ledger_flows=[],
                  ledger_positions={"TRUE-B.ST": 100, "KCC.OL": 35, "INWI.ST": 40})
    props = imp.proposals(report, import_from=date(2026, 10, 1), imports={}, fee_calc=FEES)
    assert [p.broker_id for p in props] == [22, 21, 20]


def test_a_differing_hand_booked_fill_blocks_the_proposal():
    # The ledger booked INWI as 41 shares; Montrose says 40. Booking Montrose's
    # row as well would leave 81 — so it is named, not offered with a button.
    trades = _without("INWI.ST") + [_ledger_trade("2026-10-05", "INWI.ST", "buy", 41, -7_469.0)]
    report = _run(ledger_trades=trades, ledger_positions={"TRUE-B.ST": 100, "KCC.OL": 35, "INWI.ST": 41})
    props = imp.proposals(report, import_from=date(2026, 10, 5), imports={}, fee_calc=FEES)
    inwi = next(p for p in props if p.broker_id == 4)
    assert not inwi.bookable and "41" in inwi.problem
    with pytest.raises(ValueError):
        imp.book(None, inwi)


def test_unresolved_foreign_settled_and_corporate_actions_are_not_bookable():
    txns = [_tx(30, "Buy", "2026-10-10", 999, 3, 10.0, "SEK", -31.0),
            {**_tx(31, "Buy", "2026-10-10", 610, 1, 180.0, "SEK", -18.5), "currency": "EUR"},
            _tx(32, "CorporateAction", "2026-10-10", 610, amount=12.0, text="Split 2:1")]
    report = _run(_snapshot(txns=txns), ledger_trades=[], ledger_flows=[])
    props = {p.broker_id: p for p in imp.proposals(report, import_from=date(2026, 10, 1),
                                                    imports={}, fee_calc=FEES)}
    assert "montrose-map 999" in props[30].problem
    assert "EUR" in props[31].problem
    assert props[32].kind == "manual" and not props[32].bookable


def test_a_booked_dividend_stops_being_listed_by_the_reconcile():
    report = rec.reconcile(_snapshot(), **_ledger(), universe=UNIVERSE, instruments=INSTRUMENTS,
                           booked_ids={7})
    assert [e["category"] for e in report.cash_events] == ["ForeignTax"]
    assert report.explained_cash == pytest.approx(3.0 - 12.5)


# ── Store ────────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "fund.db")
    s.set_cash(10_000.0)
    return s


def _fill(ticker="INWI.ST", shares=40, price=182.0, fee=7.28, side="buy"):
    return Transaction(ticker, side, shares, price, fee, "montrose", datetime(2026, 10, 5, 12))


def test_a_broker_fill_books_once(store):
    store.book_broker_fill(4, _fill(), "Buy", "2026-10-05", -7_287.28)
    with pytest.raises(ValueError, match="already decided"):
        store.book_broker_fill(4, _fill(), "Buy", "2026-10-05", -7_287.28)
    assert [p.shares for p in store.get_positions()] == [40]
    assert store.get_cash() == pytest.approx(10_000 - 7_287.28)
    assert store.get_broker_imports()[4]["status"] == "booked"


def test_a_refused_fill_leaves_no_import_record(store):
    with pytest.raises(ValueError, match="Insufficient cash"):
        store.book_broker_fill(4, _fill(shares=400), "Buy", "2026-10-05", -72_807.28)
    assert store.get_broker_imports() == {} and store.get_positions() == []


def test_undoing_a_booked_fill_reopens_the_broker_row(store):
    store.book_broker_fill(4, _fill(), "Buy", "2026-10-05", -7_287.28)
    store.undo_last_fill()
    assert store.get_broker_imports() == {}
    assert store.get_cash() == pytest.approx(10_000.0)


def test_a_dividend_is_return_and_a_deposit_is_a_flow(store):
    store.book_broker_cash(7, "Dividend", "2026-10-09", 50.0, "KCCo")
    assert store.get_cash() == pytest.approx(10_050.0)
    assert store.get_cash_flows() == []
    store.book_broker_cash(1, "Deposit", "2026-10-01", 2_000.0, "Från 123", flow=True)
    flows = store.get_cash_flows()
    assert [f["amount_sek"] for f in flows] == [2_000.0]
    # The flow is dated the day the ledger's cash moves, not the broker's date.
    assert flows[0]["date"] == datetime.utcnow().strftime("%Y-%m-%d")
    with pytest.raises(ValueError, match="already decided"):
        store.book_broker_cash(7, "Dividend", "2026-10-09", 50.0)
    assert store.get_cash() == pytest.approx(12_050.0)


def test_a_skipped_row_is_recorded_without_touching_the_ledger(store):
    store.skip_broker_row(8, "ForeignTax", "2026-10-09", -12.5)
    assert store.get_cash() == pytest.approx(10_000.0)
    assert store.get_broker_imports()[8]["status"] == "skipped"


# ── CLI, end to end against the fake Montrose ────────────────────────────────

@pytest.fixture
def env(fake, tmp_path, monkeypatch):  # noqa: F811 (pytest fixture)
    from fundmgr import cli as commands
    from fundmgr.notify import send

    db = tmp_path / "fund.db"
    uni = tmp_path / "universe.csv"
    uni.write_text("name,yahoo_ticker,isin,country,exchange,sector,enabled\n"
                   + "".join(f"{u.name},{u.yahoo_ticker},,SE,{u.exchange},,true\n" for u in UNIVERSE))
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"db_path: {db}\nuniverse_path: {uni}\nmontrose:\n  account_name: Simulering\n"
                   "  import_from: '2026-10-05'\n")
    monkeypatch.setenv("FUND_CONFIG", str(cfg))
    store = Store(db)
    store.set_cash(30_000.0)
    for day, t, n, px, fee in [("2026-10-02", "TRUE-B.ST", 100, 25.0, 2.5),
                               ("2026-10-02", "KCC.OL", 35, 94.2857, 3.0)]:
        store.apply_fill(Transaction(t, "buy", n, px, fee, "fill",
                                     datetime.fromisoformat(day + "T12:00:00")))
    sent: list[tuple[str, object]] = []
    monkeypatch.setattr(send, "send_telegram",
                        lambda text, parse_mode="HTML", buttons=None: sent.append((text, buttons)) or True)
    monkeypatch.setattr(rec, "default_since", lambda days, today=None: date(2026, 9, 27))
    monkeypatch.setattr(commands, "_mark_after_ledger_change", lambda *a, **k: None)
    return store, sent


def _cli(*args):
    from fundmgr.cli import cli
    return CliRunner().invoke(cli, list(args))


def test_list_book_and_refuse_a_second_booking(env):
    store, _ = env
    res = _cli("montrose-import")
    assert res.exit_code == 0, res.output
    assert [ln.split()[0] for ln in res.output.splitlines()] == ["#4", "#5", "#6", "#7", "#8"]

    res = _cli("montrose-import", "--book", "4")
    assert res.exit_code == 0, res.output
    assert "INWI.ST" in res.output
    assert {p.ticker: p.shares for p in store.get_positions()}["INWI.ST"] == 40
    assert store.get_transactions(limit=1)[0].source == "montrose"

    res = _cli("montrose-import", "--book", "4")
    assert res.exit_code != 0 and "not waiting" in res.output
    assert {p.ticker: p.shares for p in store.get_positions()}["INWI.ST"] == 40


def test_a_fill_booked_by_hand_after_the_offer_is_not_doubled(env):
    store, _ = env
    store.apply_fill(Transaction("INWI.ST", "buy", 40, 182.0, 7.28, "fill", datetime(2026, 10, 5, 12)))
    res = _cli("montrose-import", "--book", "4")
    assert res.exit_code != 0 and "not waiting" in res.output
    assert {p.ticker: p.shares for p in store.get_positions()}["INWI.ST"] == 40


def test_the_whole_window_books_back_into_sync(env):
    store, _ = env
    for bid in ("5", "6", "4", "7", "8"):
        res = _cli("montrose-import", "--book", bid)
        assert res.exit_code == 0, res.output
    res = _cli("montrose-reconcile", "--offline", "--json")
    out = json.loads(res.output)
    assert out["proposals"] == []
    assert not out["missing_in_ledger"] and not out["cash_events"]
    assert all(p["status"] == "ok" for p in out["positions"])


def test_the_daily_job_offers_each_row_once_with_buttons(env):
    _, sent = env
    res = _cli("montrose-reconcile", "--notify")
    assert res.exit_code == 0, res.output
    offers = [(t, b) for t, b in sent if b]
    assert [b[0][0][1] for _, b in offers] == ["mi:book:4", "mi:book:5", "mi:book:6",
                                              "mi:book:7", "mi:book:8"]
    sent.clear()
    _cli("montrose-reconcile", "--notify")
    assert not any(b for _, b in sent)
    # /montrose_import sends them all again on demand.
    _cli("montrose-import", "--notify")
    assert sum(1 for _, b in sent if b) == 5


def test_skip_is_remembered(env):
    store, _ = env
    assert _cli("montrose-import", "--skip", "8").exit_code == 0
    res = _cli("montrose-import")
    assert "#8" not in res.output
    assert store.get_broker_imports()[8]["status"] == "skipped"


def test_booking_is_off_without_import_from(env, tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(cfg.read_text().replace("  import_from: '2026-10-05'\n", ""))
    res = _cli("montrose-import")
    assert res.exit_code != 0 and "off" in res.output
