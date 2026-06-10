import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from fundmgr.state.models import Transaction
from fundmgr.state.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "test.db")


def test_initialise(store):
    store.initialise(50_000)
    assert store.get_cash() == pytest.approx(50_000)


def test_initialise_twice_raises(store):
    store.initialise(50_000)
    with pytest.raises(RuntimeError):
        store.initialise(50_000)


def test_buy_updates_position_and_cash(store):
    store.initialise(50_000)
    txn = Transaction(
        ticker="VOLV-B.ST",
        side="buy",
        shares=10,
        price_sek=300.0,
        fee_sek=3.0,
        source="fill",
        timestamp=datetime.utcnow(),
    )
    store.apply_fill(txn)

    positions = store.get_positions()
    assert len(positions) == 1
    assert positions[0].ticker == "VOLV-B.ST"
    assert positions[0].shares == pytest.approx(10)
    assert positions[0].avg_cost_sek == pytest.approx(300.0)

    # Cash should be 50000 - (10 * 300) - 3 = 46997
    assert store.get_cash() == pytest.approx(46_997.0)


def test_sell_updates_position_and_cash(store):
    store.initialise(50_000)
    buy = Transaction(
        ticker="VOLV-B.ST", side="buy", shares=10,
        price_sek=300.0, fee_sek=3.0, source="fill", timestamp=datetime.utcnow()
    )
    store.apply_fill(buy)

    sell = Transaction(
        ticker="VOLV-B.ST", side="sell", shares=5,
        price_sek=320.0, fee_sek=1.60, source="fill", timestamp=datetime.utcnow()
    )
    store.apply_fill(sell)

    positions = store.get_positions()
    assert positions[0].shares == pytest.approx(5)

    # Cash after buy: 46997
    # Cash after sell: 46997 + (5 * 320) - 1.60 = 46997 + 1600 - 1.60 = 48595.40
    assert store.get_cash() == pytest.approx(48_595.40)


def test_avg_cost_weighted(store):
    store.initialise(50_000)
    buy1 = Transaction(
        ticker="SAND.ST", side="buy", shares=10,
        price_sek=200.0, fee_sek=2.0, source="fill", timestamp=datetime.utcnow()
    )
    buy2 = Transaction(
        ticker="SAND.ST", side="buy", shares=10,
        price_sek=220.0, fee_sek=2.2, source="fill", timestamp=datetime.utcnow()
    )
    store.apply_fill(buy1)
    store.apply_fill(buy2)

    positions = store.get_positions()
    assert positions[0].shares == pytest.approx(20)
    assert positions[0].avg_cost_sek == pytest.approx(210.0)  # (10*200 + 10*220) / 20


def test_total_fees(store):
    store.initialise(50_000)
    store.apply_fill(Transaction(
        ticker="VOLV-B.ST", side="buy", shares=10,
        price_sek=300.0, fee_sek=3.0, source="fill", timestamp=datetime.utcnow()
    ))
    store.apply_fill(Transaction(
        ticker="SAND.ST", side="buy", shares=5,
        price_sek=200.0, fee_sek=1.0, source="fill", timestamp=datetime.utcnow()
    ))
    assert store.total_fees_paid() == pytest.approx(4.0)
