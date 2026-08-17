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


# ── Learnings retention ───────────────────────────────────────────────────────

def _lrn(store, category, body, created):
    from fundmgr.state.models import Learning
    store.save_learning(Learning(category=category, body=body,
                                 created_at=datetime.fromisoformat(created)))


def test_prune_learnings_by_category(store):
    _lrn(store, "qualitative", "Anecdote.", "2026-08-17T10:00:00")
    _lrn(store, "calibration", "Hit rate 40%.", "2026-08-17T09:00:00")

    assert len(store.find_active_learnings(category="qualitative")) == 1
    assert store.deactivate_learnings(category="qualitative") == 1

    remaining = store.get_active_learnings()
    assert [lrn.category for lrn in remaining] == ["calibration"]


def test_prune_learnings_before_date(store):
    _lrn(store, "qualitative", "Old.", "2026-08-10T10:00:00")
    _lrn(store, "qualitative", "New.", "2026-08-17T10:00:00")

    # A bare date compares as an ISO prefix: 08-17 itself is not "before 08-17".
    assert store.deactivate_learnings(before="2026-08-17") == 1
    assert [lrn.body for lrn in store.get_active_learnings()] == ["New."]


def test_prune_learnings_retires_rather_than_deletes(store):
    _lrn(store, "qualitative", "Anecdote.", "2026-08-17T10:00:00")
    store.deactivate_learnings()
    assert store.get_active_learnings() == []
    # Still on disk, just inactive — past runs stay reconstructible.
    with store._conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM learnings").fetchone()[0] == 1
