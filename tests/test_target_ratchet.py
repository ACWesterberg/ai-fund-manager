"""
Tests for take-profit ratcheting and price-alert rate limiting.

The bug these cover: TRUE-B.ST was bought with a +33% target, every run since
returned HOLD, and only buys persisted levels — so the target stayed frozen at
entry while check-stops re-fired TARGET HIT every 15 minutes, telling the user
to trim a position the consensus had just decided to hold.

Run: pytest tests/test_target_ratchet.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

# fundmgr.engine.prompt imports the shared financedata package transitively, via
# the price layer. _portfolio_block is pure string formatting and touches none
# of it, so when that sibling package isn't installed (CI containers, a fresh
# clone without ../FinanceData) stand in a stub rather than skip the test. Where
# the real package exists — the Pi, a full dev checkout — it is imported
# normally and this branch never runs.
try:  # pragma: no cover - depends on the environment, not the code under test
    import financedata  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    import types
    _stub = types.ModuleType("financedata")
    for _name in (
        "get_prices_since", "rsi", "pct_return", "ann_vol", "get_cache",
        "get_fundamentals", "ts_to_days",
    ):
        setattr(_stub, _name, lambda *a, **k: None)
    sys.modules["financedata"] = _stub

from fundmgr.levels import (  # noqa: E402
    SKIP_AT_TARGET, SKIP_BELOW_MIN, SKIP_MARKET_CLOSED, SKIP_NOT_HELD, SKIP_NO_PRICE,
    alertable_hits, deferred_note, merged_levels, record_sent_alerts, settled_sells,
)
from fundmgr.engine.prompt import _portfolio_block  # noqa: E402
from fundmgr.state.store import Store  # noqa: E402


# ── Fakes ─────────────────────────────────────────────────────────────────────

class _Action:
    """Minimal stand-in for engine.schema.Action."""
    def __init__(self, side, ticker="TRUE-B.ST", stop=None, tp=None, target_weight_pct=5.0):
        self.side = side
        self.ticker = ticker
        self.stop_loss_pct = stop
        self.take_profit_pct = tp
        self.target_weight_pct = target_weight_pct


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "test.db"))


# ── Ratcheting ────────────────────────────────────────────────────────────────

def test_hold_raises_a_stale_target():
    """The actual bug: a hold past target must be able to move the target up."""
    prior = {"stop_pct": 12.0, "take_profit_pct": 33.0}
    action = _Action("hold", tp=50.0)
    assert merged_levels(action, prior) == (12.0, 50.0)


def test_hold_without_a_target_leaves_stored_levels_alone():
    """take_profit_pct is optional — an omitted field must not erase the level."""
    prior = {"stop_pct": 12.0, "take_profit_pct": 33.0}
    assert merged_levels(_Action("hold", tp=None, stop=None), prior) is None


def test_hold_restating_only_the_stop_keeps_the_target():
    prior = {"stop_pct": 12.0, "take_profit_pct": 33.0}
    assert merged_levels(_Action("hold", stop=15.0), prior) == (15.0, 33.0)


def test_buy_still_sets_levels_from_scratch():
    assert merged_levels(_Action("buy", stop=10.0, tp=30.0), None) == (10.0, 30.0)


def test_sell_never_writes_levels():
    """Sells fall through to the clear-on-exit branch instead."""
    assert merged_levels(_Action("sell", stop=10.0, tp=30.0), {}) is None


# ── Alert rate limiting ───────────────────────────────────────────────────────

def _send(hits, kind, store, day, auto_sold=()):
    """One check-stops cycle that reaches Telegram: filter, then record."""
    out = alertable_hits(hits, kind, list(auto_sold), store, day)
    record_sent_alerts(out, kind, store, day)
    return out


def test_target_alert_fires_once_per_day(store):
    hits = [("TRUE-B.ST", 35.5, 33.0, 20.81)]
    first  = _send(hits, "target", store, "2026-08-14")
    second = _send(hits, "target", store, "2026-08-14")
    third  = _send(hits, "target", store, "2026-08-14")
    assert len(first) == 1, "first breach of the day must alert"
    assert second == [] and third == [], "repeat cycles must stay silent"


def test_alert_rearms_the_next_day(store):
    hits = [("TRUE-B.ST", 35.5, 33.0, 20.81)]
    assert _send(hits, "target", store, "2026-08-14")
    assert _send(hits, "target", store, "2026-08-15")


def test_a_send_that_never_happened_does_not_burn_the_day(store):
    """
    alertable_hits is a pure query — an unconfigured bot or a failed send skips
    record_sent_alerts, so the next cycle tries again instead of going quiet.
    """
    hits = [("TRUE-B.ST", 35.5, 33.0, 20.81)]
    assert alertable_hits(hits, "target", [], store, "2026-08-14")   # send fails here
    assert _send(hits, "target", store, "2026-08-14"), "must retry after a failed send"


# ── What actually sold ────────────────────────────────────────────────────────
#
# SAP.DE hit its +27% target and check-stops reported "AUTO-SOLD" every 15
# minutes from 19:15 to the close. XETRA shuts at 17:30 while this command runs
# on the NYSE window to 22:00, so the filler was refusing to trade a closed
# venue — correctly — and the caller counted every triggered ticker as sold
# anyway. An auto-sold ticker is exempt from the daily alert limit, so the
# false claim also repeated itself indefinitely.

_TRIGGERED = [("SAP.DE", 26.9, 27.0, 189.88)]


def test_a_skipped_fill_is_not_reported_as_sold():
    """The position is still there afterwards, so nothing was sold."""
    before = {"SAP.DE": 40.0}
    sold, deferred = settled_sells(_TRIGGERED, before, dict(before))
    assert sold == []
    assert deferred == ["SAP.DE"]


def test_a_real_sell_is_reported_as_sold():
    sold, deferred = settled_sells(_TRIGGERED, {"SAP.DE": 40.0}, {})
    assert sold == ["SAP.DE"]
    assert deferred == []


def test_a_partial_sell_still_counts_as_sold():
    """Shares left the book, so the trade happened and is worth reporting."""
    sold, _ = settled_sells(_TRIGGERED, {"SAP.DE": 40.0}, {"SAP.DE": 15.0})
    assert sold == ["SAP.DE"]


def test_a_deferred_sell_alerts_once_a_day_not_every_cycle(store):
    """The spam, end to end: with the ticker correctly absent from auto_sold,
    the daily limit applies and the 19:30 / 19:45 / 20:00 cycles stay quiet."""
    held = {"SAP.DE": 40.0}
    cycles = []
    for _ in range(4):
        sold, _deferred = settled_sells(_TRIGGERED, held, dict(held))   # venue closed
        cycles.append(_send(_TRIGGERED, "target", store, "2026-08-27", auto_sold=sold))
    assert len(cycles[0]) == 1, "the first breach of the day must still alert"
    assert cycles[1:] == [[], [], []], "later cycles must stay silent"


def test_a_genuine_auto_sell_still_bypasses_the_daily_limit(store):
    """The exemption is right — a completed trade is news, not a standing
    condition. It just has to be a trade that happened."""
    sold, _ = settled_sells(_TRIGGERED, {"SAP.DE": 40.0}, {})
    assert _send(_TRIGGERED, "target", store, "2026-08-27", auto_sold=sold)
    assert _send(_TRIGGERED, "target", store, "2026-08-27", auto_sold=sold)


def test_stop_and_target_alerts_are_independent_on_the_same_day(store):
    """
    Separate kinds must not suppress each other — the reason these live in
    their own table rather than daily_price_alerts, whose primary key is
    (ticker, alert_date) and could hold only one row per ticker per day.
    """
    assert _send([("AAA.ST", -12.0, 12.0, 50.0)], "stop", store, "2026-08-14")
    assert _send([("AAA.ST", 35.0, 33.0, 90.0)], "target", store, "2026-08-14")


def test_two_tickers_do_not_suppress_each_other(store):
    assert _send([("AAA.ST", 35.0, 33.0, 90.0)], "target", store, "2026-08-14")
    assert _send([("BBB.ST", 40.0, 33.0, 12.0)], "target", store, "2026-08-14")


def test_auto_sold_always_reports(store):
    """A completed sale is news every time, not a standing condition."""
    hits = [("TRUE-B.ST", 35.5, 33.0, 20.81)]
    assert _send(hits, "target", store, "2026-08-14", auto_sold=["TRUE-B.ST"])
    assert _send(hits, "target", store, "2026-08-14", auto_sold=["TRUE-B.ST"])


def test_store_alert_helpers_round_trip(store):
    assert not store.has_sent_position_alert("AAA.ST", "2026-08-14", "target")
    store.record_position_alert("AAA.ST", "2026-08-14", "target")
    assert store.has_sent_position_alert("AAA.ST", "2026-08-14", "target")
    assert not store.has_sent_position_alert("AAA.ST", "2026-08-14", "stop")
    assert not store.has_sent_position_alert("AAA.ST", "2026-08-15", "target")


def test_recording_the_same_alert_twice_is_harmless(store):
    store.record_position_alert("AAA.ST", "2026-08-14", "target")
    store.record_position_alert("AAA.ST", "2026-08-14", "target")
    assert store.has_sent_position_alert("AAA.ST", "2026-08-14", "target")


# ── Why a sell didn't settle ──────────────────────────────────────────────────
#
# PLT.OL hit its +30% target and the 15:00 alert read "market closed, sells on
# the next open". Oslo shuts at 16:20, so it had 80 minutes of session left —
# and the same line quoted the live price the check had just fetched. `deferred`
# only ever meant "shares didn't leave the book"; the caller picked one of the
# several reasons that can cause and asserted it for all of them.

def test_a_closed_venue_is_still_named_as_one():
    assert deferred_note(SKIP_MARKET_CLOSED) == "market closed, sells on the next open"


def test_a_missing_price_is_not_reported_as_a_closed_market():
    """The bug, at the wording layer: the one cause that wasn't PLT.OL's."""
    note = deferred_note(SKIP_NO_PRICE)
    assert "closed" not in note and "next open" not in note


def test_an_unrecorded_reason_claims_no_cause():
    """Nothing reached the caller, so the note names the consequence alone —
    including for a SKIP_* the wording hasn't caught up with yet."""
    for reason in (None, "", "a_reason_added_later"):
        note = deferred_note(reason)
        assert "closed" not in note, reason
        assert "level stands" in note, reason


def test_each_reason_says_something_of_its_own():
    known = (SKIP_MARKET_CLOSED, SKIP_NO_PRICE, SKIP_NOT_HELD, SKIP_BELOW_MIN, SKIP_AT_TARGET)
    notes = [deferred_note(r) for r in known]
    assert len(set(notes)) == len(known)
    assert deferred_note(None) not in notes


# ── What the filler reports back ──────────────────────────────────────────────
#
# The reason has to come from the one thing that knows it. These pin the codes
# the filler hands the caller, because that mapping is the whole fix.

def _universe(tmp_path: Path) -> Path:
    path = tmp_path / "universe.csv"
    path.write_text(
        "name,yahoo_ticker,isin,country,exchange,sector,enabled\n"
        "Polight ASA,PLT.OL,NO0012535832,NO,OSLO,Technology,true\n"
    )
    return path


def _auto_sell(store, tmp_path, monkeypatch, *, price, venue_open):
    """One check-stops auto-sell of a held PLT.OL, venue and price stubbed."""
    from fundmgr.config import AppConfig
    from fundmgr.engine import auto_fill

    store.set_cash(10_000.0)
    store.upsert_position("PLT.OL", 100.0, 10.0)
    monkeypatch.setattr(
        "fundmgr.data.market_hours.is_exchange_open", lambda *_a, **_k: venue_open
    )
    monkeypatch.setattr(auto_fill, "_fetch_price", lambda _t: price)
    return auto_fill.execute_paper_fills(
        [{"ticker": "PLT.OL", "side": "sell", "target_weight_pct": 0}],
        store,
        AppConfig(universe_path=_universe(tmp_path), auto_fill=True),
        notify_skips=False,
    )


def test_a_sell_with_no_price_is_reported_as_no_price(store, tmp_path, monkeypatch):
    """PLT.OL's cycle: Oslo open, the fill still didn't happen."""
    outcome = _auto_sell(store, tmp_path, monkeypatch, price=None, venue_open=True)
    assert outcome.skipped == {"PLT.OL": SKIP_NO_PRICE}
    assert "closed" not in deferred_note(outcome.skipped["PLT.OL"])
    assert store.get_positions(), "nothing sold, so the position stands"


def test_a_sell_into_a_shut_venue_is_reported_as_market_closed(store, tmp_path, monkeypatch):
    """The other half: when it really is the market, say so."""
    outcome = _auto_sell(store, tmp_path, monkeypatch, price=14.24, venue_open=False)
    assert outcome.skipped == {"PLT.OL": SKIP_MARKET_CLOSED}


def test_a_sell_that_fills_reports_no_skip_at_all(store, tmp_path, monkeypatch):
    outcome = _auto_sell(store, tmp_path, monkeypatch, price=14.24, venue_open=True)
    assert outcome.skipped == {}
    assert store.get_positions() == []


def test_a_sell_of_something_unheld_is_not_blamed_on_the_market(store, tmp_path, monkeypatch):
    from fundmgr.config import AppConfig
    from fundmgr.engine import auto_fill

    monkeypatch.setattr("fundmgr.data.market_hours.is_exchange_open", lambda *_a, **_k: True)
    monkeypatch.setattr(auto_fill, "_fetch_price", lambda _t: 14.24)
    outcome = auto_fill.execute_paper_fills(
        [{"ticker": "PLT.OL", "side": "sell", "target_weight_pct": 0}],
        store,
        AppConfig(universe_path=_universe(tmp_path), auto_fill=True),
        notify_skips=False,
    )
    assert outcome.skipped == {"PLT.OL": SKIP_NOT_HELD}


# ── Prompt visibility ─────────────────────────────────────────────────────────

class _Pos:
    def __init__(self, ticker, pnl):
        self.ticker = ticker
        self.shares = 100
        self.avg_cost_sek = 15.0
        self.current_price_sek = 20.81
        self.market_value_sek = 2081.0
        self.unrealised_pnl_pct = pnl


class _Snap:
    def __init__(self, positions):
        self.nav_sek = 51377.0
        self.cash_sek = 3500.0
        self.cash_pct = 6.8
        self.positions = positions

    def weight_pct(self, ticker):
        return 4.1


def test_portfolio_block_flags_a_position_past_its_target():
    """The model cannot re-target a level it was never shown."""
    snap = _Snap([_Pos("TRUE-B.ST", 35.5)])
    block = _portfolio_block(snap, 2.6, {"TRUE-B.ST": {"stop_pct": 12.0, "take_profit_pct": 33.0}})
    assert "target +33%" in block
    assert "PAST TARGET" in block
    assert "stop -12%" in block


def test_portfolio_block_does_not_flag_a_position_below_target():
    snap = _Snap([_Pos("INWI.ST", 8.0)])
    block = _portfolio_block(snap, 2.6, {"INWI.ST": {"stop_pct": 12.0, "take_profit_pct": 33.0}})
    assert "target +33%" in block
    assert "PAST TARGET" not in block


def test_portfolio_block_without_levels_is_unchanged():
    """Positions with no stored levels render exactly as before."""
    snap = _Snap([_Pos("OET.OL", 5.0)])
    assert "[" not in _portfolio_block(snap, 2.6, {})
    assert "[" not in _portfolio_block(snap, 2.6, None)
