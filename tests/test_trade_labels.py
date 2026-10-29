"""A buy says whether it adds to a holding or opens one, and how much to trade.

The alert listed "IVSO.ST 11%" — a top-up of a position already held — in the
same form as "BONEX.ST 5.5%", a new one, and read as a trade size.
"""
from __future__ import annotations

from fundmgr.data.prices import TickerFeatures
from fundmgr.engine.schema import Action
from fundmgr.reporting.actions import trade_kind, trade_line
from fundmgr.state.models import PortfolioSnapshot, Position


def _act(ticker, side, weight, sek):
    return Action(ticker=ticker, side=side, target_weight_pct=weight, sek_estimate=sek,
                  confidence=0.8, thesis="t")


SNAP = PortfolioSnapshot(positions=[Position("IVSO.ST", 40, 100.0, current_price_sek=200.0)],
                         cash_sek=92_000.0)          # NAV 100,000; IVSO 8%
FEATURES = {t: TickerFeatures(t, t, 200.0, "2026-10-12", 0) for t in ("IVSO.ST", "BONEX.ST")}


def test_a_buy_of_something_held_is_an_add_from_its_current_weight():
    line = trade_line(_act("IVSO.ST", "buy", 11, 3000), SNAP, FEATURES)
    assert line == "ADD IVSO.ST  8.0% → 11.0%  · buy 15 shares ≈ 3,000 SEK"


def test_a_buy_of_something_not_held_is_new():
    line = trade_line(_act("BONEX.ST", "buy", 5.5, 5500), SNAP, FEATURES)
    assert line == "NEW BONEX.ST  → 5.5%  · buy 27 shares ≈ 5,500 SEK"


def test_sells_say_whether_they_trim_or_exit():
    assert trade_kind(_act("IVSO.ST", "sell", 4, 4000), SNAP) == "TRIM"
    assert trade_kind(_act("IVSO.ST", "sell", 0, 8000), SNAP) == "EXIT"
    assert trade_line(_act("IVSO.ST", "sell", 0, 8000), SNAP, FEATURES).startswith(
        "EXIT IVSO.ST  8.0% → 0.0%  · sell 40 shares")


def test_without_a_price_the_amount_still_shows():
    line = trade_line(_act("NOPRICE.ST", "buy", 5, 5000), SNAP, FEATURES)
    assert line == "NEW NOPRICE.ST  → 5.0%  · buy ≈ 5,000 SEK"
