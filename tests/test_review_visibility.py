"""
Tests for making a review verdict visible somewhere other than a notification.

A stop breach or a target hit produces an instruction only a human can carry
out — "TRIM, sell 50%, target +50% → +65%". It was sent to Telegram and stored
nowhere a person could look it up: the dashboard showed the position, the level
it had been re-targeted to, and the fact that something had been decided, not at
all. These cover the record it now keeps, the sentence it now says, and the card
that now shows both.

Run: pytest tests/test_review_visibility.py -v
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

try:  # pragma: no cover - environment-dependent
    import financedata  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _stub = types.ModuleType("financedata")
    for _name in (
        "get_prices_since", "rsi", "pct_return", "ann_vol", "get_cache",
        "get_fundamentals", "ts_to_days",
    ):
        setattr(_stub, _name, lambda *a, **k: None)
    sys.modules["financedata"] = _stub

from fundmgr.engine.review_common import (  # noqa: E402
    follow_up,
    instruction,
    log_review,
    needs_trade,
)
from fundmgr.engine.schema import StopReview, TargetReview  # noqa: E402
from fundmgr.engine.target_review import apply_new_target  # noqa: E402
from fundmgr.state.store import Store  # noqa: E402
from fundmgr.web.views import open_reviews_context, recent_reviews_context  # noqa: E402

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "src" / "fundmgr" / "web" / "templates"
_jinja = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True)


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "test.db"))


def _target(rec="trim", trim=50.0, new_tp=65.0, ticker="TRUE-B.ST"):
    return TargetReview(
        ticker=ticker, recommendation=rec, confidence=0.85, trim_pct=trim,
        new_take_profit_pct=new_tp,
        what_changed="RSI rose to 78, crossing the prior trim threshold of 75.",
        rationale="Bank half; momentum is overextended and the recovery is unconfirmed.",
    )


def _stop(rec="exit", ticker="TRUE-B.ST"):
    return StopReview(
        ticker=ticker, recommendation=rec, confidence=0.7, trim_pct=None,
        what_changed="Guidance cut.", rationale="The thesis broke.",
    )


def _position(ticker="TRUE-B.ST", shares=428, price=23.05):
    return {"ticker": ticker, "name": "Truecaller", "shares": shares,
            "avg_cost": 15.35, "current_price": price}


# ── What the record keeps ─────────────────────────────────────────────────────

def test_a_verdict_is_stored_in_full_not_just_as_a_score(store):
    """decision_outcomes keeps 'sell'; the human needs the size and the level."""
    log_review(_target(), store, "target_review", 23.05, votes={"trim": 3}, n_samples=3,
               old_target_pct=50.0)
    row = store.get_recent_reviews()[0]
    assert (row["verdict"], row["trim_pct"], row["new_target_pct"]) == ("trim", 50.0, 65.0)
    assert row["old_target_pct"] == 50.0
    assert row["votes"] == {"trim": 3} and row["n_samples"] == 3
    assert "RSI rose to 78" in row["what_changed"]


def test_a_verdict_that_asks_for_a_trade_stays_open(store):
    for verdict in ("trim", "sell"):
        log_review(_target(verdict, ticker=f"{verdict.upper()}.ST"), store,
                   "target_review", 10.0)
    log_review(_stop("exit", ticker="EXIT.ST"), store, "stop_review", 10.0)
    assert {r["ticker"] for r in store.get_open_reviews()} == {"TRIM.ST", "SELL.ST", "EXIT.ST"}


def test_a_verdict_that_asks_for_nothing_is_noted_not_queued(store):
    """RAISE moves the plan by itself — queuing it would be a chore that isn't one."""
    log_review(_target("raise", trim=None), store, "target_review", 23.05)
    assert store.get_open_reviews() == []
    assert store.get_recent_reviews()[0]["status"] == "noted"


def test_needs_trade_marks_only_the_verdicts_a_broker_can_fill():
    assert [needs_trade(v) for v in ("exit", "sell", "trim", "add")] == [True] * 4
    assert [needs_trade(v) for v in ("raise", "hold")] == [False, False]


def test_a_later_review_supersedes_the_open_one_on_the_same_name(store):
    """Two live instructions for one position is worse than none."""
    log_review(_target("trim"), store, "target_review", 23.05)
    store.record_review(ticker="TRUE-B.ST", source="stop_review", verdict="exit",
                        review_date="2026-08-27")
    open_ids = [r["review_id"] for r in store.get_open_reviews()]
    assert open_ids == ["2026-08-27-stop_review-TRUE-B.ST"]
    assert [r["status"] for r in store.get_recent_reviews()][1] == "superseded"


def test_an_open_review_is_closed_once_you_say_what_you_did(store):
    rid = log_review(_target(), store, "target_review", 23.05)
    assert store.resolve_review(rid, "done") is True
    assert store.get_open_reviews() == []
    assert store.get_recent_reviews()[0]["status"] == "done"
    assert store.resolve_review(rid, "done") is False, "already closed"


def test_dismissing_is_recorded_as_a_decision_not_a_deletion(store):
    rid = log_review(_target(), store, "target_review", 23.05)
    store.resolve_review(rid, "dismissed")
    assert store.get_recent_reviews()[0]["status"] == "dismissed"


def test_an_unknown_status_is_refused(store):
    rid = log_review(_target(), store, "target_review", 23.05)
    with pytest.raises(ValueError):
        store.resolve_review(rid, "maybe")


def test_the_applied_target_is_tied_to_the_review_that_moved_it(store):
    store.set_position_stop("TRUE-B.ST", stop_pct=12.0, take_profit_pct=50.0)
    review = _target("raise", trim=None, new_tp=65.0)
    log_review(review, store, "target_review", 23.05, old_target_pct=50.0)
    assert apply_new_target(review, store) == (50.0, 65.0)
    row = store.get_recent_reviews()[0]
    assert (row["old_target_pct"], row["applied_target_pct"]) == (50.0, 65.0)


def test_a_target_applied_after_midnight_still_attaches_to_its_review(store):
    """The apply is a separate step; it must not lose the review to a date roll."""
    store.set_position_stop("TRUE-B.ST", stop_pct=12.0, take_profit_pct=50.0)
    store.record_review(ticker="TRUE-B.ST", source="target_review", verdict="raise",
                        new_target_pct=65.0, old_target_pct=50.0, review_date="2026-08-25")
    assert apply_new_target(_target("raise", trim=None, new_tp=65.0), store) == (50.0, 65.0)
    assert store.get_recent_reviews()[0]["applied_target_pct"] == 65.0


def test_a_refused_target_leaves_the_applied_level_empty(store):
    """Proposed and declined must not read as moved."""
    store.set_position_stop("TRUE-B.ST", stop_pct=12.0, take_profit_pct=50.0)
    review = _target("raise", trim=None, new_tp=40.0)
    log_review(review, store, "target_review", 23.05, old_target_pct=50.0)
    assert apply_new_target(review, store) is None
    row = store.get_recent_reviews()[0]
    assert row["applied_target_pct"] is None and row["new_target_pct"] == 40.0


# ── What the verdict says to do ───────────────────────────────────────────────

def test_a_trim_says_how_much_to_sell_in_shares_and_kronor():
    assert instruction("trim", trim_pct=50.0, shares=214, sek=4932) == (
        "Sell 50% of the position — about 214 shares, ≈4,932 SEK."
    )


def test_the_same_sentence_works_without_position_maths():
    """Telegram has no share count to hand; the instruction still has to parse."""
    assert instruction("trim", trim_pct=50.0) == "Sell 50% of the position."


def test_an_exit_is_the_whole_position():
    assert instruction("exit", shares=428).startswith("Sell the whole position — about 428 shares")


def test_a_raise_says_there_is_nothing_to_place():
    """The verdict that confused the reader most: a decision, but not a chore."""
    assert instruction("raise", new_target_pct=65.0) == (
        "Nothing to place: keep the position, target now +65%."
    )
    assert instruction("hold") == "Nothing to place: keep the position and its current level."


def test_the_follow_up_says_what_becomes_of_the_remainder():
    assert follow_up("trim", 50.0, 65.0, 50.0) == (
        "Keep the other 50%; its target is now +65% (was +50%)."
    )
    assert follow_up("trim", 50.0) == "Keep the other 50% as it is."
    assert follow_up("sell", None) == ""


# ── What the dashboard shows ──────────────────────────────────────────────────

def test_the_card_turns_a_percentage_into_shares_and_kronor(store):
    store.set_position_stop("TRUE-B.ST", stop_pct=12.0, take_profit_pct=50.0)
    review = _target()
    log_review(review, store, "target_review", 23.05, votes={"trim": 3}, n_samples=3,
               old_target_pct=50.0)
    apply_new_target(review, store)

    ctx = open_reviews_context(store, [_position()])
    assert ctx["count"] == 1
    row = ctx["rows"][0]
    assert row["shares_to_trade"] == 214          # half of 428
    assert row["sek_estimate"] == round(214 * 23.05)
    assert row["instruction"] == "Sell 50% of the position — about 214 shares, ≈4,933 SEK."
    assert row["follow_up"] == "Keep the other 50%; its target is now +65% (was +50%)."
    assert row["target_line"] == "target +50% → +65%"
    assert row["consensus"] == "trim×3 of 3"
    assert row["source_label"] == "Target hit"


def test_without_a_live_price_the_share_count_stands_alone(store):
    """Costing the order at entry price would understate a +50% winner by the gain."""
    log_review(_target(), store, "target_review", 23.05)
    row = open_reviews_context(store, [dict(_position(), current_price=None)])["rows"][0]
    assert row["shares_to_trade"] == 214
    assert row["sek_estimate"] is None
    assert row["instruction"] == "Sell 50% of the position — about 214 shares."


def test_an_instruction_for_a_position_you_no_longer_hold_stops_being_shown(store):
    """Selling out is the instruction being carried out, ticked off or not."""
    log_review(_stop("exit"), store, "stop_review", 23.05)
    assert open_reviews_context(store, [])["count"] == 0
    assert open_reviews_context(store, [_position()])["count"] == 1


def test_the_log_keeps_the_verdicts_that_asked_for_nothing(store):
    log_review(_target("raise", trim=None), store, "target_review", 23.05)
    log_review(_target(ticker="EVO.ST"), store, "target_review", 10.0)
    assert {r["verdict"] for r in recent_reviews_context(store, [_position()])} == {"raise", "trim"}


# ── What the page renders ─────────────────────────────────────────────────────

def _render(**ctx):
    base = {"positions": [], "cash": 0, "cash_pct": 100.0, "nav": 0, "fees_paid": 0,
            "stats": None, "has_history": False, "last_run": None, "pnl_sek": 0,
            "pnl_pct": 0.0, "active_page": "portfolio", "benchmark_label": "OMXSPI",
            "reviews": {"rows": [], "count": 0}, "recent_reviews": [],
            "review_resolve_action": "/reviews", "show_levels": True}
    base.update(ctx)
    return _jinja.get_template("index.html").render(**base)


def test_the_dashboard_prints_the_order_and_the_reason(store):
    store.set_position_stop("TRUE-B.ST", stop_pct=12.0, take_profit_pct=50.0)
    review = _target()
    rid = log_review(review, store, "target_review", 23.05, votes={"trim": 3}, n_samples=3,
                     old_target_pct=50.0)
    apply_new_target(review, store)
    html = _render(reviews=open_reviews_context(store, [_position()]))

    assert "Decisions to action" in html and "1 waiting" in html
    assert "Sell 50% of the position — about 214 shares" in html
    assert "out of 428 shares held" in html
    assert "target +50% → +65%" in html
    assert "RSI rose to 78" in html                       # what changed
    assert "momentum is overextended" in html             # why
    assert f"/reviews/{rid}/resolve" in html              # tick it off from the page
    assert "the fund never trades for you" in html


def test_nothing_waiting_reads_as_nothing_waiting(store):
    log_review(_target("raise", trim=None), store, "target_review", 23.05)
    html = _render(recent_reviews=recent_reviews_context(store, [_position()]))
    assert "nothing waiting" in html
    assert "Earlier reviews" in html
    assert "Nothing to place: keep the position" in html


def test_the_card_is_absent_when_no_review_has_ever_run():
    assert "Decisions to action" not in _render()


def test_a_position_row_shows_the_levels_it_is_measured_against():
    html = _render(positions=[dict(_position(), cost_value=6570, current_value=9865,
                                   pnl_sek=3295, pnl_pct=50.1, weight_pct=12.0,
                                   stop_pct=12.0, take_profit_pct=65.0)])
    assert "stop -12%" in html and "target +65%" in html


# ── Ticking one off ───────────────────────────────────────────────────────────

@pytest.fixture
def client(store, tmp_path, monkeypatch):
    """The real app, pointed at a throwaway store instead of the fund's own."""
    from fastapi.testclient import TestClient

    from fundmgr.config import AppConfig
    import fundmgr.web.app as web_app

    cfg = AppConfig()
    cfg.db_path = tmp_path / "test.db"
    monkeypatch.setattr(web_app, "_cfg", cfg)
    monkeypatch.setattr(web_app, "_store", store)
    monkeypatch.setattr(web_app, "_fetch_live_prices", lambda tickers: {})
    return TestClient(web_app.app)


def test_the_page_can_tick_an_instruction_off(client, store):
    rid = log_review(_target(), store, "target_review", 23.05)
    r = client.post(f"/reviews/{rid}/resolve", data={"status": "done"}, follow_redirects=False)
    assert r.status_code == 303
    assert store.get_recent_reviews()[0]["status"] == "done"


def test_ticking_the_same_one_off_twice_says_so_rather_than_erroring(client, store):
    rid = log_review(_target(), store, "target_review", 23.05)
    client.post(f"/reviews/{rid}/resolve", data={"status": "done"}, follow_redirects=False)
    again = client.post(f"/reviews/{rid}/resolve", data={"status": "done"}, follow_redirects=False)
    assert again.status_code == 303
    assert "already+closed" in again.headers["location"] and "ok=0" in again.headers["location"]


def test_an_invented_status_is_refused(client, store):
    rid = log_review(_target(), store, "target_review", 23.05)
    assert client.post(f"/reviews/{rid}/resolve", data={"status": "traded-half"}).status_code == 400
    assert store.get_open_reviews()[0]["review_id"] == rid
