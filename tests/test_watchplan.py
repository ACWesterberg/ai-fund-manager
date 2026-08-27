"""Per-position kill criteria (text + numeric) and time horizons."""
from datetime import date, datetime, timedelta, timezone

import pytest

from fundmgr import paper, watchplan
from fundmgr.state.models import NavPoint, Transaction
from fundmgr.state.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "book.db")
    s.initialise(100_000)
    s.set_meta("paper_name", "Test Sleeve")
    s.set_meta("paper_currency_map", '{"NVDA": "USD", "VOLV-B.ST": "SEK"}')
    return s


@pytest.fixture
def no_telegram(monkeypatch):
    """Capture alerts instead of sending them."""
    sent = []
    import fundmgr.notify.send as send_mod
    monkeypatch.setattr(send_mod, "send_telegram", lambda text, **kw: sent.append(text) or True)
    return sent


def _hold(store, ticker, shares, cost_sek):
    store.apply_fill(Transaction(ticker=ticker, side="buy", shares=shares,
                                 price_sek=cost_sek, fee_sek=0.0, source="fill"))


def _price(store, ticker, close):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    store.save_prices(ticker, [{"date": today, "open": close, "high": close,
                                "low": close, "close": close, "volume": 100}])


@pytest.fixture
def sek_only(monkeypatch):
    """Treat every price as already-SEK so kill maths is exact in tests."""
    monkeypatch.setattr(paper, "sek_prices_for",
                        lambda store, tickers, currency_map, native: dict(native))


# ── Horizon parsing ───────────────────────────────────────────────────────────

def test_parse_horizon_from_months():
    iso, label = watchplan.parse_horizon(months=12, today=date(2026, 8, 11))
    assert iso == "2027-08-11"
    assert label == "12 months"


def test_parse_horizon_from_explicit_date():
    iso, label = watchplan.parse_horizon("2027-02-11", today=date(2026, 8, 11))
    assert iso == "2027-02-11"
    assert label == "6 months"


def test_parse_horizon_clamps_short_months():
    """31 Jan + 1 month lands on the last valid day of February, not a crash."""
    iso, _ = watchplan.parse_horizon(months=1, today=date(2026, 1, 31))
    assert iso == "2026-02-28"


def test_parse_horizon_rejects_garbage():
    assert watchplan.parse_horizon("not-a-date") is None
    assert watchplan.parse_horizon("") is None
    assert watchplan.parse_horizon(months=0) is None


def test_days_left_counts_down():
    future = (datetime.now(timezone.utc).date() + timedelta(days=5)).isoformat()
    assert watchplan.days_left(future) == 5
    assert watchplan.days_left(None) is None
    assert watchplan.days_left("garbage") is None


# ── Setting and clearing a plan ───────────────────────────────────────────────

def test_set_plan_round_trips(store):
    plan = watchplan.set_position_plan(
        store, "nvda", kill_criterion="loses the Apple socket",
        max_drawdown_pct="25", price_below="120", horizon_months="12",
        horizon_note="second data-centre cycle should be visible",
        today=date(2026, 8, 11))
    assert plan["kill_criterion"] == "loses the Apple socket"
    assert plan["kill_rules"]["max_drawdown_pct"] == 25.0
    assert plan["kill_rules"]["price_below"] == 120.0
    assert plan["horizon"]["review_date"] == "2027-08-11"
    assert plan["horizon"]["note"] == "second data-centre cycle should be visible"
    # Shares the meta key the JSON import writes, so both paths monitor alike.
    assert watchplan.get_kill_text(store)["NVDA"] == "loses the Apple socket"


def test_set_plan_leaves_untouched_fields_alone(store):
    watchplan.set_position_plan(store, "NVDA", kill_criterion="thesis broke",
                                max_drawdown_pct="25")
    watchplan.set_position_plan(store, "NVDA", horizon_months="6")
    plan = watchplan.plan_for(store, "NVDA")
    assert plan["kill_criterion"] == "thesis broke"
    assert plan["kill_rules"]["max_drawdown_pct"] == 25.0
    assert plan["horizon"]["review_date"]


def test_empty_string_clears_a_rule(store):
    watchplan.set_position_plan(store, "NVDA", kill_criterion="x", max_drawdown_pct="25")
    watchplan.set_position_plan(store, "NVDA", kill_criterion="", max_drawdown_pct="")
    plan = watchplan.plan_for(store, "NVDA")
    assert plan["kill_criterion"] == ""
    assert plan["kill_rules"] == {}


def test_clear_position_plan_removes_everything(store):
    watchplan.set_position_plan(store, "NVDA", kill_criterion="x",
                                max_drawdown_pct="25", horizon_months="12")
    watchplan.clear_position_plan(store, "NVDA")
    assert watchplan.plan_tickers(store) == set()


def test_set_plan_requires_a_ticker(store):
    with pytest.raises(ValueError):
        watchplan.set_position_plan(store, "  ", kill_criterion="x")


# ── Numeric kill rules ────────────────────────────────────────────────────────

def test_drawdown_rule_fires_and_dedupes(store, sek_only, no_telegram):
    _hold(store, "NVDA", 10, 100.0)
    _price(store, "NVDA", 100.0)
    watchplan.set_position_plan(store, "NVDA", max_drawdown_pct="25")   # anchors at 100
    _price(store, "NVDA", 70.0)                     # −30% from the anchor

    log = watchplan.check_kill_rules("slug", store=store)
    assert any("max drop reached" in line for line in log)
    assert len(no_telegram) == 1
    assert "NVDA" in no_telegram[0]

    # Same breach next run: no repeat alert.
    assert watchplan.check_kill_rules("slug", store=store) == []
    assert len(no_telegram) == 1


def test_drawdown_rule_rearms_after_recovery(store, sek_only, no_telegram):
    _hold(store, "NVDA", 10, 100.0)
    _price(store, "NVDA", 100.0)
    watchplan.set_position_plan(store, "NVDA", max_drawdown_pct="25")
    _price(store, "NVDA", 70.0)
    watchplan.check_kill_rules("slug", store=store)

    _price(store, "NVDA", 95.0)                     # recovered well clear of the line
    log = watchplan.check_kill_rules("slug", store=store)
    assert any("re-armed" in line for line in log)

    _price(store, "NVDA", 70.0)                     # breaks again → alerts again
    watchplan.check_kill_rules("slug", store=store)
    assert len(no_telegram) == 2


def test_drawdown_within_buffer_does_not_rearm(store, sek_only, no_telegram):
    """A position hovering on its line must not flip-flop into daily alerts."""
    _hold(store, "NVDA", 10, 100.0)
    _price(store, "NVDA", 100.0)
    watchplan.set_position_plan(store, "NVDA", max_drawdown_pct="25")
    _price(store, "NVDA", 74.0)
    watchplan.check_kill_rules("slug", store=store)

    _price(store, "NVDA", 75.5)                     # back over the line, inside the buffer
    assert watchplan.check_kill_rules("slug", store=store) == []
    assert len(no_telegram) == 1


def test_price_floor_and_target(store, sek_only, no_telegram):
    _hold(store, "NVDA", 10, 100.0)
    _price(store, "NVDA", 119.0)
    watchplan.set_position_plan(store, "NVDA", price_below="120", price_above="500")
    rows = watchplan.evaluate_kill_rules(store)
    assert rows[0]["hit"] and "floor" in rows[0]["reasons"][0]

    _price(store, "NVDA", 505.0)
    watchplan.set_position_plan(store, "NVDA", price_below="", price_above="500")
    rows = watchplan.evaluate_kill_rules(store)
    assert rows[0]["hit"] and "target" in rows[0]["reasons"][0]


def test_untouched_rule_stays_quiet(store, sek_only, no_telegram):
    _hold(store, "NVDA", 10, 100.0)
    _price(store, "NVDA", 98.0)
    watchplan.set_position_plan(store, "NVDA", max_drawdown_pct="25")
    assert watchplan.check_kill_rules("slug", store=store) == []
    assert no_telegram == []


def test_editing_a_rule_rearms_it(store, sek_only, no_telegram):
    """Tightening the line after a hit should be able to fire again today."""
    _hold(store, "NVDA", 10, 100.0)
    _price(store, "NVDA", 100.0)
    watchplan.set_position_plan(store, "NVDA", max_drawdown_pct="15")
    _price(store, "NVDA", 80.0)                     # −20% from the anchor
    watchplan.check_kill_rules("slug", store=store)
    assert len(no_telegram) == 1

    watchplan.set_position_plan(store, "NVDA", max_drawdown_pct="10")
    watchplan.check_kill_rules("slug", store=store)
    assert len(no_telegram) == 2


# ── The review anchor ─────────────────────────────────────────────────────────

def test_a_new_drawdown_line_anchors_to_todays_price(store, sek_only):
    _hold(store, "NVDA", 10, 40.0)                  # bought cheap, long ago
    _price(store, "NVDA", 100.0)
    watchplan.set_position_plan(store, "NVDA", max_drawdown_pct="25",
                                today=date(2026, 8, 13))
    rule = watchplan.get_kill_rules(store)["NVDA"]
    assert rule["anchor_price_sek"] == 100.0
    assert rule["anchor_date"] == "2026-08-13"


def test_the_line_is_measured_from_the_anchor_not_from_cost(store, sek_only, no_telegram):
    """The point of the whole change: a big gain must not absorb a big fall.

    Bought at 40, reviewed at 100, now 70. That is −30% since the review — a
    kill — while still +75% on cost, which the old anchor would have read as
    nowhere near the line.
    """
    _hold(store, "NVDA", 10, 40.0)
    _price(store, "NVDA", 100.0)
    watchplan.set_position_plan(store, "NVDA", max_drawdown_pct="25")
    _price(store, "NVDA", 70.0)

    row = watchplan.evaluate_kill_rules(store)[0]
    assert row["hit"] is True
    assert row["drawdown_pct"] == -30.0
    assert row["drawdown_basis"] == "review"
    assert row["drawdown_from_cost_pct"] == 75.0      # still well up on cost
    assert "review price" in row["reasons"][0]


def test_editing_the_threshold_does_not_move_the_anchor(store, sek_only):
    """Tightening a line is a change of mind, not a fresh review — re-anchoring
    there would quietly reset a position already halfway to its kill."""
    _hold(store, "NVDA", 10, 100.0)
    _price(store, "NVDA", 100.0)
    watchplan.set_position_plan(store, "NVDA", max_drawdown_pct="25",
                                today=date(2026, 8, 13))
    _price(store, "NVDA", 80.0)
    watchplan.set_position_plan(store, "NVDA", max_drawdown_pct="15",
                                today=date(2026, 9, 1))

    rule = watchplan.get_kill_rules(store)["NVDA"]
    assert rule["anchor_price_sek"] == 100.0
    assert rule["anchor_date"] == "2026-08-13"
    assert watchplan.evaluate_kill_rules(store)[0]["drawdown_pct"] == -20.0


def test_re_anchor_moves_it_deliberately(store, sek_only):
    _hold(store, "NVDA", 10, 100.0)
    _price(store, "NVDA", 100.0)
    watchplan.set_position_plan(store, "NVDA", max_drawdown_pct="25",
                                today=date(2026, 8, 13))
    _price(store, "NVDA", 80.0)
    watchplan.set_position_plan(store, "NVDA", re_anchor=True, today=date(2026, 9, 1))

    rule = watchplan.get_kill_rules(store)["NVDA"]
    assert rule["anchor_price_sek"] == 80.0
    assert rule["anchor_date"] == "2026-09-01"
    assert rule["max_drawdown_pct"] == 25.0          # the line itself is untouched
    assert watchplan.evaluate_kill_rules(store)[0]["drawdown_pct"] == 0.0


def test_a_rule_written_before_anchors_still_measures_from_cost(store, sek_only):
    """Migrating an armed kill line by silently re-anchoring would disarm it,
    so old rules keep their meaning and say which one they have."""
    import json
    _hold(store, "NVDA", 10, 100.0)
    _price(store, "NVDA", 70.0)
    store.set_meta(watchplan.KILL_RULES_KEY,
                   json.dumps({"NVDA": {"max_drawdown_pct": 25.0}}))   # no anchor

    row = watchplan.evaluate_kill_rules(store)[0]
    assert row["drawdown_basis"] == "cost"
    assert row["drawdown_pct"] == -30.0
    assert row["hit"] is True
    assert "from cost" in row["reasons"][0]


def test_no_cached_price_leaves_the_line_on_cost(store, sek_only):
    """An anchor is only recorded when the price is already known — saving a
    plan must not block on a quote feed."""
    _hold(store, "NVDA", 10, 100.0)                  # no _price() call
    watchplan.set_position_plan(store, "NVDA", max_drawdown_pct="25")
    assert "anchor_price_sek" not in watchplan.get_kill_rules(store)["NVDA"]


def test_the_add_plans_review_price_backs_the_anchor(store, sek_only):
    """Both halves of a position's monitoring should date from the same review."""
    from fundmgr import addsignal
    _hold(store, "NVDA", 10, 100.0)                  # no cached price
    addsignal.set_plan(store, "NVDA", review_price=140.0)
    watchplan.set_position_plan(store, "NVDA", max_drawdown_pct="25")
    assert watchplan.get_kill_rules(store)["NVDA"]["anchor_price_sek"] == 140.0


def test_clearing_the_drawdown_line_drops_its_anchor(store, sek_only):
    _hold(store, "NVDA", 10, 100.0)
    _price(store, "NVDA", 100.0)
    watchplan.set_position_plan(store, "NVDA", max_drawdown_pct="25", price_below="50")
    watchplan.set_position_plan(store, "NVDA", max_drawdown_pct="")
    rule = watchplan.get_kill_rules(store)["NVDA"]
    assert "anchor_price_sek" not in rule and "anchor_date" not in rule
    assert rule["price_below"] == 50.0                # the other line survives


def test_the_dashboard_panel_breaches_on_the_same_number_as_the_watch(store, sek_only):
    """The panel used to compare P&L (from cost) against a line the watch
    measured from the anchor, so one could read breached and the other clear."""
    from fundmgr.web.paper import _watch_status
    _hold(store, "NVDA", 10, 40.0)
    _price(store, "NVDA", 100.0)
    watchplan.set_position_plan(store, "NVDA", max_drawdown_pct="25")
    _price(store, "NVDA", 70.0)

    positions_data = [{"ticker": "NVDA", "weight_pct": 100.0, "pnl_pct": 75.0,
                       "current_price": 70.0, "avg_cost": 40.0}]
    row = _watch_status(store, positions_data)["rows"][0]

    assert row["dd_breached"] is True                 # despite being +75% on cost
    assert row["drawdown_pct"] == -30.0
    assert row["drawdown_basis"] == "review"
    assert row["pnl_pct"] == 75.0                     # P&L still reported from cost
    assert row["dd_breached"] is watchplan.evaluate_kill_rules(store)[0]["hit"]


def test_drawdown_for_is_the_one_definition(store):
    """The dashboard panel and the daily watch both call this, so a line cannot
    read breached in one and clear in the other."""
    dd = watchplan.drawdown_for({"anchor_price_sek": 100.0, "anchor_date": "2026-08-13"},
                                price_sek=70.0, avg_cost_sek=40.0)
    assert dd["pct"] == pytest.approx(-30.0)         # raw; rounding is the caller's
    assert dd["from_cost_pct"] == pytest.approx(75.0)
    assert dd["basis"] == "review"
    assert dd["anchor_price_sek"] == 100.0
    assert dd["anchor_date"] == "2026-08-13"

    no_anchor = watchplan.drawdown_for({}, price_sek=70.0, avg_cost_sek=100.0)
    assert no_anchor["basis"] == "cost"
    assert no_anchor["pct"] == pytest.approx(-30.0)

    nothing = watchplan.drawdown_for({}, price_sek=None, avg_cost_sek=None)
    assert nothing["pct"] is None and nothing["from_cost_pct"] is None


def test_kill_rules_noop_without_rules(store):
    assert watchplan.evaluate_kill_rules(store) == []
    assert watchplan.check_kill_rules("slug", store=store) == []


def test_drawdown_needs_a_cost_basis(store, sek_only, no_telegram):
    """A plan-only ticker has no cost, so a drawdown rule can't fire on it."""
    _price(store, "NVDA", 10.0)
    watchplan.set_position_plan(store, "NVDA", max_drawdown_pct="25")
    rows = watchplan.evaluate_kill_rules(store)
    assert rows[0]["held"] is False
    assert rows[0]["hit"] is False


# ── Time horizons ─────────────────────────────────────────────────────────────

def _set_horizon(store, ticker, days_out, today):
    watchplan.set_position_plan(
        store, ticker, review_date=(today + timedelta(days=days_out)).isoformat())


def test_horizon_alerts_escalate_once_per_stage(store, no_telegram):
    today = date(2026, 8, 11)
    _hold(store, "NVDA", 10, 100.0)
    _set_horizon(store, "NVDA", 40, today)

    # 40 days out — outside every bucket.
    assert watchplan.check_horizons("slug", store=store, today=today) == []

    # 25 days out → the 30-day bucket fires once.
    at_25 = today + timedelta(days=15)
    assert watchplan.check_horizons("slug", store=store, today=at_25)
    assert watchplan.check_horizons("slug", store=store, today=at_25) == []
    assert len(no_telegram) == 1

    # 10 days out → the 14-day bucket.
    at_10 = today + timedelta(days=30)
    assert watchplan.check_horizons("slug", store=store, today=at_10)
    assert len(no_telegram) == 2

    # On the day → the final bucket, then silence.
    at_0 = today + timedelta(days=40)
    log = watchplan.check_horizons("slug", store=store, today=at_0)
    assert any("due today" in line for line in log)
    assert len(no_telegram) == 3
    assert watchplan.check_horizons("slug", store=store,
                                   today=at_0 + timedelta(days=3)) == []
    assert len(no_telegram) == 3


def test_horizon_set_inside_a_bucket_skips_earlier_stages(store, no_telegram):
    """A horizon set 10 days out opens at '14 days', not by replaying 30."""
    today = date(2026, 8, 11)
    _set_horizon(store, "NVDA", 10, today)
    log = watchplan.check_horizons("slug", store=store, today=today)
    assert len(log) == 1
    assert "in 10d" in log[0]
    assert store.get_meta("paper_horizon_stage:NVDA") == "14"


def test_overdue_horizon_reports_as_overdue(store, no_telegram):
    today = date(2026, 8, 11)
    watchplan.set_position_plan(store, "NVDA", review_date="2026-08-01")
    log = watchplan.check_horizons("slug", store=store, today=today)
    assert "overdue by 10d" in log[0]
    assert "10 days past horizon" in no_telegram[0]


def test_changing_the_horizon_rearms_alerts(store, no_telegram):
    today = date(2026, 8, 11)
    _set_horizon(store, "NVDA", 5, today)
    watchplan.check_horizons("slug", store=store, today=today)
    assert len(no_telegram) == 1

    _set_horizon(store, "NVDA", 20, today)          # pushed out → fresh countdown
    watchplan.check_horizons("slug", store=store, today=today)
    assert len(no_telegram) == 2


def test_horizon_alert_quotes_the_kill_criterion(store, no_telegram):
    today = date(2026, 8, 11)
    watchplan.set_position_plan(store, "NVDA", kill_criterion="loses the Apple socket")
    _set_horizon(store, "NVDA", 3, today)
    watchplan.check_horizons("slug", store=store, today=today)
    assert "loses the Apple socket" in no_telegram[0]


def test_horizon_rows_sorted_by_urgency(store):
    today = date(2026, 8, 11)
    _set_horizon(store, "NVDA", 30, today)
    _set_horizon(store, "MSFT", 2, today)
    rows = watchplan.horizon_rows(store, today)
    assert [r["ticker"] for r in rows] == ["MSFT", "NVDA"]
    assert rows[0]["near"] and not rows[0]["due"]


def test_horizons_noop_when_none_set(store, no_telegram):
    assert watchplan.check_horizons("slug", store=store) == []
    assert no_telegram == []


# ── Wiring into the daily track ───────────────────────────────────────────────

def test_track_portfolio_runs_the_new_watches(tmp_path, monkeypatch, no_telegram):
    """A drawdown breach and a due horizon both surface in `fund paper-track`."""
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path / "paper")
    monkeypatch.setattr(paper, "detect_currency", lambda t: "SEK")
    monkeypatch.setattr(paper, "sek_prices_for",
                        lambda store, tickers, currency_map, native: dict(native))
    monkeypatch.setattr(paper, "_cache_price_history", lambda store, tickers, **kw: None)
    monkeypatch.setattr(paper, "_search_symbol", lambda name: None)
    import fundmgr.data.benchmark as benchmark
    monkeypatch.setattr(benchmark, "fetch_and_cache_benchmark", lambda store, **kw: True)
    import fundmgr.data.quotes as quotes
    monkeypatch.setattr(quotes, "live_prices", lambda tickers: {t: 300.0 for t in tickers})

    slug, _log = paper.create_portfolio(
        "Watched Book", 100_000, "",
        holdings_override=[{
            "ticker": "VOLV-B.ST", "name": "Volvo B", "weight_pct": 100,
            "shares": 100, "avg_cost_sek": 300.0,
            "max_drawdown_pct": 20, "horizon_months": 12,
        }],
        kind="live", seed_holdings=True)

    _meta, store = paper.open_portfolio(slug)
    _price(store, "VOLV-B.ST", 200.0)                       # −33% from cost
    store.set_meta("paper_horizons", '{"VOLV-B.ST": {"review_date": "2020-01-01"}}')

    log = paper.track_portfolio(slug)
    assert any("max drop reached" in line for line in log)
    assert any("fresh analysis due" in line for line in log)


def test_track_survives_a_broken_watch(tmp_path, monkeypatch, no_telegram):
    """One failing watch must not stop the rest of the daily upkeep."""
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path / "paper")
    monkeypatch.setattr(paper, "detect_currency", lambda t: "SEK")
    monkeypatch.setattr(paper, "_cache_price_history", lambda store, tickers, **kw: None)
    monkeypatch.setattr(paper, "_search_symbol", lambda name: None)
    import fundmgr.data.benchmark as benchmark
    monkeypatch.setattr(benchmark, "fetch_and_cache_benchmark", lambda store, **kw: True)
    import fundmgr.data.quotes as quotes
    monkeypatch.setattr(quotes, "live_prices", lambda tickers: {t: 300.0 for t in tickers})

    slug, _ = paper.create_portfolio(
        "Fragile", 100_000, "",
        holdings_override=[{"ticker": "VOLV-B.ST", "name": "Volvo B", "weight_pct": 100,
                            "shares": 10, "avg_cost_sek": 300.0}],
        kind="live", seed_holdings=True)

    def boom(*a, **kw):
        raise RuntimeError("price feed down")
    monkeypatch.setattr(watchplan, "check_kill_rules", boom)

    log = paper.track_portfolio(slug)
    assert any("kill-rule watch failed" in line for line in log)
    assert any("NAV" in line for line in log)


# ── Seeding a book from reviewed rows ─────────────────────────────────────────

@pytest.fixture
def seeded(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path / "paper")
    monkeypatch.setattr(paper, "detect_currency", lambda t: "SEK")
    monkeypatch.setattr(paper, "_cache_price_history", lambda store, tickers, **kw: None)
    monkeypatch.setattr(paper, "_search_symbol", lambda name: None)
    import fundmgr.data.benchmark as benchmark
    monkeypatch.setattr(benchmark, "fetch_and_cache_benchmark", lambda store, **kw: True)
    import fundmgr.data.quotes as quotes
    monkeypatch.setattr(quotes, "live_prices", lambda tickers: {t: 250.0 for t in tickers})
    return None


def test_seed_holdings_books_shares_at_cost(seeded):
    slug, log = paper.create_portfolio(
        "Seeded", 100_000, "",
        holdings_override=[
            {"ticker": "VOLV-B.ST", "name": "Volvo B", "weight_pct": 60,
             "shares": 100, "avg_cost_sek": 300.0, "kill_criterion": "order book halves",
             "max_drawdown_pct": 25, "horizon_months": 12},
            {"ticker": "SAND.ST", "name": "Sandvik", "weight_pct": 40,
             "shares": 50, "avg_cost_sek": 200.0},
        ],
        kind="live", seed_holdings=True)

    _meta, store = paper.open_portfolio(slug)
    positions = {p.ticker: p for p in store.get_positions()}
    assert positions["VOLV-B.ST"].shares == 100
    assert positions["VOLV-B.ST"].avg_cost_sek == 300.0
    assert positions["SAND.ST"].shares == 50
    # Seeded at cost with no fee — cash is capital minus the cost of the book.
    assert store.get_cash() == pytest.approx(100_000 - (100 * 300 + 50 * 200))
    assert store.total_fees_paid() == 0
    assert any("Seeded 100 × VOLV-B.ST" in line for line in log)

    plan = watchplan.plan_for(store, "VOLV-B.ST")
    assert plan["kill_criterion"] == "order book halves"
    assert plan["kill_rules"]["max_drawdown_pct"] == 25.0
    assert plan["horizon"]["review_date"]
    assert watchplan.plan_for(store, "SAND.ST")["kill_rules"] == {}


def test_seed_skips_rows_without_shares_or_cost(seeded):
    slug, log = paper.create_portfolio(
        "Partial", 50_000, "",
        holdings_override=[
            {"ticker": "VOLV-B.ST", "name": "Volvo B", "weight_pct": 50,
             "shares": 10, "avg_cost_sek": 300.0},
            {"ticker": "SAND.ST", "name": "Sandvik", "weight_pct": 50, "shares": 10},
        ],
        kind="live", seed_holdings=True)

    _meta, store = paper.open_portfolio(slug)
    assert [p.ticker for p in store.get_positions()] == ["VOLV-B.ST"]
    assert any("SAND.ST: no share count / cost basis" in line for line in log)


def test_seed_with_nothing_usable_is_an_error(seeded):
    with pytest.raises(ValueError, match="No holdings could be seeded"):
        paper.create_portfolio(
            "Empty", 50_000, "",
            holdings_override=[{"ticker": "VOLV-B.ST", "name": "Volvo B", "weight_pct": 100}],
            kind="live", seed_holdings=True)


def test_seeded_book_records_a_nav_point(seeded):
    slug, _ = paper.create_portfolio(
        "Nav Check", 40_000, "",
        holdings_override=[{"ticker": "VOLV-B.ST", "name": "Volvo B", "weight_pct": 100,
                            "shares": 100, "avg_cost_sek": 300.0}],
        kind="live", seed_holdings=True)
    _meta, store = paper.open_portfolio(slug)
    navs = store.get_nav_history()
    assert navs and navs[-1].portfolio_nav_sek == pytest.approx(40_000)
    assert isinstance(navs[-1], NavPoint)


def test_seeded_targets_reflect_actual_weights(seeded):
    """Drift is only meaningful if targets match the book you actually hold."""
    import json as _json
    slug, _ = paper.create_portfolio(
        "Weighted", 40_000, "",   # fully invested: 30 000 + 10 000, no idle cash
        holdings_override=[
            {"ticker": "VOLV-B.ST", "name": "Volvo B",
             "shares": 100, "avg_cost_sek": 300.0},   # 30 000 = 75%
            {"ticker": "SAND.ST", "name": "Sandvik",
             "shares": 50, "avg_cost_sek": 200.0},    # 10 000 = 25%
        ],
        kind="live", seed_holdings=True)

    _meta, store = paper.open_portfolio(slug)
    targets = _json.loads(store.get_meta("paper_target_weights"))
    assert targets["VOLV-B.ST"] == pytest.approx(75.0)
    assert targets["SAND.ST"] == pytest.approx(25.0)


def test_seeded_targets_account_for_idle_cash(seeded):
    """A mostly-cash book keeps small real weights — not rescaled to 100%."""
    import json as _json
    slug, _ = paper.create_portfolio(
        "Cash Heavy", 1_000_000, "",
        holdings_override=[{"ticker": "VOLV-B.ST", "name": "Volvo B",
                            "shares": 10, "avg_cost_sek": 300.0}],   # 3 000 = 0.3%
        kind="live", seed_holdings=True)

    _meta, store = paper.open_portfolio(slug)
    targets = _json.loads(store.get_meta("paper_target_weights"))
    assert targets["VOLV-B.ST"] == pytest.approx(0.3)
