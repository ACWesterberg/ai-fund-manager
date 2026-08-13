"""Add signals: proof, dislocation, and the gates that stop both failure modes."""
from datetime import date, datetime, timezone

import pytest

from fundmgr import addsignal, paper, watchplan
from fundmgr.state.models import Transaction
from fundmgr.state.store import Store

TODAY = date(2026, 8, 12)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "book.db")
    s.initialise(100_000)
    s.set_meta("paper_name", "Add Sleeve")
    s.set_meta("paper_currency_map", '{"SYSR.ST": "SEK"}')
    return s


@pytest.fixture(autouse=True)
def sek_only(monkeypatch):
    monkeypatch.setattr(paper, "sek_prices_for",
                        lambda store, tickers, currency_map, native: dict(native))


@pytest.fixture
def telegram(monkeypatch):
    sent = []
    import fundmgr.notify.send as send_mod
    monkeypatch.setattr(send_mod, "send_telegram", lambda t, **k: sent.append(t) or True)
    return sent


def _hold(store, ticker="SYSR.ST", shares=100, cost=100.0):
    store.apply_fill(Transaction(ticker=ticker, side="buy", shares=shares,
                                 price_sek=cost, fee_sek=0.0, source="fill"))


def _price(store, ticker, close):
    store.save_prices(ticker, [{"date": TODAY.isoformat(), "open": close, "high": close,
                                "low": close, "close": close, "volume": 100}])


def _horizon(store, ticker, months=24):
    """A horizon is what gives the valuation gate its time axis."""
    iso, _ = watchplan.parse_horizon(months=months, today=TODAY)
    watchplan.set_position_plan(store, ticker, review_date=iso, today=TODAY)


# ── Expected return ───────────────────────────────────────────────────────────

def test_expected_return_annualises_over_the_time_left():
    # 30% upside over 12 months is ~30%/yr; over 6 months it is far more.
    assert addsignal.expected_annual_return(130, 100, 12) == pytest.approx(30.0, rel=1e-3)
    assert addsignal.expected_annual_return(130, 100, 6) == pytest.approx(69.0, abs=0.5)
    assert addsignal.expected_annual_return(130, 100, 24) == pytest.approx(14.0, abs=0.5)


def test_expected_return_is_negative_above_target():
    assert addsignal.expected_annual_return(100, 130, 12) < 0


def test_expected_return_refuses_an_expired_horizon():
    """Annualising over zero months manufactures an enormous number."""
    assert addsignal.expected_annual_return(130, 100, 0) is None
    assert addsignal.expected_annual_return(130, 100, -3) is None
    assert addsignal.expected_annual_return(130, 0, 12) is None
    assert addsignal.expected_annual_return(None, 100, 12) is None


# ── Plan storage ──────────────────────────────────────────────────────────────

def test_setting_a_target_stamps_it_with_today(store):
    plan = addsignal.set_plan(store, "sysr.st", target_price=145, today=TODAY)
    assert plan["target_price"] == 145.0
    assert plan["target_set_at"] == "2026-08-12"


def test_fields_are_independent(store):
    addsignal.set_plan(store, "SYSR.ST", book="A", max_weight_pct=13, today=TODAY)
    addsignal.set_plan(store, "SYSR.ST", target_price=145, today=TODAY)
    plan = addsignal.get_plan(store, "SYSR.ST")
    assert plan["book"] == "A" and plan["max_weight_pct"] == 13.0
    assert plan["target_price"] == 145.0


def test_an_unknown_book_is_rejected(store):
    with pytest.raises(ValueError, match="Unknown book"):
        addsignal.set_plan(store, "SYSR.ST", book="Z")


def test_clearing_a_target_drops_its_stamp(store):
    addsignal.set_plan(store, "SYSR.ST", target_price=145, today=TODAY)
    addsignal.set_plan(store, "SYSR.ST", target_price="", today=TODAY)
    assert "target_set_at" not in addsignal.get_plan(store, "SYSR.ST")


def test_proof_is_a_dated_confirmation(store):
    addsignal.set_plan(store, "SYSR.ST", proof_confirmed=True, today=TODAY)
    assert addsignal.get_plan(store, "SYSR.ST")["proof_confirmed_at"] == "2026-08-12"
    addsignal.set_plan(store, "SYSR.ST", proof_confirmed=False, today=TODAY)
    assert "proof_confirmed_at" not in addsignal.get_plan(store, "SYSR.ST")


# ── Staleness ─────────────────────────────────────────────────────────────────

def test_valuation_is_fresh_when_nothing_has_happened(store):
    addsignal.set_plan(store, "SYSR.ST", target_price=145, today=TODAY)
    assert addsignal.valuation_status(store, "SYSR.ST", today=TODAY)["status"] == "fresh"


def test_valuation_is_unset_without_a_target(store):
    assert addsignal.valuation_status(store, "SYSR.ST", today=TODAY)["status"] == "unset"


def test_an_earnings_report_since_the_target_makes_it_stale(store):
    """The failure this prevents: value halved, target didn't, price looks cheap."""
    addsignal.set_plan(store, "SYSR.ST", target_price=145, today=TODAY)
    store.set_meta("paper_earnhint:SYSR.ST:2026-08-20", "2026-08-20")
    status = addsignal.valuation_status(store, "SYSR.ST", today=TODAY)
    assert status["status"] == "stale"
    assert "1 thesis-changing event" in status["reason"]


def test_a_kill_hit_since_the_target_makes_it_stale(store):
    addsignal.set_plan(store, "SYSR.ST", target_price=145, today=TODAY)
    store.set_meta("paper_killhit:SYSR.ST:2026-09-01", "growth collapsed")
    assert addsignal.valuation_status(store, "SYSR.ST", today=TODAY)["status"] == "stale"


def test_an_event_before_the_target_does_not_make_it_stale(store):
    store.set_meta("paper_earnhint:SYSR.ST:2026-05-01", "2026-05-01")
    addsignal.set_plan(store, "SYSR.ST", target_price=145, today=TODAY)
    assert addsignal.valuation_status(store, "SYSR.ST", today=TODAY)["status"] == "fresh"


def test_another_tickers_event_is_not_confused_for_this_one(store):
    addsignal.set_plan(store, "SYSR.ST", target_price=145, today=TODAY)
    store.set_meta("paper_earnhint:EVO.ST:2026-09-01", "2026-09-01")
    assert addsignal.valuation_status(store, "SYSR.ST", today=TODAY)["status"] == "fresh"


# ── The state machine ─────────────────────────────────────────────────────────

def _setup(store, *, price, target=None, review=None, proof=False,
           book="A", max_weight=90, tranche=1.0, months=24):
    _hold(store, cost=100.0)
    _price(store, "SYSR.ST", price)
    _horizon(store, "SYSR.ST", months)
    addsignal.set_plan(store, "SYSR.ST", book=book, max_weight_pct=max_weight,
                       tranche_pct=tranche, target_price=target,
                       review_price=review, proof_confirmed=proof or None, today=TODAY)


def _state(store, **kw):
    rows = addsignal.evaluate_all(store, today=TODAY)
    return rows[0] if rows else None


def test_nothing_happening_is_hold(store):
    _setup(store, price=100, target=145, review=100)
    row = _state(store)
    assert row["state"] == addsignal.HOLD
    assert "no fundamental proof and no price dislocation" in row["why"]


def test_proof_with_clear_gates_is_add(store):
    _setup(store, price=100, target=145, review=100, proof=True)
    row = _state(store)
    assert row["state"] == addsignal.ADD
    assert row["expected_annual_return"] == pytest.approx(20.0, abs=1.0)


def test_dislocation_without_proof_is_only_add_watch(store):
    """A price fall alone is an opportunity to look, not a reason to buy."""
    _setup(store, price=85, target=145, review=100)      # −15%, book A gate −12%
    row = _state(store)
    assert row["state"] == addsignal.ADD_WATCH
    assert "without fundamental proof" in row["why"]


def test_proof_and_dislocation_together_is_strong_add(store):
    _setup(store, price=85, target=145, review=100, proof=True)
    assert _state(store)["state"] == addsignal.STRONG_ADD


def test_dislocation_is_measured_from_the_review_price_not_cost(store):
    """A winner down 15% from its review price is dislocated; from cost it isn't."""
    _hold(store, cost=100.0)
    _price(store, "SYSR.ST", 340)
    _horizon(store, "SYSR.ST")
    addsignal.set_plan(store, "SYSR.ST", book="A", max_weight_pct=90, tranche_pct=1,
                       target_price=500, review_price=400, today=TODAY)
    row = _state(store)
    assert row["dislocation_pct"] == pytest.approx(-15.0)
    assert row["state"] == addsignal.ADD_WATCH      # +240% on cost, still dislocated


def test_an_expensive_stock_with_good_news_is_not_an_add(store):
    """Proof alone must not buy something already priced for perfection."""
    _setup(store, price=140, target=145, review=100, proof=True)
    row = _state(store)
    assert row["state"] == addsignal.HOLD
    assert "below the 15% gate" in row["why"]


def test_the_book_sets_the_gate(store):
    """The same 4% expected return passes nowhere; B demands the most."""
    _setup(store, price=85, target=145, review=100, proof=True, book="B")
    row = _state(store)
    assert row["dislocation_gate"] == -20.0          # −15% is not enough for book B
    assert row["return_gate"] == 25.0
    assert row["state"] == addsignal.ADD             # proof still carries it


def test_a_stale_valuation_suppresses_the_add(store):
    _setup(store, price=85, target=145, review=100, proof=True)
    store.set_meta("paper_earnhint:SYSR.ST:2026-08-30", "2026-08-30")
    row = _state(store)
    assert row["state"] == addsignal.ADD_WATCH
    assert "stale" in row["why"] and "recalculate fair value" in row["why"]


def test_no_target_price_cannot_produce_an_add(store):
    _setup(store, price=85, target=None, review=100, proof=True)
    row = _state(store)
    assert row["state"] == addsignal.ADD_WATCH
    assert "no target review price" in row["why"]


def test_an_expired_horizon_cannot_produce_an_add(store):
    """Without months remaining the annualised return is undefined, not infinite."""
    _hold(store, cost=100.0)
    _price(store, "SYSR.ST", 100)
    watchplan.set_position_plan(store, "SYSR.ST", review_date="2026-01-01", today=TODAY)
    addsignal.set_plan(store, "SYSR.ST", book="A", target_price=145,
                       review_price=100, proof_confirmed=True, today=TODAY)
    row = _state(store)
    assert row["expected_annual_return"] is None
    assert row["state"] == addsignal.HOLD


def test_the_weight_gate_blocks_an_add_with_no_room(store):
    _hold(store, shares=500, cost=100.0)       # 50k of a 100k book → 50% weight
    _price(store, "SYSR.ST", 100)
    _horizon(store, "SYSR.ST")
    addsignal.set_plan(store, "SYSR.ST", book="A", max_weight_pct=50.5, tranche_pct=1,
                       target_price=145, review_price=100, proof_confirmed=True,
                       today=TODAY)
    row = _state(store)
    assert row["state"] == addsignal.HOLD
    assert "room below max weight" in row["why"]


def test_above_max_weight_is_a_concentration_review(store):
    _hold(store, shares=500, cost=100.0)
    _price(store, "SYSR.ST", 100)
    _horizon(store, "SYSR.ST")
    addsignal.set_plan(store, "SYSR.ST", book="A", max_weight_pct=10, tranche_pct=1,
                       target_price=145, review_price=100, proof_confirmed=True,
                       today=TODAY)
    row = _state(store)
    assert row["over_max_weight"] is True
    assert row["state"] == addsignal.HOLD
    assert "concentration review" in row["why"]


def test_kill_overrides_every_add_signal(store):
    """ADD and KILL must never coexist. Kill wins."""
    _setup(store, price=85, target=145, review=100, proof=True)
    # Reviewed at 100, now 85 — −15% against a −10% line.
    watchplan.set_position_plan(store, "SYSR.ST", max_drawdown_pct="10",
                                anchor_price_sek=100)
    row = _state(store)
    assert row["state"] == addsignal.KILL
    assert "suppressed" in row["why"]


def test_a_text_kill_verdict_also_overrides(store):
    import json as _json
    _setup(store, price=85, target=145, review=100, proof=True)
    watchplan.set_position_plan(store, "SYSR.ST", kill_criterion="growth collapses")
    store.set_meta("paper_killverdict:SYSR.ST",
                   _json.dumps({"verdict": "yes", "reason": "it collapsed"}))
    assert _state(store)["state"] == addsignal.KILL


# ── Proof expires with the report it was made about ───────────────────────────

def _reported(store, ticker="SYSR.ST", on="2026-08-20"):
    """The breadcrumb the post-earnings watch leaves when a print lands."""
    store.set_meta(f"paper_earnpost:{ticker}:{on}", on)


def test_proof_goes_stale_when_the_next_report_lands(store):
    """A confirmation is a reading of a report. Once the next one lands it is a
    statement about a quarter that has been superseded."""
    _setup(store, price=100, target=145, review=100, proof=True)
    assert addsignal.proof_status(store, "SYSR.ST")["status"] == "fresh"

    _reported(store)
    state = addsignal.proof_status(store, "SYSR.ST")
    assert state["status"] == "stale"
    assert "since proof was confirmed on 2026-08-12" in state["reason"]


def test_stale_proof_does_not_open_the_add_gate(store):
    """The failure this closes: an ADD resting on a judgement nobody has made
    about the current numbers."""
    _setup(store, price=100, target=145, review=100, proof=True)
    assert _state(store)["state"] == addsignal.ADD

    _reported(store)
    row = _state(store)
    assert row["state"] == addsignal.HOLD
    assert row["proof"] is False
    assert row["proof_status"] == "stale"
    assert "re-read the print" in row["why"]


def test_re_confirming_after_the_report_restores_proof_but_not_the_target(store):
    """Answering the proof question must not silently re-validate fair value.

    The same report that superseded the proof also superseded the target price,
    and they are separate judgements: "the thesis still holds" is not "and 145
    is still the right number". Confirming proof clears its own gate and leaves
    the valuation one asking to be recalculated.
    """
    _setup(store, price=100, target=145, review=100, proof=True)
    _reported(store, on="2026-08-20")
    addsignal.set_plan(store, "SYSR.ST", proof_confirmed=True, today=date(2026, 8, 21))

    row = _state(store)
    assert addsignal.proof_status(store, "SYSR.ST")["status"] == "fresh"
    assert row["proof"] is True
    assert row["state"] == addsignal.ADD_WATCH
    assert "valuation is stale" in row["why"]

    # Restating the target as well is what completes the review.
    addsignal.set_plan(store, "SYSR.ST", target_price=150, today=date(2026, 8, 21))
    assert _state(store)["state"] == addsignal.ADD


def test_unconfirmed_proof_is_unset_not_stale(store):
    _setup(store, price=100, target=145, review=100)
    _reported(store)
    row = _state(store)
    assert row["proof_status"] == "unset"
    assert "no fundamental proof" in row["why"]


def test_a_dislocated_name_still_moves_on_stale_proof(store):
    """Staleness removes the proof leg, not the dislocation one."""
    _setup(store, price=85, target=145, review=100, proof=True)
    _reported(store)
    row = _state(store)
    assert row["proof"] is False
    assert row["dislocated"] is True
    assert row["state"] == addsignal.ADD_WATCH      # dislocation alone, as designed


# ── The post-earnings proof prompt ────────────────────────────────────────────

def test_the_post_earnings_alert_asks_the_proof_question(store):
    _setup(store, price=100, target=145, review=100, proof=True)
    lines = paper._proof_prompt(store, "add-sleeve", "SYSR.ST")
    text = "\n".join(lines)
    assert "did this print confirm the thesis?" in text
    assert "/proof add-sleeve SYSR.ST yes" in text
    assert "this report supersedes it" in text


def test_the_prompt_flags_an_already_stale_proof(store):
    _setup(store, price=100, target=145, review=100, proof=True)
    _reported(store)
    text = "\n".join(paper._proof_prompt(store, "add-sleeve", "SYSR.ST"))
    assert "⚠" in text and "since proof was confirmed" in text


def test_no_add_plan_means_no_proof_question(store):
    """The prompt is for positions being scaled into, not every holding."""
    _hold(store)
    assert paper._proof_prompt(store, "add-sleeve", "SYSR.ST") == []


# ── The watch ─────────────────────────────────────────────────────────────────

def test_add_alerts_once_per_state_change(store, telegram):
    _setup(store, price=100, target=145, review=100, proof=True)
    log = addsignal.check_add_signals("slug", store=store, today=TODAY)
    assert any("ADD" in line for line in log)
    assert len(telegram) == 1
    assert "Suggested +1.0pp" in telegram[0]

    # Unchanged the next day: no repeat.
    assert addsignal.check_add_signals("slug", store=store, today=TODAY) == []
    assert len(telegram) == 1


def test_dropping_back_to_hold_is_logged_but_not_pushed(store, telegram):
    _setup(store, price=100, target=145, review=100, proof=True)
    addsignal.check_add_signals("slug", store=store, today=TODAY)
    addsignal.set_plan(store, "SYSR.ST", proof_confirmed=False, today=TODAY)
    log = addsignal.check_add_signals("slug", store=store, today=TODAY)
    assert any("add → hold" in line for line in log)
    assert len(telegram) == 1


def test_the_watch_is_silent_without_any_add_plan(store, telegram):
    _hold(store)
    assert addsignal.check_add_signals("slug", store=store, today=TODAY) == []
    assert telegram == []


def test_signals_are_ordered_most_actionable_first(store):
    for ticker, price in (("A.ST", 100), ("B.ST", 85), ("C.ST", 100)):
        _hold(store, ticker, 10, 100.0)
        _price(store, ticker, price)
        _horizon(store, ticker)
    store.set_meta("paper_currency_map", '{"A.ST":"SEK","B.ST":"SEK","C.ST":"SEK"}')
    addsignal.set_plan(store, "A.ST", book="A", target_price=145, review_price=100,
                       proof_confirmed=True, today=TODAY)                     # add
    addsignal.set_plan(store, "B.ST", book="A", target_price=145, review_price=100,
                       proof_confirmed=True, today=TODAY)                     # strong
    addsignal.set_plan(store, "C.ST", book="A", target_price=145, review_price=100,
                       today=TODAY)                                           # hold
    states = [r["state"] for r in addsignal.evaluate_all(store, today=TODAY)]
    assert states == [addsignal.STRONG_ADD, addsignal.ADD, addsignal.HOLD]


def test_track_portfolio_runs_the_add_watch(tmp_path, monkeypatch, telegram):
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path / "paper")
    monkeypatch.setattr(paper, "detect_currency", lambda t: "SEK")
    monkeypatch.setattr(paper, "_cache_price_history", lambda store, tickers, **kw: None)
    monkeypatch.setattr(paper, "_search_symbol", lambda name: None)
    import fundmgr.data.benchmark as benchmark
    monkeypatch.setattr(benchmark, "fetch_and_cache_benchmark", lambda store, **kw: True)
    import fundmgr.data.quotes as quotes
    monkeypatch.setattr(quotes, "live_prices", lambda tickers: {t: 100.0 for t in tickers})

    slug, _ = paper.create_portfolio(
        "Tracked Adds", 100_000, "",
        holdings_override=[{"ticker": "SYSR.ST", "name": "Systemair", "weight_pct": 100,
                            "shares": 100, "avg_cost_sek": 100.0}],
        kind="live", seed_holdings=True)
    _meta, store = paper.open_portfolio(slug)
    _price(store, "SYSR.ST", 100)
    _horizon(store, "SYSR.ST")
    addsignal.set_plan(store, "SYSR.ST", book="A", max_weight_pct=13, tranche_pct=1,
                       target_price=145, review_price=100, proof_confirmed=True,
                       today=TODAY)

    log = paper.track_portfolio(slug)
    assert any("ADD" in line for line in log)


def test_a_broken_add_watch_does_not_stop_the_track(tmp_path, monkeypatch, telegram):
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path / "paper")
    monkeypatch.setattr(paper, "detect_currency", lambda t: "SEK")
    monkeypatch.setattr(paper, "_cache_price_history", lambda store, tickers, **kw: None)
    monkeypatch.setattr(paper, "_search_symbol", lambda name: None)
    import fundmgr.data.benchmark as benchmark
    monkeypatch.setattr(benchmark, "fetch_and_cache_benchmark", lambda store, **kw: True)
    import fundmgr.data.quotes as quotes
    monkeypatch.setattr(quotes, "live_prices", lambda tickers: {t: 100.0 for t in tickers})

    slug, _ = paper.create_portfolio(
        "Fragile Adds", 100_000, "",
        holdings_override=[{"ticker": "SYSR.ST", "name": "Systemair", "weight_pct": 100,
                            "shares": 10, "avg_cost_sek": 100.0}],
        kind="live", seed_holdings=True)

    def boom(*a, **kw):
        raise RuntimeError("evaluator exploded")
    monkeypatch.setattr(addsignal, "check_add_signals", boom)

    log = paper.track_portfolio(slug)
    assert any("add-signal watch failed" in line for line in log)
    assert any("NAV" in line for line in log)
    assert datetime and timezone   # imports used by the price helper above


# ── The dashboard panel ───────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    """A live sleeve with one holding, served through the real app."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path / "paper")
    monkeypatch.setattr(paper, "detect_currency", lambda t: "SEK")
    monkeypatch.setattr(paper, "_cache_price_history", lambda store, tickers, **kw: None)
    monkeypatch.setattr(paper, "_search_symbol", lambda name: None)
    import fundmgr.data.benchmark as benchmark
    monkeypatch.setattr(benchmark, "fetch_and_cache_benchmark", lambda store, **kw: True)
    import fundmgr.data.quotes as quotes
    monkeypatch.setattr(quotes, "live_prices", lambda tickers: {t: 100.0 for t in tickers})
    monkeypatch.setattr(quotes, "live_price", lambda ticker: 100.0)

    from fundmgr.web.app import app
    paper.create_portfolio(
        "Add Sleeve", 30_000, "",
        holdings_override=[{"ticker": "SYSR.ST", "name": "Systemair", "weight_pct": 100,
                            "shares": 100, "avg_cost_sek": 100.0}],
        kind="live", seed_holdings=True)
    return TestClient(app)


def test_the_panel_is_absent_until_a_position_carries_a_plan(client):
    """No plan, no panel — an empty table would be noise on every sleeve."""
    assert "Add signals" not in client.get("/live/add-sleeve").text


def test_the_panel_shows_the_state_and_which_gate_stopped_it(client):
    client.post("/live/add-sleeve/addplan",
                data={"ticker": "SYSR.ST", "book": "A", "max_weight_pct": "100",
                      "tranche_pct": "1", "review_price": "100"},
                follow_redirects=False)

    body = client.get("/live/add-sleeve").text
    assert "Add signals" in body
    assert "SYSR.ST" in body
    assert "HOLD" in body
    # The engine's reason, rendered — the thing that was CLI-only before.
    assert "no fundamental proof and no price dislocation" in body


def test_confirming_proof_from_the_panel_moves_the_state(client):
    client.post("/live/add-sleeve/addplan",
                data={"ticker": "SYSR.ST", "book": "A", "max_weight_pct": "100",
                      "tranche_pct": "1", "target_price": "160", "review_price": "100"},
                follow_redirects=False)
    r = client.post("/live/add-sleeve/addplan",
                    data={"ticker": "SYSR.ST", "proof": "yes"}, follow_redirects=False)
    assert r.status_code == 303 and "ok=1" in r.headers["location"]

    _meta, store = paper.open_portfolio("add-sleeve")
    assert addsignal.get_plan(store, "SYSR.ST")["proof_confirmed_at"]


def test_proof_is_its_own_submit_not_a_side_effect_of_saving(client):
    """Saving an unrelated field must not stamp a dated confirmation."""
    client.post("/live/add-sleeve/addplan",
                data={"ticker": "SYSR.ST", "book": "A", "tranche_pct": "1"},
                follow_redirects=False)
    _meta, store = paper.open_portfolio("add-sleeve")
    assert "proof_confirmed_at" not in addsignal.get_plan(store, "SYSR.ST")


def test_anchoring_at_todays_price_uses_the_cached_price(client):
    client.post("/live/add-sleeve/addplan",
                data={"ticker": "SYSR.ST", "book": "A", "anchor_live": "1"},
                follow_redirects=False)
    _meta, store = paper.open_portfolio("add-sleeve")
    plan = addsignal.get_plan(store, "SYSR.ST")
    assert plan["review_price"] == 100.0
    assert plan["review_date"]


def test_clearing_removes_the_plan_and_the_panel(client):
    client.post("/live/add-sleeve/addplan",
                data={"ticker": "SYSR.ST", "book": "A", "tranche_pct": "1"},
                follow_redirects=False)
    client.post("/live/add-sleeve/addplan",
                data={"ticker": "SYSR.ST", "remove": "1"}, follow_redirects=False)

    _meta, store = paper.open_portfolio("add-sleeve")
    assert addsignal.get_plan(store, "SYSR.ST") == {}
    assert "Add signals" not in client.get("/live/add-sleeve").text


def test_a_stale_proof_is_visible_on_the_panel(client):
    """The state the panel exists for: confirmed, but for a superseded quarter."""
    client.post("/live/add-sleeve/addplan",
                data={"ticker": "SYSR.ST", "book": "A", "tranche_pct": "1",
                      "target_price": "160", "review_price": "100"},
                follow_redirects=False)
    client.post("/live/add-sleeve/addplan",
                data={"ticker": "SYSR.ST", "proof": "yes"}, follow_redirects=False)

    _meta, store = paper.open_portfolio("add-sleeve")
    store.set_meta("paper_earnpost:SYSR.ST:2099-01-01", "2099-01-01")

    body = client.get("/live/add-sleeve").text
    assert "stale" in body
    assert "re-read the print" in body
