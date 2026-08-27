"""Live-sleeve review: scope filtering, funding from sells, add-on detection,
persistence into the sleeve's own store, and the web job routes.

The LLM and every market fetch are stubbed — nothing here touches the network.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from fundmgr import paper
from fundmgr.engine import sleeve_review, whatif
from fundmgr.engine.schema import Action, DecisionRun
from fundmgr.state.models import Position, Transaction
from fundmgr.state.store import Store

UNIVERSE_CSV = """name,yahoo_ticker,isin,country,exchange,sector,enabled
Alfa AB,ALFA.ST,SE0001,SE,OMXS,Industrials,true
Beta AB,BETA.ST,SE0002,SE,OMXS,Technology,true
Gamma AB,GAMMA.ST,SE0003,SE,OMXS,Healthcare,true
Norsk ASA,NORSK.OL,NO0001,NO,OSE,Energy,true
Deutsche AG,DEUT.DE,DE0001,DE,XETRA,Technology,true
Disabled AB,DEAD.ST,SE0009,SE,OMXS,Industrials,false
Unknown Co,WEIRD,,US1234567890,NASDAQ,Unknown,true
"""

MANDATE = "You are a test fund manager. Buy good things."


def _price_rows(n: int = 260, start: float = 100.0) -> list[dict]:
    """Daily closes ending today, so features are fresh (not stale)."""
    today = datetime.utcnow().date()
    rows = []
    for i in range(n):
        px = start + i * 0.1
        rows.append({
            "date": (today - timedelta(days=n - 1 - i)).strftime("%Y-%m-%d"),
            "open": px, "high": px * 1.01, "low": px * 0.99,
            "close": px, "volume": 10_000 + i,
        })
    return rows


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A source profile on disk plus an isolated PAPER_DIR, with prices, FX,
    macro and benchmark stubbed out."""
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    config_dir.mkdir()
    data_dir.mkdir()

    (config_dir / "universe_test.csv").write_text(UNIVERSE_CSV)
    (config_dir / "mandate_test.md").write_text(MANDATE)
    db_path = data_dir / "fund_test.db"
    (config_dir / "config_test.yaml").write_text(f"""
name: "🧪 Test Fund"
capital_sek: 100000
benchmark: "^OMXSPI"
db_path: "{db_path}"
mandate_path: "{config_dir / 'mandate_test.md'}"
universe_path: "{config_dir / 'universe_test.csv'}"
llm:
  provider: openai
  model_id: "gpt-5.6-sol"
  n_samples: 1
risk:
  max_position_pct: 40
  max_positions: 5
  max_sector_pct: 90
  min_cash_pct: 5
  max_cash_pct: 10
  min_trade_sek: 1000
  max_turnover_pct: 100
  stale_after_days: 5
screener:
  top_n: 10
data:
  news_feeds:
    - "https://example.invalid/news.rss"
  sentiment:
    enabled: true
""")

    source = Store(db_path)
    source.initialise(100000)
    source.save_prices("BETA.ST", _price_rows())  # the one "warm" candidate

    monkeypatch.setattr(whatif, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(sleeve_review, "DEFAULT_REVIEW_CONFIG", "config_test.yaml")
    sleeve_review.list_scopes.cache_clear()

    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path / "paper")
    monkeypatch.setattr(paper, "detect_currency", lambda t: "SEK")
    monkeypatch.setattr(paper, "_search_symbol", lambda name: None)
    monkeypatch.setattr(paper, "_cache_price_history", lambda store, tickers, **k: None)

    import fundmgr.data.benchmark as benchmark
    import fundmgr.data.fx as fx
    import fundmgr.data.quotes as quotes
    monkeypatch.setattr(fx, "rate_to_sek", lambda cur, store=None: 1.0)
    monkeypatch.setattr(quotes, "live_prices", lambda tickers: {t: 100.0 for t in tickers})
    monkeypatch.setattr(benchmark, "fetch_and_cache_benchmark",
                        lambda store, symbol="URTH", **k: True)

    # Every ticker "fetches" successfully into the sleeve's own cache.
    def fake_fetch(tickers, store, lookback_days=252, force_refresh=False):
        for t in tickers:
            store.save_prices(t.yahoo_ticker, _price_rows())
        return {t.yahoo_ticker: True for t in tickers}

    monkeypatch.setattr(sleeve_review, "fetch_and_cache_prices", fake_fetch)
    # News is stubbed off by default so no test can reach a feed; the tests that
    # care about news caching re-patch these two with their own doubles.
    monkeypatch.setattr(sleeve_review, "fetch_news", lambda *a, **k: {})
    monkeypatch.setattr(sleeve_review, "score_and_cache_sentiment", lambda *a, **k: None)
    yield {"config_dir": config_dir, "source": source}
    sleeve_review.list_scopes.cache_clear()


@pytest.fixture
def sleeve(env):
    """A live sleeve holding 100 shares of ALFA.ST at 100 SEK, plus cash."""
    slug, _ = paper.create_portfolio(
        name="Test Sleeve", capital_sek=100_000.0,
        holdings_text="ALFA.ST 50%\nGAMMA.ST 50%",
        holdings_override=[
            {"name": "Alfa AB", "ticker": "ALFA.ST", "weight_pct": 50.0,
             "thesis": "core", "confidence": 0.8, "kill_criterion": "margins collapse"},
            {"name": "Gamma AB", "ticker": "GAMMA.ST", "weight_pct": 50.0,
             "thesis": "second", "confidence": 0.7, "kill_criterion": ""},
        ],
        kind="live", execute_buys=False, model_label="Test Model",
    )
    _meta, store = paper.open_portfolio(slug)
    store.apply_fill(Transaction(
        ticker="ALFA.ST", side="buy", shares=100, price_sek=100.0, fee_sek=0.0,
        source="paper", currency="SEK", timestamp=datetime.now(timezone.utc),
    ))
    return slug


def _stub_llm(monkeypatch, actions: list[Action], capture: dict | None = None):
    def _fake(system, user, cfg):
        if capture is not None:
            capture.update(system=system, user=user, cfg=cfg)
        decision = DecisionRun(
            run_id="stub", market_summary="Stubbed read.", actions=actions,
            cash_target_pct=8.0, notes="stub notes",
        )
        return decision, decision.model_dump_json(), None, {
            "requested": 1, "succeeded": 1, "failed": 0, "errors": []}
    monkeypatch.setattr(sleeve_review, "call_llm_consensus", _fake)


# ── Scope ─────────────────────────────────────────────────────────────────────

def test_list_scopes_counts_enabled_tickers_per_country(env):
    scopes = {s["code"]: s for s in sleeve_review.list_scopes("config_test.yaml")}
    assert scopes["SE"]["count"] == 3          # DEAD.ST is disabled, so excluded
    assert scopes["SE"]["label"] == "Sweden"
    assert scopes["NO"]["count"] == 1
    assert scopes["DE"]["count"] == 1


def test_list_scopes_drops_malformed_country_codes(env):
    """A row carrying an ISIN in the country column must not become a scope."""
    codes = {s["code"] for s in sleeve_review.list_scopes("config_test.yaml")}
    assert codes == {"SE", "NO", "DE"}
    assert not any(len(c) != 2 for c in codes)


def test_country_scope_limits_the_candidate_universe(env, sleeve, monkeypatch):
    capture = {}
    _stub_llm(monkeypatch, [Action(ticker="ALFA.ST", side="hold", target_weight_pct=50,
                                   sek_estimate=0, confidence=0.6, thesis="hold")], capture)
    result = sleeve_review.review_sleeve(sleeve, country="NO", include_macro=False)

    assert result["scope"]["country"] == "NO"
    assert "Norway only" in result["scope"]["label"]
    # Only the Norwegian name is a candidate; the Swedish ones stay out of the
    # prompt except the sleeve's own holdings, which always travel with it.
    assert "NORSK.OL" in capture["user"]
    assert "BETA.ST" not in capture["user"]
    assert "ALFA.ST" in capture["user"]


def test_global_scope_sees_every_country(env, sleeve, monkeypatch):
    capture = {}
    _stub_llm(monkeypatch, [Action(ticker="ALFA.ST", side="hold", target_weight_pct=50,
                                   sek_estimate=0, confidence=0.6, thesis="hold")], capture)
    result = sleeve_review.review_sleeve(sleeve, country=None, include_macro=False)

    assert result["scope"]["country"] == ""
    assert "global" in result["scope"]["label"]
    for ticker in ("NORSK.OL", "DEUT.DE", "BETA.ST"):
        assert ticker in capture["user"]


def test_unknown_country_is_rejected(env, sleeve, monkeypatch):
    _stub_llm(monkeypatch, [])
    with pytest.raises(ValueError, match="No enabled tickers for country 'JP'"):
        sleeve_review.review_sleeve(sleeve, country="JP", include_macro=False)


# ── Funding ───────────────────────────────────────────────────────────────────

def test_buy_is_funded_by_the_sell_in_the_same_run(env, sleeve, monkeypatch):
    """A fully-deployed sleeve has no spare cash. Selling the position must free
    the money for the add-on, or the cash-floor guardrail rejects every buy."""
    _meta, store = paper.open_portfolio(sleeve)
    store.set_cash(0.0)  # everything is in ALFA.ST

    _stub_llm(monkeypatch, [
        Action(ticker="ALFA.ST", side="sell", target_weight_pct=50,
               sek_estimate=5_000, confidence=0.9, thesis="trim, thesis half broke"),
        Action(ticker="BETA.ST", side="buy", target_weight_pct=30,
               sek_estimate=3_000, confidence=0.8, thesis="better use of the money"),
    ])
    result = sleeve_review.review_sleeve(sleeve, include_macro=False)

    by_ticker = {a["ticker"]: a for a in result["actions"]}
    assert by_ticker["ALFA.ST"]["approved"] is True, (
        by_ticker["ALFA.ST"]["status"], by_ticker["ALFA.ST"]["reason"])
    assert by_ticker["BETA.ST"]["approved"] is True, by_ticker["BETA.ST"]["reason"]
    assert result["cash_sek"] == 0
    assert result["funded_cash_sek"] > 3_000


def test_unfunded_buy_still_fails_the_cash_floor(env, sleeve, monkeypatch):
    """Funding from sells must not become a blanket exemption: a buy with no
    sell behind it is rejected exactly as before."""
    _meta, store = paper.open_portfolio(sleeve)
    store.set_cash(0.0)

    _stub_llm(monkeypatch, [
        Action(ticker="BETA.ST", side="buy", target_weight_pct=30,
               sek_estimate=9_000, confidence=0.8, thesis="unfunded"),
    ])
    result = sleeve_review.review_sleeve(sleeve, include_macro=False)

    beta = next(a for a in result["actions"] if a["ticker"] == "BETA.ST")
    assert beta["approved"] is False
    assert "cash floor" in beta["reason"]


def test_sell_proceeds_never_exceed_the_position(env, sleeve, monkeypatch):
    """An oversized sek_estimate must not conjure cash the book cannot produce."""
    _meta, store = paper.open_portfolio(sleeve)
    store.set_cash(0.0)

    _stub_llm(monkeypatch, [
        Action(ticker="ALFA.ST", side="sell", target_weight_pct=0,
               sek_estimate=999_999, confidence=0.9, thesis="exit"),
    ])
    result = sleeve_review.review_sleeve(sleeve, include_macro=False)
    # 100 shares at 100 SEK = 10,000 gross, less the fee — not 999,999.
    assert 9_000 < result["funded_cash_sek"] <= 10_000


def test_buy_is_dropped_when_the_turnover_cap_takes_its_funding_sell(env, sleeve, monkeypatch):
    """The cap drops trades lowest-confidence-first, and can take the very sell
    that was paying for a buy it keeps. The buy must not survive as unfunded."""
    from fundmgr.guardrails.rules import GuardrailResult, GuardrailVerdict
    from fundmgr.state.models import PortfolioSnapshot

    cfg = whatif.load_profile_config("config_test.yaml")
    snap = PortfolioSnapshot(
        positions=[Position(ticker="ALFA.ST", shares=100, avg_cost_sek=100.0,
                            current_price_sek=100.0)],
        cash_sek=0.0,
    )
    buy = Action(ticker="BETA.ST", side="buy", target_weight_pct=30,
                 sek_estimate=3_000, confidence=0.8, thesis="needs the sell")
    # The sell the model proposed has already been dropped by the cap, so only
    # the buy reaches this pass.
    guardrails = GuardrailResult(
        verdicts=[GuardrailVerdict(action=buy, approved=True)],
        approved_actions=[buy],
    )
    dropped = sleeve_review._drop_unfunded_buys(guardrails, snap, cfg)

    assert dropped == {"BETA.ST"}
    assert guardrails.approved_actions == []


def test_several_buys_cannot_together_overdraw_the_book(env):
    """Guardrails check each buy against cash independently; the funding pass
    walks a running balance so two individually affordable buys can't both land."""
    from fundmgr.guardrails.rules import GuardrailResult, GuardrailVerdict
    from fundmgr.state.models import PortfolioSnapshot

    cfg = whatif.load_profile_config("config_test.yaml")
    snap = PortfolioSnapshot(positions=[], cash_sek=10_000.0)
    strong = Action(ticker="BETA.ST", side="buy", target_weight_pct=30,
                    sek_estimate=8_000, confidence=0.9, thesis="best idea")
    weak = Action(ticker="GAMMA.ST", side="buy", target_weight_pct=30,
                  sek_estimate=8_000, confidence=0.5, thesis="also nice")
    guardrails = GuardrailResult(
        verdicts=[GuardrailVerdict(action=a, approved=True) for a in (strong, weak)],
        approved_actions=[strong, weak],
    )
    dropped = sleeve_review._drop_unfunded_buys(guardrails, snap, cfg)

    assert dropped == {"GAMMA.ST"}                                # lower confidence loses
    assert [a.ticker for a in guardrails.approved_actions] == ["BETA.ST"]


def test_nav_is_unchanged_by_the_funding_adjustment(env, sleeve, monkeypatch):
    """Sells move value from positions to cash; they must not inflate NAV, or
    every weight and the turnover cap would be computed against a bigger book."""
    from fundmgr.state.models import PortfolioSnapshot

    cfg = whatif.load_profile_config("config_test.yaml")
    snap = PortfolioSnapshot(
        positions=[Position(ticker="ALFA.ST", shares=100, avg_cost_sek=100.0,
                            current_price_sek=100.0)],
        cash_sek=0.0,
    )
    decision = DecisionRun(
        run_id="x", market_summary="m", cash_target_pct=5,
        actions=[Action(ticker="ALFA.ST", side="sell", target_weight_pct=0,
                        sek_estimate=10_000, confidence=0.9, thesis="exit")],
    )
    funded = sleeve_review._fund_from_sells(snap, decision, cfg)
    assert funded.nav_sek == pytest.approx(snap.nav_sek - cfg.fees.calc(10_000))
    assert funded.positions == []


# ── Decisions ─────────────────────────────────────────────────────────────────

def test_add_on_is_flagged_and_counted(env, sleeve, monkeypatch):
    _stub_llm(monkeypatch, [
        Action(ticker="ALFA.ST", side="hold", target_weight_pct=50,
               sek_estimate=0, confidence=0.6, thesis="keep"),
        Action(ticker="BETA.ST", side="buy", target_weight_pct=20,
               sek_estimate=5_000, confidence=0.8, thesis="new name"),
    ])
    result = sleeve_review.review_sleeve(sleeve, include_macro=False)

    by_ticker = {a["ticker"]: a for a in result["actions"]}
    assert by_ticker["BETA.ST"]["add_on"] is True      # never in the plan
    assert by_ticker["ALFA.ST"]["add_on"] is False     # held
    assert result["add_on_count"] == 1
    assert result["hold_count"] == 1


def test_planned_but_unfilled_name_is_not_an_add_on(env, sleeve, monkeypatch):
    """GAMMA.ST is in the imported plan but never filled. Buying it is executing
    the original plan, not a new idea, and must not be sold as an add-on."""
    _stub_llm(monkeypatch, [
        Action(ticker="GAMMA.ST", side="buy", target_weight_pct=20,
               sek_estimate=5_000, confidence=0.8, thesis="fill the plan"),
    ])
    result = sleeve_review.review_sleeve(sleeve, include_macro=False)

    gamma = next(a for a in result["actions"] if a["ticker"] == "GAMMA.ST")
    assert gamma["add_on"] is False
    assert result["add_on_count"] == 0


def test_held_ticker_outside_the_scope_is_still_reviewable(env, sleeve, monkeypatch):
    """A Norway-scoped review of a sleeve holding a Swedish name must still be
    able to sell it — a position you can't see is a position you can't exit."""
    _stub_llm(monkeypatch, [
        Action(ticker="ALFA.ST", side="sell", target_weight_pct=0,
               sek_estimate=10_000, confidence=0.9, thesis="exit"),
    ])
    result = sleeve_review.review_sleeve(sleeve, country="NO", include_macro=False)

    alfa = next(a for a in result["actions"] if a["ticker"] == "ALFA.ST")
    assert alfa["approved"] is True, alfa["reason"]


def test_sleeve_context_carries_plan_kill_lines_and_no_new_capital(env, sleeve, monkeypatch):
    capture = {}
    _stub_llm(monkeypatch, [Action(ticker="ALFA.ST", side="hold", target_weight_pct=50,
                                   sek_estimate=0, confidence=0.6, thesis="hold")], capture)
    sleeve_review.review_sleeve(sleeve, include_macro=False)

    user = capture["user"]
    assert "This Sleeve — Test Sleeve" in user
    assert "margins collapse" in user            # the per-position kill criterion
    assert "There is NO new capital" in user
    assert "Live Sleeve Review — Test Sleeve" in user


def test_kill_signal_reaches_the_prompt(env, sleeve, monkeypatch):
    _meta, store = paper.open_portfolio(sleeve)
    store.set_meta("paper_killhit:ALFA.ST:2026-08-01", "Gross margin guided down 600bps")

    capture = {}
    _stub_llm(monkeypatch, [Action(ticker="ALFA.ST", side="hold", target_weight_pct=50,
                                   sek_estimate=0, confidence=0.6, thesis="hold")], capture)
    sleeve_review.review_sleeve(sleeve, include_macro=False)

    assert "KILL SIGNAL LOGGED" in capture["user"]
    assert "Gross margin guided down 600bps" in capture["user"]


def test_add_criterion_and_target_price_reach_the_prompt(env, sleeve, monkeypatch):
    """The operator's add-side work — add criterion, target price, gates — is
    the evidence the review needs to decide whether to add to a name it holds.
    Withholding it while asking for add recommendations is the bug this fixes."""
    from fundmgr import addsignal, watchplan

    _meta, store = paper.open_portfolio(sleeve)
    watchplan.set_add_text(store, "ALFA.ST", "ARR still compounding above 25%")
    addsignal.set_plan(store, "ALFA.ST", book="A/B", max_weight_pct=20,
                       tranche_pct=3, target_price=180, review_price=120)

    capture = {}
    _stub_llm(monkeypatch, [Action(ticker="ALFA.ST", side="hold", target_weight_pct=50,
                                   sek_estimate=0, confidence=0.6, thesis="hold")], capture)
    sleeve_review.review_sleeve(sleeve, include_macro=False)

    user = capture["user"]
    assert "ARR still compounding above 25%" in user     # the add criterion
    assert "target price: 180" in user
    assert "review price" in user
    assert "book A/B" in user
    assert "max weight 20.0%" in user


def test_numeric_kill_rules_reach_the_prompt(env, sleeve, monkeypatch):
    """The kill criterion's prose was already shown; the thresholds behind it
    were not, so the model saw 'margins collapse' but never '-20% drawdown'."""
    from fundmgr import watchplan

    _meta, store = paper.open_portfolio(sleeve)
    watchplan.set_position_plan(
        store, "ALFA.ST", max_drawdown_pct=20.0, price_below=75.0, currency="SEK",
        fundamentals=[{"metric": "gross_margin", "op": "below", "value": 45, "quarters": 2}],
    )

    capture = {}
    _stub_llm(monkeypatch, [Action(ticker="ALFA.ST", side="hold", target_weight_pct=50,
                                   sek_estimate=0, confidence=0.6, thesis="hold")], capture)
    sleeve_review.review_sleeve(sleeve, include_macro=False)

    user = capture["user"]
    assert "drawdown past 20%" in user
    assert "price below 75.00 SEK" in user
    assert "gross_margin below 45 for 2 consecutive quarters" in user


def test_the_operators_levels_are_evidence_not_instructions(env, sleeve, monkeypatch):
    """The model is told it may disagree — and told to justify it when it does.
    A binding gate would be a different product; this one argues."""
    capture = {}
    _stub_llm(monkeypatch, [Action(ticker="ALFA.ST", side="hold", target_weight_pct=50,
                                   sek_estimate=0, confidence=0.6, thesis="hold")], capture)
    sleeve_review.review_sleeve(sleeve, include_macro=False)

    user = capture["user"]
    assert "not instructions you must follow" in user.replace("\n", " ")
    assert "Where you disagree" in user
    assert "give your reason" in user
    # No language that turns a stored level into a hard constraint.
    for forbidden in ("you must not buy", "is prohibited", "never buy"):
        assert forbidden not in user.lower()


def test_a_stale_target_price_is_flagged_as_unevaluable(env, sleeve, monkeypatch):
    """A target set before a material event can't price the valuation gate. The
    model must see that as missing evidence, not as a failed gate — otherwise a
    stale number silently reads as 'too expensive'."""
    from fundmgr import addsignal

    from datetime import date

    _meta, store = paper.open_portfolio(sleeve)
    addsignal.set_plan(store, "ALFA.ST", target_price=180, review_price=120,
                       max_weight_pct=20, tranche_pct=3,
                       today=date.today() - timedelta(days=7))
    # An earnings print lands after the target was set — addsignal's staleness rule.
    store.set_meta(f"paper_earnhint:ALFA.ST:{date.today().isoformat()}",
                   "Q3 report published")

    capture = {}
    _stub_llm(monkeypatch, [Action(ticker="ALFA.ST", side="hold", target_weight_pct=50,
                                   sek_estimate=0, confidence=0.6, thesis="hold")], capture)
    sleeve_review.review_sleeve(sleeve, include_macro=False)

    assert "STALE" in capture["user"]
    assert "missing evidence" in capture["user"]


def test_review_survives_an_unusable_add_signal_layer(env, sleeve, monkeypatch):
    """Add signals reach for prices and fundamentals. If that fails the review
    is still worth having — it must degrade, not abort."""
    monkeypatch.setattr("fundmgr.addsignal.evaluate_all",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))

    capture = {}
    _stub_llm(monkeypatch, [Action(ticker="ALFA.ST", side="hold", target_weight_pct=50,
                                   sek_estimate=0, confidence=0.6, thesis="hold")], capture)
    result = sleeve_review.review_sleeve(sleeve, include_macro=False)

    assert result["hold_count"] == 1
    assert "margins collapse" in capture["user"]   # the kill criterion still lands


def test_sleeve_context_precedes_the_candidate_dump(env, sleeve, monkeypatch):
    """The book's own criteria belong with the portfolio state. Trailing them
    after dozens of candidate feature blocks buries the evidence the operator
    actually wrote."""
    capture = {}
    _stub_llm(monkeypatch, [Action(ticker="ALFA.ST", side="hold", target_weight_pct=50,
                                   sek_estimate=0, confidence=0.6, thesis="hold")], capture)
    sleeve_review.review_sleeve(sleeve, include_macro=False)

    user = capture["user"]
    assert user.index("## This Sleeve") < user.index("## Universe") < user.index("## Your Task")


def _seed_outcome(store, ticker, thesis, *, action="buy", verdict=None,
                  evidence="", when="2026-06-01"):
    """A matured decision on a name, optionally with a thesis-check verdict."""
    with store._conn() as conn:
        cur = conn.execute(
            "INSERT INTO decision_outcomes (run_id, ticker, action, confidence, "
            "thesis, decision_date, outperformed) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (f"seed-{ticker}-{when}", ticker, action, 0.8, thesis, when, 1),
        )
        outcome_id = cur.lastrowid
    if verdict:
        store.set_thesis_verdict(outcome_id, verdict, evidence)
    return outcome_id


def test_past_thesis_and_its_verdict_reach_the_prompt(env, sleeve, monkeypatch):
    """thesis_check judges the claim without seeing the return, which is what
    makes its verdict worth putting in front of the next decision on that name."""
    _meta, store = paper.open_portfolio(sleeve)
    _seed_outcome(store, "ALFA.ST", "Margins expand as the new plant ramps",
                  verdict="broke", evidence="Q2 guided gross margin down 600bps")

    capture = {}
    _stub_llm(monkeypatch, [Action(ticker="ALFA.ST", side="hold", target_weight_pct=50,
                                   sek_estimate=0, confidence=0.6, thesis="hold")], capture)
    sleeve_review.review_sleeve(sleeve, include_macro=False)

    user = capture["user"]
    assert "Margins expand as the new plant ramps" in user
    assert "thesis BROKE" in user
    assert "Q2 guided gross margin down 600bps" in user


def test_an_unaudited_past_thesis_shows_without_a_verdict(env, sleeve, monkeypatch):
    """No verdict yet is not the same as 'unresolved' — the claim still belongs
    in front of the model, but nothing may be asserted about how it fared."""
    _meta, store = paper.open_portfolio(sleeve)
    _seed_outcome(store, "ALFA.ST", "Order book converts within two quarters")

    capture = {}
    _stub_llm(monkeypatch, [Action(ticker="ALFA.ST", side="hold", target_weight_pct=50,
                                   sek_estimate=0, confidence=0.6, thesis="hold")], capture)
    sleeve_review.review_sleeve(sleeve, include_macro=False)

    user = capture["user"]
    assert "Order book converts within two quarters" in user
    assert "thesis HELD" not in user
    assert "thesis BROKE" not in user


def test_book_thesis_record_reaches_the_prompt(env, sleeve, monkeypatch):
    """The held-and-lagged / broke-and-beat split is the calibration a model
    judging only on returns can never see."""
    _meta, store = paper.open_portfolio(sleeve)
    _seed_outcome(store, "ALFA.ST", "t1", verdict="held", when="2026-05-01")
    _seed_outcome(store, "GAMMA.ST", "t2", verdict="broke", when="2026-05-02")

    capture = {}
    _stub_llm(monkeypatch, [Action(ticker="ALFA.ST", side="hold", target_weight_pct=50,
                                   sek_estimate=0, confidence=0.6, thesis="hold")], capture)
    sleeve_review.review_sleeve(sleeve, include_macro=False)

    user = capture["user"]
    assert "This book's own thesis record" in user
    assert "held 1 beat / 0 lagged" in user
    assert "broke 1 beat / 0 lagged" in user
    assert "Reasoning held on 50% of 2 resolved calls." in user


def test_thesis_record_is_omitted_when_nothing_has_been_audited(env, sleeve, monkeypatch):
    """An empty cross-tab is noise, and a 0% hold rate would read as a finding."""
    capture = {}
    _stub_llm(monkeypatch, [Action(ticker="ALFA.ST", side="hold", target_weight_pct=50,
                                   sek_estimate=0, confidence=0.6, thesis="hold")], capture)
    sleeve_review.review_sleeve(sleeve, include_macro=False)

    assert "This book's own thesis record" not in capture["user"]


def test_review_caches_news_for_held_names_so_the_audit_has_evidence(env, sleeve, monkeypatch):
    """thesis_check reads its evidence from the book's own news cache. A sleeve
    store is never filled by a weekly run, so without this the audit reports
    'no evidence' forever and no verdict is ever reached."""
    seen = {}

    def fake_fetch_news(feeds, tickers, **kwargs):
        seen["tickers"] = [t.yahoo_ticker for t in tickers]
        return {t.yahoo_ticker: [{"headline": "h"}] for t in tickers}

    monkeypatch.setattr(sleeve_review, "fetch_news", fake_fetch_news)
    monkeypatch.setattr(sleeve_review, "score_and_cache_sentiment",
                        lambda *a, **k: seen.setdefault("scored", True))

    _stub_llm(monkeypatch, [Action(ticker="ALFA.ST", side="hold", target_weight_pct=50,
                                   sek_estimate=0, confidence=0.6, thesis="hold")])
    sleeve_review.review_sleeve(sleeve, include_macro=False)

    # Held and planned names only — never the whole candidate pool.
    assert set(seen["tickers"]) == {"ALFA.ST", "GAMMA.ST"}
    assert seen.get("scored") is True


def test_review_survives_a_failing_news_fetch(env, sleeve, monkeypatch):
    monkeypatch.setattr(sleeve_review, "fetch_news",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("feed down")))
    _stub_llm(monkeypatch, [Action(ticker="ALFA.ST", side="hold", target_weight_pct=50,
                                   sek_estimate=0, confidence=0.6, thesis="hold")])
    assert sleeve_review.review_sleeve(sleeve, include_macro=False)["hold_count"] == 1


# ── Persistence ───────────────────────────────────────────────────────────────

def test_review_is_saved_as_the_sleeves_latest_decision(env, sleeve, monkeypatch):
    _stub_llm(monkeypatch, [
        Action(ticker="BETA.ST", side="buy", target_weight_pct=20,
               sek_estimate=5_000, confidence=0.8, thesis="new name"),
    ])
    result = sleeve_review.review_sleeve(sleeve, country="SE", include_macro=False)

    _meta, store = paper.open_portfolio(sleeve)
    rec = store.get_last_recommendation()
    assert rec.run_id == result["id"]
    assert rec.run_id.startswith(f"review-{sleeve}-")
    actions = json.loads(rec.actions_json)
    assert [a["ticker"] for a in actions] == ["BETA.ST"]
    # Scope is remembered so the next review and the form default to it.
    assert store.get_meta(sleeve_review.META_COUNTRY) == "SE"
    assert store.get_meta(sleeve_review.META_CONFIG) == "config_test.yaml"


def test_dry_run_writes_nothing(env, sleeve, monkeypatch):
    _meta, store = paper.open_portfolio(sleeve)
    before = store.get_last_recommendation().run_id

    _stub_llm(monkeypatch, [
        Action(ticker="BETA.ST", side="buy", target_weight_pct=20,
               sek_estimate=5_000, confidence=0.8, thesis="new name"),
    ])
    result = sleeve_review.review_sleeve(sleeve, country="SE", include_macro=False, dry_run=True)

    assert result["dry_run"] is True
    _meta, store = paper.open_portfolio(sleeve)
    assert store.get_last_recommendation().run_id == before
    assert store.get_meta(sleeve_review.META_COUNTRY) is None


def test_stored_scope_is_the_default_for_the_next_review(env, sleeve, monkeypatch):
    _meta, store = paper.open_portfolio(sleeve)
    store.set_meta(sleeve_review.META_COUNTRY, "NO")
    store.set_meta(sleeve_review.META_CONFIG, "config_test.yaml")

    capture = {}
    _stub_llm(monkeypatch, [Action(ticker="ALFA.ST", side="hold", target_weight_pct=50,
                                   sek_estimate=0, confidence=0.6, thesis="hold")], capture)
    result = sleeve_review.review_sleeve(sleeve, include_macro=False)

    assert result["scope"]["country"] == "NO"
    assert "NORSK.OL" in capture["user"]


def test_review_defaults_fall_back_when_the_stored_profile_is_gone(env, sleeve):
    _meta, store = paper.open_portfolio(sleeve)
    store.set_meta(sleeve_review.META_CONFIG, "config_deleted.yaml")
    assert sleeve_review.review_defaults(store)["config"] == "config_test.yaml"


# ── Web routes ────────────────────────────────────────────────────────────────

@pytest.fixture
def client(env):
    from fundmgr.web.app import app
    return TestClient(app)


def test_review_form_renders_on_the_sleeve_dashboard(client, sleeve):
    r = client.get(f"/live/{sleeve}")
    assert r.status_code == 200
    assert "Review this sleeve" in r.text
    assert "Global — every country" in r.text
    assert "Sweden (3)" in r.text


def test_review_form_is_absent_from_paper_books(client, env):
    slug, _ = paper.create_portfolio(
        name="Just Paper", capital_sek=10_000.0, holdings_text="ALFA.ST 100%",
        holdings_override=[{"name": "Alfa AB", "ticker": "ALFA.ST",
                            "weight_pct": 100.0, "thesis": "t", "confidence": 0.5}],
        kind="paper", execute_buys=False,
    )
    r = client.get(f"/paper/{slug}")
    assert r.status_code == 200
    assert "Review this sleeve" not in r.text


def test_scopes_endpoint_lists_countries_for_a_profile(client, sleeve):
    r = client.get(f"/live/{sleeve}/review/scopes", params={"config": "config_test.yaml"})
    assert r.status_code == 200
    assert {s["code"] for s in r.json()["scopes"]} == {"SE", "NO", "DE"}


def test_review_rejects_an_unknown_scope(client, sleeve):
    r = client.post(f"/live/{sleeve}/review", json={"config": "config_test.yaml", "country": "JP"})
    assert r.status_code == 400
    assert "Unknown scope" in r.json()["detail"]


def test_review_rejects_an_unknown_profile(client, sleeve):
    r = client.post(f"/live/{sleeve}/review", json={"config": "../secrets.yaml"})
    assert r.status_code == 400


def test_review_of_a_missing_sleeve_is_404(client, env):
    assert client.post("/live/nope/review", json={}).status_code == 404


def test_review_job_runs_and_reports_its_result(client, sleeve, monkeypatch):
    _stub_llm(monkeypatch, [
        Action(ticker="BETA.ST", side="buy", target_weight_pct=20,
               sek_estimate=5_000, confidence=0.8, thesis="new name"),
    ])
    start = client.post(f"/live/{sleeve}/review",
                        json={"config": "config_test.yaml", "country": "SE",
                              "include_macro": False})
    assert start.status_code == 200
    job_id = start.json()["job_id"]

    for _ in range(200):  # the worker is a thread; give it a moment
        job = client.get(f"/live/{sleeve}/review/jobs/{job_id}").json()
        if job["status"] != "running":
            break
    assert job["status"] == "done", job.get("error")
    assert job["result"]["add_on_count"] == 1
    assert job["result"]["scope"]["country"] == "SE"


def test_unknown_review_job_is_404(client, sleeve):
    assert client.get(f"/live/{sleeve}/review/jobs/nope").status_code == 404
