"""Paper portfolios: parsing pasted picks, creation at live prices, tracking, web routes."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from fundmgr import paper

FIXTURES = Path(__file__).parent / "fixtures"


# ── Fixtures: isolate PAPER_DIR + mock all market data ────────────────────────

FAKE_NATIVE = {"AAPL": 200.0, "MSFT": 500.0, "VOLV-B.ST": 300.0}
FAKE_FX = {"USD": 10.0, "SEK": 1.0}


@pytest.fixture
def paper_dir(tmp_path, monkeypatch):
    d = tmp_path / "paper"
    monkeypatch.setattr(paper, "PAPER_DIR", d)
    return d


@pytest.fixture
def mock_market(monkeypatch):
    import fundmgr.data.fx as fx
    import fundmgr.data.quotes as quotes

    monkeypatch.setattr(quotes, "live_prices", lambda tickers: {t: FAKE_NATIVE.get(t) for t in tickers})
    monkeypatch.setattr(fx, "rate_to_sek", lambda cur, store=None: FAKE_FX.get(cur.upper()))
    monkeypatch.setattr(
        paper, "detect_currency",
        lambda t: "SEK" if t.endswith(".ST") else "USD",
    )

    def fake_bench(store, symbol="URTH", **kwargs):
        today = datetime.utcnow().strftime("%Y-%m-%d")
        store.save_benchmark([{"date": today, "close": 100.0}])
        return True

    import fundmgr.data.benchmark as benchmark
    monkeypatch.setattr(benchmark, "fetch_and_cache_benchmark", fake_bench)

    def fake_history(store, tickers, lookback_days=40):
        today = datetime.utcnow().strftime("%Y-%m-%d")
        for t in tickers:
            if t in FAKE_NATIVE:
                store.save_prices(t, [{
                    "date": today, "open": FAKE_NATIVE[t], "high": FAKE_NATIVE[t],
                    "low": FAKE_NATIVE[t], "close": FAKE_NATIVE[t], "volume": 1000,
                }])

    monkeypatch.setattr(paper, "_cache_price_history", fake_history)
    monkeypatch.setattr(paper, "_search_symbol", lambda name: None)  # no network


# ── Parsing ───────────────────────────────────────────────────────────────────

def test_parse_json_array_with_fences():
    text = """Here is a diversified portfolio:
```json
[
  {"ticker": "AAPL", "weight_pct": 40, "thesis": "Services growth", "confidence": 0.8},
  {"symbol": "MSFT", "allocation": 35, "rationale": "Azure"},
  {"ticker": "VOLV-B.ST", "weight": 25}
]
```
Good luck!"""
    h = paper.parse_holdings(text)
    assert [x["ticker"] for x in h] == ["AAPL", "MSFT", "VOLV-B.ST"]
    assert h[0]["weight_pct"] == 40
    assert h[0]["thesis"] == "Services growth"
    assert h[0]["confidence"] == 0.8
    assert h[1]["weight_pct"] == 35
    assert h[1]["thesis"] == "Azure"


def test_parse_json_object_with_holdings_key():
    text = json.dumps({"holdings": [{"ticker": "AAPL", "weight_pct": 100}], "notes": "x"})
    h = paper.parse_holdings(text)
    assert h[0]["ticker"] == "AAPL"


def test_parse_json_ticker_weight_map():
    h = paper.parse_holdings('{"AAPL": 60, "MSFT": 40}')
    assert {x["ticker"]: x["weight_pct"] for x in h} == {"AAPL": 60, "MSFT": 40}


def test_parse_plain_lines():
    text = """# my picks
AAPL 40% — durable services growth
MSFT, 35%
VOLV-B.ST 25%
"""
    h = paper.parse_holdings(text)
    assert [x["ticker"] for x in h] == ["AAPL", "MSFT", "VOLV-B.ST"]
    assert h[0]["weight_pct"] == 40
    assert "durable services growth" in h[0]["thesis"]


def test_parse_lines_without_weights():
    h = paper.parse_holdings("AAPL\nMSFT\nVOLV-B.ST")
    assert len(h) == 3
    assert all(x["weight_pct"] is None for x in h)


def test_parse_dedupes_and_uppercases():
    h = paper.parse_holdings("aapl 50%\nAAPL 30%\nmsft 20%")
    assert [x["ticker"] for x in h] == ["AAPL", "MSFT"]


def test_parse_empty_raises():
    with pytest.raises(ValueError):
        paper.parse_holdings("   ")
    with pytest.raises(ValueError):
        paper.parse_holdings("Sorry, I cannot recommend individual stocks today.")


def test_parse_prose_with_company_names():
    """The real-world case: an LLM answer in prose with company names, cluster
    headers carrying percentages, theses containing '>20%' style figures, and
    'Alphabet 6% and Amazon 6%' on one line."""
    text = (FIXTURES / "fable_prose_portfolio.txt").read_text()
    h = paper.resolve_holdings(paper.parse_holdings(text), search=False)

    by_ticker = {x["ticker"]: x for x in h}
    assert set(by_ticker) == {
        "NVDA", "TSM", "ASML", "MU", "AVGO",           # AI compute cluster
        "GEV", "ABB.ST", "MTRS.ST",                    # power & grid
        "GOOGL", "AMZN",                               # hyperscalers
        "MYCR.ST", "ATCO-A.ST", "MILDEF.ST",           # Swedish chokepoints
    }
    # Weights survive; commentary lines ("Cluster 1 …, 30% at cost", the bear
    # case's "25% index sleeve", "Cash 3%", exposure math) produce no holdings
    assert by_ticker["NVDA"]["weight_pct"] == 8
    assert by_ticker["AMZN"]["weight_pct"] == 6
    assert sum(x["weight_pct"] for x in h) == pytest.approx(72)
    # Theses keep their text, and the inline "Kill: …" tail is split out as the
    # pre-registered kill criterion — '>20%' inside it doesn't chop the line
    assert "NVLink" in by_ticker["NVDA"]["thesis"]
    assert "custom silicon" in by_ticker["NVDA"]["kill_criterion"]
    assert "book-to-bill" in by_ticker["ABB.ST"]["kill_criterion"].lower()


def test_parse_prose_unresolved_names_kept_for_review():
    h = paper.resolve_holdings(paper.parse_holdings("- Some Obscure Company 10%\n- Nvidia 5%"),
                               search=False)
    unresolved = [x for x in h if x["ticker"] is None]
    assert len(unresolved) == 1
    assert unresolved[0]["name"] == "Some Obscure Company"
    assert unresolved[0]["weight_pct"] == 10


def test_resolve_via_symbol_search(monkeypatch):
    monkeypatch.setattr(paper, "_search_symbol", lambda name: "BUFAB.ST" if "Bufab" in name else None)
    h = paper.resolve_holdings(paper.parse_holdings("- Bufab 10%\n- Nvidia 5%"))
    by_ticker = {x["ticker"]: x for x in h}
    assert by_ticker["BUFAB.ST"]["resolved_via"] == "search"
    assert "NVDA" in by_ticker


def test_parse_index_sleeve_alias():
    h = paper.resolve_holdings(paper.parse_holdings("MSCI World 25%\nNvidia 8%"), search=False)
    assert {x["ticker"] for x in h} == {"URTH", "NVDA"}


def test_parse_json_with_broker_tickers_and_kill_criteria():
    """The structured format: broker-style tickers + exchange field ('ATCO A' on
    OMX), a FUND placeholder row, a CASH row, and kill_criterion per position."""
    text = (FIXTURES / "json_portfolio.json").read_text()
    h = paper.resolve_holdings(paper.parse_holdings(text), search=False)
    by_ticker = {x["ticker"]: x for x in h}

    # OMX tickers get the .ST suffix (bare 'MTRS' is a different NYSE company),
    # 'ATCO A' → ATCO-A.ST, ASML on AMS keeps its Amsterdam listing
    assert set(by_ticker) == {"NVDA", "ASML.AS", "ABB.ST", "MTRS.ST", "ATCO-A.ST", "URTH"}
    # the INDEX_GLOBAL fund placeholder resolves via its name to the ETF proxy
    assert by_ticker["URTH"]["weight_pct"] == 25
    # the CASH row produces no holding — its 3% simply stays uninvested
    assert sum(x["weight_pct"] for x in h) == pytest.approx(55)
    # kill criteria ride along per position
    assert "accelerator unit share" in by_ticker["NVDA"]["kill_criterion"]
    assert by_ticker["ATCO-A.ST"]["cluster"] == "swedish"


def test_json_ticker_normalisation_variants():
    assert paper._normalise_json_ticker("MYCR", "OMX", "Mycronic") == "MYCR.ST"
    assert paper._normalise_json_ticker("NVDA", "NASDAQ", "Nvidia") == "NVDA"
    assert paper._normalise_json_ticker("VOLV-B.ST", "OMX", "Volvo") == "VOLV-B.ST"  # already Yahoo-style
    assert paper._normalise_json_ticker("INDEX_GLOBAL", "FUND", "x") is None
    # no exchange hint: the name's home-listing alias wins over an ambiguous bare ticker
    assert paper._normalise_json_ticker("MTRS", "", "Munters") == "MTRS.ST"


# ── Weight normalisation ──────────────────────────────────────────────────────

def test_normalise_fractions_scaled_to_pct():
    h = paper.normalise_weights([
        {"ticker": "A", "weight_pct": 0.6, "thesis": "", "confidence": None},
        {"ticker": "B", "weight_pct": 0.4, "thesis": "", "confidence": None},
    ])
    assert h[0]["weight_pct"] == 60
    assert h[1]["weight_pct"] == 40


def test_normalise_missing_weights_share_remainder():
    h = paper.normalise_weights([
        {"ticker": "A", "weight_pct": 50, "thesis": "", "confidence": None},
        {"ticker": "B", "weight_pct": None, "thesis": "", "confidence": None},
        {"ticker": "C", "weight_pct": None, "thesis": "", "confidence": None},
    ])
    assert h[1]["weight_pct"] == 25
    assert h[2]["weight_pct"] == 25


def test_normalise_all_missing_equal_weight():
    h = paper.normalise_weights([
        {"ticker": t, "weight_pct": None, "thesis": "", "confidence": None}
        for t in ("A", "B", "C", "D")
    ])
    assert all(x["weight_pct"] == 25 for x in h)


def test_normalise_over_100_scaled_down():
    h = paper.normalise_weights([
        {"ticker": "A", "weight_pct": 80, "thesis": "", "confidence": None},
        {"ticker": "B", "weight_pct": 80, "thesis": "", "confidence": None},
    ])
    assert sum(x["weight_pct"] for x in h) == pytest.approx(100)


# ── Creation ──────────────────────────────────────────────────────────────────

PICKS = """```json
[
  {"ticker": "AAPL", "weight_pct": 40, "thesis": "Services", "confidence": 0.8},
  {"ticker": "MSFT", "weight_pct": 30, "thesis": "Azure"},
  {"ticker": "VOLV-B.ST", "weight_pct": 20, "thesis": "Trucks"}
]
```"""


def test_create_portfolio_executes_buys(paper_dir, mock_market):
    slug, log = paper.create_portfolio(
        name="Fable Picks", capital_sek=100_000, holdings_text=PICKS,
        base_prompt="Pick me 3 great stocks.", model_label="Fable",
    )
    assert slug == "fable-picks"
    meta, store = paper.open_portfolio(slug)

    positions = {p.ticker: p for p in store.get_positions()}
    assert set(positions) == {"AAPL", "MSFT", "VOLV-B.ST"}

    # USD prices land in SEK cost basis (200 USD × 10 = 2000 SEK/share)
    assert positions["AAPL"].avg_cost_sek == pytest.approx(2000, rel=1e-3)
    # 40% of 100k minus fee ≈ 39,960 SEK → ~19.98 shares
    assert positions["AAPL"].shares == pytest.approx(19.98, rel=1e-2)
    # 10% uninvested stays in cash
    assert store.get_cash() == pytest.approx(10_000, rel=1e-2)

    # The creation is logged as a decision run with seeded outcomes
    rec = store.get_last_recommendation()
    assert rec is not None
    actions = json.loads(rec.actions_json)
    assert {a["ticker"] for a in actions} == {"AAPL", "MSFT", "VOLV-B.ST"}
    assert all(a["side"] == "buy" for a in actions)
    with store._conn() as conn:
        rows = conn.execute("SELECT ticker, price_at_decision FROM decision_outcomes").fetchall()
    assert {r["ticker"]: r["price_at_decision"] for r in rows}["AAPL"] == pytest.approx(200)

    # NAV point recorded, metadata retrievable
    assert len(store.get_nav_history()) == 1
    assert meta["name"] == "Fable Picks"
    assert meta["model_label"] == "Fable"
    assert meta["base_prompt"] == "Pick me 3 great stocks."
    assert meta["currency_map"]["AAPL"] == "USD"
    assert meta["currency_map"]["VOLV-B.ST"] == "SEK"


def test_create_skips_unpriced_ticker(paper_dir, mock_market):
    text = "AAPL 50%\nNOSUCHTICKER 50%"
    slug, log = paper.create_portfolio("Partial", 50_000, text)
    _, store = paper.open_portfolio(slug)
    assert [p.ticker for p in store.get_positions()] == ["AAPL"]
    assert any("NOSUCHTICKER" in line for line in log)
    # unpriced allocation stays in cash
    assert store.get_cash() == pytest.approx(25_000, rel=1e-2)


def test_create_all_unpriced_raises_and_cleans_up(paper_dir, mock_market):
    with pytest.raises(ValueError):
        paper.create_portfolio("Broken", 50_000, "NOSUCHTICKER 100%")
    assert not (paper_dir / "broken.db").exists()
    assert paper.list_portfolios() == []


def test_create_duplicate_name_raises(paper_dir, mock_market):
    paper.create_portfolio("Twice", 50_000, "AAPL 100%")
    with pytest.raises(ValueError, match="already exists"):
        paper.create_portfolio("Twice", 50_000, "MSFT 100%")


def test_create_rejects_bad_input(paper_dir, mock_market):
    with pytest.raises(ValueError):
        paper.create_portfolio("", 50_000, "AAPL 100%")
    with pytest.raises(ValueError):
        paper.create_portfolio("X", 0, "AAPL 100%")


def test_create_from_company_name_prose(paper_dir, mock_market):
    slug, _ = paper.create_portfolio(
        "Prose", 100_000, "- Apple 60% — ecosystem\n- Microsoft 40% — Azure")
    _, store = paper.open_portfolio(slug)
    assert {p.ticker for p in store.get_positions()} == {"AAPL", "MSFT"}


def test_create_logs_unresolved_names(paper_dir, mock_market):
    slug, log = paper.create_portfolio(
        "Partial Resolve", 100_000, "- Apple 50%\n- Some Unknown Industrials 25%")
    _, store = paper.open_portfolio(slug)
    assert [p.ticker for p in store.get_positions()] == ["AAPL"]
    assert any("Some Unknown Industrials" in line for line in log)
    # the skip reason lands in the creation run's notes, visible on the dashboard
    notes = json.loads(store.get_last_recommendation().llm_response)["notes"]
    assert "Some Unknown Industrials" in notes


def test_create_with_holdings_override(paper_dir, mock_market):
    slug, _ = paper.create_portfolio(
        "Edited", 100_000, "original pasted text kept as record",
        holdings_override=[
            {"ticker": "aapl", "weight_pct": 70, "thesis": "edited row"},
            {"ticker": "MSFT", "weight_pct": 30},
            {"ticker": "", "weight_pct": 10},          # blank row from the UI — dropped
        ])
    _, store = paper.open_portfolio(slug)
    positions = {p.ticker for p in store.get_positions()}
    assert positions == {"AAPL", "MSFT"}
    actions = json.loads(store.get_last_recommendation().actions_json)
    assert {a["ticker"]: a["thesis"] for a in actions}["AAPL"] == "edited row"
    assert store.get_meta("paper_pasted_text") == "original pasted text kept as record"


def test_create_stores_kill_criteria(paper_dir, mock_market):
    text = json.dumps({"positions": [
        {"ticker": "AAPL", "weight_pct": 60, "kill_criterion": "Services revenue growth <10% YoY"},
        {"ticker": "MSFT", "weight_pct": 40, "kill_criterion": "Azure decelerates two quarters"},
    ]})
    slug, _ = paper.create_portfolio("Kills", 100_000, text)
    _, store = paper.open_portfolio(slug)
    kills = json.loads(store.get_meta("paper_kill_criteria"))
    assert kills["AAPL"].startswith("Services revenue")
    # the kill criterion is also folded into the action thesis for the history page
    actions = {a["ticker"]: a for a in json.loads(store.get_last_recommendation().actions_json)}
    assert "Kill: Azure decelerates" in actions["MSFT"]["thesis"]
    assert actions["MSFT"]["kill_criterion"] == "Azure decelerates two quarters"


def test_check_kill_criteria_flags_matching_news(paper_dir, mock_market, monkeypatch):
    text = json.dumps({"positions": [
        {"ticker": "AAPL", "weight_pct": 100,
         "kill_criterion": "Custom silicon takes >20% accelerator share"},
    ]})
    slug, _ = paper.create_portfolio("Watch", 100_000, text)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(paper, "_recent_headlines",
                        lambda t, max_items=8: ["Hyperscaler custom chip hits 25% of accelerator shipments (Reuters)"])
    monkeypatch.setattr(paper, "_judge_kill_hit",
                        lambda ticker, criterion, headlines: "custom silicon crossed 25% per Reuters")
    sent = {}
    import fundmgr.notify.send as send_mod
    monkeypatch.setattr(send_mod, "send_telegram", lambda text, **k: sent.update(text=text) or True)

    log = paper.check_kill_criteria(slug)
    assert any("kill criterion may be triggering" in line for line in log)
    _, store = paper.open_portfolio(slug)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert store.get_meta(f"paper_killhit:AAPL:{today}")
    # second run same day is a no-op (already checked)
    assert paper.check_kill_criteria(slug) == []


def test_check_kill_criteria_no_hit_stays_quiet(paper_dir, mock_market, monkeypatch):
    text = json.dumps({"positions": [{"ticker": "AAPL", "weight_pct": 100,
                                      "kill_criterion": "Services growth stalls"}]})
    slug, _ = paper.create_portfolio("Quiet", 100_000, text)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(paper, "_recent_headlines", lambda t, max_items=8: ["Apple unveils new colorway"])
    monkeypatch.setattr(paper, "_judge_kill_hit", lambda *a: None)
    assert paper.check_kill_criteria(slug) == []


def test_check_kill_criteria_skips_without_api_key(paper_dir, mock_market, monkeypatch):
    text = json.dumps({"positions": [{"ticker": "AAPL", "weight_pct": 100,
                                      "kill_criterion": "x"}]})
    slug, _ = paper.create_portfolio("NoKey", 100_000, text)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    log = paper.check_kill_criteria(slug)
    assert any("no OPENAI_API_KEY" in line for line in log)


def test_delete_archives_db(paper_dir, mock_market):
    slug, _ = paper.create_portfolio("Gone", 50_000, "AAPL 100%")
    paper.delete_portfolio(slug)
    assert paper.list_portfolios() == []
    assert list(paper_dir.glob("gone.db.deleted-*"))  # archived, not destroyed


# ── Tracking ──────────────────────────────────────────────────────────────────

def test_track_portfolio_snapshots_nav(paper_dir, mock_market):
    slug, _ = paper.create_portfolio("Tracked", 100_000, PICKS)
    log = paper.track_portfolio(slug)
    _, store = paper.open_portfolio(slug)
    navs = store.get_nav_history()
    assert len(navs) == 1  # same-day upsert overwrites the creation point
    # prices unchanged → NAV ≈ capital minus fees
    assert navs[-1].portfolio_nav_sek == pytest.approx(100_000, rel=1e-2)
    assert any("NAV" in line for line in log)


def test_track_runs_evaluation_after_horizon(paper_dir, mock_market, monkeypatch):
    slug, _ = paper.create_portfolio("Matured", 100_000, "AAPL 90%")
    _, store = paper.open_portfolio(slug)

    # Age the recommendation + seed a close at decision+28d so the evaluator can pin it
    past = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    with store._conn() as conn:
        conn.execute("UPDATE recommendations SET timestamp = ?", (past,))
    eval_date = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
    store.save_prices("AAPL", [{"date": eval_date, "open": 240, "high": 240,
                                "low": 240, "close": 240, "volume": 1000}])
    store.save_benchmark([
        {"date": (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d"), "close": 100.0},
        {"date": eval_date, "close": 105.0},
    ])

    paper.track_portfolio(slug)

    with store._conn() as conn:
        row = conn.execute("SELECT * FROM decision_outcomes WHERE ticker='AAPL'").fetchone()
    assert row["outperformed"] == 1            # +20% vs +5% benchmark
    assert row["position_return_pct"] == pytest.approx(20.0)


def test_track_all_survives_broken_portfolio(paper_dir, mock_market):
    paper.create_portfolio("Ok", 50_000, "AAPL 100%")
    (paper_dir / "corrupt.db").write_text("not a sqlite file")
    log = paper.track_all()
    assert any("Ok" in line for line in log)


# ── Structured import (broker tickers → Yahoo) ────────────────────────────────

def test_montrose_to_yahoo_mapping():
    assert paper.montrose_to_yahoo("KOG", "NOK", "Kongsberg Gruppen") == "KOG.OL"
    assert paper.montrose_to_yahoo("ENR", "EUR", "Siemens Energy") == "ENR.DE"
    assert paper.montrose_to_yahoo("BESI", "EUR", "BE Semiconductor") == "BESI.AS"
    assert paper.montrose_to_yahoo("SKHY", "USD", "SK Hynix ADR") == "SKHY"  # NasdaqGS line
    assert paper.montrose_to_yahoo("ASML", "EUR", "ASML Holding NV") == "ASML.AS"
    # Bare US symbols already resolve correctly — pass through
    assert paper.montrose_to_yahoo("NVDA", "USD", "NVIDIA") == "NVDA"
    assert paper.montrose_to_yahoo("GEV", "USD", "GE Vernova") == "GEV"
    # Already-suffixed symbols are kept
    assert paper.montrose_to_yahoo("VOLV-B.ST", "SEK", "Volvo") == "VOLV-B.ST"


def test_parse_structured_portfolio():
    data = {
        "meta": {"name": "KF Sleeve", "deployable_capital_sek": 800000,
                 "excluded_holdings": [{"name": "SELLAS", "ticker": "SLS"}]},
        "portfolio_kill_criterion": {
            "trigger": "Any two of the five largest hyperscalers guide 2027 capex flat or down.",
            "action": "Halve the compute cluster.",
        },
        "positions": [
            {"ticker": "KOG", "currency": "NOK", "name": "Kongsberg Gruppen",
             "target_weight_pct": 7.5, "cluster": "defense",
             "kill_criteria": ["Order intake declining YoY", "Budget targets cut"],
             "next_earnings": "2026-07-13"},
            {"ticker": "NVDA", "currency": "USD", "name": "NVIDIA",
             "target_weight_pct": 10.0, "watch": "DC revenue vs hyperscaler capex"},
        ],
    }
    parsed = paper.parse_structured_portfolio(data)
    by_ticker = {h["ticker"]: h for h in parsed["holdings_override"]}
    assert set(by_ticker) == {"KOG.OL", "NVDA"}
    # list kill criteria joined into one clause
    assert by_ticker["KOG.OL"]["kill_criterion"] == "Order intake declining YoY; Budget targets cut"
    # excluded holding never appears
    assert "SLS" not in by_ticker and "SELLAS" not in by_ticker
    assert parsed["excluded"] == ["SLS"]
    # capex kill criterion + default hyperscaler set
    assert "two of the five" in parsed["capex_kill"]["trigger"]
    assert parsed["capex_kill"]["hyperscalers"] == paper.DEFAULT_HYPERSCALERS
    # per-position notes carried through
    assert parsed["position_meta"]["NVDA"]["watch"] == "DC revenue vs hyperscaler capex"
    assert parsed["position_meta"]["KOG.OL"]["next_earnings"] == "2026-07-13"
    assert parsed["capital_sek"] == 800000


def test_create_persists_targets_notes_capex(paper_dir, mock_market):
    slug, _ = paper.create_portfolio(
        "Meta Rich", 100_000, "record",
        holdings_override=[
            {"ticker": "AAPL", "weight_pct": 60, "thesis": "t"},
            {"ticker": "MSFT", "weight_pct": 40, "thesis": "t"},
        ],
        position_meta={"AAPL": {"watch": "Services growth", "next_earnings": "2026-08-01"}},
        capex_kill={"trigger": "2 of 5 hyperscalers flat/down", "action": "halve compute",
                    "hyperscalers": ["MSFT", "AMZN"]},
    )
    _, store = paper.open_portfolio(slug)
    targets = json.loads(store.get_meta("paper_target_weights"))
    assert targets == {"AAPL": 60.0, "MSFT": 40.0}
    notes = json.loads(store.get_meta("paper_position_notes"))
    assert notes["AAPL"]["watch"] == "Services growth"
    capex = json.loads(store.get_meta("paper_capex_kill"))
    assert capex["hyperscalers"] == ["MSFT", "AMZN"]


def test_paper_import_cli(paper_dir, mock_market, tmp_path):
    from click.testing import CliRunner
    from fundmgr.cli import cli

    data = {
        "meta": {"name": "Import Test", "deployable_capital_sek": 100000,
                 "excluded_holdings": [{"ticker": "SLS", "name": "SELLAS"}]},
        "portfolio_kill_criterion": {"trigger": "2 of 5 capex flat/down",
                                     "action": "halve compute"},
        "positions": [
            {"ticker": "AAPL", "currency": "USD", "name": "Apple", "target_weight_pct": 60},
            {"ticker": "MSFT", "currency": "USD", "name": "Microsoft", "target_weight_pct": 40},
        ],
    }
    jf = tmp_path / "picks.json"
    jf.write_text(json.dumps(data))

    result = CliRunner().invoke(cli, ["paper-import", str(jf)])
    assert result.exit_code == 0, result.output
    _, store = paper.open_portfolio("import-test")
    assert {p.ticker for p in store.get_positions()} == {"AAPL", "MSFT"}
    assert "SLS" not in {p.ticker for p in store.get_positions()}
    assert json.loads(store.get_meta("paper_capex_kill"))["trigger"].startswith("2 of 5")


def test_paper_fill_cli(paper_dir, mock_market):
    from click.testing import CliRunner
    from fundmgr.cli import cli

    paper.create_portfolio("Fillable", 100_000, "AAPL 100%")
    before = {p.ticker: p.shares for p in paper.open_portfolio("fillable")[1].get_positions()}

    result = CliRunner().invoke(
        cli, ["paper-fill", "fillable", "AAPL", "5", "2000", "1"])
    assert result.exit_code == 0, result.output
    _, store = paper.open_portfolio("fillable")
    after = {p.ticker: p.shares for p in store.get_positions()}
    assert after["AAPL"] == pytest.approx(before["AAPL"] + 5)

    # unknown slug exits non-zero
    bad = CliRunner().invoke(cli, ["paper-fill", "nope", "AAPL", "5", "2000", "1"])
    assert bad.exit_code != 0


# ── Monitors: capex kill, earnings calendar, weight drift ─────────────────────

def _capture_telegram(monkeypatch):
    sent = []
    import fundmgr.notify.send as send_mod
    monkeypatch.setattr(send_mod, "send_telegram", lambda text, **k: sent.append(text) or True)
    return sent


def test_check_capex_kill_triggers_on_two(paper_dir, mock_market, monkeypatch):
    slug, _ = paper.create_portfolio(
        "Capex", 100_000, "AAPL 100%",
        capex_kill={"trigger": "2 of 5 hyperscalers guide 2027 capex flat/down",
                    "action": "Halve the compute cluster.",
                    "hyperscalers": ["MSFT", "AMZN"]})
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(paper, "_recent_headlines", lambda t, max_items=8: [f"{t} capex headline"])
    monkeypatch.setattr(paper, "_judge_capex_signal", lambda ticker, headlines: "down")
    sent = _capture_telegram(monkeypatch)

    log = paper.check_capex_kill(slug)
    assert any("TRIGGERED" in line for line in log)
    assert any("KILL CRITERION TRIGGERED" in m for m in sent)
    assert "Halve the compute cluster" in "".join(sent)
    _, store = paper.open_portfolio(slug)
    assert store.get_meta("paper_capex_status") == "triggered"
    # already triggered → a second run sends nothing new
    sent.clear()
    paper.check_capex_kill(slug)
    assert sent == []


def test_check_capex_kill_one_is_warning(paper_dir, mock_market, monkeypatch):
    slug, _ = paper.create_portfolio(
        "CapexWarn", 100_000, "AAPL 100%",
        capex_kill={"trigger": "2 of 5 flat/down", "action": "halve",
                    "hyperscalers": ["MSFT", "AMZN"]})
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(paper, "_recent_headlines", lambda t, max_items=8: [f"{t} news"])
    # only MSFT reads down; AMZN unclear
    monkeypatch.setattr(paper, "_judge_capex_signal",
                        lambda ticker, headlines: "down" if ticker == "MSFT" else None)
    sent = _capture_telegram(monkeypatch)

    log = paper.check_capex_kill(slug)
    assert any("warning" in line for line in log)
    assert any("1 of 2" in m for m in sent)
    _, store = paper.open_portfolio(slug)
    assert store.get_meta("paper_capex_status") == "warning"


def test_check_capex_kill_noop_without_config(paper_dir, mock_market, monkeypatch):
    slug, _ = paper.create_portfolio("NoCapex", 100_000, "AAPL 100%")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert paper.check_capex_kill(slug) == []


def test_check_capex_kill_skips_without_key(paper_dir, mock_market, monkeypatch):
    slug, _ = paper.create_portfolio(
        "CapexNoKey", 100_000, "AAPL 100%",
        capex_kill={"trigger": "x", "action": "y", "hyperscalers": ["MSFT"]})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert any("no OPENAI_API_KEY" in line for line in paper.check_capex_kill(slug))


def test_check_earnings_calendar_heads_up_and_post(paper_dir, mock_market, monkeypatch):
    slug, _ = paper.create_portfolio(
        "Earnings", 100_000, "AAPL 100%",
        position_meta={"AAPL": {"watch": "Services growth vs 10%", "next_earnings": ""}})
    sent = _capture_telegram(monkeypatch)
    tomorrow = (datetime.now(timezone.utc).date() + timedelta(days=1))
    monkeypatch.setattr(paper, "_next_earnings_date", lambda t: tomorrow)

    log = paper.check_earnings_calendar(slug)
    assert any("heads-up" in line for line in log)
    assert any("reports tomorrow" in m and "Services growth" in m for m in sent)
    # dedup: same event, same day → silent
    sent.clear()
    assert paper.check_earnings_calendar(slug) == []

    # after the print (yesterday) → check-the-numbers reminder
    yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1))
    monkeypatch.setattr(paper, "_next_earnings_date", lambda t: yesterday)
    log = paper.check_earnings_calendar(slug)
    assert any("post-earnings" in line for line in log)
    assert any("reported yesterday" in m for m in sent)


def test_check_weight_drift_alerts_past_1_5x(paper_dir, mock_market, monkeypatch):
    slug, _ = paper.create_portfolio("Drift", 100_000, PICKS)  # AAPL 40 / MSFT 30 / VOLV 20
    _, store = paper.open_portfolio(slug)
    sent = _capture_telegram(monkeypatch)

    today = datetime.utcnow().strftime("%Y-%m-%d")
    # Triple AAPL's price (native 200→600) so its weight blows past 1.5× its 40% target
    store.save_prices("AAPL", [{"date": today, "open": 600, "high": 600,
                                "low": 600, "close": 600, "volume": 1000}])
    log = paper.check_weight_drift(slug, store=store)
    assert any("AAPL" in line and "×" in line for line in log)
    assert any("Rebalance rule" in m for m in sent)
    assert store.get_meta("paper_drift_state:AAPL") == "over"

    # still over next run → no repeat alert (transition-based)
    sent.clear()
    assert paper.check_weight_drift(slug, store=store) == []

    # price falls back below 1.4× → state re-arms
    store.save_prices("AAPL", [{"date": today, "open": 200, "high": 200,
                                "low": 200, "close": 200, "volume": 1000}])
    paper.check_weight_drift(slug, store=store)
    assert store.get_meta("paper_drift_state:AAPL") == "under"


def test_check_weight_drift_noop_without_targets(paper_dir, mock_market):
    slug, _ = paper.create_portfolio("NoTargets", 100_000, "AAPL 100%")
    _, store = paper.open_portfolio(slug)
    store.set_meta("paper_target_weights", "{}")
    assert paper.check_weight_drift(slug, store=store) == []


# ── Web routes ────────────────────────────────────────────────────────────────

@pytest.fixture
def client(paper_dir, mock_market):
    from fastapi.testclient import TestClient

    from fundmgr.web.app import app
    return TestClient(app)


def test_paper_home_lists_portfolios(client):
    paper.create_portfolio("Home Test", 50_000, "AAPL 100%")
    r = client.get("/paper/")
    assert r.status_code == 200
    assert "Home Test" in r.text


def test_preview_endpoint(client):
    r = client.post("/paper/preview", json={"holdings_text": "AAPL 60%\nMSFT 40%"})
    data = r.json()
    assert data["ok"] is True
    assert data["total_weight"] == pytest.approx(100)
    r = client.post("/paper/preview", json={"holdings_text": ""})
    assert r.json()["ok"] is False


def test_create_via_form_with_edited_table(client):
    """holdings_json (the edited preview table) overrides the raw paste."""
    r = client.post("/paper/create", data={
        "name": "Table Created", "capital_sek": 100000,
        "holdings_text": "- Nvidia 8% — this text is NOT what gets bought",
        "holdings_json": json.dumps([{"ticker": "AAPL", "weight_pct": 100, "thesis": "t"}]),
    }, follow_redirects=False)
    assert r.status_code == 303
    _, store = paper.open_portfolio("table-created")
    assert [p.ticker for p in store.get_positions()] == ["AAPL"]


def test_create_via_form_and_dashboard(client):
    r = client.post("/paper/create", data={
        "name": "Web Created", "capital_sek": 100000,
        "holdings_text": "AAPL 50%\nMSFT 50%",
        "base_prompt": "pick stocks", "model_label": "Fable", "benchmark": "URTH",
    }, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/paper/web-created"

    r = client.get("/paper/web-created")
    assert r.status_code == 200
    assert "PAPER PORTFOLIO" in r.text
    assert "AAPL" in r.text

    for page in ("transactions", "learnings", "prompt"):
        r = client.get(f"/paper/web-created/{page}")
        assert r.status_code == 200, page
    assert client.get("/paper/web-created/api/stats").status_code == 200
    assert client.get("/paper/web-created/api/nav").status_code == 200


def test_create_via_form_error_rerenders(client):
    r = client.post("/paper/create", data={
        "name": "Bad", "capital_sek": 100000,
        "holdings_text": "no tickers here at all.",
    })
    assert r.status_code == 200
    assert "Could not find any tickers" in r.text


def test_unknown_slug_404(client):
    assert client.get("/paper/nope").status_code == 404


def test_delete_via_form(client):
    paper.create_portfolio("Deletable", 50_000, "AAPL 100%")
    r = client.post("/paper/deletable/delete", follow_redirects=False)
    assert r.status_code == 303
    assert client.get("/paper/deletable").status_code == 404
