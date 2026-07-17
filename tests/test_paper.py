"""Paper portfolios: parsing pasted picks, creation at live prices, tracking, web routes."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from fundmgr import paper


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
