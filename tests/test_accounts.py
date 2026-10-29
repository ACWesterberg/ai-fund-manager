"""Broker accounts: snapshot parsing, syncing trades into the mirror, monitoring
plans, the target-reached alert and the /accounts web section."""
import json

import pytest

from fundmgr import accounts, addsignal, paper, watchplan
from fundmgr.engine import account_plan

# The paper-book fixtures (isolated PAPER_DIR, mocked prices/FX/history) are
# shared with test_paper.py rather than copied, so the two stay in step.
pytest_plugins = ["tests.test_paper"]


@pytest.fixture
def sent(monkeypatch):
    out: list[str] = []
    import fundmgr.notify.send as send_mod
    monkeypatch.setattr(send_mod, "send_telegram", lambda text, **kw: out.append(text) or True)
    return out


@pytest.fixture
def no_plans(monkeypatch):
    """Record which names a sync asked plans for, without an LLM."""
    calls: list[list[str]] = []

    def fake(slug, tickers=None, **kw):
        calls.append(list(tickers or []))
        return [{"ticker": t, "stance": "hold", "stance_reason": "fine",
                 "kill_criterion": "margin below 5%", "target_price": 600.0,
                 "currency": "USD", "max_drawdown_pct": 25, "horizon_months": 12,
                 "written": ["kill_criterion"]} for t in tickers or []]
    monkeypatch.setattr(account_plan, "write_plans", fake)
    return calls


SNAPSHOT = {
    "cash_sek": 1000,
    "holdings": [
        {"name": "Volvo B", "isin": "SE0000115446", "yahoo_ticker": "VOLV-B.ST",
         "shares": 10, "avg_cost_sek": 250},
        {"name": "Apple", "yahoo_ticker": "AAPL", "shares": 5, "cost_value_sek": 9000},
    ],
}


def _account(sent=None):
    slug, _log = accounts.create_account("Svenska aktier", json.dumps(SNAPSHOT))
    return slug


# ── Parsing ───────────────────────────────────────────────────────────────────

def test_parse_snapshot_accepts_broker_column_names():
    parsed = accounts.parse_snapshot({
        "likvida_medel": "1 250,50",
        "innehav": [{"kortnamn": "VOLV B", "isin": "se0000115446", "antal": "12",
                     "inköpsvärde": "3 000", "marknadsvärde": "3 600"}],
    })
    assert parsed["cash_sek"] == pytest.approx(1250.5)
    row = parsed["holdings"][0]
    assert row["raw_ticker"] == "VOLV B" and row["ticker"] is None
    assert row["isin"] == "SE0000115446"
    assert row["avg_cost_sek"] == pytest.approx(250)
    assert row["price_sek"] == pytest.approx(300)


def test_parse_snapshot_keeps_photo_importer_rows_resolved():
    parsed = accounts.parse_snapshot([{"ticker": "KOG.OL", "raw_ticker": "KOG",
                                       "name": "Kongsberg", "shares": 3, "avg_cost_sek": 400}])
    assert parsed["holdings"][0]["ticker"] == "KOG.OL"
    assert parsed["holdings"][0]["raw_ticker"] == "KOG"


def test_parse_snapshot_rejects_garbage():
    with pytest.raises(ValueError):
        accounts.parse_snapshot("not a snapshot")
    with pytest.raises(ValueError):
        accounts.parse_snapshot({"nothing": []})


# ── Creating ──────────────────────────────────────────────────────────────────

def test_create_account_seeds_at_broker_cost(paper_dir, mock_market):
    slug = _account()
    meta, store = paper.open_portfolio(slug)
    assert meta["kind"] == "account"
    pos = {p.ticker: p for p in store.get_positions()}
    assert pos["VOLV-B.ST"].shares == 10 and pos["VOLV-B.ST"].avg_cost_sek == 250
    assert pos["AAPL"].avg_cost_sek == pytest.approx(1800)
    assert store.get_cash() == pytest.approx(1000)
    assert meta["capital_sek"] == pytest.approx(12500)
    assert accounts.names(store)["VOLV-B.ST"] == "Volvo B"
    assert [m["slug"] for m in accounts.list_accounts()] == [slug]
    assert paper.list_portfolios(kind="live") == []


# ── Syncing ───────────────────────────────────────────────────────────────────

def test_sync_books_each_kind_of_trade(paper_dir, mock_market, sent, no_plans):
    slug = _account()
    _meta, store = paper.open_portfolio(slug)
    watchplan.set_position_plan(store, "AAPL", kill_criterion="services growth below 5%",
                                max_drawdown_pct=25)

    result = accounts.sync_account(slug, {
        "cash_sek": 500,
        "holdings": [
            # Same ISIN, no ticker: lands on the listing it was booked under.
            {"name": "Volvo B", "isin": "SE0000115446", "shares": 15, "avg_cost_sek": 260},
            {"name": "Microsoft", "yahoo_ticker": "MSFT", "shares": 2, "avg_cost_sek": 4800},
        ],
    })
    kinds = {c["ticker"]: c for c in result["changes"]}
    assert kinds["AAPL"]["kind"] == "closed"
    assert kinds["AAPL"]["price_sek"] == pytest.approx(2000)       # live 200 USD × 10
    assert kinds["VOLV-B.ST"]["kind"] == "added"
    assert kinds["VOLV-B.ST"]["price_sek"] == pytest.approx(280)   # from the change in cost
    assert kinds["MSFT"]["kind"] == "opened"

    meta, store = paper.open_portfolio(slug)
    pos = {p.ticker: p for p in store.get_positions()}
    assert set(pos) == {"VOLV-B.ST", "MSFT"}
    assert pos["VOLV-B.ST"].shares == 15
    assert pos["VOLV-B.ST"].avg_cost_sek == pytest.approx(260)
    # 1000 + 10000 sale − 1400 − 9600 = 0; the broker says 500 → a deposit.
    assert store.get_cash() == pytest.approx(500)
    assert meta["capital_sek"] == pytest.approx(12500)          # starting capital stays put
    assert [f["amount_sek"] for f in store.get_cash_flows()] == [pytest.approx(500)]
    assert result["cash"]["unexplained_sek"] == pytest.approx(500)

    # The closed name stops being watched; the opened one was planned.
    assert "AAPL" not in watchplan.get_kill_rules(store)
    assert "AAPL" not in watchplan.get_kill_text(store)
    assert no_plans == [["MSFT"]]
    targets = json.loads(store.get_meta("paper_target_weights"))
    assert "AAPL" not in targets and targets["MSFT"] > 0

    assert len(sent) == 1
    assert "Opened" in sent[0] and "MSFT" in sent[0] and "Closed" in sent[0]
    assert "+11.1% on cost" in sent[0]
    assert "deposit" in sent[0]


def test_sync_with_no_changes_is_quiet(paper_dir, mock_market, sent, no_plans):
    slug = _account()
    result = accounts.sync_account(slug, SNAPSHOT)
    assert result["changes"] == []
    assert sent == [] and no_plans == []


def test_sync_dry_run_writes_nothing(paper_dir, mock_market, sent, no_plans):
    slug = _account()
    result = accounts.sync_account(slug, {"holdings": [SNAPSHOT["holdings"][0]]},
                                   dry_run=True)
    assert [c["kind"] for c in result["changes"]] == ["closed"]
    _meta, store = paper.open_portfolio(slug)
    assert {p.ticker for p in store.get_positions()} == {"VOLV-B.ST", "AAPL"}
    assert sent == []


def test_sync_infers_a_deposit_for_an_unfunded_buy(paper_dir, mock_market, sent, no_plans):
    slug = _account()
    holdings = SNAPSHOT["holdings"] + [
        {"name": "Microsoft", "yahoo_ticker": "MSFT", "shares": 3, "avg_cost_sek": 5000}]
    accounts.sync_account(slug, {"holdings": holdings})       # no cash figure given
    meta, store = paper.open_portfolio(slug)
    assert {p.ticker for p in store.get_positions()} == {"VOLV-B.ST", "AAPL", "MSFT"}
    assert store.get_cash() >= 0
    # The 15 000 SEK purchase was paid with money the mirror hadn't seen: a deposit, not return.
    assert meta["capital_sek"] == pytest.approx(12500)
    assert sum(f["amount_sek"] for f in store.get_cash_flows()) == pytest.approx(14000, abs=1)


def test_sync_refuses_a_non_account_book(paper_dir, mock_market):
    slug, _ = paper.create_portfolio("Sim", 10_000, "AAPL 100%")
    with pytest.raises(ValueError):
        accounts.sync_account(slug, SNAPSHOT)


def test_sync_reports_unresolvable_rows(paper_dir, mock_market, sent, no_plans, monkeypatch):
    import fundmgr.vision.portfolio_photo as photo
    monkeypatch.setattr(photo, "_resolve_tickers", lambda rows: rows)
    slug = _account()
    result = accounts.sync_account(slug, {"holdings": SNAPSHOT["holdings"] + [
        {"name": "Mystery AB", "shares": 4, "avg_cost_sek": 10}]})
    assert result["changes"] == []
    assert any("Mystery AB" in w for w in result["warnings"])
    assert len(sent) == 1       # a warning alone is still worth telling you


# ── Plans ─────────────────────────────────────────────────────────────────────

def _fake_llm(monkeypatch, plans):
    from fundmgr.config import AppConfig
    import fundmgr.engine.client as client

    def fake_call(system, user, cfg, schema=None, **kw):
        assert "AAPL" in user and "Apple" in user
        return schema(summary="Two names, concentrated in tech.", plans=plans), "{}"
    monkeypatch.setattr(client, "call_llm", fake_call)
    monkeypatch.setattr(account_plan, "_plan_config", lambda: AppConfig())
    monkeypatch.setattr(paper, "_recent_headlines", lambda t, max_items=8: [])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def _plan(ticker, **kw):
    base = dict(ticker=ticker, stance="hold", stance_reason="Thesis intact.",
                thesis="Ecosystem lock-in.", kill_criterion="Services growth below 5%",
                add_criterion="Services growth above 10%", target_price=250.0,
                max_drawdown_pct=25, horizon_months=12)
    base.update(kw)
    return account_plan.PositionPlan(**base)


def test_write_plans_installs_and_records(paper_dir, mock_market, sent, monkeypatch):
    slug = _account()
    _fake_llm(monkeypatch, [_plan("AAPL", stance="trim", suggested_weight_pct=40,
                                  stance_reason="Over a third of the account."),
                            _plan("VOLV-B.ST", target_price=340.0)])
    out = account_plan.write_plans(slug)
    assert {p["ticker"] for p in out} == {"AAPL", "VOLV-B.ST"}

    _meta, store = paper.open_portfolio(slug)
    assert watchplan.get_kill_text(store)["AAPL"] == "Services growth below 5%"
    rule = watchplan.get_kill_rules(store)["AAPL"]
    assert rule["price_above"] == 250.0 and rule["max_drawdown_pct"] == 25
    assert rule["anchor_price_sek"]                     # measured from today, not cost
    assert addsignal.get_plan(store, "AAPL")["target_price"] == 250.0
    assert watchplan.get_horizons(store)["AAPL"]["review_date"]
    notes = json.loads(store.get_meta("paper_position_notes"))
    assert notes["AAPL"]["stance"] == "trim"

    rec = store.get_last_recommendation()
    assert rec.run_id.startswith(f"plan-{slug}-")
    actions = {a["ticker"]: a for a in json.loads(rec.actions_json)}
    assert actions["AAPL"]["side"] == "sell" and actions["AAPL"]["target_weight_pct"] == 40
    assert actions["VOLV-B.ST"]["side"] == "hold"

    assert len(sent) == 1 and "Worth acting on" in sent[0]
    # The trim leads the message, ahead of the hold.
    assert sent[0].index("AAPL") < sent[0].index("VOLV-B.ST")


def test_write_plans_keeps_hand_edits_unless_overwrite(paper_dir, mock_market, sent, monkeypatch):
    slug = _account()
    _meta, store = paper.open_portfolio(slug)
    watchplan.set_position_plan(store, "AAPL", kill_criterion="my own line", price_above=300)

    _fake_llm(monkeypatch, [_plan("AAPL")])
    out = account_plan.write_plans(slug, ["AAPL"])
    assert "kill_criterion" not in out[0]["written"]
    assert watchplan.get_kill_text(store)["AAPL"] == "my own line"
    assert watchplan.get_kill_rules(store)["AAPL"]["price_above"] == 300

    account_plan.write_plans(slug, ["AAPL"], overwrite=True)
    assert watchplan.get_kill_text(store)["AAPL"] == "Services growth below 5%"
    assert watchplan.get_kill_rules(store)["AAPL"]["price_above"] == 250.0


def test_write_plans_ignores_tickers_not_held(paper_dir, mock_market, sent, monkeypatch):
    slug = _account()
    _fake_llm(monkeypatch, [_plan("AAPL"), _plan("TSLA")])
    out = account_plan.write_plans(slug, ["AAPL"])
    assert [p["ticker"] for p in out] == ["AAPL"]


# ── Target reached is a take-profit, not a max-drop ───────────────────────────

def test_target_hit_gets_its_own_headline(paper_dir, mock_market, sent):
    slug = _account()
    _meta, store = paper.open_portfolio(slug)
    watchplan.set_position_plan(store, "VOLV-B.ST", price_above=290, currency="SEK")
    log = watchplan.check_kill_rules(slug, store=store)     # last close 300 ≥ 290
    assert any("target reached" in line for line in log)
    assert "target price reached, consider taking profit" in sent[0]
    assert "max drop" not in sent[0]


# ── Web ───────────────────────────────────────────────────────────────────────

@pytest.fixture
def client(paper_dir, mock_market, monkeypatch):
    from fastapi.testclient import TestClient

    import fundmgr.web.paper as web_paper
    from fundmgr.web.app import app
    started: list = []
    monkeypatch.setattr(web_paper, "_start_plan_job",
                        lambda slug, tickers=None, overwrite=False: started.append(slug) or True)
    monkeypatch.delenv("FUND_WEB_SHOW_ARCHIVED", raising=False)
    c = TestClient(app)
    c.started = started
    return c


def test_accounts_home_and_sidebar(client):
    r = client.get("/accounts/")
    assert r.status_code == 200
    assert "Add an account" in r.text
    assert 'href="/whatif"' in r.text
    # The comparison sims, paper and live sections are off the sidebar…
    assert 'href="/sim-claude"' not in r.text
    assert 'href="/paper"' not in r.text and 'href="/live"' not in r.text


def test_archived_sections_return_on_request(client, monkeypatch):
    monkeypatch.setenv("FUND_WEB_SHOW_ARCHIVED", "1")
    r = client.get("/accounts/")
    assert 'href="/sim-claude"' in r.text and 'href="/live"' in r.text


def test_live_pages_still_served_and_show_their_nav(client):
    r = client.get("/live/")
    assert r.status_code == 200
    assert 'href="/live"' in r.text


def test_create_account_from_the_web(client):
    r = client.post("/accounts/create-from-snapshot", data={
        "name": "Utländska aktier", "snapshot_text": json.dumps(SNAPSHOT),
        "benchmark": "URTH"}, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"].startswith("/accounts/utlandska-aktier")
    assert client.started == ["utlandska-aktier"]

    page = client.get("/accounts/utlandska-aktier")
    assert page.status_code == 200
    assert "Sync after you trade" in page.text
    assert "BROKER ACCOUNT" in page.text
    # Listed in the sidebar on every page, with its own sub-pages when open.
    assert 'href="/accounts/utlandska-aktier/transactions"' in page.text
    assert "Utländska aktier" in client.get("/accounts/").text


def test_create_account_shows_errors(client):
    r = client.post("/accounts/create-from-snapshot", data={
        "name": "Broken", "snapshot_text": "{}"})
    assert r.status_code == 200 and "holdings list" in r.text


def test_sync_endpoint_takes_json_and_forms(client, sent, no_plans):
    client.post("/accounts/create-from-snapshot", data={
        "name": "Svenska aktier", "snapshot_text": json.dumps(SNAPSHOT)})

    preview = client.post("/accounts/svenska-aktier/sync?dry_run=1",
                          json={"holdings": SNAPSHOT["holdings"][:1]})
    assert preview.status_code == 200
    assert [c["kind"] for c in preview.json()["changes"]] == ["closed"]

    r = client.post("/accounts/svenska-aktier/sync",
                    data={"snapshot_text": json.dumps({"holdings": SNAPSHOT["holdings"][:1]})},
                    follow_redirects=False)
    assert r.status_code == 303 and "closed+AAPL" in r.headers["location"]
    _meta, store = paper.open_portfolio("svenska-aktier")
    assert {p.ticker for p in store.get_positions()} == {"VOLV-B.ST"}

    assert client.post("/accounts/nope/sync", json={"holdings": []}).status_code == 404
    assert client.post("/accounts/svenska-aktier/sync", json={"x": 1}).status_code == 400


def test_recording_a_buy_of_a_new_name_starts_its_plan(client):
    client.post("/accounts/create-from-snapshot", data={
        "name": "Svenska aktier", "snapshot_text": json.dumps(SNAPSHOT)})
    client.started.clear()
    client.post("/accounts/svenska-aktier/fill", data={
        "ticker": "MSFT", "shares": "1", "price_sek": "500", "side": "buy"})
    assert client.started == ["svenska-aktier"]


def test_plans_button_starts_a_job(client):
    client.post("/accounts/create-from-snapshot", data={
        "name": "Svenska aktier", "snapshot_text": json.dumps(SNAPSHOT)})
    client.started.clear()
    r = client.post("/accounts/svenska-aktier/plans", data={}, follow_redirects=False)
    assert r.status_code == 303 and client.started == ["svenska-aktier"]
