"""
Montrose phase 3: a pre-filled order ticket for each trade the weekly run approved.

A ticket is not an order — the owner approves it in the app — but it can name
the wrong instrument or account, so those are what these tests pin down.
"""
from __future__ import annotations

from datetime import datetime

import httpx
import pytest
from click.testing import CliRunner

from fundmgr.broker import montrose as m
from fundmgr.broker import reconcile as rec
from fundmgr.broker import tickets as tk
from fundmgr.config import AppConfig
from fundmgr.data.prices import TickerFeatures
from fundmgr.engine.schema import Action
from fundmgr.state.models import Position, PortfolioSnapshot, Transaction
from fundmgr.state.store import Store

from tests.test_montrose_reconcile import ACCOUNT_ID, UNIVERSE, FakeMCP

TICKET_URL = "https://app.montrose.io/trade?ticketId=abc123"

SEARCH = {
    "KCC": [{"orderbookId": 4608, "ticker": "KCC", "name": "Klaveness Combination Carriers"}],
    "TRUE B": [{"orderbookId": 1757, "ticker": "TRUE", "name": "Truecaller B"}],
    "ATCO B": [{"orderbookId": 1938, "ticker": "ATCO A", "name": "Atlas Copco A"},
               {"orderbookId": 1939, "ticker": "ATCO B", "name": "Atlas Copco B"}],
    "ASML": [{"orderbookId": 3193, "ticker": "ASML", "name": "ASML Holding"},
             {"orderbookId": 1306, "ticker": "ASML", "name": "ASML Holding NV"}],
}


class FakeTicketMCP(FakeMCP):
    def __init__(self):
        super().__init__()
        self.url = TICKET_URL

    def _tool(self, name, args):
        if name == "search_instruments" and "query" in args:
            return {"result": SEARCH.get(args["query"], [])}
        if name == "create_trade_ticket":
            return {"url": self.url}
        return super()._tool(name, args)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    f = FakeTicketMCP()
    monkeypatch.setattr(m, "DATA_DIR", tmp_path)
    monkeypatch.setattr(m, "_http", lambda: httpx.Client(transport=httpx.MockTransport(f.handler)))
    monkeypatch.setattr(m, "access_token", lambda force_refresh=False: "at")
    return f


def _tickets(fake):
    return [a for n, a in fake.calls if n == "create_trade_ticket"]


# ── The ticket call itself ───────────────────────────────────────────────────

def test_a_ticket_names_orderbook_whole_shares_and_the_account(fake):
    with m.Session() as s:
        assert s.create_trade_ticket("Sell", 4608, 36, ACCOUNT_ID) == TICKET_URL
    assert _tickets(fake) == [{"side": "Sell", "orderbookId": 4608, "quantity": 36,
                               "accountId": ACCOUNT_ID}]


@pytest.mark.parametrize("side,obid,qty,acc", [
    ("sell", 4608, 36, ACCOUNT_ID),        # Montrose's enum is Buy/Sell
    ("Sell", "4608", 36, ACCOUNT_ID),      # never a ticker or a string id
    ("Sell", 4608, 36.5, ACCOUNT_ID),      # whole shares only
    ("Sell", 4608, 0, ACCOUNT_ID),
    ("Sell", 4608, True, ACCOUNT_ID),
    ("Sell", 4608, 36, ""),                # never the default account
])
def test_bad_ticket_arguments_are_refused_before_any_call(fake, side, obid, qty, acc):
    with m.Session() as s, pytest.raises(ValueError):
        s.create_trade_ticket(side, obid, qty, acc)
    assert _tickets(fake) == []


def test_a_ticket_url_off_montrose_trade_page_is_refused(fake):
    fake.url = "https://evil.example/trade?ticketId=1"
    with m.Session() as s, pytest.raises(m.MontroseError, match="Unexpected ticket URL"):
        s.create_trade_ticket("Buy", 4608, 1, ACCOUNT_ID)


def test_the_generic_tool_path_still_cannot_reach_writes(fake):
    with m.Session() as s:
        for tool in ("create_trade_ticket", "create_price_alert", "add_to_watchlist"):
            with pytest.raises(PermissionError):
                s.call_tool(tool, {})


# ── Instrument resolution ────────────────────────────────────────────────────

def _search(query):
    return SEARCH.get(query, [])


def test_resolution_uses_pins_then_the_account_then_one_search_hit():
    known = rec.known_symbols([], UNIVERSE)
    inst = {"1757": {"ticker": "TRUE", "name": "TRUECALLER AK B", "currency": "SEK"},
            "999": {"yahoo": "MYST.ST", "ticker": "MYS"}}
    assert tk.resolve_orderbook("MYST.ST", inst, known, _search) == (999, "MYS", None)
    assert tk.resolve_orderbook("TRUE-B.ST", inst, known, _search) == (1757, "TRUE", None)
    obid, ticker, problem = tk.resolve_orderbook("KCC.OL", inst, known, _search)
    assert (obid, ticker, problem) == (4608, "KCC", None)
    assert inst["4608"]["resolved_yahoo"] == "KCC.OL"
    # The exact class wins over its sibling in the same search.
    assert tk.resolve_orderbook("ATCO-B.ST", {}, known, _search)[0] == 1939


def test_a_name_listed_twice_gets_no_ticket():
    obid, _, problem = tk.resolve_orderbook("ASML.AS", {}, {}, _search)
    assert obid is None and "2 Montrose listings" in problem and "montrose-map" in problem
    obid, _, problem = tk.resolve_orderbook("NOPE.ST", {}, {}, _search)
    assert obid is None and "no listing" in problem


# ── Plan from a run ──────────────────────────────────────────────────────────

def _feat(ticker, price, ccy="SEK"):
    return TickerFeatures(ticker=ticker, name=ticker, last_price=price, last_date="2026-10-26",
                          data_age_trading_days=0, currency=ccy)


def _action(ticker, side, weight, sek):
    return Action(ticker=ticker, side=side, target_weight_pct=weight, sek_estimate=sek,
                  confidence=0.7, thesis="t")


def test_the_plan_sells_first_exits_everything_and_sizes_buys_in_whole_shares():
    snap = PortfolioSnapshot(positions=[Position("KCC.OL", 35.4, 95.0, 120.0),
                                        Position("INWI.ST", 41, 177.0, 190.0)], cash_sek=20_000.0)
    feats = {"KCC.OL": _feat("KCC.OL", 112.4, "NOK"), "INWI.ST": _feat("INWI.ST", 190.0),
             "TRUE-B.ST": _feat("TRUE-B.ST", 25.0)}
    actions = [_action("TRUE-B.ST", "buy", 5.0, 5_010.0),
               _action("KCC.OL", "sell", 0.0, 4_000.0),
               _action("INWI.ST", "sell", 4.0, 1_000.0),
               _action("INWI.ST", "hold", 7.0, 0.0),
               _action("MISSING.ST", "buy", 3.0, 3_000.0)]
    plan = tk.plan_from_run("r1", actions, snap, feats, AppConfig())
    lines = tk.plan_lines(plan)
    got = [(t.ticker, t.side, t.shares, t.kind) for t in lines]
    assert got == [("KCC.OL", "sell", 35, "EXIT"), ("INWI.ST", "sell", 5, "TRIM"),
                   ("TRUE-B.ST", "buy", 200, "NEW"), ("MISSING.ST", "buy", 0, "NEW")]
    assert lines[-1].problem and lines[0].currency == "NOK"


def test_trades_already_in_the_ledger_get_no_ticket():
    lines = [tk.TicketLine("KCC.OL", "sell", 35), tk.TicketLine("TRUE-B.ST", "buy", 200)]
    trades = [rec.LedgerTrade(datetime(2026, 10, 26).date(), "KCC.OL", "sell", 35, 4_000.0),
              rec.LedgerTrade(datetime(2026, 10, 20).date(), "TRUE-B.ST", "buy", 10, -250.0)]
    done = tk.mark_done(lines, trades, "2026-10-26T09:31:00+00:00")
    assert done == ["SELL 35 KCC.OL"] and [t.ticker for t in lines] == ["TRUE-B.ST"]


# ── CLI ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def env(fake, tmp_path, monkeypatch):
    from fundmgr.notify import send

    db = tmp_path / "fund.db"
    uni = tmp_path / "universe.csv"
    uni.write_text("name,yahoo_ticker,isin,country,exchange,sector,enabled\n"
                   + "".join(f"{u.name},{u.yahoo_ticker},,SE,{u.exchange},,true\n" for u in UNIVERSE))
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"db_path: {db}\nuniverse_path: {uni}\nmontrose:\n  account_name: Simulering\n")
    monkeypatch.setenv("FUND_CONFIG", str(cfg))
    store = Store(db)
    store.set_cash(30_000.0)
    sent: list[tuple[str, object]] = []
    monkeypatch.setattr(send, "send_telegram",
                        lambda text, parse_mode="HTML", buttons=None: sent.append((text, buttons)) or True)
    return store, sent


def _cli(*args):
    from fundmgr.cli import cli
    return CliRunner().invoke(cli, list(args))


def test_one_manual_ticket_goes_to_the_fund_account_as_a_button(env, fake):
    _, sent = env
    res = _cli("montrose-ticket", "kcc.ol", "sell", "36", "--notify")
    assert res.exit_code == 0, res.output
    assert _tickets(fake) == [{"side": "Sell", "orderbookId": 4608, "quantity": 36,
                               "accountId": ACCOUNT_ID}]
    text, buttons = sent[0]
    assert buttons == [[("SELL 36 KCC.OL", TICKET_URL)]] and "approve it there" in text


def test_an_ambiguous_manual_ticket_is_refused_without_a_call(env, fake):
    res = _cli("montrose-ticket", "ASML.AS", "buy", "1")
    assert res.exit_code != 0 and "2 Montrose listings" in res.output
    assert _tickets(fake) == []


def test_latest_run_tickets_skip_what_is_already_filled(env, fake):
    store, sent = env
    snap = PortfolioSnapshot(positions=[Position("KCC.OL", 35, 95.0, 120.0)], cash_sek=20_000.0)
    feats = {"KCC.OL": _feat("KCC.OL", 112.4, "NOK"), "TRUE-B.ST": _feat("TRUE-B.ST", 25.0),
             "ASML.AS": _feat("ASML.AS", 600.0, "EUR")}
    plan = tk.plan_from_run("r1", [_action("KCC.OL", "sell", 0.0, 4_000.0),
                                   _action("TRUE-B.ST", "buy", 5.0, 5_010.0),
                                   _action("ASML.AS", "buy", 5.0, 6_000.0)], snap, feats, AppConfig())
    plan["created_at"] = "2026-10-26T09:31:00+00:00"
    tk.save_plan(plan)
    store.apply_fill(Transaction("KCC.OL", "buy", 35, 95.0, 3.0, "fill", datetime(2026, 10, 1, 12)))
    store.apply_fill(Transaction("KCC.OL", "sell", 35, 120.0, 4.2, "fill", datetime(2026, 10, 26, 12)))

    res = _cli("montrose-tickets", "--notify")
    assert res.exit_code == 0, res.output
    assert [(a["orderbookId"], a["side"], a["quantity"]) for a in _tickets(fake)] == [(1757, "Buy", 200)]
    text, buttons = sent[0]
    assert buttons == [[("BUY 200 TRUE-B.ST", TICKET_URL)]]
    assert "SELL 35 KCC.OL" in text and "2 Montrose listings" in text


def test_the_weekly_run_hook_saves_the_plan_and_sends_only_when_switched_on(env, fake, tmp_path):
    from fundmgr import cli as commands
    from fundmgr.config import load_config

    store, sent = env
    cfg = load_config()
    snap = PortfolioSnapshot(positions=[], cash_sek=20_000.0)
    feats = {"TRUE-B.ST": _feat("TRUE-B.ST", 25.0)}
    acts = [_action("TRUE-B.ST", "buy", 5.0, 5_010.0)]

    commands._run_tickets(cfg, store, "r2", acts, snap, feats)
    assert tk.load_plan()["run_id"] == "r2" and _tickets(fake) == [] and sent == []

    cfg.montrose.tickets = True
    commands._run_tickets(cfg, store, "r3", acts, snap, feats)
    assert len(_tickets(fake)) == 1 and sent[0][1] == [[("BUY 200 TRUE-B.ST", TICKET_URL)]]


def test_a_montrose_outage_never_breaks_the_run(env, fake, monkeypatch):
    from fundmgr import cli as commands
    from fundmgr.config import load_config

    store, sent = env
    cfg = load_config()
    cfg.montrose.tickets = True

    def down(force_refresh=False):
        raise m.LoginRequired("expired")

    monkeypatch.setattr(m, "access_token", down)
    commands._run_tickets(cfg, store, "r4", [_action("TRUE-B.ST", "buy", 5.0, 5_010.0)],
                          PortfolioSnapshot(positions=[], cash_sek=20_000.0),
                          {"TRUE-B.ST": _feat("TRUE-B.ST", 25.0)})
    assert sent and "not sent" in sent[0][0] and "/montrose_tickets" in sent[0][0]


# ── Tickets for review orders (stop / target reviews) ────────────────────────
# The 30 Oct stop review said "TRIM IVSO.ST — sell 15 shares" and came with no
# ticket: tickets were only wired to the Monday run.

def _review_env(env, fake, tickets_on=True):
    from fundmgr.config import load_config

    store, sent = env
    cfg = load_config()
    cfg.montrose.tickets = tickets_on
    store.apply_fill(Transaction("TRUE-B.ST", "buy", 30, 25.0, 1.0, "fill", datetime(2026, 10, 5, 12)))
    return cfg, store, sent


def test_a_trim_review_gets_a_ticket_for_exactly_its_size(env, fake):
    from fundmgr import cli as commands

    cfg, store, _ = _review_env(env, fake)
    store.record_review(ticker="TRUE-B.ST", source="stop_review", verdict="trim", trim_pct=50)
    buttons = commands._owed_ticket_buttons(cfg, store, ["TRUE-B.ST"])
    assert buttons == [[("SELL 15 TRUE-B.ST", TICKET_URL)]]
    assert _tickets(fake) == [{"side": "Sell", "orderbookId": 1757, "quantity": 15,
                               "accountId": ACCOUNT_ID}]
    # The 11:00 alert repeating the same owed order reuses the ticket.
    assert commands._owed_ticket_buttons(cfg, store, ["TRUE-B.ST"]) == buttons
    assert len(_tickets(fake)) == 1


def test_an_exit_sells_everything_and_an_add_gets_no_ticket(env, fake):
    from fundmgr import cli as commands

    cfg, store, _ = _review_env(env, fake)
    store.record_review(ticker="TRUE-B.ST", source="target_review", verdict="exit")
    assert commands._owed_ticket_buttons(cfg, store, ["TRUE-B.ST"]) == [[("SELL 30 TRUE-B.ST", TICKET_URL)]]
    store.record_review(ticker="TRUE-B.ST", source="stop_review", verdict="add")
    assert commands._owed_ticket_buttons(cfg, store, ["TRUE-B.ST"]) == []


def test_no_review_tickets_when_tickets_are_off(env, fake):
    from fundmgr import cli as commands

    cfg, store, _ = _review_env(env, fake, tickets_on=False)
    store.record_review(ticker="TRUE-B.ST", source="stop_review", verdict="trim", trim_pct=50)
    assert commands._owed_ticket_buttons(cfg, store, ["TRUE-B.ST"]) == [] and _tickets(fake) == []


def test_review_tickets_go_out_as_their_own_message(env, fake):
    from fundmgr import cli as commands

    cfg, store, sent = _review_env(env, fake)
    store.record_review(ticker="TRUE-B.ST", source="stop_review", verdict="trim", trim_pct=50)
    commands._send_review_tickets(cfg, store, ["TRUE-B.ST"])
    assert sent[-1][1] == [[("SELL 15 TRUE-B.ST", TICKET_URL)]] and "review above" in sent[-1][0]
