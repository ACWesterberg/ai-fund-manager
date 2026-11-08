"""
Montrose phase 1: refresh the real account and reconcile it with the ledger.

The shapes below are Montrose's real replies (probed 2026-10-27), with made-up
numbers. No network: refresh runs against a fake MCP server.
"""
from __future__ import annotations

import hashlib
import json
import stat
from datetime import date, datetime

import httpx
import pytest
from click.testing import CliRunner

from fundmgr.broker import montrose as m
from fundmgr.broker import reconcile as rec
from fundmgr.config import UniverseTicker
from fundmgr.state.models import Transaction
from fundmgr.state.store import Store

ACCOUNT_ID = "acc-sim"


def _u(yahoo: str, exchange: str, name: str = "") -> UniverseTicker:
    return UniverseTicker(name=name or yahoo, yahoo_ticker=yahoo, isin="", country="SE",
                          exchange=exchange, sector="", enabled=True)


UNIVERSE = [_u("TRUE-B.ST", "OMXS"), _u("KCC.OL", "OSLO"), _u("INWI.ST", "OMXS"),
            _u("BONEX.ST", "OMXS"), _u("ATCO-A.ST", "OMXS"), _u("ATCO-B.ST", "OMXS"),
            _u("KCC.ST", "OMXS")]


def _pos(ticker, name, obid, qty, mv, pnl, ccy="SEK"):
    return {"instrumentName": name, "ticker": ticker, "orderbookId": obid, "possibleOrderbookIds": [],
            "quantity": qty, "marketValue": {"accountCurrency": mv, "instrumentCurrency": mv},
            "unrealizedResult": {"accountCurrency": pnl, "instrumentCurrency": pnl},
            "unrealizedResultPercent": 0.0, "instrumentCurrency": ccy, "fxRate": 1, "loanToValue": 0.7}


def _tx(tid, cat, day, obid=0, qty=0, price=0.0, ccy="", amount=0.0, text=""):
    return {"id": tid, "accountId": ACCOUNT_ID, "accountNumber": 1, "category": cat,
            "code": {"Buy": "K", "Sell": "S"}.get(cat, ""), "tradeDate": day, "settlementDate": day,
            "accountDate": day, "quantity": qty, "price": price, "priceCurrency": ccy,
            "amount": amount, "currency": "SEK", "instrumentId": "", "orderbookId": obid,
            "text": text, "isCanceled": False}


def _holdings(cash=10_000.0):
    return {
        "accountId": ACCOUNT_ID, "accountNumber": "1", "accountName": "Simulering",
        "accountType": "ISK", "currency": "SEK",
        "summary": {"totalMarketValue": 14_000.0, "availableForPurchase": cash,
                    "totalValue": 14_000.0 + cash, "currency": "SEK"},
        "currencyPositions": [{"currencyCode": "SEK", "balance": cash, "accruedInterest": 0,
                               "availableForPurchase": cash}],
        "positions": [
            _pos("TRUE", "TRUECALLER AK B", 1757, 100, 2_600.0, 100.0),
            _pos("KCC", "Klaveness Combination Carriers", 4608, 35, 4_100.0, 800.0, "NOK"),
            _pos("INWI", "Inwido", 610, 40, 7_300.0, 0.0),
        ],
        "error": None,
    }


TRANSACTIONS = [
    _tx(1, "Deposit", "2026-10-01", amount=20_000.0, text="Från 123"),
    _tx(2, "Buy", "2026-10-02", 1757, 100, 25.0, "SEK", -2_502.5),
    _tx(3, "Buy", "2026-10-02", 4608, 35, 90.0, "NOK", -3_300.0),
    _tx(4, "Buy", "2026-10-05", 610, 40, 182.0, "SEK", -7_287.28),
    _tx(5, "Buy", "2026-10-06", 692, 10, 300.0, "SEK", -3_003.0),
    _tx(6, "Sell", "2026-10-08", 692, -10, 310.0, "SEK", 3_096.9),
    _tx(7, "Dividend", "2026-10-09", 4608, amount=50.0, text="Utdelning KCCo"),
    _tx(8, "ForeignTax", "2026-10-09", 4608, amount=-12.5, text="Utdelning KCCo"),
]


def _snapshot(cash=10_000.0, txns=None):
    h = _holdings(cash)
    return {"fetched_at": "2026-10-27T11:00:00+00:00", "since": "2026-09-27",
            "account": {"accountId": ACCOUNT_ID, "accountName": "Simulering", "accountType": "ISK"},
            "currency": "SEK", "summary": h["summary"], "currency_positions": h["currencyPositions"],
            "positions": h["positions"], "transactions": TRANSACTIONS if txns is None else txns}


INSTRUMENTS = {"692": {"ticker": "BONEX", "name": "Bonesupport Holding", "currency": "SEK"},
               "1757": {"ticker": "TRUE", "name": "TRUECALLER AK B", "currency": "SEK"},
               "4608": {"ticker": "KCC", "name": "Klaveness", "currency": "NOK"},
               "610": {"ticker": "INWI", "name": "Inwido", "currency": "SEK"}}


def _ledger_trade(day, ticker, side, shares, effect):
    return rec.LedgerTrade(date.fromisoformat(day), ticker, side, shares, effect)


def _ledger(**override):
    base = {
        "ledger_cash": 10_000.0,
        "ledger_positions": {"TRUE-B.ST": 100, "KCC.OL": 35, "INWI.ST": 40},
        "ledger_costs": {"TRUE-B.ST": 25.0, "KCC.OL": 94.29, "INWI.ST": 182.18},
        "ledger_trades": [
            _ledger_trade("2026-10-02", "TRUE-B.ST", "buy", 100, -2_502.5),
            _ledger_trade("2026-10-02", "KCC.OL", "buy", 35, -3_303.0),
            _ledger_trade("2026-10-05", "INWI.ST", "buy", 40, -7_287.28),
            _ledger_trade("2026-10-06", "BONEX.ST", "buy", 10, -3_003.0),
            _ledger_trade("2026-10-08", "BONEX.ST", "sell", 10, 3_096.9),
        ],
        "ledger_flows": [{"date": "2026-10-01", "amount_sek": 20_000.0, "note": ""}],
    }
    return {**base, **override}


def _run(snapshot=None, **override):
    return rec.reconcile(snapshot or _snapshot(), **_ledger(**override), universe=UNIVERSE,
                         instruments=INSTRUMENTS, cash_tolerance=25.0)


# ── Instrument resolution ────────────────────────────────────────────────────

def test_montrose_tickers_resolve_by_ticker_and_quote_currency():
    known = rec.known_symbols([], UNIVERSE)
    assert rec.resolve_yahoo("TRUE", "TRUECALLER AK B", "SEK", known) == "TRUE-B.ST"
    # KCC is in the universe twice; the quote currency picks the Oslo line.
    assert rec.resolve_yahoo("KCC", "Klaveness", "NOK", known) == "KCC.OL"
    assert rec.resolve_yahoo("KCC", "Klaveness", "SEK", known) == "KCC.ST"


def test_share_class_comes_from_the_name_and_is_never_guessed():
    known = rec.known_symbols([], UNIVERSE)
    assert rec.resolve_yahoo("ATCO", "Atlas Copco B", "SEK", known) == "ATCO-B.ST"
    assert rec.resolve_yahoo("ATCO", "Atlas Copco", "SEK", known) is None
    assert rec.resolve_yahoo("NOPE", "Nope AB", "SEK", known) is None


def test_ledger_holdings_outside_the_universe_still_resolve():
    known = rec.known_symbols(["OLD.ST"], [])
    assert rec.resolve_yahoo("OLD", "Old AB", "SEK", known) == "OLD.ST"


# ── Reconcile ────────────────────────────────────────────────────────────────

def test_a_matching_book_is_in_sync_and_lists_what_the_ledger_does_not_book():
    r = _run()
    assert r.in_sync
    assert [p["status"] for p in r.positions] == ["ok", "ok", "ok"]
    assert not (r.missing_in_ledger or r.not_at_broker or r.flows_missing or r.unresolved)
    # The sold BONEX round trip matches through its orderbookId alone.
    assert {t.ticker for t in _ledger()["ledger_trades"]} >= {"BONEX.ST"}
    assert [(d["ticker"], round(d["diff"], 2)) for d in r.amount_diffs] == [("KCC.OL", 3.0)]
    assert [e["category"] for e in r.cash_events] == ["Dividend", "ForeignTax"]
    assert r.explained_cash == pytest.approx(3.0 + 50.0 - 12.5)
    text = rec.format_report(r)
    assert text.startswith("✅") and "Read-only" in text


def test_cash_outside_tolerance_is_drift():
    r = _run(ledger_cash=9_900.0)
    assert not r.in_sync and r.cash_diff == pytest.approx(100.0)


def test_drift_in_shares_and_trades_is_reported_both_ways():
    trades = _ledger()["ledger_trades"]
    r = _run(
        ledger_positions={"TRUE-B.ST": 100, "KCC.OL": 30, "MIPS.ST": 5},
        ledger_trades=[t for t in trades if t.ticker != "INWI.ST"]
        + [_ledger_trade("2026-10-07", "MIPS.ST", "buy", 5, -1_900.0)],
    )
    status = {p["ticker"]: p["status"] for p in r.positions}
    assert status == {"TRUE-B.ST": "ok", "KCC.OL": "shares_differ",
                      "INWI.ST": "missing_in_ledger", "MIPS.ST": "not_at_broker"}
    assert [(t["ticker"], t["side"], t["shares"]) for t in r.missing_in_ledger] == [("INWI.ST", "buy", 40)]
    assert [t["ticker"] for t in r.not_at_broker] == ["MIPS.ST"]
    assert not r.in_sync
    text = rec.format_report(r)
    assert "INWI.ST" in text and "MIPS.ST" in text and text.startswith("⚠️")


def test_a_fill_booked_on_the_wrong_day_matches_but_is_named():
    trades = [t for t in _ledger()["ledger_trades"] if t.ticker != "INWI.ST"]
    trades.append(_ledger_trade("2026-10-06", "INWI.ST", "buy", 40, -7_287.28))
    r = _run(ledger_trades=trades)
    assert not r.missing_in_ledger and not r.not_at_broker
    assert r.date_diffs == [{"ticker": "INWI.ST", "side": "buy", "shares": 40,
                             "broker_date": "2026-10-05", "ledger_date": "2026-10-06"}]


def test_unrecorded_deposit_and_unknown_instrument_are_reported():
    snap = _snapshot()
    snap["positions"].append(_pos("ZZZ", "Mystery AB", 999, 3, 300.0, 0.0))
    r = _run(snap, ledger_flows=[])
    assert [u["orderbookId"] for u in r.unresolved] == [999]
    assert [f["amount"] for f in r.flows_missing] == [20_000.0]
    assert not r.in_sync
    assert "montrose-map 999" in rec.format_report(r)


def test_a_pinned_instrument_overrides_resolution():
    snap = _snapshot()
    snap["positions"].append(_pos("ZZZ", "Mystery AB", 999, 3, 300.0, 0.0))
    instruments = {**INSTRUMENTS, "999": {"yahoo": "MYST.ST"}}
    r = rec.reconcile(snap, **_ledger(ledger_positions={**_ledger()["ledger_positions"], "MYST.ST": 3}),
                      universe=UNIVERSE, instruments=instruments)
    assert r.in_sync


def test_foreign_cash_is_never_silently_in_sync():
    snap = _snapshot()
    snap["currency_positions"].append({"currencyCode": "NOK", "balance": 120.0})
    r = _run(snap)
    assert r.foreign_cash == [{"currency": "NOK", "balance": 120.0}] and not r.in_sync


def test_ledger_inputs_reads_the_store(tmp_path):
    store = Store(tmp_path / "fund.db")
    store.set_cash(10_000)
    store.apply_fill(Transaction("INWI.ST", "buy", 40, 182.0, 7.28, "fill", datetime(2026, 10, 5, 12)))
    store.apply_fill(Transaction("OLD.ST", "buy", 1, 10.0, 0.0, "fill", datetime(2026, 8, 1, 12)))
    got = rec.ledger_inputs(store, date(2026, 9, 27))
    assert got["ledger_positions"] == {"INWI.ST": 40, "OLD.ST": 1}
    assert [(t.ticker, t.cash_effect) for t in got["ledger_trades"]] == [("INWI.ST", pytest.approx(-7_287.28))]


# ── Refresh against a fake Montrose ──────────────────────────────────────────

class FakeMCP:
    def __init__(self, accounts=None):
        self.calls: list[tuple[str, dict]] = []
        self.accounts = accounts or [
            {"accountId": "other", "accountName": "Svenska aktier", "accountType": "ISK", "credit": None},
            {"accountId": ACCOUNT_ID, "accountName": "Simulering ", "accountType": "ISK", "credit": None},
        ]

    def handler(self, req: httpx.Request) -> httpx.Response:
        if req.method == "DELETE":
            return httpx.Response(200)
        msg = json.loads(req.content)
        if "id" not in msg:
            return httpx.Response(202)
        if msg["method"] == "initialize":
            result = {"serverInfo": {"name": "montrose-mcp"}}
        else:
            name, args = msg["params"]["name"], msg["params"]["arguments"]
            self.calls.append((name, args))
            data = self._tool(name, args)
            result = {"content": [{"type": "text", "text": json.dumps(data)}], "structuredContent": data}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": msg["id"], "result": result})

    def _tool(self, name, args):
        if name == "get_user_accounts":
            return {"result": self.accounts}
        if name == "get_holdings":
            assert args == {"accountId": ACCOUNT_ID}
            return {"result": [_holdings()]}
        if name == "get_transactions":
            assert args["accountIds"] == [ACCOUNT_ID]
            if not args.get("cursor"):
                return {"transactions": TRANSACTIONS[:4], "pagination": {"nextCursor": "p2", "totalCount": 8}}
            other = {**TRANSACTIONS[0], "accountId": "other", "id": 99}
            return {"transactions": TRANSACTIONS[4:] + [other], "pagination": {"totalCount": 8}}
        if name == "search_instruments":
            return {"result": [{"orderbookId": args["orderbookId"], "ticker": "BONEX",
                                "name": "Bonesupport Holding"}]}
        raise AssertionError(f"unexpected tool {name}")


@pytest.fixture
def fake(monkeypatch, tmp_path):
    f = FakeMCP()
    monkeypatch.setattr(m, "DATA_DIR", tmp_path)
    monkeypatch.setattr(m, "_http", lambda: httpx.Client(transport=httpx.MockTransport(f.handler)))
    monkeypatch.setattr(m, "access_token", lambda force_refresh=False: "at")
    return f


def test_refresh_reads_one_account_pages_transactions_and_saves_privately(fake):
    path, snap = rec.refresh("Simulering", date(2026, 9, 27))
    assert snap["account"]["accountId"] == ACCOUNT_ID
    assert [t["id"] for t in snap["transactions"]] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert {c for c, _ in fake.calls} <= m.READ_TOOLS
    assert ("search_instruments", {"orderbookId": 692}) in fake.calls
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    inst = rec.load_instruments()
    assert inst["692"]["ticker"] == "BONEX" and inst["1757"]["ticker"] == "TRUE"
    # A second refresh needs no lookup: the sold holding's id is cached.
    fake.calls.clear()
    rec.refresh("Simulering", date(2026, 9, 27))
    assert not any(c == "search_instruments" for c, _ in fake.calls)


def test_refresh_refuses_an_ambiguous_or_missing_account(fake):
    with pytest.raises(m.MontroseError, match="No Montrose account named 'FundManager'"):
        rec.refresh("FundManager", date(2026, 9, 27))
    fake.accounts.append({**fake.accounts[1], "accountId": "dup"})
    with pytest.raises(m.MontroseError, match="More than one"):
        rec.refresh("Simulering", date(2026, 9, 27))


def test_pin_instrument_is_kept_across_refreshes(fake):
    rec.pin_instrument(1757, "true-b.st")
    rec.refresh("Simulering", date(2026, 9, 27))
    assert rec.load_instruments()["1757"] == {"yahoo": "TRUE-B.ST", "ticker": "TRUE",
                                              "name": "TRUECALLER AK B", "currency": "SEK"}


# ── CLI ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def cli_env(fake, tmp_path, monkeypatch):
    db = tmp_path / "fund.db"
    cfg = tmp_path / "cfg.yaml"
    uni = tmp_path / "universe.csv"
    uni.write_text("name,yahoo_ticker,isin,country,exchange,sector,enabled\n"
                   + "".join(f"{u.name},{u.yahoo_ticker},,SE,{u.exchange},,true\n" for u in UNIVERSE))
    cfg.write_text(f"db_path: {db}\nuniverse_path: {uni}\nmontrose:\n  account_name: Simulering\n"
                   "  lookback_days: 30\n")
    monkeypatch.setenv("FUND_CONFIG", str(cfg))
    store = Store(db)
    store.set_cash(30_000.0)
    for day, t, side, n, px, fee in [("2026-10-02", "TRUE-B.ST", "buy", 100, 25.0, 2.5),
                                     ("2026-10-02", "KCC.OL", "buy", 35, 94.2857, 3.0),
                                     ("2026-10-05", "INWI.ST", "buy", 40, 182.0, 7.28),
                                     ("2026-10-06", "BONEX.ST", "buy", 10, 300.0, 3.0),
                                     ("2026-10-08", "BONEX.ST", "sell", 10, 310.0, 3.1)]:
        store.apply_fill(Transaction(t, side, n, px, fee, "fill",
                                     datetime.fromisoformat(day + "T12:00:00")))
    sent: list[str] = []
    from fundmgr.notify import send
    monkeypatch.setattr(send, "send_telegram", lambda text, parse_mode="HTML": sent.append(text) or True)
    monkeypatch.setattr(rec, "default_since", lambda days, today=None: date(2026, 9, 27))
    return db, sent


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reconcile_command_writes_nothing_to_the_fund(cli_env):
    from fundmgr.cli import cli

    db, sent = cli_env
    before = _digest(db)
    res = CliRunner().invoke(cli, ["montrose-reconcile", "--notify"])
    assert res.exit_code == 0, res.output
    assert "3/3 match" in res.output and "Read-only" in res.output
    assert _digest(db) == before
    # The deposit isn't in this ledger, so the report is off and is sent.
    assert "Deposit" in res.output and sent and "Deposit" in sent[0]


def test_reconcile_offline_uses_the_saved_snapshot_and_json(cli_env, fake):
    from fundmgr.cli import cli

    CliRunner().invoke(cli, ["montrose-refresh"])
    fake.calls.clear()
    res = CliRunner().invoke(cli, ["montrose-reconcile", "--offline", "--json"])
    assert res.exit_code == 0, res.output
    assert fake.calls == []
    out = json.loads(res.output)
    assert out["account"] == "Simulering" and out["cash_diff"] == pytest.approx(10_000.0 - out["ledger_cash"])


def test_login_lapse_is_notified(cli_env, monkeypatch):
    from fundmgr.cli import cli

    _, sent = cli_env

    def lapsed(force_refresh=False):
        raise m.LoginRequired("Montrose token expired and there is no refresh token.")

    monkeypatch.setattr(m, "access_token", lapsed)
    res = CliRunner().invoke(cli, ["montrose-reconcile", "--notify"])
    assert res.exit_code != 0
    assert sent and "/montrose_login" in sent[0]
