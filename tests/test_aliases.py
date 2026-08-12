"""Learned ticker aliases: correct a symbol once, have it applied everywhere."""
import json

import pytest

from fundmgr import aliases
from fundmgr.vision import portfolio_photo as pp


@pytest.fixture(autouse=True)
def alias_file(tmp_path, monkeypatch):
    """Isolate the alias store from the real DATA_DIR."""
    path = tmp_path / "ticker_aliases.json"
    monkeypatch.setenv("FUND_ALIAS_PATH", str(path))
    return path


# ── Keys ──────────────────────────────────────────────────────────────────────

def test_isin_key_is_normalised():
    aliases.learn("VOLV-B.ST", isin=" se0000115446 ")
    assert aliases.lookup(isin="SE0000115446")[0] == "VOLV-B.ST"


def test_malformed_isin_is_not_stored():
    aliases.learn("VOLV-B.ST", isin="SE00001154")     # 10 chars
    assert aliases.lookup(isin="SE00001154") is None


def test_name_key_is_normalised_the_same_way_as_resolution():
    aliases.learn("VSSAB-B.ST", name="Vestum AB (publ)")
    assert aliases.lookup(name="vestum  ab   publ")[0] == "VSSAB-B.ST"


def test_a_too_short_name_is_not_keyed():
    aliases.learn("AB.ST", name="AB")
    assert aliases.lookup(name="AB") is None


def test_broker_alias_equal_to_the_answer_is_not_stored():
    """'NVDA'→'NVDA' is noise and would shadow a later correction."""
    written = aliases.learn("NVDA", raw_ticker="NVDA", name="Nvidia")
    assert "broker" not in written
    assert "name" in written


def test_learn_requires_a_ticker():
    assert aliases.learn("", isin="SE0000115446") == []


# ── Precedence ────────────────────────────────────────────────────────────────

def test_isin_beats_broker_and_name():
    aliases.learn("RIGHT.ST", isin="SE0000115446")
    aliases.learn("WRONG.F", raw_ticker="4HY", name="Some Co")
    hit = aliases.lookup(isin="SE0000115446", raw_ticker="4HY", name="Some Co")
    assert hit == ("RIGHT.ST", "isin")


def test_broker_beats_name():
    aliases.learn("BYTICKER.ST", raw_ticker="4HY")
    aliases.learn("BYNAME.ST", name="Ambiguous Holdings")
    assert aliases.lookup(raw_ticker="4HY", name="Ambiguous Holdings")[1] == "broker"


def test_lookup_misses_cleanly():
    assert aliases.lookup(isin="SE0000000000", name="Nothing Here") is None
    assert aliases.lookup() is None


# ── Persistence ───────────────────────────────────────────────────────────────

def test_alias_survives_a_reload(alias_file):
    aliases.learn("VSSAB-B.ST", name="Vestum", raw_ticker="4HY")
    assert alias_file.exists()
    stored = json.loads(alias_file.read_text())
    assert stored["broker"]["4HY"]["ticker"] == "VSSAB-B.ST"
    assert aliases.lookup(raw_ticker="4HY")[0] == "VSSAB-B.ST"


def test_relearning_overwrites_but_keeps_the_first_seen_date():
    aliases.learn("WRONG.F", raw_ticker="4HY")
    first = aliases.load()["broker"]["4HY"]["first_learned_at"]
    aliases.learn("VSSAB-B.ST", raw_ticker="4HY")
    entry = aliases.load()["broker"]["4HY"]
    assert entry["ticker"] == "VSSAB-B.ST"        # newest correction wins
    assert entry["first_learned_at"] == first


def test_corrupt_alias_file_is_survivable(alias_file):
    alias_file.write_text("{ not json")
    assert aliases.load() == {"isin": {}, "broker": {}, "name": {}}
    assert aliases.lookup(name="Anything") is None
    aliases.learn("VOLV-B.ST", name="Volvo B")     # and recovers on the next write
    assert aliases.lookup(name="Volvo B")[0] == "VOLV-B.ST"


def test_forget_removes_every_kind():
    aliases.learn("VSSAB-B.ST", raw_ticker="4HY", name="4HY")
    assert aliases.forget("4HY")
    assert aliases.lookup(raw_ticker="4HY") is None
    assert aliases.forget("4HY") == []


def test_entries_lists_everything_stored():
    aliases.learn("VOLV-B.ST", isin="SE0000115446", name="Volvo B")
    kinds = {e["kind"] for e in aliases.entries()}
    assert kinds == {"isin", "name"}


# ── Resolution uses them first ────────────────────────────────────────────────

def _row(**kw):
    base = {"name": "", "raw_ticker": "", "isin": "", "ticker": None,
            "currency": "SEK", "resolved_via": None}
    base.update(kw)
    return base


def test_a_learned_alias_beats_the_universe_isin_map():
    """A human correction outranks every automatic guess, including universe.csv."""
    aliases.learn("CORRECTED.ST", isin="SE0000115446")
    out = pp._resolve_tickers([_row(isin="SE0000115446", name="Volvo B")])
    assert out[0]["ticker"] == "CORRECTED.ST"
    assert out[0]["resolved_via"] == "learned (isin)"


def test_a_learned_alias_beats_the_broker_map():
    aliases.learn("KOG-CORRECTED.OL", raw_ticker="KOG")
    out = pp._resolve_tickers([_row(raw_ticker="KOG", name="Kongsberg")])
    assert out[0]["ticker"] == "KOG-CORRECTED.OL"


def test_a_learned_name_rescues_a_row_search_could_not_resolve(monkeypatch):
    monkeypatch.setattr(pp, "_search_symbol_preferring", lambda name, currency: None)
    unknown = _row(name="Obscure Nordic Holdings")
    assert pp._resolve_tickers([dict(unknown)])[0]["ticker"] is None

    aliases.learn("OBSC.ST", name="Obscure Nordic Holdings")
    out = pp._resolve_tickers([dict(unknown)])
    assert out[0]["ticker"] == "OBSC.ST"
    assert out[0]["resolved_via"] == "learned (name)"


def test_resolution_without_aliases_is_unchanged():
    out = pp._resolve_tickers([_row(isin="SE0000115446", name="Volvo B")])
    assert out[0]["ticker"] == "VOLV-B.ST"
    assert out[0]["resolved_via"] == "isin"


# ── Learned on import ─────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from fundmgr import paper

    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path / "paper")
    monkeypatch.setattr(paper, "detect_currency", lambda t: "SEK")
    monkeypatch.setattr(paper, "_cache_price_history", lambda store, tickers, **kw: None)
    monkeypatch.setattr(paper, "_search_symbol", lambda name: None)
    import fundmgr.data.benchmark as benchmark
    monkeypatch.setattr(benchmark, "fetch_and_cache_benchmark", lambda store, **kw: True)
    import fundmgr.data.quotes as quotes
    monkeypatch.setattr(quotes, "live_prices", lambda tickers: {t: 300.0 for t in tickers})

    from fundmgr.web.app import app
    return TestClient(app)


def test_import_learns_the_corrected_ticker(client):
    rows = [{
        "ticker": "VSSAB-B.ST",          # what the human typed
        "suggested_ticker": "4HY.F",     # what extraction had guessed
        "isin": "SE0007278523", "raw_ticker": "4HY", "name": "Vestum",
        "shares": "100", "avg_cost_sek": "20",
    }]
    r = client.post("/live/create-from-photo", data={
        "name": "Learn Sleeve", "holdings_json": json.dumps(rows), "cash_sek": "0",
    }, follow_redirects=False)
    assert r.status_code == 303
    assert "Remembered+1+ticker+correction" in r.headers["location"]

    # All three identifiers now point at the corrected symbol.
    assert aliases.lookup(isin="SE0007278523")[0] == "VSSAB-B.ST"
    assert aliases.lookup(raw_ticker="4HY")[0] == "VSSAB-B.ST"
    assert aliases.lookup(name="Vestum")[0] == "VSSAB-B.ST"


def test_an_uncorrected_row_is_still_learned_but_not_counted(client):
    rows = [{
        "ticker": "VOLV-B.ST", "suggested_ticker": "VOLV-B.ST",
        "isin": "SE0000115446", "raw_ticker": "VOLV B", "name": "Volvo B",
        "shares": "100", "avg_cost_sek": "300",
    }]
    r = client.post("/live/create-from-photo", data={
        "name": "Quiet Sleeve", "holdings_json": json.dumps(rows), "cash_sek": "0",
    }, follow_redirects=False)
    assert "Remembered" not in r.headers["location"]
    assert aliases.lookup(isin="SE0000115446")[0] == "VOLV-B.ST"


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli(*args):
    from click.testing import CliRunner
    from fundmgr.cli import cli
    return CliRunner().invoke(cli, ["aliases", *args])


def test_cli_lists_nothing_when_empty():
    assert "No learned aliases yet" in _cli().output


def test_cli_sets_and_lists_and_forgets():
    out = _cli("--set", "SE0000115446", "VOLV-B.ST").output
    assert "SE0000115446 → VOLV-B.ST" in out
    assert aliases.lookup(isin="SE0000115446")[0] == "VOLV-B.ST"

    listing = _cli().output
    assert "isin" in listing and "VOLV-B.ST" in listing

    assert "Forgot" in _cli("--forget", "SE0000115446").output
    assert aliases.lookup(isin="SE0000115446") is None


def test_cli_set_accepts_a_company_name():
    _cli("--set", "Obscure Nordic Holdings", "OBSC.ST")
    assert aliases.lookup(name="Obscure Nordic Holdings")[0] == "OBSC.ST"


def test_cli_forget_reports_a_miss():
    assert "No learned alias matching" in _cli("--forget", "NOPE").output
