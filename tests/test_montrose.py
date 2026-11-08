"""
Tests for the read-only Montrose MCP client (phase 0).

No network: every HTTP call goes through an httpx.MockTransport that plays
Montrose's OAuth and MCP endpoints.

Run: pytest tests/test_montrose.py -v
"""
from __future__ import annotations

import json
import stat
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from fundmgr.broker import montrose as m  # noqa: E402

META = {
    "issuer": m.ISSUER,
    "authorization_endpoint": f"{m.ISSUER}/authorize",
    "token_endpoint": f"{m.ISSUER}/token",
    "registration_endpoint": f"{m.ISSUER}/register",
}


class FakeMontrose:
    def __init__(self) -> None:
        self.token_calls: list[dict] = []
        self.tool_calls: list[str] = []
        self.tool_args: list[dict] = []
        self.registrations = 0
        self.mcp_401_once = False
        self.sse = False

    def handler(self, req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/.well-known/oauth-authorization-server":
            return httpx.Response(200, json=META)
        if path == "/register":
            self.registrations += 1
            body = json.loads(req.content)
            return httpx.Response(201, json={"client_id": "cid", "client_secret": "csecret",
                                             "redirect_uris": body["redirect_uris"]})
        if path == "/token":
            form = {k: v[0] for k, v in parse_qs(req.content.decode()).items()}
            self.token_calls.append(form)
            n = len(self.token_calls)
            return httpx.Response(200, json={"access_token": f"at{n}", "refresh_token": f"rt{n}",
                                             "expires_in": 3600, "token_type": "Bearer"})
        if path == "/":
            if req.method == "DELETE":
                return httpx.Response(200)
            if self.mcp_401_once:
                self.mcp_401_once = False
                return httpx.Response(401)
            msg = json.loads(req.content)
            if "id" not in msg:
                return httpx.Response(202)
            result = self._result(msg)
            reply = {"jsonrpc": "2.0", "id": msg["id"], "result": result}
            if self.sse:
                return httpx.Response(200, headers={"content-type": "text/event-stream",
                                                    "mcp-session-id": "s1"},
                                      text=f"event: message\ndata: {json.dumps(reply)}\n\n")
            return httpx.Response(200, json=reply, headers={"mcp-session-id": "s1"})
        return httpx.Response(404)

    def _result(self, msg: dict) -> dict:
        if msg["method"] == "initialize":
            return {"serverInfo": {"name": "montrose"}}
        if msg["method"] == "tools/list":
            # Shapes as in Montrose's real tools/list: every argument optional.
            return {"tools": [
                {"name": "get_user_accounts", "inputSchema": {"type": "object", "properties": {}}},
                {"name": "get_transactions_categories",
                 "inputSchema": {"type": "object", "properties": {}}},
                {"name": "get_holdings", "inputSchema": {
                    "type": "object", "properties": {"accountId": {"type": ["string", "null"]}}}},
                {"name": "get_transactions", "inputSchema": {
                    "type": "object", "properties": {"from": {}, "to": {}, "pageSize": {}}}},
                {"name": "search_instruments", "inputSchema": {
                    "type": "object", "properties": {"query": {}, "orderbookId": {}}}},
                {"name": "create_trade_ticket", "inputSchema": {"type": "object"}},
            ]}
        name = msg["params"]["name"]
        self.tool_calls.append(name)
        self.tool_args.append(msg["params"]["arguments"])
        data = {
            "get_user_accounts": {"result": [{"accountId": "3fa85f64", "accountNumber": "123456",
                                              "accountType": "ISK"}]},
            "get_transactions_categories": {"result": [{"type": "Buy", "description": "Köp"}]},
            "get_holdings": {"positions": [{"name": "Volvo B", "quantity": 12, "marketValue": 3500.0}]},
            "get_transactions": {"items": [{"category": "Buy", "price": 291.5}]},
        }[name]
        return {"content": [{"type": "text", "text": json.dumps(data)}]}


@pytest.fixture
def fake(monkeypatch, tmp_path):
    f = FakeMontrose()
    monkeypatch.setattr(m, "DATA_DIR", tmp_path)
    monkeypatch.setattr(m, "_http", lambda: httpx.Client(transport=httpx.MockTransport(f.handler)))
    return f


def _callback(url: str, **override: str) -> str:
    q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
    params = {"code": "thecode", "state": q["state"], "iss": m.ISSUER, **override}
    return f"{m.REDIRECT_URI}?" + "&".join(f"{k}={v}" for k, v in params.items())


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_login_round_trip_with_pkce_and_private_files(fake):
    url = m.start_login()
    q = parse_qs(urlparse(url).query)
    assert url.startswith(f"{m.ISSUER}/authorize?")
    assert q["redirect_uri"] == [m.REDIRECT_URI]
    assert q["code_challenge_method"] == ["S256"]
    assert q["resource"] == [m.RESOURCE]

    status = m.complete_login("Here you go: " + _callback(url) + " thanks")
    assert status["logged_in"] and status["refresh_token"]
    form = fake.token_calls[0]
    assert form["grant_type"] == "authorization_code"
    assert form["code"] == "thecode"
    assert form["client_secret"] == "csecret"
    assert len(form["code_verifier"]) >= 43

    d = m.state_dir()
    assert _mode(d) == 0o700
    for name in ("client.json", "token.json"):
        assert _mode(d / name) == 0o600
    assert json.loads((d / "pending_login.json").read_text()) == {}
    assert _mode(d / "pending_login.json") == 0o600


def test_registration_is_reused(fake):
    m.start_login()
    m.start_login()
    assert fake.registrations == 1


def test_state_mismatch_is_refused(fake):
    url = m.start_login()
    with pytest.raises(m.MontroseError, match="doesn't match"):
        m.complete_login(_callback(url, state="forged"))
    assert fake.token_calls == []


def test_an_earlier_link_still_works_after_a_newer_one(fake):
    first = m.start_login()
    second = m.start_login()
    m.complete_login(_callback(first))
    first_state = parse_qs(urlparse(first).query)["state"][0]
    assert first_state not in json.loads((m.state_dir() / "pending_login.json").read_text())
    m.complete_login(_callback(second))
    assert len(fake.token_calls) == 2
    with pytest.raises(m.MontroseError):
        m.complete_login(_callback(first))


def test_wrong_issuer_is_refused(fake):
    url = m.start_login()
    with pytest.raises(m.MontroseError, match="issuer"):
        m.complete_login(_callback(url, iss="https://evil.example"))
    assert fake.token_calls == []


def test_expired_pending_login_is_refused(fake, monkeypatch):
    url = m.start_login()
    later = m._now() + m.PENDING_TTL_S + 1
    monkeypatch.setattr(m, "_now", lambda: later)
    with pytest.raises(m.MontroseError, match="expired"):
        m.complete_login(_callback(url))


def test_callback_is_single_use(fake):
    url = m.start_login()
    m.complete_login(_callback(url))
    with pytest.raises(m.MontroseError, match="No login in progress"):
        m.complete_login(_callback(url))


def test_error_in_callback_is_reported(fake):
    m.start_login()
    with pytest.raises(m.MontroseError, match="access_denied"):
        m.complete_login(f"{m.REDIRECT_URI}?error=access_denied&state=x")


def test_not_logged_in_raises_login_required(fake):
    with pytest.raises(m.LoginRequired):
        m.access_token()


def test_expired_token_refreshes_and_rotates(fake, monkeypatch):
    m.complete_login(_callback(m.start_login()))
    later = m._now() + 3600
    monkeypatch.setattr(m, "_now", lambda: later)
    assert m.access_token() == "at2"
    assert fake.token_calls[-1]["grant_type"] == "refresh_token"
    assert fake.token_calls[-1]["refresh_token"] == "rt1"
    assert m.token_status()["refresh_count"] == 1


def test_write_tools_are_refused_without_a_request(fake):
    m.complete_login(_callback(m.start_login()))
    with m.Session() as s:
        with pytest.raises(PermissionError):
            s.call_tool("create_trade_ticket", {"side": "buy"})
    assert fake.tool_calls == []


def test_401_refreshes_once(fake):
    m.complete_login(_callback(m.start_login()))
    fake.mcp_401_once = True
    with m.Session() as s:
        assert s.call_tool("get_user_accounts")["result"][0]["accountType"] == "ISK"
    assert fake.token_calls[-1]["grant_type"] == "refresh_token"


def test_sse_responses_are_parsed(fake):
    m.complete_login(_callback(m.start_login()))
    fake.sse = True
    with m.Session() as s:
        assert [t["name"] for t in s.list_tools()][0] == "get_user_accounts"


def test_probe_reads_only_and_summary_hides_values(fake):
    m.complete_login(_callback(m.start_login()))
    path, summary = m.probe()
    assert fake.tool_calls == ["get_user_accounts", "get_transactions_categories",
                               "get_holdings", "get_transactions"]
    assert fake.tool_args[2] == {}
    assert set(fake.tool_args[3]) == {"from", "pageSize"}
    assert _mode(path) == 0o600
    raw = json.loads(path.read_text())
    assert raw["calls"]["get_holdings"]["result"]["positions"][0]["name"] == "Volvo B"

    text = json.dumps(summary)
    for secret in ("123456", "3fa85f64", "Volvo", "3500", "291.5"):
        assert secret not in text
    assert summary["calls"]["get_holdings"]["shape"] == {
        "positions": [{"name": "str", "quantity": "int", "marketValue": "float"}, "×1"]}
    assert summary["calls"]["get_transactions"]["args"] == ["from", "pageSize"]
    assert summary["read_tools_missing"] == []
    assert "create_trade_ticket" in summary["tools"]
