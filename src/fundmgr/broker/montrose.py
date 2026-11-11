"""
Montrose MCP client.

Montrose exposes the real ISK/KF account over MCP at mcp.montrose.io with
standard MCP OAuth (dynamic client registration, PKCE S256, client secret,
refresh tokens). This module talks to it from plain code; no model sits in
the loop. Montrose's terms allow own tools and scheduled jobs; orders are
always approved by the owner in the Montrose app.

There is no read-only login: any token can also create trade tickets. So the
limits live here. `call_tool` refuses anything outside `READ_TOOLS`. The one
write — a trade *ticket*, a pre-filled order screen the owner still has to
approve in the app — has its own method, `create_trade_ticket`, which takes
only an orderbookId, a whole share count and an account, and returns the URL
only if it is Montrose's own https trade page. Watchlists, price alerts and
anything else stay unreachable.

Login renewal has to work from a phone away from home. The Pi keeps the PKCE
verifier and state; the owner opens the authorize link on the phone and does
BankID there. Montrose then redirects to http://localhost:8765/callback, which
fails to load on the phone — but the code is already in the address bar, and
pasting that URL to the Telegram bot completes the exchange on the Pi. The
code is single-use, short-lived and worthless without the verifier, which
never leaves the Pi. A localhost redirect is used because the forum reports
Montrose rejecting non-localhost return addresses for self-built clients.

Everything is stored under DATA_DIR/montrose/ with 0600 files in a 0700
directory: the client registration, the token, the pending login, and probe
dumps (which contain the real holdings).
"""
from __future__ import annotations

import base64
import contextlib
import fcntl
import hashlib
import json
import os
import secrets
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from fundmgr.config import DATA_DIR

ISSUER = "https://mcp.montrose.io"
RESOURCE = "https://mcp.montrose.io"
# The MCP endpoint is the server root; /mcp answers 404 once authenticated.
MCP_URL = "https://mcp.montrose.io"
REDIRECT_URI = "http://localhost:8765/callback"
SCOPE = "mcp"
CLIENT_NAME = "fundmgr (ai-fund-manager)"
PROTOCOL_VERSION = "2025-06-18"

READ_TOOLS = frozenset({
    "get_user_accounts", "get_holdings", "get_transactions", "get_transactions_categories",
    # Transactions name instruments by orderbookId only; this turns a sold
    # holding's id back into a ticker. A lookup, it changes nothing.
    "search_instruments",
})

TICKET_TOOL = "create_trade_ticket"
# Seen live 2026-10-28: https://app.montrose.io/trade?ticketId=<id>.
TICKET_HOST = "app.montrose.io"
TICKET_PATH = "/trade"

PENDING_TTL_S = 10 * 60
REFRESH_MARGIN_S = 60
TIMEOUT_S = 30.0


class MontroseError(RuntimeError):
    pass


class LoginRequired(MontroseError):
    """No usable token: the owner has to do BankID again (/montrose_login)."""


# ── Storage ──────────────────────────────────────────────────────────────────

def state_dir() -> Path:
    d = DATA_DIR / "montrose"
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    return d


def _write_private(path: Path, data: dict | list) -> None:
    """Atomic write, created 0600 — never world-readable, not even briefly."""
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@contextlib.contextmanager
def _lock() -> Iterator[None]:
    """Serialise token refresh: refresh tokens rotate, and two cron jobs
    refreshing at once would leave one holding a revoked token."""
    fd = os.open(state_dir() / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _http() -> httpx.Client:
    return httpx.Client(timeout=TIMEOUT_S, follow_redirects=False)


def _now() -> float:
    return time.time()


# ── OAuth ────────────────────────────────────────────────────────────────────

def _metadata(client: httpx.Client) -> dict:
    r = client.get(f"{ISSUER}/.well-known/oauth-authorization-server")
    r.raise_for_status()
    meta = r.json()
    if meta.get("issuer") != ISSUER:
        raise MontroseError(f"Unexpected issuer in metadata: {meta.get('issuer')!r}")
    for key in ("authorization_endpoint", "token_endpoint", "registration_endpoint"):
        if not str(meta.get(key, "")).startswith(ISSUER + "/"):
            raise MontroseError(f"{key} is not on {ISSUER}: {meta.get(key)!r}")
    return meta


def _client_registration(client: httpx.Client, meta: dict) -> dict:
    path = state_dir() / "client.json"
    reg = _read(path)
    if reg and reg.get("client_id") and REDIRECT_URI in reg.get("redirect_uris", []):
        expires = reg.get("client_secret_expires_at") or 0
        if not expires or expires > _now() + 3600:
            return reg
    r = client.post(meta["registration_endpoint"], json={
        "client_name": CLIENT_NAME,
        "redirect_uris": [REDIRECT_URI],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "client_secret_post",
        "scope": SCOPE,
    })
    if r.status_code >= 400:
        raise MontroseError(f"Client registration refused ({r.status_code}): {r.text[:300]}")
    reg = r.json()
    if not reg.get("client_id"):
        raise MontroseError("Client registration returned no client_id")
    reg.setdefault("redirect_uris", [REDIRECT_URI])
    reg["registered_at"] = _now()
    _write_private(path, reg)
    return reg


def _pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode()).digest()
    return verifier, base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def start_login() -> str:
    """Register (once), stash verifier + state, return the URL to open on the phone."""
    with _http() as client:
        meta = _metadata(client)
        reg = _client_registration(client, meta)
    verifier, challenge = _pkce()
    state = secrets.token_urlsafe(24)
    # Keyed by state, so every link sent in the last 10 minutes still works:
    # the first live test pasted the callback of an earlier /montrose_login,
    # which a single pending slot had already overwritten.
    path = state_dir() / "pending_login.json"
    pending = _live_pending(_read(path))
    pending[state] = {"verifier": verifier, "created_at": _now()}
    _write_private(path, pending)
    return meta["authorization_endpoint"] + "?" + urlencode({
        "response_type": "code",
        "client_id": reg["client_id"],
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "resource": RESOURCE,
    })


def _live_pending(pending: dict | None) -> dict:
    now = _now()
    return {s: p for s, p in (pending or {}).items()
            if isinstance(p, dict) and now - p.get("created_at", 0) <= PENDING_TTL_S}


def complete_login(pasted: str) -> dict:
    """Exchange the code in a pasted callback URL. Returns token_status()."""
    start = pasted.find("http")
    url = pasted[start:].split()[0] if start >= 0 else pasted.strip()
    params = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
    if "error" in params:
        raise MontroseError(f"Montrose refused the login: {params['error']} "
                            f"{params.get('error_description', '')}".strip())

    path = state_dir() / "pending_login.json"
    stored = _read(path) or {}
    pending = _live_pending(stored)
    state = params.get("state", "")
    if not stored:
        raise MontroseError("No login in progress — start one with /montrose_login.")
    if state not in pending:
        if state in stored:
            reason = "That login link expired (10 min)."
        elif not pending:
            reason = "No login started in the last 10 minutes."
        else:
            reason = ("This address doesn't match any login link sent in the last 10 minutes "
                      "(was the whole address pasted?).")
        raise MontroseError(f"{reason} Send /montrose_login and use that newest link.")
    if "iss" in params and params["iss"] != ISSUER:
        raise MontroseError(f"Unexpected issuer in callback: {params['iss']!r}")
    code = params.get("code")
    if not code:
        raise MontroseError("No code in that URL — paste the whole address.")
    # Single use either way: a failed exchange must not leave a reusable verifier.
    verifier = pending.pop(state)["verifier"]
    _write_private(path, pending)

    with _http() as client:
        meta = _metadata(client)
        reg = _read(state_dir() / "client.json") or {}
        tok = _token_request(client, meta, reg, {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "code_verifier": verifier,
            "resource": RESOURCE,
        })
    with _lock():
        _save_token(tok, previous=None)
    return token_status()


def _token_request(client: httpx.Client, meta: dict, reg: dict, form: dict) -> dict:
    if not reg.get("client_id"):
        raise LoginRequired("No client registration — run /montrose_login.")
    form = {**form, "client_id": reg["client_id"]}
    auth = None
    if reg.get("token_endpoint_auth_method") == "client_secret_basic":
        auth = (reg["client_id"], reg.get("client_secret", ""))
    elif reg.get("client_secret"):
        form["client_secret"] = reg["client_secret"]
    r = client.post(meta["token_endpoint"], data=form, auth=auth)
    if r.status_code in (400, 401):
        raise LoginRequired(f"Token endpoint refused ({r.status_code}): {r.text[:300]}")
    r.raise_for_status()
    tok = r.json()
    if not tok.get("access_token"):
        raise MontroseError("Token response had no access_token")
    return tok


def _save_token(tok: dict, previous: dict | None) -> dict:
    now = _now()
    saved = {
        "access_token": tok["access_token"],
        # Not every server rotates; keep the old refresh token when none comes back.
        "refresh_token": tok.get("refresh_token") or (previous or {}).get("refresh_token"),
        "token_type": tok.get("token_type", "Bearer"),
        "scope": tok.get("scope"),
        "obtained_at": now,
        "expires_at": now + float(tok["expires_in"]) if tok.get("expires_in") else None,
        "refresh_expires_at": (now + float(tok["refresh_token_expires_in"])
                               if tok.get("refresh_token_expires_in")
                               else (previous or {}).get("refresh_expires_at")),
        "login_at": (previous or {}).get("login_at", now),
        "refresh_count": (previous or {}).get("refresh_count", -1) + 1,
        "extra_fields": sorted(set(tok) - {"access_token", "refresh_token"}),
    }
    _write_private(state_dir() / "token.json", saved)
    return saved


def access_token(force_refresh: bool = False) -> str:
    with _lock():
        tok = _read(state_dir() / "token.json")
        if not tok:
            raise LoginRequired("Not logged in to Montrose — run /montrose_login.")
        fresh = tok.get("expires_at") and tok["expires_at"] - REFRESH_MARGIN_S > _now()
        if fresh and not force_refresh:
            return tok["access_token"]
        if not tok.get("refresh_token"):
            raise LoginRequired("Montrose token expired and there is no refresh token.")
        with _http() as client:
            meta = _metadata(client)
            reg = _read(state_dir() / "client.json") or {}
            new = _token_request(client, meta, reg, {
                "grant_type": "refresh_token",
                "refresh_token": tok["refresh_token"],
                "resource": RESOURCE,
            })
        return _save_token(new, previous=tok)["access_token"]


def _iso(ts: float | None) -> str | None:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds") if ts else None


def token_status() -> dict:
    tok = _read(state_dir() / "token.json")
    if not tok:
        return {"logged_in": False}
    return {
        "logged_in": True,
        "login_at": _iso(tok.get("login_at")),
        "access_expires_at": _iso(tok.get("expires_at")),
        "refresh_token": bool(tok.get("refresh_token")),
        "refresh_expires_at": _iso(tok.get("refresh_expires_at")),
        "refresh_count": tok.get("refresh_count", 0),
        "token_fields": tok.get("extra_fields", []),
    }


# ── MCP over streamable HTTP ─────────────────────────────────────────────────

def _parse_rpc(r: httpx.Response, want_id: int) -> dict:
    ctype = r.headers.get("content-type", "")
    if "text/event-stream" in ctype:
        for block in r.text.split("\n\n"):
            data = "\n".join(ln[5:].lstrip() for ln in block.splitlines() if ln.startswith("data:"))
            if not data:
                continue
            msg = json.loads(data)
            if msg.get("id") == want_id:
                return msg
        raise MontroseError(f"No response with id {want_id} in event stream")
    return r.json()


class Session:
    """One MCP session. Refreshes the token once on a 401, then gives up."""

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or _http()
        self._session_id: str | None = None
        self._next_id = 0
        self._token = access_token()
        self.server_info: dict = {}

    def __enter__(self) -> Session:
        init = self._rpc("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "fundmgr", "version": "0.1"},
        })
        self.server_info = init
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return self

    def __exit__(self, *exc: object) -> None:
        if self._session_id:
            with contextlib.suppress(Exception):
                self._client.delete(MCP_URL, headers=self._headers())
        self._client.close()

    def _headers(self) -> dict:
        h = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if self._session_id:
            h["Mcp-Session-Id"] = self._session_id
            h["MCP-Protocol-Version"] = PROTOCOL_VERSION
        return h

    def _post(self, body: dict) -> httpx.Response:
        r = self._client.post(MCP_URL, json=body, headers=self._headers())
        if r.status_code == 401:
            self._token = access_token(force_refresh=True)
            r = self._client.post(MCP_URL, json=body, headers=self._headers())
        if r.status_code == 401:
            raise LoginRequired("Montrose rejected the token after a refresh — log in again.")
        if r.status_code >= 400:
            raise MontroseError(f"MCP HTTP {r.status_code}: {r.text[:300]}")
        if r.headers.get("mcp-session-id"):
            self._session_id = r.headers["mcp-session-id"]
        return r

    def _rpc(self, method: str, params: dict) -> dict:
        self._next_id += 1
        msg = _parse_rpc(self._post({"jsonrpc": "2.0", "id": self._next_id,
                                     "method": method, "params": params}), self._next_id)
        if "error" in msg:
            raise MontroseError(f"{method}: {msg['error']}")
        return msg.get("result", {})

    def list_tools(self) -> list[dict]:
        return self._rpc("tools/list", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict | None = None) -> Any:
        if name not in READ_TOOLS:
            raise PermissionError(f"{name} is not a read-only tool; refusing to call it")
        result = self._rpc("tools/call", {"name": name, "arguments": arguments or {}})
        if result.get("isError"):
            raise MontroseError(f"{name}: {_text_content(result)[:500]}")
        if "structuredContent" in result:
            return result["structuredContent"]
        text = _text_content(result)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def create_trade_ticket(self, side: str, orderbook_id: int, quantity: int,
                            account_id: str) -> str:
        """A pre-filled order screen in the Montrose app; the owner approves it there.

        Only the safest identifiers: an orderbookId (never a ticker or name,
        which Montrose itself calls ambiguous), a whole share count (never an
        amount, whose currency it would guess) and an explicit account, so a
        ticket can't land in the wrong ISK. No price: the owner sets the limit.
        """
        if side not in ("Buy", "Sell"):
            raise ValueError(f"side must be Buy or Sell, got {side!r}")
        if isinstance(orderbook_id, bool) or not isinstance(orderbook_id, int) or orderbook_id <= 0:
            raise ValueError(f"orderbookId must be a positive int, got {orderbook_id!r}")
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            raise ValueError(f"quantity must be a positive whole number, got {quantity!r}")
        if not account_id:
            raise ValueError("a ticket needs the account it is for")
        result = self._rpc("tools/call", {"name": TICKET_TOOL, "arguments": {
            "side": side, "orderbookId": orderbook_id, "quantity": quantity,
            "accountId": account_id}})
        if result.get("isError"):
            raise MontroseError(f"{TICKET_TOOL}: {_text_content(result)[:500]}")
        url = str((result.get("structuredContent") or {}).get("url") or "")
        u = urlparse(url)
        if u.scheme != "https" or u.netloc != TICKET_HOST or u.path != TICKET_PATH:
            raise MontroseError(f"Unexpected ticket URL from Montrose: {url[:120]!r}")
        return url


def _text_content(result: dict) -> str:
    return "\n".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")


# ── Phase 0 probe ────────────────────────────────────────────────────────────

def _account_ids(obj: Any) -> list[str]:
    found: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (str, int)) and k.lower().replace("_", "") in ("accountid", "id", "accountnumber"):
                found.append(str(v))
            else:
                found += _account_ids(v)
    elif isinstance(obj, list):
        for item in obj:
            found += _account_ids(item)
    return list(dict.fromkeys(found))


def _guess_args(schema: dict, account_id: str | None) -> tuple[dict, list[str]]:
    """Fill required arguments we can infer; report the ones we can't."""
    props = schema.get("properties", {}) or {}
    args: dict = {}
    missing: list[str] = []
    today = datetime.now(timezone.utc).date()
    for name in schema.get("required", []) or []:
        low = name.lower().replace("_", "")
        is_date = "date" in low or low in ("from", "to", "start", "end", "since", "until")
        if "account" in low and account_id is not None:
            args[name] = account_id
        elif is_date and low.startswith(("from", "start", "since")):
            args[name] = (today - timedelta(days=90)).isoformat()
        elif is_date and low.startswith(("to", "end", "until")):
            args[name] = today.isoformat()
        else:
            missing.append(f"{name} ({props.get(name, {}).get('type', '?')})")
    return args, missing


def shape(obj: Any, depth: int = 0, max_depth: int = 6) -> Any:
    """Keys and types only — safe to paste into a chat; no amounts, no names."""
    if depth >= max_depth:
        return "…"
    if isinstance(obj, dict):
        return {k: shape(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [shape(obj[0], depth + 1), f"×{len(obj)}"] if obj else []
    return type(obj).__name__


def probe() -> tuple[Path, dict]:
    """Call every read tool once; save raw replies (private) and return their shape."""
    out: dict = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "calls": {}}
    with Session() as s:
        out["server"] = s.server_info.get("serverInfo")
        tools = s.list_tools()
        out["tools"] = tools
        schemas = {t["name"]: t.get("inputSchema", {}) for t in tools}
        out["read_tools_missing"] = sorted(READ_TOOLS - set(schemas))

        accounts: list[str] = []
        if "get_user_accounts" in schemas:
            args, missing = _guess_args(schemas["get_user_accounts"], None)
            try:
                res = s.call_tool("get_user_accounts", args) if not missing else None
                out["calls"]["get_user_accounts"] = {"args": args, "missing": missing, "result": res}
                accounts = _account_ids(res)
            except MontroseError as exc:
                out["calls"]["get_user_accounts"] = {"args": args, "error": str(exc)}

        optional = {
            "get_transactions": {
                "from": (datetime.now(timezone.utc).date() - timedelta(days=90)).isoformat(),
                "pageSize": 50,
            },
        }
        for name in ("get_transactions_categories", "get_holdings", "get_transactions"):
            if name not in schemas:
                continue
            needs_account = any("account" in r.lower() for r in schemas[name].get("required", []))
            for i, acc in enumerate((accounts if needs_account else [None]) or [None]):
                # Index, not the account number: the summary is meant to be pasted.
                key = f"{name}[account {i + 1}]" if acc else name
                args, missing = _guess_args(schemas[name], acc)
                args = {**optional.get(name, {}), **args}
                if missing:
                    out["calls"][key] = {"args": args, "missing": missing}
                    continue
                try:
                    out["calls"][key] = {"args": args, "result": s.call_tool(name, args)}
                except MontroseError as exc:
                    out["calls"][key] = {"args": args, "error": str(exc)}

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = state_dir() / f"probe-{stamp}.json"
    _write_private(path, out)

    summary = {
        "server": out.get("server"),
        "tools": sorted(schemas),
        "read_tools_missing": out["read_tools_missing"],
        "calls": {k: ({"error": v["error"]} if "error" in v
                      else {"missing_args": v["missing"]} if v.get("missing")
                      else {"args": sorted(v["args"]), "shape": shape(v.get("result"))})
                  for k, v in out["calls"].items()},
        "token": token_status(),
    }
    return path, summary
