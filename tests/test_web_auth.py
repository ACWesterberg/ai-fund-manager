from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from fundmgr.web.auth import protect_dashboard


def client(headers=None):
    app = FastAPI()
    app.middleware("http")(protect_dashboard)

    @app.get("/")
    @app.post("/fill")
    @app.post("/deploy")
    def endpoint():
        return {"ok": True}

    return TestClient(app, headers=headers)


def test_unconfigured_dashboard_fails_closed(monkeypatch):
    monkeypatch.delenv("FUND_WEB_PASSWORD")
    assert client().get("/").status_code == 503
    assert client().post("/fill").status_code == 503


def test_missing_or_wrong_credentials_are_rejected_without_a_browser_prompt():
    """The WWW-Authenticate challenge is what made the browser pop its dialog."""
    for header in ("", "Basic invalid", "Bearer secret", "Basic bm9wZTpub3Bl"):
        response = client({"Authorization": header}).post("/fill")
        assert response.status_code == 401
        assert "www-authenticate" not in response.headers


def test_authenticated_dashboard_access():
    assert client().get("/").status_code == 200
    assert client().post("/fill").status_code == 200


def test_cross_origin_form_is_rejected_even_when_authenticated():
    assert client().post("/fill", headers={"Origin": "https://attacker.example"}).status_code == 403
    assert client().post("/fill", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert client().post("/fill", headers={"Origin": "http://testserver"}).status_code == 200


def test_webhook_uses_its_own_authentication(monkeypatch):
    monkeypatch.delenv("FUND_WEB_PASSWORD")
    assert client({"Authorization": ""}).post("/deploy").status_code == 200


def test_real_app_protects_routes_and_verifies_webhook(monkeypatch):
    from fundmgr.web.app import app

    actual = TestClient(app, headers={"Authorization": ""})
    assert actual.get("/", headers={"Accept": "application/json"}).status_code == 401
    assert actual.post("/live/example/fill").status_code == 401
    monkeypatch.setenv("DEPLOY_WEBHOOK_SECRET", "webhook-test-secret")
    assert actual.post("/deploy", json={}).status_code == 401


# ── The login page and its session ────────────────────────────────────────────

def _anon():
    return client({"Authorization": ""})


def _login(c, password="test-password", next_path="/"):
    return c.post("/login", data={"password": password, "next": next_path},
                  follow_redirects=False)


def test_a_page_visit_goes_to_the_login_page_and_comes_back(monkeypatch):
    c = _anon()
    first = c.get("/sim?tab=x", headers={"Accept": "text/html"}, follow_redirects=False)
    assert first.status_code == 303 and first.headers["location"] == "/login?next=/sim%3Ftab%3Dx"
    page = c.get("/login?next=/sim", headers={"Accept": "text/html"})
    assert page.status_code == 200 and 'name="password"' in page.text
    done = _login(c, next_path="/")
    assert done.status_code == 303 and done.headers["location"] == "/"
    assert "fund_session" in done.headers["set-cookie"]
    assert "httponly" in done.headers["set-cookie"].lower()
    assert "samesite=lax" in done.headers["set-cookie"].lower()
    assert c.get("/").status_code == 200          # the cookie carries the session
    assert c.post("/fill").status_code == 200


def test_a_wrong_code_stays_on_the_login_page():
    c = _anon()
    response = _login(c, password="nope")
    assert response.status_code == 401 and "Fel kod" in response.text
    assert "set-cookie" not in response.headers
    assert c.get("/", headers={"Accept": "application/json"}).status_code == 401


def test_repeated_wrong_codes_are_throttled(monkeypatch):
    from fundmgr.web import auth
    monkeypatch.setattr(auth, "_failures", {})
    c = _anon()
    for _ in range(auth.MAX_FAILURES):
        assert _login(c, password="nope").status_code == 401
    assert _login(c).status_code == 429             # even the right code waits


def test_a_forged_or_expired_session_is_refused(monkeypatch):
    from fundmgr.web import auth
    c = _anon()
    c.cookies.set(auth.COOKIE, "9999999999.deadbeef")
    assert c.get("/", headers={"Accept": "application/json"}).status_code == 401
    c.cookies.set(auth.COOKIE, auth._sign(1, "test-password"))          # expired in 1970
    assert c.get("/", headers={"Accept": "application/json"}).status_code == 401
    monkeypatch.setenv("FUND_WEB_PASSWORD", "changed")                   # old sessions die
    c.cookies.set(auth.COOKIE, auth._sign(9999999999, "test-password"))
    assert c.get("/", headers={"Accept": "application/json"}).status_code == 401


def test_the_login_never_redirects_off_site():
    for evil in ("https://attacker.example/", "//attacker.example/x", "javascript:alert(1)"):
        assert _login(_anon(), next_path=evil).headers["location"] == "/"


def test_a_cross_site_post_is_refused_even_with_a_session():
    c = _anon()
    _login(c)
    assert c.post("/fill", headers={"Origin": "https://attacker.example"}).status_code == 403
    assert c.post("/login", data={"password": "test-password"},
                  headers={"Origin": "https://attacker.example"}).status_code == 403


def test_logout_ends_the_session():
    c = _anon()
    _login(c)
    asked = c.get("/logout", follow_redirects=False)
    assert asked.status_code == 200 and 'action="/logout"' in asked.text
    assert c.get("/", headers={"Accept": "application/json"}).status_code == 200  # still signed in
    out = c.post("/logout", follow_redirects=False)
    assert out.status_code == 303 and out.headers["location"] == "/login"
    assert c.get("/", headers={"Accept": "application/json"}).status_code == 401


# ── Every credential check is throttled ──────────────────────────────────────

import base64  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_failures(monkeypatch):
    from fundmgr.web import auth
    monkeypatch.setattr(auth, "_failures", {})


def _basic(code: str) -> dict:
    return {"Authorization": "Basic " + base64.b64encode(f"fund:{code}".encode()).decode()}


def test_basic_guesses_are_throttled_like_logins():
    from fundmgr.web import auth
    c = _anon()
    for i in range(auth.MAX_FAILURES):
        assert c.get("/", headers=_basic(f"guess{i}")).status_code == 401
    assert c.get("/", headers=_basic("test-password")).status_code == 429


def test_forged_cookies_are_throttled_and_cleared():
    from fundmgr.web import auth
    c = _anon()
    for i in range(auth.MAX_FAILURES):
        c.cookies.set(auth.COOKIE, f"9999999999.{i:064x}")
        response = c.get("/", headers={"Accept": "application/json"})
        assert response.status_code == 401
        assert 'fund_session=""' in response.headers.get("set-cookie", "")
    c.cookies.set(auth.COOKIE, auth._sign(9999999999, "test-password"))
    assert c.get("/").status_code == 429


def test_an_expired_session_is_not_a_guess():
    """The wall display's lapsed cookie must not lock the owner out."""
    from fundmgr.web import auth
    c = _anon()
    c.cookies.set(auth.COOKIE, auth._sign(1, "test-password"))
    for _ in range(auth.MAX_FAILURES + 2):
        assert c.get("/", headers={"Accept": "application/json"}).status_code == 401
    assert _login(c).status_code == 303


def test_a_spoofed_cloudflare_header_does_not_reset_the_count():
    """Off the tunnel the header is the client's own claim."""
    from fundmgr.web import auth
    c = _anon()
    for i in range(auth.MAX_FAILURES):
        assert _login(c, password="nope").status_code == 401
        c.headers["cf-connecting-ip"] = f"10.0.0.{i}"
    assert _login(c).status_code == 429


def test_the_tunnel_header_is_trusted_from_loopback():
    from types import SimpleNamespace

    from fundmgr.web import auth
    req = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"),
                          headers={"cf-connecting-ip": "203.0.113.9"})
    assert auth._client(req) == "203.0.113.9"
    req.client.host = "192.168.1.20"
    assert auth._client(req) == "192.168.1.20"


def test_non_ascii_credentials_are_refused_not_a_server_error():
    from fundmgr.web import auth
    raw = f"{auth.COOKIE}=9999999999.".encode() + b"\xe9"   # arrives as latin-1 'é'
    assert _anon().get("/", headers={"Accept": "application/json", "Cookie": raw}).status_code == 401


def test_the_webhook_refuses_oversized_and_non_ascii_requests(monkeypatch):
    from fundmgr.web.app import MAX_WEBHOOK_BYTES, app
    monkeypatch.setenv("DEPLOY_WEBHOOK_SECRET", "webhook-test-secret")
    actual = TestClient(app, headers={"Authorization": ""})
    assert actual.post("/deploy", content=b"x" * (MAX_WEBHOOK_BYTES + 1)).status_code == 413
    assert actual.post("/deploy", content=b"{}",
                       headers={"X-Hub-Signature-256": b"sha256=\xe9"}).status_code == 401



def test_another_site_cannot_sign_the_owner_out():
    c = _anon()
    _login(c)
    assert c.post("/logout", headers={"Origin": "https://attacker.example"}).status_code == 403
    assert c.get("/", headers={"Accept": "application/json"}).status_code == 200



# ── Sessions are signed with a host key, not the password alone ──────────────

def test_a_cookie_cannot_be_forged_from_the_password_alone(monkeypatch):
    import hashlib
    import hmac as _hmac

    from fundmgr.web import auth
    old_style = hashlib.sha256(b"fund-session:test-password").digest()
    expiry = 9999999999
    forged = f"{expiry}." + _hmac.new(old_style, str(expiry).encode(), hashlib.sha256).hexdigest()
    c = _anon()
    c.cookies.set(auth.COOKIE, forged)
    assert c.get("/", headers={"Accept": "application/json"}).status_code == 401
    c.cookies.set(auth.COOKIE, auth._sign(expiry, "test-password"))
    assert c.get("/", headers={"Accept": "application/json"}).status_code == 200


def test_the_host_key_is_created_once_private_and_reused(tmp_path, monkeypatch):
    import stat

    from fundmgr.web import auth
    monkeypatch.delenv("FUND_WEB_SESSION_SECRET")
    monkeypatch.setattr(auth, "SESSION_KEY_FILE", tmp_path / "web" / "session.key")
    monkeypatch.setattr(auth, "_session_key", None)
    first = auth._sign(9999999999, "pw")
    mode = stat.S_IMODE((tmp_path / "web" / "session.key").stat().st_mode)
    assert mode == 0o600
    monkeypatch.setattr(auth, "_session_key", None)           # another process, same file
    assert auth._sign(9999999999, "pw") == first
    assert auth._sign(9999999999, "changed") != first         # a new password signs out
