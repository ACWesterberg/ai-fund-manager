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
    out = c.get("/logout", follow_redirects=False)
    assert out.status_code == 303 and out.headers["location"] == "/login"
    assert c.get("/", headers={"Accept": "application/json"}).status_code == 401
