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


def test_missing_or_wrong_credentials_are_rejected():
    for header in ("", "Basic invalid", "Bearer secret", "Basic bm9wZTpub3Bl"):
        response = client({"Authorization": header}).post("/fill")
        assert response.status_code == 401
        assert response.headers["www-authenticate"].startswith("Basic")


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
    assert actual.get("/").status_code == 401
    assert actual.post("/live/example/fill").status_code == 401
    monkeypatch.setenv("DEPLOY_WEBHOOK_SECRET", "webhook-test-secret")
    assert actual.post("/deploy", json={}).status_code == 401
