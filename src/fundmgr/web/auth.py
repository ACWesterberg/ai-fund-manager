"""Password protection for the dashboard, including its paid and mutating routes.

A login page and a signed session cookie, not HTTP Basic. The browser's Basic
prompt kept coming back: it caches credentials per origin, so moving between
the Nordic dashboard (:8010) and the sim dashboards (:8011) asked again, and a
browser restart forgot it. Cookies are scoped by host, not port, so one login
covers both. Basic credentials are still accepted (scripts, tests), but nothing
sends the `WWW-Authenticate` challenge that made the browser pop the dialog.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import html
import os
import time
from collections.abc import Awaitable, Callable
from urllib.parse import quote, urlsplit

from dotenv import load_dotenv
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from fundmgr.config import ROOT

load_dotenv(ROOT / ".env")

COOKIE = "fund_session"
SESSION_DAYS = 90            # the wall display should not have to log in every week
MAX_FAILURES = 8             # per client, per window, before logins are refused
FAILURE_WINDOW = 15 * 60
_failures: dict[str, list[float]] = {}


def _secret(password: str) -> bytes:
    # Derived from the password, so changing it signs every old session out.
    return hashlib.sha256(f"fund-session:{password}".encode()).digest()


def _sign(expiry: int, password: str) -> str:
    mac = hmac.new(_secret(password), str(expiry).encode(), hashlib.sha256).hexdigest()
    return f"{expiry}.{mac}"


def _valid_session(value: str, password: str) -> bool:
    expiry, _, mac = (value or "").partition(".")
    if not expiry.isdigit() or int(expiry) < time.time():
        return False
    return hmac.compare_digest(_sign(int(expiry), password), value)


def _valid_basic(header: str, password: str) -> bool:
    if not header.lower().startswith("basic "):
        return False
    username = os.getenv("FUND_WEB_USERNAME", "fund")
    try:
        decoded = base64.b64decode(header[6:], validate=True)
    except (ValueError, binascii.Error):
        return False
    return hmac.compare_digest(decoded, f"{username}:{password}".encode())


def _client(request: Request) -> str:
    # Behind the Cloudflare tunnel every request arrives from 127.0.0.1.
    return request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "?")


def _throttled(key: str) -> bool:
    now = time.time()
    recent = [t for t in _failures.get(key, []) if now - t < FAILURE_WINDOW]
    _failures[key] = recent
    return len(recent) >= MAX_FAILURES


def _safe_next(target: str | None) -> str:
    """Only a path on this site — never an open redirect to somewhere else."""
    target = target or "/"
    parts = urlsplit(target)
    if parts.scheme or parts.netloc or not target.startswith("/") or target.startswith("//"):
        return "/"
    return target


def _cross_origin(request: Request) -> bool:
    origin = request.headers.get("origin")
    return (request.headers.get("sec-fetch-site") == "cross-site"
            or (origin is not None
                and (urlsplit(origin).scheme, urlsplit(origin).netloc)
                != (request.url.scheme, request.url.netloc)))


def _is_https(request: Request) -> bool:
    return (request.url.scheme == "https"
            or request.headers.get("x-forwarded-proto", "").lower() == "https")


def _login_page(next_path: str, message: str = "", status: int = 200) -> HTMLResponse:
    note = f'<p class="err">{html.escape(message)}</p>' if message else ""
    body = f"""<!doctype html>
<html lang="sv"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fund Manager — logga in</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; min-height:100vh; display:grid; place-items:center;
         background:#0b1220; color:#e2e8f0; font:16px system-ui, sans-serif; }}
  form {{ width:min(320px, calc(100vw - 32px)); padding:28px; border-radius:16px;
         background:#111a2e; border:1px solid #1e293b; }}
  h1 {{ margin:0 0 18px; font-size:18px; font-weight:600; }}
  input {{ width:100%; box-sizing:border-box; padding:12px 14px; border-radius:10px;
          border:1px solid #334155; background:#0b1220; color:inherit; font-size:18px;
          letter-spacing:.08em; }}
  button {{ width:100%; margin-top:14px; padding:12px; border:0; border-radius:10px;
           background:#2563eb; color:#fff; font-size:16px; font-weight:600; }}
  .err {{ color:#f87171; margin:12px 0 0; font-size:14px; }}
</style></head>
<body><form method="post" action="/login">
  <h1>Fund Manager</h1>
  <input type="password" name="password" placeholder="Kod" autocomplete="current-password" autofocus required>
  <input type="hidden" name="next" value="{html.escape(next_path, quote=True)}">
  <button type="submit">Logga in</button>{note}
</form></body></html>"""
    return HTMLResponse(body, status_code=status)


async def _login(request: Request, password: str) -> Response:
    if request.method != "POST":
        return _login_page(_safe_next(request.query_params.get("next")))
    if _cross_origin(request):
        return JSONResponse({"detail": "Cross-origin writes are not allowed"}, status_code=403)
    key = _client(request)
    form = await request.form()
    next_path = _safe_next(str(form.get("next") or "/"))
    if _throttled(key):
        return _login_page(next_path, "För många försök. Vänta en stund och försök igen.", 429)
    if not hmac.compare_digest(str(form.get("password") or "").encode(), password.encode()):
        _failures.setdefault(key, []).append(time.time())
        return _login_page(next_path, "Fel kod.", 401)
    _failures.pop(key, None)
    expiry = int(time.time()) + SESSION_DAYS * 86400
    response = RedirectResponse(next_path, status_code=303)
    response.set_cookie(COOKIE, _sign(expiry, password), max_age=SESSION_DAYS * 86400,
                        httponly=True, samesite="lax", secure=_is_https(request), path="/")
    return response


async def protect_dashboard(
    request: Request, call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    path = request.url.path
    # The GitHub endpoint verifies its own HMAC over the request body.
    if path == "/deploy" and request.method == "POST":
        return await call_next(request)
    password = os.getenv("FUND_WEB_PASSWORD", "")
    if not password:
        return JSONResponse({"detail": "Set FUND_WEB_PASSWORD to enable the dashboard"}, status_code=503)
    if path == "/login":
        return await _login(request, password)
    if path == "/logout":
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(COOKIE, path="/")
        return response

    valid = (_valid_session(request.cookies.get(COOKIE, ""), password)
             or _valid_basic(request.headers.get("authorization", ""), password))
    if not valid:
        wants_page = (request.method in {"GET", "HEAD"}
                      and "text/html" in request.headers.get("accept", ""))
        if wants_page:
            target = path + (f"?{request.url.query}" if request.url.query else "")
            return RedirectResponse(f"/login?next={quote(target, safe='/')}", status_code=303)
        # No WWW-Authenticate: that header is what made the browser pop its dialog.
        return JSONResponse({"detail": "Authentication required"}, status_code=401)
    if request.method not in {"GET", "HEAD", "OPTIONS"} and _cross_origin(request):
        # The session cookie rides along on a cross-site form post just as Basic
        # credentials did. SameSite=Lax already withholds it; this is belt and braces.
        return JSONResponse({"detail": "Cross-origin writes are not allowed"}, status_code=403)
    return await call_next(request)
