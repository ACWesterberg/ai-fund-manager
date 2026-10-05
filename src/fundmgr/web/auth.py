"""Password protection for the dashboard, including its paid and mutating routes."""
from __future__ import annotations

import base64
import binascii
import hmac
import os
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

from dotenv import load_dotenv
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from fundmgr.config import ROOT

load_dotenv(ROOT / ".env")


async def protect_dashboard(
    request: Request, call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    # The GitHub endpoint verifies its own HMAC over the request body.
    if request.url.path == "/deploy" and request.method == "POST":
        return await call_next(request)
    password = os.getenv("FUND_WEB_PASSWORD", "")
    if not password:
        return JSONResponse({"detail": "Set FUND_WEB_PASSWORD to enable the dashboard"}, status_code=503)
    username = os.getenv("FUND_WEB_USERNAME", "fund")
    supplied = request.headers.get("authorization", "")
    valid = False
    if supplied.lower().startswith("basic "):
        try:
            decoded = base64.b64decode(supplied[6:], validate=True)
            valid = hmac.compare_digest(decoded, f"{username}:{password}".encode())
        except (ValueError, binascii.Error):
            pass
    if not valid:
        return JSONResponse({"detail": "Authentication required"}, status_code=401,
                            headers={"WWW-Authenticate": 'Basic realm="Fund Manager", charset="UTF-8"'})
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        # Browsers resend Basic credentials automatically. Reject cross-site
        # form submissions even when the victim is already authenticated.
        origin = request.headers.get("origin")
        if (request.headers.get("sec-fetch-site") == "cross-site"
                or (origin is not None and
                    (urlsplit(origin).scheme, urlsplit(origin).netloc) !=
                    (request.url.scheme, request.url.netloc))):
            return JSONResponse({"detail": "Cross-origin writes are not allowed"}, status_code=403)
    return await call_next(request)
