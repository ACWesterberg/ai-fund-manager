"""What the sidebar lists, shared by every template environment.

The sidebar is rendered by four routers with four Jinja environments, and the
accounts section has to list whatever accounts exist on every page — so it reads
them through a global rather than each route threading them into its context.
"""
from __future__ import annotations

import os

from jinja2 import Environment


def nav_accounts() -> list[dict]:
    """Broker accounts for the sidebar. Fails open: a broken book must not take
    every page down with it."""
    try:
        from fundmgr import accounts
        return [{"slug": m["slug"], "name": m["name"]} for m in accounts.list_accounts()]
    except Exception:
        return []


def show_archived() -> bool:
    """The model-comparison sims, paper portfolios and live sleeves are off the
    sidebar; their pages still work. FUND_WEB_SHOW_ARCHIVED=1 lists them again."""
    return os.getenv("FUND_WEB_SHOW_ARCHIVED", "").strip().lower() in ("1", "true", "yes")


def install(env: Environment) -> Environment:
    env.globals["nav_accounts"] = nav_accounts
    env.globals["show_archived"] = show_archived
    return env
