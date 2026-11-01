"""Fundamentals fetched in this repo: the same keys as before, ownership with
them, statement ratios kept across a refresh, and an ownership line that only
reaches the prompt when switched on."""
from __future__ import annotations

import time
from datetime import datetime

import pytest

from fundmgr.config import AppConfig
from fundmgr.data import fundamentals as fm
from fundmgr.data.prices import TickerFeatures
from fundmgr.engine.prompt import OWNERSHIP_NOTE, build_prompt
from fundmgr.state.models import PortfolioSnapshot, Position
from fundmgr.state.store import Store

INFO = {"quoteType": "EQUITY", "trailingPE": 14.2, "currency": "SEK", "sector": "Industrials",
        "numberOfAnalystOpinions": 9.0, "profitMargins": float("nan"),
        "heldPercentInsiders": 0.0189, "heldPercentInstitutions": 0.64264}


def test_info_maps_onto_the_cache_keys_with_ownership():
    data = fm.parse_info(INFO)
    assert set(data) == set(fm.FIELD_MAP.values())
    assert data["pe_ratio"] == 14.2 and data["analyst_count"] == 9
    assert data["currency"] == "SEK" and data["profit_margin"] is None
    assert data["held_insiders"] == 0.0189 and data["held_institutions"] == 0.64264
    assert fm.parse_info({"quoteType": "NONE"}) is None and fm.parse_info({}) is None
    assert fm.parse_info({"trailingPegRatio": None}) is None       # Yahoo's reply to an unknown symbol


def test_days_to_an_event():
    assert fm.ts_to_days(time.time() + 3 * 86400 + 60) == 3
    assert fm.ts_to_days(time.time() - 86400 * 2) is None
    assert fm.ts_to_days("junk") is None and fm.ts_to_days(None) is None


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "f.db")


def test_a_refresh_fetches_only_stale_names_and_keeps_statement_ratios(store, monkeypatch):
    store.save_fundamentals("OLD.ST", {"pe_ratio": 9.0, "equity_to_assets": 0.37})
    store.save_fundamentals("FRESH.ST", {"pe_ratio": 20.0})
    with store._conn() as conn:
        conn.execute("UPDATE fundamentals_cache SET fetched_at = '2020-01-01' WHERE ticker = 'OLD.ST'")
    asked = []
    monkeypatch.setattr(fm, "fetch_info", lambda t: asked.append(t) or fm.parse_info(INFO))
    assert fm.fetch_and_cache_fundamentals(["OLD.ST", "FRESH.ST", "NEW.ST"], store) == 2
    assert sorted(asked) == ["NEW.ST", "OLD.ST"]
    old = store.get_fundamentals("OLD.ST")
    assert old["pe_ratio"] == 14.2 and old["equity_to_assets"] == 0.37
    assert store.get_fundamentals("FRESH.ST") == {"pe_ratio": 20.0}


def test_a_failed_fetch_leaves_the_old_entry_to_retry(store, monkeypatch):
    store.save_fundamentals("OLD.ST", {"pe_ratio": 9.0})
    with store._conn() as conn:
        conn.execute("UPDATE fundamentals_cache SET fetched_at = '2020-01-01'")
    monkeypatch.setattr(fm, "fetch_info", lambda t: None)
    assert fm.fetch_and_cache_fundamentals(["OLD.ST"], store) == 0
    assert store.get_fundamentals("OLD.ST") == {"pe_ratio": 9.0}
    assert store.get_stale_fundamentals_tickers(["OLD.ST"]) == ["OLD.ST"]


def test_the_ownership_line():
    assert fm.ownership_line({"held_insiders": 0.4712, "held_institutions": 0.3196}) == \
        "Owners: insiders 47%, institutions 32% of capital"
    assert fm.ownership_line({"held_institutions": 1.03}) == "Owners: institutions 100% of capital"
    assert fm.ownership_line({"held_insiders": 7.5}) is None          # junk, not a share
    assert fm.ownership_line({"pe_ratio": 9}) is None and fm.ownership_line(None) is None


def _prompt(cfg, store):
    features = {t: TickerFeatures(t, t, 100, "2026-10-30", 0) for t in ("HELD.ST", "CAND.ST")}
    snap = PortfolioSnapshot([Position("HELD.ST", 10, 90, 100)], 100000, timestamp=datetime(2026, 11, 2))
    _system, _user, fields = build_prompt(cfg, snap, features, store, "weekly-1")
    return fields["universe"]


def test_ownership_reaches_the_prompt_only_when_switched_on(store):
    store.save_fundamentals("HELD.ST", {"held_insiders": 0.02, "held_institutions": 0.64})
    store.save_fundamentals("CAND.ST", {"held_insiders": 0.47})
    cfg = AppConfig(db_path=store.db_path)
    off = _prompt(cfg, store)
    assert "Owners" not in off and OWNERSHIP_NOTE not in off

    cfg.data.ownership_in_prompt = True
    on = _prompt(cfg, store)
    assert on.count(OWNERSHIP_NOTE) == 1
    assert "[HELD.ST] HELD.ST" in on and "  Owners: insiders 2%, institutions 64% of capital" in on
    assert "  Owners: insiders 47% of capital" in on
    assert on.replace(OWNERSHIP_NOTE + "\n\n", "").replace(
        "\n  Owners: insiders 2%, institutions 64% of capital", "").replace(
        "\n  Owners: insiders 47% of capital", "") == off
