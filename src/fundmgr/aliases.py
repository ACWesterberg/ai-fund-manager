"""
Learned ticker aliases — corrections you make once, remembered everywhere.

Photo import resolves a holding to a Yahoo symbol through a ladder of guesses:
ISIN lookup, broker→Yahoo map, name alias, symbol search. Any of them can miss,
and a name Yahoo doesn't rank well ends up needing a human. That correction is
knowledge, and throwing it away means retyping it every time the name shows up
again — in another screenshot, another sleeve, another account.

Everything imported through the review table is recorded here against whatever
identifiers the row carried, and consulted first on the next import. The file
lives in DATA_DIR (outside the repo, alongside the databases) because it is
your data, not the project's, and it is plain JSON so it can be inspected and
hand-edited.

Three key kinds, strongest first:

  isin   — unambiguous, survives renames and re-listings
  broker — the ticker as your broker writes it ("4HY", "KOG")
  name   — the company name, normalised; the weakest and the most likely to
           collide, so it is consulted last

An alias always wins over automatic resolution. It was entered by a human who
was looking at the actual position.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fundmgr.config import DATA_DIR

# Key kinds, in the order they are consulted.
KINDS = ("isin", "broker", "name")


def alias_path() -> Path:
    """Location of the alias file. Overridable for tests via FUND_ALIAS_PATH."""
    override = os.getenv("FUND_ALIAS_PATH")
    return Path(override) if override else DATA_DIR / "ticker_aliases.json"


def _normalise(kind: str, key: str) -> str | None:
    """Canonical form of a lookup key, or None when it isn't usable."""
    key = (key or "").strip()
    if not key:
        return None
    if kind == "isin":
        key = key.upper().replace(" ", "")
        return key if len(key) == 12 else None
    if kind == "broker":
        return key.upper()
    if kind == "name":
        from fundmgr.paper import _clean_name
        cleaned = _clean_name(key)
        # A one-word fragment is too collision-prone to key on.
        return cleaned if len(cleaned) >= 3 else None
    return None


def load() -> dict[str, dict]:
    """The whole alias table, {kind: {key: entry}}. Empty when nothing is stored."""
    path = alias_path()
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {kind: {} for kind in KINDS}
    if not isinstance(raw, dict):
        return {kind: {} for kind in KINDS}
    return {
        kind: {k: v for k, v in (raw.get(kind) or {}).items() if isinstance(v, dict)}
        for kind in KINDS
    }


def _save(table: dict[str, dict]) -> None:
    """Write atomically — a half-written alias file would break every import."""
    path = alias_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(table, fh, indent=2, ensure_ascii=False, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def lookup(isin: str | None = None, raw_ticker: str | None = None,
           name: str | None = None, table: dict | None = None) -> tuple[str, str] | None:
    """Find a learned ticker for a holding. Returns (ticker, matched_kind).

    Tries the strongest identifier first, so a row carrying an ISIN is never
    resolved by a name that happens to look similar.
    """
    table = load() if table is None else table
    for kind, value in (("isin", isin), ("broker", raw_ticker), ("name", name)):
        key = _normalise(kind, value or "")
        if not key:
            continue
        entry = (table.get(kind) or {}).get(key)
        if entry and entry.get("ticker"):
            return str(entry["ticker"]).upper(), kind
    return None


def learn(ticker: str, *, isin: str | None = None, raw_ticker: str | None = None,
          name: str | None = None, source: str = "photo import") -> list[str]:
    """Record a confirmed ticker against every identifier the row carried.

    Returns the kinds actually written. Re-learning the same mapping refreshes
    its timestamp; learning a *different* ticker for a known key overwrites it,
    because the newer correction is the one the human just looked at.
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return []

    table = load()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    written: list[str] = []
    for kind, value in (("isin", isin), ("broker", raw_ticker), ("name", name)):
        key = _normalise(kind, value or "")
        if not key:
            continue
        # Don't record a broker alias that is already the answer — "NVDA"→"NVDA"
        # is noise, and it would shadow a later, better correction.
        if kind == "broker" and key == ticker:
            continue
        existing = (table.get(kind) or {}).get(key) or {}
        table.setdefault(kind, {})[key] = {
            "ticker": ticker,
            "name": (name or existing.get("name") or "").strip(),
            "source": source,
            "learned_at": now,
            "first_learned_at": existing.get("first_learned_at", now),
        }
        written.append(kind)

    if written:
        _save(table)
    return written


def forget(key: str, kind: str | None = None) -> list[str]:
    """Remove an alias by key. Without `kind`, removes it from every kind."""
    table = load()
    removed: list[str] = []
    for k in ([kind] if kind else list(KINDS)):
        normalised = _normalise(k, key)
        if normalised and normalised in (table.get(k) or {}):
            table[k].pop(normalised)
            removed.append(k)
    if removed:
        _save(table)
    return removed


def entries() -> list[dict]:
    """Flat listing for display: [{kind, key, ticker, name, learned_at}]."""
    table = load()
    out = []
    for kind in KINDS:
        for key, entry in sorted((table.get(kind) or {}).items()):
            out.append({
                "kind": kind, "key": key,
                "ticker": entry.get("ticker", ""),
                "name": entry.get("name", ""),
                "source": entry.get("source", ""),
                "learned_at": entry.get("learned_at", ""),
            })
    return out
