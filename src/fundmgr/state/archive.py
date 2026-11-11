"""Point-in-time archive of what each weekly run could see, written once per run.

Prices can be re-downloaded later; almost nothing else here can. The fundamentals
cache is overwritten in place every seven days, FinanceData refuses historical
fundamentals, and the screener's ranking of the names it did *not* pass on was
never stored. Without this, three questions stay unanswerable forever: does the
screener's rank predict returns, did the model's picks beat the candidates it
was shown, and how did valuations and analyst targets move before a decision.

One immutable gzip file per run under ``<db parent>/archive/<fund>/``, beside
the trading database rather than inside it, so the store the fund trades from
stays small and a backup can copy new files incrementally instead of re-sending
a growing database every day. Writing is advisory: any failure is logged and the
run carries on, because a missing archive must never cost a decision.
"""
from __future__ import annotations

import gzip
import json
import logging
import math
import os
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fundmgr.config import AppConfig
    from fundmgr.data.prices import TickerFeatures

logger = logging.getLogger(__name__)

VERSION = 1
_RUN_ID = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9a-f]{6}$")
# Headlines reach only candidates and are archived in the frozen prompt already.
_DROP_FIELDS = {"headlines"}


def archive_dir(cfg: "AppConfig") -> Path:
    return cfg.db_path.parent / "archive" / cfg.db_path.stem


def _clean(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _compact(feat: "TickerFeatures") -> dict:
    return {
        k: _clean(v) for k, v in asdict(feat).items()
        if k not in _DROP_FIELDS and v is not None
    }


def build_run_archive(
    cfg: "AppConfig",
    run_id: str,
    features: dict[str, "TickerFeatures"],
    selected: set[str],
    scores: dict[str, float],
    fundamentals: dict[str, dict],
    held: set[str],
    pinned: set[str],
    universe_size: int,
) -> dict:
    ranked = sorted(scores, key=lambda t: (-scores[t], t))
    rank = {t: i + 1 for i, t in enumerate(ranked)}
    population = [
        {
            "ticker": t,
            "score": round(scores[t], 4) if t in scores else None,
            "rank": rank.get(t),
            "selected": t in selected,
            "held": t in held,
            "pinned": t in pinned,
            "features": _compact(features[t]),
        }
        for t in sorted(features)
    ]
    return {
        "version": VERSION,
        "run_id": run_id,
        "fund_id": cfg.db_path.stem,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "universe_size": universe_size,
        "screen": {"top_n": cfg.screener.top_n, "scored": len(scores),
                   "selected": len(selected)},
        "population": population,
        "fundamentals": {t: _clean(fundamentals[t]) for t in sorted(fundamentals)},
    }


def write_run_archive(cfg: "AppConfig", run_id: str, payload: dict) -> Path | None:
    """Write the archive for ``run_id`` once. Returns its path, or None on failure.

    An existing file is never overwritten: a run id names one decision, and a
    second write could only replace what was seen then with what is seen now.
    """
    if not _RUN_ID.match(run_id):
        logger.warning("Run archive skipped: unexpected run id %r", run_id)
        return None
    try:
        directory = archive_dir(cfg)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{run_id}.json.gz"
        if path.exists():
            logger.warning("Run archive %s already exists; left untouched", path)
            return None
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"), allow_nan=False)
        os.replace(tmp, path)
        return path
    except Exception as exc:
        logger.warning("Run archive for %s failed: %s", run_id, exc)
        return None


def read_run_archive(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)
