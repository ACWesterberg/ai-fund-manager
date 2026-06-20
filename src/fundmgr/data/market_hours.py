"""
Exchange trading-hours / holiday checks for paper fills.

Maps a universe `exchange` code to its exchange_calendars MIC calendar and
answers "is this exchange open right now?" — accounting for weekends, session
hours, and bank holidays. Used to avoid booking paper fills at stale prices
when the relevant market is closed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache

# Universe exchange code (config.csv `exchange` column) -> exchange_calendars MIC.
# Ambiguous venues map to their primary calendar (close enough for holidays).
_EXCHANGE_TO_CALENDAR: dict[str, str] = {
    # Nordic
    "OMXS": "XSTO", "OMXS-FN": "XSTO", "SPOTLIGHT": "XSTO", "NGM": "XSTO",
    "OMXC": "XCSE", "OMXC-FN": "XCSE",
    "OSLO": "XOSL",
    "OMXH": "XHEL",
    "OMXI": "XICE",
    # Global
    "NYSE": "XNYS", "NASDAQ": "XNAS", "OTC": "XNYS",
    "LSE": "XLON",
    "XETRA": "XETR",
    "EURONEXT": "XPAR",
    "SIX": "XSWX",
    "TSE": "XTKS",
    "TSX": "XTSE",
    "ASX": "XASX",
    "HKEX": "XHKG",
}


@lru_cache(maxsize=64)
def _calendar(mic: str):
    import exchange_calendars as ec
    return ec.get_calendar(mic)


def is_exchange_open(exchange_code: str, when: datetime | None = None) -> bool | None:
    """Is `exchange_code` open at `when` (UTC now by default)?

    Returns True/False when determinable, or None when it can't be decided
    (unknown exchange code, library/calendar error) so callers can fail-open
    rather than wrongly blocking a fill.
    """
    mic = _EXCHANGE_TO_CALENDAR.get((exchange_code or "").upper())
    if not mic:
        return None
    try:
        import pandas as pd
        ts = pd.Timestamp(when or datetime.now(timezone.utc))
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return bool(_calendar(mic).is_open_on_minute(ts))
    except Exception:
        return None
