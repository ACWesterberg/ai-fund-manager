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


# XNYS and XNAS keep the same holiday schedule (US federal calendar + the NYSE
# rules), so for "is this market trading today" they are one venue. Nothing
# else here is grouped: the Nordic exchanges genuinely differ from each other
# on national holidays, so folding them together would answer wrongly.
_CALENDAR_GROUP: dict[str, str] = {"XNAS": "XNYS"}

# How far ahead to look for the next session. A market closed for longer than
# this is not a holiday, and reporting "unknown" beats reporting a wrong date.
_NEXT_SESSION_HORIZON_DAYS = 14


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


def dominant_calendar(universe_path) -> str | None:
    """The trading calendar most of a universe's enabled names sit on.

    Answers "whose holidays decide whether a run for this profile can execute
    at all". Counting raw MICs answers that wrongly for the global universe:
    LSE has more names than either US venue alone, but fewer than the two
    together — and LSE trades on US holidays. Hence the grouping.

    None when the universe can't be read or maps to no known calendar, so
    callers fail open rather than blocking a run on a guess.
    """
    from collections import Counter

    try:
        from fundmgr.config import get_enabled_tickers
        tickers = get_enabled_tickers(universe_path)
    except Exception:
        return None

    counts: Counter = Counter()
    for t in tickers:
        mic = _EXCHANGE_TO_CALENDAR.get((t.exchange or "").upper())
        if mic:
            counts[_CALENDAR_GROUP.get(mic, mic)] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def is_trading_day(mic: str, when: datetime | None = None) -> bool | None:
    """Does `mic` hold a session on `when`'s date? None when undeterminable.

    Session-level, not minute-level: a run fires at a fixed cron time and its
    fills follow minutes later, so the question is whether the market trades
    today at all — not whether it happens to be open at that instant.
    """
    try:
        import pandas as pd
        ts = pd.Timestamp(when or datetime.now(timezone.utc))
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return bool(_calendar(mic).is_session(ts.normalize()))
    except Exception:
        return None


def next_session(mic: str, after: datetime | None = None):
    """First session strictly after `after`'s date, or None if undeterminable."""
    try:
        import pandas as pd
        ts = pd.Timestamp(after or datetime.now(timezone.utc))
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        start = ts.normalize() + pd.Timedelta(days=1)
        end = start + pd.Timedelta(days=_NEXT_SESSION_HORIZON_DAYS)
        sessions = _calendar(mic).sessions_in_range(start, end)
        return sessions[0].date() if len(sessions) else None
    except Exception:
        return None
