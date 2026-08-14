"""
Stop / take-profit level policy, and the cadence of the alerts they raise.

Kept free of the data layer so it can be imported (and tested) without pulling
in yfinance or financedata — these are decisions about stored numbers, not
about fetching prices.
"""
from __future__ import annotations


def merged_levels(action, prior: dict | None) -> tuple[float | None, float | None] | None:
    """
    Stop and take-profit to store for `action`, or None to leave it alone.

    Holds re-target as well as buys: a winner past its take-profit is held
    precisely because the consensus sees further upside, and a level frozen at
    entry would keep firing TARGET HIT against the decision the same run just
    made. Levels the model omitted fall back to the stored ones — take_profit_pct
    is optional, so treating an absent field as "clear it" would silently strip
    the stop off every held position the model didn't restate.
    """
    if action.side not in ("buy", "hold"):
        return None
    if not (action.stop_loss_pct or action.take_profit_pct):
        return None
    prior = prior or {}
    return (
        action.stop_loss_pct or prior.get("stop_pct"),
        action.take_profit_pct or prior.get("take_profit_pct"),
    )


def alertable_hits(
    hits: list[tuple],
    kind: str,
    auto_sold: list[str],
    store,
    today: str,
) -> list[tuple]:
    """
    Filter breach hits down to the ones worth pinging about.

    A breached level stays breached until the position is closed or re-targeted,
    and check-stops runs every 15 min through the session — so each ticker gets
    one `kind` alert per day. An auto-sold position always reports: that is a
    completed trade, not a standing condition. Detection itself is never
    suppressed, because the same lists drive auto-sell and the stop review.

    Pure query: pair with `record_sent_alerts` once the message is actually
    away, so a failed send or an unconfigured bot doesn't burn the day's alert.
    """
    return [
        hit for hit in hits
        if hit[0] in auto_sold or not store.has_sent_position_alert(hit[0], today, kind)
    ]


def record_sent_alerts(hits: list[tuple], kind: str, store, today: str) -> None:
    """Mark `hits` as alerted today, so the next cycle stays quiet."""
    for hit in hits:
        store.record_position_alert(hit[0], today, kind)
