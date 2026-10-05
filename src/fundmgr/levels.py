"""
Stop / take-profit level policy, and the cadence of the alerts they raise.

Kept free of the data layer so it can be imported (and tested) without pulling
in yfinance or financedata — these are decisions about stored numbers, not
about fetching prices.
"""
from __future__ import annotations


# Why an action didn't fill. The filler is the only thing that knows, so it
# reports in this vocabulary and callers describe what it reported — see
# `deferred_note` for the failure that motivated saying it that way round.
SKIP_MARKET_CLOSED = "market_closed"
SKIP_NO_PRICE = "no_price"
SKIP_NOT_HELD = "not_held"
SKIP_BELOW_MIN = "below_min"
SKIP_AT_TARGET = "at_target"
SKIP_RISK = "risk_limit"

_DEFERRED_NOTES: dict[str, str] = {
    SKIP_MARKET_CLOSED: "market closed, sells on the next open",
    SKIP_NO_PRICE: "no live price, retries next cycle",
    SKIP_NOT_HELD: "no longer held",
    SKIP_BELOW_MIN: "below the minimum trade size",
    SKIP_AT_TARGET: "already at target weight",
    SKIP_RISK: "fill rejected by portfolio limits",
}

# Said when the filler recorded no reason at all. Names the consequence, which
# holds whatever the cause was, and claims nothing about the cause.
_DEFERRED_UNKNOWN = "not filled this cycle, the level stands"


def deferred_note(reason: str | None) -> str:
    """How to describe a triggered sell that didn't settle.

    The reason has to come from the filler. `check-stops` runs one fund-wide
    schedule (15:00–22:00 CET) across venues that shut at different times, so
    "market closed" is true of some deferrals and false of others — and it was
    asserted for all of them. It reached Telegram about a Norwegian name at
    15:00 with Oslo trading for another 80 minutes, in the same line that
    quoted the live price the alert had just fetched.

    A reason we don't have is reported as a reason we don't have.
    """
    return _DEFERRED_NOTES.get(reason or "", _DEFERRED_UNKNOWN)


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


def settled_sells(
    triggered: list[tuple],
    held_before: dict[str, float],
    held_after: dict[str, float],
) -> tuple[list[str], list[str]]:
    """Split triggered tickers into (sold, deferred), read off the book.

    Whether an auto-sell happened cannot be assumed from having asked for one:
    the filler refuses to trade a venue that is closed, and check-stops runs on
    one fund-wide schedule that outlasts some of its holdings' exchanges. A
    European name hitting its target after XETRA shuts is skipped on every
    cycle until the next open.

    Getting this wrong is not cosmetic. An auto-sold ticker is exempt from the
    once-a-day alert limit — correctly, since a completed trade is news rather
    than a standing condition — so counting a skipped fill as sold both claims a
    trade that never happened and repeats the claim every cycle.
    """
    sold, deferred = [], []
    for ticker, *_ in triggered:
        if held_after.get(ticker, 0.0) < held_before.get(ticker, 0.0):
            sold.append(ticker)
        else:
            deferred.append(ticker)
    return sold, deferred


def record_sent_alerts(hits: list[tuple], kind: str, store, today: str) -> None:
    """Mark `hits` as alerted today, so the next cycle stays quiet."""
    for hit in hits:
        store.record_position_alert(hit[0], today, kind)
