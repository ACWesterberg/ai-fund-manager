"""Does heavy volume say anything here, and does its direction matter? Free.

The screener and the prompt reward a day whose volume is 2–3× its 20-day
average whatever the price did, so heavy selling earns the same bonus as heavy
buying. Before that weight changes, this measures it on the fund's own price
cache: every stock-day whose volume cleared the bar, split by whether the
stock rose or fell that day, against every stock-day as the baseline, with the
abnormal return (stock less the index) over the next 5, 20 and 28 trading days.
No model calls, no fetches beyond the benchmark, nothing written.

The windows overlap and heavy days cluster on market-wide news, so `n`
overstates the independent evidence; the distinct-dates count says by how much.
The intervals are a normal approximation for the same reason a bootstrap would
be: neither knows about the overlap, and this one is instant at 300k rows.
"""
from __future__ import annotations

import math
from statistics import mean, median, stdev
from typing import Callable

from fundmgr.engine.report_events import HORIZONS, _ensure_benchmark, _fmt_row, forward

WINDOW = 20
DEFAULT_RATIO = 2.5          # where `_signal_score` starts rewarding volume


def day_rows(series: list[dict]) -> list[tuple[int, float, float]]:
    """(index, volume ÷ prior 20-session average, day return %) for every usable day."""
    out = []
    for i in range(WINDOW, len(series)):
        vol = series[i].get("volume")
        prior = [r.get("volume") for r in series[i - WINDOW:i]]
        prev, cur = series[i - 1].get("close"), series[i].get("close")
        if not vol or not prev or not cur or any(not v for v in prior):
            continue
        out.append((i, vol / mean(prior), (cur / prev - 1) * 100))
    return out


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    m = mean(values)
    half = 1.645 * (stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0)
    return {"n": len(values), "mean": m, "median": median(values),
            "pos_share": sum(v > 0 for v in values) / len(values), "ci": (m - half, m + half)}


def run_study(store, cfg, tickers: list, ratio: float = DEFAULT_RATIO,
              echo: Callable[[str], None] = print) -> dict:
    bench = _ensure_benchmark(store, cfg, 400)
    groups: dict[str, dict[int, list[float]]] = {
        k: {n: [] for n in HORIZONS} for k in ("all", "heavy_up", "heavy_down")}
    dates: dict[str, set[str]] = {k: set() for k in groups}
    covered = 0
    for t in tickers:
        series = store.get_prices(t.yahoo_ticker)
        rows = day_rows(series)
        if rows:
            covered += 1
        for i, vol_ratio, day_ret in rows:
            fwd = forward(series, bench, i)
            keys = ["all"]
            if vol_ratio >= ratio and day_ret > 0:
                keys.append("heavy_up")
            elif vol_ratio >= ratio and day_ret < 0:
                keys.append("heavy_down")
            for k in keys:
                dates[k].add(series[i]["date"])
                for n in HORIZONS:
                    if fwd[n] is not None:
                        groups[k][n].append(fwd[n])
    echo(f"  {covered}/{len(tickers)} names have at least {WINDOW + 1} cached days.")
    return {"ratio": ratio, "n_tickers": len(tickers), "covered": covered,
            "groups": groups, "dates": {k: len(v) for k, v in dates.items()}}


def format_study(result: dict) -> str:
    labels = {"all": "every day", "heavy_up": "heavy, up", "heavy_down": "heavy, down"}
    d = result["dates"]
    lines = [
        f"\n─── Heavy volume — base rates (volume ≥ {result['ratio']:g}× its 20-day average) ───",
        f"  {result['covered']}/{result['n_tickers']} names in the price cache. Heavy days fell on "
        f"{d['heavy_up']} distinct dates (up) and {d['heavy_down']} (down).",
        "  Then: abnormal return (stock less the index) from that day's close, N trading days on.",
    ]
    for n in HORIZONS:
        lines.append(f"\n  +{n} trading days:")
        for key, label in labels.items():
            lines.append(_fmt_row(label, describe(result["groups"][key][n])))
    lines.append(
        "\n  Read: the screener rewards both heavy rows the same today. If the down-day row"
        "\n  trails the baseline, that bonus is backwards for it. Intervals ignore the"
        "\n  overlap, so treat anything near zero as no answer."
    )
    return "\n".join(lines)
