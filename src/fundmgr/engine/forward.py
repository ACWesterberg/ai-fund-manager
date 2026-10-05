"""Opt-in forward shadow experiments and immutable outcome collection.

This module never books trades, changes guidance, or sends notifications.
A recorded pair is evaluated at a common closing session AFTER both decisions.
"""
from __future__ import annotations

import copy
import json
import math
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from fundmgr.data.market_hours import _EXCHANGE_TO_CALENDAR, is_trading_day
from fundmgr.engine.experiments import (
    Case, FORWARD_CONVENTION, Outcomes, compare_guidance, digest, load_case,
    project_allocation, score_comparison, validate_candidate, write_report,
)


def experiment_root(cfg) -> Path:
    return cfg.db_path.parent / "shadow" / cfg.db_path.stem


def _session(calendar: str, day: date) -> bool:
    result = is_trading_day(calendar, datetime.combine(day, datetime.min.time(), timezone.utc))
    if result is None:
        raise ValueError(f"Unknown or unavailable calendar: {calendar}")
    return result


def _check_hash(payload: dict, key: str) -> None:
    if payload.get(key) != digest({k: v for k, v in payload.items() if k != key}):
        raise ValueError(f"Modified artifact: {key}")


def record_forward(cfg, snapshot: dict, run_id: str, tickers: list) -> Path | None:
    """Called only by a new, saved weekly run when shadow.candidate is configured.

    Reserve the run before calls. A crash/invalid pair cannot silently be retried
    and cherry-picked. Disabled profiles don't create files or call a provider.
    """
    if not cfg.shadow.candidate:
        return None
    case = load_case(snapshot)
    candidate = json.loads(Path(cfg.shadow.candidate).read_text())
    created = validate_candidate(case, candidate)
    now = datetime.now(timezone.utc)
    if not (created <= case.decision_time <= now
            and now - case.decision_time < timedelta(hours=2)):
        raise ValueError("Forward recording requires a fresh case and a pre-existing candidate")
    if case.fund_id != cfg.db_path.stem:
        raise ValueError("Forward case belongs to a different fund")
    for feature in case.features.values():
        if date.fromisoformat(feature.last_date) > case.decision_time.date():
            raise ValueError("Frozen quote date is later than the recorded decision")
    currency, calendar = cfg.shadow.benchmark_currency, cfg.shadow.benchmark_calendar
    if len(currency) != 3 or currency != currency.upper() or not calendar:
        raise ValueError("Configure shadow benchmark_currency and benchmark_calendar explicitly")
    venues = {t.yahoo_ticker: _EXCHANGE_TO_CALENDAR.get(t.exchange.upper()) for t in tickers}
    calendars = {t: venues.get(t) for t in case.features}
    if not all(calendars.values()):
        raise ValueError("Every shown ticker needs a known exchange calendar")
    for mic in set(calendars.values()) | {calendar}:
        _session(mic, now.date())
    path = experiment_root(cfg) / digest({"run_id": run_id})[:24]
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()  # Exclusive reservation, including across concurrent processes.
    except FileExistsError:
        return path
    registration = {"version": 1, "run_id": run_id, "registered_at": now.isoformat(),
                    "snapshot": snapshot, "candidate": candidate,
                    "calendars": calendars, "benchmark_calendar": calendar,
                    "benchmark_currency": currency,
                    "execution_convention": FORWARD_CONVENTION}
    registration["registration_hash"] = digest(registration)
    write_report(path / "registration.json", registration)
    try:
        pair = compare_guidance(snapshot, candidate)
        pair.pop("comparison_hash")
        pair.update(mode="registered_forward", registration_hash=registration["registration_hash"])
        pair["comparison_hash"] = digest(pair)
        write_report(path / "comparison.json", pair)
    except Exception as exc:
        write_report(path / "recording_error.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise
    return path


def execution_date(registration: dict, pair: dict) -> date:
    """First common session strictly after completion (up to 14 calendar days)."""
    after = datetime.fromisoformat(pair["completed_at"]).astimezone(timezone.utc).date()
    calendars = set(registration["calendars"].values()) | {registration["benchmark_calendar"]}
    for offset in range(1, 15):
        day = after + timedelta(days=offset)
        if all(_session(mic, day) for mic in calendars):
            return day
    raise ValueError("No common execution session in the next 14 calendar days")


def fetch_history(symbol: str, start: date, end: date) -> dict:
    """Dedicated adjusted history; never relies on the trading cache's freshness.

    yfinance's adjusted daily closes account for splits and cash distributions.
    Preserve provider rows/actions and retrieval time so a result is reproducible
    even if the provider later revises its history. No repair/fill of missing rows.
    """
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    frame = ticker.history(start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                           interval="1d", auto_adjust=True, actions=True, repair=False)
    if frame.empty or "Close" not in frame:
        raise ValueError(f"No adjusted history for {symbol}")
    rows = []
    for index, row in frame.iterrows():
        day = index.date()
        if start <= day <= end:
            rows.append({"date": day.isoformat(), "close": float(row["Close"]),
                         "dividends": float(row.get("Dividends", 0)),
                         "splits": float(row.get("Stock Splits", 0))})
    metadata = ticker.get_history_metadata()
    payload = {"symbol": symbol, "currency": metadata.get("currency"),
               "exchange_timezone": metadata.get("exchangeTimezoneName"),
               "requested_start": start.isoformat(), "requested_end": end.isoformat(),
               "source": "Yahoo Finance via yfinance", "provider_version": yf.__version__,
               "auto_adjust": True, "repair": False,
               "retrieved_at": datetime.now(timezone.utc).isoformat(), "rows": rows}
    digest(payload)  # Reject NaN/Infinity instead of dropping evidence silently.
    return payload


def _series(bundle: dict, symbol: str, currency: str) -> dict[date, float]:
    source = bundle[symbol]
    if source["symbol"] != symbol or source.get("currency") != currency or source.get("auto_adjust") is not True:
        raise ValueError(f"Unverified currency/adjustment convention for {symbol}")
    result = {}
    for row in source["rows"]:
        day, value = date.fromisoformat(row["date"]), row["close"]
        if day in result or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid or duplicate close for {symbol} on {day}")
        result[day] = value
    return result


def _mark(series: dict, day: date, calendar: str) -> float:
    # Carry only across confirmed closed sessions; a missing open session fails.
    for offset in range(8):
        previous = day - timedelta(days=offset)
        if calendar == "FX":
            opened = previous.weekday() < 5
        else:
            opened = _session(calendar, previous)
        if opened:
            if previous not in series:
                raise ValueError(f"Missing {calendar} session close on {previous}")
            return series[previous]
    raise ValueError(f"No recent session mark on {day}")


def derive_forward(registration: dict, pair: dict, bundle: dict) -> tuple[dict, Outcomes]:
    """Revalue the starting book at future entry, then apply the frozen targets.

    The derived case is accounting context only. Original prompts/decisions stay
    in comparison.json; no model sees future entry marks or collected labels.
    """
    _check_hash(registration, "registration_hash")
    _check_hash(pair, "comparison_hash")
    original = load_case(registration["snapshot"])
    if (pair.get("registration_hash") != registration["registration_hash"]
            or pair["case_hash"] != digest(original.model_dump(mode="json"))
            or pair["candidate_hash"] != digest(registration["candidate"])
            or pair["case"] != original.model_dump(mode="json")
            or pair["candidate"] != registration["candidate"]):
        raise ValueError("Registration and comparison lineage disagree")
    registered = datetime.fromisoformat(registration["registered_at"])
    started = datetime.fromisoformat(pair["started_at"])
    completed = datetime.fromisoformat(pair["completed_at"])
    if (pair.get("mode") != "registered_forward"
            or not original.decision_time <= registered <= started <= completed
            or validate_candidate(original, registration["candidate"]) > original.decision_time):
        raise ValueError("Comparison was not recorded after forward registration")
    if any(a["status"] != "ready" for a in pair["arms"].values()):
        raise ValueError("Invalid decision arm; no performance score")
    entry = execution_date(registration, pair)
    end = entry + timedelta(days=original.horizon_days)
    if end >= datetime.now(timezone.utc).date():
        raise ValueError("Wait until the day after the full outcome horizon")
    series = {t: _series(bundle, t, f.currency) for t, f in original.features.items()}
    benchmark = _series(bundle, original.benchmark, registration["benchmark_currency"])
    currencies = {f.currency for f in original.features.values()} | {registration["benchmark_currency"]}
    fx = {c: _series(bundle, f"{c}SEK=X", "SEK") for c in currencies - {"SEK"}} if original.basis == "SEK" else {}

    def rate(currency, day):
        return _mark(fx[currency], day, "FX") if currency in fx else 1.0

    def native(ticker, day):
        feat = original.features[ticker]
        base = date.fromisoformat(feat.last_date)
        # Normalize revised adjusted histories to the actual archived quote.
        if base not in series[ticker]:
            raise ValueError(f"Missing exact frozen quote date for {ticker}: {base}")
        return feat.last_price * _mark(series[ticker], day, registration["calendars"][ticker]) / series[ticker][base]

    case = original.model_copy(deep=True)
    case.decision_time = datetime.combine(entry, datetime.min.time(), timezone.utc)
    case.fx_rates = {c: rate(c, entry) for c in fx}
    for t, feat in case.features.items():
        feat.last_price, feat.last_date = native(t, entry), entry.isoformat()
    for position in case.positions:
        position.current_price_sek = case.price(position.ticker)
    derived = copy.deepcopy(pair)
    derived.pop("comparison_hash")
    derived.update(case=case.model_dump(mode="json"), case_hash=digest(case.model_dump(mode="json")),
                   original_comparison_hash=pair["comparison_hash"], execution_convention=FORWARD_CONVENTION,
                   context_role="accounting_only_original_prompts_in_source_comparison")
    for arm in derived["arms"].values():
        arm["allocation"] = project_allocation(case, arm["approved"])
    derived["comparison_hash"] = digest(derived)
    observations = []
    for offset in range(original.horizon_days + 1):
        day = entry + timedelta(days=offset)
        observations.append({"date": day.isoformat(),
            "prices": {t: native(t, day) * rate(f.currency, day) for t, f in original.features.items()},
            "benchmark": _mark(benchmark, day, registration["benchmark_calendar"]) * rate(registration["benchmark_currency"], day)})
    outcomes = Outcomes.model_validate({"case_hash": derived["case_hash"], "basis": case.basis,
        "benchmark": case.benchmark, "source": f"Archived adjusted histories sha256:{digest(bundle)}",
        "observations": observations})
    return derived, outcomes


def collect_forward(cfg, fetcher=None) -> list[dict]:
    """Collect matured registered pairs. No model calls; retries are append-only.

    A per-case process lock prevents competing collectors. Missing evidence stays
    pending. Completed scores are immutable and never fetched or scored again.
    """
    import fcntl

    fetcher = fetcher or fetch_history
    results = []
    for path in sorted(experiment_root(cfg).glob("*/registration.json")):
        directory = path.parent
        with (directory / ".collection.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                results.append({"case": directory.name, "status": "busy"})
                continue
            if (directory / "result.json").exists():
                results.append({"case": directory.name, "status": "complete"})
                continue
            attempt = None
            try:
                registration = json.loads(path.read_text())
                _check_hash(registration, "registration_hash")
                if not (directory / "comparison.json").exists():
                    raise ValueError("Recording incomplete; no automatic model retry")
                pair = json.loads((directory / "comparison.json").read_text())
                _check_hash(pair, "comparison_hash")
                case = Case.model_validate(pair["case"])
                if case.fund_id != cfg.db_path.stem:
                    raise ValueError("Case belongs to a different fund")
                if any(a["status"] != "ready" for a in pair["arms"].values()):
                    raise ValueError("Invalid decision arm; no automatic model retry")
                entry = execution_date(registration, pair)
                end = entry + timedelta(days=case.horizon_days)
                if end >= datetime.now(timezone.utc).date():
                    results.append({"case": directory.name, "status": "waiting", "horizon_end": end.isoformat()})
                    continue
                attempt = directory / "collections" / uuid.uuid4().hex
                attempt.mkdir(parents=True)
                start = min([date.fromisoformat(f.last_date) for f in case.features.values()] + [entry - timedelta(days=7)])
                symbols = set(case.features) | {case.benchmark}
                if case.basis == "SEK":
                    currencies = {f.currency for f in case.features.values()} | {registration["benchmark_currency"]}
                    symbols |= {f"{c}SEK=X" for c in currencies - {"SEK"}}
                bundle = {s: fetcher(s, start, end) for s in sorted(symbols)}
                write_report(attempt / "history.json", bundle)
                derived, outcomes = derive_forward(registration, pair, bundle)
                score = score_comparison(derived, outcomes)
                score.update(original_comparison_hash=pair["comparison_hash"],
                             registration_hash=registration["registration_hash"],
                             history_hash=digest(bundle), execution_convention=FORWARD_CONVENTION,
                             recorded_decision_time=case.decision_time.isoformat(),
                             collected_at=datetime.now(timezone.utc).isoformat())
                write_report(attempt / "valuation_case.json", derived)
                write_report(attempt / "outcomes.json", outcomes.model_dump(mode="json"))
                write_report(directory / "result.json", score)
                results.append({"case": directory.name, "status": "scored", "score": str(directory / "result.json")})
            except Exception as exc:
                error = {"case": directory.name, "status": "pending", "error": f"{type(exc).__name__}: {exc}"}
                if attempt:
                    write_report(attempt / "error.json", error)
                results.append(error)
    return results
