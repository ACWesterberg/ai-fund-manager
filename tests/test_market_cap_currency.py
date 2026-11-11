from __future__ import annotations

from pathlib import Path

from fundmgr.data.fundamentals import apply_to_features
from fundmgr.data.prices import TickerFeatures
from fundmgr.state.store import Store


def _feat(ticker: str, currency: str = "SEK") -> TickerFeatures:
    return TickerFeatures(
        ticker=ticker, name=ticker, last_price=100.0, last_date="2026-11-10",
        data_age_trading_days=1, currency=currency,
    )


def _apply(tmp_path: Path, rows: dict[str, dict], feats: dict, rates: dict[str, float]):
    store = Store(tmp_path / "t.db")
    store.initialise(1000)
    for ticker, data in rows.items():
        store.save_fundamentals(ticker, data)
    asked: list[str] = []

    def rate(cur: str) -> float | None:
        asked.append(cur)
        return rates.get(cur)

    apply_to_features(feats, store, rate_to_sek=rate)
    return asked


def test_usd_market_cap_is_converted_to_sek(tmp_path):
    """The mandates' size bands are in SEK; a native USD figure read ~10x too small."""
    feats = {"NVDA": _feat("NVDA", "USD")}
    _apply(tmp_path, {"NVDA": {"market_cap": 4.0e12, "currency": "USD"}}, feats, {"USD": 10.0})

    f = feats["NVDA"]
    assert f.market_cap_m == 4_000_000
    assert f.market_cap_currency == "USD"
    assert f.market_cap_msek == 40_000_000
    assert "mktcap 40,000,000M SEK" in f.to_prompt_block()


def test_pence_quoted_cap_converts_at_the_pound_rate(tmp_path):
    feats = {"VOD.L": _feat("VOD.L", "GBp")}
    asked = _apply(tmp_path, {"VOD.L": {"market_cap": 2.0e10, "currency": "GBp"}},
                   feats, {"GBP": 13.0})
    assert asked == ["GBP"]
    assert feats["VOD.L"].market_cap_msek == 260_000


def test_missing_rate_shows_the_native_figure_with_its_code(tmp_path):
    feats = {"X": _feat("X", "EUR")}
    _apply(tmp_path, {"X": {"market_cap": 3.0e9, "currency": "EUR"}}, feats, {})

    f = feats["X"]
    assert f.market_cap_msek is None
    assert "mktcap 3,000M EUR" in f.to_prompt_block()


def test_sek_needs_no_lookup_beyond_identity_and_rates_are_memoised(tmp_path):
    feats = {"A.ST": _feat("A.ST"), "B.ST": _feat("B.ST")}
    asked = _apply(
        tmp_path,
        {"A.ST": {"market_cap": 5.0e9, "currency": "SEK"},
         "B.ST": {"market_cap": 1.0e9, "currency": "SEK"}},
        feats, {"SEK": 1.0},
    )
    assert asked == ["SEK"]
    assert feats["A.ST"].market_cap_msek == 5_000
    assert "mktcap 5,000M SEK" in feats["A.ST"].to_prompt_block()


def test_a_failing_rate_lookup_fails_open(tmp_path):
    store = Store(tmp_path / "t.db")
    store.initialise(1000)
    store.save_fundamentals("X", {"market_cap": 1.0e9, "currency": "USD"})
    feats = {"X": _feat("X", "USD")}

    def boom(cur: str) -> float | None:
        raise RuntimeError("offline")

    apply_to_features(feats, store, rate_to_sek=boom)
    assert feats["X"].market_cap_msek is None
    assert feats["X"].market_cap_m == 1_000
