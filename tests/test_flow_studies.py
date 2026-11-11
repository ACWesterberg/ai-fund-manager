"""FI's insider register reader and the two free base-rate studies."""
from __future__ import annotations

from datetime import date, datetime, timedelta

import httpx
import pytest

from fundmgr.config import AppConfig, UniverseTicker
from fundmgr.data.insiders import InsiderDataError, InsiderTrade, fetch_trades, parse_export
from fundmgr.engine import insider_study, volume_study
from fundmgr.engine.report_events import _Bench
from fundmgr.state.store import Store

HEADER = ("Publiceringsdatum;Emittent;LEI-kod;Anmälningsskyldig;Person i ledande ställning;"
          "Befattning;Närstående;Korrigering;Beskrivning av korrigering;Är förstagångsrapportering;"
          "Är kopplad till aktieprogram;Karaktär;Instrumenttyp;Instrumentnamn;ISIN;"
          "Transaktionsdatum;Volym;Volymsenhet;Pris;Valuta;Handelsplats;Status;")


def row(person="Anna A", karaktar="Förvärv", typ="Aktie", program="", status="Aktuell",
        volym="10000,0", pris="25,50", isin="SE0000000001", published="2026-10-01 18:00:00"):
    return (f"{published};Alfa AB;LEI;{person};{person};Verkställande\xa0direktör\xa0(VD);;;;;"
            f"{program};{karaktar};{typ};Alfa B;{isin};2026-09-29 00:00:00;{volym};Antal;"
            f"{pris};SEK;XSTO;{status};")


def export(*rows: str) -> bytes:
    return "\r\n".join([HEADER, *rows]).encode("utf-16")


def test_the_export_keeps_open_market_share_trades_only():
    trades = parse_export(export(
        row(),                                          # kept: a buy
        row(karaktar="Avyttring", person="Bo B"),       # kept: a sell
        row(program="Ja"),                              # a programme grant is pay, not a view
        row(status="Reviderad"),                        # superseded by its correction
        row(typ="Teckningsoption"),                     # not a share
        row(karaktar="Tilldelning"),                    # an allotment
        row(isin=""),                                   # can't be matched
    ))
    assert [(t.side, t.person) for t in trades] == [("buy", "Anna A"), ("sell", "Bo B")]
    assert trades[0].shares == 10000 and trades[0].price == 25.5
    assert trades[0].role == "Verkställande direktör (VD)"
    assert trades[0].published == datetime(2026, 10, 1, 18, 0)


def test_a_changed_export_format_is_said_out_loud():
    with pytest.raises(InsiderDataError, match="lacks columns"):
        parse_export("Publiceringsdatum;Emittent\r\n2026-10-01;Alfa".encode("utf-16"))
    with pytest.raises(InsiderDataError, match="header"):
        parse_export(b"<html>maintenance</html>")


def test_the_register_is_read_two_weeks_at_a_time():
    asked = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append((request.url.params["Publiceringsdatum.From"],
                      request.url.params["Publiceringsdatum.To"]))
        return httpx.Response(200, content=export(row()))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        trades = fetch_trades(date(2026, 10, 1), date(2026, 10, 20), client=client, pause=0)
    assert asked == [("2026-10-01", "2026-10-14"), ("2026-10-15", "2026-10-20")]
    assert len(trades) == 2


def trade(person: str, day: int, side: str = "buy", shares: float = 10_000, price: float = 20):
    return InsiderTrade(published=datetime(2026, 9, day, 18), traded=date(2026, 9, day),
                        issuer="Alfa AB", isin="SE1", person=person, role="VD", side=side,
                        shares=shares, price=price, currency="SEK", closely_associated=False)


def test_a_second_manager_buying_within_a_month_is_a_cluster():
    events = insider_study.find_events(
        [trade("Anna", 1), trade("Bo", 12), trade("Anna", 15)], {"SE1": "ALFA-B.ST"})
    assert [(e["kind"], e["published"]) for e in events] == [
        ("buy", "2026-09-01"), ("cluster", "2026-09-12")]


def test_a_managers_fills_on_one_day_count_together_against_the_threshold():
    small = [trade("Anna", 3, shares=2_000, price=20)] * 2           # 80k in two fills
    assert insider_study.find_events(small, {"SE1": "ALFA-B.ST"}) == []
    enough = [trade("Anna", 3, shares=3_000, price=20)] * 2          # 120k
    assert [e["kind"] for e in insider_study.find_events(enough, {"SE1": "ALFA-B.ST"})] == ["buy"]


def _series(start: date, closes: list[float], volumes: list[float]) -> list[dict]:
    return [{"date": (start + timedelta(days=i)).isoformat(), "open": c, "high": c, "low": c,
             "close": c, "volume": v} for i, (c, v) in enumerate(zip(closes, volumes))]


def test_insider_events_enter_after_publication_and_score_against_the_index(tmp_path, monkeypatch):
    store = Store(tmp_path / "f.db")
    start = date(2026, 8, 1)
    closes = [100.0] * 32 + [110.0] * 40                 # jumps the day after publication
    store.save_prices("ALFA-B.ST", _series(start, closes, [1000.0] * len(closes)))
    bench = _Bench([{"date": (start + timedelta(days=i)).isoformat(), "close": 1000.0}
                    for i in range(80)])
    monkeypatch.setattr(insider_study, "_fetch_history", lambda *a, **k: None)
    monkeypatch.setattr(insider_study, "_ensure_benchmark", lambda *a, **k: bench)
    universe = [UniverseTicker("Alfa", "ALFA-B.ST", "SE1", "SE", "OMXS", "Tech", True)]
    result = insider_study.run_study(store, AppConfig(), universe,
                                     fetch=lambda s, e: [trade("Anna", 1)], echo=lambda m: None)
    [event] = result["events"]
    assert event["published"] == "2026-09-01"
    assert event["fwd"][5] == pytest.approx(0.0)         # entered at the first close after: 110
    assert "insider buy" in insider_study.format_study(result)


def test_heavy_volume_days_split_by_direction(tmp_path, monkeypatch):
    store = Store(tmp_path / "f.db")
    start = date(2026, 1, 1)
    closes = [100.0] * 25 + [108.0] + [108.0] * 10 + [100.0] + [100.0] * 40
    volumes = [1000.0] * 25 + [5000.0] + [1000.0] * 10 + [5000.0] + [1000.0] * 40
    store.save_prices("ALFA-B.ST", _series(start, closes, volumes))
    bench = _Bench([{"date": (start + timedelta(days=i)).isoformat(), "close": 1000.0}
                    for i in range(90)])
    monkeypatch.setattr(volume_study, "_ensure_benchmark", lambda *a, **k: bench)
    universe = [UniverseTicker("Alfa", "ALFA-B.ST", "SE1", "SE", "OMXS", "Tech", True)]
    result = volume_study.run_study(store, AppConfig(), universe, echo=lambda m: None)
    assert result["dates"] == {"all": len(closes) - 20, "heavy_up": 1, "heavy_down": 1}
    assert result["groups"]["heavy_down"][5] == [pytest.approx(0.0)]
    assert "heavy, down" in volume_study.format_study(result)
