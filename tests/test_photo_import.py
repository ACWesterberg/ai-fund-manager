"""Portfolio photo → holdings extraction, and the web import flow it feeds."""
import json
from io import BytesIO

import pytest

from fundmgr import paper
from fundmgr.vision import portfolio_photo as pp


# ── Number coercion (Nordic broker formatting) ────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("1 234,50", 1234.50),      # space thousands + decimal comma (Avanza/Montrose)
    ("1.234,50", 1234.50),      # dot thousands + decimal comma (German/Nordic)
    ("1,234.50", 1234.50),      # US
    ("12 500", 12500.0),
    ("291,50 SEK", 291.50),
    ("-12,5", -12.5),
    ("(3 000)", -3000.0),       # parenthesised negative
    ("8,2 %", 8.2),
    ("1,234", 1234.0),          # 3 trailing digits → thousands separator
    ("1,23", 1.23),             # 2 trailing digits → decimal comma
    (1234.5, 1234.5),
    (12, 12.0),
])
def test_to_float_parses_broker_numbers(raw, expected):
    assert pp.to_float(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", [None, "", "  ", "n/a", "—", "-", True, False])
def test_to_float_rejects_non_numbers(raw):
    assert pp.to_float(raw) is None


# ── JSON recovery from a chatty model ─────────────────────────────────────────

def test_parse_json_plain():
    assert pp._parse_json('{"holdings": []}') == {"holdings": []}


def test_parse_json_from_fenced_prose():
    raw = 'Here you go:\n```json\n{"holdings": [{"name": "Volvo"}]}\n```\nHope that helps!'
    assert pp._parse_json(raw)["holdings"][0]["name"] == "Volvo"


def test_parse_json_handles_braces_inside_strings():
    raw = '{"notes": "row 3 said {unreadable}", "holdings": []}'
    assert pp._parse_json(raw)["notes"] == "row 3 said {unreadable}"


def test_parse_json_accepts_a_bare_array():
    assert pp._parse_json('[{"name": "Volvo"}]')["holdings"][0]["name"] == "Volvo"


def test_parse_json_gives_up_cleanly():
    assert pp._parse_json("no json at all") is None
    assert pp._parse_json("") is None


# ── Row normalisation ─────────────────────────────────────────────────────────

def test_rows_derive_cost_basis_from_sek_purchase_value():
    """Inköpsvärde ÷ antal is the honest cost basis — it beats a native GAV."""
    rows, _warn = pp._normalise_rows([{
        "name": "Alphabet", "ticker": "GOOGL", "shares": 10,
        "avg_cost_native": 180.0, "cost_value_sek": 34_290, "currency": "USD",
    }])
    assert rows[0]["avg_cost_sek"] == pytest.approx(3429.0)
    assert rows[0]["avg_cost_basis"] == "cost_value_sek"


def test_rows_use_native_gav_when_the_row_is_already_sek():
    rows, _ = pp._normalise_rows([{
        "name": "Volvo B", "ticker": "VOLV-B", "shares": 100,
        "avg_cost_native": 291.5, "currency": "SEK",
    }])
    assert rows[0]["avg_cost_sek"] == pytest.approx(291.5)
    assert rows[0]["avg_cost_basis"] == "avg_cost_native"


def test_rows_fall_back_to_market_value_and_flag_it():
    rows, warnings = pp._normalise_rows([{
        "name": "Sandvik", "ticker": "SAND", "shares": 50,
        "market_value_sek": 10_000, "currency": "SEK",
    }])
    assert rows[0]["avg_cost_sek"] == pytest.approx(200.0)
    assert rows[0]["avg_cost_basis"] == "market_value_sek"
    assert not warnings  # a derivable basis isn't a warning; the UI flags it in amber


def test_rows_warn_on_missing_numbers():
    rows, warnings = pp._normalise_rows([{"name": "Mystery Corp", "ticker": "MYST"}])
    assert rows[0]["shares"] is None and rows[0]["avg_cost_sek"] is None
    assert any("no share count" in w for w in warnings)
    assert any("no cost basis" in w for w in warnings)


def test_rows_drop_summary_and_cash_lines():
    rows, _ = pp._normalise_rows([
        {"name": "Totalt", "shares": None},
        {"name": "Likvida medel", "shares": None},
        {"name": "Volvo B", "ticker": "VOLV-B", "shares": 10, "avg_cost_native": 300,
         "currency": "SEK"},
    ])
    assert [r["name"] for r in rows] == ["Volvo B"]


def test_rows_reject_a_malformed_isin():
    rows, warnings = pp._normalise_rows([{
        "name": "Volvo B", "isin": "SE00001154", "shares": 10,
        "avg_cost_native": 300, "currency": "SEK",
    }])
    assert rows[0]["isin"] == ""
    assert any("not a valid ISIN" in w for w in warnings)


def test_rows_keep_a_well_formed_isin():
    rows, _ = pp._normalise_rows([{"name": "Volvo B", "isin": "se0000115446"}])
    assert rows[0]["isin"] == "SE0000115446"


def test_rows_clamp_confidence():
    rows, _ = pp._normalise_rows([
        {"name": "A", "confidence": 5},
        {"name": "B", "confidence": -1},
        {"name": "C"},
    ])
    assert [r["confidence"] for r in rows] == [1.0, 0.0, 0.5]


def test_rows_skip_entries_with_no_identity():
    rows, _ = pp._normalise_rows([{"shares": 10}, "not a dict", {"name": "Volvo"}])
    assert [r["name"] for r in rows] == ["Volvo"]


# ── Ticker resolution ─────────────────────────────────────────────────────────

def test_isin_resolves_to_universe_ticker():
    holdings = [{"name": "Volvo B", "raw_ticker": "", "isin": "SE0000115446",
                 "ticker": None, "currency": "SEK", "resolved_via": None}]
    assert pp._resolve_tickers(holdings)[0]["ticker"] == "VOLV-B.ST"
    assert holdings[0]["resolved_via"] == "isin"


def test_broker_ticker_maps_to_the_yahoo_symbol():
    holdings = [{"name": "Kongsberg", "raw_ticker": "KOG", "isin": "",
                 "ticker": None, "currency": "NOK", "resolved_via": None}]
    assert pp._resolve_tickers(holdings)[0]["ticker"] == "KOG.OL"


def test_name_only_row_falls_back_to_the_alias_map(monkeypatch):
    monkeypatch.setattr(paper, "_search_symbol", lambda name: None)  # no network
    holdings = [{"name": "Nvidia", "raw_ticker": "", "isin": "",
                 "ticker": None, "currency": "USD", "resolved_via": None}]
    assert pp._resolve_tickers(holdings)[0]["ticker"] == "NVDA"


def test_unresolvable_row_comes_back_blank(monkeypatch):
    monkeypatch.setattr(paper, "_search_symbol", lambda name: None)
    holdings = [{"name": "Totally Made Up Ltd", "raw_ticker": "", "isin": "",
                 "ticker": None, "currency": "USD", "resolved_via": None}]
    assert pp._resolve_tickers(holdings)[0]["ticker"] is None


# ── extract_holdings orchestration (LLM mocked) ───────────────────────────────

SAMPLE_ANSWER = json.dumps({
    "holdings": [
        {"name": "Volvo B", "ticker": "VOLV-B", "isin": "SE0000115446", "shares": "100",
         "avg_cost_native": "291,50", "currency": "SEK", "confidence": 0.9},
        {"name": "Nvidia", "ticker": "NVDA", "shares": "12",
         "cost_value_sek": "41 148", "currency": "USD", "confidence": 0.8},
        {"name": "Totalt", "shares": None},
    ],
    "cash_sek": "12 500",
    "total_value_sek": "83 798",
    "account_name": "Montrose KF",
    "notes": "",
})


@pytest.fixture
def fake_vision(monkeypatch):
    """Stub the vision call and the image/OCR layers — no network, no Pillow."""
    calls = {}

    def _fake_call(model, api_key, content):
        calls["model"] = model
        calls["content"] = content
        return SAMPLE_ANSWER

    monkeypatch.setattr(pp, "_call_vision", _fake_call)
    monkeypatch.setattr(pp, "_prepare_image", lambda raw: ("Zm9v", "image/jpeg"))
    monkeypatch.setattr(pp, "ocr_text", lambda raw: "ISIN: SE0000115446  Antal: 100")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    return calls


def test_extract_holdings_end_to_end(fake_vision):
    result = pp.extract_holdings([b"img"], resolve=False)

    assert [h["name"] for h in result["holdings"]] == ["Volvo B", "Nvidia"]
    assert result["holdings"][0]["avg_cost_sek"] == pytest.approx(291.5)
    assert result["holdings"][1]["avg_cost_sek"] == pytest.approx(41148 / 12)
    assert result["cash_sek"] == pytest.approx(12500)
    assert result["total_value_sek"] == pytest.approx(83798)
    assert result["account_name"] == "Montrose KF"
    assert result["ocr_used"] is True


def test_extract_holdings_sends_image_and_ocr_hint(fake_vision):
    pp.extract_holdings([b"img1", b"img2"], resolve=False)
    content = fake_vision["content"]
    assert sum(1 for part in content if part["type"] == "image_url") == 2
    assert "SE0000115446" in content[0]["text"]      # OCR transcript rides along
    assert "2 screenshots" in content[0]["text"]


def test_extract_holdings_without_ocr(monkeypatch, fake_vision):
    monkeypatch.setattr(pp, "ocr_text", lambda raw: "")
    result = pp.extract_holdings([b"img"], resolve=False)
    assert result["ocr_used"] is False
    assert "OCR" not in fake_vision["content"][0]["text"]


def test_extract_holdings_resolves_tickers(fake_vision, monkeypatch):
    monkeypatch.setattr(paper, "_search_symbol", lambda name: None)
    result = pp.extract_holdings([b"img"], resolve=True)
    assert result["holdings"][0]["ticker"] == "VOLV-B.ST"   # via ISIN
    assert result["holdings"][1]["ticker"] == "NVDA"


def test_extract_holdings_needs_an_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(pp.PhotoExtractionError, match="OPENAI_API_KEY"):
        pp.extract_holdings([b"img"])


def test_extract_holdings_rejects_no_images():
    with pytest.raises(pp.PhotoExtractionError, match="No images"):
        pp.extract_holdings([])


def test_extract_holdings_caps_the_image_count(fake_vision):
    with pytest.raises(pp.PhotoExtractionError, match="Too many images"):
        pp.extract_holdings([b"img"] * (pp.MAX_IMAGES + 1))


def test_extract_holdings_errors_on_unparseable_answer(monkeypatch, fake_vision):
    monkeypatch.setattr(pp, "_call_vision", lambda *a: "I can't read that image, sorry.")
    with pytest.raises(pp.PhotoExtractionError, match="readable JSON"):
        pp.extract_holdings([b"img"], resolve=False)


def test_extract_holdings_errors_when_no_rows_survive(monkeypatch, fake_vision):
    monkeypatch.setattr(pp, "_call_vision", lambda *a: '{"holdings": []}')
    with pytest.raises(pp.PhotoExtractionError, match="No holdings could be read"):
        pp.extract_holdings([b"img"], resolve=False)


def test_prepare_image_downscales_a_large_screenshot():
    pytest.importorskip("PIL")
    from PIL import Image
    buf = BytesIO()
    Image.new("RGB", (3000, 2000), (10, 10, 10)).save(buf, format="PNG")
    b64, mime = pp._prepare_image(buf.getvalue())
    assert mime == "image/jpeg"

    import base64
    out = Image.open(BytesIO(base64.b64decode(b64)))
    assert max(out.size) == pp._MAX_EDGE


def test_prepare_image_flattens_transparency():
    """A dark-mode PNG with alpha must not come out black-on-black."""
    pytest.importorskip("PIL")
    from PIL import Image
    buf = BytesIO()
    Image.new("RGBA", (40, 40), (0, 0, 0, 0)).save(buf, format="PNG")
    b64, _ = pp._prepare_image(buf.getvalue())

    import base64
    out = Image.open(BytesIO(base64.b64decode(b64)))
    assert out.mode == "RGB"
    assert out.getpixel((5, 5)) == (255, 255, 255)


def test_prepare_image_rejects_a_non_image():
    pytest.importorskip("PIL")
    with pytest.raises(pp.PhotoExtractionError, match="Could not read image"):
        pp._prepare_image(b"this is not an image")


# ── Web flow ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path / "paper")
    monkeypatch.setattr(paper, "detect_currency", lambda t: "SEK")
    monkeypatch.setattr(paper, "_cache_price_history", lambda store, tickers, **kw: None)
    monkeypatch.setattr(paper, "_search_symbol", lambda name: None)
    import fundmgr.data.benchmark as benchmark
    monkeypatch.setattr(benchmark, "fetch_and_cache_benchmark", lambda store, **kw: True)
    import fundmgr.data.quotes as quotes
    monkeypatch.setattr(quotes, "live_prices", lambda tickers: {t: 300.0 for t in tickers})

    from fundmgr.web.app import app
    return TestClient(app)


def test_live_page_offers_photo_import(client):
    r = client.get("/live/")
    assert r.status_code == 200
    assert "Import from a photo of your portfolio" in r.text
    assert "/live/extract-photos" in r.text


def test_extract_photos_returns_review_rows(client, monkeypatch):
    monkeypatch.setattr(
        "fundmgr.vision.portfolio_photo.extract_holdings",
        lambda images, **kw: {
            "holdings": [{"name": "Volvo B", "ticker": "VOLV-B.ST", "shares": 100,
                          "avg_cost_sek": 291.5, "avg_cost_basis": "avg_cost_native",
                          "raw_ticker": "VOLV-B", "isin": "SE0000115446",
                          "confidence": 0.9, "resolved_via": "isin"}],
            "cash_sek": 12500.0, "total_value_sek": 41650.0,
            "account_name": "Montrose KF", "notes": "", "warnings": [],
            "ocr_used": True, "model": "gpt-4o-mini",
        })

    r = client.post("/live/extract-photos", files={"photos": ("shot.png", b"fake", "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["holdings"][0]["ticker"] == "VOLV-B.ST"
    assert body["cash_sek"] == 12500.0
    assert body["unresolved"] == 0


def test_extract_photos_reports_unresolved_rows(client, monkeypatch):
    monkeypatch.setattr(
        "fundmgr.vision.portfolio_photo.extract_holdings",
        lambda images, **kw: {
            "holdings": [{"name": "Mystery", "ticker": None, "shares": 1,
                          "avg_cost_sek": 10.0, "avg_cost_basis": None, "raw_ticker": "",
                          "isin": "", "confidence": 0.3, "resolved_via": None}],
            "cash_sek": None, "total_value_sek": None, "account_name": None,
            "notes": "one row was cut off", "warnings": ["Mystery: no share count read"],
            "ocr_used": False, "model": "gpt-4o-mini",
        })

    body = client.post("/live/extract-photos",
                       files={"photos": ("s.png", b"x", "image/png")}).json()
    assert body["unresolved"] == 1
    assert body["warnings"] == ["Mystery: no share count read"]
    assert body["notes"] == "one row was cut off"


def test_extract_photos_surfaces_extraction_errors(client, monkeypatch):
    def boom(images, **kw):
        raise pp.PhotoExtractionError("No holdings could be read from the photo.")
    monkeypatch.setattr("fundmgr.vision.portfolio_photo.extract_holdings", boom)

    r = client.post("/live/extract-photos", files={"photos": ("s.png", b"x", "image/png")})
    assert r.status_code == 400
    assert "No holdings could be read" in r.json()["error"]


def test_extract_photos_rejects_an_empty_upload(client):
    r = client.post("/live/extract-photos", files={"photos": ("s.png", b"", "image/png")})
    assert r.status_code == 400
    assert r.json()["ok"] is False


def test_extract_photos_rejects_an_oversized_upload(client):
    from fundmgr.web.paper import _MAX_UPLOAD_BYTES
    big = b"x" * (_MAX_UPLOAD_BYTES + 1)
    r = client.post("/live/extract-photos", files={"photos": ("huge.png", big, "image/png")})
    assert r.status_code == 400
    assert "larger than" in r.json()["error"]


def test_create_from_photo_seeds_the_sleeve(client):
    rows = [
        {"ticker": "VOLV-B.ST", "name": "Volvo B", "shares": "100", "avg_cost_sek": "291.5",
         "kill_criterion": "order book halves", "max_drawdown_pct": "25",
         "horizon_months": "12"},
        {"ticker": "SAND.ST", "name": "Sandvik", "shares": "50", "avg_cost_sek": "200",
         "kill_criterion": "", "max_drawdown_pct": "", "horizon_months": ""},
    ]
    r = client.post("/live/create-from-photo", data={
        "name": "Photo Sleeve", "holdings_json": json.dumps(rows),
        "cash_sek": "12500", "benchmark": "URTH",
    }, follow_redirects=False)
    assert r.status_code == 303

    _meta, store = paper.open_portfolio("photo-sleeve")
    positions = {p.ticker: p for p in store.get_positions()}
    assert positions["VOLV-B.ST"].shares == 100
    assert positions["VOLV-B.ST"].avg_cost_sek == pytest.approx(291.5)
    assert store.get_cash() == pytest.approx(12500)

    from fundmgr import watchplan
    plan = watchplan.plan_for(store, "VOLV-B.ST")
    assert plan["kill_criterion"] == "order book halves"
    assert plan["kill_rules"]["max_drawdown_pct"] == 25.0
    assert plan["horizon"]["review_date"]


def test_photo_sleeve_dashboard_shows_the_watch_plan(client):
    rows = [{"ticker": "VOLV-B.ST", "name": "Volvo B", "shares": "100",
             "avg_cost_sek": "291.5", "kill_criterion": "order book halves",
             "max_drawdown_pct": "25", "horizon_months": "12"}]
    client.post("/live/create-from-photo", data={
        "name": "Dash Sleeve", "holdings_json": json.dumps(rows), "cash_sek": "0",
    }, follow_redirects=False)

    r = client.get("/live/dash-sleeve")
    assert r.status_code == 200
    assert "Watch status" in r.text
    assert "order book halves" in r.text
    assert "−25% from cost" in r.text
    assert "Set a kill criterion" in r.text


def test_create_from_photo_rejects_rows_without_cost(client):
    rows = [{"ticker": "VOLV-B.ST", "name": "Volvo B", "shares": "100"}]
    r = client.post("/live/create-from-photo", data={
        "name": "No Cost", "holdings_json": json.dumps(rows), "cash_sek": "0",
    })
    assert r.status_code == 200
    assert "missing its share count or cost basis" in r.text


def test_create_from_photo_rejects_bad_json(client):
    r = client.post("/live/create-from-photo", data={
        "name": "Broken", "holdings_json": "{not json", "cash_sek": "0",
    })
    assert r.status_code == 200
    assert "Could not read the reviewed holdings" in r.text


def test_create_from_photo_rejects_an_empty_list(client):
    r = client.post("/live/create-from-photo", data={
        "name": "Empty", "holdings_json": "[]", "cash_sek": "0",
    })
    assert r.status_code == 200
    assert "No holdings to import" in r.text


# ── Editing the plan from the dashboard ───────────────────────────────────────

def _sleeve(client, name="Edit Sleeve"):
    rows = [{"ticker": "VOLV-B.ST", "name": "Volvo B", "shares": "100",
             "avg_cost_sek": "300"}]
    client.post("/live/create-from-photo", data={
        "name": name, "holdings_json": json.dumps(rows), "cash_sek": "0",
    }, follow_redirects=False)
    return paper.slugify(name)


def test_watchplan_form_saves_criteria(client):
    slug = _sleeve(client)
    r = client.post(f"/live/{slug}/watchplan", data={
        "ticker": "volv-b.st", "kill_criterion": "order book halves",
        "max_drawdown_pct": "20", "price_below": "150", "horizon_months": "6",
        "horizon_note": "capital goods cycle should have turned",
    }, follow_redirects=False)
    assert r.status_code == 303

    from fundmgr import watchplan
    _meta, store = paper.open_portfolio(slug)
    plan = watchplan.plan_for(store, "VOLV-B.ST")
    assert plan["kill_criterion"] == "order book halves"
    assert plan["kill_rules"]["max_drawdown_pct"] == 20.0
    assert plan["kill_rules"]["price_below"] == 150.0
    assert plan["horizon"]["note"] == "capital goods cycle should have turned"


def test_watchplan_form_clears_a_position(client):
    slug = _sleeve(client, "Clear Sleeve")
    client.post(f"/live/{slug}/watchplan",
                data={"ticker": "VOLV-B.ST", "kill_criterion": "x", "max_drawdown_pct": "20"},
                follow_redirects=False)
    client.post(f"/live/{slug}/watchplan",
                data={"ticker": "VOLV-B.ST", "remove": "1"}, follow_redirects=False)

    from fundmgr import watchplan
    _meta, store = paper.open_portfolio(slug)
    assert watchplan.plan_tickers(store) == set()


def test_watchplan_form_rejects_a_bad_date(client):
    slug = _sleeve(client, "Bad Date Sleeve")
    r = client.post(f"/live/{slug}/watchplan", data={
        "ticker": "VOLV-B.ST", "horizon_date": "11/08/2027",
    }, follow_redirects=False)
    assert r.status_code == 303
    assert "Bad+horizon+date" in r.headers["location"] or "Bad horizon date" in r.headers["location"]
    assert "ok=0" in r.headers["location"]


def test_watchplan_form_requires_a_ticker(client):
    slug = _sleeve(client, "No Ticker Sleeve")
    r = client.post(f"/live/{slug}/watchplan", data={"ticker": "  "},
                    follow_redirects=False)
    assert "ok=0" in r.headers["location"]


def test_watchplan_form_404s_on_unknown_sleeve(client):
    r = client.post("/live/nope/watchplan", data={"ticker": "NVDA"})
    assert r.status_code == 404


# ── Regression: the value column read as the share count ──────────────────────
#
# A real Montrose ISK import produced 13 rows whose "shares" were each the
# position's SEK market value. Cost derived as value/shares = exactly 1.00, and
# the dashboard then multiplied a currency amount by a live share price: a
# 1,004,314 SEK portfolio showed a NAV of 264,869,938 SEK.

def test_share_count_equal_to_market_value_is_rejected():
    rows, warnings = pp._normalise_rows([{
        "name": "Evolution", "ticker": "EVO", "shares": 109783,
        "market_value_sek": 109783, "currency": "SEK",
    }])
    assert rows[0]["shares"] is None, "a value masquerading as a count must not survive"
    assert rows[0]["avg_cost_sek"] is None, "and must not become a 1.00 cost basis"
    assert any("read as a value, not a count" in w for w in warnings)


def test_share_count_equal_to_purchase_value_is_rejected():
    rows, warnings = pp._normalise_rows([{
        "name": "Vitec", "ticker": "VIT-B", "shares": 98332,
        "cost_value_sek": 98332, "currency": "SEK",
    }])
    assert rows[0]["shares"] is None
    assert any("purchase value" in w for w in warnings)


def test_a_genuine_row_where_count_and_value_differ_survives():
    rows, warnings = pp._normalise_rows([{
        "name": "Volvo B", "ticker": "VOLV-B", "shares": 100,
        "cost_value_sek": 29150, "currency": "SEK",
    }])
    assert rows[0]["shares"] == 100
    assert rows[0]["avg_cost_sek"] == pytest.approx(291.5)
    assert warnings == []


def test_sanity_check_flags_a_book_of_unit_costs():
    holdings = [{"shares": 1000.0, "avg_cost_sek": 1.0},
                {"shares": 2000.0, "avg_cost_sek": 1.0}]
    out = pp._sanity_check(holdings, None)
    assert any("exactly 1.00" in w for w in out)


def test_sanity_check_accepts_real_costs():
    holdings = [{"shares": 100.0, "avg_cost_sek": 291.5},
                {"shares": 50.0, "avg_cost_sek": 200.0}]
    assert pp._sanity_check(holdings, 39_150) == []


def test_sanity_check_flags_a_total_that_does_not_reconcile():
    """The 264× discrepancy the real import produced, caught before import."""
    holdings = [{"shares": 109783.0, "avg_cost_sek": 737.80}]
    out = pp._sanity_check(holdings, 1_004_314)
    assert any("read from the wrong column" in w for w in out)


def test_sanity_check_says_so_when_there_is_no_total_to_check_against():
    """"Could not check" must not look like "checked and fine" — the same
    distinction the kill judge draws between unread and passing."""
    holdings = [{"shares": 100.0, "avg_cost_sek": 291.5}]
    out = pp._sanity_check(holdings, None)
    assert any("No portfolio total was visible" in w for w in out)


def test_sanity_check_ignores_rows_without_both_numbers():
    assert pp._sanity_check([{"shares": None, "avg_cost_sek": None}], 1000) == []


# ── Regression: Nordic names resolving to a foreign listing ────────────────────
#
# The same import resolved Swedish small caps to Frankfurt lines (52SA.F, 4HY.F,
# X5T.F, 51R.F) and AstraZeneca to London (AZN.L). Those quote in the wrong
# currency, so the price looks real and the position is wrong.

def _mock_search(monkeypatch, symbols):
    class _Search:
        def __init__(self, query, max_results=10):
            self.quotes = [{"symbol": s, "quoteType": "EQUITY"} for s in symbols]
    import yfinance as yf
    monkeypatch.setattr(yf, "Search", _Search)


def test_sek_row_prefers_the_stockholm_listing(monkeypatch):
    _mock_search(monkeypatch, ["4HY.F", "4HY.DE", "VSSAB-B.ST"])
    assert pp._search_symbol_preferring("Vestum", "SEK") == "VSSAB-B.ST"


def test_gbp_row_prefers_london(monkeypatch):
    _mock_search(monkeypatch, ["AZN.F", "AZN.L", "AZN"])
    assert pp._search_symbol_preferring("AstraZeneca", "GBP") == "AZN.L"


def test_usd_row_prefers_the_bare_us_symbol(monkeypatch):
    _mock_search(monkeypatch, ["NVD.DE", "NVDA"])
    assert pp._search_symbol_preferring("Nvidia", "USD") == "NVDA"


def test_falls_back_to_plain_search_when_no_home_listing(monkeypatch):
    _mock_search(monkeypatch, ["FOO.F"])
    monkeypatch.setattr(paper, "_search_symbol", lambda name: "FALLBACK")
    assert pp._search_symbol_preferring("Obscure Co", "SEK") == "FALLBACK"


def test_unknown_currency_falls_straight_through(monkeypatch):
    monkeypatch.setattr(paper, "_search_symbol", lambda name: "PLAIN")
    assert pp._search_symbol_preferring("Whatever", "XYZ") == "PLAIN"


def test_resolution_uses_the_market_aware_search(monkeypatch):
    monkeypatch.setattr(pp, "_search_symbol_preferring",
                        lambda name, currency: "VSSAB-B.ST" if currency == "SEK" else None)
    holdings = [{"name": "Vestum", "raw_ticker": "", "isin": "", "ticker": None,
                 "currency": "SEK", "resolved_via": None}]
    assert pp._resolve_tickers(holdings)[0]["ticker"] == "VSSAB-B.ST"


# ── Deriving the share count when the screen has no "Antal" column ────────────
#
# A Montrose ISK holdings view shows name, price and market value but no share
# count. The count is value ÷ price, and it lands on whole numbers.

def test_share_count_is_derived_from_value_and_price():
    rows, warnings = pp._normalise_rows([{
        "name": "Evolution", "ticker": "EVO", "shares": None,
        "last_price_native": 736.8, "market_value_sek": 109783, "currency": "SEK",
    }])
    assert rows[0]["shares"] == 149          # 109783 / 736.8 = 149.0
    assert rows[0]["shares_basis"] == "derived"
    assert any("derived 149" in w for w in warnings)


@pytest.mark.parametrize("price,value,expected", [
    (736.8, 109783, 149), (244.0, 98332, 403), (29.6, 80778, 2729),
    (1523.5, 76175, 50), (53.6, 71073, 1326),
])
def test_derivation_matches_the_real_portfolio(price, value, expected):
    rows, _ = pp._normalise_rows([{
        "name": "X", "ticker": "X", "shares": None,
        "last_price_native": price, "market_value_sek": value, "currency": "SEK",
    }])
    assert rows[0]["shares"] == expected


def test_derivation_uses_avg_cost_when_no_last_price():
    rows, _ = pp._normalise_rows([{
        "name": "Vitec", "ticker": "VIT-B", "shares": None,
        "avg_cost_native": 244.0, "market_value_sek": 98332, "currency": "SEK",
    }])
    assert rows[0]["shares"] == 403


def test_derivation_keeps_precision_when_not_a_whole_number():
    rows, _ = pp._normalise_rows([{
        "name": "Frac", "ticker": "F", "shares": None,
        "last_price_native": 100.0, "market_value_sek": 1050, "currency": "SEK",
    }])
    assert rows[0]["shares"] == pytest.approx(10.5)


def test_derivation_is_skipped_for_a_foreign_currency_row():
    """value is SEK and price is native — dividing them would be nonsense."""
    rows, _ = pp._normalise_rows([{
        "name": "Nvidia", "ticker": "NVDA", "shares": None,
        "last_price_native": 180.0, "market_value_sek": 41148, "currency": "USD",
    }])
    assert rows[0]["shares"] is None


def test_a_read_share_count_is_not_overwritten_by_derivation():
    rows, _ = pp._normalise_rows([{
        "name": "Volvo B", "ticker": "VOLV-B", "shares": 100,
        "last_price_native": 291.5, "market_value_sek": 29150, "currency": "SEK",
    }])
    assert rows[0]["shares"] == 100
    assert rows[0]["shares_basis"] == "read"


def test_derivation_rescues_the_column_confusion_case():
    """Both fixes together: the bogus count is cleared, then rebuilt correctly."""
    rows, warnings = pp._normalise_rows([{
        "name": "Evolution", "ticker": "EVO",
        "shares": 109783,                      # the value, misread as a count
        "market_value_sek": 109783, "last_price_native": 736.8, "currency": "SEK",
    }])
    assert rows[0]["shares"] == 149
    assert rows[0]["shares_basis"] == "derived"
    assert any("read as a value, not a count" in w for w in warnings)
    # Cost comes back as value ÷ derived shares, i.e. the price it started from
    # (to rounding). This screen shows no purchase price, so cost basis is the
    # market price and P&L necessarily starts at zero.
    assert rows[0]["avg_cost_sek"] == pytest.approx(736.8, rel=1e-4)


# ── Benchmark labelling ───────────────────────────────────────────────────────
#
# The chart and KPI strip said "OMXSPI" for every book regardless of its actual
# benchmark, so a sleeve tracking URTH (MSCI World) claimed to be measured
# against the Swedish all-share.

def test_benchmark_label_strips_the_index_caret():
    from fundmgr.reporting.dashboard import benchmark_label
    assert benchmark_label("^OMXSPI") == "OMXSPI"
    assert benchmark_label("URTH") == "URTH"
    assert benchmark_label("  ^omxsgi ") == "OMXSGI"
    assert benchmark_label(None) == "Benchmark"
    assert benchmark_label("") == "Benchmark"


def test_chart_series_uses_the_given_label():
    from datetime import datetime
    from fundmgr.reporting.dashboard import nav_chart_json
    from fundmgr.state.models import NavPoint

    history = [NavPoint(date="2026-08-01", portfolio_nav_sek=100.0,
                        benchmark_value=10.0, cash_sek=0.0),
               NavPoint(date="2026-08-02", portfolio_nav_sek=110.0,
                        benchmark_value=11.0, cash_sek=0.0)]
    chart = json.loads(nav_chart_json(history, "URTH"))
    names = {s["name"] for s in chart["data"]}
    assert "URTH" in names and "OMXSPI" not in names
    assert datetime  # keep the import meaningful for the NavPoint construction


def test_sleeve_dashboard_shows_its_own_benchmark(client):
    rows = [{"ticker": "VOLV-B.ST", "name": "Volvo B", "shares": "100",
             "avg_cost_sek": "300"}]
    client.post("/live/create-from-photo", data={
        "name": "Bench Sleeve", "holdings_json": json.dumps(rows),
        "cash_sek": "0", "benchmark": "^OMXSPI",
    }, follow_redirects=False)

    body = client.get("/live/bench-sleeve").text
    assert "Portfolio vs OMXSPI" in body


def test_a_urth_sleeve_does_not_claim_omxspi(client):
    rows = [{"ticker": "VOLV-B.ST", "name": "Volvo B", "shares": "100",
             "avg_cost_sek": "300"}]
    client.post("/live/create-from-photo", data={
        "name": "World Sleeve", "holdings_json": json.dumps(rows),
        "cash_sek": "0", "benchmark": "URTH",
    }, follow_redirects=False)

    body = client.get("/live/world-sleeve").text
    assert "Portfolio vs URTH" in body
    assert "OMXSPI" not in body


# ── Regression: the Montrose import of 2026-08-13 ─────────────────────────────
#
# 15 rows, of which the first 7 were read perfectly and the rest degraded:
# Verisure booked at 8,000 SEK instead of 86,667 (its EUR price multiplied by
# the share count and labelled SEK), three share counts copied from adjacent
# rows, and four values rounded to exact thousands. The book came to 879,261
# against the screen's own 950,138 and nothing complained, because the only
# reconciliation compared a cost sum against a market total at a 0.5–1.5×
# tolerance. Each check below is anchored to what that screenshot actually
# contained.

def _montrose_rows():
    """The extraction as it came back, with the broker's true figures alongside."""
    return [
        # name,            shares,  avg_cost_sek, currency, avg_cost_native
        ("Evolution",        107.0,  667.0467, "SEK", 667.04),
        ("AstraZeneca",       65.0, 1555.6923, "SEK", 1555.69),
        ("BeammWave B",     2193.0,   13.2225, "SEK", 13.22),
        ("Systemair",       1316.0,   82.4962, "SEK", 82.49),
        ("Bonesupport",      172.0,  227.8256, "SEK", 227.82),
        ("Upsales",         1705.0,   27.7994, "SEK", 27.79),
        ("Vitec B",          451.0,  245.2794, "SEK", 245.27),
        ("Cheffelo",         272.0,  117.6471, "SEK", 117.79),
        ("Assa Abloy B",     272.0,  360.2941, "SEK", 360.86),
        ("Verisure",         752.0,   10.6383, "EUR", 10.47),
        ("Mildef",           272.0,  224.2647, "SEK", 225.22),
        ("Nordea",           254.0,  181.1024, "SEK", 182.71),
        ("Hacksaw",          743.0,   76.7160, "SEK", 76.94),
        ("Balder B",         506.0,   59.2885, "SEK", 53.11),
        ("Dynavox",          506.0,   79.0514, "SEK", 77.82),
    ]


def _as_holdings(rows):
    return [{"name": n, "shares": s, "avg_cost_sek": c, "currency": cur,
             "avg_cost_native": nat} for n, s, c, cur, nat in rows]


def test_the_montrose_shortfall_is_caught_against_the_stated_cost_total():
    """879,261 read against the screen's own 950,138 — 7.5% short, and the old
    0.5-1.5x tolerance let it through as healthy."""
    out = pp._sanity_check(_as_holdings(_montrose_rows()), 946_456, 950_138)
    reconcile = [w for w in out if "of purchase value" in w]
    assert len(reconcile) == 1
    assert "-70,877" in reconcile[0]
    assert "-7.5%" in reconcile[0]


def test_a_foreign_row_priced_in_its_own_currency_is_caught():
    """Verisure: 752 x 10.47 EUR booked as 8,000 SEK. For a EUR row the SEK cost
    and the native price differ by the FX rate — equal means unconverted."""
    out = pp._sanity_check(_as_holdings(_montrose_rows()), None, 950_138)
    hits = [w for w in out if w.startswith("Verisure")]
    assert len(hits) == 1
    assert "looks like EUR, not SEK" in hits[0]


def test_a_sek_row_is_not_flagged_by_the_currency_check():
    """The native price and the SEK cost are *supposed* to match here."""
    holdings = [{"name": "Evolution", "shares": 107.0, "avg_cost_sek": 667.0467,
                 "currency": "SEK", "avg_cost_native": 667.04}]
    assert not [w for w in pp._sanity_check(holdings, None, 71_374) if "looks like" in w]


def test_repeated_share_counts_are_reported():
    """272 on three rows and 506 on two — each copied off a neighbour."""
    out = pp._sanity_check(_as_holdings(_montrose_rows()), None, 950_138)
    repeats = [w for w in out if "share the same count" in w]
    assert any("272" in w and "Cheffelo" in w and "Mildef" in w for w in repeats)
    assert any("506" in w and "Balder B" in w and "Dynavox" in w for w in repeats)


def test_values_rounded_to_exact_thousands_are_reported():
    out = pp._sanity_check(_as_holdings(_montrose_rows()), None, 950_138)
    assert any("exact round thousand" in w for w in out)


def test_a_clean_book_passes_every_check():
    """The guards must stay quiet on a correctly-read portfolio, or they are
    noise and get ignored — which is worse than not having them."""
    holdings = [
        {"name": "Evolution", "shares": 107.0, "avg_cost_sek": 667.0467,
         "currency": "SEK", "avg_cost_native": 667.04},
        {"name": "AstraZeneca", "shares": 65.0, "avg_cost_sek": 1555.6923,
         "currency": "SEK", "avg_cost_native": 1555.69},
        {"name": "Verisure", "shares": 752.0, "avg_cost_sek": 115.2487,
         "currency": "EUR", "avg_cost_native": 10.47},
    ]
    total = sum(h["shares"] * h["avg_cost_sek"] for h in holdings)
    assert pp._sanity_check(holdings, None, total) == []
