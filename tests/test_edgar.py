"""SEC EDGAR as a source for figures no market feed carries.

The parsing is what these cover, because the fetching is three lines and the
parsing is where a plausible-but-wrong number comes from: a concept's fact list
mixes quarterly, year-to-date and annual durations under one name, and reading
the wrong one produces growth that never happened.
"""
import pytest

from fundmgr import evidence
from fundmgr.data import edgar
from fundmgr.state.store import Store

# Shaped like the real payload: quarterly facts, a nine-month year-to-date fact
# and a full-year fact under the same concept, plus a restatement.
FACTS = {
    "cik": 1045810,
    "entityName": "NVIDIA CORP",
    "facts": {"us-gaap": {"Revenues": {"label": "Revenues", "units": {"USD": [
        {"start": "2025-08-01", "end": "2025-10-31", "val": 35_000, "form": "10-Q",
         "filed": "2025-11-20"},
        {"start": "2025-11-01", "end": "2026-01-31", "val": 39_000, "form": "10-K",
         "filed": "2026-02-26"},
        {"start": "2026-02-01", "end": "2026-04-30", "val": 44_000, "form": "10-Q",
         "filed": "2026-05-28"},
        # Same quarter, restated in a later filing — the correction wins.
        {"start": "2026-02-01", "end": "2026-04-30", "val": 44_500, "form": "10-Q/A",
         "filed": "2026-08-14"},
        # Nine-month year-to-date: same concept, must never read as a quarter.
        {"start": "2025-08-01", "end": "2026-04-30", "val": 118_000, "form": "10-Q",
         "filed": "2026-05-28"},
        # Full year.
        {"start": "2025-02-01", "end": "2026-01-31", "val": 130_000, "form": "10-K",
         "filed": "2026-02-26"},
        # An instant (balance-sheet style) — no duration at all.
        {"end": "2026-04-30", "val": 9_999, "form": "10-Q", "filed": "2026-05-28"},
    ]}}}},
}

TICKER_MAP = {"0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
              "1": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "book.db")
    s.initialise(100_000)
    return s


# ── Picking the right facts ───────────────────────────────────────────────────

def test_only_quarterly_durations_are_kept():
    """Nine-month and full-year facts share the concept. Reading 118,000 as a
    quarter would show growth that never happened."""
    series = edgar.quarterly_series(FACTS, "Revenues")["values"]
    assert set(series) == {"2025-10-31", "2026-01-31", "2026-04-30"}
    assert 118_000 not in series.values()
    assert 130_000 not in series.values()


def test_an_instant_fact_is_not_a_quarter():
    assert 9_999 not in edgar.quarterly_series(FACTS, "Revenues")["values"].values()


def test_a_restatement_supersedes_the_original():
    """A later filing for the same period is the company correcting itself."""
    assert edgar.quarterly_series(FACTS, "Revenues")["values"]["2026-04-30"] == 44_500


def test_values_are_keyed_by_the_period_they_cover():
    """Not the date fetched — that is what makes the quarters machinery work."""
    series = edgar.quarterly_series(FACTS, "Revenues")["values"]
    assert series["2025-10-31"] == 35_000
    assert series["2026-01-31"] == 39_000


def test_an_unknown_concept_is_empty_not_an_error():
    assert edgar.quarterly_series(FACTS, "NoSuchConcept")["values"] == {}
    assert edgar.quarterly_series({}, "Revenues")["values"] == {}


def test_the_unit_with_the_most_facts_wins():
    """A filer reporting in another currency must still work."""
    facts = {"facts": {"us-gaap": {"Revenues": {"units": {
        "USD": [{"start": "2026-02-01", "end": "2026-04-30", "val": 1, "filed": "x"}],
        "SEK": [{"start": "2026-02-01", "end": "2026-04-30", "val": 11, "filed": "x"},
                {"start": "2025-11-01", "end": "2026-01-31", "val": 10, "filed": "x"}],
    }}}}}
    out = edgar.quarterly_series(facts, "Revenues")
    assert out["values"]["2026-04-30"] == 11
    assert out["unit"] == "SEK"


def test_malformed_facts_are_skipped_not_fatal():
    facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        "not a dict",
        {"start": "bad-date", "end": "2026-04-30", "val": 1, "filed": "x"},
        {"start": "2026-02-01", "end": "2026-04-30", "val": "not a number", "filed": "x"},
        {"start": "2026-02-01", "end": "2026-04-30", "val": 42, "filed": "x"},
    ]}}}}}
    assert edgar.quarterly_series(facts, "Revenues")["values"] == {"2026-04-30": 42.0}


# ── Ticker → CIK ──────────────────────────────────────────────────────────────

def test_ticker_map_parses_the_sec_shape():
    assert edgar._parse_ticker_map(TICKER_MAP) == {"NVDA": 1045810, "AAPL": 320193}


def test_a_suffixed_ticker_is_not_looked_up(monkeypatch):
    """VOLV-B.ST is a Stockholm listing; EDGAR knows plain US symbols, and
    asking would be a wasted request with a misleading answer."""
    monkeypatch.setattr(edgar, "_ticker_cik", {"NVDA": 1045810})
    assert edgar.cik_for("VOLV-B.ST") is None
    assert edgar.cik_for("nvda") == 1045810


def test_an_unlisted_ticker_has_no_cik(monkeypatch):
    monkeypatch.setattr(edgar, "_ticker_cik", {"NVDA": 1045810})
    assert edgar.cik_for("NOSUCH") is None


# ── Only mapped custom metrics are filled ─────────────────────────────────────

def test_nothing_is_read_without_a_mapping(store):
    evidence.define_custom_metric(store, "organic_growth", "Organic growth", "%")
    assert edgar.read_ticker(store, "NVDA", FACTS) == {}


def test_a_mapped_metric_is_filled_from_the_filings(store):
    evidence.define_custom_metric(store, "segment_revenue", "Segment revenue",
                                  "USD", edgar="Revenues")
    out = edgar.read_ticker(store, "NVDA", FACTS)
    assert set(out) == {"segment_revenue"}
    assert out["segment_revenue"]["values"]["2026-04-30"] == 44_500
    assert out["segment_revenue"]["unit"] == "USD"


def test_a_built_in_field_cannot_be_filled_from_edgar(store):
    """Two providers behind one key would give a series that changes definition
    halfway — worse than one that is merely incomplete."""
    with pytest.raises(ValueError, match="already a built-in"):
        evidence.define_custom_metric(store, "revenue_growth", edgar="Revenues")


def test_a_company_that_does_not_file_yields_nothing(store):
    evidence.define_custom_metric(store, "segment_revenue", unit="SEK", edgar="Revenues")
    assert edgar.read_ticker(store, "BALD-B.ST", {}) == {}


# ── The SEC's User-Agent requirement is stated, not swallowed ─────────────────

def test_a_missing_user_agent_says_so(monkeypatch):
    monkeypatch.delenv("FUND_SEC_USER_AGENT", raising=False)
    with pytest.raises(edgar.EdgarError, match="FUND_SEC_USER_AGENT"):
        edgar._user_agent()


def test_a_configured_user_agent_is_used(monkeypatch):
    monkeypatch.setenv("FUND_SEC_USER_AGENT", "Test test@example.com")
    assert edgar._user_agent() == "Test test@example.com"


# ── End to end into the book ──────────────────────────────────────────────────

def test_filled_quarters_feed_a_multi_quarter_criterion(store):
    """The point of keying by period: a 'two consecutive quarters' rule works
    on filings the moment they are pulled, with no waiting."""
    evidence.define_custom_metric(store, "segment_revenue", "Segment revenue",
                                  "USD", edgar="Revenues")
    series = edgar.read_ticker(store, "NVDA", FACTS)["segment_revenue"]["values"]
    for period, value in series.items():
        evidence.record_reported(store, "NVDA", {"segment_revenue": value}, period)

    history = evidence.metric_history(store, "NVDA", "segment_revenue")
    assert [r["period"] for r in history] == ["2026-04-30", "2026-01-31", "2025-10-31"]

    run = evidence.sustained_breach(store, "NVDA", "segment_revenue", "above", 30_000, 3)
    assert run["hit"] is True and run["periods_seen"] == 3


# ── The filing's unit is not the book's guess ────────────────────────────────

def test_the_filing_unit_travels_with_the_figures():
    """NVIDIA files in USD. Dropping that let its revenue display under a metric
    declared in SEK — the number right, the label a lie."""
    assert edgar.quarterly_series(FACTS, "Revenues")["unit"] == "USD"


def test_a_currency_code_is_a_valid_unit(store):
    for unit in ("USD", "SEK", "EUR", "%", "x", ""):
        evidence.define_custom_metric(store, f"m_{unit or 'none'}", unit=unit)
    # Keys normalise to lower case; the unit is stored as given.
    assert evidence.field_meta(store)["m_usd"]["unit"] == "USD"
    assert evidence.field_meta(store)["m_sek"]["unit"] == "SEK"


def test_a_nonsense_unit_is_still_refused(store):
    for bad in ("dollars", "US$", "furlongs", "usd"):
        with pytest.raises(ValueError, match="Unit must be"):
            evidence.define_custom_metric(store, "x_metric", unit=bad)


def test_the_cli_accepts_every_unit_the_registry_does(tmp_path, monkeypatch):
    """One rule, one validator. A second copy in the click option is how the
    error message came to recommend a command the CLI then rejected."""
    from click.testing import CliRunner

    from fundmgr import paper
    book = Store(tmp_path / "unit.db")
    book.initialise(1000)
    monkeypatch.setattr(paper, "open_portfolio", lambda slug: ({"name": slug}, book))

    from fundmgr.cli import cli
    runner = CliRunner()
    for unit in ("USD", "SEK", "EUR", "%", "x"):
        result = runner.invoke(cli, ["paper-metric", "book", f"m_{unit}",
                                     "--unit", unit])
        assert result.exit_code == 0, f"{unit}: {result.output}"
    assert evidence.field_meta(book)["m_usd"]["unit"] == "USD"

    bad = runner.invoke(cli, ["paper-metric", "book", "nope", "--unit", "dollars"])
    assert bad.exit_code != 0 and "Unit must be" in bad.output


def test_only_the_quarters_the_series_can_hold_are_written(tmp_path, monkeypatch):
    """64 quarters pulled, 12 retained. Writing all 64 made the count claim
    sixteen years of history that the next append would trim away."""
    from click.testing import CliRunner

    from fundmgr import paper
    book = Store(tmp_path / "cap.db")
    book.initialise(1000)
    monkeypatch.setattr(paper, "open_portfolio", lambda slug: ({"name": slug}, book))
    evidence.define_custom_metric(book, "segment_revenue", "Segment revenue",
                                  "USD", edgar="Revenues")

    # Twenty quarters of filings — more than the series keeps.
    quarters = [(f"20{y:02d}-{m:02d}-30", 1000 + i)
                for i, (y, m) in enumerate([(20 + n // 4, [3, 6, 9, 12][n % 4])
                                            for n in range(20)])]
    facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        {"start": f"{end[:4]}-{int(end[5:7]) - 2:02d}-01", "end": end,
         "val": val, "filed": end}
        for end, val in quarters]}}}}}
    monkeypatch.setattr(edgar, "fetch_facts", lambda t: facts)

    from fundmgr.cli import cli
    out = CliRunner().invoke(cli, ["paper-edgar", "book", "NVDA", "--apply"])
    assert out.exit_code == 0, out.output
    assert "older quarter(s) skipped" in out.output

    history = evidence.metric_history(book, "NVDA", "segment_revenue")
    assert len(history) <= evidence.MAX_SNAPSHOTS


def test_the_metric_listing_names_a_metric_this_book_has(tmp_path, monkeypatch):
    """A hardcoded example invites the 'Unknown metric' error from a line the
    tool itself printed."""
    from click.testing import CliRunner

    from fundmgr import paper
    book = Store(tmp_path / "hint.db")
    book.initialise(1000)
    monkeypatch.setattr(paper, "open_portfolio", lambda slug: ({"name": slug}, book))
    evidence.define_custom_metric(book, "book_to_bill", "Book-to-bill", "x")

    from fundmgr.cli import cli
    out = CliRunner().invoke(cli, ["paper-metric", "book"])
    assert "--set book_to_bill=" in out.output
    assert "organic_growth" not in out.output
