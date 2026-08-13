"""Balance-sheet derived metrics — the capital structure `.info` never carries."""
import pandas as pd
import pytest

from fundmgr.data import statements
from fundmgr.state.store import Store

# Balder-shaped: a leveraged property company, the holding whose entire
# criterion set lives in these rows.
BALANCE = pd.DataFrame(
    {pd.Timestamp("2026-06-30"): {
        "Total Assets": 200_000.0,
        "Stockholders Equity": 74_000.0,
        "Total Debt": 105_000.0,
        "Cash And Cash Equivalents": 4_200.0,
    }}
)
INCOME = pd.DataFrame(
    {pd.Timestamp("2026-06-30"): {
        "EBIT": 9_000.0,
        "EBITDA": 8_500.0,
        "Interest Expense": 3_400.0,
        "Total Revenue": 12_000.0,
        "Operating Expense": 5_400.0,
        "Net Income": 4_000.0,
    }}
)
CASHFLOW = pd.DataFrame(
    {pd.Timestamp("2026-06-30"): {"Free Cash Flow": 3_600.0,
                                  "Operating Cash Flow": 5_000.0}}
)


def _frames(**over):
    base = {"balance": BALANCE, "income": INCOME, "cashflow": CASHFLOW,
            "quarterly_cashflow": None}
    base.update(over)
    return base


# ── The ratios ────────────────────────────────────────────────────────────────

def test_derives_capital_structure():
    m = statements.derive("BALD-B.ST", frames=_frames())
    assert m["equity_to_assets"] == pytest.approx(0.37)              # 74k/200k
    assert m["net_debt_to_assets"] == pytest.approx(0.504)           # (105k−4.2k)/200k
    assert m["net_debt_to_ebitda"] == pytest.approx(11.86, abs=0.01)
    assert m["interest_coverage"] == pytest.approx(2.65, abs=0.01)   # 9000/3400
    assert m["cost_to_income"] == pytest.approx(0.45)                # 5400/12000
    assert m["fcf_to_net_income"] == pytest.approx(0.9)              # 3600/4000


def test_interest_expense_sign_does_not_flip_coverage():
    """Filers report interest expense positive or negative; coverage is positive."""
    income = INCOME.copy()
    income.loc["Interest Expense"] = -3_400.0
    assert statements.derive("X", frames=_frames(income=income))["interest_coverage"] \
        == pytest.approx(2.65, abs=0.01)


# ── Never guess, never fabricate ──────────────────────────────────────────────

def test_alternative_row_labels_are_accepted():
    """yfinance renames rows between releases; each metric names its spellings."""
    balance = pd.DataFrame({pd.Timestamp("2026-06-30"): {
        "Total Assets": 200_000.0,
        "Total Stockholder Equity": 74_000.0,        # older label
        "Total Debt": 105_000.0,
        "Cash Cash Equivalents And Short Term Investments": 4_200.0,
    }})
    m = statements.derive("X", frames=_frames(balance=balance))
    assert m["equity_to_assets"] == pytest.approx(0.37)
    assert m["net_debt_to_assets"] == pytest.approx(0.504)


def test_an_unknown_label_yields_none_not_a_wrong_number():
    balance = pd.DataFrame({pd.Timestamp("2026-06-30"): {
        "Assets Grand Total": 200_000.0}})          # a spelling we do not accept
    m = statements.derive("X", frames=_frames(balance=balance))
    assert m["equity_to_assets"] is None
    assert m["net_debt_to_assets"] is None


def test_a_zero_denominator_is_unknown_not_infinite():
    """Coverage of 1e18 because interest rounded to zero would clear any gate."""
    income = INCOME.copy()
    income.loc["Interest Expense"] = 0.0
    income.loc["EBITDA"] = 0.0
    m = statements.derive("X", frames=_frames(income=income))
    assert m["interest_coverage"] is None
    assert m["net_debt_to_ebitda"] is None


def test_nan_values_are_dropped():
    balance = BALANCE.copy()
    balance.loc["Total Assets"] = float("nan")
    assert statements.derive("X", frames=_frames(balance=balance))["equity_to_assets"] is None


def test_missing_frames_yield_all_none():
    m = statements.derive("X", frames={"balance": None, "income": None,
                                       "cashflow": None, "quarterly_cashflow": None})
    assert set(m) == set(statements.STATEMENT_METRICS)
    assert all(v is None for v in m.values())


def test_empty_frames_are_survivable():
    empty = pd.DataFrame()
    m = statements.derive("X", frames=_frames(balance=empty, income=empty, cashflow=empty))
    assert all(v is None for v in m.values())


def test_the_most_recent_period_is_used():
    balance = pd.DataFrame({
        pd.Timestamp("2026-06-30"): {"Total Assets": 200_000.0, "Stockholders Equity": 74_000.0},
        pd.Timestamp("2025-06-30"): {"Total Assets": 100_000.0, "Stockholders Equity": 50_000.0},
    })
    assert statements.derive("X", frames=_frames(balance=balance))["equity_to_assets"] \
        == pytest.approx(0.37)


# ── Cash runway ───────────────────────────────────────────────────────────────

def test_cash_runway_for_a_burning_company():
    """BeammWave-shaped: cash, no revenue, a quarterly burn."""
    quarterly = pd.DataFrame({
        pd.Timestamp("2026-06-30"): {"Operating Cash Flow": -24.0},
        pd.Timestamp("2026-03-31"): {"Operating Cash Flow": -22.0},
    })
    balance = pd.DataFrame({pd.Timestamp("2026-06-30"): {
        "Cash And Cash Equivalents": 144.6}})
    m = statements.derive("BEAMMW-B.ST",
                          frames=_frames(balance=balance, quarterly_cashflow=quarterly))
    assert m["cash_runway_months"] == pytest.approx(18.1, abs=0.2)   # 144.6 / (24/3)


def test_cash_runway_reads_an_annual_period_as_twelve_months():
    annual = pd.DataFrame({
        pd.Timestamp("2026-06-30"): {"Operating Cash Flow": -96.0},
        pd.Timestamp("2025-06-30"): {"Operating Cash Flow": -80.0},
    })
    balance = pd.DataFrame({pd.Timestamp("2026-06-30"): {
        "Cash And Cash Equivalents": 144.6}})
    m = statements.derive("X", frames=_frames(balance=balance, quarterly_cashflow=annual))
    assert m["cash_runway_months"] == pytest.approx(18.1, abs=0.2)   # 144.6 / (96/12)


def test_a_cash_generative_company_has_no_runway():
    """Not 'a very large number of months' — the question doesn't apply."""
    assert statements.derive("X", frames=_frames())["cash_runway_months"] is None


# ── Merging into the cache ────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "book.db")
    s.initialise(100_000)
    return s


def test_refresh_merges_and_does_not_clobber_info_keys(monkeypatch, store):
    """financedata owns the .info keys and rewrites the blob; these fold in."""
    store.save_fundamentals("BALD-B.ST", {"revenue_growth": 0.04, "pe_ratio": 12.0})
    monkeypatch.setattr(statements, "fetch_frames", lambda t: _frames())

    assert statements.refresh(store, ["BALD-B.ST"]) == 1
    cached = store.get_fundamentals("BALD-B.ST")
    assert cached["revenue_growth"] == 0.04           # untouched
    assert cached["equity_to_assets"] == pytest.approx(0.37)


def test_refresh_skips_a_ticker_with_no_usable_statements(monkeypatch, store):
    monkeypatch.setattr(statements, "fetch_frames", lambda t: {})
    assert statements.refresh(store, ["NOPE"]) == 0
    assert store.get_fundamentals("NOPE") is None


def test_refresh_survives_a_failing_ticker(monkeypatch, store):
    def flaky(ticker):
        if ticker == "BAD":
            raise RuntimeError("statement fetch exploded")
        return _frames()
    monkeypatch.setattr(statements, "fetch_frames", flaky)
    assert statements.refresh(store, ["BAD", "BALD-B.ST"]) == 1


def test_fetch_frames_survives_a_dead_feed(monkeypatch):
    import yfinance as yf

    def boom(ticker):
        raise RuntimeError("network down")
    monkeypatch.setattr(yf, "Ticker", boom)
    assert statements.fetch_frames("X") == {}


# ── Wiring ────────────────────────────────────────────────────────────────────

def test_only_reconciled_metrics_are_offered():
    """Offered only after reconciling against the company's own reporting.

    equity/assets reads ~3pp low (the equity row excludes minority and hybrid
    capital) and net debt/EBITDA reads 24% low (income-statement EBITDA carries
    revaluation gains the company excludes) — both would silently misjudge a
    criterion written to the company's definition."""
    from fundmgr.evidence import FUND_FIELD_META
    offered = set(statements.STATEMENT_METRICS) & set(FUND_FIELD_META)
    assert offered == {"net_debt_to_assets", "interest_coverage",
                       "cost_to_income", "fcf_to_net_income"}
    assert "equity_to_assets" not in FUND_FIELD_META
    assert "net_debt_to_ebitda" not in FUND_FIELD_META


def test_every_statement_metric_declares_its_unit():
    for key, (label, is_fraction) in statements.STATEMENT_METRICS.items():
        assert label and isinstance(is_fraction, bool), key


# ── Period selection: stocks fresh, flows annual ──────────────────────────────

QUARTERLY_BALANCE = pd.DataFrame({pd.Timestamp("2026-06-30"): {
    "Total Assets": 276_292.0,
    "Total Equity Gross Minority Interest": 106_491.0,
    "Stockholders Equity": 93_852.0,
    "Total Debt": 155_536.0,
    "Cash And Cash Equivalents": 5_394.0,
}})


def test_total_equity_prefers_the_gross_row():
    """Excluding minority interest understates solvency: 34.0% vs 38.5%."""
    m = statements.derive("BALD-B.ST", frames=_frames(balance=QUARTERLY_BALANCE))
    assert m["equity_to_assets"] == pytest.approx(0.385, abs=0.002)


def test_stockholders_equity_is_still_accepted_as_a_fallback():
    balance = QUARTERLY_BALANCE.drop(index=["Total Equity Gross Minority Interest"])
    m = statements.derive("X", frames=_frames(balance=balance))
    assert m["equity_to_assets"] == pytest.approx(0.340, abs=0.002)


def test_the_quarterly_balance_sheet_wins_over_the_annual():
    """An annual sheet is up to a year stale against an interim criterion."""
    annual = pd.DataFrame({pd.Timestamp("2025-12-31"): {
        "Total Assets": 100_000.0, "Total Equity Gross Minority Interest": 20_000.0}})
    m = statements.derive("X", frames={
        "balance": annual, "quarterly_balance": QUARTERLY_BALANCE,
        "income": INCOME, "cashflow": CASHFLOW, "quarterly_cashflow": None})
    assert m["equity_to_assets"] == pytest.approx(0.385, abs=0.002)   # not 0.20


def test_the_annual_balance_sheet_is_used_when_there_is_no_interim():
    m = statements.derive("X", frames={
        "balance": QUARTERLY_BALANCE, "quarterly_balance": None,
        "income": INCOME, "cashflow": CASHFLOW, "quarterly_cashflow": None})
    assert m["equity_to_assets"] == pytest.approx(0.385, abs=0.002)


def test_flows_stay_annual_so_leverage_is_not_inflated():
    """A single quarter of EBITDA would make net debt/EBITDA four times too high."""
    m = statements.derive("X", frames={
        "balance": None, "quarterly_balance": QUARTERLY_BALANCE,
        "income": INCOME, "cashflow": CASHFLOW, "quarterly_cashflow": None})
    net_debt = 155_536.0 - 5_394.0
    assert m["net_debt_to_ebitda"] == pytest.approx(net_debt / 8_500.0, rel=1e-3)
