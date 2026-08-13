"""
Balance-sheet and income-statement derived metrics.

`.info` — the payload financedata maps — is a normalised snapshot: valuation
multiples, headline margins, growth rates. It carries nothing about capital
structure, so a criterion phrased around leverage, solvency or interest cover
has no field behind it at all.

Those figures are not missing. yfinance serves the full statements
(`.balance_sheet`, `.income_stmt`, `.cashflow`) from the same free source; they
are simply never queried. This module asks for them and derives the ratios the
criteria are actually written in, merging the results into the fundamentals
cache so everything downstream — the evidence pack, the snapshots, the
criterion analyser, the numeric rules — picks them up with no change.

Two rules throughout:

  • **Never guess a label.** yfinance's statement row names drift between
    versions and differ by filer. Each metric names every spelling it accepts
    and yields None when it finds none of them, because a metric that silently
    computes from the wrong row is worse than one that reports nothing.
  • **Never fabricate a ratio.** A zero or missing denominator returns None,
    not infinity — an interest coverage of 1e18 because interest expense
    rounded to zero would sail through any "coverage ≥ 2.5" gate.

Reconciled against Balder's own Q2 reporting. A derived figure is not
automatically the company's figure, and both the size and the *direction* of
the difference decide whether it can back a criterion:

  equity/assets     37.0% vs 37.0% reported — exact, offered. Reaching it took
                    both fixes below: the gross equity row and the interim
                    sheet. Either one alone lands ~3pp out.
  net debt/assets   52.8% vs 50.4% reported — offered, with a known +2.4pp
                    bias. yfinance's "Total Debt" appears to fold in lease
                    liabilities the company keeps out of its own net debt.
                    Reading leverage *high* is the safe direction for a gate
                    written "below X" — it kills early, never late — but a
                    criterion set within ~3pp of the current ratio will trip on
                    the definition rather than on the business.
  net debt/EBITDA   10.4x vs 12.8x reported — derived, NOT offered.
                    Income-statement EBITDA for a property company includes
                    revaluation gains the company excludes from its leverage
                    metric. This one flatters, which is the wrong direction for
                    a leverage gate, and no choice of row label fixes it.

An earlier version of this note recorded net debt/assets as 50.6% vs 50.4% and
called it near-exact. That agreement was an artefact: an annual balance sheet
compared against an interim reported ratio. Once both sides came from the same
quarter the real gap appeared.
"""
from __future__ import annotations

import math

# Derived metric → (label, is_fraction). Mirrors the shape evidence._FUND_FIELDS
# uses, so promoting one to an offered kill/add metric is a copy of this row.
STATEMENT_METRICS: dict[str, tuple[str, bool]] = {
    "equity_to_assets":    ("Equity/assets", True),
    "net_debt_to_assets":  ("Net debt/assets", True),
    "net_debt_to_ebitda":  ("Net debt/EBITDA", False),
    "interest_coverage":   ("Interest coverage", False),
    "cost_to_income":      ("Cost/income", True),
    "fcf_to_net_income":   ("FCF/net income", True),
    "cash_runway_months":  ("Cash runway (months)", False),
}

# Statement row labels, most standard first. yfinance renames rows between
# releases and filers differ, so every lookup accepts a list.
_ROWS = {
    "total_assets":     ("Total Assets",),
    # Gross first: a solvency ratio is reported on total equity, and excluding
    # minority interest understates it (Balder: 34.0% vs 38.5% on the same
    # balance sheet). Narrower rows are the fallback, not the preference.
    "total_equity":     ("Total Equity Gross Minority Interest", "Stockholders Equity",
                         "Total Stockholder Equity", "Common Stock Equity"),
    "total_debt":       ("Total Debt",),
    "cash":             ("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments",
                         "Cash And Cash Equivalents At Carrying Value"),
    "ebit":             ("EBIT", "Operating Income", "Total Operating Income As Reported"),
    "ebitda":           ("EBITDA", "Normalized EBITDA"),
    "interest_expense": ("Interest Expense", "Interest Expense Non Operating"),
    "revenue":          ("Total Revenue", "Operating Revenue"),
    "operating_expense": ("Operating Expense", "Total Expenses"),
    "net_income":       ("Net Income", "Net Income Common Stockholders",
                         "Net Income From Continuing Operation Net Minority Interest"),
    "free_cash_flow":   ("Free Cash Flow",),
    "operating_cash_flow": ("Operating Cash Flow", "Cash Flow From Continuing Operating Activities"),
}


def _num(value) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) or math.isinf(out) else out


def _pick(frame, key: str) -> float | None:
    """Most recent value of a statement row, trying each accepted label."""
    if frame is None or getattr(frame, "empty", True):
        return None
    for label in _ROWS.get(key, ()):
        try:
            if label not in frame.index:
                continue
            series = frame.loc[label]
        except Exception:
            continue
        try:
            series = series.dropna()
        except Exception:
            return _num(series)
        if len(series):
            # Statement columns are period-end dates, newest first.
            return _num(series.iloc[0])
    return None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Guarded division — a zero denominator is unknown, never infinite."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    out = numerator / denominator
    return None if math.isnan(out) or math.isinf(out) else out


def derive(ticker: str, *, frames: dict | None = None) -> dict[str, float | None]:
    """Derived statement metrics for one ticker. Values may be None.

    `frames` lets tests supply statements directly; normally they are fetched.
    """
    if frames is None:
        frames = fetch_frames(ticker)
    # Balance-sheet items are stocks: take the freshest, which is the interim
    # if there is one — an annual sheet is up to a year stale, and a criterion
    # written against the latest report would be judged on last year's balance.
    # Flows (income, cash flow) stay annual: one quarter of EBITDA would make
    # net debt/EBITDA four times too high.
    quarterly_balance = frames.get("quarterly_balance")
    balance = quarterly_balance if quarterly_balance is not None else frames.get("balance")
    income = frames.get("income")
    cashflow = frames.get("cashflow")
    quarterly_cashflow = frames.get("quarterly_cashflow")

    assets = _pick(balance, "total_assets")
    equity = _pick(balance, "total_equity")
    debt = _pick(balance, "total_debt")
    cash = _pick(balance, "cash")
    ebit = _pick(income, "ebit")
    ebitda = _pick(income, "ebitda")
    interest = _pick(income, "interest_expense")
    revenue = _pick(income, "revenue")
    opex = _pick(income, "operating_expense")
    net_income = _pick(income, "net_income")
    fcf = _pick(cashflow, "free_cash_flow")

    net_debt = (debt - cash) if (debt is not None and cash is not None) else None
    # `a or b` on a DataFrame raises — its truth value is ambiguous — so the
    # fallback to annual figures has to be an explicit None check.
    runway_frame = quarterly_cashflow if quarterly_cashflow is not None else cashflow

    return {
        "equity_to_assets":   _ratio(equity, assets),
        "net_debt_to_assets": _ratio(net_debt, assets),
        "net_debt_to_ebitda": _ratio(net_debt, ebitda),
        # Interest expense is reported positive or negative depending on filer.
        "interest_coverage":  _ratio(ebit, abs(interest) if interest is not None else None),
        "cost_to_income":     _ratio(opex, revenue),
        "fcf_to_net_income":  _ratio(fcf, net_income),
        "cash_runway_months": _cash_runway(cash, runway_frame),
    }


def _cash_runway(cash: float | None, cashflow) -> float | None:
    """Months of cash left at the current burn. None for anything cash-generative.

    A runway is only meaningful while operations consume cash; for a profitable
    company the answer is not "a very large number of months", it is "not
    applicable", and returning the former invites a nonsense comparison.
    """
    if not cash or cash <= 0:
        return None
    operating = _pick(cashflow, "operating_cash_flow")
    if operating is None or operating >= 0:
        return None
    # Statement periods are quarterly or annual; infer which from the column span.
    months = 3.0
    try:
        columns = list(getattr(cashflow, "columns", []))
        if len(columns) >= 2:
            span = abs((columns[0] - columns[1]).days)
            months = 12.0 if span > 200 else 3.0
    except Exception:
        pass
    burn_per_month = abs(operating) / months
    return _ratio(cash, burn_per_month)


def fetch_frames(ticker: str) -> dict:
    """The statement frames for one ticker. Missing ones come back as None."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
    except Exception:
        return {}

    def _get(name):
        try:
            return getattr(t, name)
        except Exception:
            return None

    return {
        "balance": _get("balance_sheet"),
        "quarterly_balance": _get("quarterly_balance_sheet"),
        "income": _get("income_stmt"),
        "cashflow": _get("cashflow"),
        "quarterly_cashflow": _get("quarterly_cashflow"),
    }


def refresh(store, tickers: list[str], ttl_days: int = 7,
            max_workers: int = 8) -> int:
    """Derive statement metrics and merge them into the fundamentals cache.

    Merges rather than replaces: financedata owns the `.info` keys and rewrites
    the whole blob on its own refresh, so these are folded in afterwards on the
    same schedule. Returns the number of tickers updated.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    stale = store.get_stale_fundamentals_tickers(list(tickers), ttl_days=0) \
        if ttl_days == 0 else list(tickers)
    if not stale:
        return 0

    updated = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(derive, t): t for t in stale}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                metrics = future.result()
            except Exception:
                continue
            metrics = {k: v for k, v in metrics.items() if v is not None}
            if not metrics:
                continue
            existing = store.get_fundamentals(ticker) or {}
            existing.update(metrics)
            store.save_fundamentals(ticker, existing)
            updated += 1
    return updated
