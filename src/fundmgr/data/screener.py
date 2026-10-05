from __future__ import annotations

from fundmgr import regions, styles
from fundmgr.data.prices import TickerFeatures


def _score(feat: TickerFeatures) -> float:
    score = 0.0

    # ── Momentum (price-based) ────────────────────────────────────────────────
    if feat.return_1d_pct is not None:
        score += feat.return_1d_pct * 0.10
    if feat.return_5d_pct is not None:
        score += feat.return_5d_pct * 0.20
    if feat.return_20d_pct is not None:
        score += feat.return_20d_pct * 0.30
    if feat.return_60d_pct is not None:
        score += feat.return_60d_pct * 0.15

    # ── Trend alignment ───────────────────────────────────────────────────────
    if feat.above_ma50 is True:
        score += 2.0
    elif feat.above_ma50 is False:
        score -= 1.0
    if feat.above_ma200 is True:
        score += 1.5

    # ── RSI — exclude overbought, reward room to run ──────────────────────────
    if feat.rsi_14 is not None:
        if feat.rsi_14 > 75:
            score -= 12.0
        elif feat.rsi_14 > 70:
            score -= 5.0
        elif feat.rsi_14 < 30:
            score += 4.0
        elif feat.rsi_14 < 40:
            score += 1.5

    # ── Price positioning vs 52w high ─────────────────────────────────────────
    # Stocks near their 52w high are in strong uptrends; very far below = distressed
    if feat.pct_from_52w_high is not None:
        pct = feat.pct_from_52w_high  # negative value (0 = at high, -50 = half of high)
        if pct >= -10:
            score += 2.0   # near 52w high — strong trend
        elif pct >= -20:
            score += 0.5
        elif pct <= -50:
            score -= 2.0   # deep in a hole — avoid unless clear catalyst

    # ── Sentiment ─────────────────────────────────────────────────────────────
    if feat.sentiment_label == "positive" and feat.sentiment_score is not None:
        score += feat.sentiment_score * 5.0
    elif feat.sentiment_label == "negative" and feat.sentiment_score is not None:
        score -= feat.sentiment_score * 5.0

    # ── Fundamentals: growth ──────────────────────────────────────────────────
    # Accelerating revenue/earnings growth is a strong buy signal
    if feat.revenue_growth_pct is not None:
        if feat.revenue_growth_pct > 20:
            score += 3.0
        elif feat.revenue_growth_pct > 10:
            score += 1.5
        elif feat.revenue_growth_pct < -10:
            score -= 2.0
    if feat.earnings_growth_pct is not None:
        if feat.earnings_growth_pct > 25:
            score += 2.5
        elif feat.earnings_growth_pct > 10:
            score += 1.0
        elif feat.earnings_growth_pct < -20:
            score -= 2.0

    # ── Fundamentals: quality ─────────────────────────────────────────────────
    # High ROE + healthy margins = quality business worth paying up for
    if feat.roe_pct is not None:
        if feat.roe_pct > 20:
            score += 2.0
        elif feat.roe_pct > 10:
            score += 0.5
        elif feat.roe_pct < 0:
            score -= 2.5   # negative ROE = distressed
    if feat.profit_margin_pct is not None:
        if feat.profit_margin_pct > 15:
            score += 1.5
        elif feat.profit_margin_pct < 0:
            score -= 2.0   # unprofitable

    # ── Fundamentals: valuation ───────────────────────────────────────────────
    # Reward reasonable valuations; penalise extreme over-valuation relative to earnings
    if feat.ev_to_ebitda is not None:
        if feat.ev_to_ebitda < 10:
            score += 1.5   # cheap on EV/EBITDA
        elif feat.ev_to_ebitda > 30:
            score -= 1.0   # expensive, needs strong growth to justify
    if feat.forward_pe is not None:
        if feat.forward_pe < 12:
            score += 1.5   # value territory
        elif feat.forward_pe > 40:
            score -= 1.0   # priced for perfection

    # ── Fundamentals: balance sheet risk ──────────────────────────────────────
    if feat.debt_to_equity is not None:
        if feat.debt_to_equity > 200:
            score -= 2.0   # highly leveraged
        elif feat.debt_to_equity > 100:
            score -= 0.5

    # ── Analyst consensus ─────────────────────────────────────────────────────
    # Significant analyst upside with reasonable coverage = institutional backing
    if feat.analyst_target_pct is not None and feat.analyst_count is not None:
        if feat.analyst_count >= 5:
            if feat.analyst_target_pct > 20:
                score += 2.0   # strong consensus upside
            elif feat.analyst_target_pct > 10:
                score += 1.0
            elif feat.analyst_target_pct < -10:
                score -= 1.5   # consensus sell

    # ── Earnings proximity ────────────────────────────────────────────────────
    # Binary-event risk: avoid new buys right before earnings
    if feat.days_to_earnings is not None:
        if 0 <= feat.days_to_earnings <= 2:
            score -= 8.0   # imminent — high uncertainty, skip
        elif 0 <= feat.days_to_earnings <= 5:
            score -= 3.0   # this week — caution

    # ── Volume confirmation ───────────────────────────────────────────────────
    # Elevated relative volume signals institutional interest or news catalyst
    if feat.rel_volume is not None:
        if feat.rel_volume > 3.0:
            score += 2.5
        elif feat.rel_volume > 2.0:
            score += 1.0
        elif feat.rel_volume < 0.3:
            score -= 1.5   # extremely thin — liquidity risk

    # ── Stale data is almost never actionable ────────────────────────────────
    if feat.is_stale:
        score -= 30.0

    return round(score, 3)


def screen(
    features: dict[str, TickerFeatures],
    held_tickers: set[str],
    top_n: int = 75,
    pinned_tickers: set[str] | None = None,
    region_quotas: dict[str, int] | None = None,
    excluded_regions: set[str] | None = None,
    style_quotas: dict[str, int] | None = None,
    excluded_styles: set[str] | None = None,
) -> tuple[dict[str, TickerFeatures], int]:
    """Return top_n candidates by score, always including held + pinned positions.

    The quota arguments reserve slots per bucket — geography for
    `region_quotas` (see fundmgr.regions), risk/quality character for
    `style_quotas` (fundmgr.styles) — before the ranking is allowed to spend the
    rest. Without them an allocation target is unbuildable rather than merely
    hard: the score is blind to both, so a week where momentum sits in US large
    caps hands the model a list with four Nordic names in it and no way to reach
    30% Nordics from there. Reserved slots are a floor on choice, not a cap —
    the free remainder still goes to whatever scored best.

    Both dials reserve against the same top_n, and a name already picked counts
    towards every bucket it belongs to, so a Nordic compounder settles a Nordic
    slot and a quality slot at once rather than consuming two. Regions are
    reserved first: geography is a fact on the universe row, while a style is a
    reading of fundamentals that may not have landed yet, and the surer dial
    should not be the one squeezed when both are set.

    The exclusion arguments drop a bucket the mix asks for none of, so the
    prompt isn't paying to show names whose buys the guardrails would reject
    anyway. Held and pinned names are never dropped by either: you must be able
    to sell what you own, whatever it is or wherever it is listed.

    Returns (filtered_features, total_screened_out).
    """
    pinned = pinned_tickers or set()
    always = held_tickers | pinned
    region_of_ticker = regions.regions_of(features)
    style_of_ticker = styles.styles_of(features)

    scored = sorted(
        ((sym, _score(feat), feat) for sym, feat in features.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    for bucket_of, dropped in ((region_of_ticker, excluded_regions),
                               (style_of_ticker, excluded_styles)):
        if dropped:
            scored = [
                row for row in scored
                if row[0] in always or bucket_of.get(row[0]) not in dropped
            ]

    selected: dict[str, TickerFeatures] = {}

    for sym, _, feat in scored:
        if sym in always:
            selected[sym] = feat

    _reserve(scored, selected, top_n, region_of_ticker,
             _quota_order(region_quotas, regions.SCHEME))
    _reserve(scored, selected, top_n, style_of_ticker,
             _quota_order(style_quotas, styles.SCHEME))

    remaining = max(0, top_n - len(selected))
    count = 0
    for sym, _, feat in scored:
        if count >= remaining:
            break
        if sym not in selected:
            selected[sym] = feat
            count += 1

    screened_out = len(features) - len(selected)
    return selected, screened_out


def _quota_order(quotas: dict[str, int] | None, scheme) -> list[tuple[str, int]]:
    """Quotas in a fixed bucket order, so a screen is reproducible."""
    quotas = quotas or {}
    return [(b.code, quotas[b.code]) for b in scheme.buckets if quotas.get(b.code)]


def _reserve(
    scored: list,
    selected: dict[str, TickerFeatures],
    top_n: int,
    bucket_of: dict[str, str],
    quotas: list[tuple[str, int]],
) -> None:
    """Fill each bucket's reserved slots from the top of the ranking, in place.

    Names already selected — held, pinned, or reserved by the other dial — count
    towards the quota, so the reservation reads as "the list carries at least N
    of these" rather than "spend N more slots on these".
    """
    for code, quota in quotas:
        need = quota - sum(1 for sym in selected if bucket_of.get(sym) == code)
        for sym, _, feat in scored:
            if need <= 0 or len(selected) >= top_n:
                break
            if sym in selected or bucket_of.get(sym) != code:
                continue
            selected[sym] = feat
            need -= 1
