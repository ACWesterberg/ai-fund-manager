from __future__ import annotations

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

    # ── Stale data is almost never actionable ────────────────────────────────
    if feat.is_stale:
        score -= 30.0

    return round(score, 3)


def screen(
    features: dict[str, TickerFeatures],
    held_tickers: set[str],
    top_n: int = 75,
) -> tuple[dict[str, TickerFeatures], int]:
    """Return top_n candidates by score, always including held positions.

    Returns (filtered_features, total_screened_out).
    """
    scored = sorted(
        ((sym, _score(feat), feat) for sym, feat in features.items()),
        key=lambda x: x[1],
        reverse=True,
    )

    selected: dict[str, TickerFeatures] = {}

    # Held positions always make the cut regardless of score
    for sym, _, feat in scored:
        if sym in held_tickers:
            selected[sym] = feat

    # Fill remaining slots with top scorers
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
