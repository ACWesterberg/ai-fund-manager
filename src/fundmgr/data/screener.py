from __future__ import annotations

from fundmgr.data.prices import TickerFeatures


def _score(feat: TickerFeatures) -> float:
    score = 0.0

    # Momentum — 20d is the primary signal, 5d adds recency, 60d adds trend context
    if feat.return_1d_pct is not None:
        score += feat.return_1d_pct * 0.10
    if feat.return_5d_pct is not None:
        score += feat.return_5d_pct * 0.25
    if feat.return_20d_pct is not None:
        score += feat.return_20d_pct * 0.45
    if feat.return_60d_pct is not None:
        score += feat.return_60d_pct * 0.20

    # Trend alignment
    if feat.above_ma50 is True:
        score += 3.0
    elif feat.above_ma50 is False:
        score -= 1.0
    if feat.above_ma200 is True:
        score += 2.0

    # RSI — exclude overbought, reward room to run
    if feat.rsi_14 is not None:
        if feat.rsi_14 > 75:
            score -= 15.0
        elif feat.rsi_14 > 70:
            score -= 7.0
        elif feat.rsi_14 < 30:
            score += 5.0
        elif feat.rsi_14 < 40:
            score += 2.0

    # Sentiment
    if feat.sentiment_label == "positive" and feat.sentiment_score is not None:
        score += feat.sentiment_score * 6.0
    elif feat.sentiment_label == "negative" and feat.sentiment_score is not None:
        score -= feat.sentiment_score * 6.0

    # Stale data is almost never actionable
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
