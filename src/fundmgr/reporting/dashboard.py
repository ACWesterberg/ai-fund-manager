from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fundmgr.config import AppConfig
from fundmgr.state.models import NavPoint, PortfolioSnapshot
from fundmgr.state.store import Store


def compute_stats(nav_history: list[NavPoint], initial_capital: float) -> dict:
    if not nav_history:
        return {}

    navs = [n.portfolio_nav_sek for n in nav_history]
    benches = [n.benchmark_value for n in nav_history]

    # Time-weighted return
    twr = (navs[-1] / navs[0] - 1) * 100 if navs[0] else 0.0

    # Benchmark return over same window
    bench_return = (benches[-1] / benches[0] - 1) * 100 if benches[0] else 0.0

    # Max drawdown
    peak = navs[0]
    max_dd = 0.0
    for nav in navs:
        if nav > peak:
            peak = nav
        dd = (peak - nav) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Volatility (daily returns std, annualised)
    vol = None
    if len(navs) > 5:
        daily_returns = [(navs[i] / navs[i - 1] - 1) for i in range(1, len(navs))]
        import math
        mean = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean) ** 2 for r in daily_returns) / len(daily_returns)
        vol = math.sqrt(variance) * math.sqrt(252) * 100

    return {
        "twr_pct": round(twr, 2),
        "benchmark_return_pct": round(bench_return, 2),
        "alpha_pct": round(twr - bench_return, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "volatility_ann_pct": round(vol, 1) if vol is not None else None,
        "days_running": len(nav_history),
        "first_date": nav_history[0].date,
        "last_date": nav_history[-1].date,
        "nav_start": navs[0],
        "nav_current": navs[-1],
        "initial_capital": initial_capital,
    }


def nav_chart_json(nav_history: list[NavPoint]) -> str:
    """Return Plotly-compatible JSON for the NAV vs benchmark chart."""
    if not nav_history:
        return json.dumps({"data": [], "layout": {}})

    nav0 = nav_history[0].portfolio_nav_sek
    if nav0 == 0:
        return json.dumps({"data": [], "layout": {}})

    nav_indexed = [round(n.portfolio_nav_sek / nav0 * 100, 2) for n in nav_history]
    dates = [n.date for n in nav_history]

    data = [
        {
            "x": dates,
            "y": nav_indexed,
            "type": "scatter",
            "mode": "lines",
            "name": "Portfolio",
            "line": {"color": "#2563eb", "width": 2},
        },
    ]

    # Only render benchmark line if we have real (non-zero) values to index from
    bench_vals_raw = [n.benchmark_value for n in nav_history]
    # Forward-fill zeros from the last known value so gaps don't crater the line
    bench_filled: list[float] = []
    last_valid = next((v for v in bench_vals_raw if v > 0), None)
    for v in bench_vals_raw:
        if v > 0:
            last_valid = v
        bench_filled.append(last_valid or 0.0)

    bench0 = bench_filled[0] if bench_filled else 0.0
    if bench0 > 0:
        bench_indexed = [round(v / bench0 * 100, 2) if v else None for v in bench_filled]
        data.append({
            "x": dates,
            "y": bench_indexed,
            "type": "scatter",
            "mode": "lines",
            "name": "OMXSPI",
            "line": {"color": "#9ca3af", "width": 1.5, "dash": "dot"},
            "connectgaps": True,
        })

    layout = {
        "xaxis": {"title": "Date", "showgrid": False, "color": "#94a3b8"},
        "yaxis": {"title": "Indexed to 100", "showgrid": True, "gridcolor": "rgba(255,255,255,0.06)", "color": "#94a3b8"},
        "plot_bgcolor": "transparent",
        "paper_bgcolor": "transparent",
        "legend": {"orientation": "h", "y": -0.2, "font": {"color": "#94a3b8"}},
        "margin": {"l": 50, "r": 20, "t": 10, "b": 60},
        "hovermode": "x unified",
        "font": {"color": "#94a3b8"},
    }

    return json.dumps({"data": data, "layout": layout})


def format_text_report(store: Store, cfg: AppConfig) -> str:
    nav_history = store.get_nav_history()
    positions = store.get_positions()
    cash = store.get_cash()
    fees_paid = store.total_fees_paid()
    stats = compute_stats(nav_history, cfg.capital_sek)

    lines = ["═" * 58]
    lines.append("  Performance Report")
    lines.append(f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("═" * 58)

    if not stats:
        nav = sum(p.shares * p.avg_cost_sek for p in positions) + cash
        lines.append(f"\n  NAV (cost basis):  {nav:>12,.0f} SEK")
        lines.append(f"  Cash:              {cash:>12,.0f} SEK")
        lines.append(f"  Fees paid:         {fees_paid:>12.2f} SEK")
        lines.append("\n  No NAV history yet — run 'fund run' to start tracking.")
        return "\n".join(lines)

    lines.append(f"\n  Period:         {stats['first_date']} → {stats['last_date']} ({stats['days_running']} days)")
    lines.append(f"  Starting NAV:   {stats['nav_start']:>12,.0f} SEK")
    lines.append(f"  Current NAV:    {stats['nav_current']:>12,.0f} SEK")
    lines.append("")
    lines.append(f"  TWR:            {stats['twr_pct']:>+11.2f}%")
    lines.append(f"  OMXSPI:         {stats['benchmark_return_pct']:>+11.2f}%")
    lines.append(f"  Alpha:          {stats['alpha_pct']:>+11.2f}%  {'▲ outperforming' if stats['alpha_pct'] > 0 else '▼ underperforming'}")
    lines.append(f"  Max drawdown:   {-stats['max_drawdown_pct']:>+11.2f}%")
    if stats["volatility_ann_pct"] is not None:
        lines.append(f"  Volatility:     {stats['volatility_ann_pct']:>11.1f}% (ann.)")
    lines.append("")
    lines.append(f"  Fees paid:      {fees_paid:>12.2f} SEK  ({fees_paid/stats['nav_current']*100:.2f}% of NAV)")

    if positions:
        lines.append("\n  ── Open Positions ──────────────────────────────────")
        lines.append(f"  {'Ticker':<16} {'Shares':>8} {'Avg Cost':>10} {'P&L':>8}")
        for p in sorted(positions, key=lambda x: x.shares * x.avg_cost_sek, reverse=True):
            cost_val = p.shares * p.avg_cost_sek
            pnl_str = f"{p.unrealised_pnl_pct:+.1f}%" if p.current_price_sek else "  n/a"
            lines.append(f"  {p.ticker:<16} {p.shares:>8.0f} {p.avg_cost_sek:>10.2f} {pnl_str:>8}")

    lines.append("═" * 58)
    return "\n".join(lines)


def generate_html_report(store: Store, cfg: AppConfig, output_path: Path) -> None:
    """Generate a standalone HTML report with embedded Plotly chart."""
    nav_history = store.get_nav_history()
    stats = compute_stats(nav_history, cfg.capital_sek)
    chart_json = nav_chart_json(nav_history)

    twr = f"{stats.get('twr_pct', 0):+.2f}%" if stats else "n/a"
    bench = f"{stats.get('benchmark_return_pct', 0):+.2f}%" if stats else "n/a"
    alpha = f"{stats.get('alpha_pct', 0):+.2f}%" if stats else "n/a"
    dd = f"{-stats.get('max_drawdown_pct', 0):.2f}%" if stats else "n/a"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI Fund Manager — Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<style>body{{background:#f9fafb;}} .stat-card{{background:#fff;border-radius:8px;padding:1rem 1.5rem;box-shadow:0 1px 3px rgba(0,0,0,.08);}}</style>
</head>
<body class="p-4">
<div class="container-fluid" style="max-width:960px">
  <h1 class="h4 mb-1">AI Fund Manager</h1>
  <p class="text-muted small mb-4">Report generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</p>

  <div class="row g-3 mb-4">
    <div class="col-6 col-md-3"><div class="stat-card text-center">
      <div class="text-muted small">Portfolio TWR</div>
      <div class="h4 mb-0 {'text-success' if stats and stats.get('twr_pct',0)>0 else 'text-danger'}">{twr}</div>
    </div></div>
    <div class="col-6 col-md-3"><div class="stat-card text-center">
      <div class="text-muted small">OMXSPI</div>
      <div class="h4 mb-0">{bench}</div>
    </div></div>
    <div class="col-6 col-md-3"><div class="stat-card text-center">
      <div class="text-muted small">Alpha</div>
      <div class="h4 mb-0 {'text-success' if stats and stats.get('alpha_pct',0)>0 else 'text-danger'}">{alpha}</div>
    </div></div>
    <div class="col-6 col-md-3"><div class="stat-card text-center">
      <div class="text-muted small">Max Drawdown</div>
      <div class="h4 mb-0 text-danger">{dd}</div>
    </div></div>
  </div>

  <div class="bg-white rounded p-3 shadow-sm mb-4">
    <div id="nav-chart" style="height:360px"></div>
  </div>
</div>
<script>
var chartData = {chart_json};
Plotly.newPlot('nav-chart', chartData.data, chartData.layout, {{responsive: true}});
</script>
</body>
</html>"""

    output_path.write_text(html)
