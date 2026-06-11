from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta

import click

from fundmgr.config import load_config, get_enabled_tickers
from fundmgr.data.benchmark import fetch_and_cache_benchmark, get_benchmark_return_pct
from fundmgr.data.macro_context import build_macro_block, fetch_macro_headlines, fetch_macro_indicators
from fundmgr.data.news import attach_sentiment_to_features, check_news_triggers, fetch_news, score_and_cache_sentiment
from fundmgr.data.prices import build_all_features, fetch_and_cache_prices
from fundmgr.data.screener import screen
from fundmgr.engine.client import LLMError, call_llm
from fundmgr.reporting.dashboard import format_text_report, generate_html_report
from fundmgr.engine.evaluator import evaluate_pending_outcomes, generate_learnings
from fundmgr.engine.prompt import build_prompt, snapshot_to_dict
from fundmgr.guardrails.rules import apply_guardrails
from fundmgr.reporting.actions import format_action_list
from fundmgr.state.models import PortfolioSnapshot, RecommendationLog, Transaction
from fundmgr.state.store import Store


def _get_store(cfg=None) -> tuple:
    if cfg is None:
        cfg = load_config()
    store = Store(cfg.db_path)
    return cfg, store


@click.group()
def cli():
    """AI Fund Manager — weekly LLM-driven portfolio decisions for Nordic equities."""
    pass


@cli.command()
@click.option("--capital", type=float, default=None, help="Override starting capital (SEK)")
def init(capital: float | None):
    """Initialise the portfolio with starting cash. Run once before anything else."""
    cfg, store = _get_store()
    starting_capital = capital or cfg.capital_sek
    try:
        store.initialise(starting_capital)
        click.echo(f"✓ Portfolio initialised with {starting_capital:,.0f} SEK")
        click.echo(f"  Database: {cfg.db_path}")
        click.echo(f"  Universe: {len(get_enabled_tickers())} enabled tickers")
        click.echo("\nNext step: run 'fund run' to generate your first decision.")
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--dry-run", is_flag=True, help="Run pipeline but skip saving recommendation")
@click.option("--force-refresh", is_flag=True, help="Re-fetch all prices even if cached")
@click.option("--skip-news", is_flag=True, help="Skip Nordic RSS + FinBERT sentiment step (faster)")
@click.option("--skip-macro", is_flag=True, help="Skip global macro context fetch (no yfinance indicator or news fetch)")
def run(dry_run: bool, force_refresh: bool, skip_news: bool, skip_macro: bool):
    """Ingest data, call the LLM, apply guardrails, and emit the action list."""
    cfg, store = _get_store()
    if not store.is_initialised():
        click.echo("Portfolio not initialised. Run 'fund init' first.", err=True)
        sys.exit(1)

    tickers = get_enabled_tickers()
    click.echo(f"\n{'═'*56}")
    click.echo(f"  AI Fund Manager — Weekly Run")
    click.echo(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    click.echo(f"{'═'*56}")

    # ── Step 1: Fetch prices ──────────────────────────────────────────────────
    click.echo(f"\n[1/5] Fetching prices for {len(tickers)} tickers…")
    fetch_result = fetch_and_cache_prices(tickers, store, cfg.data.lookback_days, force_refresh)
    ok = sum(1 for v in fetch_result.values() if v)
    failed = [sym for sym, v in fetch_result.items() if not v]
    click.echo(f"      {ok}/{len(tickers)} tickers resolved")
    if failed:
        click.echo(f"      ✗ Failed: {', '.join(failed)}")

    # ── Step 2: Fetch benchmark ────────────────────────────────────────────────
    click.echo(f"\n[2/5] Fetching benchmark ({cfg.benchmark})…")
    bench_ok = fetch_and_cache_benchmark(store, cfg.benchmark, cfg.data.lookback_days, force_refresh)
    click.echo(f"      {'✓ OK' if bench_ok else '✗ Failed'}")

    # ── Step 3: Nordic news + FinBERT sentiment ───────────────────────────────
    since_news = (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d")
    if not skip_news and cfg.data.news_feeds:
        click.echo(f"\n[3/5] Fetching Nordic news from {len(cfg.data.news_feeds)} feeds…")
        ticker_news = fetch_news(cfg.data.news_feeds, tickers, max_age_hours=72)
        total_headlines = sum(len(v) for v in ticker_news.values())
        click.echo(f"      {total_headlines} matched headlines across {len(ticker_news)} tickers")
        if total_headlines > 0 and cfg.data.sentiment.enabled:
            click.echo(f"      Scoring with FinBERT ({cfg.data.sentiment.model})…")
            score_and_cache_sentiment(
                ticker_news, store, cfg.data.sentiment.model, cfg.data.sentiment.device
            )
    else:
        click.echo(f"\n[3/5] Nordic news skipped.")

    # ── Step 4: Global macro context ─────────────────────────────────────────
    macro_block = ""
    if not skip_macro:
        click.echo(f"\n[4/5] Fetching global macro context…")
        macro_indicators = fetch_macro_indicators()
        macro_headlines = fetch_macro_headlines(cfg.data.macro_feeds) if cfg.data.macro_feeds else []
        macro_block = build_macro_block(macro_indicators, macro_headlines)
        ind_ok = sum(1 for i in macro_indicators if i.price is not None)
        click.echo(f"      {ind_ok}/{len(macro_indicators)} indicators | {len(macro_headlines)} global headlines")
    else:
        click.echo(f"\n[4/5] Global macro context skipped.")

    # ── Step 5: Compute features ──────────────────────────────────────────────
    click.echo(f"\n[5/5] Computing features…")
    features = build_all_features(tickers, store, cfg, fetch_result)
    attach_sentiment_to_features(features, store, since_date=since_news)

    stale = [sym for sym, f in features.items() if f.is_stale]
    click.echo(f"      {len(features)} tickers with features computed")
    if stale:
        click.echo(f"      ⚠ Stale data (>{cfg.risk.stale_after_days}d): {', '.join(stale)}")

    # ── Pre-screen: cut to top_n candidates before LLM ───────────────────────
    held_tickers = {p.ticker for p in store.get_positions()}
    screened_features, screened_out = screen(features, held_tickers, top_n=cfg.screener.top_n)
    if screened_out > 0:
        click.echo(f"      Screener: {len(screened_features)} candidates → LLM "
                   f"({screened_out} filtered out, held positions always included)")

    # ── Data quality summary ──────────────────────────────────────────────────
    click.echo(f"\n{'─'*56}")
    click.echo(f"  Data Quality Report")
    click.echo(f"{'─'*56}")
    _print_feature_table(features, cfg)

    # ── Step 5: Retrospective evaluation + learnings ─────────────────────────
    updated = evaluate_pending_outcomes(store)
    if updated:
        new_learnings = generate_learnings(store)
        click.echo(f"\n[*] Evaluated {updated} past decisions; {len(new_learnings)} new learnings generated.")

    # ── Step 6: Build portfolio snapshot (attach live prices) ────────────────
    positions = store.get_positions()
    for p in positions:
        feat = features.get(p.ticker)
        if feat:
            p.current_price_sek = feat.last_price
    snap = PortfolioSnapshot(positions=positions, cash_sek=store.get_cash())

    # ── Step 7: Assemble prompt (use screened candidates) ────────────────────
    run_id = f"{datetime.utcnow().strftime('%Y-%m-%d')}-{__import__('uuid').uuid4().hex[:6]}"
    system_msg, user_msg = build_prompt(cfg, snap, screened_features, store, run_id, macro_block=macro_block)

    # ── Call LLM ─────────────────────────────────────────────────────────────
    click.echo(f"\n[→] Calling {cfg.llm.provider}/{cfg.llm.model_id}…")
    try:
        decision, raw_response = call_llm(system_msg, user_msg, cfg)
    except LLMError as e:
        click.echo(f"  ✗ LLM call failed: {e}", err=True)
        sys.exit(1)
    click.echo(f"      ✓ Got {len(decision.actions)} action(s)")

    # ── Step 9: Apply guardrails ──────────────────────────────────────────────
    universe_tickers = {t.yahoo_ticker for t in tickers}
    guardrail_result = apply_guardrails(decision, snap, features, universe_tickers, cfg)

    rejected_count = sum(1 for v in guardrail_result.verdicts if not v.approved)
    clipped_count = sum(1 for v in guardrail_result.verdicts if v.clipped)
    if rejected_count:
        click.echo(f"      Guardrails: {rejected_count} rejected, {clipped_count} clipped")

    # ── Step 10: Save recommendation log ─────────────────────────────────────
    if not dry_run:
        rec = RecommendationLog(
            run_id=run_id,
            timestamp=datetime.utcnow(),
            prompt_snapshot=snapshot_to_dict(snap, system_msg, user_msg),
            llm_response=raw_response,
            guardrail_log=json.dumps(guardrail_result.to_log()),
            actions_json=json.dumps([a.model_dump() for a in guardrail_result.approved_actions]),
        )
        store.save_recommendation(rec)
        store.seed_outcomes_for_run(
            run_id,
            json.dumps([a.model_dump() for a in guardrail_result.approved_actions]),
        )
        click.echo(f"      Recommendation saved (run_id: {run_id})")

    # ── Step 11: Print action list ────────────────────────────────────────────
    action_list = format_action_list(decision, guardrail_result, snap, features, cfg)
    click.echo("\n" + action_list)

    # Save action list to file
    if not dry_run:
        report_path = cfg.db_path.parent / "reports" / f"actions_{run_id}.md"
        report_path.write_text(action_list)
        click.echo(f"\n  Action list saved to: {report_path}")

    # ── Telegram notification ─────────────────────────────────────────────────
    if not dry_run:
        import urllib.parse
        import urllib.request as _req
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id   = os.getenv("TELEGRAM_CHAT_ID", "")
        if bot_token and chat_id:
            buys  = [a for a in guardrail_result.approved_actions if a.side == "buy"]
            sells = [a for a in guardrail_result.approved_actions if a.side == "sell"]
            lines = [f"<b>📊 Fund Run Complete</b>  <code>{run_id}</code>"]
            lines.append(decision.market_summary)
            if buys:
                lines.append("")
                lines.append("<b>🟢 BUYS</b>")
                for a in buys:
                    lines.append(f"  {a.ticker}  {a.target_weight_pct:.0f}%  conf {a.confidence:.2f}")
                    lines.append(f"  <i>{a.thesis[:140]}</i>")
            if sells:
                lines.append("")
                lines.append("<b>🔴 SELLS</b>")
                for a in sells:
                    lines.append(f"  {a.ticker}  → {a.target_weight_pct:.0f}%  conf {a.confidence:.2f}")
                    lines.append(f"  <i>{a.thesis[:140]}</i>")
            if not buys and not sells:
                lines.append("No trades this run — holding cash.")
            if decision.notes:
                lines.append(f"\n<i>{decision.notes[:200]}</i>")
            msg = "\n".join(lines)
            try:
                data = urllib.parse.urlencode({
                    "chat_id": chat_id, "text": msg, "parse_mode": "HTML",
                }).encode()
                _req.urlopen(
                    f"https://api.telegram.org/bot{bot_token}/sendMessage",
                    data, timeout=10,
                )
                click.echo("  Telegram notification sent.")
            except Exception as e:
                click.echo(f"  ⚠ Telegram notification failed: {e}", err=True)


def _print_feature_table(features, cfg):
    from fundmgr.data.prices import TickerFeatures
    if not features:
        click.echo("  No features available.")
        return

    # Sort by 20d return descending
    sorted_f = sorted(
        features.values(),
        key=lambda f: f.return_20d_pct or 0,
        reverse=True,
    )

    click.echo(f"  {'Ticker':<16} {'Last Price':>10} {'1d%':>6} {'5d%':>6} {'20d%':>7} {'RSI':>5} {'Vol%':>6} {'Senti'}")
    click.echo(f"  {'─'*16} {'─'*10} {'─'*6} {'─'*6} {'─'*7} {'─'*5} {'─'*6} {'─'*8}")
    for f in sorted_f[:25]:  # cap at 25 rows
        r1 = f"{f.return_1d_pct:+.1f}" if f.return_1d_pct is not None else "  n/a"
        r5 = f"{f.return_5d_pct:+.1f}" if f.return_5d_pct is not None else "  n/a"
        r20 = f"{f.return_20d_pct:+.1f}" if f.return_20d_pct is not None else "   n/a"
        rsi = f"{f.rsi_14:.0f}" if f.rsi_14 is not None else " n/a"
        vol = f"{f.vol_20d_ann_pct:.0f}" if f.vol_20d_ann_pct is not None else "  n/a"
        senti = f.sentiment_label[:3].upper() if f.sentiment_label else "  -"
        stale_flag = " ⚠" if f.is_stale else ""
        click.echo(f"  {f.ticker:<16} {f.last_price:>10.2f} {r1:>6} {r5:>6} {r20:>7} {rsi:>5} {vol:>6} {senti}{stale_flag}")


@cli.command()
@click.argument("ticker")
@click.argument("shares", type=float)
@click.argument("price", type=float)
@click.argument("fee", type=float)
@click.option("--side", type=click.Choice(["buy", "sell"]), default="buy", show_default=True)
@click.option("--date", "trade_date", default=None, metavar="YYYY-MM-DD",
              help="Trade date (defaults to today). Use when recording a past fill.")
def fill(ticker: str, shares: float, price: float, fee: float, side: str, trade_date: str | None):
    """Record an actual fill from the broker.

    \b
    Example:
        fund fill VOLV-B.ST 12 291.50 2.91
        fund fill SAND.ST 8 217.80 1.74 --side sell
        fund fill LIME.ST 200 199.40 39.88 --date 2026-06-10
    """
    cfg, store = _get_store()
    if not store.is_initialised():
        click.echo("Portfolio not initialised. Run 'fund init' first.", err=True)
        sys.exit(1)

    if trade_date:
        try:
            ts = datetime.strptime(trade_date, "%Y-%m-%d").replace(hour=12, minute=0)
        except ValueError:
            click.echo(f"Invalid date '{trade_date}' — expected YYYY-MM-DD", err=True)
            sys.exit(1)
    else:
        ts = datetime.utcnow()

    ticker = ticker.upper()
    txn = Transaction(
        ticker=ticker,
        side=side,
        shares=shares,
        price_sek=price,
        fee_sek=fee,
        source="fill",
        timestamp=ts,
    )

    store.apply_fill(txn)

    gross = shares * price
    direction = "Bought" if side == "buy" else "Sold"
    click.echo(f"✓ {direction} {shares} × {ticker} @ {price:.2f} SEK = {gross:,.0f} SEK (fee: {fee:.2f} SEK)")
    cash = store.get_cash()
    click.echo(f"  Cash remaining: {cash:,.0f} SEK")


@cli.command()
def status():
    """Print current portfolio snapshot: positions, cash, NAV."""
    cfg, store = _get_store()
    if not store.is_initialised():
        click.echo("Portfolio not initialised. Run 'fund init' first.", err=True)
        sys.exit(1)

    positions = store.get_positions()
    cash = store.get_cash()

    # We don't have live prices yet (Phase 1), so show cost-basis values
    click.echo("\n─── Portfolio Status ───────────────────────────────")
    if not positions:
        click.echo("  No open positions.")
    else:
        click.echo(f"  {'Ticker':<15} {'Shares':>8} {'Avg Cost':>10} {'Cost Value':>12}")
        click.echo(f"  {'─'*15} {'─'*8} {'─'*10} {'─'*12}")
        total_cost_value = 0.0
        for p in positions:
            cost_value = p.shares * p.avg_cost_sek
            total_cost_value += cost_value
            click.echo(f"  {p.ticker:<15} {p.shares:>8.2f} {p.avg_cost_sek:>10.2f} {cost_value:>12,.0f} SEK")

    click.echo(f"\n  Cash:       {cash:>12,.0f} SEK")
    total_cost_value = sum(p.shares * p.avg_cost_sek for p in positions)
    nav = total_cost_value + cash
    click.echo(f"  NAV (cost): {nav:>12,.0f} SEK")
    click.echo(f"  Fees paid:  {store.total_fees_paid():>12,.2f} SEK")
    click.echo("────────────────────────────────────────────────────\n")


@cli.command()
@click.option("--limit", default=20, show_default=True, help="Number of transactions to show")
def transactions(limit: int):
    """Show recent transaction history."""
    cfg, store = _get_store()
    txns = store.get_transactions(limit=limit)

    if not txns:
        click.echo("No transactions recorded yet.")
        return

    click.echo(f"\n─── Last {limit} Transactions ──────────────────────────────────────────")
    click.echo(f"  {'Date':<20} {'Ticker':<15} {'Side':<6} {'Shares':>8} {'Price':>10} {'Fee':>6} {'Source'}")
    click.echo(f"  {'─'*20} {'─'*15} {'─'*6} {'─'*8} {'─'*10} {'─'*6} {'─'*10}")
    for t in txns:
        date_str = t.timestamp.strftime("%Y-%m-%d %H:%M")
        click.echo(
            f"  {date_str:<20} {t.ticker:<15} {t.side:<6} {t.shares:>8.2f} "
            f"{t.price_sek:>10.2f} {t.fee_sek:>6.2f} {t.source}"
        )
    click.echo()


@cli.command()
@click.option("--html", is_flag=True, help="Also generate an HTML report with chart")
def report(html: bool):
    """Show NAV vs benchmark chart and performance summary."""
    cfg, store = _get_store()
    click.echo(format_text_report(store, cfg))
    if html:
        out = cfg.db_path.parent / "reports" / "report_latest.html"
        generate_html_report(store, cfg, out)
        click.echo(f"\n  HTML report saved to: {out}")


@cli.command("check-stops")
@click.option("--quiet", is_flag=True, help="Suppress output unless a stop or warning fires")
def check_stops(quiet: bool):
    """Fetch live prices and flag any position breaching its stop-loss."""
    import urllib.parse
    import urllib.request as _req
    import yfinance as yf

    cfg, store = _get_store()
    positions = store.get_positions()
    if not positions:
        if not quiet:
            click.echo("No open positions.")
        return

    last_rec = store.get_last_recommendation()
    stop_map: dict[str, float] = {}
    if last_rec:
        try:
            actions = json.loads(last_rec.actions_json)
            for a in actions:
                if a.get("stop_loss_pct") and a.get("ticker"):
                    stop_map[a["ticker"]] = a["stop_loss_pct"]
        except Exception:
            pass

    if not quiet:
        click.echo("\n─── Stop-Loss Check ─────────────────────────────────")
        click.echo(f"  {'Ticker':<16} {'Avg Cost':>9} {'Live Price':>10} {'Chg%':>7} {'Stop':>7} {'Status'}")
        click.echo(f"  {'─'*16} {'─'*9} {'─'*10} {'─'*7} {'─'*7} {'─'*8}")

    stops_hit = []
    warnings = []

    for p in positions:
        try:
            live_price = yf.Ticker(p.ticker).fast_info.last_price
        except Exception:
            live_price = None

        if not live_price:
            if not quiet:
                click.echo(f"  {p.ticker:<16} {p.avg_cost_sek:>9.2f} {'n/a':>10}")
            continue

        chg = (live_price / p.avg_cost_sek - 1) * 100
        stop_pct = stop_map.get(p.ticker)
        stop_str = f"-{stop_pct:.0f}%" if stop_pct else "  n/a"

        if stop_pct and chg <= -stop_pct:
            status = "🚨 STOP HIT"
            stops_hit.append((p.ticker, chg, stop_pct, live_price))
        elif chg < -5:
            status = "⚠ watch"
            warnings.append((p.ticker, chg, live_price))
        else:
            status = "✓ ok"

        if not quiet:
            click.echo(
                f"  {p.ticker:<16} {p.avg_cost_sek:>9.2f} {live_price:>10.2f} "
                f"{chg:>+7.1f}% {stop_str:>7} {status}"
            )

    if not quiet:
        click.echo()
        if stops_hit:
            click.echo(f"  ⚠ Stop-loss triggered for: {', '.join(t for t, *_ in stops_hit)}")
            click.echo("  Consider selling — run 'fund run' to get updated recommendations.")
        else:
            click.echo("  All positions within stop-loss thresholds.")

    # ── Telegram alert ────────────────────────────────────────────────────────
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if (stops_hit or warnings) and bot_token and chat_id:
        lines = ["<b>📉 Price Alert</b>"]
        for ticker, chg, stop_pct, price in stops_hit:
            lines.append(f"🚨 <b>{ticker}</b> {chg:+.1f}% — STOP HIT (stop -{stop_pct:.0f}%)  live {price:.2f}")
        for ticker, chg, price in warnings:
            lines.append(f"⚠ <b>{ticker}</b> {chg:+.1f}% — down &gt;5%  live {price:.2f}")
        if stops_hit:
            lines.append("\nConsider reviewing — trigger <code>/run</code> for updated recommendation.")
        msg = "\n".join(lines)
        try:
            data = urllib.parse.urlencode({
                "chat_id": chat_id, "text": msg, "parse_mode": "HTML",
            }).encode()
            _req.urlopen(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                data, timeout=10,
            )
            click.echo("  Telegram alert sent.")
        except Exception as e:
            click.echo(f"  ⚠ Telegram send failed: {e}", err=True)


@cli.command("check-news")
@click.option("--auto-run/--no-auto-run", default=True, show_default=True,
              help="Automatically trigger 'fund run' when a high-severity event fires")
@click.option("--max-age-hours", default=8, show_default=True,
              help="Only consider articles published within this window")
def check_news(auto_run: bool, max_age_hours: int):
    """Scan recent headlines and trigger an early run if a held position gets bad news."""
    import os
    import subprocess
    import urllib.parse
    import urllib.request

    cfg, store = _get_store()
    if not cfg.data.news_feeds:
        click.echo("No news feeds configured — set data.news_feeds in config.yaml")
        return

    tickers = get_enabled_tickers()
    held_tickers = {p.ticker for p in store.get_positions()}

    click.echo(f"\n[check-news] Scanning {len(cfg.data.news_feeds)} feeds "
               f"(last {max_age_hours}h, {len(held_tickers)} held positions)…")

    triggers = check_news_triggers(
        cfg.data.news_feeds,
        tickers,
        held_tickers,
        store,
        cfg.data.sentiment,
        max_age_hours=max_age_hours,
    )

    if not triggers:
        click.echo("  No trigger-worthy articles found.")
        return

    # ── Print and notify ──────────────────────────────────────────────────────
    held_triggers = [t for t in triggers if t["is_held"]]
    watch_triggers = [t for t in triggers if not t["is_held"]]

    click.echo(f"\n  {len(triggers)} trigger(s) found:")
    for t in triggers:
        flag = "🔴 HELD" if t["is_held"] else "🟡 WATCH"
        click.echo(f"  {flag} {t['ticker']}  [{t['sentiment_label'].upper()} {t['sentiment_score']:.2f}]")
        click.echo(f"       {t['headline'][:100]}")

    # Send Telegram notification
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if bot_token and chat_id:
        lines = ["<b>⚡ News Trigger</b>"]
        for t in triggers:
            flag = "🔴" if t["is_held"] else "🟡"
            label = t["sentiment_label"].upper()
            lines.append(f"{flag} <b>{t['ticker']}</b> [{label} {t['sentiment_score']:.2f}]")
            lines.append(f"  {t['headline'][:120]}")
        if auto_run and held_triggers:
            lines.append("\n▶ Triggering early decision run…")
        msg = "\n".join(lines)
        try:
            data = urllib.parse.urlencode({
                "chat_id": chat_id,
                "text": msg,
                "parse_mode": "HTML",
            }).encode()
            urllib.request.urlopen(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                data,
                timeout=10,
            )
        except Exception as e:
            click.echo(f"  ⚠ Telegram send failed: {e}", err=True)

    # ── Trigger early run if a held position is affected ─────────────────────
    if auto_run and held_triggers:
        click.echo("\n  Held position affected — triggering early 'fund run'…")
        tickers_affected = ", ".join(t["ticker"] for t in held_triggers)
        click.echo(f"  Affected: {tickers_affected}")
        try:
            fund_bin = sys.argv[0]  # same binary we're running as
            subprocess.run([fund_bin, "run"], check=True)
        except subprocess.CalledProcessError as e:
            click.echo(f"  ✗ fund run failed (exit {e.returncode})", err=True)
    elif held_triggers:
        click.echo("\n  (--no-auto-run set — skipping automatic run)")


@cli.command()
def reconcile():
    """Sync read-only holdings from Montrose MCP and flag drift."""
    click.echo("[ Reconciliation with Montrose — implemented in Phase 4 ]")


@cli.command()
def universe():
    """List the enabled tickers in the universe."""
    tickers = get_enabled_tickers()
    click.echo(f"\n─── Universe ({len(tickers)} enabled tickers) ───────────────────────────────")
    click.echo(f"  {'Name':<30} {'Ticker':<15} {'Country':<8} {'Sector'}")
    click.echo(f"  {'─'*30} {'─'*15} {'─'*8} {'─'*20}")
    for t in tickers:
        click.echo(f"  {t.name:<30} {t.yahoo_ticker:<15} {t.country:<8} {t.sector}")
    click.echo()


if __name__ == "__main__":
    cli()
