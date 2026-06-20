from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta

import click

from fundmgr.config import load_config, get_enabled_tickers
from fundmgr.data.benchmark import fetch_and_cache_benchmark, get_benchmark_return_pct
from fundmgr.data.fundamentals import apply_to_features, fetch_and_cache_fundamentals
from fundmgr.data.macro_context import build_macro_block, fetch_macro_headlines, fetch_macro_indicators
from fundmgr.data.news import attach_sentiment_to_features, check_news_triggers, fetch_news, score_and_cache_sentiment
from fundmgr.data.prices import build_all_features, fetch_and_cache_prices
from fundmgr.data.screener import screen
from fundmgr.engine.client import LLMError, call_llm_consensus
from fundmgr.reporting.dashboard import format_text_report, generate_html_report
from fundmgr.engine.evaluator import evaluate_pending_outcomes, generate_learnings, generate_qualitative_learnings
from fundmgr.engine.prompt import build_prompt, snapshot_to_dict
from fundmgr.guardrails.rules import apply_guardrails
from fundmgr.reporting.actions import format_action_list
from fundmgr.state.models import NavPoint, PortfolioSnapshot, RecommendationLog, Transaction
from fundmgr.state.store import Store


# After this many logged runs (≈ weeks at weekly cadence) the post-run Telegram
# notification nudges once to check the Refine gate. Re-arms after `fund reset`.
REFINE_REMINDER_AFTER_RUNS = 4


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
        click.echo(f"  Universe: {len(get_enabled_tickers(cfg.universe_path))} enabled tickers")
        click.echo("\nNext step: run 'fund run' to generate your first decision.")
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--capital", type=float, default=None,
              help="Fresh starting capital (SEK). Defaults to the config's capital_sek.")
@click.option("--purge-cache", is_flag=True,
              help="Also clear market-data caches (prices, benchmark, fundamentals, news).")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt (for scripted use).")
def reset(capital: float | None, purge_cache: bool, yes: bool):
    """Wipe ALL portfolio state + decision history and start fresh.

    Destructive: clears positions, transactions, recommendations, NAV history,
    decision outcomes, learnings and stops for the fund selected by FUND_CONFIG,
    then re-initialises cash to a fresh balance. Market-data caches are kept
    unless --purge-cache is given.
    """
    cfg, store = _get_store()
    starting_capital = capital or cfg.capital_sek

    click.echo("⚠ This will PERMANENTLY erase this fund's portfolio and history:")
    click.echo(f"    Database: {cfg.db_path}")
    click.echo(f"    Fresh capital: {starting_capital:,.0f} SEK")
    click.echo(f"    Market-data caches: {'CLEARED' if purge_cache else 'kept'}")
    if not yes and not click.confirm("Proceed?", default=False):
        click.echo("Aborted — nothing changed.")
        return

    deleted = store.reset(starting_capital, purge_cache=purge_cache)
    total = sum(deleted.values())
    click.echo(f"✓ Reset complete — {total} row(s) cleared across {len(deleted)} table(s).")
    for table, n in deleted.items():
        if n:
            click.echo(f"    {table:<20} {n:>6} row(s)")
    click.echo(f"✓ Portfolio re-initialised with {starting_capital:,.0f} SEK")
    click.echo("\nNext step: run 'fund run' to generate the first decision of the fresh sim.")


@cli.command()
@click.option("--dry-run", is_flag=True, help="Run pipeline but skip saving recommendation")
@click.option("--force-refresh", is_flag=True, help="Re-fetch all prices even if cached")
@click.option("--skip-news", is_flag=True, help="Skip Nordic RSS + FinBERT sentiment step (faster)")
@click.option("--skip-macro", is_flag=True, help="Skip global macro context fetch (no yfinance indicator or news fetch)")
@click.option("--skip-fundamentals", is_flag=True, help="Skip fundamentals cache refresh (use cached data as-is)")
def run(dry_run: bool, force_refresh: bool, skip_news: bool, skip_macro: bool, skip_fundamentals: bool):
    """Ingest data, call the LLM, apply guardrails, and emit the action list."""
    cfg, store = _get_store()
    if not store.is_initialised():
        click.echo("Portfolio not initialised. Run 'fund init' first.", err=True)
        sys.exit(1)

    tickers = get_enabled_tickers(cfg.universe_path)
    click.echo(f"\n{'═'*56}")
    click.echo(f"  AI Fund Manager — Weekly Run")
    click.echo(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    click.echo(f"{'═'*56}")

    # ── Step 1: Fetch prices ──────────────────────────────────────────────────
    click.echo(f"\n[1/6] Fetching prices for {len(tickers)} tickers…")
    fetch_result = fetch_and_cache_prices(tickers, store, cfg.data.lookback_days, force_refresh)
    ok = sum(1 for v in fetch_result.values() if v)
    failed = [sym for sym, v in fetch_result.items() if not v]
    click.echo(f"      {ok}/{len(tickers)} tickers resolved")
    if failed:
        click.echo(f"      ✗ Failed: {', '.join(failed)}")

    # ── Step 2: Fetch / refresh fundamentals cache (weekly TTL) ───────────────
    ticker_symbols = [t.yahoo_ticker for t in tickers]
    if not skip_fundamentals:
        stale_count = len(store.get_stale_fundamentals_tickers(ticker_symbols, ttl_days=7))
        click.echo(f"\n[2/6] Fundamentals cache — {stale_count} tickers need refresh…")
        if stale_count > 0:
            refreshed = fetch_and_cache_fundamentals(ticker_symbols, store, ttl_days=7, max_workers=12)
            click.echo(f"      {refreshed}/{stale_count} refreshed")
        else:
            click.echo(f"      All fresh (TTL 7d)")
    else:
        click.echo(f"\n[2/6] Fundamentals refresh skipped.")

    # ── Step 3: Fetch benchmark ────────────────────────────────────────────────
    click.echo(f"\n[3/6] Fetching benchmark ({cfg.benchmark})…")
    bench_ok = fetch_and_cache_benchmark(store, cfg.benchmark, cfg.data.lookback_days, force_refresh)
    click.echo(f"      {'✓ OK' if bench_ok else '✗ Failed'}")

    # ── Step 4: Nordic news + FinBERT sentiment ───────────────────────────────
    since_news = (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d")
    if not skip_news and cfg.data.news_feeds:
        click.echo(f"\n[4/6] Fetching Nordic news from {len(cfg.data.news_feeds)} feeds…")
        ticker_news = fetch_news(cfg.data.news_feeds, tickers, max_age_hours=72)
        total_headlines = sum(len(v) for v in ticker_news.values())
        click.echo(f"      {total_headlines} matched headlines across {len(ticker_news)} tickers")
        if total_headlines > 0 and cfg.data.sentiment.enabled:
            click.echo(f"      Scoring with FinBERT ({cfg.data.sentiment.model})…")
            score_and_cache_sentiment(
                ticker_news, store, cfg.data.sentiment.model, cfg.data.sentiment.device
            )
    else:
        click.echo(f"\n[4/6] Nordic news skipped.")

    # ── Step 5: Global macro context ─────────────────────────────────────────
    macro_block = ""
    if not skip_macro:
        click.echo(f"\n[5/6] Fetching global macro context…")
        macro_indicators = fetch_macro_indicators()
        macro_headlines = fetch_macro_headlines(cfg.data.macro_feeds) if cfg.data.macro_feeds else []
        macro_block = build_macro_block(macro_indicators, macro_headlines)
        ind_ok = sum(1 for i in macro_indicators if i.price is not None)
        click.echo(f"      {ind_ok}/{len(macro_indicators)} indicators | {len(macro_headlines)} global headlines")
    else:
        click.echo(f"\n[5/6] Global macro context skipped.")

    # ── Step 6: Compute features + attach fundamentals & sentiment ────────────
    click.echo(f"\n[6/6] Computing features…")
    features = build_all_features(tickers, store, cfg, fetch_result)
    apply_to_features(features, store)
    attach_sentiment_to_features(features, store, since_date=since_news)

    fund_count = sum(1 for f in features.values() if f.ev_to_ebitda is not None or f.revenue_growth_pct is not None)
    stale = [sym for sym, f in features.items() if f.is_stale]
    click.echo(f"      {len(features)} tickers with features  ({fund_count} with fundamentals data)")
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
    evaluated = evaluate_pending_outcomes(store)
    if evaluated:
        stat_learnings = generate_learnings(store)
        qual_learnings = generate_qualitative_learnings(store, evaluated)
        total_learnings = len(stat_learnings) + len(qual_learnings)
        click.echo(
            f"\n[*] Evaluated {len(evaluated)} past decisions; "
            f"{total_learnings} new learnings generated "
            f"({len(qual_learnings)} qualitative, {len(stat_learnings)} calibration)."
        )

    # ── Step 6: Build portfolio snapshot (attach live prices) ────────────────
    positions = store.get_positions()
    for p in positions:
        feat = features.get(p.ticker)
        if feat:
            p.current_price_sek = feat.last_price
    snap = PortfolioSnapshot(positions=positions, cash_sek=store.get_cash())

    # ── Step 7: Assemble prompt (use screened candidates) ────────────────────
    run_id = f"{datetime.utcnow().strftime('%Y-%m-%d')}-{__import__('uuid').uuid4().hex[:6]}"
    # ── Cold-start: lift turnover cap when deploying from near-100% cash ─────
    import copy
    effective_cfg = cfg
    if snap.cash_pct >= cfg.risk.cold_start_cash_threshold:
        effective_cfg = copy.copy(cfg)
        effective_cfg.risk = copy.copy(cfg.risk)
        effective_cfg.risk.max_turnover_pct = cfg.risk.cold_start_turnover_pct
        click.echo(f"\n      Cold start detected (cash {snap.cash_pct:.0f}%): "
                   f"turnover cap → {cfg.risk.cold_start_turnover_pct:.0f}%")

    system_msg, user_msg, prompt_fields = build_prompt(effective_cfg, snap, screened_features, store, run_id, macro_block=macro_block)

    # ── Call LLM (with optional consensus sampling) ───────────────────────────
    n_samples = effective_cfg.llm.n_samples
    if n_samples > 1:
        click.echo(f"\n[→] Calling {cfg.llm.provider}/{cfg.llm.model_id} × {n_samples} (consensus mode)…")
    else:
        click.echo(f"\n[→] Calling {cfg.llm.provider}/{cfg.llm.model_id}…")
    try:
        decision, raw_response, vote_counts, sampling = call_llm_consensus(system_msg, user_msg, effective_cfg)
    except LLMError as e:
        click.echo(f"  ✗ LLM call failed: {e}", err=True)
        sys.exit(1)
    if sampling["failed"]:
        click.echo(f"      ⚠ {sampling['failed']}/{sampling['requested']} sample(s) failed to parse")
    if vote_counts is not None and n_samples > 1:
        unanimous = sum(1 for v in vote_counts.values() if v == n_samples)
        majority  = len(vote_counts) - unanimous
        click.echo(
            f"      ✓ Consensus: {len(decision.actions)} action(s) "
            f"({unanimous} unanimous, {majority} majority-only)"
        )
    else:
        click.echo(f"      ✓ Got {len(decision.actions)} action(s)")

    # ── Step 9: Apply guardrails ──────────────────────────────────────────────
    universe_tickers = {t.yahoo_ticker for t in tickers}
    guardrail_result = apply_guardrails(decision, snap, features, universe_tickers, effective_cfg)

    rejected_count = sum(1 for v in guardrail_result.verdicts if not v.approved)
    clipped_count = sum(1 for v in guardrail_result.verdicts if v.clipped)
    if rejected_count:
        click.echo(f"      Guardrails: {rejected_count} rejected, {clipped_count} clipped")

    # ── Step 10: Save recommendation log ─────────────────────────────────────
    if not dry_run:
        rec = RecommendationLog(
            run_id=run_id,
            timestamp=datetime.utcnow(),
            prompt_snapshot=snapshot_to_dict(snap, system_msg, user_msg, prompt_fields, effective_cfg),
            llm_response=raw_response,
            guardrail_log=json.dumps(guardrail_result.to_log()),
            actions_json=json.dumps([a.model_dump() for a in guardrail_result.approved_actions]),
            sampling_log=json.dumps(sampling),
        )
        store.save_recommendation(rec)
        store.seed_outcomes_for_run(
            run_id,
            json.dumps([a.model_dump() for a in guardrail_result.approved_actions]),
        )
        click.echo(f"      Recommendation saved (run_id: {run_id})")

        # Persist stop/take-profit levels per position so check-stops survives multiple runs
        for action in guardrail_result.approved_actions:
            if action.side == "buy" and (action.stop_loss_pct or action.take_profit_pct):
                store.set_position_stop(
                    action.ticker,
                    stop_pct=action.stop_loss_pct,
                    take_profit_pct=action.take_profit_pct,
                )
            elif action.side == "sell" and action.target_weight_pct == 0:
                store.clear_position_stop(action.ticker)

        # Record NAV snapshot for portfolio chart
        bench_rows = store.get_benchmark()
        bench_val = bench_rows[-1]["close"] if bench_rows else 0.0
        store.upsert_nav(NavPoint(
            date=datetime.utcnow().strftime("%Y-%m-%d"),
            portfolio_nav_sek=snap.nav_sek,
            benchmark_value=bench_val,
            cash_sek=store.get_cash(),
        ))

    # ── Step 11: Print action list ────────────────────────────────────────────
    action_list = format_action_list(
        decision, guardrail_result, snap, features, cfg,
        vote_counts=vote_counts, n_samples=n_samples,
    )
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
            holds = [a for a in guardrail_result.approved_actions if a.side == "hold"]
            tg_consensus = vote_counts is not None and n_samples > 1
            tg_vote = (lambda t: f" [{vote_counts[t]}/{n_samples}]" if tg_consensus and t in vote_counts else "")
            header = f"<b>📊 Fund Run Complete</b>  <code>{run_id}</code>"
            if tg_consensus:
                header += f"  <i>({n_samples}-run consensus)</i>"
            lines = [header]
            lines.append(decision.market_summary)
            # NAV vs benchmark performance line
            from fundmgr.data.benchmark import get_benchmark_return_pct
            nav_history = store.get_nav_history()
            if nav_history:
                first = nav_history[0]
                fund_ret = (snap.nav_sek / first.portfolio_nav_sek - 1) * 100 if first.portfolio_nav_sek else None
                bench_ret = get_benchmark_return_pct(store, since_date=first.date)
                if fund_ret is not None:
                    bench_str = f"  vs {cfg.benchmark} {bench_ret:+.1f}%" if bench_ret is not None else ""
                    lines.append(f"NAV {snap.nav_sek:,.0f} SEK ({fund_ret:+.1f}%{bench_str})")
            if buys:
                lines.append("")
                lines.append("<b>🟢 BUYS</b>")
                for a in buys:
                    lines.append(f"  {a.ticker}  {a.target_weight_pct:.0f}%  conf {a.confidence:.2f}{tg_vote(a.ticker)}")
                    lines.append(f"  <i>{a.thesis}</i>")
            if sells:
                lines.append("")
                lines.append("<b>🔴 SELLS</b>")
                for a in sells:
                    lines.append(f"  {a.ticker}  → {a.target_weight_pct:.0f}%  conf {a.confidence:.2f}{tg_vote(a.ticker)}")
                    lines.append(f"  <i>{a.thesis}</i>")
            if holds:
                lines.append("")
                lines.append("<b>⏸ HOLDS</b>")
                for a in holds:
                    lines.append(f"  {a.ticker}{tg_vote(a.ticker)}  <i>{a.thesis[:120]}</i>")
            if not buys and not sells:
                lines.append("No trades this run — holding cash.")
            if decision.notes:
                lines.append(f"\n<i>{decision.notes}</i>")
            # One-shot reminder: once enough weeks of runs have accumulated,
            # nudge to check the Refine gate. Fires once, re-arms after reset.
            show_refine_reminder = False
            try:
                rec_count = store.count_recommendations()
                if rec_count >= REFINE_REMINDER_AFTER_RUNS and not store.get_meta("refine_gate_reminded"):
                    rs = store.get_rejection_stats()
                    show_refine_reminder = True
                    lines.append(
                        f"\n<b>🔬 Refine-gate check</b>\n"
                        f"{rec_count} runs logged — enough data to decide on Refine.\n"
                        f"  malformed samples: {rs['sample_failure_pct']}%\n"
                        f"  guardrail rejected: {rs['guardrail_reject_pct']}%  clipped: {rs['guardrail_clip_pct']}%\n"
                        f"Run /reject_rates for detail. Build Refine only if a rate is materially non-zero."
                    )
            except Exception:
                pass
            # Split into ≤4096-char chunks at line boundaries
            full_msg = "\n".join(lines)
            chunks, current = [], ""
            for line in full_msg.split("\n"):
                candidate = (current + "\n" + line) if current else line
                if len(candidate) > 4096:
                    chunks.append(current)
                    current = line
                else:
                    current = candidate
            if current:
                chunks.append(current)
            try:
                for chunk in chunks:
                    data = urllib.parse.urlencode({
                        "chat_id": chat_id, "text": chunk, "parse_mode": "HTML",
                    }).encode()
                    _req.urlopen(
                        f"https://api.telegram.org/bot{bot_token}/sendMessage",
                        data, timeout=10,
                    )
                click.echo(f"  Telegram notification sent ({len(chunks)} message(s)).")
                if show_refine_reminder:
                    store.set_meta("refine_gate_reminded", datetime.utcnow().isoformat())
            except Exception as e:
                click.echo(f"  ⚠ Telegram notification failed: {e}", err=True)

    # ── Auto-fill (paper trading simulation) ─────────────────────────────────
    if not dry_run and cfg.auto_fill and guardrail_result.approved_actions:
        from fundmgr.engine.auto_fill import execute_paper_fills
        click.echo("\n[→] Auto-fill: executing paper trades…")
        # Attach live prices to positions before computing trade sizes
        for p in store.get_positions():
            feat = features.get(p.ticker)
            if feat:
                p.current_price_sek = feat.last_price
        fill_log = execute_paper_fills(
            [a.model_dump() for a in guardrail_result.approved_actions],
            store,
            cfg,
        )
        for line in fill_log:
            click.echo(line)


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

    # Record a NAV snapshot (cost-basis) so the chart shows fill events
    try:
        bench_rows = store.get_benchmark()
        bench_val = bench_rows[-1]["close"] if bench_rows else 0.0
        positions_after = store.get_positions()
        nav_cost = sum(p.shares * p.avg_cost_sek for p in positions_after) + cash
        store.upsert_nav(NavPoint(
            date=ts.strftime("%Y-%m-%d"),
            portfolio_nav_sek=nav_cost,
            benchmark_value=bench_val,
            cash_sek=cash,
        ))
    except Exception:
        pass


@cli.command("undo-fill")
def undo_fill():
    """Reverse the most recent fill (position + cash restored, transaction deleted)."""
    cfg, store = _get_store()
    txn = store.undo_last_fill()
    if txn is None:
        click.echo("No transactions to undo.")
        return
    direction = "BUY" if txn.side == "buy" else "SELL"
    click.echo(f"✓ Undone: {direction} {txn.shares} × {txn.ticker} @ {txn.price_sek:.2f} SEK  "
               f"(fee {txn.fee_sek:.2f} SEK, recorded {txn.timestamp.strftime('%Y-%m-%d %H:%M')})")
    click.echo(f"  Cash now: {store.get_cash():,.0f} SEK")


@cli.command("backfill-nav")
def backfill_nav():
    """Reconstruct NAV history from transaction log (one point per trading day)."""
    cfg, store = _get_store()
    if not store.is_initialised():
        click.echo("Portfolio not initialised. Run 'fund init' first.", err=True)
        sys.exit(1)

    txns = store.get_transactions(limit=10_000)
    if not txns:
        click.echo("No transactions found — nothing to backfill.")
        return

    # Sort ascending (get_transactions returns DESC)
    txns = sorted(txns, key=lambda t: t.timestamp)

    # Build benchmark lookup {date_str: close}, sorted by date
    bench_rows = store.get_benchmark()
    bench_lookup: dict[str, float] = {r["date"]: r["close"] for r in bench_rows}
    bench_dates_sorted = sorted(bench_lookup.keys())

    def nearest_bench(date_str: str) -> float:
        """Return benchmark close for date_str, falling back to nearest available date."""
        if date_str in bench_lookup:
            return bench_lookup[date_str]
        if not bench_dates_sorted:
            return 0.0
        # Find closest date (prefer earlier dates so we don't look into the future)
        earlier = [d for d in bench_dates_sorted if d <= date_str]
        return bench_lookup[earlier[-1]] if earlier else bench_lookup[bench_dates_sorted[0]]

    # Replay transactions: simulate portfolio state at each transaction date
    sim_positions: dict[str, tuple[float, float]] = {}  # ticker -> (shares, avg_cost)
    cash = cfg.capital_sek

    # Record starting NAV one day before first transaction (all cash)
    from datetime import timedelta
    start_date = (txns[0].timestamp - timedelta(days=1)).strftime("%Y-%m-%d")
    store.upsert_nav(NavPoint(
        date=start_date,
        portfolio_nav_sek=cfg.capital_sek,
        benchmark_value=nearest_bench(start_date),
        cash_sek=cfg.capital_sek,
    ))

    inserted = 1
    for txn in txns:
        if txn.side == "buy":
            existing_shares, existing_cost = sim_positions.get(txn.ticker, (0.0, 0.0))
            new_shares = existing_shares + txn.shares
            new_cost = (existing_shares * existing_cost + txn.shares * txn.price_sek) / new_shares
            sim_positions[txn.ticker] = (new_shares, new_cost)
            cash -= txn.shares * txn.price_sek + txn.fee_sek
        elif txn.side == "sell":
            existing_shares, existing_cost = sim_positions.get(txn.ticker, (0.0, 0.0))
            new_shares = existing_shares - txn.shares
            if new_shares <= 0.001:
                sim_positions.pop(txn.ticker, None)
            else:
                sim_positions[txn.ticker] = (new_shares, existing_cost)
            cash += txn.shares * txn.price_sek - txn.fee_sek

        date_str = txn.timestamp.strftime("%Y-%m-%d")
        nav_cost = sum(s * c for s, c in sim_positions.values()) + cash
        store.upsert_nav(NavPoint(
            date=date_str,
            portfolio_nav_sek=nav_cost,
            benchmark_value=nearest_bench(date_str),
            cash_sek=cash,
        ))
        inserted += 1

    click.echo(f"✓ Backfilled {inserted} NAV points from {txns[0].timestamp.date()} to {txns[-1].timestamp.date()}")
    click.echo("  (NAV uses cost-basis, not live prices — next 'fund run' will update today's point with live prices)")


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

    # Markets are closed on weekends — nothing to check
    if datetime.utcnow().weekday() >= 5:
        if not quiet:
            click.echo("Weekend — markets closed, skipping stop-loss check.")
        return

    cfg, store = _get_store()
    positions = store.get_positions()
    if not positions:
        if not quiet:
            click.echo("No open positions.")
        return

    stop_map = store.get_position_stops()

    if not quiet:
        click.echo("\n─── Price Level Check ───────────────────────────────")
        click.echo(f"  {'Ticker':<16} {'Avg Cost':>9} {'Live':>8} {'Chg%':>7} {'Levels':>14} {'Status'}")
        click.echo(f"  {'─'*16} {'─'*9} {'─'*8} {'─'*7} {'─'*14} {'─'*12}")

    stops_hit: list[tuple] = []
    profits_hit: list[tuple] = []
    warnings: list[tuple] = []

    today = datetime.utcnow().strftime("%Y-%m-%d")
    today_date = datetime.utcnow().date()

    for p in positions:
        try:
            hist = yf.Ticker(p.ticker).history(period="2d")
        except Exception:
            hist = None

        if hist is None or hist.empty:
            if not quiet:
                click.echo(f"  {p.ticker:<16} {p.avg_cost_sek:>9.2f} {'n/a':>8}")
            continue

        live_price = float(hist["Close"].iloc[-1])

        # Only compute daily change if today's bar is actually present —
        # at market open yfinance still shows yesterday's close as last_price,
        # which would make the "daily" change show yesterday's move instead
        daily_chg = None
        if len(hist) >= 2:
            last_bar_date = hist.index[-1].date()
            if last_bar_date == today_date:
                prev_day_close = float(hist["Close"].iloc[-2])
                daily_chg = (live_price / prev_day_close - 1) * 100
        levels = stop_map.get(p.ticker, {})
        stop_pct = levels.get("stop_pct")
        tp_pct = levels.get("take_profit_pct")

        parts = []
        if stop_pct:
            parts.append(f"-{stop_pct:.0f}%")
        if tp_pct:
            parts.append(f"+{tp_pct:.0f}%")
        levels_str = " / ".join(parts) if parts else "n/a"

        if stop_pct and chg <= -stop_pct:
            status = "🚨 STOP HIT"
            stops_hit.append((p.ticker, chg, stop_pct, live_price))
        elif tp_pct and chg >= tp_pct:
            status = "🎯 TARGET HIT"
            profits_hit.append((p.ticker, chg, tp_pct, live_price))
        elif daily_chg is not None and daily_chg < -5 and not store.has_sent_daily_drop_alert(p.ticker, today):
            status = "⚠ watch"
            warnings.append((p.ticker, chg, daily_chg, live_price, stop_pct))
            store.record_daily_drop_alert(p.ticker, today)
        else:
            status = "✓ ok"

        if not quiet:
            daily_str = f" ({daily_chg:+.1f}% today)" if daily_chg is not None else ""
            click.echo(
                f"  {p.ticker:<16} {p.avg_cost_sek:>9.2f} {live_price:>8.2f} "
                f"{chg:>+7.1f}%{daily_str} {levels_str:>14} {status}"
            )

    if not quiet:
        click.echo()
        if stops_hit:
            click.echo(f"  ⚠ Stop triggered: {', '.join(t for t, *_ in stops_hit)}")
            if cfg.auto_fill:
                click.echo("  Auto-executing stop sells (simulation fund)…")
            else:
                click.echo("  Consider selling — run 'fund run' for updated recommendations.")
        if profits_hit:
            click.echo(f"  🎯 Take-profit triggered: {', '.join(t for t, *_ in profits_hit)}")
            if cfg.auto_fill:
                click.echo("  Auto-executing take-profit sells (simulation fund)…")
            else:
                click.echo("  Consider trimming — run 'fund run' for updated recommendations.")
        if not stops_hit and not profits_hit:
            click.echo("  All positions within thresholds.")

    # ── Auto-execute stops/profits for simulation fund ────────────────────────
    auto_sold: list[str] = []
    triggered = stops_hit + profits_hit
    if triggered and cfg.auto_fill:
        from fundmgr.engine.auto_fill import execute_paper_fills
        sell_actions = [
            {"ticker": t, "side": "sell", "target_weight_pct": 0, "sek_estimate": price}
            for t, _chg, _pct, price in triggered
        ]
        # notify_skips=False: check-stops runs every 15 min, so a closed-market
        # skip here must not spam Telegram (the weekly run path handles reminders).
        fill_log = execute_paper_fills(sell_actions, store, cfg, notify_skips=False)
        for line in fill_log:
            click.echo(f"  {line}")
        for ticker, *_ in triggered:
            store.clear_position_stop(ticker)
            auto_sold.append(ticker)

    # ── Telegram alert ────────────────────────────────────────────────────────
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if (stops_hit or profits_hit or warnings) and bot_token and chat_id:
        lines = ["<b>📉 Price Alert</b>"]
        for ticker, chg, stop_pct, price in stops_hit:
            note = " — <b>AUTO-SOLD</b>" if ticker in auto_sold else " — review &amp; sell"
            lines.append(f"🚨 <b>{ticker}</b> {chg:+.1f}% — STOP HIT (stop -{stop_pct:.0f}%)  live {price:.2f}{note}")
        for ticker, chg, tp_pct, price in profits_hit:
            note = " — <b>AUTO-SOLD</b>" if ticker in auto_sold else " — consider trimming"
            lines.append(f"🎯 <b>{ticker}</b> {chg:+.1f}% — TARGET HIT (+{tp_pct:.0f}%)  live {price:.2f}{note}")
        for ticker, chg, daily_chg, price, stop_pct in warnings:
            if stop_pct:
                remaining = stop_pct + chg  # e.g. stop=-15, chg=-10.6 → 4.4pp left
                lines.append(
                    f"⚠ <b>{ticker}</b> {daily_chg:+.1f}% today  ({chg:+.1f}% since entry)  "
                    f"stop -{stop_pct:.0f}% ({remaining:.1f}pp away)  live {price:.2f}"
                )
            else:
                lines.append(
                    f"⚠ <b>{ticker}</b> {daily_chg:+.1f}% today  ({chg:+.1f}% since entry)  live {price:.2f}"
                )
        if (stops_hit or profits_hit) and not auto_sold:
            lines.append("\nTrigger <code>/run</code> for updated recommendation.")
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

    tickers = get_enabled_tickers(cfg.universe_path)
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

    # Send Telegram notification — held positions only
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if held_triggers and bot_token and chat_id:
        lines = ["<b>⚡ News Trigger — held position</b>"]
        for t in held_triggers:
            label = t["sentiment_label"].upper()
            lines.append(f"🔴 <b>{t['ticker']}</b> [{label} {t['sentiment_score']:.2f}]")
            lines.append(f"  {t['headline'][:120]}")
        if auto_run:
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
    tickers = get_enabled_tickers(cfg.universe_path)
    click.echo(f"\n─── Universe ({len(tickers)} enabled tickers) ───────────────────────────────")
    click.echo(f"  {'Name':<30} {'Ticker':<15} {'Country':<8} {'Sector'}")
    click.echo(f"  {'─'*30} {'─'*15} {'─'*8} {'─'*20}")
    for t in tickers:
        click.echo(f"  {t.name:<30} {t.yahoo_ticker:<15} {t.country:<8} {t.sector}")
    click.echo()


@cli.command("score-runs")
def score_runs():
    """Score completed weekly runs by excess return vs benchmark (for DSPy training data)."""
    cfg, store = _get_store()
    scored = store.score_runs()
    if not scored:
        click.echo("No new runs to score (either too recent or already scored).")
        return
    click.echo(f"\n  Scored {len(scored)} run(s):\n")
    click.echo(f"  {'Run ID':<30} {'Date':<12} {'Score':>10}  {'NAV start':>12} {'NAV end':>12}")
    click.echo(f"  {'─'*30} {'─'*12} {'─'*10}  {'─'*12} {'─'*12}")
    for r in scored:
        sign = "+" if r["score"] >= 0 else ""
        click.echo(
            f"  {r['run_id']:<30} {r['timestamp'][:10]:<12} "
            f"{sign}{r['score']*100:>8.3f}%  "
            f"{r['nav_start']:>12,.0f} {r['nav_end']:>12,.0f}"
        )
    click.echo()


@cli.command("export-dspy")
@click.option("--output", default="data/dspy_dataset.jsonl", show_default=True,
              help="Output path for the JSONL dataset")
@click.option("--score-first", is_flag=True, default=True, show_default=True,
              help="Run score-runs before exporting")
def export_dspy(output: str, score_first: bool):
    """Export scored runs to JSONL for DSPy/MIPRO prompt optimisation."""
    import os
    cfg, store = _get_store()
    if score_first:
        newly_scored = store.score_runs()
        if newly_scored:
            click.echo(f"  Scored {len(newly_scored)} new run(s) before export.")
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
    count = store.export_dspy(output)
    if count == 0:
        click.echo("No scored runs available yet — run 'fund score-runs' after at least one full week.")
    else:
        click.echo(f"  Exported {count} example(s) → {output}")


@cli.command("reject-rates")
def reject_rates():
    """Report malformed-sample and guardrail drop/clip rates (Refine-gate data)."""
    cfg, store = _get_store()
    s = store.get_rejection_stats()
    if s["runs"] == 0:
        click.echo("No runs logged yet — nothing to measure.")
        return

    click.echo(f"\n  Rejection rates across {s['runs']} run(s):\n")
    click.echo("  Malformed samples (the case Refine would retry)")
    click.echo(f"    samples: {s['samples_failed']}/{s['samples_requested']} failed "
               f"= {s['sample_failure_pct']}%")
    click.echo(f"    runs with ≥1 failed sample: {s['runs_with_any_failure']}/{s['runs']}")
    click.echo("\n  Guardrail verdicts (the case Refine could pre-empt)")
    click.echo(f"    rejected: {s['guardrail_rejected']}/{s['guardrail_verdicts']} "
               f"= {s['guardrail_reject_pct']}%")
    click.echo(f"    clipped:  {s['guardrail_clipped']}/{s['guardrail_verdicts']} "
               f"= {s['guardrail_clip_pct']}%")
    click.echo("\n  Gate: Refine earns its call-volume cost only if one of these "
               "rates is materially non-zero.\n")


if __name__ == "__main__":
    cli()
