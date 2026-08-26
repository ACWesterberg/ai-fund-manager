from __future__ import annotations

import html
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import click

from fundmgr.config import load_config, get_enabled_tickers
from fundmgr.data.benchmark import fetch_and_cache_benchmark, get_benchmark_return_pct
from fundmgr.data.fundamentals import apply_to_features, fetch_and_cache_fundamentals
from fundmgr.data.macro_context import build_macro_block, fetch_macro_headlines, fetch_macro_indicators
from fundmgr.data.news import attach_sentiment_to_features, check_news_triggers, fetch_news, score_and_cache_sentiment
from fundmgr.data.prices import build_all_features, fetch_and_cache_prices
from fundmgr.data.screener import screen
from fundmgr.data.universe_selection import (
    news_watch_tickers,
    select_tickers_for_price_fetch,
    tickers_for_feature_build,
)
from fundmgr.engine.client import LLMError, call_llm_consensus
from fundmgr.reporting.dashboard import format_text_report, generate_html_report
from fundmgr.engine.evaluator import evaluate_pending_outcomes, generate_learnings, generate_qualitative_learnings
from fundmgr.engine.prompt import build_prompt, snapshot_to_dict
from fundmgr.engine.thesis_check import verify_theses
from fundmgr.guardrails.rules import apply_guardrails
from fundmgr.levels import alertable_hits, merged_levels, record_sent_alerts
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
    # Every subcommand gets the environment, not just those that happen to load
    # a config. `paper-track` does not — it opens each book's store directly —
    # so under cron it ran for months with no OPENAI_API_KEY and skipped every
    # text-criterion watch, reporting the skip in a log nobody reads. The web
    # and bot processes were unaffected (systemd hands them the same file),
    # which is why the gap was invisible from the dashboard.
    from dotenv import load_dotenv

    from fundmgr.config import ROOT
    load_dotenv(ROOT / ".env")


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
    held_tickers = {p.ticker for p in store.get_positions()}
    fetch_tickers, fetch_note = select_tickers_for_price_fetch(tickers, held_tickers, cfg.screener)
    click.echo(f"\n{'═'*56}")
    click.echo(f"  AI Fund Manager — Weekly Run")
    click.echo(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    click.echo(f"{'═'*56}")

    # ── Step 1: Fetch prices ──────────────────────────────────────────────────
    click.echo(
        f"\n[1/6] Fetching prices for {len(fetch_tickers)}/{len(tickers)} tickers "
        f"({fetch_note})…"
    )
    fetch_result = fetch_and_cache_prices(fetch_tickers, store, cfg.data.lookback_days, force_refresh)
    ok = sum(1 for v in fetch_result.values() if v)
    failed = [sym for sym, v in fetch_result.items() if not v]
    click.echo(f"      {ok}/{len(tickers)} tickers resolved")
    if failed:
        click.echo(f"      ✗ Failed: {', '.join(failed)}")

    # ── Step 2: Fetch / refresh fundamentals cache (weekly TTL) ───────────────
    feature_tickers = tickers_for_feature_build(tickers, fetch_tickers, store)
    ticker_symbols = [t.yahoo_ticker for t in feature_tickers]
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

    # ── Step 4: Global macro context ─────────────────────────────────────────
    since_news = (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d")
    macro_block = ""
    if not skip_macro:
        click.echo(f"\n[4/6] Fetching global macro context…")
        macro_indicators = fetch_macro_indicators()
        macro_headlines = fetch_macro_headlines(cfg.data.macro_feeds) if cfg.data.macro_feeds else []
        macro_block = build_macro_block(macro_indicators, macro_headlines)
        ind_ok = sum(1 for i in macro_indicators if i.price is not None)
        click.echo(f"      {ind_ok}/{len(macro_indicators)} indicators | {len(macro_headlines)} global headlines")
    else:
        click.echo(f"\n[4/6] Global macro context skipped.")

    # ── Step 5: Compute features + pre-screen ─────────────────────────────────
    click.echo(f"\n[5/6] Computing features…")
    features = build_all_features(feature_tickers, store, cfg, fetch_result)
    apply_to_features(features, store)

    fund_count = sum(1 for f in features.values() if f.ev_to_ebitda is not None or f.revenue_growth_pct is not None)
    stale = [sym for sym, f in features.items() if f.is_stale]
    click.echo(
        f"      {len(features)} tickers with features from cache "
        f"({len(tickers):,} in universe)  ({fund_count} with fundamentals data)"
    )
    if stale:
        click.echo(f"      ⚠ Stale data (>{cfg.risk.stale_after_days}d): {', '.join(stale)}")

    pinned = set(cfg.screener.pinned_tickers)
    screened_features, screened_out = screen(
        features,
        held_tickers,
        top_n=cfg.screener.top_n,
        pinned_tickers=pinned,
    )
    if screened_out > 0:
        click.echo(f"      Screener: {len(screened_features)} candidates → LLM "
                   f"({screened_out} filtered out, held positions always included)")

    # ── Step 6: News + FinBERT — screener candidates only (not full universe) ─
    if not skip_news and cfg.data.news_feeds:
        news_symbols = set(screened_features.keys()) | held_tickers
        news_tickers = [t for t in tickers if t.yahoo_ticker in news_symbols]
        click.echo(
            f"\n[6/6] Fetching news for {len(news_tickers)} screener candidates "
            f"(from {len(cfg.data.news_feeds)} feeds)…"
        )
        ticker_news = fetch_news(cfg.data.news_feeds, news_tickers, max_age_hours=72)
        total_headlines = sum(len(v) for v in ticker_news.values())
        click.echo(f"      {total_headlines} matched headlines across {len(ticker_news)} tickers")
        if total_headlines > 0 and cfg.data.sentiment.enabled:
            click.echo(f"      Scoring with FinBERT ({cfg.data.sentiment.model})…")
            score_and_cache_sentiment(
                ticker_news, store, cfg.data.sentiment.model, cfg.data.sentiment.device
            )
    else:
        click.echo(f"\n[6/6] News skipped.")

    attach_sentiment_to_features(features, store, since_date=since_news)

    # ── Data quality summary ──────────────────────────────────────────────────
    click.echo(f"\n{'─'*56}")
    click.echo(f"  Data Quality Report")
    click.echo(f"{'─'*56}")
    _print_feature_table(features, cfg)

    # ── Step 5: Retrospective evaluation + learnings ─────────────────────────
    horizon = cfg.evaluation_horizon_days
    evaluated = evaluate_pending_outcomes(store, lookback_days=horizon)
    if evaluated:
        # Judge the reasoning before distilling: a lesson needs to know whether a
        # position that beat did so because the thesis held or in spite of it.
        funnel = verify_theses(store, evaluated, cfg, lookback_days=horizon)
        verdicts = ", ".join(f"{n} {v}" for v, n in sorted(funnel["verdicts"].items()))
        click.echo(
            f"\n[*] Thesis check: {funnel['outcomes']} outcome(s) → "
            f"{funnel['with_thesis']} with a thesis → "
            f"{funnel['with_evidence']} with news to judge → "
            f"{verdicts or 'no verdicts'}"
        )
        stat_learnings = generate_learnings(store, horizon_days=horizon)
        qual_learnings = generate_qualitative_learnings(store, evaluated, cfg)
        total_learnings = len(stat_learnings) + len(qual_learnings)
        click.echo(
            f"\n[*] Evaluated {len(evaluated)} past decisions; "
            f"{total_learnings} new learnings generated "
            f"({len(qual_learnings)} qualitative, {len(stat_learnings)} calibration)."
        )

    # ── Step 6: Build portfolio snapshot (attach live prices, native→SEK) ────
    from fundmgr.data.fx import rate_to_sek
    fx_cache: dict[str, float] = {}
    cur_by_ticker = {t.yahoo_ticker: t.currency for t in tickers}
    positions = store.get_positions()
    for p in positions:
        feat = features.get(p.ticker)
        if feat:
            price = feat.last_price
            cur = cur_by_ticker.get(p.ticker, "SEK")
            if cfg.fx_to_sek and cur != "SEK":  # convert native market price to SEK
                rate = fx_cache.get(cur) or rate_to_sek(cur, store) or 1.0
                fx_cache[cur] = rate
                price *= rate
            p.current_price_sek = price
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
        decision_prices = {
            a.ticker: features[a.ticker].last_price
            for a in guardrail_result.approved_actions
            if a.ticker in features and features[a.ticker].last_price
        }
        store.seed_outcomes_for_run(
            run_id,
            json.dumps([a.model_dump() for a in guardrail_result.approved_actions]),
            prices=decision_prices,
        )
        click.echo(f"      Recommendation saved (run_id: {run_id})")

        # Persist stop/take-profit levels per position so check-stops survives
        # multiple runs.
        existing_levels = store.get_effective_stops()
        for action in guardrail_result.approved_actions:
            merged = merged_levels(action, existing_levels.get(action.ticker))
            if merged:
                store.set_position_stop(
                    action.ticker,
                    stop_pct=merged[0],
                    take_profit_pct=merged[1],
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
            header = f"<b>{html.escape(cfg.display_name)}</b>\n📊 Run Complete  <code>{run_id}</code>"
            if tg_consensus:
                header += f"  <i>({n_samples}-run consensus)</i>"
            lines = [header]
            lines.append(html.escape(decision.market_summary))
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
                    lines.append(f"  <i>{html.escape(a.thesis)}</i>")
            if sells:
                lines.append("")
                lines.append("<b>🔴 SELLS</b>")
                for a in sells:
                    lines.append(f"  {a.ticker}  → {a.target_weight_pct:.0f}%  conf {a.confidence:.2f}{tg_vote(a.ticker)}")
                    lines.append(f"  <i>{html.escape(a.thesis)}</i>")
            if holds:
                lines.append("")
                lines.append("<b>⏸ HOLDS</b>")
                for a in holds:
                    lines.append(f"  {a.ticker}{tg_vote(a.ticker)}  <i>{html.escape(a.thesis[:120])}</i>")
            if not buys and not sells:
                lines.append("No trades this run — holding cash.")
            if decision.notes:
                lines.append(f"\n<i>{html.escape(decision.notes)}</i>")
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
            def _send_chunk(chunk: str) -> None:
                """Send as HTML; on a parse error (HTTP 400) retry as plain text
                so a stray character can never swallow the whole notification."""
                for mode in ("HTML", None):
                    payload = {"chat_id": chat_id, "text": chunk}
                    if mode:
                        payload["parse_mode"] = mode
                    try:
                        _req.urlopen(
                            f"https://api.telegram.org/bot{bot_token}/sendMessage",
                            urllib.parse.urlencode(payload).encode(), timeout=10,
                        )
                        return
                    except Exception:
                        if mode is None:
                            raise  # plain text also failed — genuine send error

            try:
                for chunk in chunks:
                    _send_chunk(chunk)
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


@cli.command("set-cash")
@click.argument("amount", type=float)
def set_cash_cmd(amount: float):
    """Set the cash balance to AMOUNT (SEK). For manual corrections."""
    cfg, store = _get_store()
    if amount < 0:
        click.echo("Amount must be >= 0.", err=True)
        sys.exit(1)
    old = store.get_cash()
    store.set_cash(amount)
    click.echo(f"✓ Cash set: {old:,.2f} → {amount:,.2f} SEK")


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

    stop_map = store.get_effective_stops()

    # Native→SEK for comparing live prices against the SEK cost basis.
    from fundmgr.data.fx import rate_to_sek
    cur_by_ticker = {t.yahoo_ticker: t.currency for t in get_enabled_tickers(cfg.universe_path)}
    fx_cache: dict[str, float] = {}

    def _to_sek(price: float, ticker: str) -> float:
        cur = cur_by_ticker.get(ticker, "SEK")
        if not cfg.fx_to_sek or cur == "SEK":
            return price
        rate = fx_cache.get(cur) or rate_to_sek(cur, store) or 1.0
        fx_cache[cur] = rate
        return price * rate

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

        live_native = float(hist["Close"].iloc[-1])
        live_price = _to_sek(live_native, p.ticker)  # SEK, for cost-basis comparison & display

        # % change since entry — the metric stop/take-profit levels are measured against.
        chg = (live_price / p.avg_cost_sek - 1) * 100 if p.avg_cost_sek else 0.0

        # Only compute daily change if today's bar is actually present —
        # at market open yfinance still shows yesterday's close as last_price,
        # which would make the "daily" change show yesterday's move instead
        daily_chg = None
        if len(hist) >= 2:
            last_bar_date = hist.index[-1].date()
            if last_bar_date == today_date:
                prev_day_close = float(hist["Close"].iloc[-2])
                daily_chg = (live_native / prev_day_close - 1) * 100  # native ratio
        levels = stop_map.get(p.ticker, {})
        stop_pct = levels.get("stop_pct")
        tp_pct = levels.get("take_profit_pct")

        parts = []
        if stop_pct:
            parts.append(f"-{stop_pct:.0f}%")
        if tp_pct:
            parts.append(f"+{tp_pct:.0f}%")
        levels_str = " / ".join(parts) if parts else "n/a"

        # These lists drive auto-sell and the stop review as well as the alert,
        # so every breach is recorded on every cycle. Rate-limiting happens at
        # the Telegram step alone — suppressing detection here would mean a
        # stop-loss whose auto-sell failed never got retried.
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

    # ── Stop-loss review (advisory) for non-auto-fill (real-money) funds ───────
    # On a stop hit, run a focused N-sample reassessment so a recent "add" thesis
    # gets weighed before deciding to dip out. Once per ticker per day (this
    # command runs every 15 min) to avoid repeat LLM calls + Telegram spam.
    review_snippets: list[str] = []
    if stops_hit and not cfg.auto_fill:
        from fundmgr.engine.stop_review import review_position, format_review_html
        for ticker, _chg, _stop, live in stops_hit:
            meta_key = f"stop_review:{ticker}:{today}"
            if store.get_meta(meta_key):
                continue
            try:
                result = review_position(ticker, store, cfg, live_price=live)
            except Exception as e:
                click.echo(f"  ⚠ stop review failed for {ticker}: {e}", err=True)
                continue
            if result is None:
                continue
            consensus, votes = result
            store.set_meta(meta_key, today)
            review_snippets.append(format_review_html(consensus, votes, max(1, cfg.llm.n_samples)))
            click.echo(
                f"  Stop review {ticker}: {consensus.recommendation.upper()} "
                f"(conf {consensus.confidence:.2f})"
            )

    # ── Take-profit review ────────────────────────────────────────────────────
    # Symmetric to the stop review above: reaching a target is a decision, not a
    # notification. The review banks the gain, trims, or raises the target — and
    # a raise is written straight back, because the whole failure mode here was
    # a stale target re-alerting with nothing able to move it.
    reviewed_targets: set[str] = set()
    if profits_hit and not cfg.auto_fill:
        from fundmgr.engine.target_review import (
            apply_new_target,
            review_position as review_target,
            format_review_html as format_target_html,
        )
        for ticker, _chg, _tp, live in profits_hit:
            meta_key = f"target_review:{ticker}:{today}"
            if store.get_meta(meta_key):
                continue
            try:
                result = review_target(ticker, store, cfg, live_price=live)
            except Exception as e:
                click.echo(f"  ⚠ target review failed for {ticker}: {e}", err=True)
                continue
            if result is None:
                continue
            consensus, votes = result
            store.set_meta(meta_key, today)
            reviewed_targets.add(ticker)
            moved = apply_new_target(consensus, store)
            review_snippets.append(
                format_target_html(consensus, votes, max(1, cfg.llm.n_samples), moved)
            )
            moved_str = f"; target {moved[0]:+.0f}% → {moved[1]:+.0f}%" if moved else ""
            click.echo(
                f"  Target review {ticker}: {consensus.recommendation.upper()} "
                f"(conf {consensus.confidence:.2f}{moved_str})"
            )

    # ── Telegram alert ────────────────────────────────────────────────────────
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    alert_stops   = alertable_hits(stops_hit, "stop", auto_sold, store, today) if bot_token and chat_id else []
    alert_profits = alertable_hits(profits_hit, "target", auto_sold, store, today) if bot_token and chat_id else []

    if (alert_stops or alert_profits or warnings) and bot_token and chat_id:
        lines = [f"<b>{cfg.display_name}</b>\n📉 Price Alert"]
        for ticker, chg, stop_pct, price in alert_stops:
            note = " — <b>AUTO-SOLD</b>" if ticker in auto_sold else " — review &amp; sell"
            lines.append(f"🚨 <b>{ticker}</b> {chg:+.1f}% — STOP HIT (stop -{stop_pct:.0f}%)  live {price:.2f}{note}")
        for ticker, chg, tp_pct, price in alert_profits:
            if ticker in auto_sold:
                note = " — <b>AUTO-SOLD</b>"
            elif ticker in reviewed_targets:
                note = ""  # the review below carries the call — don't pre-empt it
            else:
                note = " — consider trimming"
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
        for snip in review_snippets:
            lines.append("")
            lines.append(snip)
        if (alert_stops or alert_profits) and not auto_sold and not review_snippets:
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
            # Only now is the day's alert spent — a send that never landed must
            # be retried on the next cycle, not silently swallowed.
            record_sent_alerts(alert_stops, "stop", store, today)
            record_sent_alerts(alert_profits, "target", store, today)
            click.echo("  Telegram alert sent.")
        except Exception as e:
            click.echo(f"  ⚠ Telegram send failed: {e}", err=True)


@cli.command("review-stop")
@click.argument("ticker", required=False)
@click.option("--notify/--no-notify", default=True, show_default=True,
              help="Send the result to Telegram. Disable when a relay (e.g. the bot) already forwards stdout.")
def review_stop(ticker: str | None, notify: bool):
    """Advisory stop-loss review (EXIT/TRIM/HOLD/ADD).

    With a TICKER: review that held position (the Yahoo suffix is optional —
    `evo` resolves to `EVO.ST` if unambiguous). With no argument: scan all
    holdings and review every position currently below its stop-loss.
    """
    from fundmgr.engine.client import LLMError
    from fundmgr.engine.stop_review import (
        review_position, find_stop_breaches, format_review_text, format_review_html,
    )
    from fundmgr.notify.send import send_telegram

    cfg, store = _get_store()
    n = max(1, cfg.llm.n_samples)
    held = [p.ticker for p in store.get_positions()]

    # Decide which tickers to review.
    if ticker:
        q = ticker.upper()
        # Exact, then suffix-insensitive (EVO → EVO.ST), then prefix.
        matches = [h for h in held if h == q] or \
                  [h for h in held if h.split(".")[0] == q] or \
                  [h for h in held if h.startswith(q + ".")]
        matches = sorted(set(matches))
        if not matches:
            click.echo(f"{q} is not a current holding. Held: {', '.join(held) or '(none)'}", err=True)
            sys.exit(1)
        if len(matches) > 1:
            click.echo(f"{q} is ambiguous — matches {', '.join(matches)}. Use the full ticker.", err=True)
            sys.exit(1)
        targets = [{"ticker": matches[0], "live": None}]
    else:
        click.echo("Scanning holdings for stop-loss breaches…")
        scan = find_stop_breaches(store, cfg)
        breaches, skipped = scan["breaches"], scan["skipped"]
        if skipped:
            click.echo("  Could not evaluate: " + ", ".join(
                f"{s['ticker']} ({s['reason']})" for s in skipped))
        if not breaches:
            msg = "No positions are currently below their stop-loss."
            if skipped:
                msg += f" ({len(skipped)} could not be evaluated — see above.)"
            click.echo(msg)
            if notify and skipped:
                send_telegram(
                    "<b>📉 Stop scan</b>\nNo breaches found, but couldn't evaluate:\n"
                    + "\n".join(f"  {s['ticker']} — {s['reason']}" for s in skipped)
                )
            return
        click.echo(f"  {len(breaches)} below stop: {', '.join(b['ticker'] for b in breaches)}")
        targets = breaches

    html_blocks: list[str] = []
    for t in targets:
        tk = t["ticker"]
        click.echo(f"\nReviewing {tk} ({n}-sample consensus)…")
        try:
            result = review_position(tk, store, cfg, live_price=t.get("live"))
        except LLMError as e:
            click.echo(f"  Review failed for {tk}: {e}", err=True)
            continue
        if result is None:
            continue
        consensus, votes = result
        click.echo(format_review_text(consensus, votes, n))
        html_blocks.append(format_review_html(consensus, votes, n))

    # Only push to Telegram directly when no relay is forwarding our stdout.
    if notify and html_blocks:
        body = "<b>📉 Stop Review (manual)</b>\n\n" + "\n\n".join(html_blocks)
        if send_telegram(body):
            click.echo("\n  (sent to Telegram)")


@cli.command("review-target")
@click.argument("ticker", required=False)
@click.option("--notify/--no-notify", default=True, show_default=True,
              help="Send the result to Telegram. Disable when a relay (e.g. the bot) already forwards stdout.")
@click.option("--apply/--no-apply", default=True, show_default=True,
              help="Persist a raised take-profit level. --no-apply reviews without changing anything.")
def review_target_cmd(ticker: str | None, notify: bool, apply: bool):
    """Take-profit review (SELL/TRIM/RAISE/HOLD) for a position at its target.

    With a TICKER: review that held position. With no argument: review every
    holding currently at or above its take-profit level. A RAISE or TRIM writes
    the new target back unless --no-apply is given.
    """
    from fundmgr.engine.client import LLMError
    from fundmgr.engine.target_review import (
        apply_new_target, find_target_hits, review_position as review_target,
        format_review_text as fmt_text, format_review_html as fmt_html,
    )
    from fundmgr.notify.send import send_telegram

    cfg, store = _get_store()
    n = max(1, cfg.llm.n_samples)
    held = [p.ticker for p in store.get_positions()]

    if ticker:
        q = ticker.upper()
        matches = [h for h in held if h == q] or \
                  [h for h in held if h.split(".")[0] == q] or \
                  [h for h in held if h.startswith(q + ".")]
        matches = sorted(set(matches))
        if not matches:
            click.echo(f"{q} is not a current holding. Held: {', '.join(held) or '(none)'}", err=True)
            sys.exit(1)
        if len(matches) > 1:
            click.echo(f"{q} is ambiguous — matches {', '.join(matches)}. Use the full ticker.", err=True)
            sys.exit(1)
        targets = [{"ticker": matches[0], "live": None}]
    else:
        click.echo("Scanning holdings for positions at their take-profit target…")
        hits = find_target_hits(store, cfg)
        if hits["skipped"]:
            click.echo("  Could not evaluate: " + ", ".join(
                f"{s['ticker']} ({s['reason']})" for s in hits["skipped"]))
        if not hits["hits"]:
            click.echo("No positions are currently at their take-profit target.")
            return
        click.echo(f"  {len(hits['hits'])} at target: {', '.join(h['ticker'] for h in hits['hits'])}")
        targets = hits["hits"]

    html_blocks: list[str] = []
    for t in targets:
        tk = t["ticker"]
        click.echo(f"\nReviewing {tk} ({n}-sample consensus)…")
        try:
            result = review_target(tk, store, cfg, live_price=t.get("live"))
        except LLMError as e:
            click.echo(f"  Review failed for {tk}: {e}", err=True)
            continue
        if result is None:
            continue
        consensus, votes = result
        moved = apply_new_target(consensus, store) if apply else None
        click.echo(fmt_text(consensus, votes, n, moved))
        html_blocks.append(fmt_html(consensus, votes, n, moved))

    if notify and html_blocks:
        body = "<b>🎯 Take-Profit Review (manual)</b>\n\n" + "\n\n".join(html_blocks)
        if send_telegram(body):
            click.echo("\n  (sent to Telegram)")


@cli.command("decisions")
@click.option("--all", "show_all", is_flag=True,
              help="Include reviews already actioned, dismissed or superseded.")
@click.option("--done", "done_ticker", default=None, metavar="TICKER",
              help="Mark that name's open decision as actioned (you placed the order).")
@click.option("--dismiss", "dismiss_ticker", default=None, metavar="TICKER",
              help="Mark that name's open decision as not being acted on.")
def decisions(show_all: bool, done_ticker: str | None, dismiss_ticker: str | None):
    """Review verdicts still waiting on an order from you.

    A stop or take-profit review decides — EXIT, SELL, TRIM, ADD — and then needs
    a human to place the trade. This is the list of those that haven't been
    ticked off, the same one the dashboard shows.
    """
    from fundmgr.web.views import review_row

    cfg, store = _get_store()
    positions = {p.ticker: p for p in store.get_positions()}

    for ticker, status in ((done_ticker, "done"), (dismiss_ticker, "dismissed")):
        if not ticker:
            continue
        tk = ticker.upper()
        openrows = store.get_open_reviews()
        match = next(
            (r for r in openrows if r["ticker"] == tk
             or r["ticker"].split(".")[0] == tk
             or r["ticker"].startswith(tk + ".")),
            None,
        )
        if match is None:
            waiting = ", ".join(r["ticker"] for r in openrows) or "(none)"
            click.echo(f"No open decision for {tk}. Waiting: {waiting}", err=True)
            sys.exit(1)
        store.resolve_review(match["review_id"], status)
        click.echo(f"{match['ticker']}: {match['verdict'].upper()} marked {status}.")

    def _pos(ticker: str) -> dict | None:
        p = positions.get(ticker)
        # No price: a share count is the part that is certain offline, and a
        # SEK figure taken from cost basis would be wrong for a name at target.
        return None if p is None else {
            "ticker": p.ticker, "name": p.ticker, "shares": p.shares,
            "current_price": None,
        }

    raw = (store.get_recent_reviews(limit=20) if show_all
           else store.get_open_reviews(tickers=list(positions)))
    rows = [review_row(r, _pos(r["ticker"])) for r in raw]
    if not rows:
        click.echo("Nothing waiting — every review verdict has been actioned.")
        return

    click.echo(f"\n─── {cfg.display_name} — decisions ─────────────────────────")
    for r in rows:
        tail = f"  [{r['status']}]" if show_all else ""
        click.echo(f"\n  {r['verdict_label']}  {r['ticker']}  ({r['source_label']}, {r['when']}){tail}")
        click.echo(f"    {r['instruction']}")
        if r["follow_up"]:
            click.echo(f"    {r['follow_up']}")
        if r["target_line"]:
            click.echo(f"    {r['target_line']}")
        if r["rationale"]:
            click.echo(f"    Why: {r['rationale']}")
    if not show_all:
        click.echo("\n  Tick one off with: fund decisions --done TICKER\n")


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
    news_tickers = news_watch_tickers(tickers, held_tickers, cfg.screener)

    click.echo(f"\n[check-news] Scanning {len(cfg.data.news_feeds)} feeds "
               f"(last {max_age_hours}h, {len(news_tickers)} tickers: "
               f"{len(held_tickers)} held + pinned)…")

    triggers = check_news_triggers(
        cfg.data.news_feeds,
        news_tickers,
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


def _parse_holdings_snapshot(path: str) -> tuple[dict[str, tuple[float, float | None]], float | None]:
    """Parse a broker snapshot file → ({ticker: (shares, avg_cost|None)}, cash|None).

    One entry per line: 'TICKER SHARES [AVG_COST]', or 'CASH AMOUNT' for the
    balance. Blank lines and '#' comments ignored; commas are thousands seps.
    """
    holdings: dict[str, tuple[float, float | None]] = {}
    cash: float | None = None
    for lineno, raw in enumerate(open(path).read().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        key = parts[0].upper()
        try:
            if key == "CASH":
                cash = float(parts[1].replace(",", ""))
            else:
                shares = float(parts[1].replace(",", ""))
                avg = float(parts[2].replace(",", "")) if len(parts) > 2 else None
                holdings[key] = (shares, avg)
        except (IndexError, ValueError):
            raise click.ClickException(f"{path}:{lineno}: cannot parse '{raw.strip()}'")
    return holdings, cash


@cli.command()
@click.option("--holdings", "holdings_path", type=click.Path(exists=True, dir_okay=False),
              default=None, metavar="PATH",
              help="Broker snapshot: lines 'TICKER SHARES [AVG_COST]' and optional 'CASH AMOUNT'.")
@click.option("--cash", "cash_actual", type=float, default=None,
              help="Actual broker cash (SEK). Reconciles cash alone, or overrides the file's CASH line.")
@click.option("--full", is_flag=True,
              help="Snapshot is the COMPLETE holdings list — book tickers missing from it are zeroed.")
@click.option("--apply", "do_apply", is_flag=True, help="Write the corrections (default is a dry-run report).")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt when applying.")
def reconcile(holdings_path: str | None, cash_actual: float | None,
              full: bool, do_apply: bool, yes: bool):
    """Diff the book against the broker's real holdings + cash and flag drift.

    There is no live broker feed, so you supply the truth: read your broker
    statement into a --holdings file and/or pass --cash. This catches dividends,
    splits, and un-logged fills in one pass. Dry-run by default; --apply syncs
    the book (positions + cash) to the snapshot.
    """
    cfg, store = _get_store()
    if not store.is_initialised():
        click.echo("Portfolio not initialised. Run 'fund init' first.", err=True)
        sys.exit(1)
    if holdings_path is None and cash_actual is None:
        click.echo("Nothing to reconcile — pass --holdings and/or --cash.", err=True)
        sys.exit(1)

    actual_holdings: dict[str, tuple[float, float | None]] = {}
    file_cash: float | None = None
    if holdings_path:
        actual_holdings, file_cash = _parse_holdings_snapshot(holdings_path)
    actual_cash = cash_actual if cash_actual is not None else file_cash

    book = {p.ticker: p for p in store.get_positions()}
    book_cash = store.get_cash()

    # Which tickers to reconcile: everything in the snapshot; book tickers only
    # if the snapshot is declared complete (--full), else leave them untouched.
    reconcile_shares = holdings_path is not None
    tickers = set(actual_holdings)
    if full:
        tickers |= set(book)

    rows: list[tuple[str, float, float | None, float | None]] = []  # ticker, book, actual, avg
    for t in sorted(tickers):
        book_shares = book[t].shares if t in book else 0.0
        if t in actual_holdings:
            actual_shares, avg = actual_holdings[t]
        elif full:  # in book, missing from a complete snapshot → zeroed
            actual_shares, avg = 0.0, None
        else:
            continue
        rows.append((t, book_shares, actual_shares, avg))

    drift_rows = [r for r in rows if abs(r[1] - (r[2] or 0.0)) > 1e-6]
    cash_drift = actual_cash is not None and abs(actual_cash - book_cash) > 0.005

    click.echo("\n─── Reconciliation ─────────────────────────────────")
    if reconcile_shares:
        if drift_rows:
            click.echo(f"  {'Ticker':<16} {'Book':>10} {'Broker':>10} {'Drift':>10}")
            click.echo(f"  {'─'*16} {'─'*10} {'─'*10} {'─'*10}")
            for t, bshares, ashares, _avg in drift_rows:
                delta = (ashares or 0.0) - bshares
                tag = "  NEW" if t not in book else ("  GONE" if (ashares or 0.0) == 0 else "")
                click.echo(f"  {t:<16} {bshares:>10.2f} {(ashares or 0.0):>10.2f} {delta:>+10.2f}{tag}")
        else:
            click.echo("  Positions: in sync ✓")
        if not full and set(book) - set(actual_holdings):
            untouched = ", ".join(sorted(set(book) - set(actual_holdings)))
            click.echo(f"  (not in snapshot, left untouched: {untouched} — use --full to zero them)")

    if actual_cash is not None:
        if cash_drift:
            click.echo(f"\n  Cash:  book {book_cash:,.2f} → broker {actual_cash:,.2f} "
                       f"({actual_cash - book_cash:+,.2f} SEK)")
        else:
            click.echo("\n  Cash: in sync ✓")

    if not drift_rows and not cash_drift:
        click.echo("\n  ✓ Book matches the broker — nothing to apply.")
        return

    if not do_apply:
        click.echo("\n  Dry run — re-run with --apply to write these corrections.")
        return

    # New tickers with no cost basis can't be added without inventing a basis.
    unpriced = [r[0] for r in drift_rows if r[0] not in book and r[3] is None and (r[2] or 0.0) > 0]
    if unpriced:
        click.echo(f"\n  ⚠ New holdings with no AVG_COST given: {', '.join(unpriced)}")
        click.echo("    Add a third column (cost basis) or record them with 'fund fill'; skipping.")

    if not yes and not click.confirm("\n  Apply these corrections to the book?"):
        click.echo("  Aborted — nothing written.")
        return

    for t, bshares, ashares, avg in drift_rows:
        ashares = ashares or 0.0
        if t not in book and avg is None and ashares > 0:
            continue  # unpriced new holding — skipped above
        new_avg = avg if avg is not None else (book[t].avg_cost_sek if t in book else 0.0)
        store.upsert_position(t, ashares, new_avg)
        click.echo(f"  ✓ {t}: {bshares:.2f} → {ashares:.2f} shares")
    if cash_drift:
        store.set_cash(actual_cash)
        click.echo(f"  ✓ Cash set to {actual_cash:,.2f} SEK")

    # Record a cost-basis NAV snapshot so the chart reflects the reconciled book.
    try:
        bench_rows = store.get_benchmark()
        bench_val = bench_rows[-1]["close"] if bench_rows else 0.0
        positions_after = store.get_positions()
        cash_after = store.get_cash()
        nav_cost = sum(p.shares * p.avg_cost_sek for p in positions_after) + cash_after
        store.upsert_nav(NavPoint(
            date=datetime.utcnow().strftime("%Y-%m-%d"),
            portfolio_nav_sek=nav_cost,
            benchmark_value=bench_val,
            cash_sek=cash_after,
        ))
    except Exception:
        pass
    click.echo("\n  Reconciliation applied.")


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

    # The two channels that shape every prompt without ever being measured.
    for key, label in (("learnings_hash", "learnings"), ("guidance_hash", "MIPRO guidance")):
        buckets = store.score_by_regime(key)
        if len(buckets) < 2:
            continue  # only one regime seen — nothing to compare against yet
        click.echo(f"\n  Mean score by {label} regime:")
        for b in buckets:
            name = b["value"] or "(none — unguided)"
            click.echo(f"    {name:<22} {b['runs']:>3} run(s)  {b['mean_score']*100:>+8.3f}%")
        click.echo("    (descriptive only — weekly runs are few and not randomised)")
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


@cli.command()
@click.option("--min-outcomes", type=int, default=None,
              help="Override the evaluated-outcome threshold from config")
@click.option("--dry-run", is_flag=True, help="Report trainset stats without running MIPRO")
def optimize(min_outcomes: int | None, dry_run: bool):
    """Optimize the decision prompt from evaluated outcomes (DSPy MIPROv2).

    Builds one training example per past run whose 28-day per-ticker outcomes
    vs the benchmark are known, searches instruction space for the
    WeeklyDecision signature with an alpha-weighted metric, and saves the
    winning instructions as guidance injected into every future 'fund run'.
    """
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg, store = _get_store()
    from fundmgr.engine.optimizer import build_pooled_trainset, guidance_path, run_optimization

    threshold = min_outcomes if min_outcomes is not None else cfg.optimizer.min_outcomes
    evaluated = store.get_evaluated_outcomes()
    examples = build_pooled_trainset(cfg)
    resolved = sum(
        1 for e in examples for v in (e.get("ticker_theses") or {}).values()
        if v in ("held", "broke")
    )

    click.echo("\n─── Prompt Optimizer ───────────────────────────────")
    click.echo(f"  Evaluated outcomes:   {len(evaluated)} (need {threshold})")
    click.echo(f"  Usable run examples:  {len(examples)} (need {cfg.optimizer.min_examples})")
    if cfg.optimizer.pool_configs:
        by_source: dict[str, int] = {}
        for e in examples:
            by_source[e.get("source") or "?"] = by_source.get(e.get("source") or "?", 0) + 1
        click.echo("    pooled from:        " + ", ".join(
            f"{src} ({n})" for src, n in sorted(by_source.items())
        ))
    click.echo(f"  Resolved theses:      {resolved} (metric signal beyond raw alpha)")
    click.echo(f"  Guidance artifact:    {guidance_path(cfg)}")

    if dry_run:
        click.echo("  (dry run — MIPRO not executed)")
        return

    if run_optimization(cfg, store, min_outcomes=min_outcomes):
        click.echo("\n  ✓ New guidance saved — it will be injected into the next 'fund run'.")
    else:
        click.echo("\n  No new guidance produced (threshold not met or optimization failed).")


@cli.command("repair-outcomes")
@click.option("--dry-run", is_flag=True, help="Report what would change without writing")
@click.option("--deactivate-learnings", is_flag=True,
              help="Also deactivate all active learnings (they were distilled from the corrupted returns)")
def repair_outcomes_cmd(dry_run: bool, deactivate_learnings: bool):
    """Repair decision outcomes poisoned by the price_at_decision seeding bug.

    Historical rows stored the trade's SEK estimate as the entry price (and
    measured the benchmark from the start of the cache), so every evaluated
    return was wrong. This re-derives the true entry price from each run's
    stored prompt snapshot and recomputes the returns over the correct window.
    """
    from fundmgr.engine.evaluator import repair_outcomes

    cfg, store = _get_store()
    stats = repair_outcomes(store, dry_run=dry_run)

    mode = "DRY RUN — nothing written" if dry_run else "applied"
    click.echo(f"\n─── Outcome Repair ({mode}) ─────────────────────────")
    click.echo(f"  Rows checked:              {stats['checked']}")
    click.echo(f"  Entry prices fixed:        {stats['price_fixed']}")
    click.echo(f"  Evaluations recomputed:    {stats['recomputed']}")
    click.echo(f"  Reset to pending:          {stats['reset_pending']}")
    click.echo(f"  Unrecoverable (cleared):   {stats['unrecoverable']}")

    if deactivate_learnings and not dry_run:
        n = store.deactivate_all_learnings()
        click.echo(f"  Learnings deactivated:     {n}")
    elif stats["price_fixed"] or stats["recomputed"]:
        click.echo("\n  ⚠ Existing learnings were distilled from the corrupted returns —")
        click.echo("    consider re-running with --deactivate-learnings.")


@cli.command("thesis-report")
def thesis_report():
    """Did the reasoning hold, and did the price agree? The four-cell readout."""
    cfg, store = _get_store()
    stats = store.get_thesis_stats()

    cov = store.get_thesis_coverage()
    click.echo(f"\n─── Thesis vs outcome — {cfg.display_name} ───────────")
    click.echo(f"\n  Coverage: {cov['evaluated']} evaluated outcome(s)")
    click.echo(f"    no thesis recorded:      {cov['no_thesis']}")
    click.echo(f"    thesis, no news to judge: {cov['no_evidence']}")
    click.echo(f"    audited:                 {cov['audited']}")
    if cov["no_evidence"] > cov["audited"]:
        click.echo("    ⚠ most theses are lost to missing news, not to being")
        click.echo("      unfalsifiable — widen what the news fetch covers.")

    if not stats["n"]:
        click.echo("  No checked theses yet. Verdicts are written when outcomes")
        click.echo("  mature (~4 weeks after each run).")
        return

    click.echo(f"\n  {'':<12} {'beat':>8} {'lagged':>8}")
    click.echo(f"  {'─'*12} {'─'*8} {'─'*8}")
    for verdict in ("held", "broke", "unresolved"):
        cell = stats["cells"].get(verdict)
        if not cell:
            continue
        click.echo(f"  {verdict:<12} {cell['beat']:>8} {cell['lagged']:>8}")

    if stats["hold_rate"] is not None:
        click.echo(
            f"\n  Theses that held: {stats['hold_rate']:.0%} "
            f"of {stats['resolved']} resolved ({stats['n']} checked)"
        )
    lucky = (stats["cells"].get("broke") or {}).get("beat", 0)
    if lucky:
        click.echo(f"  ⚠ {lucky} position(s) beat the benchmark on a thesis that broke —")
        click.echo("    judged on return alone these are indistinguishable from skill.")
    click.echo()


@cli.command("prune-learnings")
@click.option("--category", default=None, metavar="NAME",
              help="Only this category (e.g. 'qualitative').")
@click.option("--before", default=None, metavar="YYYY-MM-DD",
              help="Only lessons created before this date.")
@click.option("--dry-run", is_flag=True, help="Show what would be retired, write nothing.")
@click.option("--all-books", is_flag=True,
              help="Every fund in config/ and every paper book, not just this one.")
def prune_learnings_cmd(category: str | None, before: str | None, dry_run: bool, all_books: bool):
    """Retire active learnings so they stop being injected into the prompt.

    For clearing out a batch that turned out to be noise — e.g. the per-trade
    lessons written before batch distillation, which explained a whole run's
    outcomes by the one macro line every trade in that run shared. Rows are
    deactivated, not deleted, so past runs remain reconstructible.

    Learnings are per-book: each fund and each paper portfolio distils from its
    own outcomes into its own database. Without --all-books this touches one
    book — the fund named by FUND_CONFIG, or the default config.
    """
    books = _all_learning_books() if all_books else [_current_learning_book()]

    scope = category or "all categories"
    window = f" created before {before}" if before else ""
    click.echo(f"\n─── Prune learnings ({scope}{window}) ───────────────")

    total_matched = total_retired = 0
    for label, store in books:
        matched = store.find_active_learnings(category=category, before=before)
        if not matched:
            if all_books:
                click.echo(f"\n  {label}: nothing matches.")
            continue

        total_matched += len(matched)
        click.echo(f"\n  {label}:")
        for lrn in matched:
            body = lrn.body if len(lrn.body) <= 88 else lrn.body[:85] + "..."
            click.echo(f"    [{lrn.created_at.strftime('%Y-%m-%d')}] {lrn.category}: {body}")

        if not dry_run:
            total_retired += store.deactivate_learnings(category=category, before=before)
            click.echo(f"    → {len(store.get_active_learnings())} active learning(s) remain.")

    if not total_matched:
        click.echo("\n  Nothing matches — no learnings retired.")
    elif dry_run:
        click.echo(f"\n  DRY RUN — {total_matched} would be retired across "
                   f"{len(books)} book(s), nothing written.")
    else:
        click.echo(f"\n  Retired {total_retired} across {len(books)} book(s).")


def _current_learning_book() -> tuple[str, Store]:
    cfg, store = _get_store()
    return cfg.display_name, store


def _all_learning_books() -> list[tuple[str, Store]]:
    """Every book that distils learnings: each config/*.yaml fund, each paper book.

    Both run the same learning pipeline against their own database, so a prune
    that only reached the fund selected by FUND_CONFIG would silently leave the
    other funds' lessons live in their prompts.
    """
    from fundmgr import paper
    from fundmgr.config import CONFIG_DIR, load_config

    books: list[tuple[str, Store]] = []
    seen: set[Path] = set()

    for path in sorted(CONFIG_DIR.glob("config*.yaml")):
        try:
            cfg = load_config(path)
        except Exception as exc:
            click.echo(f"  ⚠ skipping {path.name}: {exc}")
            continue
        if cfg.db_path in seen or not cfg.db_path.exists():
            continue
        seen.add(cfg.db_path)
        books.append((f"{cfg.display_name} [{path.name}]", Store(cfg.db_path)))

    for meta in paper.list_portfolios():
        try:
            _, store = paper.open_portfolio(meta["slug"])
        except Exception as exc:
            click.echo(f"  ⚠ skipping paper book {meta['slug']}: {exc}")
            continue
        books.append((f"{meta['name']} [paper/{meta['slug']}]", store))

    return books


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


@cli.command("paper-track")
@click.option("--slug", default=None, help="Track a single paper portfolio instead of all.")
def paper_track(slug: str | None):
    """Daily upkeep for the paper portfolios created on the web (/paper):

    refresh price + benchmark caches, snapshot NAV at live prices, evaluate
    matured decision outcomes and distil learnings — the same pipeline the
    LLM funds run, scoped to each portfolio's own store."""
    from fundmgr.paper import track_all, track_portfolio
    lines = track_portfolio(slug) if slug else track_all()
    for line in lines:
        click.echo(f"  {line}")


@cli.command("paper-kill-check")
@click.option("--slug", default=None, help="Check a single paper portfolio instead of all.")
def paper_kill_check(slug: str | None):
    """Check each paper portfolio's holdings against their pre-registered kill
    criteria using recent news (gpt-4o-mini judge; needs OPENAI_API_KEY).

    Hits are recorded and pushed to Telegram. This also runs automatically as
    part of 'fund paper-track'."""
    from fundmgr.paper import check_kill_criteria, list_portfolios
    slugs = [slug] if slug else [m["slug"] for m in list_portfolios()]
    if not slugs:
        click.echo("No paper portfolios yet.")
        return
    for s in slugs:
        for line in check_kill_criteria(s):
            click.echo(f"  {line}")


@cli.command("aliases")
@click.option("--set", "set_pair", nargs=2, default=None,
              metavar="KEY TICKER", help="Teach a mapping, e.g. --set SE0000115446 VOLV-B.ST")
@click.option("--forget", default=None, metavar="KEY", help="Remove a learned mapping.")
def aliases_cmd(set_pair, forget):
    """List, add or remove learned ticker aliases.

    Photo import records how every reviewed row resolved, so a ticker you fix
    once is applied automatically the next time that ISIN, broker ticker or
    company name appears — in any sleeve. KEY may be an ISIN, a broker ticker
    or a company name.
    """
    from fundmgr import aliases as alias_store

    if set_pair:
        key, ticker = set_pair
        kinds = alias_store.learn(
            ticker,
            isin=key if len(key.strip()) == 12 else None,
            raw_ticker=key, name=key, source="manual",
        )
        if not kinds:
            raise click.ClickException(f"Could not use '{key}' as an alias key.")
        click.echo(f"✓ {key} → {ticker.upper()}  (as {', '.join(kinds)})")
        return

    if forget:
        removed = alias_store.forget(forget)
        if removed:
            click.echo(f"✓ Forgot '{forget}' ({', '.join(removed)}).")
        else:
            click.echo(f"No learned alias matching '{forget}'.")
        return

    rows = alias_store.entries()
    if not rows:
        click.echo("No learned aliases yet — they're recorded as you import photos.")
        return
    click.echo(f"\n─── {len(rows)} learned alias(es) — {alias_store.alias_path()} ───")
    click.echo(f"  {'KIND':<8} {'KEY':<28} {'TICKER':<14} {'LEARNED'}")
    for r in rows:
        click.echo(f"  {r['kind']:<8} {r['key'][:27]:<28} {r['ticker']:<14} "
                   f"{r['learned_at'][:10]}")
    click.echo("\n  Remove one with:  fund aliases --forget KEY\n")


@cli.command("fundamentals-check")
@click.argument("ticker")
def fundamentals_check(ticker: str):
    """Show which fundamentals fields the data layer really returns for TICKER.

    Every metric the kill-criterion analyser can offer needs data behind it: a
    key that is always None shows up as a watched line on the dashboard but can
    never evaluate. yfinance `.info` is a passthrough of Yahoo's payload, so a
    mis-mapped key name fails silently — run this after changing financedata's
    _FIELD_MAP to confirm the new keys actually arrive.
    """
    tkr = ticker.upper()
    try:
        from financedata import get_fundamentals
    except ImportError as e:
        raise click.ClickException(f"financedata is not installed: {e}")
    from fundmgr.evidence import FUND_FIELD_META

    data = (get_fundamentals([tkr], ttl_days=0) or {}).get(tkr)
    if not data:
        raise click.ClickException(
            f"No fundamentals returned for {tkr} — bad ticker, or the fetch failed.")

    # A criterion reads the merged cache, not one provider, so the check has to
    # show both sources or it answers the wrong question. Statement metrics are
    # derived live here rather than read from a book's store — this command is
    # about what a ticker *can* yield, independent of any portfolio.
    from fundmgr.data.statements import STATEMENT_METRICS, derive
    try:
        statement_metrics = derive(tkr)
    except Exception as e:
        statement_metrics = {}
        click.echo(f"  ⚠ statement metrics unavailable: {e}")
    data.update({k: v for k, v in statement_metrics.items() if v is not None})

    offered = set(FUND_FIELD_META)
    populated = {k: v for k, v in sorted(data.items()) if v is not None}
    empty = sorted(k for k, v in data.items() if v is None)

    click.echo(f"\n─── {tkr}: {len(populated)}/{len(data)} fields populated ───")
    for key, value in populated.items():
        click.echo(f"  {'★' if key in offered else ' '} {key:<22} {value}")
    if empty:
        click.echo(f"\n  Returned but empty for this ticker:\n    {', '.join(empty)}")

    # Statement metrics get their own section: an absent one almost always means
    # a row label this codebase doesn't accept, which is silent everywhere else.
    if statement_metrics:
        # fiscal_period_end rides along in the same payload but is a period
        # label, not a metric — counting it would inflate the coverage figure.
        ratios = {k: v for k, v in statement_metrics.items() if k in STATEMENT_METRICS}
        derived_ok = [k for k, v in ratios.items() if v is not None]
        derived_missing = [k for k, v in ratios.items() if v is None]
        period = statement_metrics.get("fiscal_period_end")
        click.echo(f"\n  Derived from the financial statements "
                   f"({len(derived_ok)}/{len(STATEMENT_METRICS)}"
                   f"{f', period ending {period}' if period else ''}):")
        for key, (label, is_fraction) in STATEMENT_METRICS.items():
            value = statement_metrics.get(key)
            if value is None:
                if key == "cash_runway_months" and (data.get("operating_cash_flow") or 0) > 0:
                    shown = "—  (n/a — cash generative)"
                else:
                    shown = "—  (no matching statement row)"
            else:
                shown = f"{value * 100:,.1f}%" if is_fraction else f"{value:,.2f}"
            click.echo(f"    {'★' if key in offered else ' '} {key:<22} {shown}")
        unexplained = [k for k in derived_missing
                       if not (k == "cash_runway_months"
                               and (data.get("operating_cash_flow") or 0) > 0)]
        if unexplained:
            click.echo(f"    → {len(unexplained)} unexplained: either this filer "
                       "doesn't report them, or the row label needs adding.")

    missing = sorted(offered - set(data))
    if missing:
        click.echo(f"\n  ⚠ offered as kill-criterion metrics but NOT in the payload "
                   f"at all:\n    {', '.join(missing)}")
        click.echo("    → these can never evaluate; fix the mapping or drop them.")
    else:
        click.echo("\n  ✓ every offered kill-criterion metric is present in the payload.")

    # The other direction: a field the data layer now returns that no criterion
    # can yet be checked against — the cue to add it to evidence._FUND_FIELDS.
    spare = sorted(k for k in populated if k not in offered
                   and isinstance(populated[k], (int, float)))
    if spare:
        click.echo(f"\n  Available but not offered as a metric:\n    {', '.join(spare)}")
    click.echo("\n  ★ = offered to the criterion analyser as a checkable metric\n")


@cli.command("paper-watch")
@click.option("--slug", default=None, help="Check a single portfolio instead of all.")
def paper_watch(slug: str | None):
    """Check numeric kill lines and time horizons across the portfolios.

    Kill lines (max drawdown from the review anchor, price floor, price target) are compared
    with current prices; horizons alert at 30/14/7/1 days out and on the day.
    Neither needs an API key. Both also run as part of 'fund paper-track'."""
    from fundmgr.paper import list_portfolios
    from fundmgr.watchplan import check_horizons, check_kill_rules
    slugs = [slug] if slug else [m["slug"] for m in list_portfolios()]
    if not slugs:
        click.echo("No paper portfolios yet.")
        return
    quiet = True
    for s in slugs:
        for line in check_kill_rules(s) + check_horizons(s):
            click.echo(f"  {line}")
            quiet = False
    if quiet:
        click.echo("  No kill lines crossed, no horizons due.")


@cli.command("paper-plan")
@click.argument("slug")
@click.argument("ticker")
@click.option("--kill", default=None, help="Kill criterion text (empty string clears it).")
@click.option("--add", "add_criterion", default=None,
              help="ADD criterion — what must still be true to add (empty string clears it).")
@click.option("--max-drop", default=None, help="Max % drawdown from the review price before alerting.")
@click.option("--price-below", default=None, help="Alert when price falls to/below this (native currency).")
@click.option("--price-above", default=None, help="Alert when price rises to/above this (native currency).")
@click.option("--horizon", default=None, help="Review date, YYYY-MM-DD.")
@click.option("--months", default=None, help="…or a horizon this many months out.")
@click.option("--note", default=None, help="Note on what should be true by the horizon.")
@click.option("--re-anchor", is_flag=True,
              help="Measure the drawdown line from today's price (do this after a fresh analysis).")
@click.option("--clear", is_flag=True, help="Remove the whole watch plan for this ticker.")
def paper_plan(slug: str, ticker: str, kill: str | None, add_criterion: str | None,
               max_drop: str | None, price_below: str | None, price_above: str | None,
               horizon: str | None, months: str | None, note: str | None,
               re_anchor: bool, clear: bool):
    """Set a position's kill criteria and time horizon.

    The CLI twin of the dashboard's watch-plan editor:

        fund paper-plan my-sleeve NVDA --kill "loses the Apple socket" \\
            --max-drop 25 --months 12
    """
    from fundmgr import watchplan
    from fundmgr.paper import open_portfolio
    try:
        meta, store = open_portfolio(slug)
    except KeyError:
        raise click.ClickException(f"No portfolio '{slug}' — see 'fund paper-list'.")

    tkr = ticker.upper()
    if clear:
        watchplan.clear_position_plan(store, tkr)
        click.echo(f"✓ Cleared the watch plan for {tkr} in {meta['name']}.")
        return

    plan = watchplan.set_position_plan(
        store, tkr, kill_criterion=kill, max_drawdown_pct=max_drop,
        price_below=price_below, price_above=price_above,
        currency=meta["currency_map"].get(tkr), review_date=horizon,
        horizon_months=months, horizon_note=note, re_anchor=re_anchor,
    )
    # The ADD criterion goes through the same analyser as the kill criterion,
    # with the framing that reads its conditions as ones that must hold.
    if add_criterion is not None:
        previous = watchplan.get_add_text(store).get(tkr, "")
        text = watchplan.set_add_text(store, tkr, add_criterion)
        if text and text != previous:
            watchplan.save_add_analysis(
                store, tkr, watchplan.analyse_criterion(text, kind="add", store=store))
        elif not text:
            watchplan.save_add_analysis(store, tkr, None)

    click.echo(f"✓ {tkr} in {meta['name']}:")
    click.echo(f"  Kill criterion: {plan['kill_criterion'] or '—'}")
    click.echo(f"  ADD criterion:  {watchplan.get_add_text(store).get(tkr) or '—'}")
    rules = plan["kill_rules"]
    click.echo(f"  Max drop: {rules.get('max_drawdown_pct') or '—'}   "
               f"Floor: {rules.get('price_below') or '—'}   "
               f"Target: {rules.get('price_above') or '—'}")
    if rules.get("max_drawdown_pct"):
        anchor = rules.get("anchor_price_sek")
        click.echo(f"  Measured from: {anchor:,.2f} SEK on {rules.get('anchor_date', '')}"
                   if anchor else
                   "  Measured from: cost basis (no price cached when the line was set)")
    for f in rules.get("fundamentals") or []:
        span = (f" for {f['quarters']} straight quarters"
                if int(f.get("quarters") or 1) > 1 else "")
        click.echo(f"  Fundamental: {f['metric']} {f['op']} {f['value']:g}{span}")
    hz = plan["horizon"]
    if hz.get("review_date"):
        click.echo(f"  Horizon: {hz['review_date']} ({hz.get('label', '')}, "
                   f"{plan['days_left']}d away)")
    else:
        click.echo("  Horizon: —")


@cli.command("paper-add-plan")
@click.argument("slug")
@click.argument("ticker")
@click.option("--book", default=None, help="Risk book: A, A/B or B (sets the gates).")
@click.option("--max-weight", default=None, help="Ceiling weight % while an add thesis is active.")
@click.option("--tranche", default=None, help="Size of one add step, in weight points.")
@click.option("--target", default=None, help="Target review price (stamped with today's date).")
@click.option("--review-price", default=None,
              help="Anchor for price dislocation — the thesis-confirmed price. "
                   "Use 'live' to anchor at the current price.")
@click.option("--confirm-proof", is_flag=True, help="Record that the fundamental proof criterion is met.")
@click.option("--drop-proof", is_flag=True, help="Withdraw the proof confirmation.")
@click.option("--clear", is_flag=True, help="Remove the whole add plan for this ticker.")
def paper_add_plan(slug, ticker, book, max_weight, tranche, target, review_price,
                   confirm_proof, drop_proof, clear):
    """Set a position's add plan, then show the resulting signal.

    An add needs (proof OR dislocation) AND valuation AND weight AND no kill:

        fund paper-add-plan my-sleeve SYSR.ST --book A --max-weight 13 \\
            --tranche 1 --target 145 --review-price live
    """
    from fundmgr import addsignal
    from fundmgr.paper import open_portfolio
    try:
        meta, store = open_portfolio(slug)
    except KeyError:
        raise click.ClickException(f"No portfolio '{slug}' — see 'fund paper-list'.")

    tkr = ticker.upper()
    if clear:
        addsignal.clear_plan(store, tkr)
        click.echo(f"✓ Cleared the add plan for {tkr}.")
        return

    if review_price and str(review_price).strip().lower() == "live":
        from fundmgr.data.quotes import live_price
        from fundmgr.paper import to_sek_price
        native = live_price(tkr)
        if not native:
            raise click.ClickException(f"Could not fetch a live price for {tkr}.")
        # Dislocation is compared against SEK prices, so the anchor must be SEK.
        currency = meta["currency_map"].get(tkr, "SEK")
        review_price = (native if currency == "SEK"
                        else to_sek_price(native, currency, store))
        if not review_price:
            raise click.ClickException(f"No {currency}→SEK rate to anchor {tkr}.")
        click.echo(f"  Anchoring review price at {review_price:,.2f} SEK")

    addsignal.set_plan(
        store, tkr, book=book, max_weight_pct=max_weight, tranche_pct=tranche,
        target_price=target, review_price=review_price,
        proof_confirmed=True if confirm_proof else (False if drop_proof else None),
    )

    row = next((r for r in addsignal.evaluate_all(store) if r["ticker"] == tkr), None)
    if row is None:
        click.echo(f"✓ Saved. No signal computed for {tkr} yet.")
        return

    click.echo(f"\n─── {tkr} in {meta['name']} — {row['state'].upper()} ───")
    click.echo(f"  {row['why']}")
    click.echo(f"  Book {row['book']}  ·  proof {'yes' if row['proof'] else 'no'}"
               f"{' (' + row['proof_confirmed_at'] + ')' if row['proof_confirmed_at'] else ''}")
    if row["dislocation_pct"] is not None:
        click.echo(f"  Dislocation {row['dislocation_pct']:+.1f}% vs review "
                   f"{row['review_price']:,.2f} (gate {row['dislocation_gate']:+.0f}%)")
    click.echo(f"  Valuation {row['valuation_status']} — {row['valuation_reason']}")
    if row["expected_annual_return"] is not None:
        click.echo(f"  Expected {row['expected_annual_return']:.0f}%/yr over "
                   f"{row['months_left']:.0f}m (gate {row['return_gate']:.0f}%)")
    if row["max_weight_pct"]:
        click.echo(f"  Weight {row['weight_pct']:.1f}% of max {row['max_weight_pct']:.1f}% "
                   f"(tranche {row['tranche_pct']:.1f}pp)")
    click.echo("")


@cli.command("paper-adds")
@click.option("--slug", default=None, help="Check a single portfolio instead of all.")
def paper_adds(slug):
    """Show the current add signal for every position carrying an add plan."""
    from fundmgr import addsignal
    from fundmgr.paper import list_portfolios, open_portfolio
    slugs = [slug] if slug else [m["slug"] for m in list_portfolios()]
    shown = False
    for s in slugs:
        try:
            meta, store = open_portfolio(s)
        except KeyError:
            continue
        rows = addsignal.evaluate_all(store)
        if not rows:
            continue
        shown = True
        click.echo(f"\n─── {meta['name']} ───")
        click.echo(f"  {'TICKER':<14} {'STATE':<11} {'EXP/YR':>7} {'DISLOC':>8}  WHY")
        for r in rows:
            exp = f"{r['expected_annual_return']:.0f}%" if r["expected_annual_return"] is not None else "—"
            dis = f"{r['dislocation_pct']:+.1f}%" if r["dislocation_pct"] is not None else "—"
            click.echo(f"  {r['ticker']:<14} {r['state']:<11} {exp:>7} {dis:>8}  {r['why']}")
    if not shown:
        click.echo("No add plans yet — set one with 'fund paper-add-plan'.")


@cli.command("paper-list")
def paper_list():
    """List the paper portfolios and their current state."""
    from fundmgr.paper import list_portfolios, open_portfolio
    portfolios = list_portfolios()
    if not portfolios:
        click.echo("No paper portfolios yet — create one on the web at /paper.")
        return
    for meta in portfolios:
        _, store = open_portfolio(meta["slug"])
        navs = store.get_nav_history()
        nav = navs[-1].portfolio_nav_sek if navs else meta["capital_sek"]
        pnl = (nav / meta["capital_sek"] - 1) * 100 if meta["capital_sek"] else 0.0
        click.echo(
            f"  {meta['slug']:<24} {meta['name']:<28} "
            f"NAV {nav:>10,.0f} SEK  {pnl:>+6.2f}%  "
            f"(started {meta['created_at'][:10]}, benchmark {meta['benchmark']})"
        )


@cli.command("paper-import")
@click.argument("json_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--name", default=None, help="Override the portfolio name.")
@click.option("--capital", type=float, default=None,
              help="Override deployable capital (SEK). Defaults to meta.deployable_capital_sek.")
@click.option("--benchmark", default=None, help="Benchmark symbol (default: URTH).")
@click.option("--model", "model_label", default="Claude Fable",
              help="Label for who produced the picks.")
@click.option("--execute", is_flag=True, default=False,
              help="Buy every position now at live prices. Default: plan only — "
                   "import the idea and let positions fill as you record trades.")
def paper_import(json_file: str, name: str | None, capital: float | None,
                 benchmark: str | None, model_label: str, execute: bool):
    """Import a monitored 'live' sleeve from a structured LLM answer (JSON).

    Maps broker/Montrose tickers to Yahoo symbols, drops excluded_holdings, and
    stores per-position kill criteria, target weights, notes and the
    portfolio-level capex kill criterion. By default nothing is bought — the
    plan is imported and positions appear as you record fills ('fund paper-fill'
    / Telegram screenshot); pass --execute to open everything now. Either way
    the book is watched daily by 'fund paper-track' (kill criteria, capex,
    earnings, drift → Telegram).
    """
    from fundmgr import paper

    data = json.loads(open(json_file).read())
    parsed = paper.parse_structured_portfolio(data)
    cap = capital if capital is not None else parsed["capital_sek"]
    if not cap:
        click.echo("No capital found in the JSON — pass --capital.", err=True)
        sys.exit(1)
    if parsed["excluded"]:
        click.echo(f"  Excluded (never bought/sized): {', '.join(parsed['excluded'])}")

    try:
        slug, log = paper.create_portfolio(
            name=name or parsed["name"],
            capital_sek=float(cap),
            holdings_text=json.dumps(data, indent=2, ensure_ascii=False),
            model_label=model_label,
            benchmark=(benchmark or paper.DEFAULT_BENCHMARK),
            holdings_override=parsed["holdings_override"],
            position_meta=parsed["position_meta"],
            capex_kill=parsed["capex_kill"],
            kind="live",
            execute_buys=execute,
        )
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    for line in log:
        click.echo(f"  {line}")
    if parsed["capex_kill"]:
        click.echo(f"  ✓ Portfolio kill criterion armed: {parsed['capex_kill']['trigger']}")
    click.echo(f"\n✓ Imported as '{slug}'. Watched daily by 'fund paper-track'; "
               f"record fills with 'fund paper-fill {slug} …'.")


@cli.command("paper-fill")
@click.argument("slug")
@click.argument("ticker")
@click.argument("shares", type=float)
@click.argument("price", type=float)
@click.argument("fee", type=float)
@click.option("--side", type=click.Choice(["buy", "sell"]), default="buy", show_default=True)
@click.option("--date", "trade_date", default=None, metavar="YYYY-MM-DD",
              help="Trade date (defaults to today). Use when recording a past fill.")
def paper_fill(slug: str, ticker: str, shares: float, price: float, fee: float,
               side: str, trade_date: str | None):
    """Record a real broker fill into a mirror portfolio (price in SEK, like 'fund fill').

    \b
    Example:
        fund paper-fill kf-chokepoint-satellite VRT 20 610.00 39.00
        fund paper-fill kf-chokepoint-satellite ASML.AS 5 8420 39 --side sell
    """
    from fundmgr import paper

    try:
        meta, store = paper.open_portfolio(slug)
    except KeyError:
        click.echo(f"No paper portfolio '{slug}'. See 'fund paper-list'.", err=True)
        sys.exit(1)

    if trade_date:
        try:
            ts = datetime.strptime(trade_date, "%Y-%m-%d").replace(hour=12, minute=0)
        except ValueError:
            click.echo(f"Invalid date '{trade_date}' — expected YYYY-MM-DD", err=True)
            sys.exit(1)
    else:
        ts = datetime.utcnow()

    ticker, snap_note = paper.snap_ticker_to_plan(store, ticker)
    if snap_note:
        click.echo(f"  {snap_note}")
    currency = meta["currency_map"].get(ticker, "SEK")
    store.apply_fill(Transaction(
        ticker=ticker, side=side, shares=shares,
        price_sek=price, fee_sek=fee, source="fill",
        currency=currency, timestamp=ts,
    ))

    gross = shares * price
    direction = "Bought" if side == "buy" else "Sold"
    click.echo(f"✓ {direction} {shares:g} × {ticker} @ {price:.2f} SEK = "
               f"{gross:,.0f} SEK (fee {fee:.2f} SEK)")
    click.echo(f"  Cash remaining: {store.get_cash():,.0f} SEK")

    # Cost-basis NAV snapshot so the chart reflects the fill event
    try:
        bench_rows = store.get_benchmark()
        positions_after = store.get_positions()
        nav_cost = sum(p.shares * p.avg_cost_sek for p in positions_after) + store.get_cash()
        store.upsert_nav(NavPoint(
            date=ts.strftime("%Y-%m-%d"),
            portfolio_nav_sek=nav_cost,
            benchmark_value=bench_rows[-1]["close"] if bench_rows else 0.0,
            cash_sek=store.get_cash(),
        ))
    except Exception:
        pass


@cli.command("paper-retag")
@click.argument("slug")
@click.argument("old")
@click.argument("new", required=False)
def paper_retag(slug: str, old: str, new: str | None):
    """Fix a mis-tagged position: move OLD ticker to its correct Yahoo symbol.

    NEW defaults to the plan's matching symbol (e.g. `paper-retag <slug> ENR`
    moves ENR → ENR.DE). Cash is untouched — this only corrects the instrument
    and its currency, so a wrong-ticker holding is priced right again.
    """
    from fundmgr import paper

    try:
        meta, store = paper.open_portfolio(slug)
    except KeyError:
        click.echo(f"No paper portfolio '{slug}'. See 'fund paper-list'.", err=True)
        sys.exit(1)

    old = old.upper()
    new = (new or "").upper() or paper.intended_plan_symbol(store, old)
    if not new:
        click.echo(f"Couldn't infer the target symbol for {old} — pass it explicitly "
                   f"(e.g. fund paper-retag {slug} {old} {old}.DE).", err=True)
        sys.exit(1)
    try:
        res = paper.retag_position(store, old, new)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if res["shares"]:
        click.echo(f"✓ Retagged {res['shares']:g} × {res['old']} → {res['new']} "
                   f"(cash unchanged).")
    else:
        click.echo(f"✓ Moved the plan for {res['old']} → {res['new']} "
                   f"(no position was held under {res['old']}).")
    if res.get("moved"):
        click.echo(f"  Plan carried over: {', '.join(res['moved'])}.")
    if res.get("dropped"):
        click.echo(f"  Forgot {', '.join(sorted(set(res['dropped'])))} — that data "
                   f"described the wrong instrument.")
    if any("kill rules" in m or "add plan" in m for m in res.get("moved", [])):
        click.echo("  Price anchors were cleared: the drawdown line and review "
                   "price were set from the wrong instrument. Re-anchor after a "
                   "fresh look.")
    try:
        bench_rows = store.get_benchmark()
        nav_cost = sum(p.shares * p.avg_cost_sek for p in store.get_positions()) + store.get_cash()
        store.upsert_nav(NavPoint(
            date=datetime.utcnow().strftime("%Y-%m-%d"),
            portfolio_nav_sek=nav_cost,
            benchmark_value=bench_rows[-1]["close"] if bench_rows else 0.0,
            cash_sek=store.get_cash(),
        ))
    except Exception:
        pass


@cli.command("paper-setcost")
@click.argument("slug")
@click.argument("ticker")
@click.argument("avg_sek", type=float)
def paper_setcost(slug: str, ticker: str, avg_sek: float):
    """Correct a position's SEK cost basis to the broker's actual figure.

    Use the broker's Inköpsvärde ÷ shares (SEK per share) so the dashboard P&L
    matches Montrose. Cash is adjusted by the difference so NAV stays
    consistent — nothing else moves.
    """
    from fundmgr import paper

    try:
        meta, store = paper.open_portfolio(slug)
    except KeyError:
        click.echo(f"No paper portfolio '{slug}'. See 'fund paper-list'.", err=True)
        sys.exit(1)
    try:
        res = paper.set_cost_basis(store, ticker, avg_sek)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"✓ {res['ticker']}: cost basis {res['old_avg']:,.2f} → "
               f"{res['new_avg']:,.2f} SEK/share ({res['shares']:g} sh). "
               f"Cash {res['cash_delta']:+,.0f} SEK.")
    try:
        bench_rows = store.get_benchmark()
        nav_cost = sum(p.shares * p.avg_cost_sek for p in store.get_positions()) + store.get_cash()
        store.upsert_nav(NavPoint(
            date=datetime.utcnow().strftime("%Y-%m-%d"),
            portfolio_nav_sek=nav_cost,
            benchmark_value=bench_rows[-1]["close"] if bench_rows else 0.0,
            cash_sek=store.get_cash(),
        ))
    except Exception:
        pass


@cli.command("paper-status")
@click.argument("slug")
def paper_status(slug: str):
    """Print a mirror portfolio's positions, cash and cost-basis NAV."""
    from fundmgr import paper

    try:
        meta, store = paper.open_portfolio(slug)
    except KeyError:
        click.echo(f"No paper portfolio '{slug}'. See 'fund paper-list'.", err=True)
        sys.exit(1)

    positions = store.get_positions()
    cash = store.get_cash()
    click.echo(f"\n─── {meta['name']} ({slug}) ───")
    if not positions:
        click.echo("  No open positions.")
    else:
        click.echo(f"  {'Ticker':<14} {'Shares':>10} {'Avg Cost':>10} {'Cost Value':>13}")
        click.echo(f"  {'─'*14} {'─'*10} {'─'*10} {'─'*13}")
        for p in sorted(positions, key=lambda x: x.shares * x.avg_cost_sek, reverse=True):
            cv = p.shares * p.avg_cost_sek
            click.echo(f"  {p.ticker:<14} {p.shares:>10.4g} {p.avg_cost_sek:>10.2f} {cv:>10,.0f} SEK")
    nav = sum(p.shares * p.avg_cost_sek for p in positions) + cash
    click.echo(f"\n  Cash:       {cash:>12,.0f} SEK")
    click.echo(f"  NAV (cost): {nav:>12,.0f} SEK")


if __name__ == "__main__":
    cli()


@cli.command("paper-metric")
@click.argument("slug")
@click.argument("key", required=False)
@click.option("--label", default=None, help="How it reads on the dashboard.")
# No click.Choice here on purpose: define_custom_metric owns this rule, and a
# second copy of it in the option definition is how the error message came to
# recommend a command the CLI then rejected.
@click.option("--unit", default="",
              help="Unit the figure is printed in: %, x, or a currency code (SEK, USD…).")
@click.option("--note", default="", help="The company's definition, in your words.")
@click.option("--edgar", default="", metavar="CONCEPT",
              help="us-gaap concept to fill this from SEC filings (US filers only).")
@click.option("--forget", is_flag=True, help="Remove this metric from the book.")
def paper_metric(slug, key, label, unit, note, edgar, forget):
    """Define a metric this book can write criteria against.

    For the figures no provider carries because they are company-defined rather
    than accounting standards — organic growth, ARR, adjusted EBITA, cash
    conversion, order backlog. Once defined, a criterion may name it and
    `paper-report` supplies the numbers from the company's own report.

        fund paper-metric my-sleeve organic_growth --label "Organic growth" --unit %
        fund paper-metric my-sleeve                      # list what is defined
    """
    from fundmgr.evidence import custom_metrics, define_custom_metric, forget_custom_metric
    from fundmgr.paper import open_portfolio
    try:
        _meta, store = open_portfolio(slug)
    except KeyError:
        raise click.ClickException(f"No portfolio '{slug}' — see 'fund paper-list'.")

    if key and forget:
        if forget_custom_metric(store, key):
            click.echo(f"✓ Removed '{key}'. Criteria naming it will report it as unknown.")
        else:
            click.echo(f"No custom metric '{key}' in this book.")
        return
    if key:
        try:
            define_custom_metric(store, key, label or "", unit, note, edgar)
        except ValueError as e:
            raise click.ClickException(str(e))

    defined = custom_metrics(store)
    if not defined:
        click.echo("No custom metrics yet. Add one:\n"
                   "  fund paper-metric <slug> organic_growth --label 'Organic growth' --unit %")
        return
    click.echo(f"\n─── Custom metrics for {slug} ───")
    for k, meta in sorted(defined.items()):
        src = f"  ← SEC {meta['edgar']}" if meta.get("edgar") else ""
        click.echo(f"  {k:<24} {meta.get('label', ''):<24} "
                   f"{meta.get('unit', '') or '—'}{src}")
        if meta.get("note"):
            click.echo(f"    {meta['note']}")
    # Name a metric this book actually has. A hardcoded example invites the
    # "Unknown metric" error from a line the tool itself suggested.
    example = sorted(defined)[0]
    click.echo(f"\n  Enter figures with: fund paper-report {slug} <TICKER> "
               f"--period <quarter-end> --set {example}=<value>")
    if any(m.get("edgar") for m in defined.values()):
        click.echo(f"  Or pull the SEC-mapped ones: fund paper-edgar {slug} <TICKER>")
    click.echo()


@cli.command("paper-report")
@click.argument("slug")
@click.argument("ticker")
@click.option("--period", required=True,
              help="Period end the figures describe, YYYY-MM-DD (not today's date).")
@click.option("--set", "pairs", multiple=True, metavar="KEY=VALUE",
              help="A figure from the report. Repeatable.")
def paper_report(slug, ticker, period, pairs):
    """Record figures read off a company's own quarterly report.

    They land in the same cache a provider's numbers do, so criteria, kill
    rules and the quarters machinery read them identically.

        fund paper-report my-sleeve SYSR.ST --period 2026-06-30 \\
            --set organic_growth=6.4 --set adj_ebit_margin=9.3

    The period is the quarter the figures describe, not the day you typed them:
    that is what lets a late entry still count as its own quarter, and what stops
    one report's numbers counting as two quarters of evidence.
    """
    from datetime import datetime

    from fundmgr.evidence import field_meta, record_reported
    from fundmgr.paper import open_portfolio
    try:
        _meta, store = open_portfolio(slug)
    except KeyError:
        raise click.ClickException(f"No portfolio '{slug}' — see 'fund paper-list'.")
    try:
        datetime.strptime(period, "%Y-%m-%d")
    except ValueError:
        raise click.ClickException(f"Bad period '{period}' — use YYYY-MM-DD.")

    known = field_meta(store)
    values, unknown = {}, []
    for pair in pairs:
        if "=" not in pair:
            raise click.ClickException(f"Expected KEY=VALUE, got '{pair}'.")
        k, v = pair.split("=", 1)
        k = k.strip()
        if k not in known:
            unknown.append(k)
            continue
        try:
            values[k] = float(v.strip().replace(",", "."))
        except ValueError:
            raise click.ClickException(f"'{v}' is not a number (for {k}).")
    if unknown:
        raise click.ClickException(
            f"Unknown metric(s): {', '.join(unknown)}. Define them first with "
            f"'fund paper-metric {slug} <key>' — see 'fund paper-metric {slug}'.")
    if not values:
        raise click.ClickException("Nothing to record — pass at least one --set KEY=VALUE.")

    written = record_reported(store, ticker, values, period)
    click.echo(f"✓ {ticker.upper()} — period ending {period}:")
    for k, v in sorted(written.items()):
        meta = known[k]
        click.echo(f"    {meta['label']:<26} {v:,.2f}{meta.get('unit', '')}")


@cli.command("paper-read")
@click.argument("slug")
@click.argument("ticker")
@click.option("--period", required=True, help="Period end the report covers, YYYY-MM-DD.")
@click.option("--apply", "apply_", is_flag=True,
              help="Record the extracted figures (default: show them only).")
def paper_read(slug, ticker, period, apply_):
    """Read this book's tracked figures out of coverage of a company's report.

    Finds what was published around the reporting date, fetches it, and pulls
    out the metrics defined with 'fund paper-metric'. Nothing is stored without
    --apply: a figure that gates money should not enter the book on a model's
    say-so.

        fund paper-read my-sleeve SYSR.ST --period 2026-06-30
        fund paper-read my-sleeve SYSR.ST --period 2026-06-30 --apply
    """
    from datetime import datetime

    from fundmgr.data.release import read_report
    from fundmgr.evidence import record_reported
    from fundmgr.paper import open_portfolio
    try:
        _meta, store = open_portfolio(slug)
    except KeyError:
        raise click.ClickException(f"No portfolio '{slug}' — see 'fund paper-list'.")
    try:
        datetime.strptime(period, "%Y-%m-%d")
    except ValueError:
        raise click.ClickException(f"Bad period '{period}' — use YYYY-MM-DD.")

    tkr = ticker.upper()
    click.echo(f"⏳ Looking for {tkr} coverage around {period}…")
    out = read_report(store, tkr, period)

    for src in out["sources"]:
        click.echo(f"  · {src['publisher']}: {src['headline'][:70]}")
    if out.get("notes"):
        click.echo(f"  note: {out['notes']}")

    if not out["figures"] and not out["rejected"]:
        click.echo("\n  Nothing extractable found. Enter the figures by hand:\n"
                   f"    fund paper-report {slug} {tkr} --period {period} --set <key>=<value>")
        return

    if out["figures"]:
        click.echo(f"\n─── Found in the coverage ({len(out['figures'])}) ───")
        for f in out["figures"]:
            mark = "" if f["quote_verbatim"] else "  ~paraphrased quote"
            click.echo(f"  {f['label']:<26} {f['value']:>10,.2f}{f['unit']:<4}"
                       f"  conf {f['confidence']:.1f}{mark}")
            click.echo(f"      “{f['quote'][:100]}”")
            if f["period"]:
                click.echo(f"      period as stated: {f['period']}")
    if out["rejected"]:
        click.echo(f"\n─── Discarded ({len(out['rejected'])}) ───")
        for f in out["rejected"]:
            click.echo(f"  {f['label']:<26} {f['value']:>10,.2f}  — {f['reason']}")

    if not apply_:
        click.echo("\n  Check these against the report, then re-run with --apply "
                   "to record them.")
        return

    written = record_reported(store, tkr, {f["metric"]: f["value"] for f in out["figures"]},
                              period)
    click.echo(f"\n✓ Recorded {len(written)} figure(s) for period ending {period}.")


@cli.command("paper-edgar")
@click.argument("slug")
@click.argument("ticker")
@click.option("--quarters", default=4, help="How many recent quarters to show.")
@click.option("--apply", "apply_", is_flag=True, help="Record the figures.")
def paper_edgar(slug, ticker, quarters, apply_):
    """Fill EDGAR-mapped metrics from a US filer's own XBRL facts.

    Map a metric to a us-gaap concept first, then pull every quarter the
    company has filed:

        fund paper-metric kf datacenter_revenue --unit SEK --edgar Revenues
        fund paper-edgar  kf NVDA
        fund paper-edgar  kf NVDA --apply

    Values are stored against the period they cover, so the `quarters` part of
    a criterion works on them straight away. US filers only — a foreign issuer
    is simply absent from EDGAR, which the command says rather than guessing.
    """
    from fundmgr.data.edgar import EdgarError, read_ticker
    from fundmgr.evidence import MAX_SNAPSHOTS, custom_metrics, field_meta, record_reported
    from fundmgr.paper import open_portfolio
    try:
        _meta, store = open_portfolio(slug)
    except KeyError:
        raise click.ClickException(f"No portfolio '{slug}' — see 'fund paper-list'.")

    tkr = ticker.upper()
    try:
        series = read_ticker(store, tkr)
    except EdgarError as e:
        raise click.ClickException(str(e))
    except Exception as e:
        raise click.ClickException(f"EDGAR request failed: {e}")

    if not series:
        click.echo(f"Nothing to fill for {tkr}. Either this book maps no metric to a "
                   f"us-gaap concept (see 'fund paper-metric {slug}'), or {tkr} does "
                   f"not file with the SEC — only US filers do.")
        return

    known = field_meta(store)
    mismatched = []
    for metric, found in sorted(series.items()):
        meta = known.get(metric, {})
        declared, filed_unit = meta.get("unit", ""), found["unit"]
        values = found["values"]
        click.echo(f"\n─── {meta.get('label', metric)} — filed in {filed_unit} ───")
        for period in sorted(values, reverse=True)[:quarters]:
            click.echo(f"    {period}   {values[period]:>18,.0f} {filed_unit}")
        # A currency the metric does not claim makes every threshold written
        # against it mean something else. Say so, and refuse to store it.
        if declared and filed_unit and declared != filed_unit:
            mismatched.append((metric, declared, filed_unit))

    for metric, declared, filed_unit in mismatched:
        click.echo(f"\n  ⚠ {metric} is declared in {declared} but the company files "
                   f"in {filed_unit}. A threshold written against it would mean "
                   f"something else. Fix it with:\n"
                   f"      fund paper-metric {slug} {metric} --unit {filed_unit} "
                   f"--edgar {custom_metrics(store)[metric]['edgar']}")
    if mismatched:
        raise click.ClickException("Nothing recorded — correct the unit(s) first.")

    if not apply_:
        click.echo("\n  Check these against the filing, then re-run with --apply.")
        return

    # One call per period so each figure lands in the quarter it describes.
    # Only the periods the series can hold are written: the older ones would be
    # trimmed on the next append anyway, and writing them makes the count report
    # sixteen years of history that is not there.
    all_periods = sorted({p for f in series.values() for p in f["values"]})
    periods = all_periods[-MAX_SNAPSHOTS:]
    written = 0
    for period in periods:
        batch = {m: f["values"][period] for m, f in series.items()
                 if period in f["values"]}
        written += len(record_reported(store, tkr, batch, period))
    older = len(all_periods) - len(periods)
    tail = f" ({older} older quarter(s) skipped — the series holds {MAX_SNAPSHOTS})" if older else ""
    click.echo(f"\n✓ Recorded {written} figure(s) across {len(periods)} quarter(s)"
               f"{tail}, {periods[0]} to {periods[-1]}.")
