# AI Fund Manager — Claude Code Context

This file gives Claude Code enough context to resume work on this project in any session (CLI, web, or mobile).

---

## What this project is

**Primary purpose:** improve future investment decisions by learning from outcomes and constructing better decision prompts. Prioritize reliable evidence, faithful prompt evaluation, and measured improvement. The fund portfolios are the feedback and testing environment for that purpose. See `docs/LEARNING.md` for the current implementation sequence.

A **discretionary fund manager driven by an LLM**, running on a Raspberry Pi 5. One book holds **real money** (🇸🇪 Nordic, Montrose KF/ISK); the rest are simulations that run the identical pipeline so decision quality can be compared across models and mandates. Every week each fund is handed its mandate, its portfolio state, a screened slice of its universe and its own past lessons, and returns a structured `DecisionRun` of buy/sell/hold actions. Mechanical **guardrails** then approve, clip or reject each one — the model proposes, the guardrails dispose.

The fund does not place orders. For the real book you execute in Montrose and record fills; the sims auto-fill at live prices.

**Not to be confused with DeepSwing**, the sibling project on the same Pi: that one is a *swing trading* simulator (15-min-bar decisions, days-long holds, DSPy/MIPRO-compiled prompts). This one is a *portfolio allocator* (weekly decisions, 28-day evaluation horizon, whole-book reasoning). They share the `financedata` library and nothing else.

---

## The five funds

Each `config/config*.yaml` bundles a mandate + universe + risk limits + model, so a profile can never mix e.g. the Buffett mandate with the Nordic universe.

| Config | Fund | Universe (enabled) | `top_n` | Model | Money |
|---|---|---|---|---|---|
| `config.yaml` | 🇸🇪 Nordic — REAL money | `universe.csv` (1,605) | 75 | `gpt-5.6-sol` | **REAL** |
| `config_global.yaml` | 🌍 Global SIM — GPT-5.6-sol | `universe_global.csv` (17,117) | 120 | `gpt-5.6-sol` | sim |
| `config_claude.yaml` | 🤖 Global SIM — Claude | `universe_global.csv` (17,117) | 120 | `claude-opus-4-8` | sim |
| `config_buffett_gpt.yaml` | 🧱 Buffett Screen — GPT-5.6 | `universe_buffett.csv` (96) | 100 | `gpt-5.6-sol` | sim |
| `config_buffett_claude.yaml` | 🧱 Buffett Screen — Claude | `universe_buffett.csv` (96) | 100 | `claude-opus-4-8` | sim |

All five run `reasoning_effort: high` and `n_samples: 3`.

The two Global and two Buffett funds are deliberately **paired across providers on an identical universe and mandate**, so the only difference is the decision model. Keep `learning_model_id` pinned to the same id everywhere for the same reason: the comparison should be a difference in decision-making, not in how each fund is coached.

`FUND_CONFIG=config/config_x.yaml` selects the profile for any `fund` command; unset means `config.yaml`.

---

## Key design decisions (don't re-litigate these)

- **The model proposes, guardrails dispose.** `guardrails/rules.py` re-checks every action against the mandate mechanically: universe membership, stale-data block on buys, min trade size, max position weight (clipped, not rejected), sector concentration, max open positions, cash floor, and a turnover cap that drops lowest-confidence trades until the run fits. Nothing reaches the book without passing. A rejected action is logged with its reason, so `guardrail_log` is the audit trail of what the model wanted versus what it got.
- **Consensus sampling, not one shot.** `n_samples: 3` runs the decision call 3× in parallel and majority-votes each ticker. Unanimous and majority-only actions are labelled separately in the report and on Telegram. A ticker needs more than half the runs to survive.
- **Reasoning effort is `high` everywhere it decides.** Weekly runs are ~9 heavy calls a week in total — a rounding error — and a weekly real-money allocation across a whole portfolio with 3-way voting is exactly where deep reasoning earns its price. The small extraction jobs (release reading, kill-criterion judging, OCR fill parsing) sit on `gpt-4o-mini` with tight `max_tokens` and no reasoning at all. Don't collapse those two tiers into one.
- **`fund run` refuses to run into a shut market.** Cron fires on a fixed weekly schedule and knows nothing about holidays, while the only calendar check in the pipeline sits down in `auto_fill`, per ticker, *after* the model has been paid for — so on US Labor Day a US-dominant fund bought a full consensus run and then skipped every fill one at a time. Skipped fills are not queued for the next open either: they wait for the next *weekly* run. `_skip_on_market_holiday` resolves the universe's **dominant calendar** and returns before fetching anything, naming the next session on stdout and Telegram. `--force` overrides.
  - Dominant, not per-name: a run whose main market is shut is deciding on stale prices whatever the minority can still trade.
  - Dominant by **holiday schedule**, not by MIC. XNYS and XNAS share the US calendar, and counting them separately makes **London** the plurality of `universe_global.csv` (3,940 vs 3,782) — LSE trades on US holidays, so the naive version would have let exactly that run proceed into a closed market. Nothing else is grouped: the Nordic exchanges genuinely differ from each other on national holidays.
  - Fails open throughout — unreadable universe, unmapped exchange or calendar error all run as before. Blocking a run on a guess is worse than the spend it saves.
- **A style mix is the same dial over a judgement, and says so.** `styles.py` buckets each candidate into Buffett-style quality / growth / higher-risk-speculative / unclassified from the fundamentals `screener.py` already scores and the criteria `mandate_buffett.md` states in prose, so a run can be asked for "40% compounders and 20% higher-risk" inside one mandate. It shares `allocation.py` with the regional dial, and differs in exactly one way that governs everything else: **a region is a fact on the universe row, a style is a reading of a 7-day-TTL cache**. So a name is bucketed only on a *positive finding* — missing figures mean `unclassified`, which is reported, counted against no target and blocked by none, because rejecting a buy on a cache miss is failing closed on a gap. The honest cost is that a style ceiling under-counts, and the prompt, the result and the UI all say so rather than implying it is airtight. Selection is always *within* the profile's own universe: a Global run asking for quality gets the best quality names in the global universe, never names imported from the Buffett screen — mandate and universe travel together. A free-text **brief** covers the tilt no bucket captures; it steers selection inside the mandate and relaxes nothing.
- **A regional mix is reserved in the screen, capped in the guardrails, and reported at the end.** `regions.py` groups the universe's `country` column into Nordics / North America / UK & Ireland / Europe ex-Nordics / Asia-Pacific / Other, and the What-If Lab and sleeve reviews take a per-region % of NAV. Three layers have to agree on it: the **screener** reserves candidate slots in proportion to each target — without that the mix is unbuildable rather than merely hard, since the score is blind to geography and a week where momentum sits in US large caps hands the model four Nordic names; the **prompt** states the brief and tags each candidate with its region; the **guardrails** reject a buy past target + tolerance. Only the ceiling binds. A guardrail can refuse a trade but cannot invent one, so "at least 30% Nordics" is a brief and a number reported back, while "at most 40%" is mechanical — anything claiming to enforce a floor is claiming a guardrail can create a buy. A region nobody named is unconstrained; an explicit **0%** is an exclusion, so it caps at zero rather than at the tolerance band and drops that region's names from the candidate list (held and pinned excepted — you must be able to sell what you own).
- **The universe is screened before the model sees it.** `screener.py` scores every ticker on momentum (1/5/20/60-day, weighted), trend alignment (above MA50/MA200), and RSI (penalising overbought, rewarding room to run), then passes the top `screener.top_n`. **Held and pinned tickers are always included** regardless of score — the model must be able to sell what it owns. At `top_n: 100` against a 96-name universe the Buffett funds see everything; the Global funds see 120 of 17k, and the Nordic fund 75 of 1,605.
- **Fills never book at a stale price.** `auto_fill` checks each ticker's exchange calendar and skips the fill when that venue is closed, with a Telegram reminder. `check-stops` is the same idea on a faster clock: pure price arithmetic, **no LLM calls**, every 15 min during trading hours — separate from the weekly decision entirely. (DeepSwing copied this split.)
- **Outcomes are scored at a horizon the mandate chooses.** `evaluation_horizon_days` (28) is a property of the mandate, not a system constant — a momentum book and a quality-compounder screen are not the same question asked at 28 days. The learning loop reads the horizon back and will tell a fund to stop taking positions it cannot score inside it, so shortening it quietly rewrites the strategy.
- **A thesis is judged before a lesson is drawn from it.** `verify_theses` asks whether a position that beat did so *because* the thesis held or *in spite of* it, and the verdict (held / broke) is what the qualitative learning is distilled from. Without that, the lesson learner rewards being right by accident.
- **Learnings are injected, not just logged.** `build_prompt` folds surviving lessons and — once `fund optimize` has compiled one — the optimized **Decision Guidance** artifact into every subsequent run's prompt.
- **Cold start lifts the turnover cap.** A book sitting at ≥ `cold_start_cash_threshold` cash gets `cold_start_turnover_pct` instead of the weekly cap, so deploying from near-100% cash isn't spread over months by a rule meant to damp churn.
- **Prices are native; the real fund converts to SEK.** `fx_to_sek` is True only for the real Nordic book (the broker settles SEK). Sims run native-consistent — turning it on for them would mix a currency bet into the model comparison.

---

## The weekly run (`fund run`)

```
holiday gate ──▶ prices ──▶ fundamentals (7d TTL) ──▶ benchmark ──▶ macro
     │                                                                │
     └── skips before spending anything if the dominant market is shut │
                                                                       ▼
features + screen (top_n, held/pinned forced in) ──▶ news + FinBERT (candidates only)
                                                                       │
                          evaluate matured outcomes ──▶ verify theses ──▶ learnings
                                                                       │
   portfolio snapshot ──▶ build_prompt (mandate + macro + state + limits + universe
                          + learnings + optimized guidance)
                                                                       │
                          call_llm_consensus (n_samples, majority vote)
                                                                       ▼
                          apply_guardrails ──▶ approved / clipped / rejected
                                                                       ▼
      save recommendation + seed outcomes + persist stops + NAV point + Telegram
                                                                       ▼
                          auto_fill (sims) — skips any venue that is closed
```

News and FinBERT run on **screener candidates only**, never the full universe — that is the step that would otherwise dominate the run.

---

## The What-If Lab (`/whatif`)

Generates a hypothetical from-scratch portfolio for any fund profile against a **synthetic clean-slate snapshot** (full amount in cash, zero positions), so it answers "what would this mandate buy today with this money" rather than "what should it do next". Model, sample count, amount, macro and price-refresh are all per-run overrides. Results are written to disk and listed newest-first; generation runs in a daemon thread with a single job slot.

Three things it does that the weekly run doesn't:

- **Monitoring plan (opt-in toggle).** `DecisionRun` has carried `kill_criterion`, `add_criterion`, `target_price`, `max_weight_pct` and `tranche_pct` since the sleeve review learned to specify monitoring, but only `sleeve_review` asks for them. The Lab asks too when the toggle is on. It is **off by default** because it costs output tokens and only earns them on a run that might become real. The directive deliberately mirrors `sleeve_review`'s wording — that is the prompt these fields were designed against, and the two should not drift apart. Plan values are stored **only when requested**: OpenAI structured outputs put the field descriptions in front of the model whether or not you ask, so an unrequested value is a guess rather than an answer, and promotion must not read it as a plan.
- **Regional and style mixes (optional), plus a free-text brief.** Per-bucket % of NAV, blank = unconstrained, 0 = excluded. The result reports target vs achieved per bucket and names any bucket the run left short — the floor is not enforceable, so the shortfall is the run's own caveat rather than something the pipeline quietly papers over. Both mixes and the brief travel on promotion, so a sleeve promoted from a 30%-Nordics, 40%-quality run is reviewed against that mix instead of drifting back to whatever the ranking favours.
- **Promote to a live sleeve.** `promote_to_sleeve` turns a stored result into a real monitored sleeve via `paper.create_portfolio(kind="live")`. Only **approved buys** carry over — a rejected or turnover-dropped action was never part of the book the run proposed. It defaults to **plan-only** (like `fund paper-import`), so promoting never spends money by surprise. Promoting without a monitoring plan is **reported, not blocked** — the toggle exists so the choice is the user's — but the response and the UI name the positions `paper-track` will have nothing to watch.
  - It builds the holdings **directly**, not through `paper.parse_structured_portfolio`. That function exists to translate *broker* tickers; these are canonical Yahoo symbols read from the profile's `universe.csv`, and routing them through the broker map would let an entry like `ASML` be silently rewritten to a different listing than the one screened.
  - `run_id` reaches `load_result` from a URL path segment, so it is matched against the generated id format rather than joined onto a directory.

---

## Sleeves: paper vs live

`create_portfolio(kind=...)` makes two different things from the same machinery:

- **`paper`** — a simulation book. `execute_buys=True`: every position opens now at live prices.
- **`live`** — a mirror of a real broker account. `execute_buys=False`: the *plan* is imported and positions appear as you record fills (`fund paper-fill`, or a Telegram screenshot through OCR). Watched daily by `fund paper-track` against its kill criteria, capex trigger, earnings dates and drift.

`sleeve_review` re-decides a live sleeve against its current book. It borrows a source profile for universe, mandate and risk limits, and carries four remembered settings of its own: the country scope, per-sleeve risk caps, the regional mix, and the style mix with its free-text brief. All four are stored on the sleeve, so a book keeps being reviewed the way it was set up rather than needing them retyped each run.

See **[docs/MONITORING.md](docs/MONITORING.md)** for the full monitoring model — kill and add criteria, how criterion text is read, staleness, add signals, and the evidence sources behind each.

---

## Risk limits (`RiskConfig`, enforced in `guardrails/rules.py`)

| Limit | Default | Notes |
|---|---|---|
| `max_position_pct` | 18 | Clipped, not rejected — the trade shrinks to fit |
| `max_positions` | 10 | |
| `max_sector_pct` | 35 | NAV weight in any one GICS sector |
| `min_cash_pct` / `max_cash_pct` | 12 / 25 | per-fund — the real book runs **5 / 10** (ISK is taxed on total balance) |
| `min_trade_sek` | 2,500 | below this, fees eat the edge |
| `max_turnover_pct` | 25 | per run; excess dropped lowest-confidence first |
| `stale_after_days` | 5 | stale data blocks **buys** only |
| `cold_start_*` | 80 / 50 | lift the turnover cap when deploying from cash |
| `region_targets` | `{}` | % of NAV per region; only the ceiling (target + tolerance) is enforced |
| `region_tolerance_pct` | 10 | band around each regional target |
| `style_targets` | `{}` | % of NAV per style bucket; ceiling only, and it under-counts what it could not classify |
| `style_tolerance_pct` | 10 | band around each style target |

Fees: `rate` 0.10% with a 1–99 SEK floor/ceiling (`FeeConfig.calc`).

---

## Scheduling (crontab — see `deploy/cron.example`)

Everything is cron on the Pi; there is no in-process scheduler.

| Job | When | LLM? |
|---|---|---|
| `fund run` × 5 funds | Mon, staggered 09:30–17:30 CET | yes — the main spend |
| `fund check-news` | weekdays ×3–4 | no (local FinBERT); `--auto-run` can trigger a run |
| `fund check-stops` | every 15 min, trading hours | no |
| `fund paper-watch` | hourly, trading hours | cheap `gpt-4o-mini` only |
| `fund paper-track` | weekdays 22:15 | yes, gated |
| `fund optimize` × 5 | Sun 02:00–04:00 | yes — **gated, see below** |
| `deploy/backup.sh` | daily 03:00 + post-run | no |

Run times are staggered so the funds don't collide, and each is placed 30 min after its market's open. `fund run` gates itself on holidays; the others don't need to.

---

## Learning loop (how the fund improves)

- **`fund optimize`** compiles a *Decision Guidance* artifact with DSPy MIPROv2 from `(decision → mandate-horizon outcome)` pairs, as an **inactive candidate** under `compiled/candidates/<fund>/`. Compilation no longer replaces active guidance; `build_prompt` only reads the existing active guidance artifact. `fund compare-guidance` reuses frozen prompts, consensus and guardrails, then checks cumulative allocation feasibility; `fund score-guidance` measures buy-and-hold portfolio returns after fees from supplied daily valuations. Opt-in `shadow` config records fresh weekly comparisons before outcomes; `fund collect-guidance` (also called after saved weekly runs) collects adjusted histories/FX and scores allocations at a future common close. Artifacts are immutable under the database parent’s `shadow/<fund>/`; invalid/incomplete cases never auto-retry model calls or promote. `fund evaluate-guidance` reconstructs archived scores and reports coverage and descriptive performance across non-overlapping periods, stratified by exact candidate/incumbent/configuration. Statistical promotion thresholds and a promotion gate remain to be implemented (see `docs/LEARNING.md`). It is **hard-gated**: `min_outcomes` (30) evaluated outcomes *and* `min_examples` (25) usable run-level examples. Below either, it logs and skips at zero cost — MIPRO holds out 20% and picks winning instructions on that slice, so at 8 examples it was selecting on 2 runs against ~2pp of weekly excess-return noise, and the artifact would have looked like an improvement while being none. As of Sept 2026 these gates have never been met, so no guidance has ever compiled. That is the threshold working, not a bug.
- **Two kinds of learning.** `generate_learnings` produces *calibration* lessons statistically (Wilson intervals over hit rates by confidence band); `generate_qualitative_learnings` distils *what happened and why* from a batch review of matured outcomes. `surviving_lessons` checks new lessons for minimum supporting tickers; it does not yet test their forward predictive value.
- **Pooling.** `optimizer.pool_configs` lets a fund train on other funds' examples, sharing the trainset while each still compiles and applies its own guidance.

---

## File map (key files)

```
config/config*.yaml           One per fund: mandate + universe + risk + model + cadence
config/mandate*.md            The system prompt — investment policy in prose
config/universe*.csv          name,yahoo_ticker,isin,country,exchange,sector,enabled
src/fundmgr/config.py         AppConfig + all sub-configs; get_enabled_tickers, load_universe
src/fundmgr/cli.py            Every `fund` command; the weekly run pipeline lives here
src/fundmgr/engine/client.py  call_llm / call_llm_consensus — OpenAI structured outputs, Anthropic JSON
src/fundmgr/engine/schema.py  Action + DecisionRun — the contract the model answers in
src/fundmgr/engine/prompt.py  build_prompt: mandate + macro + state + limits + universe + learnings
src/fundmgr/engine/evaluator.py   Score matured outcomes; calibration + qualitative learnings
src/fundmgr/engine/thesis_check.py  Did it work *because* of the thesis, or in spite of it
src/fundmgr/engine/optimizer.py    MIPROv2 → Decision Guidance artifact (gated)
src/fundmgr/engine/whatif.py       What-If Lab: generate, load_result, promote_to_sleeve
src/fundmgr/engine/sleeve_review.py  Review a live sleeve; the monitoring-plan prompt lives here
src/fundmgr/engine/auto_fill.py    Paper fills; skips closed venues
src/fundmgr/guardrails/rules.py    Every mechanical risk check; the audit log
src/fundmgr/allocation.py     The mix arithmetic both dials share: bands, quotas, exposure, reporting
src/fundmgr/regions.py        Geographic buckets (a fact on the universe row)
src/fundmgr/styles.py         Risk/quality buckets (a reading of fundamentals — see its docstring)
src/fundmgr/data/screener.py       Momentum + trend + RSI score → top_n candidates
src/fundmgr/data/market_hours.py   Exchange calendars: is_exchange_open, dominant_calendar,
                                   is_trading_day, next_session
src/fundmgr/data/prices.py         TickerFeatures + feature build
src/fundmgr/data/release.py        Statement/report reading (gpt-4o-mini)
src/fundmgr/paper.py               Sleeves: create_portfolio, parse_structured_portfolio, kill judging
src/fundmgr/watchplan.py           Kill/add criteria plumbing
src/fundmgr/state/store.py         SQLite — positions, cash, transactions, recommendations,
                                   decision_outcomes, learnings, caches, stops, alerts, reviews
src/fundmgr/web/app.py             FastAPI; mounts /sim* per fund, /paper, /live, /whatif
src/fundmgr/notify/                Telegram send + bot (OCR fill import)
tools/discover_tickers.py          Validates a ticker list against yfinance → universe.csv rows
```

`financedata` is a **shared local library** installed from `../FinanceData` (not on PyPI). It provides prices, fundamentals, FX, news and macro. It is not installable in a sandbox, so the full test suite only runs where it is present.

---

## Running locally

```bash
uv sync                                   # or: pip install -e .
cp .env.example .env                      # API keys + Telegram
fund init                                 # initialise a portfolio
fund run --dry-run                        # full pipeline, nothing saved
FUND_CONFIG=config/config_global.yaml fund run
uvicorn fundmgr.web.app:app --reload      # dashboard
```

Useful flags: `--force` (ignore the holiday gate), `--force-refresh` (re-fetch prices), `--skip-news` / `--skip-macro` / `--skip-fundamentals` (faster iteration).

---

## Deployment

`deploy/deploy.sh` runs **on the Pi**, pulls the `deploy` branch, checks the `FinanceData` checkout is on the right branch, waits for any in-flight `fund run` to finish, then restarts `fundmgr-bot`, `fundmgr-web` and `fundmgr-global-web` and verifies each came back. Triggered by GitHub Actions over SSH or by `deploy/poll-deploy.sh` (5-min cron). Full provisioning: **[deploy/SETUP.md](deploy/SETUP.md)**; backups: **[deploy/BACKUP.md](deploy/BACKUP.md)**.

---

## Style conventions

- No comments unless the WHY is non-obvious — but *do* record the failure that motivated a rule
- Type hints on all function signatures
- `from __future__ import annotations` at the top of every file
- Fail open on anything advisory (calendars, aliases, notifications); fail closed on anything that moves money
- Model output is never `innerHTML` — build DOM nodes and set `textContent`
