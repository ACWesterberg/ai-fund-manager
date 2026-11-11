# Roadmap / Ideas

Forward-looking ideas noted for later — not yet built. Keep short; promote to an
issue or a `fund` command when picked up.

---

## Current priority: evidence-driven prompt improvement

See [LEARNING.md](LEARNING.md). New optimizer output is staged as an inactive
candidate. Frozen paired comparisons and offline portfolio scoring are available;
opt-in weekly forward registration and automated outcome collection are available.
Offline evidence auditing and non-overlapping-period aggregation are available.
The next milestone is validated promotion thresholds and a tested promotion gate. The older exploration notes below are background, not the
current delivery order.

---

## Decision memory and Cognee — deferred until the Mac Studio move

**Picked up when** the project runs on the Mac Studio. Assessed 2026-11 against
[Cognee](https://github.com/topoteretes/cognee) (agent memory: entity extraction
→ knowledge graph + vectors → retrieval, with an `improve` step that keeps
rewriting the graph).

**Verdict: don't put Cognee in the decision path.** Hardware was never the main
objection; three others stand regardless of machine:
- **Retrieval isn't the bottleneck, evidence is.** At most `PROMPT_LEARNING_LIMIT`
  (8) short lessons reach the prompt; the open problems are labels, reward and
  the promotion gate ([LEARNING.md](LEARNING.md)).
- **Point-in-time correctness.** A self-updating graph with no documented as-of
  query leaks future knowledge into replays and frozen comparisons, and makes
  "what the model saw" unreproducible.
- **Measurement.** Per-run retrieved context escapes the learnings fingerprint
  and the paired guidance comparison, so its effect can't be attributed.

**Step 1 — per-ticker decision memory (build first, no new infrastructure).**
The prompt carries general lessons and the current book, but nothing about what
*this fund* previously concluded about a name that has come back into the
candidate list. Add a block to `build_prompt` from the existing SQLite store:
prior thesis, held/broke verdict, realised alpha, fired kill criterion.
- Filter strictly by `run_date < snapshot date` — replays must not leak.
- Candidates and held names only, one or two lines each.
- Archive it in the snapshot and fingerprint it like `learnings_fingerprint`.

**Step 2 — Cognee as a research track over filings.** Local extraction and
embedding models (or a local LLM) make a continuously maintained graph over
filings/releases cheap on the Mac: supplier/customer links, capex exposure,
concentration. Run it as a separate process feeding **monitoring** first
(`paper-track` kill/add criteria — advisory, fails open, easy to check by hand).
Enter the weekly decision prompt only once it can answer "what did we know on
date X", and only through the paired guidance comparison.

**Later, if ever.** Relevance-based lesson selection (embeddings in SQLite) only
once the active lesson count well exceeds 8 *and* the learnings fingerprint shows
the channel helps at all.


## Decision-quality backlog (gap analysis, 2026-11)

Done 2026-11: sells scored from their own side in lessons and the thesis
cross-tab; the screener's coverage-biased sentiment term removed; market cap in
SEK; and the per-run archive (`state/archive.py`) plus per-sample consensus
logging, which the items below depend on. In priority order:

1. **Score the unbought.** Compare the picks with an equal-weight basket of the
   shown candidates, and score shown-but-not-chosen names. Today only approved
   actions get outcomes and holds score zero in the optimizer reward
   (`optimizer.py` side handling). Backfillable from 2026-09-22 via archived
   candidate features.
2. **Screener rank vs forward return** (rank correlation per week, e.g.
   alphalens-reloaded) from the run archive. The prompt re-sorts by
   `_signal_score`; measure both orderings.
3. **Per-ticker decision memory** — see the Mac-move section above.
4. **Surface fetched-but-dropped quality data**: operating/EBITDA margin, ROA,
   FCF, debt/cash are cached by FinanceData but dropped in `apply_to_features`;
   `data/statements.py` (net debt/EBITDA, interest cover) is wired to sleeves
   only. The Buffett mandate asks for exactly these.
5. **Earnings and analyst detail**: last surprise, estimate revisions
   (`eps_trend`/`eps_revisions` must be snapshotted — current-only in yfinance),
   upgrades/downgrades. Needs FinanceData support. `days_to_earnings` is
   calendar days though labelled trading days.
6. **Portfolio-aware features**: ADV in SEK (mandate liquidity rule is
   unchecked), correlation to current holdings, sector-relative momentum.
7. **Unused sources**: `get_insider_summary` (SEC Form 4 + Swedish FI register)
   is never called; news window is 3 days on a 28–90 day horizon;
   `get_macro_context` (policy rates) unused.
8. **Measurement**: confidence/size vs alpha regression; GPT-vs-Claude paired
   weekly differences with a block bootstrap; return attribution; forward test
   of lessons via the archived `learnings_hash`.

Outside projects worth borrowing from: Graphiti (bi-temporal graph memory —
preferred over Cognee for the filings track, as it answers as-of queries), GEPA
(`dspy.GEPA`, reflective prompt evolution that consumes textual feedback such as
thesis verdicts — a candidate proposal step for the bounded optimizer), a
bull/bear critique pass as in TradingAgents (only via `compare-guidance`),
Börsdata API for Nordic historical fundamentals (check Pro+/non-commercial terms).

---

## Historical backtest / replay (bootstrap the optimizer's trainset)

**Idea.** Replay the fund over historical weeks to generate `(decision → 28-day
outcome)` pairs without waiting months of live runs, then feed them into
calibration learnings + the MIPRO trainset. Collapses "months of live outcomes"
into an afternoon of replay.

**Why it's valuable.** The optimizer is data-starved: each live outcome takes ~4
weeks to mature, so the prompt needs months to get strong. A backtest bootstraps
that.

**The hard part — the fund's decision *is* the LLM.** DeepSwing backtests only its
*mechanical* signal path (point-in-time price slices → screener/risk), never the
LLM, which dodges the leakage problem. The fund manager's picks come from the
model, so a real backtest has to replay the LLM — where both the value and the
danger live.

**Landmines (must design around):**
- **Point-in-time data / look-ahead.** Prices + technicals are clean (slice OHLCV
  to the as-of date). yfinance **fundamentals are only a current snapshot** →
  severe leakage if used for a past decision. Historical news/sentiment is hard
  to reconstruct.
- **LLM knowledge-cutoff leakage (unique to LLM backtests).** If the as-of date
  precedes the model's training cutoff, it may already "know" the outcome. Only
  mitigable (replay dates *after* cutoff; ticker/number anonymisation — imperfect).
- **Survivorship bias** (today's universe drops delisted names), execution/capacity
  realism, and overfitting the prompt to the backtest.

**Honest feasible version.** A **prices-and-technicals-only** replay over dates
*after* the model's cutoff, fundamentals/news deliberately dropped → every input
is genuinely point-in-time. Treat results as *directional* bootstrap, not a truth
oracle; let clean live outcomes (pinned-window evaluator + `repair-outcomes`)
correct backtest artifacts over time.

**Sketch of the harness** (mostly reuses existing parts):
`fund backtest --from … --to … --prices-only` → for each historical week: build a
point-in-time `TickerFeatures` (prices/technicals only) → `build_prompt` →
`call_llm_consensus` → score each pick 28 trading days later → write as
`decision_outcomes` tagged `source="backtest"` so they stay quarantined from the
live track record. DeepSwing's point-in-time slicing (`src/backtesting/engine.py`)
is directly portable.

---

## Should MIPRO optimize the whole prompt, or stay an appended guidance block?

**Current design (deliberate).** MIPRO optimizes the `WeeklyDecision` signature's
*instruction* (`engine/dspy_program.py`), with the mandate held fixed as an
`InputField`. The live run path (`engine/prompt.py` + `client.py`) does **not**
execute the DSPy program — it extracts the winning instruction string
(`optimizer._compiled_instructions`) and *appends* it to the human-authored
mandate as a labeled "Decision Guidance" block. So MIPRO owns the full
instruction layer but never the mandate, and at runtime its output is grafted on
rather than run as the program.

**Two known gaps this creates.**
- **Context drift** — the instruction is tuned inside DSPy's assembly
  (ChainOfThought scaffolding, mandate-as-input-field) but applied at runtime
  inside a differently-built prompt. Detectable via the per-run `guidance_hash`
  (`prompt.snapshot_to_dict` regime) → guided vs unguided alpha.
- **Discarded demos** — MIPRO also proposes few-shot demonstrations; we keep only
  the instruction string and drop them.

**Decision (2026-07): change nothing now.** No fund has history yet — MIPRO can't
compile until ~30 evaluated outcomes + 8 scored runs (≈autumn). The fixed mandate
is what keeps the five funds' GPT-vs-Claude / global-vs-Buffett comparison
legible, and the drift gap only bites once guidance exists. Keep `fund optimize`
enabled (harmless no-op until thresholds met) and let data decide.

**Revisit when** the first guidance artifacts compile: group each fund's runs by
`guidance_hash` and compare realized alpha, guided vs unguided. *Then*, and only
then:
- If guidance clearly helps but looks capped → wire the live path to the DSPy
  `FundManager` program (closes drift + recovers demos; also swallows the N=3
  consensus + JSON-parsing path, so it's a meaty change — earn it with evidence).
- **Do not** fold the mandate into the optimizable instruction — that trades away
  the auditable baseline and cross-fund comparability for little gain.

---

## Align dashboard design with DeepSwing

DeepSwing = single-page **tabbed** SPA (Comparison / per-track / Decisions /
Heuristics / Prompts) for two paper tracks. Fund Manager = **multipage sidebar**
app for one real fund + two sims (sub-routes `/sim`, `/sim-claude`). Fund Manager
already has richer composition viz (NAV vs benchmark, bubbles, weight bars) than
DeepSwing.

Open question: what "align" means — a head-to-head **Comparison** view across the
three funds (DeepSwing's headline tab, which FM lacks), a tabbed fund-switcher vs
the current sidebar, or a fuller visual reskin. Decide scope before touching the
real-money UI.
