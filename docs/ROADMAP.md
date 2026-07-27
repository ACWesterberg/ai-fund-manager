# Roadmap / Ideas

Forward-looking ideas noted for later — not yet built. Keep short; promote to an
issue or a `fund` command when picked up.

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
