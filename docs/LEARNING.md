# Learning-driven prompt improvement

The product is a decision prompt that improves from experience. The objective is
benchmark-relative return after estimated costs within the mandate's risk limits.
Use the fund's intended horizon (28 days by default, 90 days for the Buffett
profiles), with downside and turnover checks. Thesis judgments explain decisions;
they are not interchangeable with profitable allocations.

## Implemented foundation

- **Candidate reward:** historical thesis verdicts no longer reward new theses
  about the same ticker. The search metric is versioned `directional_alpha_v2`.
  It remains an interim directional-return surrogate: it does not yet measure
  allocation sizing, fees, feasibility, or full portfolio returns. Duplicate
  ticker actions cannot multiply its reward.
- **Historical price labels:** use the latest valid cached close on or before the
  requested horizon, no more than seven calendar days earlier. Store the actual
  evaluation date. No live-price fallback; missing history or benchmark data
  leaves the outcome pending instead of generating an incomplete lesson batch.
- **Thesis evidence:** include only news both published and cached inside the
  inclusive UTC date window, ending at the earlier of the actual evaluation date
  and the requested horizon. Undated/unparseable news is excluded. Repeated cache
  copies do not crowd distinct headlines out of the evidence cap.
- **Decision identity:** each thesis judgment carries an outcome ID. Distinct
  decisions on the same ticker retain their own evidence and verdicts. Legacy
  ticker-only responses are accepted only when the ticker is unique in the batch.
- **Inactive candidates:** `fund optimize` writes an instruction artifact and guidance
  manifest under `config/compiled/candidates/<fund>/<candidate>/`. The manifest
  records the metric, config fingerprint and incumbent guidance fingerprint.
  The prompt page lists candidates separately from active and archived guidance.
  Compilation never changes the active guidance file.

Existing active guidance and stored outcomes/learnings are not rewritten. Old
labels may still contain the previously permitted price/news leakage. They must
be audited and rebuilt from historical evidence before serving as a clean
promotion dataset. Rebuilding historical labels is a separate, explicit operation.
A news article first cached after a window is deliberately ineligible, even if it
claims an earlier publication date. This conservative rule can reduce evidence
coverage until a reliable historical availability source is implemented.

## Bounded optimizer: inspect cost before running

`fund optimize` now uses a small native instruction-only search, replacing MIPRO's
bootstrapping and few-shot search. It proposes **one** alternative, then evaluates
that alternative and the incumbent on the same historical cases. It emits a
candidate only after every call succeeds and its mean directional score exceeds
the incumbent's. This search reward is still a surrogate, not proof of portfolio
improvement. No DSPy or Optuna installation is needed for this path; the optional
DSPy prototype and older artifacts remain available separately.

Default settings (independent of the live fund's reasoning/output settings):

```yaml
optimizer:
  max_calls: 7
  max_total_tokens: 200000
  max_output_tokens: 2048
  reasoning_effort: low
  validation_runs: 3
```

The chronological 80/20 split remains; at most three runs from the validation
portion are used, selected by date before performance is examined. The proposal
uses bounded summaries of up to six training runs; no validation labels enter
that request. Validation calls retain their full historical context, with no
few-shot demonstrations or extra textual chain-of-thought field. The configured
fund model is used for evaluation and, by default, proposal writing. An explicit
`prompt_model_id` still overrides the latter. Live consensus settings are not
changed; search uses one sample per case and requires independent forward review.

First inspect the data, checkpoint path, and complete worst-case reservation:

```sh
FUND_CONFIG=config/config.yaml .venv/bin/fund optimize --dry-run
```

A new run refuses to start if the entire planned search exceeds either limit.
Large historical contexts may exceed the default budget; inspect that result
before explicitly choosing a larger limit. There is no automatic budget increase.
`max_total_tokens` is an **admission-control reservation**, calculated from UTF-8
text bytes, schema text twice, an 8,192-token protocol allowance per request, and
the output cap. It deliberately overestimates normal text tokenization and never
refunds unused output allocations. It is not measured provider usage, an invoice
estimate, or a guaranteed dollar cap. Each provider request also receives the
actual output token cap; OpenAI's completion limit includes reasoning tokens
([official reasoning documentation](https://developers.openai.com/api/docs/guides/reasoning)).
Budgets apply to one search, not the account or all scheduled funds combined.

When satisfied with the displayed limits, invoke `fund optimize` without
`--dry-run`. SDK retries are disabled. Billing/quota, transport, truncation and
schema failures stop immediately; no fallback model calls or zero-valued scores
hide failures. Small output limits can cause truncation; this stops the search
rather than silently paying for a larger response.

Each request is reserved on disk **before** transmission. Completed structured
responses and proposals are saved atomically under
`config/compiled/searches/<fund>/<plan-hash>.json`, together with the frozen
training/validation selection, settings and cumulative reservations. A per-file
lock prevents concurrent callers from executing the same search. Successful
responses are reused on resume, and completed searches return their existing
result rather than starting over.

```sh
fund optimize --resume config/compiled/searches/fund/PLAN_HASH.json --dry-run
fund optimize --resume config/compiled/searches/fund/PLAN_HASH.json
```

Failed/interrupted requests remain charged against both budgets because they may
already have been billed. Retrying one requires explicit `--retry-failed`; if the
extra attempt would exceed the cap, explicitly increase the **cumulative** limit:

```sh
fund optimize --resume config/compiled/searches/fund/PLAN_HASH.json \
  --retry-failed --max-calls 8
```

The token reservation cap can similarly be changed with `--max-total-tokens`.
Changing model, mandate, incumbent guidance, output cap, reasoning effort, or
validation count rejects resume: those change the experiment and require a new
search. Resumes use the saved data even if more history has since arrived.
Checkpoints are local recovery records with checksums, not tamper-proof receipts.
Do not delete a checkpoint to bypass limits. Ctrl+C preserves completed work;
a hard process kill can leave an uncertain in-flight request that also requires
explicit retry authorization. Older MIPRO runs cannot be resumed through this
checkpoint format, and their cache is not imported.

## Frozen paired comparisons

New weekly snapshots archive the structured features for every shown candidate,
portfolio marks, risk limits, fees, model settings, mandate, guidance, and FX.
The prompt renderer now retains the full upstream screen instead of silently
cutting it to 75 names. Older snapshots without this context are rejected; current
market data is never substituted to fill historical gaps.

Run an explicit comparison against a saved weekly run:

```sh
fund compare-guidance --run-id RUN_ID \
  --candidate config/compiled/candidates/FUND/CANDIDATE/guidance.json \
  --output comparison.json
fund score-guidance comparison.json --outcomes outcomes.json --output score.json
```

Use the same fund configuration/environment as the saved run for the first
command. It makes **two sets of paid model calls** with the archived sample count.
Both arms use the same frozen user prompt, model settings, production consensus,
and guardrails. Only decision guidance changes. Incomplete samples and infeasible
allocations are recorded as invalid, making the pair unscoreable. Neither command
books trades or activates guidance. Reports require a new output filename.

The scorer is offline. Supply an outcomes file shaped as follows (illustrative
only; actual tickers, dates, horizon, and hash must match the comparison):

```json
{
  "case_hash": "COPY_FROM_COMPARISON",
  "basis": "SEK",
  "benchmark": "^OMXSPI",
  "source": "Describe the price/FX source and valuation convention",
  "observations": [
    {"date": "2026-01-01", "prices": {"A": 100, "B": 100}, "benchmark": 100},
    {"date": "2026-01-02", "prices": {"A": 95, "B": 80}, "benchmark": 99},
    {"date": "2026-01-03", "prices": {"A": 90, "B": 120}, "benchmark": 101}
  ]
}
```

Provide every calendar date through the exact matured horizon, explicitly
carrying non-trading marks forward, and **every shown ticker**, including names
neither arm bought. Initial marks must match the frozen snapshot; subsequent
marks should use consistent closing valuations. For SEK cases, convert all
security valuations to SEK at each observation and supply benchmark levels on
the same currency basis. For synthetic simulations, copy `synthetic_native` from
the case instead. The importer checks shape, completeness, identity and finite
values; it cannot independently establish the accuracy or historical availability
of a user-supplied source. Splits, distributions and other corporate actions need
consistent total-return treatment in the supplied valuation series.

Execution is an explicit **fractional-share, buy-and-hold research convention**:
targets are sized from initial NAV at archived quotes, sells precede buys, estimated
fees reduce cash, omitted holdings remain invested, and cash earns zero. Extra
cumulative checks cover cash, position count, turnover, sector and allocation
ceilings. Unknown sectors prevent buys when the sector ceiling cannot be checked.
Existing concentration breaches may be reduced or retained, but not increased by
buying into the affected bucket. This is not broker execution parity: slippage,
lot sizes, intraday exits, ongoing rebalancing and funding costs are not modeled.

Scores report net portfolio return, benchmark-relative return, daily valuation
drawdown, turnover, fees and candidate advantage, with artifact hashes and source
lineage. Hashes detect accidental changes; they are not signatures or proof of
when a decision was made. Historical replays are labeled `retrospective_diagnostic`;
`same_day_shadow` requires the candidate to predate the frozen case and both calls
to finish on that UTC day. This label alone does not establish unseen evidence.
**Every report remains ineligible for promotion.**

## Automatic forward collection (opt-in)

Configure one inactive candidate in the fund's YAML file:

```yaml
shadow:
  candidate: config/compiled/candidates/FUND/CANDIDATE/guidance.json
  benchmark_currency: SEK
  benchmark_calendar: XSTO
```

Currency and calendar must describe the configured benchmark; the example is for
`^OMXSPI`. Do not reuse these values blindly for another index. Relative candidate
paths resolve from the repository root. Profiles are disabled by default, and
none were enabled as part of implementation.

A saved `fund run` now registers the frozen case and candidate before running two
additional model sample sets. The candidate must predate the case; the case must
be less than two hours old, with known exchange calendars for all shown names.
Dry runs never register comparisons. Shadow failures are reported separately
from the normal portfolio run. Registration exclusively reserves that run ID;
failed, interrupted or invalid model runs are retained and never automatically
rerolled. A reservation that lacks `comparison.json` needs inspection; the next
scheduled weekly run will create a separate experiment.

Registered experiments live under `data/shadow/<fund>/` beside the fund database
(or the corresponding database parent's `shadow/` directory). Each contains an
immutable registration and original comparison. Collection runs at the end of
subsequent saved weekly runs, even if the candidate has since been disabled. It
can also be triggered without any model calls or trading:

```sh
fund collect-guidance
```

No new scheduler/service is installed. Existing weekly runs drive collection;
this standalone command can also be used by an operator between weekly runs.

The forward convention differs from historical replay: **execution is the first
common trading date strictly after both decisions finish**, across all shown
securities and the benchmark. If no common session exists within 14 days, the
case remains pending. The starting book is revalued at that future close, approved
target weights are applied with fees, and cumulative feasibility is checked
again. The evaluation horizon starts at this entry date. Collection waits until
the following UTC day after the full horizon to avoid partial closing bars.
Returns before entry are excluded from the scored period.

Collection fetches adjusted daily histories directly through yfinance for every
shown security, the exact benchmark, and (for SEK cases) historical currency/SEK
rates. It archives the provider rows, dividends/splits, request range, currency,
retrieval timestamp and provider version. Adjusted-close ratios include the
provider's treatment of splits and cash distributions; histories are normalized
to the archived quote to avoid artificial losses when older prices are revised
for a split. These are **synthetic total-return units**, including reinvestment
implicit in the provider's adjustment, not raw-share broker fills. Benchmark
returns follow the configured index: an adjusted price index is still a price
index, not a dividend-inclusive total-return index.

Only confirmed exchange closures permit carrying a security/index close forward.
Missing open-session bars, unverified currencies, duplicate/invalid prices,
unknown calendars or infeasible entry allocations leave the case pending. FX
permits weekend carry only; a missing weekday FX bar also leaves it pending.
Minor-unit currency mismatches such as GBP versus GBp are rejected rather than
silently converted. Missing or delisted alternatives are not dropped, so some
cases require a better data source before they can ever score.

A collection attempt archives `history.json`, then on success a derived
`valuation_case.json` and `outcomes.json`. The derived case is accounting context;
the original decision prompts and model responses remain in `comparison.json`.
Failed collection attempts retain their errors. Successful `result.json` files
bind registration, original comparison and history hashes, and are never
recomputed on later invocations. Publication is atomic and refuses overwrites;
per-case locks prevent concurrent collectors. To investigate provider revisions,
use the preserved evidence rather than replacing an existing score.

Yahoo's adjustments are provider claims, not independently audited corporate
actions. There is no delisting settlement feed, tax model, slippage model, lot
rounding or intraday execution simulation. Collection timestamps establish when
this system retrieved evidence, not when the provider first published it. All
forward results therefore remain **ineligible for automatic promotion**.

## Aggregate evidence audit

```sh
fund evaluate-guidance --output evidence.json
# Or inspect full directories for several funds (results remain separate):
fund evaluate-guidance --root data/shadow/fund --root data/shadow/fund_global \
  --min-periods 8 --output evidence.json
```

This offline command scans every reservation in each supplied fund directory,
including interrupted recordings and cases without results. It verifies artifact
hashes and chronology, reconstructs future-entry allocations and daily outcomes
from the archived provider history, and recomputes performance. Modified or
missing evidence is reported as invalid, not silently skipped. Identical copied
experiments count once; conflicting copies invalidate that experiment. Output
must be a new file. No network, model, trading or promotion calls occur.

Groups separate fund, exact candidate artifact, full incumbent instructions,
mandate, model settings, risk limits, fees, benchmark, currency convention,
horizon and execution convention. There is no pooled return across funds or
candidates. Within each group, entry dates choose the earliest available period,
then exclude any subsequent period whose inclusive date window overlaps it.
Ties use recorded decision time and registration hash. This selection uses dates
before inspecting scores; failed or missing periods still reserve their window.
Thus an overlapping winner cannot replace an inconvenient missing observation.
Non-overlap reduces repeated exposure but **does not prove statistical independence**.

The report includes every case status, selected and overlapping registrations,
coverage of matured selected windows, mean/median/worst paired advantage, wins,
mean net and benchmark-relative returns, worst period returns, drawdown, turnover
and fees for both arms. These are descriptive averages of separate experiments,
not a compounded backtest or an annualized investment return. Cases with unknown
windows remain visible and prevent a group's positive label.

Labels are exploratory:

- `incomplete_evidence`: the group has unresolved or invalid cases, even when
  some remaining periods performed well.
- `insufficient_periods`: fewer than the requested number of scored,
  non-overlapping periods (default eight; not a statistically validated threshold).
- `mixed_results`: enough periods, but return, drawdown or turnover checks do not
  all support the candidate.
- `promising_descriptive`: positive mean and median advantage, more wins than
  half the sample, no worse mean/worst drawdown, and no higher mean turnover.

No label establishes significance or authorizes promotion. Waiting cases are
shown but excluded from the matured coverage denominator. A report can audit only
the directories supplied: it cannot detect deleted or intentionally omitted
experiments, prove a source was historically available, or correct provider data.
Invalid artifacts produce a nonzero CLI exit after saving the audit report.

## Next milestone: validated promotion

There is no promotion command yet. Do not copy an unevaluated candidate over the
active guidance. The optimizer still searches with the interim directional metric;
portfolio scoring is a separate comparison step, not its training reward.

1. Validate the provider against independent corporate-action and delisting data;
   expose evidence gaps and coverage before interpreting returns.
2. Group validation by decision period across funds and exclude training outcomes
   unavailable at the validation decision time. Stratify mandates and horizons.
   The optimizer's existing 80/20 split is search validation, not promotion proof.
3. Validate promotion thresholds on held-out forward evidence, including uncertainty
   and repeated candidate selection. Descriptive aggregation is implemented, but
   does not establish independence or significance. Retain the incumbent when
   results are inconclusive. Promotion must bind evaluated artifact
   hashes, support rollback, and fail if the candidate or incumbent has changed.
4. Compare base mandate, selected lessons, and optimized guidance separately to
   establish which learning channel adds value. Then add structured provisional,
   active, and retired lessons with applicability and contradicting evidence.

Regression tests use temporary stores, mocked providers and mocked model output.
They establish accounting and isolation behavior, not investment performance.


### Avoiding repeated search spending

The default `optimizer.min_new_periods: 3` requires three new distinct evaluated
own-fund decision dates after a paid search, including an interrupted search.
Pooled funds do not advance this gate. The initial outcome/example minimums still
apply. Dates are a scheduling proxy, not proof of independent evidence. Scheduled
weekly commands can therefore check eligibility without paying every week.
Resume interrupted work explicitly; `--force-search` bypasses only this scheduling
gate and records the override, preserving call and token limits.

With `optimizer.reuse_evaluations: true` (default), successful search evaluation
responses are cached under `config/compiled/request_cache/`. Keys include exact
system/user content, model settings, output schema, schema hint and transport
version. Proposals are not shared. Identical requests across searches or funds
sharing this directory can reuse one response; changed requests make fresh calls.
Scores are recomputed against the current labels. Reused responses are the same
sample, not additional independent evidence. This cache does not change production
or forward evaluation calls, and does not automatically span other projects.

`FUND_CONFIG=config/config.yaml .venv/bin/fund optimizer-usage` is an offline report
across saved searches in the configured compiled directory. It reports provider
input/output counts and available cache/reasoning subsets, without double-counting
local cache hits. Reasoning is part of output; cache reads/writes are part of input.
Missing counters and interrupted calls without a usage response remain unknown.
The report does not reconstruct old MIPRO costs or other application spending and
is not an invoice or a dollar budget. Conservative admission reservations remain
separate from provider usage. Keep checkpoints and caches private and preserve
checkpoints for accounting; deleting them also removes scheduling history.

### Inspect and compare compact historical context

Every `fund optimize --dry-run` now prints UTF-8 byte totals for each selected
validation field, plus full/compact user-context sizes. These are not tokenizer
counts. Field totals count each selected case once; reservations also include
both evaluations, the proposal, schema/protocol allowance and output ceilings.
The original `full` mode remains the default.

The experimental `--context-mode compact` uses `asset_templates_v2`: repeated
metric-row labels become shared templates with exact string values per asset.
It also considers the earlier exact-line deduplication, choosing the smallest
representation, including the original text. All characters, numbers, attribution, uncertainty and ordering can be
reconstructed exactly. The mandate remains unchanged. Unique text is not dropped,
ranked, truncated or paraphrased. If the representation is not smaller, the
original text is used. The original fields and packed inputs stay in the frozen
checkpoint, and outcome labels never enter evaluation context. This provides no
promise of savings on mostly unique evidence, or of identical model behavior.

First inspect the real history without API calls:

```bash
FUND_CONFIG=config/config.yaml .venv/bin/fund optimize --dry-run
FUND_CONFIG=config/config.yaml .venv/bin/fund optimize --context-mode compact --dry-run
FUND_CONFIG=config/config.yaml .venv/bin/fund optimize --context-mode compare --dry-run
```

`--context-mode compare` runs up to two requests per selected validation case,
using the same incumbent guidance/model/settings with full and compact inputs.
It creates no proposal or guidance candidate. Remove `--dry-run` only when ready
to pay for that bounded diagnostic. The existing call/token ceilings, persistent
request cache, explicit failed-request retries, and `--resume` all apply. Resume
uses the frozen context mode; a conflicting explicit mode is rejected. Comparison
calls appear in `optimizer-usage`, but do not consume the new-evidence scheduling
gate for instruction searches.

The checkpoint's `comparisons` records action details and cash-target differences;
`plan.cases` retains the frozen risk limits and evidence for manual review. Check
changed tickers, buy/sell/hold choices, position sizes, stops/targets, missing or
misattributed evidence, and compliance with those original risk limits. One small
paired sample cannot establish risk or performance equivalence, and model sampling
can also cause differences. Cached results are reused observations. No comparison
automatically enables compression. Production and forward evaluation retain their
existing context; compact search winners still require full forward evaluation.

### Small learning requests, compact evaluations, optional OpenAI batches

The proposal request contains the mandate, current guidance and bounded training
summaries, **not the asset universe**. The universe belongs to the historical
candidate/incumbent evaluations. Dry runs now split those reservations and show
how much universe text consists of recognized metric rows versus other text
(names, news, warnings and unrecognized historical formats).

Asset templates keep all supplied eligible tickers and every rendered value,
including signs, units, dates and held-position markers. They do not round numbers
or use later outcomes to select assets. News, contradictions, uncertain evidence,
names and warnings remain literal. Savings depend on the actual history: large
unique news snippets will remain large. Frozen old checkpoints retain their saved
representation on resume. Compact format is still experimental and opt-in; test
full versus compact behavior before relying on it.

For OpenAI funds, `--execution batch` sends outstanding historical evaluations via
the [OpenAI Batch API](https://developers.openai.com/api/docs/guides/batch), which
provides 50% lower input/output token pricing with a 24-hour processing window.
The one proposal call stays direct. The model, reasoning effort and output limits
remain configured as before. Anthropic batch execution is not implemented here;
selecting it fails before the proposal. Nothing automatically changes cron jobs.

```bash
# Free inspection of both context size and reservations:
FUND_CONFIG=config/config.yaml .venv/bin/fund optimize --context-mode compact --execution batch --dry-run

# When the inspected plan fits the budget, remove --dry-run to start it.
# Submission prints a checkpoint path; collect later without submitting again:
FUND_CONFIG=config/config.yaml .venv/bin/fund optimize --resume /absolute/path/to/checkpoint.json
```

Batch pricing changes dollars per token, not the token reservation. An oversized
plan still refuses to start. No API dollars are estimated and no limits are raised
automatically. Each batch item counts toward the call cap, not just the HTTP batch
submission. Reservations are saved before submission and never refunded on failure.
Completed cached responses are reused, and identical requests within a batch are
coalesced. Separate simultaneously pending searches are not globally coalesced.

Pending batches exit without waiting and resume checks their status once. Results
are matched by `custom_id`, not line order. Successful results and usage are saved
before scoring. Missing, refused, truncated and invalid responses cannot become
zero scores or incomplete winners. Partial failure stops; explicitly use
`--resume ... --retry-failed` and sufficient cumulative limits to batch only the
remaining failed requests. There is no direct evaluation fallback.

An ambiguous submit timeout never automatically resubmits, even with
`--retry-failed`. Recover the batch ID in the provider dashboard and use
`--resume ... --batch-id batch_...`; its input file and optimizer metadata must
match the checkpoint. Preserve checkpoints while batches run. Retrieve completed
results promptly, before provider file expiry. Cancellation can be performed in
the provider dashboard; resume collects any completed items and records failures.

Batch can also run `--context-mode compare`; it creates no proposal or candidate.
That diagnostic remains a paid bounded comparison and does not automatically
activate compact context. `optimizer-usage` includes saved per-item provider usage
for both normal and batch calls, including available usage on invalid responses.

### Token counts and an advisory dollar estimate

`fund optimize --context-mode full --execution batch --dry-run` now reports a
whole-plan token and USD estimate separately from the byte-based admission
reservation. Install the updated normal dependencies to obtain `tiktoken` (for an
existing environment, `uv pip install --python .venv/bin/python -e .`). Its first
use may download a public tokenizer vocabulary; prompt text is tokenized locally,
without a model request or upload of fund data.

The initial verified price snapshot supports OpenAI `gpt-5.6-sol`: $4/M input,
$20/M output, checked 2026-09-23 against the official model page. Batch evaluations
receive the 50% batch discount; the proposal remains standard-priced. The
proposal's configured model is priced separately, never assumed identical to the
evaluation model. Long-context multipliers apply per request above 272,000 tokens.
Prices expire for display after 2026-10-23 and require source review; unknown
models, missing tokenizers and unsupported providers give an explicit unavailable
result, never a guessed or partial dollar total. There is no local Claude-tokenizer
substitution and no automatic external token-counting request.

The count uses the model's tiktoken encoding on actual system/user text, the textual
schema hint, and the strict response schema, plus a 32-token estimated protocol
allowance per call. This is not the provider's exact billed count. A fresh plan
cannot know the candidate instruction yet: the range adds up to 16,000 input
tokens per candidate evaluation (the 4,000-character instruction limit in UTF-8).
A saved proposal uses its actual instruction text instead. The low dollar scenario
uses known inputs with zero output; the high scenario uses that candidate allowance
and the full configured output allowance, including reasoning. Actual serialized
schema/framing can differ; the displayed range is **not a billing cap**.

The report covers the whole plan, including on resume, before local/provider cache
savings. It is not remaining spend or a bill. It excludes retries, taxes, regional
surcharges and cache-write premiums. It never changes call/token reservations,
automatically raises limits, or starts a paid run. Inspect the estimate first;
only explicitly adjust `--max-total-tokens` once the cost is acceptable. The actual
provider counts remain available through `fund optimizer-usage` after execution.

### Automatic batch collection and Telegram notification

`fund optimizer-watch` checks saved batch searches for the selected `FUND_CONFIG`.
It collects existing remote results, finishes scoring, and sends a Telegram message
when complete or when manual intervention is required. It does not create a search,
generate a proposal, submit a batch, retry failed model requests, or activate a
candidate. Pending batches and busy fund locks do not trigger notifications.
Collection also works if the current configured token reservation is below the
amount previously reserved by an explicitly authorized run; no new reservation is
made. Changed model/mandate/guidance settings still block scoring for review.

The existing `.env` values `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are used.
Successful delivery receipts live in `config/compiled/searches/<fund>/notifications/`.
Failed Telegram delivery is retried at the next check without repeating model work.
Receipts suppress repeated notifications under normal operation; a crash between
Telegram accepting a message and receipt persistence can cause a duplicate.
Transient provider connection failures are logged and checked again next time.
Terminal batch failures produce an attention notice, never automatic paid retries.

After deploying the code, run `.venv/bin/fund optimizer-watch` once to collect/check
existing batches. On the Pi, add the `optimizer-watch` entry from
`deploy/cron.example` to the existing crontab with `crontab -e` (do not overwrite
other scheduled tasks). It runs every 15 minutes. The repository example does not
install or modify the Pi's live crontab automatically. Add a corresponding entry
for each other OpenAI fund you want watched. Results need to be retrieved before
the provider's output files expire.
