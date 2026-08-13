# Monitoring a real sleeve as a mirror portfolio

This is how the **KF Chokepoint Satellite** (Montrose KF 2561058) is watched:
as a paper "mirror" portfolio tracked at real Yahoo prices, with four Telegram
watches running daily. The monitor never touches the broker — you execute in
Montrose, then record fills so the mirror stays honest.

## 1a. Create a sleeve from a photo of your portfolio

The fastest path from "what I actually hold" to "a watched book" — no JSON, no
typing. On the **Live** section (`/live`), the **Import from a photo of your
portfolio** panel takes one or more screenshots of a broker holdings screen
(Montrose, Avanza, Nordnet, IBKR, Degiro…) and turns them into positions:

1. **Read.** Each image goes through local OCR (free, `tesseract`) *and* a
   vision model, together in one prompt — the OCR transcript is the numeric
   cross-check, the image preserves the table layout. Set `FUND_VISION_MODEL`
   to change the model (default `gpt-4o-mini`, needs `OPENAI_API_KEY`).
   Multiple screenshots of a scrolled list are merged and de-duplicated.
2. **Resolve.** Each row is matched to a Yahoo symbol: ISIN from
   `universe.csv` → broker→Yahoo map (`KOG`→`KOG.OL`) → company-name alias →
   Yahoo symbol search. Rows that don't resolve come back with an amber ticker
   box for you to fill in or untick.
3. **Review.** Everything lands in an editable table — ticker, shares, cost
   basis, plus a **kill criterion**, a **max drop** and a **time horizon** per
   stock. *Nothing is written until you press Import.*
4. **Seed.** Each holding is booked at the share count and SEK cost basis you
   confirmed (fee 0 — those were paid at the real broker), so P&L runs from
   your actual entry rather than today's price. No orders are placed anywhere.

Cost basis is read as **Inköpsvärde ÷ antal** where the screen shows it, since
that is the SEK you actually paid; a native GAV is used only when the row is
already in SEK. A row where neither was visible falls back to market value and
is flagged in amber — check those against the broker before importing.

## 1b. Kill criteria and time horizons

Every position carries up to three pre-registered exit conditions, set on the
photo review table, in the dashboard's **Set a kill criterion & time horizon**
editor (the pencil on any Watch-status row pre-fills it), or from the CLI:

```bash
fund paper-plan kf-chokepoint-satellite NVDA \
    --kill "loses the Apple socket" --max-drop 25 --months 12
fund paper-plan kf-chokepoint-satellite NVDA --clear
```

| Condition | Set with | Checked by |
|-----------|----------|------------|
| **Kill criterion** (text) | what would falsify the thesis | daily judge (gpt-4o-mini) over an evidence pack |
| **Kill rules** (numeric) | max % drop from the review anchor, price floor, price target | prices, every run — no API key needed |
| **Time horizon** | a date, or N months out | days remaining, every run |

#### What a drawdown line is measured from

Not cost. A max-drop line anchors to the price **on the day you set it**, because
that is when the thesis it tests was written. Bought at 40, reviewed at 100, now
70: that is −30% against a −25% line — a kill — even though the position is still
+75% on cost and would look untouched from there. The anchor is the same idea the
add plan's **review price** already uses for dislocation, and a new drawdown line
falls back to it when no price is cached yet.

Three rules keep the anchor honest:

- **Editing the threshold does not move it.** Tightening 25% → 20% is a change of
  mind about the line, not a fresh review; re-anchoring there would quietly reset
  a position already halfway to its kill.
- **Moving it is a deliberate act** — the *Re-anchor to today* button, or
  `fund paper-plan <slug> <TICKER> --re-anchor`. Do it after you have actually
  re-run the analysis.
- **Rules written before anchors existed keep measuring from cost**, and say so
  in the badge and in the alert text. Silently re-anchoring an armed kill line
  would have disarmed it.

The badge on the Watch panel carries both the line and the live number against
it (`−25% from review · −12%`), and the P&L column stays measured from cost — two
different questions, so they are no longer mixed in one cell. Panel and daily
watch both call `watchplan.drawdown_for`, so a line cannot read breached in one
and clear in the other.

### How your criterion text is read

The software never parses the string itself — it is stored verbatim and handed
to the model. To make that reading visible, saving a criterion runs **one
analysis call** and shows how it decomposed, right in the Watch panel:

```
Recurring growth falls below 8% while EBITA/FCF deteriorates, or
acquisitions begin generating clearly weaker returns
▸ read as (A and B) or C
   [figure] recurring growth falls below 8%   → revenue growth below 8
            ⚠ total revenue growth, not recurring specifically
   [figure] EBITA/FCF deteriorates            → profit margin below 9
            ⚠ net margin — EBITA and FCF are not in the data set
   [manual] acquisitions begin generating clearly weaker returns
            ⚠ deal-level returns are not disclosed in any feed
```

Each condition is tagged **figure** (a threshold on a metric we cache),
**news** (an event that could show up in coverage) or **manual** (needs the
full report). The ⚠ lines are the analyser stating how the available metric
differs from what you wrote — a proxy is used, but never silently.

**The metric menu is what financedata fetches and Yahoo actually populates:**

| Group | Metrics |
|-------|---------|
| Growth | revenue growth, earnings growth |
| Margins | gross, operating, EBITDA, net |
| Returns | ROE, ROA |
| Cash & debt | free cash flow, operating cash flow, total cash, total debt, debt/equity |
| Capital structure | equity/assets, net debt/assets, interest coverage, cost/income, FCF/net income |
| Valuation | EV/EBITDA, P/E, forward P/E, price/book, dividend yield |

Cash and debt figures are in the company's own reporting currency, so prefer a
sign test ("free cash flow below 0") over an absolute amount.

The **capital structure** row is not in `.info` at all — that payload is a
normalised snapshot with nothing about leverage or solvency, so a criterion
phrased "net debt/assets below 50%" had no field behind it. Those five are
derived from the full statements (`fundmgr/data/statements.py`), taking balance
-sheet stocks from the interim sheet and flows from the annual one, and merged
into the same fundamentals cache. Everything downstream reads them without
knowing the difference.

Two more are derived but deliberately **not** offered, because a derived figure
is not automatically the company's figure:

| Metric | Derived | Balder reports | Offered? |
|--------|---------|----------------|----------|
| equity/assets | 37.0% | 37.0% | yes — exact |
| net debt/assets | 52.8% | 50.4% | yes, +2.4pp — reads leverage high, which kills early rather than late |
| net debt/EBITDA | 10.4x | 12.8x | **no** — property-company EBITDA carries revaluation gains the company excludes; it flatters |
| cash runway | — | — | no — only defined while operations burn cash |

Set a leverage gate more than ~3pp away from the current ratio, or it will trip
on the debt definition rather than on the business.

Two rules keep this list honest, both enforced in `tests/test_evidence.py`:
every metric offered must exist in financedata's `_FIELD_MAP`, and a mapped key
that Yahoo returns null for does **not** belong here — `price_to_sales` is
mapped but omitted for exactly that reason. Verify with:

```bash
fund fundamentals-check VOLV-B.ST
```

It forces a refresh and reports drift in both directions: metrics offered but
missing from the payload (broken — a rule on one could never evaluate), and
fields the data layer returns that no criterion can use yet. Run it after any
change to `_FIELD_MAP`: yfinance `.info` is a passthrough of Yahoo's payload,
so a mis-mapped key name fails silently rather than raising.

Where it finds a real threshold, the panel offers **"Also check N of these as
a hard rule"**. One click lifts them into deterministic fundamentals rules
(`metric` + `below`/`above` + value) checked against the cached figures every
run — no LLM, no interpretation. The text criterion is left exactly as written,
so the same line is both judged in context *and* compared against a number.

Set the same thing from the CLI with repeated `--fundamental`-style entries via
the dashboard, or inspect what is stored with `fund paper-plan <slug> <TICKER>`.
The analysis needs `OPENAI_API_KEY`; without it the criterion still saves and is
still judged, only the preview is missing.

A metric with nothing cached is reported as **unread**, never as passing.

### What the text judge reads

A criterion is only as good as the evidence it's judged against, so the judge
gets an **evidence pack** (`fundmgr/evidence.py`), not just headlines:

- **News with substance** — headline, publisher, date and the article
  *summary*, plus a best-effort fetch of the article body for the items whose
  summary is thin. Set `FUND_KILL_FETCH_ARTICLES=0` to skip body fetching.
- **Fundamentals with a trend** — the cached growth/margin/cash figures and how
  they've moved since the oldest snapshot on file, so "deteriorates" is
  measurable. `fund paper-track` refreshes and snapshots these; a snapshot
  taken *today* is never used as the earlier reference, so day one reports "no
  history" rather than a false "flat".
- **The position** — shares and cost basis.

The judge decomposes a compound criterion first (`A while B, or C` → all of
A and B, *or* C) and returns one of three verdicts:

| Verdict | Meaning |
|---------|---------|
| **YES** | evidence positively shows it met — Telegram alert, recorded as a kill hit |
| **NO** | the evidence covers the question and the thesis is intact |
| **INSUFFICIENT** | the criterion needs facts this evidence can't settle |

That third verdict is the point. A criterion like *"recurring growth falls
below 8% while EBITA/FCF deteriorates, or acquisitions begin generating clearly
weaker returns"* has legs that only the quarterly report settles. Previously
that returned NO — indistinguishable from a healthy thesis. Now it's reported
as unverifiable: flagged **not auto-checkable** on the dashboard with the
specific conditions it couldn't check, and pushed to Telegram **once per
wording** (rewording re-notifies, since a new phrasing may be checkable).
Those criteria still get quoted back at you around each earnings print.

Kill-rule alerts are transition-based: one Telegram push when a line is
crossed, then silence until the position recovers past it by 2 points — a
name sitting on its line doesn't nag daily. Editing a rule re-arms it.

Horizons escalate through **30 / 14 / 7 / 1 days and the day itself**, one
alert per stage, so a 12-month horizon produces five nudges over its life
rather than a daily countdown. A horizon reached is a prompt to **re-run the
thesis**, not a sell signal — the message says so.

Check them on demand (both also run inside `fund paper-track`):

```bash
fund paper-watch                 # kill lines + horizons, all books
fund paper-watch --slug my-sleeve
```

The dashboard's Watch-status panel shows the live drop against each drawdown line
and the days left on each horizon, most urgent first.

## 1. Create the mirror from a structured LLM answer

Save the picks JSON (the Fable answer, with its `positions[]`,
`portfolio_kill_criterion`, and `excluded_holdings`) somewhere local, then:

```bash
fund paper-import path/to/kf_chokepoint.json
```

`paper-import`:

- maps broker/Montrose tickers to Yahoo symbols
  (`KOG`→`KOG.OL`, `ENR`→`ENR.DE`, `BESI`→`BESI.AS`, `ASML`→`ASML.AS`; bare US
  symbols like `TSM`/`NVDA`/`GEV`/`VRT`/`CEG`/`SKHY` pass through — SK Hynix
  trades as `SKHY` on NasdaqGS in USD).
- **drops `excluded_holdings`** (SELLAS) entirely — never bought, never sized.
- stores per-position kill criteria, **target weights**, per-position notes
  (`watch`, `next_earnings`), and the **portfolio-level capex kill criterion**.
- **imports the plan only — it does NOT buy.** The sleeve starts 100% cash;
  positions appear as you record actual fills (below). Pass `--execute` to
  `paper-import` (or nothing on the CLI) only if you want every position opened
  at live prices immediately. The watches run against the *plan* (target
  tickers), so you get kill-criterion and earnings alerts before you've bought.

Capital defaults to `meta.deployable_capital_sek`; override with `--capital`.

Or import from the web: the **Live** section (`/live`) has a "Import a sleeve
from JSON" form that does the same thing (kind=`live`, plan-only). Live sleeves
get real-money framing (never "paper / not real money") and their own dashboard
with a **Watch-status panel** — the capex kill meter, the full plan (intended
tickers, weights, kill lines, next earnings), and per-position weight drift once
held — separate from the `/paper` simulation section.

## 2. What gets watched (daily, via `fund paper-track`)

Runs from cron after NYSE close (see `deploy/cron.example`). Each pushes to
Telegram only when something fires:

| Watch | Fires when |
|-------|-----------|
| Per-position kill criteria | Recent news plausibly meets a position's pre-registered kill line |
| **Numeric kill rules** | A position drops past its max-drop line from the review anchor, or through a price floor / up to a price target |
| **Time horizons** | A position's review date is 30 / 14 / 7 / 1 days out, or has arrived — time for fresh analysis |
| **Portfolio capex kill** | 1 of the 5 largest hyperscalers guides 2027 capex flat/down → **warning**; 2+ → **KILL: halve the compute cluster** |
| **Earnings calendar** | Day before/of a holding's report → heads-up (quotes its `watch` + kill lines); day after → check-the-print reminder |
| **Weight drift** | A position appreciates past **1.5× its target weight** (rebalance rule); re-arms after it falls back below 1.4× |

The news/capex judges need `OPENAI_API_KEY` (gpt-4o-mini); they skip cleanly
without it. The numeric kill rules and the horizon watch need no API key at
all. All watches no-op on portfolios that don't carry the relevant config, so
they're safe to run across every paper book.

## 3. Record fills as you execute the tranches

Same workflow as the main fund, pointed at the mirror book. In Telegram:

```
/plist                         # find the slug, e.g. kf-chokepoint-satellite
/ptarget kf-chokepoint-satellite   # route fills + screenshots here
/pfill VRT 20 610.00 39.00     # or just send a Montrose confirmation screenshot
/pstatus                       # snapshot
/ptarget off                   # switch back to the main fund
```

Prices are entered in **SEK** (the KF account settles in SEK), exactly like
`fund fill`. On the CLI: `fund paper-fill <slug> <TICKER> <SHARES> <PRICE> <FEE>`.

Or from the browser: the live dashboard has a **"Record a fill"** form
(ticker / side / shares / price SEK / fee / optional date). Use it to **seed a
holding you already own** — enter the shares at your Montrose average cost with
fee 0 — and to log buys and trims (a trim is a Sell). Positions and drift update
immediately.

**Ticker auto-snap:** a fill ticker is matched onto the book's plan symbol when
there's a single base match — type `ASML` and it records `ASML.AS`, `ENR` →
`ENR.DE`, etc. — so a fill lands on the intended, correctly-priced instrument
instead of a mismatched one. A symbol that genuinely isn't in the plan is kept
as-is with a warning. Applies to `fund paper-fill`, the web form, `/pfill`, and
the screenshot flow alike.

**Fixing a mis-tagged holding:** if a position was recorded under the wrong
symbol before snap existed (e.g. a bare `ENR` that the dashboard prices as the
wrong instrument), retag it in place — no cash movement:
`fund paper-retag <slug> ENR` (NEW defaults to the plan symbol, here `ENR.DE`),
or on Telegram `/ptarget <slug>` then `/pretag ENR`.

**Fixing the cost basis of a foreign holding:** if the dashboard P&L on a
USD/EUR position disagrees with Montrose, the seeded cost basis was likely the
purchase price converted at *today's* FX instead of the SEK you actually paid.
Set it to the broker's **Inköpsvärde ÷ shares** (SEK/share):
`fund paper-setcost <slug> GOOGL 3429` or Telegram `/psetcost GOOGL 3429`. Cash
is adjusted by the difference so NAV stays consistent.

## Learned ticker aliases

Photo import resolves each row through ISIN → broker→Yahoo map → name alias →
symbol search, and any of those can miss. When you fix a ticker in the review
table, the correction is recorded against every identifier that row carried —
its ISIN, the broker's own ticker, and the company name — and consulted **first**
on every future import, in any sleeve. Fix `4HY` → `VSSAB-B.ST` once and the
next screenshot resolves it silently.

Aliases live in `data/ticker_aliases.json` (outside the repo, alongside the
databases) and always beat automatic resolution, since a human entered them
while looking at the actual position.

```bash
fund aliases                                   # what has been learned
fund aliases --set SE0000115446 VOLV-B.ST      # teach one by hand
fund aliases --forget 4HY                      # remove a bad one
```

KEY may be an ISIN, a broker ticker or a company name. An ISIN is the strongest
key — it survives renames and re-listings — and is tried before the others.

## Add signals — the other half of monitoring

A kill criterion asks *"has the thesis become materially worse?"*. An add
criterion asks a different question: *"has the probability-weighted return
become materially better?"* — which happens when the business improves, when
the price falls while the business does not, or both.

The naive version of each is a trap. "Good news → buy" adds to an already
expensive stock; "price down 20% → buy" automates catching falling knives. So a
signal is never a trigger on its own:

```
(proof OR dislocation) AND valuation AND weight AND NOT killed
```

| Input | Meaning |
|-------|---------|
| **Proof** | the company-specific fundamental test — organic growth, ARR, adjusted margins. Almost none of it is in any price feed, so it's a **human-confirmed flag with a date**, refreshed at each report. |
| **Dislocation** | price fall since the last **thesis-confirmed review price** — never since original cost. A winner down 15% from its review price is dislocated; measured from cost it looks untouched. |
| **Valuation** | `(target / price) ^ (12 / months_left) − 1` against the book's gate. |
| **Weight** | is there room for a tranche below the ceiling. |

Gates are set per risk book:

| Book | Dislocation | Min expected return |
|------|-------------|---------------------|
| A | −12% | 15%/yr |
| A/B | −15% | 20%/yr |
| B | −20% | 25%/yr |

### States

**HOLD** nothing actionable · **ADD-WATCH** dislocation without proof, or a
valuation that needs recalculating · **ADD** proof confirmed and every gate
clear · **STRONG ADD** proof *and* dislocation together · **KILL** overrides
everything. ADD and KILL never coexist.

Alerts are transition-based: an ADD that stays ADD for six weeks is one
message, not forty.

### Staleness — the safeguard that matters most

The dangerous failure mode is a stock halving because intrinsic value halved,
while the engine compares the new price with a target set before the news and
declares it cheap. So a target price carries the date it was set, and any
thesis-changing event recorded after that date — an earnings report, a kill hit
— marks it **STALE**. A stale valuation suppresses every add signal until fair
value is recalculated. Staleness reads the breadcrumbs the earnings and kill
watches already leave, so noticing a report landed needs no new bookkeeping.

### Three weights, not one

Target is the normal allocation, current is where you are, **max** is the
ceiling justified by an active positive thesis. Passing the gates suggests one
tranche toward max; drifting *above* max is a concentration review, not an add.

```bash
fund paper-add-plan my-sleeve SYSR.ST --book A --max-weight 13 \
    --tranche 1 --target 145 --review-price live
fund paper-add-plan my-sleeve SYSR.ST --confirm-proof   # after a good report
fund paper-adds                                          # current signals
```

Both run inside `fund paper-track`, after the kill watches so a position that
has just tripped a kill can never also be suggested.
