# Monitoring a real sleeve as a mirror portfolio

This is how the **KF Chokepoint Satellite** (Montrose KF 2561058) is watched:
as a paper "mirror" portfolio tracked at real Yahoo prices, with four Telegram
watches running daily. The monitor never touches the broker — you execute in
Montrose, then record fills so the mirror stays honest.

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
- buys at live prices, then the book is tracked like any paper portfolio.

Capital defaults to `meta.deployable_capital_sek`; override with `--capital`.

## 2. What gets watched (daily, via `fund paper-track`)

Runs from cron after NYSE close (see `deploy/cron.example`). Each pushes to
Telegram only when something fires:

| Watch | Fires when |
|-------|-----------|
| Per-position kill criteria | Recent news plausibly meets a position's pre-registered kill line |
| **Portfolio capex kill** | 1 of the 5 largest hyperscalers guides 2027 capex flat/down → **warning**; 2+ → **KILL: halve the compute cluster** |
| **Earnings calendar** | Day before/of a holding's report → heads-up (quotes its `watch` + kill lines); day after → check-the-print reminder |
| **Weight drift** | A position appreciates past **1.5× its target weight** (rebalance rule); re-arms after it falls back below 1.4× |

The news/capex judges need `OPENAI_API_KEY` (gpt-4o-mini); they skip cleanly
without it. All watches no-op on portfolios that don't carry the relevant
config, so they're safe to run across every paper book.

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
