# Moving off FinanceData

**Goal:** the fund fetches its own market data, so it no longer needs the sibling
`FinanceData` repo installed, checked out in CI, or kept on the right branch on the Pi.
DeepSwing keeps using FinanceData; nothing here changes that repo.

**Why:** FinanceData is one more repo that has to be in step for a deploy to work
(`deploy.sh` checks its branch, CI checks it out with its own token, `pyproject`
installs it from `../FinanceData`). Its failures have also been silent from the
fund's side. Its insider reader returned 404 for months. `debtToEquity` was cached
but never reached the fund. The Alpha Vantage fallback has never run for the fund,
because the Pi's `.env` sets `ALPHA_VANTAGE_API_KEY` and FinanceData reads
`ALPHA_VANTAGE_KEY`. Data the decision depends on is easier to trust when it is
fetched, cached and tested in the same repo as the decision.

**What it costs:** the Pi's shared cache (`~/.financedata/cache.db`, 681 MB: 5.1M
price rows, 40k news items, 8.9k fundamentals) stops being shared, so the fund
makes its own Yahoo calls where it used to reuse DeepSwing's. The fund already
keeps its own copy of prices, fundamentals and news in its store (FinanceData's
results are mirrored there today), so that copy becomes the cache. Watch for
Yahoo rate limits after each phase that moves a fetch.

## What the fund uses today

| Area | FinanceData calls | Fund adapter | Size there | Fallback already here | Phase |
|---|---|---|---|---|---|
| Indicators | `rsi`, `pct_return`, `ann_vol` | `data/prices.py` | ~30 of 144 lines | none | 1 |
| FX | `get_fx_rate` | `data/fx.py` | 97 | `_local_rate_to_sek` | 1 |
| Live quotes | `get_live_price(s)` | `data/quotes.py` | 144 | `_local_live_price` (unbatched) | 1 |
| **Fundamentals** | `get_fundamentals`, `ts_to_days` | `data/fundamentals.py` | 155 | **moved (phase 2a, done)** | 2a ✓ |
| Prices | `get_prices_since`, `get_cache().get_prices` | `data/prices.py` | 387 + cache | none | 2b |
| Macro | `get_macro_indicators_cached`, `build_macro_block`, `Indicator` | `data/macro_context.py` | 231 | none (headlines are already local) | 3 |
| News + FinBERT | `get_news_cached`, `score_sentiment`, `score_and_save`, `get_cache().get_news` | `data/news.py` | 817 | none | 4 |
| Universe | `data/universe_snapshot.csv` (EODHD) | `scripts/sync_universe_from_financedata.py`, run by `deploy.sh` | 605 + snapshot | committed `config/universe*.csv` | 5 |
| Install, CI, deploy | path dependency, CI checkout, branch check | `pyproject.toml`, `tests.yml`, `deploy.sh`, `deploy/test-branch-check.sh`, 4 test stubs | | | 6 |

The Pi's `.env` has keys for FRED, NewsAPI and Finnhub, and the shared cache shows
NewsAPI and Yahoo news both producing items in the last 30 days. Phases 3 and 4
have to carry those providers over, not just Yahoo.

## Rules for every phase

- **One phase per change**, with the full suite green, deployed and watched through one
  Monday run before the next phase starts.
- **The prompt must not change as a side effect.** Each phase that feeds the prompt
  (indicators, fundamentals, macro, news) adds a golden test that renders the same
  inputs before and after the move. A changed prompt is a changed strategy, and the
  shadow comparison's results are only comparable while the prompt holds still.
- **Do not add fields to `TickerFeatures`.** Shadow comparisons freeze each case's
  features and check them by hash (`experiments.Case`, `case_hash`). A new field
  changes every archived case's hash and invalidates the evidence collected since
  2026-10-12. New per-ticker data travels beside the features, as ownership does
  (`fundamentals.ownership_lines`).
- **Money paths fail closed.** A missing price or FX rate stays missing (`None`),
  never a default. Sizing, stops, NAV and fills already rely on that. Advisory data
  (news, macro, fundamentals) fails open.
- **Keep the fund's store as the only cache.** No second SQLite file.

## Phases

**1. Indicators, FX, live quotes.** Small, with local versions already written. Port
the three indicator functions with tests pinned to FinanceData's outputs on a fixed
series. Make `_local_rate_to_sek` the only FX path. It reads the same Yahoo
`XXXSEK=X` pair for the spot rate. Outcome scoring reads dated closes from the
price cache, not from here. Batch
`_local_live_price` the way `get_live_prices` does, because `check-stops` prices
every holding every 15 minutes. *Done when* `data/fx.py`, `data/quotes.py` and the
indicator imports no longer name `financedata`.

**2a. Fundamentals (done, 2026-11-01).** `data/fundamentals.py` fetches `.info`
itself with the same key map, plus `heldPercentInsiders` and
`heldPercentInstitutions` (ownership, checklist item 23). A refresh now replaces
only the `.info` keys, so the statement-derived ratios survive it. An unknown
symbol is no longer cached as an empty entry for a week.
`fund fundamentals-check` uses the same fetch. The first refresh after deploy
happens as each name's 7-day entry expires, so the ownership figures fill in over
a week.

**2b. Prices.** Fetch daily bars from Yahoo straight into `store.save_prices` (batch
download, per-ticker retry), the same rows `compute_features` reads now. Drop Alpha
Vantage: it has never run for the fund, and 25 requests a day can't cover a
75-name screen. Before switching, run both fetchers for one week on a copy of the
Pi's database and diff the rows: dates, closes, split and dividend adjustment.
Outcome scoring and the split check (`REBASE_TOLERANCE`) depend on adjustment
matching. *Done when* `data/prices.py` has no `financedata` import and the diff is
clean.

**3. Macro.** Port the FRED, Riksbank and ECB fetchers and `build_macro_block` with
the 6-hour cache moved into `app_meta`. Add a golden test against the current
block for fixed indicator values. The headline half is already local.

**4. News and FinBERT.** The largest phase: RSS keyword matching, NewsAPI with its
cooldown breaker, Finnhub with its rate limit, Yahoo news as the fallback, and
FinBERT scoring. Before porting, measure which providers the fund's own candidates
actually get items from. A provider that adds nothing for this universe is
dropped rather than ported. MFN (`data/mfn.py`) is the obvious addition for Nordic
names. Fix the dead `mfn.se/a/all.rss` feed in `config.yaml` (checklist P5) here
or before.

**5. Universe source (your decision).** Today `deploy.sh` regenerates
`config/universe.csv` and `universe_global.csv` from FinanceData's committed EODHD
snapshot on every deploy. The shared cache's own `universe` table is empty on the
Pi. Recommended: make the committed CSVs the source of truth, stop syncing on
deploy, and keep the script as an occasional `fund universe-sync --from
<snapshot.csv>` run by hand. The alternative is porting the EODHD refresh, which
needs `EODHD_API_TOKEN` here.

**6. Remove the dependency.** Delete `financedata` from `pyproject.toml` and
`[tool.uv.sources]`. Remove the CI checkout and the `FINANCEDATA_READ_TOKEN`
secret, the FinanceData update and branch check in `deploy.sh` and
`deploy/test-branch-check.sh`, and the `financedata` stubs in
`test_target_ratchet.py`, `test_review_visibility.py`, `test_target_review.py`
and `test_review_decisions.py`. Update `CLAUDE.md`. Leave
`~/.financedata/cache.db` alone, since DeepSwing still uses it. *Done when* a
fresh clone runs the full suite with no sibling checkout.
