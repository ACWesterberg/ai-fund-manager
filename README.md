# AI Fund Manager

A **learning-driven decision-prompt system**, running on a Raspberry Pi 5. Its main purpose is to improve future investment choices by turning evaluated decisions into better prompts. The fund portfolios provide the decisions, evidence and outcomes for that loop. Once a week each fund is handed its mandate, its portfolio state, a screened slice of its universe and its own past lessons, and returns a structured set of buy/sell/hold actions. Mechanical guardrails then approve, clip or reject each one.

**One book holds real money.** The other four are simulations running the identical pipeline, so decision quality can be compared across models and mandates.

The fund never places orders. For the real book you execute in your broker and record the fills; the simulations auto-fill at live prices.

---

## The five funds

| Fund | Universe | Model | Money |
|---|---|---|---|
| 🇸🇪 Nordic — REAL money | Nordic (1,605 names) | `gpt-5.6-sol` | **real** |
| 🌍 Global SIM | Global (17,117) | `gpt-5.6-sol` | sim |
| 🤖 Global SIM | Global (17,117) | `claude-opus-4-8` | sim |
| 🧱 Buffett Screen | Quality screen (96) | `gpt-5.6-sol` | sim |
| 🧱 Buffett Screen | Quality screen (96) | `claude-opus-4-8` | sim |

The Global and Buffett funds are deliberately **paired across providers on an identical universe and mandate**, so the only variable is the decision model. Each config bundles mandate + universe + risk limits + model, so a profile can never mix e.g. the Buffett mandate with the Nordic universe.

---

## What it does

- **Weekly decision run** — prices → fundamentals (7-day cache) → benchmark → macro → features → screen → news + FinBERT sentiment on candidates only → evaluate matured outcomes → build prompt → LLM → guardrails → save + notify
- **Consensus sampling** — every decision is run 3× in parallel and majority-voted per ticker; unanimous and majority-only actions are labelled separately
- **Guardrails, not trust** — universe membership, stale-data block on buys, min trade size, position-weight clipping, sector concentration, max open positions, cash floor, and a turnover cap that drops the lowest-confidence trades. Every verdict is logged, so you can see what the model wanted versus what it got
- **Won't run into a shut market** — cron knows nothing about holidays, so `fund run` resolves its universe's dominant trading calendar and skips before spending anything, naming the next session
- **Learns from realised outcomes** — decisions are scored at a mandate-chosen horizon (28 days by default; 90 for Buffett), each thesis is judged on whether it *held* or merely *paid*, and the resulting lessons are injected into later prompts. Once enough outcomes accumulate, DSPy/MIPROv2 compiles an inactive decision-guidance candidate for evaluation
- **Monitored sleeves** — mirror a real broker account as a paper book with per-position kill criteria, target prices and add gates, watched daily via Telegram
- **What-If Lab** — generate a hypothetical from-scratch portfolio for any fund profile against a clean-slate snapshot, optionally with a full monitoring plan, and promote one you like into a live sleeve
- **Web dashboard** — one route per fund, plus paper books, live sleeves and the Lab

---

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| LLM clients | `openai` (structured outputs), `anthropic` (JSON mode) |
| Prompt optimization | `dspy-ai` — MIPROv2, gated on 30 evaluated outcomes |
| Sentiment | FinBERT (`ProsusAI/finbert`) — local, no API cost |
| Market calendars | `exchange_calendars` — holidays, half-days, session hours |
| Data | shared `financedata` library (prices, fundamentals, FX, news, macro), `yfinance`, SEC EDGAR |
| Database | SQLite |
| Web | `fastapi` + `uvicorn` + Jinja2 |
| Scheduling | cron on the Pi (no in-process scheduler) |
| Notifications | Telegram — run summaries, stop alerts, kill-criterion triggers, OCR fill import |

---

## Quick Start

```bash
uv sync                                   # or: pip install -e .
cp .env.example .env                      # API keys, Telegram + FUND_WEB_PASSWORD
fund init                                 # initialise a portfolio
fund run --dry-run                        # full pipeline, nothing saved
uvicorn fundmgr.web.app:app --reload      # dashboard at localhost:8000
```

The dashboard requires `FUND_WEB_PASSWORD` in `.env` (username defaults to `fund`).
Use HTTPS when accessing it remotely. Until a password is configured, dashboard
requests return 503; the separately signed deployment webhook remains available.

Select a fund with `FUND_CONFIG`; unset means the Nordic book:

```bash
FUND_CONFIG=config/config_global.yaml fund run
```

Useful flags: `--force` (ignore the market-holiday gate), `--force-refresh` (re-fetch prices), `--skip-news` / `--skip-macro` / `--skip-fundamentals` for faster iteration.

`financedata` is a shared local library installed from `../FinanceData` — it is not on PyPI, so the full test suite only runs where it is present.

---

## Documentation

| Doc | What it covers |
|---|---|
| [CLAUDE.md](CLAUDE.md) | **Start here.** Purpose, every design decision and the failure behind it, the run pipeline, risk limits, schedule, file map |
| [docs/MONITORING.md](docs/MONITORING.md) | Monitoring a real sleeve — kill and add criteria, how criterion text is read, staleness, add signals, evidence sources |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Ideas not yet built |
| [deploy/SETUP.md](deploy/SETUP.md) | Raspberry Pi provisioning and auto-deploy |
| [deploy/BACKUP.md](deploy/BACKUP.md) | SQLite → Google Drive backups and restore |
| [deploy/cron.example](deploy/cron.example) | The full schedule, annotated |
| `config/mandate*.md` | The investment policy each fund is given as its system prompt |

---

## A note on risk

This drives a **real brokerage account**. The guardrails are the safety layer and they are mechanical on purpose — the model's output is a proposal, never an instruction. Before changing anything in `guardrails/rules.py`, read why each check exists in [CLAUDE.md](CLAUDE.md).

Sibling project on the same Pi: **DeepSwing** — a swing-trading simulator (intraday scans, days-long holds, paper money only). Different problem, shared `financedata` library.


## Learning-loop implementation status

The first evaluation-safety changes are implemented: historical thesis judgments
are diagnostic only (not rewards for new candidate theses), outcome prices cannot
look past their horizon, and thesis evidence must have been published and cached
inside the evaluation window. Missing history keeps outcomes pending.

`fund optimize` now writes immutable candidates under
`config/compiled/candidates/<fund>/`. The prompt page lists these separately;
compilation never replaces active guidance. Existing active guidance is preserved.
`fund compare-guidance` records paired decisions on complete frozen weekly inputs;
`fund score-guidance` evaluates their allocations after fees against supplied daily
valuations. Comparisons require explicit model calls; scoring is offline. There is
no activation command yet. Opt-in `shadow` configuration records fresh weekly
comparisons; subsequent runs or `fund collect-guidance` automatically collect
matured adjusted-price/FX evidence and score future-entry portfolios.
`fund evaluate-guidance` audits archived evidence and summarizes non-overlapping
periods separately for each candidate/configuration. Validated statistical
thresholds and a tested promotion gate remain to be implemented.
See [the learning roadmap](docs/LEARNING.md) for scope and acceptance criteria.
