from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]  # project root (ai-fund-manager/)
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"


@dataclass
class LLMConfig:
    provider: str = "openai"
    model_id: str = "gpt-4o"
    temperature: float = 0.2
    max_tokens: int = 4096
    reasoning_effort: str | None = None  # "low" | "medium" | "high" — for o1/o3/gpt-5+ models
    n_samples: int = 1  # run LLM this many times and take majority-vote consensus; 1 = disabled


@dataclass
class RiskConfig:
    max_position_pct: float = 18.0
    max_positions: int = 10
    max_sector_pct: float = 35.0             # max NAV weight in any single GICS sector
    min_cash_pct: float = 12.0
    max_cash_pct: float = 25.0
    min_trade_sek: float = 2500.0
    max_turnover_pct: float = 25.0
    stale_after_days: int = 5
    cold_start_cash_threshold: float = 80.0  # if cash% above this, use cold_start_turnover_pct
    cold_start_turnover_pct: float = 50.0    # turnover cap when deploying from near-100% cash
    # Geographic mix, {region code: % of NAV} — see fundmgr.regions. Empty means
    # the book's geography is whatever the screener's ranking happens to produce.
    # Only named regions constrain anything; the ceiling (target + tolerance) is
    # enforced here, the floor is not, because no guardrail can force a buy.
    region_targets: dict[str, float] = field(default_factory=dict)
    region_tolerance_pct: float = 10.0       # band around each regional target
    # Risk/quality mix, {style code: % of NAV} — see fundmgr.styles. Same shape
    # as the regional dial, but over a judgement rather than a fact: a name with
    # no fundamentals on file is `unclassified`, counted against no target and
    # blocked by none, so these caps under-count rather than bind tightly.
    style_targets: dict[str, float] = field(default_factory=dict)
    style_tolerance_pct: float = 10.0        # band around each style target


@dataclass
class FeeConfig:
    rate: float = 0.001
    min_sek: float = 1.0
    max_sek: float = 99.0

    def calc(self, trade_sek: float) -> float:
        return max(self.min_sek, min(self.max_sek, trade_sek * self.rate))


@dataclass
class SentimentConfig:
    enabled: bool = True
    model: str = "ProsusAI/finbert"
    device: str = "cpu"
    trigger_threshold_negative: float = 0.80
    trigger_threshold_positive: float = 0.85
    trigger_cooldown_hours: int = 6


@dataclass
class DataConfig:
    price_provider: str = "yfinance"
    lookback_days: int = 252
    news_feeds: list[str] = field(default_factory=list)
    macro_feeds: list[str] = field(default_factory=list)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)


@dataclass
class WebConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class ScreenerConfig:
    top_n: int = 75  # candidates passed to LLM; held + pinned always added on top
    pinned_tickers: list[str] = field(default_factory=list)  # always price-fetch + screen
    price_fetch_limit: int | None = None  # cap weekly refreshes for large universes
    rotate_weeks: int = 8  # spread the rest across N ISO-week buckets


def default_heavy_model(provider: str) -> str:
    """The heaviest reasoner for a provider.

    Used for the roles where output quality matters far more than call volume —
    writing the optimizer's candidate instructions, and distilling learnings.
    Both run a handful of times a week and their output conditions every
    subsequent decision, so neither should be served by a cheap model.
    """
    return "claude-opus-4-8" if provider == "anthropic" else "gpt-5.6-sol"


@dataclass
class OptimizerConfig:
    # Bounded instruction-only search, separate from live decision settings.
    max_calls: int = 7
    max_total_tokens: int = 200000  # conservative request reservations, not billed usage
    max_output_tokens: int = 2048
    reasoning_effort: str = "low"
    validation_runs: int = 3
    min_new_periods: int = 3  # own-fund matured decision dates since prior paid search
    reuse_evaluations: bool = True
    # Optional model override for the single instruction-writing call.
    # None uses the fund's decision model.
    prompt_model_id: str | None = None
    min_outcomes: int = 30       # evaluated outcomes required before optimization runs
    # Keep the minimum historical evidence gate even for the smaller search.
    # Search validation alone never qualifies an artifact for promotion.
    min_examples: int = 25
    # Other funds' configs to pool training examples from (filenames in config/,
    # or absolute paths). Empty = this fund's own history only. Pooling shares
    # the trainset, never the guidance artifact — each fund still compiles and
    # applies its own.
    pool_configs: list[str] = field(default_factory=list)
    compiled_dir: Path = field(default_factory=lambda: CONFIG_DIR / "compiled")


@dataclass
class ShadowConfig:
    # Explicit opt-in: each new weekly run makes two additional sample sets.
    candidate: str | None = None
    benchmark_currency: str = ""
    benchmark_calendar: str = ""
    # Named variants (experiments.VARIANTS), each a further pair per run, e.g.
    # ["no_lessons"]. Created once with `fund shadow-variant NAME`.
    variants: list[str] = field(default_factory=list)


@dataclass
class ReportWatchConfig:
    # The report-reaction experiment (engine/report_events.py). Off unless a
    # fund opts in; it judges drops but never trades or touches the prompt.
    enabled: bool = False
    move_threshold_pct: float = 5.0     # |abnormal| reaction that counts as an event
    judge_threshold_pct: float = 5.0    # drops at least this large get a verdict
    max_judged_per_day: int = 10        # largest drops first; the rest stay unjudged
    reasoning_effort: str = "medium"
    max_tokens: int = 8000


@dataclass
class EarningsWatchConfig:
    # Reads each held name's report against the thesis it was bought on
    # (engine/earnings_watch.py). Advice only: it never trades.
    enabled: bool = False
    reasoning_effort: str = "medium"
    max_tokens: int = 8000
    max_checks_per_report: int = 3      # report-day morning, evening, next morning


@dataclass
class MontroseConfig:
    # The real broker account this fund's ledger mirrors (phase 1: read and
    # compare only). Empty means the fund has no broker account.
    account_name: str = ""
    lookback_days: int = 60          # transaction window the reconcile walks
    cash_tolerance_sek: float = 25.0  # fee rounding across a month of fills
    cost_tolerance_pct: float = 1.0
    # Phase 2: Montrose rows dated on/after this are offered for booking into
    # the ledger (one tap each). Earlier history was booked by hand — some of
    # it folded into cash with /setcash — so it is never offered. Empty = off.
    import_from: str = ""
    # Phase 3: after each weekly run, send a pre-filled Montrose order ticket
    # per approved trade. Off until the owner has tried /montrose_ticket once;
    # the plan is saved either way, so /montrose_tickets works on demand.
    tickets: bool = False


@dataclass
class AppConfig:
    capital_sek: float = 50000.0
    cadence: str = "weekly"
    benchmark: str = "^OMXSPI"
    llm: LLMConfig = field(default_factory=LLMConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    fees: FeeConfig = field(default_factory=FeeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    web: WebConfig = field(default_factory=WebConfig)
    screener: ScreenerConfig = field(default_factory=ScreenerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    shadow: ShadowConfig = field(default_factory=ShadowConfig)
    report_watch: ReportWatchConfig = field(default_factory=ReportWatchConfig)
    montrose: MontroseConfig = field(default_factory=MontroseConfig)
    earnings_watch: EarningsWatchConfig = field(default_factory=EarningsWatchConfig)
    db_path: Path = field(default_factory=lambda: DATA_DIR / "fund.db")
    mandate_path: Path = field(default_factory=lambda: CONFIG_DIR / "mandate.md")
    universe_path: Path = field(default_factory=lambda: CONFIG_DIR / "universe.csv")
    auto_fill: bool = False  # if True, paper-execute fills automatically after each run
    fx_to_sek: bool = False  # convert foreign-currency holdings to SEK for cash/NAV
                             # (real fund). Sims run native-consistent; leave False.
    name: str = ""           # display name for notifications (which fund this is)
    # Paused: the book stays as it is and keeps being valued, but nothing that
    # costs money runs for it — no decision runs, reviews, optimizer searches or
    # news-triggered runs — and nothing trades, mechanical stops included.
    # Un-pausing is deleting the line.
    paused: bool = False
    # Model that distils evaluated outcomes into learnings. None → this fund's
    # provider's heavy model. Pin the same id across every fund's config to keep
    # one lesson-writer system-wide, so the GPT-vs-Claude comparison is a
    # difference in decision model rather than in how each fund is coached.
    learning_model_id: str | None = None
    # Days after a decision before its outcome is scored. This is a property of
    # the mandate, not a system constant: a momentum book and a quality-
    # compounder screen are not the same question asked at 28 days. Set it to
    # match how long a thesis in this fund is expected to take to resolve — the
    # learning loop reads the horizon back and will tell the fund to stop taking
    # positions it cannot score inside it, so too short a horizon quietly
    # rewrites the strategy.
    evaluation_horizon_days: int = 28

    @property
    def display_name(self) -> str:
        return self.name or f"{self.llm.provider}/{self.llm.model_id}"

    @property
    def learning_model(self) -> str:
        """Resolved id of the model that writes this fund's learnings."""
        return self.learning_model_id or default_heavy_model(self.llm.provider)

    def config_hash(self) -> str:
        """Short hash of the decision-shaping config, for within-repo regime drift.

        Scope: risk + llm + fees — the parameters that change the model's
        decision surface. Deliberately excludes universe (churns weekly; the
        actual tickers shown are captured per-run in the snapshot's `universe`
        field) and data-source / non-semantic config (feeds, paths, web, etc.).
        Cross-repo hashes are NOT expected to match — this only flags drift
        within one repo (e.g. when capital or risk limits change).
        """
        payload = {
            "risk": asdict(self.risk),
            "llm": asdict(self.llm),
            "fees": asdict(self.fees),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]


_SUFFIX_CURRENCY: dict[str, str] = {
    "ST": "SEK", "CO": "DKK", "OL": "NOK", "HE": "EUR", "IC": "ISK",
    "WA": "PLN", "MC": "EUR", "DE": "EUR", "PA": "EUR", "AS": "EUR", "L": "GBP",
}

_EXCHANGE_CURRENCY: dict[str, str] = {
    # Nordic
    "OMXS": "SEK",
    "OMXS-FN": "SEK",
    "SPOTLIGHT": "SEK",
    "NGM": "SEK",
    "OMXC": "DKK",
    "OMXC-FN": "DKK",
    "OSLO": "NOK",
    "OMXH": "EUR",
    "OMXI": "ISK",
    "STOCKHOLM": "SEK",
    "HELSINKI": "EUR",
    "COPENHAGEN": "DKK",
    "WSE": "PLN",
    "BME": "EUR",
    # Global
    "NYSE": "USD",
    "NASDAQ": "USD",
    "LSE": "GBP",
    "XETRA": "EUR",
    "EURONEXT": "EUR",
    "SIX": "CHF",
    "TSE": "JPY",
    "TSX": "CAD",
    "ASX": "AUD",
    "HKEX": "HKD",
    "OTC": "USD",
}


@dataclass
class UniverseTicker:
    name: str
    yahoo_ticker: str
    isin: str
    country: str
    exchange: str
    sector: str
    enabled: bool

    @property
    def currency(self) -> str:
        """Quote currency: the exchange's, else the Yahoo suffix's, else SEK.

        The SEK default used to apply to any exchange code missing from the
        table, which priced Orion Oyj (ORNBV.HE, EUR, listed as "HELSINKI")
        at an eleventh of its value in the real fund's SEK maths.
        """
        known = _EXCHANGE_CURRENCY.get((self.exchange or "").upper())
        if known:
            return known
        suffix = self.yahoo_ticker.rsplit(".", 1)[-1].upper() if "." in self.yahoo_ticker else ""
        return _SUFFIX_CURRENCY.get(suffix, "SEK")

    @property
    def needs_fx(self) -> bool:
        return self.currency != "SEK"


def load_config(config_path: Path | None = None) -> AppConfig:
    load_dotenv(ROOT / ".env")

    # FUND_CONFIG env var lets systemd services select which fund to run
    env_config = os.getenv("FUND_CONFIG")
    path = config_path or (Path(env_config) if env_config else CONFIG_DIR / "config.yaml")
    raw = yaml.safe_load(path.read_text())

    cfg = AppConfig(
        capital_sek=float(os.getenv("FUND_CAPITAL_SEK", raw.get("capital_sek", 50000))),
        cadence=raw.get("cadence", "weekly"),
        benchmark=raw.get("benchmark", "^OMXSPI"),
    )

    # Paths — resolve relative to project root
    if db := raw.get("db_path"):
        cfg.db_path = ROOT / db
    if mandate := raw.get("mandate_path"):
        cfg.mandate_path = ROOT / mandate
    if universe := raw.get("universe_path"):
        cfg.universe_path = ROOT / universe
    if "paused" in raw:
        cfg.paused = bool(raw["paused"])
    if "auto_fill" in raw:
        cfg.auto_fill = bool(raw["auto_fill"])
    if "fx_to_sek" in raw:
        cfg.fx_to_sek = bool(raw["fx_to_sek"])
    if "name" in raw:
        cfg.name = str(raw["name"])
    if "learning_model_id" in raw:
        cfg.learning_model_id = str(raw["learning_model_id"])
    if "evaluation_horizon_days" in raw:
        cfg.evaluation_horizon_days = int(raw["evaluation_horizon_days"])

    if opt_raw := raw.get("optimizer"):
        if "min_outcomes" in opt_raw:
            cfg.optimizer.min_outcomes = int(opt_raw["min_outcomes"])
        if "min_examples" in opt_raw:
            cfg.optimizer.min_examples = int(opt_raw["min_examples"])
        if "prompt_model_id" in opt_raw:
            cfg.optimizer.prompt_model_id = str(opt_raw["prompt_model_id"])
        if "pool_configs" in opt_raw:
            cfg.optimizer.pool_configs = [str(x) for x in (opt_raw["pool_configs"] or [])]

    if llm_raw := raw.get("llm"):
        cfg.llm = LLMConfig(
            provider=os.getenv("FUND_LLM_PROVIDER", llm_raw.get("provider", "openai")),
            model_id=os.getenv("FUND_MODEL_ID", llm_raw.get("model_id", "gpt-4o")),
            temperature=llm_raw.get("temperature", 0.2),
            max_tokens=llm_raw.get("max_tokens", 4096),
            reasoning_effort=llm_raw.get("reasoning_effort", None),
            n_samples=int(llm_raw.get("n_samples", 1)),
        )

    if risk_raw := raw.get("risk"):
        cfg.risk = RiskConfig(**risk_raw)

    if fee_raw := raw.get("fees"):
        cfg.fees = FeeConfig(**fee_raw)

    if data_raw := raw.get("data"):
        sent_raw = data_raw.pop("sentiment", {})
        cfg.data = DataConfig(
            **data_raw,
            sentiment=SentimentConfig(**sent_raw) if sent_raw else SentimentConfig(),
        )

    if web_raw := raw.get("web"):
        cfg.web = WebConfig(**web_raw)

    if screener_raw := raw.get("screener"):
        cfg.screener = ScreenerConfig(**screener_raw)

    if opt_raw := raw.get("optimizer"):
        compiled = opt_raw.pop("compiled_dir", None)
        cfg.optimizer = OptimizerConfig(**opt_raw)
        if compiled:
            cfg.optimizer.compiled_dir = ROOT / compiled
    if rw_raw := raw.get("report_watch"):
        cfg.report_watch = ReportWatchConfig(**rw_raw)
    if ew_raw := raw.get("earnings_watch"):
        cfg.earnings_watch = EarningsWatchConfig(**ew_raw)
    if montrose_raw := raw.get("montrose"):
        cfg.montrose = MontroseConfig(**montrose_raw)
    if shadow_raw := raw.get("shadow"):
        cfg.shadow = ShadowConfig(**shadow_raw)
        if cfg.shadow.candidate:
            cfg.shadow.candidate = str(ROOT / cfg.shadow.candidate)
    if env_prompt_model := os.getenv("FUND_OPTIMIZER_PROMPT_MODEL"):
        cfg.optimizer.prompt_model_id = env_prompt_model

    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "reports").mkdir(exist_ok=True)
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)

    return cfg


def load_universe(universe_path: Path | None = None) -> list[UniverseTicker]:
    path = universe_path or CONFIG_DIR / "universe.csv"
    tickers = []
    with path.open() as f:
        for row in csv.DictReader(f):
            tickers.append(
                UniverseTicker(
                    name=row["name"],
                    yahoo_ticker=row["yahoo_ticker"],
                    isin=row["isin"],
                    country=row["country"],
                    exchange=row["exchange"],
                    sector=row["sector"],
                    enabled=row["enabled"].strip().lower() == "true",
                )
            )
    return tickers


def get_enabled_tickers(universe_path: Path | None = None) -> list[UniverseTicker]:
    return [t for t in load_universe(universe_path) if t.enabled]


def holding_rows(universe_path: Path | None, held: set[str]) -> list[UniverseTicker]:
    """A universe row for every held ticker, enabled or not.

    A position outlives its row: a name disabled in the CSV (Hemnet) or pinned
    with `montrose-map` to a symbol the CSV never listed still has to be priced,
    converted and sold. Built from enabled rows only, such a holding had no
    price, which blocked every trade in the book, and no currency but SEK. A
    symbol with no row gets a bare one whose currency comes from its suffix.
    """
    by_yahoo = {t.yahoo_ticker: t for t in load_universe(universe_path)}
    return [
        by_yahoo.get(t) or UniverseTicker(name=t, yahoo_ticker=t, isin="", country="",
                                          exchange="", sector="", enabled=False)
        for t in sorted(held)
    ]


def get_isin_map(universe_path: Path | None = None) -> dict[str, str]:
    """Return ISIN -> yahoo_ticker for all tickers (enabled and disabled)."""
    return {t.isin: t.yahoo_ticker for t in load_universe(universe_path) if t.isin}
