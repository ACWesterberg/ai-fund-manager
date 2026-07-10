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


@dataclass
class OptimizerConfig:
    # Heavy model that *writes* candidate instructions (MIPRO prompt_model).
    # None → derived from llm.provider at run time (anthropic → claude-opus-4-8, openai → gpt-5.5).
    prompt_model_id: str | None = None
    min_outcomes: int = 30       # evaluated outcomes required before optimization runs
    min_examples: int = 8        # usable run-level examples required after reconstruction
    compiled_dir: Path = field(default_factory=lambda: CONFIG_DIR / "compiled")


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
    db_path: Path = field(default_factory=lambda: DATA_DIR / "fund.db")
    mandate_path: Path = field(default_factory=lambda: CONFIG_DIR / "mandate.md")
    universe_path: Path = field(default_factory=lambda: CONFIG_DIR / "universe.csv")
    auto_fill: bool = False  # if True, paper-execute fills automatically after each run
    fx_to_sek: bool = False  # convert foreign-currency holdings to SEK for cash/NAV
                             # (real fund). Sims run native-consistent; leave False.
    name: str = ""           # display name for notifications (which fund this is)

    @property
    def display_name(self) -> str:
        return self.name or f"{self.llm.provider}/{self.llm.model_id}"

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
        return _EXCHANGE_CURRENCY.get(self.exchange, "SEK")

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
    if "auto_fill" in raw:
        cfg.auto_fill = bool(raw["auto_fill"])
    if "fx_to_sek" in raw:
        cfg.fx_to_sek = bool(raw["fx_to_sek"])
    if "name" in raw:
        cfg.name = str(raw["name"])

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


def get_isin_map(universe_path: Path | None = None) -> dict[str, str]:
    """Return ISIN -> yahoo_ticker for all tickers (enabled and disabled)."""
    return {t.isin: t.yahoo_ticker for t in load_universe(universe_path) if t.isin}
