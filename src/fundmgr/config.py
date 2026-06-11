from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
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


@dataclass
class RiskConfig:
    max_position_pct: float = 18.0
    max_positions: int = 10
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
    top_n: int = 75  # candidates passed to LLM; held positions always added on top


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
    db_path: Path = field(default_factory=lambda: DATA_DIR / "fund.db")
    mandate_path: Path = field(default_factory=lambda: CONFIG_DIR / "mandate.md")


_EXCHANGE_CURRENCY: dict[str, str] = {
    "OMXS": "SEK",
    "OMXC": "DKK",
    "OSLO": "NOK",
    "OMXH": "EUR",
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

    path = config_path or CONFIG_DIR / "config.yaml"
    raw = yaml.safe_load(path.read_text())

    cfg = AppConfig(
        capital_sek=float(os.getenv("FUND_CAPITAL_SEK", raw.get("capital_sek", 50000))),
        cadence=raw.get("cadence", "weekly"),
        benchmark=raw.get("benchmark", "^OMXSPI"),
    )

    if llm_raw := raw.get("llm"):
        cfg.llm = LLMConfig(
            provider=os.getenv("FUND_LLM_PROVIDER", llm_raw.get("provider", "openai")),
            model_id=os.getenv("FUND_MODEL_ID", llm_raw.get("model_id", "gpt-4o")),
            temperature=llm_raw.get("temperature", 0.2),
            max_tokens=llm_raw.get("max_tokens", 4096),
            reasoning_effort=llm_raw.get("reasoning_effort", None),
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

    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "reports").mkdir(exist_ok=True)

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
