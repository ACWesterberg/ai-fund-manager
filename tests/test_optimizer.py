import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from fundmgr.config import AppConfig
from fundmgr.engine.evaluator import repair_outcomes
from fundmgr.engine.optimizer import (
    build_trainset,
    decision_metric,
    fields_from_snapshot,
    guidance_fingerprint,
    guidance_path,
    load_guidance,
    price_from_snapshot,
)
from fundmgr.engine.prompt import build_prompt
from fundmgr.state.models import PortfolioSnapshot, RecommendationLog
from fundmgr.state.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "test.db")


@pytest.fixture
def cfg(tmp_path):
    cfg = AppConfig()
    cfg.mandate_path = tmp_path / "mandate.md"
    cfg.mandate_path.write_text("# Mandate\nBeat the benchmark.")
    cfg.db_path = tmp_path / "fund.db"
    cfg.optimizer.compiled_dir = tmp_path / "compiled"
    return cfg


def _example(**alphas):
    return SimpleNamespace(ticker_alphas=alphas)


def _prediction(*actions):
    return SimpleNamespace(
        decision=SimpleNamespace(
            actions=[SimpleNamespace(ticker=t, side=s) for t, s in actions]
        )
    )


# ── Metric ────────────────────────────────────────────────────────────────────

def test_metric_buying_winner_beats_sitting_out():
    ex = _example(**{"AAA.ST": 5.0})
    assert decision_metric(ex, _prediction(("AAA.ST", "buy"))) > decision_metric(ex, _prediction())


def test_metric_sitting_out_beats_buying_loser():
    ex = _example(**{"AAA.ST": -5.0})
    assert decision_metric(ex, _prediction()) > decision_metric(ex, _prediction(("AAA.ST", "buy")))


def test_metric_selling_underperformer_rewarded():
    ex = _example(**{"AAA.ST": -8.0})
    assert decision_metric(ex, _prediction(("AAA.ST", "sell"))) > 0.5


def test_metric_mixes_multiple_tickers():
    ex = _example(**{"AAA.ST": 6.0, "BBB.ST": -6.0})
    both_right = decision_metric(ex, _prediction(("AAA.ST", "buy"), ("BBB.ST", "sell")))
    both_wrong = decision_metric(ex, _prediction(("AAA.ST", "sell"), ("BBB.ST", "buy")))
    assert both_right > 0.5 > both_wrong


def test_metric_unknown_tickers_and_holds_are_neutral():
    ex = _example(**{"AAA.ST": 4.0})
    with_noise = decision_metric(
        ex, _prediction(("AAA.ST", "buy"), ("ZZZ.ST", "buy"), ("AAA.ST", "hold"))
    )
    plain = decision_metric(ex, _prediction(("AAA.ST", "buy")))
    assert with_noise == pytest.approx(plain)


def test_metric_no_known_alphas_neutral():
    assert decision_metric(_example(), _prediction(("AAA.ST", "buy"))) == 0.5


def test_metric_bounded():
    ex = _example(**{"AAA.ST": 1000.0})
    assert 0.0 < decision_metric(ex, _prediction(("AAA.ST", "buy"))) <= 1.0


# ── Snapshot helpers ──────────────────────────────────────────────────────────

_V2_FIELDS = {
    "mandate": "# Mandate",
    "macro": "## Global Macro Context",
    "portfolio_state": "## Current Portfolio State\nNAV: 150,000 SEK",
    "risk_limits": "## Risk Limits",
    "universe": "## Universe — Ticker Feature Blocks\n★ [AAA.ST] Test AB\n  Price: 100.00",
    "learnings": "",
}

_V1_USER_MSG = "\n".join([
    "# Weekly Decision Run",
    "",
    "## Global Macro Context  (fetched 2026-06-01)",
    "  Equity indices: OMXS30 +1.2%",
    "",
    "## Current Portfolio State",
    "NAV: 150,000 SEK  |  Cash: 15,000 SEK (10.0%)",
    "",
    "## Risk Limits (hard constraints)",
    "  Max single-name weight: 18.0%",
    "",
    "## Universe — Ticker Feature Blocks",
    "",
    "★ [AAA.ST] Test AB",
    "  Price: 100.00  (as of 2026-06-01)",
    "  Returns: 20d +8.0%",
    "",
    "  [BBB.ST] Other AB",
    "  Price: 42.50  (as of 2026-06-01)",
    "",
])


def test_fields_from_snapshot_v2():
    fields = fields_from_snapshot({"snapshot_version": 2, "fields": _V2_FIELDS})
    assert fields["universe"].startswith("## Universe")
    assert fields["mandate"] == "# Mandate"


def test_fields_from_snapshot_v1_reconstruction():
    snap = {
        "system_message": "# Mandate\nBeat it.\n\n---\nReturn ONLY a valid JSON object...",
        "user_message": _V1_USER_MSG,
    }
    fields = fields_from_snapshot(snap)
    assert fields["mandate"] == "# Mandate\nBeat it."
    assert "Equity indices" in fields["macro"]
    assert "NAV: 150,000 SEK" in fields["portfolio_state"]
    assert "Max single-name weight" in fields["risk_limits"]
    assert "[AAA.ST]" in fields["universe"] and "[BBB.ST]" in fields["universe"]


def test_price_from_snapshot():
    snap = {"user_message": _V1_USER_MSG}
    assert price_from_snapshot(snap, "AAA.ST") == pytest.approx(100.0)
    assert price_from_snapshot(snap, "BBB.ST") == pytest.approx(42.50)
    assert price_from_snapshot(snap, "ZZZ.ST") is None


# ── Store + trainset ──────────────────────────────────────────────────────────

def _save_run(store, run_id, ts, snapshot=None):
    store.save_recommendation(RecommendationLog(
        run_id=run_id, timestamp=ts,
        prompt_snapshot=json.dumps(snapshot or {"snapshot_version": 2, "fields": _V2_FIELDS}),
        llm_response="{}", guardrail_log="{}", actions_json="[]",
    ))


def _seed(store, run_id, ticker, side="buy", price=None):
    actions = [{"ticker": ticker, "side": side, "confidence": 0.7,
                "sek_estimate": 9000.0, "thesis": "t"}]
    prices = {ticker: price} if price is not None else None
    store.seed_outcomes_for_run(run_id, json.dumps(actions), prices=prices)


def _evaluate(store, run_id, ticker, position_return, bench_return):
    from fundmgr.state.models import DecisionOutcome
    store.update_outcome(DecisionOutcome(
        run_id=run_id, ticker=ticker, action="buy",
        price_at_evaluation=110.0, position_return_pct=position_return,
        benchmark_return_pct=bench_return,
        outperformed=position_return > bench_return, evaluation_date="2026-07-01",
    ))


def test_seed_uses_real_price_not_sek_estimate(store):
    _save_run(store, "r1", datetime(2026, 6, 1))
    _seed(store, "r1", "AAA.ST", price=100.0)
    _seed(store, "r1x", "BBB.ST", price=None)  # unknown price -> NULL, never sek_estimate
    _save_run(store, "r1x", datetime(2026, 6, 1))

    rows = {o.ticker: o for o in store.get_all_outcomes()}
    assert rows["AAA.ST"].price_at_decision == pytest.approx(100.0)
    assert rows["BBB.ST"].price_at_decision is None


def test_pending_outcomes_carry_decision_date(store):
    _save_run(store, "r1", datetime(2026, 6, 1))
    _seed(store, "r1", "AAA.ST", price=100.0)
    pending = store.get_pending_outcomes(older_than_days=0)
    assert pending[0].decision_date == "2026-06-01"


def test_trainset_groups_alphas_per_run(store):
    _save_run(store, "r1", datetime(2026, 6, 1))
    _seed(store, "r1", "AAA.ST", price=100.0)
    _evaluate(store, "r1", "AAA.ST", position_return=10.0, bench_return=6.0)

    examples = build_trainset(store)
    assert len(examples) == 1
    assert examples[0]["ticker_alphas"] == {"AAA.ST": pytest.approx(4.0)}
    assert examples[0]["universe"].startswith("## Universe")


def test_trainset_v1_snapshot_fallback(store):
    snap = {"system_message": "# M\n\n---\nReturn ONLY...", "user_message": _V1_USER_MSG}
    _save_run(store, "r-old", datetime(2026, 5, 1), snapshot=snap)
    _seed(store, "r-old", "AAA.ST", price=100.0)
    _evaluate(store, "r-old", "AAA.ST", position_return=-3.0, bench_return=1.0)

    examples = build_trainset(store)
    assert len(examples) == 1
    assert examples[0]["ticker_alphas"]["AAA.ST"] == pytest.approx(-4.0)
    assert "[AAA.ST]" in examples[0]["universe"]


# ── Repair ────────────────────────────────────────────────────────────────────

def _bench(store, rows):
    store.save_benchmark(rows)


def test_repair_fixes_poisoned_rows(store):
    snap = {"user_message": _V1_USER_MSG}
    _save_run(store, "r1", datetime(2026, 6, 1), snapshot=snap)
    # Simulate the old bug: seed with sek_estimate as the price
    store.seed_outcomes_for_run("r1", json.dumps([
        {"ticker": "AAA.ST", "side": "buy", "confidence": 0.7, "sek_estimate": 9000.0, "thesis": "t"},
        {"ticker": "ZZZ.ST", "side": "buy", "confidence": 0.5, "sek_estimate": 5000.0, "thesis": "t"},
    ]), prices={"AAA.ST": 9000.0, "ZZZ.ST": 5000.0})
    # AAA was "evaluated" against the wrong price + wrong benchmark window
    _evaluate(store, "r1", "AAA.ST", position_return=-98.78, bench_return=42.0)
    _bench(store, [
        {"date": "2026-06-01", "close": 1000.0},
        {"date": "2026-07-01", "close": 1020.0},
        {"date": "2026-07-04", "close": 1500.0},  # after evaluation_date — must be excluded
    ])

    stats = repair_outcomes(store)
    assert stats["price_fixed"] == 1       # AAA's SEK estimate replaced by the true price
    assert stats["recomputed"] == 1        # AAA re-scored
    assert stats["unrecoverable"] == 1     # ZZZ has no snapshot block — price cleared

    rows = {o.ticker: o for o in store.get_all_outcomes()}
    aaa = rows["AAA.ST"]
    assert aaa.price_at_decision == pytest.approx(100.0)
    assert aaa.position_return_pct == pytest.approx(10.0)   # 110 / 100 - 1
    assert aaa.benchmark_return_pct == pytest.approx(2.0)   # 1020/1000 within the window
    assert aaa.outperformed is True
    assert rows["ZZZ.ST"].price_at_decision is None


def test_repair_dry_run_writes_nothing(store):
    snap = {"user_message": _V1_USER_MSG}
    _save_run(store, "r1", datetime(2026, 6, 1), snapshot=snap)
    store.seed_outcomes_for_run("r1", json.dumps([
        {"ticker": "AAA.ST", "side": "buy", "confidence": 0.7, "sek_estimate": 9000.0, "thesis": "t"},
    ]), prices={"AAA.ST": 9000.0})

    stats = repair_outcomes(store, dry_run=True)
    assert stats["price_fixed"] == 1
    assert store.get_all_outcomes()[0].price_at_decision == pytest.approx(9000.0)


def test_repair_leaves_correct_rows_alone(store):
    snap = {"user_message": _V1_USER_MSG}
    _save_run(store, "r1", datetime(2026, 6, 1), snapshot=snap)
    _seed(store, "r1", "AAA.ST", price=100.0)

    stats = repair_outcomes(store)
    assert stats["price_fixed"] == 0
    assert stats["recomputed"] == 0


# ── Guidance artifact + prompt injection ──────────────────────────────────────

def _write_guidance(cfg, text):
    path = guidance_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"instructions": text}))


def test_load_guidance_missing_returns_empty(cfg):
    assert load_guidance(cfg) == ""


def test_load_guidance_corrupt_file(cfg):
    path = guidance_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    assert load_guidance(cfg) == ""


def test_guidance_paths_are_per_fund(cfg):
    nordic = guidance_path(cfg)
    cfg.db_path = cfg.db_path.with_name("fund_claude.db")
    assert guidance_path(cfg) != nordic


def test_build_prompt_injects_guidance(cfg, store):
    _write_guidance(cfg, "Prefer momentum entries with RSI < 65.")
    snap = PortfolioSnapshot(positions=[], cash_sek=150_000)

    system, user, fields = build_prompt(cfg, snap, {}, store, run_id="r1")

    assert "## Decision Guidance (optimized from your realized outcomes)" in system
    assert "Prefer momentum entries with RSI < 65." in system
    # Guidance sits between the mandate and the JSON-format instruction
    assert system.index("Beat the benchmark.") < system.index("Prefer momentum") < system.index("DecisionRun schema")
    # The fielded snapshot keeps the raw mandate — guidance must not leak into training data
    assert "Prefer momentum" not in fields["mandate"]


def test_build_prompt_without_guidance_unchanged(cfg, store):
    snap = PortfolioSnapshot(positions=[], cash_sek=150_000)
    system, _, _ = build_prompt(cfg, snap, {}, store, run_id="r1")
    assert "Decision Guidance" not in system


# ── Guidance fingerprint (A/B tracking) ───────────────────────────────────────

def test_guidance_fingerprint_none_without_artifact(cfg):
    assert guidance_fingerprint(cfg) is None


def test_guidance_fingerprint_changes_with_content(cfg):
    _write_guidance(cfg, "Rule A.")
    fp_a = guidance_fingerprint(cfg)
    assert fp_a and len(fp_a) == 12
    _write_guidance(cfg, "Rule B — different.")
    assert guidance_fingerprint(cfg) != fp_a


def test_snapshot_regime_records_guidance_hash(cfg, store):
    from fundmgr.engine.prompt import snapshot_to_dict

    snap = PortfolioSnapshot(positions=[], cash_sek=150_000)
    # No guidance yet → null in the regime
    plain = json.loads(snapshot_to_dict(snap, "sys", "user", {}, cfg))
    assert plain["regime"]["guidance_hash"] is None

    _write_guidance(cfg, "Prefer momentum entries.")
    guided = json.loads(snapshot_to_dict(snap, "sys", "user", {}, cfg))
    assert guided["regime"]["guidance_hash"] == guidance_fingerprint(cfg)
