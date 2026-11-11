from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from fundmgr.config import AppConfig
from fundmgr.data.prices import TickerFeatures
from fundmgr.data.screener import score_all
from fundmgr.engine import client
from fundmgr.engine.schema import Action, DecisionRun
from fundmgr.state.archive import (
    archive_dir, build_run_archive, read_run_archive, write_run_archive,
)


def _cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig()
    cfg.db_path = tmp_path / "fund_x.db"
    return cfg


def _feat(ticker: str, r20: float) -> TickerFeatures:
    f = TickerFeatures(ticker=ticker, name=ticker, last_price=100.0,
                       last_date="2026-11-10", data_age_trading_days=1)
    f.return_20d_pct = r20
    f.headlines = [{"headline": "noise"}]
    return f


def _payload(cfg: AppConfig) -> dict:
    features = {"A": _feat("A", 10.0), "B": _feat("B", -5.0), "C": _feat("C", 3.0)}
    return build_run_archive(
        cfg, "2026-11-09-abc123", features, selected={"A", "B"},
        scores=score_all(features),
        fundamentals={"A": {"pe_ratio": 12.0, "analyst_target_price": float("nan")}},
        held={"B"}, pinned=set(), universe_size=10,
    )


def test_archive_keeps_the_names_the_model_never_saw(tmp_path):
    """Screened-out names are the baseline the picks have to beat; they were never stored."""
    payload = _payload(_cfg(tmp_path))
    rows = {r["ticker"]: r for r in payload["population"]}

    assert set(rows) == {"A", "B", "C"}
    assert rows["C"]["selected"] is False
    assert rows["B"]["selected"] and rows["B"]["held"]
    assert [rows[t]["rank"] for t in ("A", "C", "B")] == [1, 2, 3]
    assert rows["A"]["features"]["return_20d_pct"] == 10.0
    assert "headlines" not in rows["A"]["features"]
    assert payload["screen"] == {"top_n": AppConfig().screener.top_n, "scored": 3, "selected": 2}
    # Non-finite values cannot be written as strict JSON; they become null.
    assert payload["fundamentals"]["A"]["analyst_target_price"] is None


def test_archive_is_written_once_and_never_overwritten(tmp_path):
    cfg = _cfg(tmp_path)
    path = write_run_archive(cfg, "2026-11-09-abc123", _payload(cfg))

    assert path == archive_dir(cfg) / "2026-11-09-abc123.json.gz"
    assert path.parent == tmp_path / "archive" / "fund_x"
    assert read_run_archive(path)["run_id"] == "2026-11-09-abc123"

    assert write_run_archive(cfg, "2026-11-09-abc123", {"run_id": "replaced"}) is None
    assert read_run_archive(path)["run_id"] == "2026-11-09-abc123"
    with gzip.open(path, "rt") as fh:
        assert "replaced" not in fh.read()


def test_archive_rejects_a_run_id_that_could_escape_the_directory(tmp_path):
    cfg = _cfg(tmp_path)
    assert write_run_archive(cfg, "../../etc/passwd", {}) is None
    assert not (tmp_path / "archive").exists()


def test_archive_failure_fails_open(tmp_path):
    cfg = _cfg(tmp_path)
    (tmp_path / "archive").write_text("a file where the directory should be")
    assert write_run_archive(cfg, "2026-11-09-abc123", _payload(cfg)) is None


def _run(ticker: str) -> DecisionRun:
    return DecisionRun(
        run_id="t", market_summary="m", cash_target_pct=12,
        actions=[Action(ticker=ticker, side="buy", target_weight_pct=5,
                        sek_estimate=5000, confidence=0.7, thesis="t")],
    )


def test_consensus_keeps_every_sample_and_its_vote(monkeypatch):
    """The minority proposals and the agreement level were discarded after voting."""
    replies = iter([_run("A"), _run("A"), _run("B")])
    monkeypatch.setattr(client, "call_llm", lambda *a, **k: (next(replies), "{}"))
    cfg = AppConfig()
    cfg.llm.n_samples = 3

    decision, _raw, votes, sampling = client.call_llm_consensus("s", "u", cfg)

    assert [a.ticker for a in decision.actions] == ["A"]
    tickers = sorted(s["actions"][0]["ticker"] for s in sampling["samples"])
    assert tickers == ["A", "A", "B"]
    # The vote only records survivors; the losing proposal exists only in the samples.
    assert sampling["vote_counts"] == votes == {"A": 2}


@pytest.mark.parametrize("bad", ["2026-11-09", "2026-11-09-ABC123", "x/2026-11-09-abc123"])
def test_only_generated_run_ids_are_archived(tmp_path, bad):
    assert write_run_archive(_cfg(tmp_path), bad, {}) is None
