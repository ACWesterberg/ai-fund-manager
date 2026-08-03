import json
from datetime import datetime
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

from fundmgr.config import AppConfig
from fundmgr.engine.optimizer import guidance_path, guidance_versions
from fundmgr.state.models import Learning
from fundmgr.state.store import Store
from fundmgr.web.views import learnings_context, prompt_context

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "src" / "fundmgr" / "web" / "templates"
_jinja = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True)


@pytest.fixture
def cfg(tmp_path):
    c = AppConfig()
    c.name = "🇸🇪 Nordic — REAL money"
    c.db_path = tmp_path / "fund.db"
    c.mandate_path = tmp_path / "mandate.md"
    c.mandate_path.write_text("# Mandate\nBeat OMXSPI.")
    c.optimizer.compiled_dir = tmp_path / "compiled"
    return c


@pytest.fixture
def store(cfg):
    return Store(cfg.db_path)


def _learning(store, category, body, run_ids=()):
    store.save_learning(Learning(category=category, body=body, run_ids=list(run_ids),
                                 created_at=datetime(2026, 6, 1)))


def _write_guidance(cfg, text, archived=False, stamp="20260601_020000"):
    p = guidance_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    if archived:
        p = p.with_name(f"{p.stem}_{stamp}.json")
    p.write_text(json.dumps({
        "created_at": "2026-06-01T02:00:00+00:00",
        "instructions": text, "prompt_model": "claude-opus-4-8",
        "n_train_runs": 8, "n_val_runs": 2, "n_outcomes": 34,
    }))


# ── learnings_context ─────────────────────────────────────────────────────────

def test_learnings_context_groups_by_category(cfg, store):
    _learning(store, "calibration", "High-conf buys hit 70%.", run_ids=["r1", "r2"])
    _learning(store, "qualitative", "Momentum names faded post-earnings.", run_ids=["r3"])
    _learning(store, "calibration", "Low-conf buys underperform.")
    ctx = learnings_context(cfg, store)
    assert ctx["total"] == 3
    assert ctx["categories"] == ["calibration", "qualitative"]
    assert len(ctx["by_category"]["calibration"]) == 2
    assert ctx["by_category"]["qualitative"][0]["run_count"] == 1
    assert ctx["fund_label"] == "🇸🇪 Nordic — REAL money"
    assert ctx["active_page"] == "learnings"


def test_learnings_context_empty(cfg, store):
    ctx = learnings_context(cfg, store)
    assert ctx["total"] == 0 and ctx["by_category"] == {}


def test_learnings_are_per_fund(tmp_path):
    cfg_a = AppConfig()
    cfg_a.db_path = tmp_path / "a.db"
    cfg_b = AppConfig()
    cfg_b.db_path = tmp_path / "b.db"
    store_a, store_b = Store(cfg_a.db_path), Store(cfg_b.db_path)
    _learning(store_a, "calibration", "A-only lesson.")
    assert learnings_context(cfg_a, store_a)["total"] == 1
    assert learnings_context(cfg_b, store_b)["total"] == 0  # isolated


# ── prompt_context / guidance_versions ────────────────────────────────────────

def test_prompt_context_no_guidance(cfg):
    ctx = prompt_context(cfg)
    assert ctx["mandate"].startswith("# Mandate")
    assert ctx["mandate_filename"] == "mandate.md"
    assert ctx["guidance"]["current"] is None
    assert ctx["guidance"]["history"] == []


def test_guidance_versions_current_and_archive(cfg):
    _write_guidance(cfg, "Prefer momentum with RSI < 65.")
    _write_guidance(cfg, "Old rule.", archived=True, stamp="20260525_020000")
    _write_guidance(cfg, "Older rule.", archived=True, stamp="20260518_020000")
    v = guidance_versions(cfg)
    assert v["current"]["instructions"] == "Prefer momentum with RSI < 65."
    assert v["current"]["prompt_model"] == "claude-opus-4-8"
    assert [h["instructions"] for h in v["history"]] == ["Old rule.", "Older rule."]  # newest-first
    assert v["history"][0]["_filename"].endswith("20260525_020000.json")


def test_guidance_versions_per_fund(tmp_path):
    cfg = AppConfig()
    cfg.db_path = tmp_path / "fund_claude.db"
    cfg.optimizer.compiled_dir = tmp_path / "compiled"
    _write_guidance(cfg, "Claude-only guidance.")
    assert guidance_versions(cfg)["current"]["instructions"] == "Claude-only guidance."
    # A different fund (different db stem) sees no guidance
    other = AppConfig()
    other.db_path = tmp_path / "fund.db"
    other.optimizer.compiled_dir = tmp_path / "compiled"
    assert guidance_versions(other)["current"] is None


# ── template rendering (catches template bugs, both fund + sim contexts) ───────

def test_learnings_template_renders_main_and_sim(cfg, store):
    _learning(store, "calibration", "A lesson.")
    ctx = learnings_context(cfg, store)
    main = _jinja.get_template("learnings.html").render(**ctx)
    assert "A lesson." in main and "Learnings" in main
    # Sim context adds sim_prefix — must still render
    sim = _jinja.get_template("learnings.html").render(**ctx, sim_prefix="/sim-claude", sim_label="Claude")
    assert "A lesson." in sim


def test_prompt_template_renders_with_and_without_guidance(cfg):
    empty = _jinja.get_template("prompt.html").render(**prompt_context(cfg))
    assert "No optimized guidance yet" in empty and "Beat OMXSPI." in empty
    _write_guidance(cfg, "Prefer momentum entries.")
    filled = _jinja.get_template("prompt.html").render(**prompt_context(cfg), sim_prefix="/sim")
    assert "Prefer momentum entries." in filled and "claude-opus-4-8" in filled
