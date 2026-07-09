"""Shared context builders for the Learnings and Prompt pages.

Both the real-money fund (web/app.py) and each simulation (web/sim.py) render
these, always scoped to that one fund's own store + config + guidance artifact —
the GPT and Claude sims share a mandate file but learn separately.
"""
from __future__ import annotations

from fundmgr.config import AppConfig
from fundmgr.engine.optimizer import guidance_versions
from fundmgr.state.store import Store


def learnings_context(cfg: AppConfig, store: Store) -> dict:
    learnings = store.get_active_learnings()
    by_category: dict[str, list[dict]] = {}
    for lrn in learnings:
        by_category.setdefault(lrn.category, []).append({
            "body": lrn.body,
            "created": lrn.created_at.strftime("%Y-%m-%d"),
            "run_count": len(lrn.run_ids),
        })
    return {
        "fund_label": cfg.display_name,
        "total": len(learnings),
        "by_category": by_category,
        "categories": sorted(by_category.keys()),
        "active_page": "learnings",
    }


def prompt_context(cfg: AppConfig) -> dict:
    try:
        mandate = cfg.mandate_path.read_text().strip()
    except OSError:
        mandate = "(mandate file not found)"
    return {
        "fund_label": cfg.display_name,
        "mandate": mandate,
        "mandate_filename": cfg.mandate_path.name,
        "guidance": guidance_versions(cfg),
        "active_page": "prompt",
    }
