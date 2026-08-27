"""Shared context builders for the Learnings, Prompt and dashboard-review views.

Both the real-money fund (web/app.py) and each simulation (web/sim.py) render
these, always scoped to that one fund's own store + config + guidance artifact —
the GPT and Claude sims share a mandate file but learn separately.
"""
from __future__ import annotations

from fundmgr.config import AppConfig
from fundmgr.engine.optimizer import guidance_versions
from fundmgr.engine.prompt import PROMPT_LEARNING_LIMIT, select_prompt_learnings
from fundmgr.engine.review_common import follow_up, instruction, votes_str
from fundmgr.state.store import Store


def learnings_context(cfg: AppConfig, store: Store) -> dict:
    learnings = store.get_active_learnings()
    # The prompt only carries the top slice, so the page must say which ones
    # those are — "N active lessons injected into every prompt" was false for
    # every lesson below the cut.
    injected_ids = {id(sel) for sel in select_prompt_learnings(learnings)}

    by_category: dict[str, list[dict]] = {}
    for lrn in learnings:
        by_category.setdefault(lrn.category, []).append({
            "body": lrn.body,
            "created": lrn.created_at.strftime("%Y-%m-%d"),
            "run_count": len(lrn.run_ids),
            "injected": id(lrn) in injected_ids,
        })
    return {
        "fund_label": cfg.display_name,
        "total": len(learnings),
        "injected": len(injected_ids),
        "prompt_limit": PROMPT_LEARNING_LIMIT,
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


# ── Position reviews ──────────────────────────────────────────────────────────
# A stop breach or a take-profit hit produces a verdict that only a human can
# carry out. It used to exist as a Telegram message and nothing else: swipe the
# notification away and the instruction was gone, with the dashboard showing a
# position that looked untouched. These build the dashboard's side of it.

_SOURCE_LABEL = {"target_review": "Target hit", "stop_review": "Stop hit"}
_VERDICT_TONE = {
    "sell": "amber", "trim": "amber", "exit": "red", "add": "emerald",
    "raise": "sky", "hold": "slate",
}


def _shares_to_trade(verdict: str, shares: float | None, trim_pct: float | None) -> float | None:
    """How many shares the order is for, where the verdict implies a size."""
    if not shares:
        return None
    if verdict in ("exit", "sell"):
        return shares
    if verdict == "trim" and trim_pct:
        return shares * trim_pct / 100
    return None


def _target_line(r: dict) -> str:
    """What happened to the take-profit level, in the terms the alert used."""
    applied, proposed = r.get("applied_target_pct"), r.get("new_target_pct")
    old = r.get("old_target_pct")
    if applied:
        was = f"+{old:.0f}% → " if old else ""
        return f"target {was}+{applied:.0f}%"
    if proposed and r["verdict"] in ("raise", "trim"):
        return f"target unchanged (proposed +{proposed:.0f}%)"
    return ""


def review_row(r: dict, position: dict | None = None) -> dict:
    """One review verdict, rendered as an instruction rather than a conclusion.

    `position` is the dashboard's row for the same ticker, when it is still held:
    it turns "sell 50%" into a share count and a rough SEK figure, which is the
    difference between a verdict you have to do arithmetic on and one you can act
    on from the page.
    """
    verdict = r["verdict"]
    shares = (position or {}).get("shares")
    to_trade = _shares_to_trade(verdict, shares, r.get("trim_pct"))
    # Only the live price: valuing the order at cost basis would understate a
    # position that has just hit a +50% target by exactly the gain being banked.
    price = (position or {}).get("current_price")
    sek = to_trade * price if (to_trade and price) else None
    applied = r.get("applied_target_pct")
    created = r.get("created_at") or ""
    return {
        "review_id": r["review_id"],
        "ticker": r["ticker"],
        "name": (position or {}).get("name") or r["ticker"],
        "verdict": verdict,
        "verdict_label": verdict.upper(),
        "tone": _VERDICT_TONE.get(verdict, "slate"),
        "source_label": _SOURCE_LABEL.get(r["source"], r["source"]),
        "instruction": instruction(
            verdict, trim_pct=r.get("trim_pct"), new_target_pct=applied,
            shares=to_trade, sek=sek,
        ),
        "follow_up": follow_up(verdict, r.get("trim_pct"), applied, r.get("old_target_pct")),
        "target_line": _target_line(r),
        "shares_to_trade": round(to_trade) if to_trade else None,
        "shares_held": round(shares) if shares else None,
        "sek_estimate": round(sek) if sek else None,
        "what_changed": r.get("what_changed") or "",
        "rationale": r.get("rationale") or "",
        "confidence": r.get("confidence"),
        "consensus": votes_str(r["votes"], r["n_samples"]) if r.get("votes") and r.get("n_samples") else "",
        "price_at_review": r.get("price_at_review"),
        "when": created[:10],
        "when_time": created[11:16],
        "status": r.get("status", "open"),
    }


def open_reviews_context(store: Store, positions: list[dict]) -> dict:
    """The dashboard's "Decisions to action" card: what is still owed a trade.

    Scoped to names still held — an EXIT that was carried out leaves no position
    behind, and the instruction should stop being shown whether or not anyone
    remembered to tick it off.
    """
    by_ticker = {p["ticker"]: p for p in positions}
    rows = [
        review_row(r, by_ticker.get(r["ticker"]))
        for r in store.get_open_reviews(tickers=list(by_ticker))
    ]
    return {"rows": rows, "count": len(rows)}


def recent_reviews_context(store: Store, positions: list[dict], limit: int = 8) -> list[dict]:
    """The log behind the open ones — including the verdicts that asked nothing."""
    by_ticker = {p["ticker"]: p for p in positions}
    return [review_row(r, by_ticker.get(r["ticker"])) for r in store.get_recent_reviews(limit)]
