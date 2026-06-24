from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from fundmgr.state.models import DecisionOutcome, Learning, NavPoint, Position, RecommendationLog, Transaction

SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    ticker      TEXT PRIMARY KEY,
    shares      REAL NOT NULL DEFAULT 0,
    avg_cost_sek REAL NOT NULL DEFAULT 0,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cash (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    balance_sek REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS transactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    side        TEXT NOT NULL,
    shares      REAL NOT NULL,
    price_sek   REAL NOT NULL,
    fee_sek     REAL NOT NULL,
    source      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recommendations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL UNIQUE,
    timestamp       TEXT NOT NULL,
    prompt_snapshot TEXT NOT NULL,
    llm_response    TEXT NOT NULL,
    guardrail_log   TEXT NOT NULL,
    actions_json    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS nav_history (
    date                TEXT PRIMARY KEY,
    portfolio_nav_sek   REAL NOT NULL,
    benchmark_value     REAL NOT NULL,
    cash_sek            REAL NOT NULL
);

-- Tracks the outcome of each LLM action recommendation once enough time has passed.
-- Populated retrospectively (e.g. 4 weeks after the decision).
CREATE TABLE IF NOT EXISTS decision_outcomes (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL,
    ticker              TEXT NOT NULL,
    action              TEXT NOT NULL,       -- buy | sell | hold
    confidence          REAL,
    price_at_decision   REAL,
    price_at_evaluation REAL,
    benchmark_return_pct REAL,              -- benchmark return over same window
    position_return_pct REAL,               -- stock return over same window
    outperformed        INTEGER,            -- 1=yes, 0=no, NULL=not yet evaluated
    evaluation_date     TEXT,
    thesis              TEXT,               -- LLM's stated thesis at time of decision
    UNIQUE(run_id, ticker)
);

-- Distilled lessons fed back into future prompts.
-- Can be rule-generated (calibration stats) or LLM-generated summaries.
CREATE TABLE IF NOT EXISTS learnings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    category    TEXT NOT NULL,   -- calibration | sector_bias | timing | general
    body        TEXT NOT NULL,   -- plain-text lesson (injected into future prompts)
    run_ids     TEXT,            -- JSON list of run_ids this learning derives from
    is_active   INTEGER NOT NULL DEFAULT 1,
    superseded_by INTEGER REFERENCES learnings(id)
);

-- Daily OHLCV cache so we don't re-fetch on every run.
CREATE TABLE IF NOT EXISTS price_cache (
    ticker      TEXT NOT NULL,
    date        TEXT NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL NOT NULL,
    volume      REAL,
    fetched_at  TEXT NOT NULL,
    PRIMARY KEY (ticker, date)
);

-- Benchmark (^OMXSPI) daily close cache.
CREATE TABLE IF NOT EXISTS benchmark_cache (
    date        TEXT PRIMARY KEY,
    close       REAL NOT NULL,
    fetched_at  TEXT NOT NULL
);

-- Weekly fundamentals cache (valuation, growth, quality from yfinance .info).
-- Refreshed at most once per ttl_days to keep runs fast.
CREATE TABLE IF NOT EXISTS fundamentals_cache (
    ticker      TEXT PRIMARY KEY,
    data_json   TEXT NOT NULL,
    fetched_at  TEXT NOT NULL
);

-- Per-position stop-loss and take-profit levels, persisted independently of run history.
-- Set when a buy is approved; cleared on full exit.
CREATE TABLE IF NOT EXISTS position_stops (
    ticker          TEXT PRIMARY KEY,
    stop_pct        REAL,
    take_profit_pct REAL,
    set_at          TEXT NOT NULL
);

-- Per-ticker news sentiment cache.
CREATE TABLE IF NOT EXISTS news_cache (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    headline        TEXT NOT NULL,
    source_url      TEXT,
    published_at    TEXT,
    sentiment_label TEXT,   -- positive | negative | neutral
    sentiment_score REAL,
    fetched_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_news_ticker ON news_cache (ticker, fetched_at);

-- Tracks news-triggered early runs to prevent duplicate alerts.
CREATE TABLE IF NOT EXISTS news_triggers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    triggered_at    TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    headline        TEXT NOT NULL,
    sentiment_label TEXT NOT NULL,
    sentiment_score REAL NOT NULL,
    article_hash    TEXT NOT NULL UNIQUE  -- SHA1(ticker+headline) — deduplicates across polls
);
CREATE INDEX IF NOT EXISTS idx_triggers_ticker ON news_triggers (ticker, triggered_at);
CREATE TABLE IF NOT EXISTS daily_price_alerts (
    ticker      TEXT NOT NULL,
    alert_date  TEXT NOT NULL,
    PRIMARY KEY (ticker, alert_date)
);

-- Small key-value store for fund-level flags (e.g. one-shot reminders).
CREATE TABLE IF NOT EXISTS app_meta (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);
"""


class Store:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA)
        # Migrations: add columns to existing tables that predate schema changes
        for stmt in [
            "ALTER TABLE position_stops ADD COLUMN take_profit_pct REAL",
            "ALTER TABLE recommendations ADD COLUMN score REAL",
            "ALTER TABLE recommendations ADD COLUMN sampling_log TEXT",
        ]:
            with self._conn() as conn:
                try:
                    conn.execute(stmt)
                except Exception:
                    pass  # column already exists

    # ── Cash ──────────────────────────────────────────────────────────────────

    def get_cash(self) -> float:
        with self._conn() as conn:
            row = conn.execute("SELECT balance_sek FROM cash WHERE id = 1").fetchone()
            return float(row["balance_sek"]) if row else 0.0

    def set_cash(self, balance_sek: float) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO cash (id, balance_sek) VALUES (1, ?) "
                "ON CONFLICT(id) DO UPDATE SET balance_sek = excluded.balance_sek",
                (balance_sek,),
            )

    def adjust_cash(self, delta_sek: float) -> float:
        new_balance = self.get_cash() + delta_sek
        self.set_cash(new_balance)
        return new_balance

    # ── Meta flags ────────────────────────────────────────────────────────────

    def get_meta(self, key: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute("SELECT value FROM app_meta WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO app_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def count_recommendations(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) AS n FROM recommendations").fetchone()["n"]

    # ── Positions ─────────────────────────────────────────────────────────────

    def get_positions(self) -> list[Position]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT ticker, shares, avg_cost_sek, updated_at FROM positions WHERE shares > 0"
            ).fetchall()
        return [
            Position(
                ticker=r["ticker"],
                shares=r["shares"],
                avg_cost_sek=r["avg_cost_sek"],
                updated_at=datetime.fromisoformat(r["updated_at"]),
            )
            for r in rows
        ]

    def upsert_position(self, ticker: str, shares: float, avg_cost_sek: float) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO positions (ticker, shares, avg_cost_sek, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(ticker) DO UPDATE SET "
                "shares = excluded.shares, avg_cost_sek = excluded.avg_cost_sek, updated_at = excluded.updated_at",
                (ticker, shares, avg_cost_sek, datetime.utcnow().isoformat()),
            )

    def apply_fill(self, txn: Transaction) -> None:
        """Apply a fill to positions and cash atomically."""
        with self._conn() as conn:
            # Record the transaction
            conn.execute(
                "INSERT INTO transactions (timestamp, ticker, side, shares, price_sek, fee_sek, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    txn.timestamp.isoformat(),
                    txn.ticker,
                    txn.side,
                    txn.shares,
                    txn.price_sek,
                    txn.fee_sek,
                    txn.source,
                ),
            )

            # Update position
            row = conn.execute(
                "SELECT shares, avg_cost_sek FROM positions WHERE ticker = ?", (txn.ticker,)
            ).fetchone()

            if txn.side == "buy":
                cur_shares = float(row["shares"]) if row else 0.0
                cur_cost = float(row["avg_cost_sek"]) if row else 0.0
                new_shares = cur_shares + txn.shares
                # Weighted average cost basis
                new_avg = (cur_shares * cur_cost + txn.shares * txn.price_sek) / new_shares
                conn.execute(
                    "INSERT INTO positions (ticker, shares, avg_cost_sek, updated_at) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(ticker) DO UPDATE SET "
                    "shares = excluded.shares, avg_cost_sek = excluded.avg_cost_sek, updated_at = excluded.updated_at",
                    (txn.ticker, new_shares, new_avg, txn.timestamp.isoformat()),
                )
                # Deduct from cash
                conn.execute(
                    "UPDATE cash SET balance_sek = balance_sek - ? WHERE id = 1",
                    (txn.gross_sek + txn.fee_sek,),
                )
            else:  # sell
                cur_shares = float(row["shares"]) if row else 0.0
                new_shares = max(0.0, cur_shares - txn.shares)
                avg_cost = float(row["avg_cost_sek"]) if row else 0.0
                conn.execute(
                    "UPDATE positions SET shares = ?, updated_at = ? WHERE ticker = ?",
                    (new_shares, txn.timestamp.isoformat(), txn.ticker),
                )
                # Add proceeds to cash (minus fee)
                conn.execute(
                    "UPDATE cash SET balance_sek = balance_sek + ? WHERE id = 1",
                    (txn.gross_sek - txn.fee_sek,),
                )

    # ── Transactions ──────────────────────────────────────────────────────────

    def get_transactions(self, limit: int = 100) -> list[Transaction]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM transactions ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            Transaction(
                id=r["id"],
                timestamp=datetime.fromisoformat(r["timestamp"]),
                ticker=r["ticker"],
                side=r["side"],
                shares=r["shares"],
                price_sek=r["price_sek"],
                fee_sek=r["fee_sek"],
                source=r["source"],
            )
            for r in rows
        ]

    def undo_last_fill(self) -> Transaction | None:
        """
        Atomically reverse the most recent transaction and return it, or None if none exists.
        Restores position shares/avg_cost and cash to their pre-fill state.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM transactions ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if not row:
                return None

            txn = Transaction(
                id=row["id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                ticker=row["ticker"],
                side=row["side"],
                shares=float(row["shares"]),
                price_sek=float(row["price_sek"]),
                fee_sek=float(row["fee_sek"]),
                source=row["source"],
            )

            pos = conn.execute(
                "SELECT shares, avg_cost_sek FROM positions WHERE ticker = ?", (txn.ticker,)
            ).fetchone()
            cur_shares = float(pos["shares"]) if pos else 0.0
            cur_avg    = float(pos["avg_cost_sek"]) if pos else 0.0

            if txn.side == "buy":
                # Reverse weighted average: recover old_shares and old_cost
                old_shares = cur_shares - txn.shares
                if old_shares <= 0:
                    # Position didn't exist before this fill — remove it entirely
                    conn.execute("DELETE FROM positions WHERE ticker = ?", (txn.ticker,))
                else:
                    old_avg = (cur_avg * cur_shares - txn.shares * txn.price_sek) / old_shares
                    conn.execute(
                        "UPDATE positions SET shares = ?, avg_cost_sek = ? WHERE ticker = ?",
                        (old_shares, old_avg, txn.ticker),
                    )
                # Return cash that was spent
                conn.execute(
                    "UPDATE cash SET balance_sek = balance_sek + ? WHERE id = 1",
                    (txn.gross_sek + txn.fee_sek,),
                )
            else:  # sell
                # Restore shares; avg_cost unchanged on sells
                old_shares = cur_shares + txn.shares
                conn.execute(
                    "UPDATE positions SET shares = ? WHERE ticker = ?",
                    (old_shares, txn.ticker),
                )
                # Deduct proceeds that were credited
                conn.execute(
                    "UPDATE cash SET balance_sek = balance_sek - ? WHERE id = 1",
                    (txn.gross_sek - txn.fee_sek,),
                )

            conn.execute("DELETE FROM transactions WHERE id = ?", (txn.id,))

        return txn

    def has_sent_daily_drop_alert(self, ticker: str, date: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM daily_price_alerts WHERE ticker = ? AND alert_date = ?",
                (ticker, date),
            ).fetchone()
            return row is not None

    def record_daily_drop_alert(self, ticker: str, date: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO daily_price_alerts (ticker, alert_date) VALUES (?, ?)",
                (ticker, date),
            )

    def total_fees_paid(self) -> float:
        with self._conn() as conn:
            row = conn.execute("SELECT COALESCE(SUM(fee_sek), 0) as total FROM transactions").fetchone()
            return float(row["total"])

    # ── Recommendations ───────────────────────────────────────────────────────

    def save_recommendation(self, rec: RecommendationLog) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO recommendations (run_id, timestamp, prompt_snapshot, llm_response, guardrail_log, actions_json, sampling_log) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    rec.run_id,
                    rec.timestamp.isoformat(),
                    rec.prompt_snapshot,
                    rec.llm_response,
                    rec.guardrail_log,
                    rec.actions_json,
                    rec.sampling_log,
                ),
            )

    def get_last_recommendation(self) -> RecommendationLog | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM recommendations ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        return RecommendationLog(
            id=row["id"],
            run_id=row["run_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            prompt_snapshot=row["prompt_snapshot"],
            llm_response=row["llm_response"],
            guardrail_log=row["guardrail_log"],
            actions_json=row["actions_json"],
            sampling_log=row["sampling_log"] or "",
        )

    def get_recommendation_by_run_id(self, run_id: str) -> RecommendationLog | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM recommendations WHERE run_id = ?", (run_id,)
            ).fetchone()
        if not row:
            return None
        return RecommendationLog(
            id=row["id"],
            run_id=row["run_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            prompt_snapshot=row["prompt_snapshot"],
            llm_response=row["llm_response"],
            guardrail_log=row["guardrail_log"],
            actions_json=row["actions_json"],
            sampling_log=row["sampling_log"] or "",
        )

    # ── NAV history ───────────────────────────────────────────────────────────

    def upsert_nav(self, nav: NavPoint) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO nav_history (date, portfolio_nav_sek, benchmark_value, cash_sek) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(date) DO UPDATE SET "
                "portfolio_nav_sek = excluded.portfolio_nav_sek, "
                "benchmark_value = excluded.benchmark_value, "
                "cash_sek = excluded.cash_sek",
                (nav.date, nav.portfolio_nav_sek, nav.benchmark_value, nav.cash_sek),
            )

    def get_nav_history(self) -> list[NavPoint]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM nav_history ORDER BY date ASC"
            ).fetchall()
        return [
            NavPoint(
                date=r["date"],
                portfolio_nav_sek=r["portfolio_nav_sek"],
                benchmark_value=r["benchmark_value"],
                cash_sek=r["cash_sek"],
            )
            for r in rows
        ]

    # ── DSPy / MIPRO scoring and export ───────────────────────────────────────

    def score_runs(self, min_days: int = 7) -> list[dict]:
        """
        Score completed weekly runs by excess return vs benchmark over the
        following week.  Only scores runs that are at least min_days old and
        have no score yet.

        Returns a list of dicts: {run_id, timestamp, score, nav_start, nav_end}.
        """
        import json as _json
        from datetime import datetime, timedelta

        cutoff = (datetime.utcnow() - timedelta(days=min_days)).strftime("%Y-%m-%d")

        with self._conn() as conn:
            runs = conn.execute(
                "SELECT run_id, timestamp FROM recommendations "
                "WHERE score IS NULL AND substr(timestamp, 1, 10) <= ? "
                "ORDER BY timestamp ASC",
                (cutoff,),
            ).fetchall()
            nav_rows = conn.execute(
                "SELECT date, portfolio_nav_sek, benchmark_value FROM nav_history ORDER BY date ASC"
            ).fetchall()

        if not nav_rows:
            return []

        nav_by_date = {r["date"]: r for r in nav_rows}
        nav_dates = sorted(nav_by_date.keys())

        def nearest_nav(target_date: str) -> dict | None:
            """Return the NAV row closest to target_date (within ±3 days)."""
            best, best_delta = None, 999
            for d in nav_dates:
                delta = abs((datetime.strptime(d, "%Y-%m-%d") - datetime.strptime(target_date, "%Y-%m-%d")).days)
                if delta < best_delta:
                    best, best_delta = nav_by_date[d], delta
            return best if best_delta <= 3 else None

        scored = []
        for run in runs:
            run_date = run["timestamp"][:10]
            run_dt = datetime.strptime(run_date, "%Y-%m-%d")
            end_date = (run_dt + timedelta(days=7)).strftime("%Y-%m-%d")

            nav_start = nearest_nav(run_date)
            nav_end   = nearest_nav(end_date)
            if nav_start is None or nav_end is None:
                continue

            nav_ret   = nav_end["portfolio_nav_sek"] / nav_start["portfolio_nav_sek"] - 1
            bench_ret = nav_end["benchmark_value"]   / nav_start["benchmark_value"]   - 1
            score     = round(nav_ret - bench_ret, 6)

            with self._conn() as conn:
                conn.execute(
                    "UPDATE recommendations SET score = ? WHERE run_id = ?",
                    (score, run["run_id"]),
                )
            scored.append({
                "run_id":    run["run_id"],
                "timestamp": run["timestamp"],
                "nav_start": nav_start["portfolio_nav_sek"],
                "nav_end":   nav_end["portfolio_nav_sek"],
                "score":     score,
            })
        return scored

    def export_dspy(self, output_path: str) -> int:
        """
        Export all scored recommendations to a JSONL file for DSPy/MIPRO.

        Each line: {run_id, timestamp, score, system_message, user_message, llm_response}

        Returns the number of examples written.
        """
        import json as _json

        with self._conn() as conn:
            rows = conn.execute(
                "SELECT run_id, timestamp, prompt_snapshot, llm_response, score "
                "FROM recommendations WHERE score IS NOT NULL ORDER BY timestamp ASC"
            ).fetchall()

        count = 0
        with open(output_path, "w") as f:
            for r in rows:
                try:
                    snap = _json.loads(r["prompt_snapshot"])
                except Exception:
                    continue
                example = {
                    "run_id":           r["run_id"],
                    "timestamp":        r["timestamp"],
                    "score":            r["score"],
                    "snapshot_version": snap.get("snapshot_version", 1),
                    # Flat — exact replay / audit (always present)
                    "system_message":   snap.get("system_message", ""),
                    "user_message":     snap.get("user_message", ""),
                    # Fielded + regime — present for v2+, null for legacy v1 rows
                    "fields":           snap.get("fields"),
                    "regime":           snap.get("regime"),
                    "llm_response":     r["llm_response"],
                }
                f.write(_json.dumps(example) + "\n")
                count += 1
        return count

    def get_rejection_stats(self) -> dict:
        """Aggregate the two rates the Refine gate depends on, across all runs:

          - malformed-sample rate (from sampling_log): how often individual
            consensus samples fail to parse — the case Refine would retry.
          - guardrail drop/clip rate (from guardrail_log): how often the model
            proposes actions that violate hard limits — the case Refine could
            pre-empt before guardrails clip/reject them.

        High either rate ⇒ Refine may pay for its call-volume cost; low ⇒ not
        worth it. Pure measurement, no decision baked in.
        """
        import json as _json

        with self._conn() as conn:
            rows = conn.execute(
                "SELECT sampling_log, guardrail_log FROM recommendations"
            ).fetchall()

        runs = runs_with_failure = 0
        samples_requested = samples_failed = 0
        verdicts_total = verdicts_rejected = verdicts_clipped = 0

        for r in rows:
            runs += 1
            if r["sampling_log"]:
                try:
                    s = _json.loads(r["sampling_log"])
                    samples_requested += s.get("requested", 0)
                    samples_failed += s.get("failed", 0)
                    if s.get("failed", 0):
                        runs_with_failure += 1
                except Exception:
                    pass
            if r["guardrail_log"]:
                try:
                    for v in _json.loads(r["guardrail_log"]):
                        verdicts_total += 1
                        if v.get("status") == "REJECTED":
                            verdicts_rejected += 1
                        elif v.get("status") == "CLIPPED":
                            verdicts_clipped += 1
                except Exception:
                    pass

        def pct(num: int, den: int) -> float:
            return round(100 * num / den, 2) if den else 0.0

        return {
            "runs": runs,
            "samples_requested": samples_requested,
            "samples_failed": samples_failed,
            "sample_failure_pct": pct(samples_failed, samples_requested),
            "runs_with_any_failure": runs_with_failure,
            "guardrail_verdicts": verdicts_total,
            "guardrail_rejected": verdicts_rejected,
            "guardrail_clipped": verdicts_clipped,
            "guardrail_reject_pct": pct(verdicts_rejected, verdicts_total),
            "guardrail_clip_pct": pct(verdicts_clipped, verdicts_total),
        }

    def get_decisions_for_ticker(self, ticker: str, limit: int = 3) -> list[dict]:
        """Most recent decisions on a ticker (newest first), for stop-loss review.

        Returns dicts: {timestamp, action, confidence, thesis, price_at_decision}.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT r.timestamp AS timestamp, do.action AS action, do.confidence AS confidence, "
                "do.thesis AS thesis, do.price_at_decision AS price_at_decision "
                "FROM decision_outcomes do JOIN recommendations r ON do.run_id = r.run_id "
                "WHERE do.ticker = ? ORDER BY r.timestamp DESC LIMIT ?",
                (ticker.upper(), limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Decision outcomes ─────────────────────────────────────────────────────

    def seed_outcomes_for_run(self, run_id: str, actions_json: str) -> None:
        """Seed unevaluated outcome rows when a run is saved, for later evaluation."""
        import json as _json
        actions = _json.loads(actions_json)
        with self._conn() as conn:
            for a in actions:
                conn.execute(
                    "INSERT OR IGNORE INTO decision_outcomes "
                    "(run_id, ticker, action, confidence, price_at_decision, thesis) VALUES (?, ?, ?, ?, ?, ?)",
                    (run_id, a["ticker"], a["side"], a.get("confidence"), a.get("sek_estimate"), a.get("thesis")),
                )

    def update_outcome(self, outcome: "DecisionOutcome") -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE decision_outcomes SET "
                "price_at_evaluation = ?, benchmark_return_pct = ?, position_return_pct = ?, "
                "outperformed = ?, evaluation_date = ? "
                "WHERE run_id = ? AND ticker = ?",
                (
                    outcome.price_at_evaluation,
                    outcome.benchmark_return_pct,
                    outcome.position_return_pct,
                    1 if outcome.outperformed else 0 if outcome.outperformed is not None else None,
                    outcome.evaluation_date,
                    outcome.run_id,
                    outcome.ticker,
                ),
            )

    def get_pending_outcomes(self, older_than_days: int = 28) -> list["DecisionOutcome"]:
        """Return outcomes not yet evaluated that are old enough to assess."""
        from fundmgr.state.models import DecisionOutcome as DO
        cutoff = datetime.utcnow().isoformat()[:10]
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT do.*, r.timestamp as run_ts FROM decision_outcomes do "
                "JOIN recommendations r ON do.run_id = r.run_id "
                "WHERE do.outperformed IS NULL AND DATE(r.timestamp) <= DATE(?, ?)",
                (cutoff, f"-{older_than_days} days"),
            ).fetchall()
        return [
            DO(
                id=r["id"], run_id=r["run_id"], ticker=r["ticker"],
                action=r["action"], confidence=r["confidence"],
                price_at_decision=r["price_at_decision"], thesis=r["thesis"],
            )
            for r in rows
        ]

    def get_calibration_stats(self) -> dict:
        """Return accuracy stats by confidence bucket for use in learnings generation."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT confidence, outperformed FROM decision_outcomes "
                "WHERE outperformed IS NOT NULL AND action = 'buy'"
            ).fetchall()

        if not rows:
            return {}

        buckets: dict[str, list[int]] = {"high": [], "medium": [], "low": []}
        for r in rows:
            c = r["confidence"] or 0.0
            bucket = "high" if c >= 0.7 else "medium" if c >= 0.4 else "low"
            buckets[bucket].append(r["outperformed"])

        return {
            k: {"n": len(v), "hit_rate": sum(v) / len(v) if v else None}
            for k, v in buckets.items()
        }

    # ── Learnings ─────────────────────────────────────────────────────────────

    def save_learning(self, learning: "Learning") -> int:
        import json as _json
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO learnings (created_at, category, body, run_ids, is_active) VALUES (?, ?, ?, ?, 1)",
                (
                    learning.created_at.isoformat(),
                    learning.category,
                    learning.body,
                    _json.dumps(learning.run_ids),
                ),
            )
            return cur.lastrowid

    def get_active_learnings(self) -> list["Learning"]:
        import json as _json
        from fundmgr.state.models import Learning as L
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM learnings WHERE is_active = 1 ORDER BY created_at DESC"
            ).fetchall()
        return [
            L(
                id=r["id"],
                category=r["category"],
                body=r["body"],
                run_ids=_json.loads(r["run_ids"] or "[]"),
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in rows
        ]

    def supersede_learning(self, old_id: int, new_id: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE learnings SET is_active = 0, superseded_by = ? WHERE id = ?",
                (new_id, old_id),
            )

    # ── Price cache ───────────────────────────────────────────────────────────

    def save_prices(self, ticker: str, rows: list[dict]) -> None:
        """rows: list of {date, open, high, low, close, volume}."""
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO price_cache (ticker, date, open, high, low, close, volume, fetched_at) "
                "VALUES (:ticker, :date, :open, :high, :low, :close, :volume, :fetched_at)",
                [{**r, "ticker": ticker, "fetched_at": now} for r in rows],
            )

    def get_prices(self, ticker: str, since_date: str | None = None) -> list[dict]:
        with self._conn() as conn:
            if since_date:
                rows = conn.execute(
                    "SELECT date, open, high, low, close, volume FROM price_cache "
                    "WHERE ticker = ? AND date >= ? ORDER BY date ASC",
                    (ticker, since_date),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT date, open, high, low, close, volume FROM price_cache "
                    "WHERE ticker = ? ORDER BY date ASC",
                    (ticker,),
                ).fetchall()
        return [dict(r) for r in rows]

    def latest_price_date(self, ticker: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MAX(date) as d FROM price_cache WHERE ticker = ?", (ticker,)
            ).fetchone()
        return row["d"] if row and row["d"] else None

    def save_benchmark(self, rows: list[dict]) -> None:
        """rows: list of {date, close}."""
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO benchmark_cache (date, close, fetched_at) VALUES (:date, :close, :fetched_at)",
                [{**r, "fetched_at": now} for r in rows],
            )

    def get_benchmark(self, since_date: str | None = None) -> list[dict]:
        with self._conn() as conn:
            if since_date:
                rows = conn.execute(
                    "SELECT date, close FROM benchmark_cache WHERE date >= ? ORDER BY date ASC",
                    (since_date,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT date, close FROM benchmark_cache ORDER BY date ASC"
                ).fetchall()
        return [dict(r) for r in rows]

    def save_news_sentiment(self, ticker: str, items: list[dict]) -> None:
        """items: list of {headline, source_url, published_at, sentiment_label, sentiment_score}."""
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO news_cache (ticker, headline, source_url, published_at, sentiment_label, sentiment_score, fetched_at) "
                "VALUES (:ticker, :headline, :source_url, :published_at, :sentiment_label, :sentiment_score, :fetched_at)",
                [{**item, "ticker": ticker, "fetched_at": now} for item in items],
            )

    def get_recent_news(self, ticker: str, since_date: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT headline, published_at, sentiment_label, sentiment_score FROM news_cache "
                "WHERE ticker = ? AND fetched_at >= ? ORDER BY fetched_at DESC",
                (ticker, since_date),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── News triggers ─────────────────────────────────────────────────────────

    def has_triggered(self, article_hash: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM news_triggers WHERE article_hash = ?", (article_hash,)
            ).fetchone()
        return row is not None

    def last_trigger_at(self, ticker: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MAX(triggered_at) as ts FROM news_triggers WHERE ticker = ?", (ticker,)
            ).fetchone()
        return row["ts"] if row and row["ts"] else None

    def record_trigger(
        self,
        ticker: str,
        headline: str,
        sentiment_label: str,
        sentiment_score: float,
        article_hash: str,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO news_triggers "
                "(triggered_at, ticker, headline, sentiment_label, sentiment_score, article_hash) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), ticker, headline, sentiment_label, sentiment_score, article_hash),
            )

    # ── Fundamentals cache ────────────────────────────────────────────────────

    def save_fundamentals(self, ticker: str, data: dict) -> None:
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO fundamentals_cache (ticker, data_json, fetched_at) "
                "VALUES (?, ?, ?)",
                (ticker, json.dumps(data), now),
            )

    def get_fundamentals(self, ticker: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT data_json FROM fundamentals_cache WHERE ticker = ?", (ticker,)
            ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def get_stale_fundamentals_tickers(self, tickers: list[str], ttl_days: int = 7) -> list[str]:
        """Return tickers whose fundamentals cache entry is missing or older than ttl_days."""
        from datetime import timedelta
        cutoff = (datetime.utcnow() - timedelta(days=ttl_days)).isoformat()
        with self._conn() as conn:
            fresh = {
                row["ticker"]
                for row in conn.execute(
                    "SELECT ticker FROM fundamentals_cache WHERE fetched_at > ?", (cutoff,)
                ).fetchall()
            }
        return [t for t in tickers if t not in fresh]

    def get_all_fundamentals(self, tickers: list[str]) -> dict[str, dict]:
        """Return cached fundamentals for all requested tickers as {ticker: data_dict}."""
        if not tickers:
            return {}
        placeholders = ",".join("?" * len(tickers))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT ticker, data_json FROM fundamentals_cache WHERE ticker IN ({placeholders})",
                tickers,
            ).fetchall()
        return {r["ticker"]: json.loads(r["data_json"]) for r in rows}

    # ── Position stops ────────────────────────────────────────────────────────

    def set_position_stop(
        self,
        ticker: str,
        stop_pct: float | None = None,
        take_profit_pct: float | None = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO position_stops (ticker, stop_pct, take_profit_pct, set_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(ticker) DO UPDATE SET "
                "stop_pct = excluded.stop_pct, take_profit_pct = excluded.take_profit_pct, set_at = excluded.set_at",
                (ticker, stop_pct, take_profit_pct, datetime.utcnow().isoformat()),
            )

    def clear_position_stop(self, ticker: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM position_stops WHERE ticker = ?", (ticker,))

    def get_position_stops(self) -> dict[str, dict]:
        """Return {ticker: {stop_pct, take_profit_pct}} for all stored positions."""
        with self._conn() as conn:
            rows = conn.execute("SELECT ticker, stop_pct, take_profit_pct FROM position_stops").fetchall()
        return {
            r["ticker"]: {
                "stop_pct": float(r["stop_pct"]) if r["stop_pct"] is not None else None,
                "take_profit_pct": float(r["take_profit_pct"]) if r["take_profit_pct"] is not None else None,
            }
            for r in rows
        }

    def get_effective_stops(self) -> dict[str, dict]:
        """Stops for held positions, falling back to the level the fund decided
        at buy time when none was explicitly persisted.

        Positions bought under older code never had their stop_loss_pct written
        to position_stops. This recovers them from the most recent recommendation
        action carrying a stop/take-profit, so stop checks aren't silently blind
        to those holdings. Explicit position_stops always take precedence.
        """
        import json as _json

        stops = self.get_position_stops()
        held = {p.ticker for p in self.get_positions()}
        missing = held - {t for t, lv in stops.items() if lv.get("stop_pct") is not None}
        if not missing:
            return stops

        with self._conn() as conn:
            rows = conn.execute(
                "SELECT actions_json FROM recommendations ORDER BY timestamp DESC"
            ).fetchall()
        for r in rows:
            if not missing:
                break
            try:
                actions = _json.loads(r["actions_json"])
            except Exception:
                continue
            for a in actions:
                tk = (a.get("ticker") or "").upper()
                if tk in missing and (a.get("stop_loss_pct") or a.get("take_profit_pct")):
                    stops[tk] = {
                        "stop_pct": a.get("stop_loss_pct"),
                        "take_profit_pct": a.get("take_profit_pct"),
                        "from_recommendation": True,
                    }
                    missing.discard(tk)
        return stops

    # ── Helpers ───────────────────────────────────────────────────────────────

    def is_initialised(self) -> bool:
        return self.get_cash() > 0

    def initialise(self, capital_sek: float) -> None:
        if self.is_initialised():
            raise RuntimeError("Portfolio already initialised — use 'fund status' to inspect.")
        self.set_cash(capital_sek)

    # Fund state + history — wiped on reset. Market-data caches are NOT in this
    # list (they're regime-neutral and expensive to refetch).
    _STATE_TABLES = (
        "positions",
        "transactions",
        "recommendations",
        "nav_history",
        "decision_outcomes",
        "learnings",
        "position_stops",
        "news_triggers",
        "daily_price_alerts",
        "app_meta",
    )
    _CACHE_TABLES = (
        "price_cache",
        "benchmark_cache",
        "fundamentals_cache",
        "news_cache",
    )

    def reset(self, capital_sek: float, purge_cache: bool = False) -> dict[str, int]:
        """Wipe all portfolio state + decision history and re-initialise to a
        fresh `capital_sek` balance, as if the fund just started.

        Market-data caches (prices, benchmark, fundamentals, news) are preserved
        by default since they're regime-neutral and costly to refetch; pass
        purge_cache=True to clear those too.

        Returns {table: rows_deleted} for reporting.
        """
        tables = list(self._STATE_TABLES)
        if purge_cache:
            tables += list(self._CACHE_TABLES)

        deleted: dict[str, int] = {}
        with self._conn() as conn:
            for t in tables:
                cur = conn.execute(f"DELETE FROM {t}")
                deleted[t] = cur.rowcount
            # Reset cash to the fresh baseline (single-row table).
            conn.execute("DELETE FROM cash")
            conn.execute("INSERT INTO cash (id, balance_sek) VALUES (1, ?)", (capital_sek,))
        return deleted
