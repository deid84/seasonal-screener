"""
db.py
Persistence layer for screening and backtest results.
Uses the same SQLite file as iv_archive (default: iv_archive.db) so all
project data lives in a single file.

Tables managed here:
  screening_results  — one row per ticker per run of screener.py
  backtest_results   — one row per ticker+strategy per run of backtest.py
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DB = "iv_archive.db"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            ticker     TEXT PRIMARY KEY,
            added_date TEXT NOT NULL,
            notes      TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS screening_results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date    TEXT NOT NULL,
            run_ts      TEXT NOT NULL,
            ticker      TEXT NOT NULL,
            years       INTEGER,
            score       REAL,
            result_json TEXT NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sr_ticker_date "
        "ON screening_results(ticker, run_date)"
    )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS backtest_results (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date     TEXT NOT NULL,
            run_ts       TEXT NOT NULL,
            ticker       TEXT NOT NULL,
            years        INTEGER,
            strategy     TEXT,
            entry_avg    REAL,
            entry_wr     REAL,
            summary_json TEXT NOT NULL,
            trades_json  TEXT NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_bt_ticker_date "
        "ON backtest_results(ticker, run_date)"
    )
    conn.commit()
    return conn


class _SafeEncoder(json.JSONEncoder):
    """Handles pandas and numpy types that default JSON encoder rejects."""
    def default(self, obj):
        if isinstance(obj, pd.DataFrame):
            return obj.reset_index().to_dict(orient="records")
        if isinstance(obj, pd.Series):
            return {k: self.default(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                    for k, v in obj.items()}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return super().default(obj)


def _to_json(data) -> str:
    return json.dumps(data, cls=_SafeEncoder)


# ---------------------------------------------------------------------------
# Screening results
# ---------------------------------------------------------------------------

def save_screening_result(ticker: str, years: int, report: dict, score: float,
                           db_path: str = DEFAULT_DB):
    """Persist a single ticker's screening report to the database."""
    now = datetime.utcnow()
    conn = _connect(db_path)
    try:
        conn.execute("""
            INSERT INTO screening_results (run_date, run_ts, ticker, years, score, result_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            now.strftime("%Y-%m-%d"),
            now.isoformat(timespec="seconds"),
            ticker, years,
            score,
            _to_json(report),
        ))
        conn.commit()
    finally:
        conn.close()


def load_latest_screening_run(db_path: str = DEFAULT_DB) -> list[dict]:
    """
    Returns all ticker results from the most recent run date,
    sorted by score descending.
    """
    if not Path(db_path).exists():
        return []
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT MAX(run_date) FROM screening_results").fetchone()
        latest_date = row[0] if row else None
        if not latest_date:
            return []
        rows = conn.execute(
            "SELECT ticker, score, result_json, run_ts FROM screening_results "
            "WHERE run_date = ? "
            "GROUP BY ticker HAVING run_ts = MAX(run_ts) "
            "ORDER BY score DESC",
            (latest_date,),
        ).fetchall()
        return [
            {"ticker": r[0], "score": r[1], "result": json.loads(r[2]), "run_ts": r[3]}
            for r in rows
        ]
    finally:
        conn.close()


def load_ticker_score_history(ticker: str, db_path: str = DEFAULT_DB) -> list[dict]:
    """Returns (date, score) pairs for a ticker — used for trend sparklines."""
    if not Path(db_path).exists():
        return []
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT run_date, score FROM screening_results "
            "WHERE ticker = ? ORDER BY run_date ASC LIMIT 90",
            (ticker,),
        ).fetchall()
        return [{"date": r[0], "score": r[1]} for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Backtest results
# ---------------------------------------------------------------------------

def save_backtest_result(ticker: str, years: int, strategy: str | None,
                          entry_avg: float, entry_wr: float,
                          summary: dict, trades_df: pd.DataFrame,
                          db_path: str = DEFAULT_DB):
    """Persist a backtest run (price-only or options-aware) to the database."""
    now = datetime.utcnow()
    conn = _connect(db_path)
    try:
        conn.execute("""
            INSERT INTO backtest_results
            (run_date, run_ts, ticker, years, strategy, entry_avg, entry_wr,
             summary_json, trades_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now.strftime("%Y-%m-%d"),
            now.isoformat(timespec="seconds"),
            ticker, years, strategy,
            entry_avg, entry_wr,
            _to_json(summary),
            trades_df.to_json(orient="records"),
        ))
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------

def add_to_watchlist(ticker: str, notes: str | None = None,
                     db_path: str = DEFAULT_DB):
    """Add a ticker to the persistent watchlist. Silently ignored if already present."""
    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO watchlist (ticker, added_date, notes) VALUES (?, ?, ?)",
            (ticker.upper(), datetime.utcnow().strftime("%Y-%m-%d"), notes),
        )
        conn.commit()
    finally:
        conn.close()


def remove_from_watchlist(ticker: str, db_path: str = DEFAULT_DB):
    """Remove a ticker from the watchlist. No-op if not present."""
    conn = _connect(db_path)
    try:
        conn.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker.upper(),))
        conn.commit()
    finally:
        conn.close()


def load_watchlist(db_path: str = DEFAULT_DB) -> list[dict]:
    """Returns all watchlist entries sorted alphabetically."""
    if not Path(db_path).exists():
        return []
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT ticker, added_date, notes FROM watchlist ORDER BY ticker ASC"
        ).fetchall()
        return [{"ticker": r[0], "added_date": r[1], "notes": r[2]} for r in rows]
    finally:
        conn.close()


def load_latest_backtest(ticker: str, strategy: str | None = None,
                          db_path: str = DEFAULT_DB) -> dict | None:
    """Returns the most recent backtest result for a ticker + strategy."""
    if not Path(db_path).exists():
        return None
    conn = _connect(db_path)
    try:
        if strategy:
            row = conn.execute(
                "SELECT summary_json, trades_json, run_ts FROM backtest_results "
                "WHERE ticker = ? AND strategy = ? ORDER BY run_date DESC LIMIT 1",
                (ticker, strategy),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT summary_json, trades_json, run_ts FROM backtest_results "
                "WHERE ticker = ? AND strategy IS NULL ORDER BY run_date DESC LIMIT 1",
                (ticker,),
            ).fetchone()
        if not row:
            return None
        return {
            "summary": json.loads(row[0]),
            "trades": json.loads(row[1]),
            "run_ts": row[2],
        }
    finally:
        conn.close()
