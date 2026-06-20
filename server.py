"""
server.py
FastAPI server that exposes the screening and backtest results stored in
iv_archive.db and serves the static frontend.

The server also runs a built-in daily scheduler (APScheduler) so no separate
cron container is needed — a single Docker service with restart: always is enough.

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1

NOTE: always use --workers 1. Multiple workers each start their own scheduler,
which would cause duplicate screening runs and concurrent SQLite writes.

Scheduler env vars:
    TICKERS        Comma-separated list of tickers to screen (default: GLD,XRT,EQT)
    YEARS          Years of history (default: 5)
    SCHEDULE_HOUR  UTC hour to run (default: 22)
    SCHEDULE_MIN   UTC minute to run (default: 0)
"""
import logging
import os
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import db
from iv_archive import load_iv_history

DB_PATH = os.environ.get("IV_ARCHIVE_DB", "iv_archive.db")

logger = logging.getLogger("py-screener")


# ---------------------------------------------------------------------------
# Scheduled screening job
# ---------------------------------------------------------------------------

def _run_screening():
    """Runs the full screening pipeline and persists results to the DB."""
    from screener import analyze_ticker, score_opportunity
    from db import save_screening_result

    tickers = [t.strip().upper() for t in os.environ.get("TICKERS", "GLD,XRT,EQT").split(",")]
    years = int(os.environ.get("YEARS", "5"))

    logger.info("Scheduled screening started — tickers=%s years=%d", tickers, years)
    for ticker in tickers:
        try:
            report = analyze_ticker(ticker, years, iv_db=DB_PATH)
            save_screening_result(ticker, years, report, score_opportunity(report), DB_PATH)
            logger.info("  %s saved", ticker)
        except Exception as exc:
            logger.error("  %s failed: %s", ticker, exc)
    logger.info("Scheduled screening complete")


@asynccontextmanager
async def lifespan(app: FastAPI):
    hour = int(os.environ.get("SCHEDULE_HOUR", "22"))
    minute = int(os.environ.get("SCHEDULE_MIN", "0"))

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        _run_screening,
        CronTrigger(day_of_week="mon-fri", hour=hour, minute=minute, timezone="UTC"),
        id="daily_screening",
        misfire_grace_time=3600,
    )
    scheduler.start()
    logger.info("Scheduler ready — daily screening at %02d:%02d UTC (Mon–Fri)", hour, minute)
    yield
    scheduler.shutdown(wait=False)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="py-screener", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/results")
def get_latest_results():
    results = db.load_latest_screening_run(DB_PATH)
    if not results:
        return {"data": [], "message": "No screening results yet. Wait for the scheduled run or POST /api/run."}
    return {"data": results, "run_ts": results[0]["run_ts"]}


@app.get("/api/results/{ticker}")
def get_ticker_detail(ticker: str):
    ticker = ticker.upper()
    history = db.load_ticker_score_history(ticker, DB_PATH)
    if not history:
        raise HTTPException(status_code=404, detail=f"No results found for {ticker}")
    return {"ticker": ticker, "history": history}


@app.get("/api/iv-history/{ticker}")
def get_iv_history(ticker: str, days: int = 365):
    ticker = ticker.upper()
    rows = load_iv_history(ticker, days=days, db_path=DB_PATH)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No IV history for {ticker}")
    return {"ticker": ticker, "history": [{"date": r[0], "iv": r[1]} for r in rows]}


@app.get("/api/backtest/{ticker}")
def get_backtest(ticker: str, strategy: str | None = None):
    ticker = ticker.upper()
    result = db.load_latest_backtest(ticker, strategy, DB_PATH)
    if not result:
        detail = f"No backtest found for {ticker}"
        if strategy:
            detail += f" (strategy: {strategy})"
        raise HTTPException(status_code=404, detail=detail)
    return {"ticker": ticker, "strategy": strategy, **result}


@app.post("/api/run", status_code=202)
def trigger_screening():
    """Manually triggers an immediate screening run (runs in background thread)."""
    import threading
    threading.Thread(target=_run_screening, daemon=True).start()
    return {"message": "Screening started in background"}


# ---------------------------------------------------------------------------
# Static frontend — must be last so API routes take priority
# ---------------------------------------------------------------------------

app.mount("/", StaticFiles(directory="static", html=True), name="static")
