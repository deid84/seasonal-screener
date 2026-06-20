"""
server.py
FastAPI server that exposes the screening and backtest results stored in
iv_archive.db and serves the static frontend.

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000

Production (with auto-reload disabled):
    uvicorn server:app --host 0.0.0.0 --port 8000 --workers 2
"""
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import db
from iv_archive import load_iv_history

DB_PATH = os.environ.get("IV_ARCHIVE_DB", "iv_archive.db")

app = FastAPI(title="py-screener", version="1.0")

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
    """
    Returns all ticker results from the most recent screening run,
    sorted by score descending.
    """
    results = db.load_latest_screening_run(DB_PATH)
    if not results:
        return {"data": [], "message": "No screening results yet. Run screener.py --output db first."}
    return {"data": results, "run_ts": results[0]["run_ts"]}


@app.get("/api/results/{ticker}")
def get_ticker_detail(ticker: str):
    """
    Returns the score history (last 90 days) for a specific ticker.
    Useful for sparklines / trend charts.
    """
    ticker = ticker.upper()
    history = db.load_ticker_score_history(ticker, DB_PATH)
    if not history:
        raise HTTPException(status_code=404, detail=f"No results found for {ticker}")
    return {"ticker": ticker, "history": history}


@app.get("/api/iv-history/{ticker}")
def get_iv_history(ticker: str, days: int = 365):
    """
    Returns the stored IV history (date, iv_mid_pct) for the given ticker.
    Used to draw the IV chart in the frontend.
    """
    ticker = ticker.upper()
    rows = load_iv_history(ticker, days=days, db_path=DB_PATH)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No IV history for {ticker}")
    return {"ticker": ticker, "history": [{"date": r[0], "iv": r[1]} for r in rows]}


@app.get("/api/backtest/{ticker}")
def get_backtest(ticker: str, strategy: str | None = None):
    """
    Returns the most recent backtest result for a ticker.
    Pass ?strategy=long-call to get a specific options backtest.
    """
    ticker = ticker.upper()
    result = db.load_latest_backtest(ticker, strategy, DB_PATH)
    if not result:
        detail = f"No backtest found for {ticker}"
        if strategy:
            detail += f" (strategy: {strategy})"
        raise HTTPException(status_code=404, detail=detail)
    return {"ticker": ticker, "strategy": strategy, **result}


# ---------------------------------------------------------------------------
# Static frontend — must be last so API routes take priority
# ---------------------------------------------------------------------------

app.mount("/", StaticFiles(directory="static", html=True), name="static")
