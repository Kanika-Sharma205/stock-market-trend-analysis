"""
Prediction router — fetches live OHLCV data via yfinance and runs DL inference.
"""
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["predict"])


class PredictRequest(BaseModel):
    ticker: str = "AAPL"


@router.post("/dl")
def predict_dl(req: PredictRequest):
    ticker = req.ticker.upper().strip()
    if not ticker:
        raise HTTPException(400, "ticker is required")

    try:
        import yfinance as yf
    except ImportError:
        raise HTTPException(503, "yfinance not installed — run: pip install yfinance")

    try:
        raw = yf.download(ticker, period="90d", auto_adjust=True, progress=False)
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch data for {ticker}: {e}")

    if raw is None or len(raw) < 35:
        raise HTTPException(404, f"Not enough data for ticker '{ticker}'. Need ≥35 trading days.")

    # Flatten MultiIndex columns if present
    if hasattr(raw.columns, "levels"):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]

    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(set(raw.columns)):
        raise HTTPException(422, f"Missing columns for {ticker}. Got: {list(raw.columns)}")

    raw = raw[list(required)].dropna()

    try:
        from app.services.dl_service import predict_from_ohlcv
        result = predict_from_ohlcv(raw)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(500, f"Prediction failed: {e}")

    last_close = float(raw["close"].iloc[-1])
    last_date  = str(raw.index[-1].date())

    return {
        "ticker":     ticker,
        "last_date":  last_date,
        "last_close": round(last_close, 2),
        **result,
    }
