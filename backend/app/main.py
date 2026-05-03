import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.routers import metrics, predict

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="MarketPulse API",
    description="CNN+LSTM+Attention and ML model metrics + live prediction for stock trend direction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(metrics.router, prefix="/api")
app.include_router(predict.router, prefix="/api")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "MarketPulse API",
        "docs":    "/docs",
        "endpoints": {
            "ml_metrics":  "/api/metrics/ml",
            "dl_metrics":  "/api/metrics/dl",
            "dl_predict":  "/api/predict/dl  (POST {ticker: 'AAPL'})",
        },
        }

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")