"""
Feature engineering pipeline — replicates Chapter 2A DL preprocessing exactly.
Requires: open, high, low, close, volume columns (chronologically sorted DataFrame).
Returns a DataFrame with the 16 DL features ready for the scaler.
"""
import numpy as np
import pandas as pd


def compute_dl_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects a DataFrame with columns: open, high, low, close, volume (lowercase).
    Returns a new DataFrame with the 16 DL feature columns.
    Needs at least ~60 rows to give stable EMA/ATR values; caller should pass ≥60 rows.
    """
    d = df.copy()
    d = d.sort_index()

    # ── Raw signals ───────────────────────────────────────────────────────────
    d["return"] = d["close"].pct_change()
    d["volume_change"] = np.log(d["volume"] / d["volume"].shift(1)).clip(-5, 5)

    # ── Moving averages ───────────────────────────────────────────────────────
    d["ma5"]  = d["close"].rolling(5).mean()
    d["ma20"] = d["close"].rolling(20).mean()
    d["ma5_dist"]  = (d["close"] - d["ma5"])  / d["ma5"]
    d["ma20_dist"] = (d["close"] - d["ma20"]) / d["ma20"]
    d["ma_cross_strength"] = d["ma5"] - d["ma20"]

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_mid   = d["close"].rolling(20).mean()
    bb_std   = d["close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    d["bb_width"]    = (bb_upper - bb_lower) / bb_mid
    d["bb_position"] = (d["close"] - bb_lower) / (bb_upper - bb_lower + 1e-9)

    # ── RSI (14) ──────────────────────────────────────────────────────────────
    delta = d["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    d["rsi14"] = 100 - 100 / (1 + rs)

    # ── MACD ─────────────────────────────────────────────────────────────────
    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"]      = ema12 - ema26
    d["signal"]    = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["signal"]

    # ── ATR (14) ──────────────────────────────────────────────────────────────
    high_low   = d["high"] - d["low"]
    high_close = (d["high"] - d["close"].shift(1)).abs()
    low_close  = (d["low"]  - d["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    d["atr14"] = tr.ewm(com=13, adjust=False).mean()

    # ── OBV change (20-day rolling z-score) ──────────────────────────────────
    direction = np.sign(d["close"].diff()).fillna(0)
    obv = (direction * d["volume"]).cumsum()
    obv_diff = obv.diff()
    obv_mean = obv_diff.rolling(20).mean()
    obv_std  = obv_diff.rolling(20).std()
    d["obv_change"] = ((obv_diff - obv_mean) / (obv_std + 1e-9)).clip(-3, 3)

    # ── Candle body ratio ─────────────────────────────────────────────────────
    hl_range = (d["high"] - d["low"]).replace(0, np.nan)
    d["body_ratio"] = (d["close"] - d["open"]) / hl_range
    d["body_ratio"] = d["body_ratio"].fillna(0)

    # ── Day of week ───────────────────────────────────────────────────────────
    d["day_of_week"] = pd.to_datetime(d.index).dayofweek.astype(float)

    feature_cols = [
        "volume", "return", "volume_change", "rsi14", "bb_width",
        "macd", "signal", "ma_cross_strength", "bb_position", "macd_hist",
        "ma5_dist", "ma20_dist", "atr14", "obv_change", "body_ratio", "day_of_week",
    ]
    return d[feature_cols]


def build_window(features_df: pd.DataFrame, window_size: int = 30) -> np.ndarray:
    """Return the last `window_size` rows as a (1, window_size, n_features) array."""
    arr = features_df.values[-window_size:]
    if len(arr) < window_size:
        raise ValueError(f"Need at least {window_size} rows; got {len(arr)}")
    return arr.reshape(1, window_size, arr.shape[1]).astype(np.float32)
