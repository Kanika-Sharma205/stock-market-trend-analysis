# Prediction Pipeline — End to End

When you type a ticker and click **Predict**, here is exactly what happens across all 7 steps.

---

## Step 1 — Frontend HTTP Request

```
POST /api/predict/dl
Content-Type: application/json
{ "ticker": "AAPL" }
```

The frontend sends the ticker symbol to the FastAPI backend. The backend validates that the ticker is non-empty and proceeds.

---

## Step 2 — Live Data Fetch

```python
import yfinance as yf
raw = yf.download("AAPL", period="90d", auto_adjust=True)
# Returns ~63 trading days of OHLCV data
```

90 calendar days gives approximately 63 trading days — more than enough for:
- The 30-day window the model expects
- MACD EMA-26 lookback to stabilise (26 days)
- A few rows of NaN drop from rolling computations

---

## Step 3 — Feature Engineering (16 features)

The same pipeline used during training is re-run on live data:

| Feature | Computation |
|---|---|
| `return` | `(close − prev_close) / prev_close` |
| `volume_change` | `log(volume / prev_volume)`, clipped ±5 |
| `rsi14` | Wilder EMA of up/down moves, 14-period |
| `macd` | EMA(12) − EMA(26) |
| `signal` | EMA(9) of MACD |
| `macd_hist` | `macd − signal` |
| `ma5_dist` | `(close − MA5) / MA5` |
| `ma20_dist` | `(close − MA20) / MA20` |
| `ma_cross_strength` | `MA5 − MA20` |
| `bb_width` | `(upper − lower) / mid` |
| `bb_position` | `(close − lower) / (upper − lower)` |
| `atr14` | Wilder smoothed True Range, 14-period |
| `obv_change` | 20-day rolling z-score of OBV differences, clipped ±3 |
| `body_ratio` | `(close − open) / (high − low)` |
| `day_of_week` | 0=Monday … 4=Friday |
| `volume` | Raw volume |

After computation, the last 30 rows are extracted as a window:
```
window shape: (1, 30, 16)
```

---

## Step 4 — Scaling

```python
# scaler.pkl was fitted on 2010-2020 training data only — never on val or test
X_flat   = window.reshape(-1, 16)           # (30, 16)
X_scaled = scaler.transform(X_flat)         # MinMaxScaler
X_scaled = X_scaled.reshape(1, 30, 16)     # (1, 30, 16) for model
```

The `MinMaxScaler` was fitted exclusively on training data. This means live values outside the training range will be clipped at 0 or 1 — an accepted behaviour for features like volume that can occasionally spike beyond historical bounds.

---

## Step 5 — Model Inference

```python
prob = model.predict(X_scaled)[0][0]   # sigmoid output ∈ (0, 1)
```

The trained CNN + Causal LSTM + Bahdanau Attention model runs a single forward pass on the 30-day scaled sequence. The final sigmoid activation outputs P(UP) — the model's estimated probability that tomorrow's adjusted close will be higher than today's.

---

## Step 6 — Decision + Confidence

```python
threshold = 0.3247          # selected on validation set at max-F1; test set never touched

prediction = "UP" if prob >= threshold else "DOWN"

diff = abs(prob - 0.5)
if diff >= 0.20:
    confidence = "High"
elif diff >= 0.10:
    confidence = "Medium"
else:
    confidence = "Low"
```

The threshold is asymmetric (0.325, not 0.5) because the model's output distribution skews low — it tends to output lower probabilities even for genuine UP days, because both classes present similarly uncertain patterns during training.

The confidence tier is based on distance from the midpoint (0.5), not from the threshold:

| Tier | Condition | Historical Accuracy |
|---|---|---|
| Low | `\|p − 0.5\| < 0.10` | 52.4% |
| Medium | `0.10 ≤ \|p − 0.5\| < 0.20` | 57.6% |
| High | `\|p − 0.5\| ≥ 0.20` | **58.3%** |

---

## Step 7 — Response

```json
{
  "ticker":      "AAPL",
  "last_date":   "2026-05-01",
  "last_close":  280.14,
  "prediction":  "UP",
  "probability": 0.4591,
  "confidence":  "Low",
  "threshold":   0.3247
}
```

The frontend renders a probability bar with the threshold marker fixed at 32.47%, so you can visually gauge how close the prediction is to the decision boundary.

---

## Important Caveat

The model was trained and calibrated exclusively on AAPL data. Running it on other tickers uses the AAPL-fitted scaler and AAPL-trained weights. Results for non-AAPL symbols are directionally interesting as a demonstration of the pipeline, but should not be treated as reliable predictions — the scaler's range bounds and the learned temporal patterns are AAPL-specific.
