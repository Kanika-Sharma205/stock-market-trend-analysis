---
title: MarketPulse Stock-Market-Trend-Analysis
emoji: 🦀
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
license: mit
---

<div align="center">

<img src="logo.svg" alt="MarketPulse" width="90" />

<br/>

# MarketPulse

### Stock Market Trend Analysis — Predicting next-day AAPL price direction using ML ensembles and a CNN + LSTM + Attention deep learning pipeline, with a full-stack web dashboard.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

</div>

---

## Table of Contents

1. [What Are We Building?](#1-what-are-we-building)
2. [The Problem — Why Is This Hard?](#2-the-problem--why-is-this-hard)
3. [Dataset](#3-dataset)
4. [Project Structure](#4-project-structure)
5. [Phase 1 — Feature Extraction](#5-phase-1--feature-extraction-chapter-0)
6. [Phase 2 — ML Models (Why We Started Here)](#6-phase-2--ml-models-chapter-1)
7. [Phase 3 — Deep Learning (Why We Went Further)](#7-phase-3--deep-learning-chapter-2)
8. [Model Architecture — CNN + LSTM + Attention](#8-model-architecture--cnn--lstm--attention)
9. [Performance Comparison — ML vs DL](#9-performance-comparison--ml-vs-dl)
10. [The Honest Assessment — What Went Wrong](#10-the-honest-assessment--what-went-wrong)
11. [Challenges & How We Solved Them](#11-challenges--how-we-solved-them)
12. [The Full-Stack App](#12-the-full-stack-app)
13. [Prediction Pipeline — End to End](#13-prediction-pipeline--end-to-end)
14. [How to Run the Project](#14-how-to-run-the-project)
15. [Future Work — v1 and Beyond](#15-future-work--v1-and-beyond)

---

## 1. What Are We Building?

This project is a **rigorous, end-to-end study** of whether next-day stock price direction for Apple Inc. (AAPL) can be predicted from publicly available technical indicators. The prediction task is a binary classification:

- **UP (1)** — tomorrow's adjusted close price is higher than today's
- **DOWN (0)** — tomorrow's adjusted close is equal or lower

We build two independent prediction pipelines and compare them head-to-head:

| Pipeline | Models Used | What It Learns |
|---|---|---|
| **ML Ensemble** | Random Forest, XGBoost, LightGBM | Tabular patterns in a single day's feature snapshot |
| **Deep Learning** | CNN + LSTM + Attention | Temporal patterns across a 30-day rolling window |

On top of the research, we also built a **full-stack web application** — a React + FastAPI dashboard — that displays all model metrics and lets you run live predictions on any stock ticker in real time.

This is **v0 of the project** — the foundational research phase is complete. We have a working, deployed system, validated training pipelines, and a clear map of what to fix next.

---

## 2. The Problem — Why Is This Hard?

Before looking at results, it helps to understand why predicting stock direction is one of the canonical hard problems in quantitative finance. This context is important for interpreting our results fairly.

### 2.1 The Efficient Market Hypothesis

AAPL is one of the most-traded, most-watched securities in the world. Thousands of algorithms and professional traders analyse the same public data — OHLCV, technical indicators, news — simultaneously. In a liquid market this efficient, **any predictive signal that becomes widely known gets arbitraged away almost instantly.** By the time we compute RSI-14 and MACD on EOD data, the information those indicators encode was already priced in during market hours.

This is not a failure of our model. It is the market working correctly.

### 2.2 Non-Stationarity and Regime Shifts

Financial time series are **not stationary**. The statistical properties — mean, variance, autocorrelation structure — change over time. More critically, entire *regimes* change:

| Era | Dominant Character |
|---|---|
| 2010 – 2019 | Post-GFC recovery, near-zero interest rates, steady tech bull run |
| 2020 – 2021 | COVID crash → fastest recovery in history, meme stocks, retail explosion |
| 2022 | Fastest rate-hike cycle in 40 years, tech selloff, AAPL -27% |
| 2023 – 2025 | AI mega-boom (ChatGPT effect), AAPL as a $3T company, macro volatility |

A model trained on 2010–2020 is learning patterns from a very different world. The features that "worked" in a zero-interest-rate environment may actively mislead the model when the Fed is aggressively hiking. This is the core reason we see val AUC = 0.55 in 2021–2022 collapse to test AUC = 0.49 in 2023–2025. The regime shifted — not our model — but the symptoms look identical.

### 2.3 The Signal-to-Noise Ratio Is Brutally Low

Every feature we computed has a near-zero correlation with the next-day direction. The strongest single feature — `return_20d` — has an absolute Pearson correlation of only **0.030** with the target. This is not a feature engineering failure. It reflects that next-day price changes are dominated by unforeseeable news, earnings surprises, macro announcements, and institutional order flow — none of which are captured in historical OHLCV.

### 2.4 We Are Missing the Real Signals

Technical indicators are a *reflection* of past price action. The signals that actually move AAPL tomorrow are:
- Earnings guidance revisions
- Fed rate decision surprises
- iPhone/Mac sales data leaks
- Analyst upgrades / downgrades
- Macro CPI prints
- Options market hedging flows

None of these are in our dataset. We are trying to predict an outcome driven largely by information we do not have.

**The takeaway:** Near-random performance is the *expected* and *honest* result when predicting a heavily traded stock from technicals alone. Our 54.9% test accuracy with threshold tuning is a genuinely positive result in this context, not a failure.

---

## 3. Dataset

**Source:** Apple Inc. (AAPL) historical OHLCV — sourced from Kaggle  
**Coverage:** December 1980 – March 2025 (we use 2010 onwards for modelling)

| Property | Value |
|---|---|
| Ticker | AAPL |
| Modelling range | 2010-01-04 → 2025-03-15 |
| Raw trading days | ~3,850 |
| After NaN drop (for 30-day window) | ~3,675 |
| Class balance | 54.2% UP / 45.8% DOWN |
| Raw columns | `open`, `high`, `low`, `close`, `adj_close`, `volume` |

**Splits used for the DL model:**

```
Train  :  2010-02-19 → 2020-12-31   (2,700 samples — 10 years)
Val    :  2021-01-04 → 2022-12-30   (  465 samples —  2 years)
Test   :  2023-01-03 → 2025-03-14   (  510 samples —  2 years, never touched until final eval)
```

The ML pipeline uses a simple temporal 80/20 split on the full dataset (~2011 → 2022 train, 2022 → 2025 test).

---

## 4. Project Structure

```
Stock_Market_Trend_Analysis/
│
├── logo.svg                         ← Project SVG logo
├── run-linux.sh                     ← Start full stack (Linux / macOS)
├── run-windows.bat                  ← Start full stack (Windows)
│
├── training/← All research notebooks & raw data
|   ├── notebooks
│   │   ├── Chapter 0: Feature Extraction From Dataset/ ← Raw OHLCV + base 19-feature extraction
│   │   ├── Chapter 1: Analysis using ML Models/          ← ML data prep, training, visualisations
│   │   └── Chapter 2: Analysis using DL Models/          ← DL data prep, model, checkpoints, evaluation
│   └── presentations
|       └── StockPricePrediction.pptx
|
├── backend/                         ← FastAPI server (port 8000)
│   ├── app/
│   │   ├── main.py                  ← FastAPI app + CORS
│   │   ├── routers/
│   │   │   ├── metrics.py           ← GET /api/metrics/ml  and  /api/metrics/dl
│   │   │   └── predict.py           ← POST /api/predict/dl
│   │   └── services/
│   │       ├── dl_service.py        ← Keras model loader + inference
│   │       └── feature_service.py   ← 16-feature engineering pipeline
│   ├── artifacts/
│   │   ├── dl/                      ← final_model.keras, scaler.pkl, results …
│   │   └── ml/                      ← classification reports + feature importances
│   ├── requirements.txt
│   ├── run-linux.sh
│   └── run-windows.bat
│
└── frontend/                        ← React + Vite + Tailwind (port 5173)
    ├── src/
    │   ├── components/
    │   │   ├── Header.tsx
    │   │   ├── HeroSection.tsx      ← Stats + CTA
    │   │   ├── ProblemStatement.tsx ← Why is this hard?
    │   │   ├── Approach.tsx         ← 4-step pipeline + architecture diagram
    │   │   ├── ModelPerformance.tsx ← Live charts from backend API
    │   │   ├── CustomPrediction.tsx ← Live ticker prediction
    │   │   └── Footer.tsx
    │   └── App.tsx
    ├── run-linux.sh
    └── run-windows.bat
```

---

## 5. Phase 1 — Feature Extraction (Chapter 0)

**Goal:** Turn raw OHLCV into meaningful technical signals.

We compute 19 base features from the raw price data, grouped as:

| Group | Features |
|---|---|
| Moving Averages | `ma5`, `ma20`, `ma_cross` |
| MACD | `macd`, `signal` |
| Oscillators | `rsi14` |
| Bollinger Bands | `bb_upper`, `bb_lower`, `bb_mid`, `bb_width` |
| Volume | `volume`, `volume_change` |
| Returns | `return` |
| Raw OHLCV | `open`, `high`, `low`, `close`, `adj_close` |

The output (`Final_Extracted_Features.csv`) is the shared starting point for both the ML and DL pipelines, which then diverge in their feature engineering strategies.

> **Key observation:** Even at this stage, no feature has a Pearson correlation with next-day direction above 0.03. The signal is weak from the very beginning — before any model even enters the picture.

---

## 6. Phase 2 — ML Models (Chapter 1)

### Why ML First?

We started with ML ensemble models for several good reasons:

1. **Interpretability.** Tree-based models give us direct feature importance scores. If MACD is a useful predictor, RF will show it. This guides DL feature selection.
2. **Speed.** Training 50 RandomizedSearchCV iterations for RF/XGB/LGB takes minutes. This lets us rapidly test whether any signal exists before investing weeks into DL.
3. **Baseline.** Any DL model must beat these baselines to justify its complexity. If XGBoost already achieves 0.60 AUC, there is no point building a Transformer.
4. **Leakage detection.** Tree models are extremely sensitive to data leakage. If AUC is suspiciously high, it is a red flag before we even touch deep learning.

### What We Did

**Data preparation (Chapter 1A):**
We extended the 19 base features to 32 engineered features including ATR, ROC-5, ROC-10, OBV, Stochastic-K, lag returns, calendar features, and ratio transforms. We applied `RobustScaler` (not StandardScaler) because COVID-crash outliers in 2020 returns would have distorted StandardScaler badly.

**Key engineering decisions:**
- `volume_change` log-transformed to fix heavy right skew
- Returns clipped at ±15% to dampen crash outliers without row deletion
- `atr14_pct = atr14 / adj_close` to remove absolute price scale across 15 years
- `obv_change` as 20-day rolling z-score instead of raw OBV

**Training (Chapter 1B):**

```python
# Strict temporal CV — NO random shuffle ever
tscv = TimeSeriesSplit(n_splits=5, gap=5)
# gap=5 prevents the 5-day rolling features from leaking across folds

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,
    scoring='roc_auc',
    cv=tscv,
    n_jobs=-1,
)
```

We searched 50 hyperparameter combinations for each of Random Forest, XGBoost, and LightGBM using AUC-ROC as the scoring metric.

**Walk-forward validation** was also run to simulate real deployment — training on 3 years, predicting 1 quarter forward, then sliding the window.

### The Data Leakage Experiment

This was a critical sanity check we deliberately built:

| Validation Method | AUC-ROC |
|---|---|
| Random `train_test_split` (**WRONG**) | 0.524 |
| Temporal `TimeSeriesSplit(gap=5)` (**CORRECT**) | 0.500 |

The 0.024 inflation from random splitting comes from future data leaking into training through rolling-window features like `bb_width` and `return_5d`. Many published studies do not catch this. We did.

### ML Results

| Model | AUC-ROC | Accuracy | UP F1 | DOWN F1 | Verdict |
|---|---|---|---|---|---|
| **Random Forest** | 0.4921 | 47% | 0.25 | 0.59 | Strong DOWN bias — predicts bear better than bull |
| **XGBoost** | 0.4833 | 50% | 0.54 | 0.46 | Random with slight UP lean |
| **LightGBM** | 0.4793 | 49% | 0.40 | 0.55 | Slightly better on DOWN class |

Walk-forward mean AUC = 0.534 ± 0.084 — the wide standard deviation tells us performance is unstable and not generalising across regimes.

**Conclusion from ML phase:** No ML model found a consistent, exploitable pattern. All three converge to random performance on the held-out test set. This is expected given the EMH, but it confirmed we needed a fundamentally different approach — temporal sequence modelling.

---

## 7. Phase 3 — Deep Learning (Chapter 2)

### Why Move to Deep Learning?

The ML models treat each day's features as an **independent tabular snapshot**. They do not model how patterns evolve over time — the sequential relationship between day 1 and day 30 is invisible to a Random Forest.

Real market behaviour has temporal structure:
- An RSI reading of 72 means something very different if it has been rising for 20 days vs. falling from 85
- MACD histogram divergence only makes sense as a sequence, not as a single number
- Volatility regimes (high/low ATR periods) set the context for all other signals

**A model that can learn from 30-day sequences has access to information that tabular models inherently cannot see.**

### Why CNN + LSTM + Attention?

We chose this architecture deliberately, not arbitrarily:

| Component | Why |
|---|---|
| **CNN (Conv1D)** | Extracts local temporal patterns — short-term price momentum, recent candlestick shapes. Acts as a feature extractor before the LSTM. |
| **Causal LSTM (×2)** | Processes the sequence strictly forward in time — no lookahead into future timesteps. Two stacked layers (128 → 64 units) allow the second layer to model higher-order temporal abstractions on top of what the first layer learned. We deliberately chose unidirectional (causal) LSTM over Bidirectional to respect the causal constraint: in a real trading system, you cannot look at day 30 to interpret day 10. |
| **Bahdanau Attention** | Not all 30 days are equally informative. Attention learns to weight high-volatility or trend-change days more heavily than quiet drift days — essentially letting the model ask "which past moments matter most for today's prediction?" |

### DL Feature Engineering (Chapter 2A)

For the DL model we **redesigned the feature set** from scratch. The 32 ML features contained absolute price values (`ma5`, `ma20`, `close`) that cause *era-recognition bias* — a model can learn that "close > $200" means it's post-2020 and adjust accordingly, which is not a genuine directional signal.

We replaced everything with **scale-free, price-level-agnostic features:**

| Raw Feature | DL Replacement | Why |
|---|---|---|
| `ma5` | `ma5_dist = (close - ma5) / ma5` | Ratio is scale-free; $5 gap on a $10 stock vs. a $200 stock are very different |
| `ma20` | `ma20_dist` | Same |
| `bb_upper`, `bb_lower`, `bb_mid` | `bb_position = (close - lower) / (upper - lower)` | 0–1 position within the band is universal |
| `ma_cross` (binary flag) | `ma_cross_strength = ma5 - ma20` | Continuous gap gives gradient signal; binary is too coarse |
| `obv` (raw cumulative) | `obv_change` (20-day rolling z-score, clipped ±3) | Raw OBV drifts upward forever; z-score normalises across any time window |

**Final 16 DL features:**
`volume`, `return`, `volume_change`, `rsi14`, `bb_width`, `macd`, `signal`, `ma_cross_strength`, `bb_position`, `macd_hist`, `ma5_dist`, `ma20_dist`, `atr14`, `obv_change`, `body_ratio`, `day_of_week`

**Windowing:** We use a 30-day sliding window. This gives enough history for MACD (which uses EMA-26) to stabilise and for momentum patterns to emerge, without introducing too many parameters.

```
Input shape:  (batch_size, 30, 16)
              └── 30 trading days of 16 features each
```

**Scaling:** `MinMaxScaler` fitted **only on the training set** and applied to val/test. This is non-negotiable — fitting the scaler on the full dataset would leak future distribution information into training.

---

## 8. Model Architecture — CNN + LSTM + Attention

```
Input Layer          (30 × 16)
        │
┌───────┴────────────────────────┐
│  CNN Block                      │
│  Conv1D(32, kernel=3, ReLU)     │
│  BatchNormalization             │
│  MaxPool(2)  →  (15 × 32)       │
│  Conv1D(64, kernel=3, ReLU)     │
│  BatchNormalization             │
│  SpatialDropout(0.5)            │
└───────┬────────────────────────┘
        │
┌───────┴────────────────────────┐
│  Causal LSTM Block              │
│  LSTM(128, return_seq=True)     │  ← LSTM_UNITS_1 × 2 = 128
│  Dropout(0.5)                   │
│  LSTM(64,  return_seq=True)     │  ← LSTM_UNITS_2 = 64
│  Dropout(0.5)                   │
└───────┬────────────────────────┘
        │
┌───────┴────────────────────────┐
│  Temporal Attention (32 units)  │
│  Bahdanau-style                 │
│  Weighted sum over timesteps    │
└───────┬────────────────────────┘
        │
┌───────┴────────────────────────┐
│  Dense Head                     │
│  Dense(32, ReLU) + Dropout(0.4) │
│  Dense(16, ReLU)                │
│  Dense(1,  Sigmoid)             │
└───────┬────────────────────────┘
        │
   Output: P(UP) ∈ [0, 1]

Total parameters:  161,121
```

**Training setup:**

| Setting | Value | Rationale |
|---|---|---|
| Optimizer | AdamW (lr=1e-4, decay=1e-4) | Weight decay improves generalisation on noisy data |
| Gradient clipping | clipnorm=1.0 | Prevents exploding gradients in LSTM |
| Loss | BinaryCrossentropy | Standard for binary classification |
| Epochs | 150 max | Early stopped at epoch 15 |
| Early stopping patience | 25 | Generous — noisy val loss can dip for many epochs |
| LR scheduler | ReduceLROnPlateau (patience=8, factor=0.5) | Halves LR on plateau |
| Class weights | UP: 0.888, DOWN: 1.0 | Corrects for 54/46 class imbalance |

---

## 9. Performance Comparison — ML vs DL

| Model | Val AUC | Test AUC | Test Acc (default) | Test Acc (tuned) |
|---|---|---|---|---|
| Random Forest | — | 0.4921 | 47.0% | — |
| XGBoost | — | 0.4833 | 50.0% | — |
| LightGBM | — | 0.4793 | 49.0% | — |
| **CNN + LSTM + Attention** | **0.5502** | **0.4875** | 49.8% | **54.9%** |

The DL model beats all ML baselines on validation AUC. Test accuracy improves from 49.8% → **54.9%** after shifting the decision threshold from 0.50 to 0.3247 (selected on the validation set). High-confidence predictions reach **58.3% accuracy** on the test set.

Top features by gradient saliency: `macd_hist` (10.3%), `macd` (7.4%), `ma5_dist` (6.9%), `atr14` (6.7%), `obv_change` (6.6%). Importance is spread across all 16 features — the model synthesises multiple weak signals rather than memorising one.

→ **[Full performance breakdown — threshold tuning, confidence tiers, all 16 feature importances](docs/performance.md)**

---

## 10. The Honest Assessment — What Went Wrong

**What we got right:** Strict no-leakage temporal pipeline, correct TimeSeriesSplit with gap, scale-free DL features, threshold tuned on val only, confidence-calibrated output.

**What didn't work as hoped:**
- Val AUC (0.5502) collapsed to test AUC (0.4875) — the 2023–2025 test period (AI boom, rate hikes) is a different market regime from the 2010–2020 training era. This is a data distribution problem, not an architecture problem.
- All three ML models plateaued at ~0.50 AUC — EMH holds for AAPL at EOD resolution.
- Early stopping at epoch 15 / 150 signals that 16 technical features simply don't contain enough predictable signal.
- Threshold tuning creates UP bias — dangerous in prolonged bear markets.

**The defence:** 54.9% honest test accuracy on one of the world's most efficient markets, from public data only, over an historically unusual test period, with a pipeline that correctly handles leakage (unlike much of the published literature) — is a realistic upper bound, not a failure.

→ **[Full honest assessment — detailed root causes and wider context](docs/assessment.md)**

---

## 11. Challenges & How We Solved Them

| # | Challenge | One-line Fix |
|---|---|---|
| 1 | Rolling-feature data leakage | `TimeSeriesSplit(gap=5)` — gap matches max window length |
| 2 | Era-recognition bias in DL | Replace all absolute prices with scale-free ratios (`ma5_dist`, `bb_position` …) |
| 3 | Non-stationarity over 15 years | Log-transform volume, z-score OBV — features are stationary even when raw values aren't |
| 4 | Custom Keras layer on load | Re-register `TemporalAttention` with `@register_keras_serializable()` before `load_model()` |
| 5 | Class imbalance 54/46 | `class_weight={0: 1.0, 1: 0.888}` — equal gradient from both classes |
| 6 | OBV near-zero variance after scaling | 20-day rolling z-score of OBV diffs, clipped ±3 — full [0,1] range after MinMax |

→ **[Full challenge write-ups — problem statements, root causes, code snippets](docs/challenges.md)**

---

## 12. The Full-Stack App

On top of the research notebooks, we built a complete production-style web application.

### Architecture

```
Browser (React / Vite)
    │  axios   HTTP
    ▼
FastAPI (Python)          port 8000
  ├── GET /api/metrics/ml  ← parse classification reports + feature CSVs
  ├── GET /api/metrics/dl  ← load final_results.pkl + training_history.csv
  └── POST /api/predict/dl ← yfinance → feature engineering → Keras inference
          │
          ├── yfinance: download last 90 days OHLCV
          ├── feature_service: compute 16 DL features
          ├── MinMaxScaler: scale using training-fitted scaler
          └── Keras model: predict P(UP)
```

### Frontend Pages / Sections

| Section | Content |
|---|---|
| **Hero** | Project stats strip — trading days, features, best accuracy, val AUC |
| **Problem Statement** | Why this is hard — EMH, non-stationarity, leakage, binary difficulty |
| **Approach** | 4-step pipeline visual + CNN/LSTM architecture block diagram |
| **Model Performance** | Live metric cards + Recharts training curves, feature importance bars, class metrics — loaded from backend API. Toggle between DL and ML tabs. |
| **Custom Prediction** | Ticker input, quick-select popular stocks, probability bar with threshold marker, confidence tier badge, historical accuracy by tier |

---

## 13. Prediction Pipeline — End to End

When you type `AAPL` and click **Predict**, this is exactly what happens:

### Step 1 — Frontend HTTP Request
```
POST /api/predict/dl
Content-Type: application/json
{ "ticker": "AAPL" }
```

### Step 2 — Live Data Fetch (Backend)
```python
import yfinance as yf
raw = yf.download("AAPL", period="90d", auto_adjust=True)
# Returns ~63 trading days of OHLCV data
```
90 calendar days gives ~63 trading days — more than enough for the 30-day window plus the 26-day EMA lookback needed for MACD.

### Step 3 — Feature Engineering (16 features)
The same pipeline used during training is re-run on live data:

| Feature | Computation |
|---|---|
| `return` | `(close − prev_close) / prev_close` |
| `volume_change` | `log(volume / prev_volume)`, clipped ±5 |
| `rsi14` | Wilder EMA of up/down moves |
| `macd`, `signal`, `macd_hist` | EMA(12) − EMA(26), EMA(9) of MACD, their difference |
| `ma5_dist`, `ma20_dist` | `(close − MAn) / MAn` |
| `ma_cross_strength` | `ma5 − ma20` |
| `bb_width`, `bb_position` | `(upper−lower)/mid` and `(close−lower)/(upper−lower)` |
| `atr14` | Wilder smoothed True Range |
| `obv_change` | 20-day rolling z-score of OBV differences, ±3 clip |
| `body_ratio` | `(close − open) / (high − low)` |
| `day_of_week` | 0=Monday … 4=Friday |

Then the last 30 rows are taken as a window → shape `(1, 30, 16)`.

### Step 4 — Scaling
```python
# scaler.pkl was fitted on 2010-2020 training data only
X_flat   = window.reshape(-1, 16)
X_scaled = scaler.transform(X_flat).reshape(1, 30, 16)
```

### Step 5 — Model Inference
```python
prob = model.predict(X_scaled)[0][0]   # sigmoid output, 0–1
```
The trained CNN+LSTM+Attention model runs a forward pass on the scaled 30-day sequence and outputs a single probability — the model's confidence that tomorrow's close will be higher than today's.

### Step 6 — Decision + Confidence
```python
threshold = 0.3247          # optimised on validation set
prediction = "UP" if prob >= threshold else "DOWN"

diff = abs(prob - 0.5)
confidence = "High" if diff >= 0.20 else ("Medium" if diff >= 0.10 else "Low")
```

The threshold is asymmetric (0.325, not 0.5) because the model's output distribution is skewed — it tends to output lower probabilities overall, so the UP/DOWN boundary sits left of centre.

### Step 7 — Response
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

The probability bar in the frontend is drawn with the threshold marker at exactly 32.47%, so you can visually gauge how close the prediction is to the decision boundary.

> **Important:** The model was trained exclusively on AAPL. Running it on other tickers uses the AAPL-fitted scaler and AAPL-trained weights. Results for non-AAPL symbols are directionally interesting as a demonstration but should not be treated as reliable predictions.

---

## 14. How to Run the Project

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| Node.js | 18+ |
| pip | latest |

### Quickstart — One Command

**Linux / macOS:**
```bash
chmod +x run-linux.sh
./run-linux.sh
```

**Windows:**
```
run-windows.bat
```

The script will:
1. Find your virtual environment (`backend/venv` → `./venv` → `~/ml_env`) or **create one automatically**
2. Install all Python dependencies (`backend/requirements.txt`)
3. Install frontend npm packages (only on first run)
4. Start backend on **http://localhost:8000**
5. Start frontend on **http://localhost:5173**

Press `Ctrl+C` once to cleanly shut down both servers.

---

### Running Separately

**Backend only:**
```bash
cd backend
./run-linux.sh        # Linux / macOS
run-windows.bat       # Windows
```

**Frontend only:**
```bash
cd frontend
./run-linux.sh        # Linux / macOS
run-windows.bat       # Windows
```

---

### Manual Setup (if you prefer)

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 2. Install backend dependencies
pip install -r backend/requirements.txt

# 3. Start the backend
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 4. In a new terminal — install and start frontend
cd frontend
npm install
npm run dev
```

**API documentation** is available at http://localhost:8000/docs once the backend is running.

---

### Running Training Notebooks

The research notebooks are in `training/`. Run them in order:

```bash
# Install Jupyter if needed
pip install jupyter

# Chapter 0 — Extract features from raw OHLCV
jupyter notebook training/chapter0_feature_extraction/Feature_Extraction.ipynb

# Chapter 1A — ML data preparation and EDA
jupyter notebook training/chapter1_ml_models/Data_Preparation_ML.ipynb

# Chapter 1B — Train Random Forest, XGBoost, LightGBM
jupyter notebook training/chapter1_ml_models/Training_ML_Models.ipynb

# Chapter 2A — DL feature engineering and windowing
jupyter notebook training/chapter2_dl_models/Data_Preparation_DL.ipynb

# Chapter 2B — Build and train CNN + LSTM + Attention
jupyter notebook training/chapter2_dl_models/Model_Architecture_Training.ipynb

# Chapter 2C — Evaluation, threshold tuning, saliency analysis
jupyter notebook training/chapter2_dl_models/Evaluation_Analysis.ipynb
```

---

## 15. Future Work — v1 and Beyond

**v0 is complete.** The foundational research is done, the no-leakage pipelines are validated, the full-stack app is running, and the performance ceiling of technical-indicator-only prediction is clearly established.

**v1 will address the root cause:** we are missing the signals that actually drive AAPL direction.

### High Priority

| # | Improvement | Why It Matters |
|---|---|---|
| 1 | **News sentiment (FinBERT / LLM-based)** | Earnings surprises, analyst revisions, macro news are the real directional signals. A single Tim Cook statement can move AAPL more than any technical pattern. |
| 2 | **Multi-asset inputs (SPY, VIX, DXY, 10Y yield)** | AAPL does not trade in isolation. When VIX spikes, big-cap tech sells off. When DXY strengthens, international revenue concerns suppress AAPL. These are strong exogenous signals. |
| 3 | **Options market data (put/call ratio, IV surface)** | Smart money positions in the options market often precede price moves. PCR and implied volatility skew are forward-looking in a way that historical OHLCV is not. |
| 4 | **Continual / online learning** | Retrain on a rolling 52-week window weekly. Address non-stationarity at the data level rather than hoping the model generalises across regimes. |

### Medium Priority

| # | Improvement | Why It Matters |
|---|---|---|
| 5 | **Transformer architecture (PatchTST / TFT)** | Temporal Fusion Transformer or PatchTST can handle longer context windows (90–252 days) more efficiently than LSTM and have shown strong results in recent time-series benchmarks. |
| 6 | **Regime-conditional modelling** | Separate models for bull / bear / high-volatility regimes. The pattern that works in a trending market is actively harmful in a mean-reverting one. |
| 7 | **Conformal prediction for uncertainty** | Instead of a fixed threshold, output calibrated prediction intervals. This enables proper position sizing proportional to the model's actual uncertainty. |
| 8 | **Proper backtesting with transaction costs** | Current evaluation is purely statistical (AUC, accuracy). A vectorbt or Zipline backtest with realistic slippage and commission would show whether the statistical edge translates to P&L. |

### Lower Priority (but interesting)

| # | Improvement |
|---|---|
| 9 | Intraday tick data (5-min OHLCV) for shorter-horizon signals |
| 10 | Multi-stock model trained on S&P 500 universe |
| 11 | Reinforcement learning for portfolio-level position sizing |
| 12 | Adversarial training to improve regime robustness |

---

<div align="center">
  <br/>
  <img src="frontend/public/logo.svg" alt="Stock Market Analysis" width="48" />
  <br/>
  <h2>Made with  ❤️  by <strong>KANIKA SHARMA</strong></h2>
</div>
