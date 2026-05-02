# Phase 2 — ML Models (Chapter 1)

## Why ML First?

We started with ML ensemble models for several good reasons:

1. **Interpretability.** Tree-based models give us direct feature importance scores. If MACD is a useful predictor, RF will show it. This guides DL feature selection.
2. **Speed.** Training 50 RandomizedSearchCV iterations for RF/XGB/LGB takes minutes. This lets us rapidly test whether any signal exists before investing weeks into DL.
3. **Baseline.** Any DL model must beat these baselines to justify its complexity. If XGBoost already achieves 0.60 AUC, there is no point building a Transformer.
4. **Leakage detection.** Tree models are extremely sensitive to data leakage. If AUC is suspiciously high, it is a red flag before we even touch deep learning.

---

## Data Preparation (Chapter 1A)

We extended the 19 base features to **32 engineered features** including ATR, ROC-5, ROC-10, OBV, Stochastic-K, lag returns, calendar features, and ratio transforms. We applied `RobustScaler` (not StandardScaler) because COVID-crash outliers in 2020 returns would have distorted StandardScaler badly.

**Key engineering decisions:**

| Decision | Reason |
|---|---|
| `volume_change` log-transformed | Fixes heavy right skew in daily volume |
| Returns clipped at ±15% | Dampens crash outliers without row deletion |
| `atr14_pct = atr14 / adj_close` | Removes absolute price scale across 15 years |
| `obv_change` as 20-day z-score | Raw cumulative OBV is non-stationary |

---

## Training (Chapter 1B)

```python
# Strict temporal CV — NO random shuffle ever
tscv = TimeSeriesSplit(n_splits=5, gap=5)
# gap=5 prevents 5-day rolling features from leaking across folds

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

**Walk-forward validation** was also run to simulate real deployment — training on 3 years, predicting 1 quarter forward, then sliding the window. Walk-forward mean AUC = 0.534 ± 0.084 — the wide standard deviation confirms performance is unstable and not generalising across regimes.

---

## The Data Leakage Experiment

This was a critical sanity check we deliberately built:

| Validation Method | AUC-ROC |
|---|---|
| Random `train_test_split` (**WRONG**) | 0.524 |
| Temporal `TimeSeriesSplit(gap=5)` (**CORRECT**) | 0.500 |

The 0.024 inflation from random splitting comes from future data leaking into training through rolling-window features like `bb_width` and `return_5d`. Many published studies do not catch this. We did.

**How leakage happens:** row 1000 in the test set contains `bb_width` computed from rows 981–1000. If row 990 ends up in the training set under a random split, the training set has "seen" part of the test row's feature — the model gets credit for predicting future data it already partially saw.

---

## ML Results

| Model | AUC-ROC | Accuracy | UP F1 | DOWN F1 | Verdict |
|---|---|---|---|---|---|
| **Random Forest** | 0.4921 | 47% | 0.25 | 0.59 | Strong DOWN bias — predicts bear better than bull |
| **XGBoost** | 0.4833 | 50% | 0.54 | 0.46 | Random with slight UP lean |
| **LightGBM** | 0.4793 | 49% | 0.40 | 0.55 | Slightly better on DOWN class |

**Conclusion:** No ML model found a consistent, exploitable pattern. All three converge to random performance on the held-out test set. This is expected given the EMH, but it confirmed we needed a fundamentally different approach — **temporal sequence modelling** that can see how patterns evolve across 30 days rather than treating each day as an isolated snapshot.
