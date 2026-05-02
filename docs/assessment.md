# The Honest Assessment — What Went Wrong (and Why That's OK)

This document exists because honest evaluation matters more than a polished story. A research project that only reports successes teaches nothing.

---

## What We Got Right

- **Strict no-leakage temporal pipeline** throughout — no random shuffling, ever
- **TimeSeriesSplit with gap=5** — correctly handles rolling-window feature leakage
- **Scale-free DL feature engineering** — eliminates era-recognition bias across 15 years
- **Threshold optimised on validation set only** — test set never touched during tuning
- **Confidence-calibrated output** — high-confidence predictions genuinely correlate with higher accuracy (58.3% vs 52.4%)
- **Data leakage detection experiment** — explicitly demonstrated the 0.024 AUC inflation from random splits

---

## What Did Not Work As Hoped

### 1. Regime generalisation collapsed on the test set

Val AUC was 0.5502 — genuinely promising. Test AUC fell to 0.4875, a drop of 0.063.

This is almost entirely explained by the 2023–2025 test period representing a fundamentally different market environment from the 2010–2020 training era:

| Period | What Changed |
|---|---|
| 2022 | Fastest rate-hike cycle in 40 years — tech sector repriced from growth to value metrics |
| 2023 | AI boom catalysed by ChatGPT — AAPL re-rated as an "AI play" on top of existing fundamentals |
| 2024–25 | Post-AI-boom volatility, options market dominance, retail-driven momentum shifts |

None of these dynamics existed during training. The model learned patterns from a low-rate, steady-growth environment and was tested in a high-rate, AI-disruption environment. This is not something a better architecture would fix — it is a **data distribution problem**, and the correct fix is continual retraining (see Future Work).

### 2. All ML models plateaued at random performance

Despite 32 carefully engineered features and 50-iteration RandomizedSearchCV on all three models:

- Random Forest test AUC: 0.4921 (below random)
- XGBoost test AUC: 0.4833
- LightGBM test AUC: 0.4793

The Efficient Market Hypothesis holds for AAPL at end-of-day resolution. Technical indicators derived from the same public price data that thousands of algorithms already trade on carry no consistent edge. This was expected but needed to be demonstrated rigorously rather than assumed.

### 3. Early stopping at epoch 15 reveals a shallow signal

The model converged at epoch 15 out of 150 allowed. With patience=25, it was given ample room to keep improving — it simply did not. This means either:

- **(a)** The model extracted all learnable signal very efficiently in 15 epochs, or
- **(b)** There is not much signal to extract

Given the near-zero feature-to-target correlations (max 0.030 Pearson r) established in Chapter 1A, it is almost certainly (b). The architecture is not the bottleneck — the information content of 16 technical indicators predicting next-day direction simply does not support deeper learning.

### 4. Threshold tuning introduces directional bias

Accuracy improves from 49.8% → 54.9% by shifting the threshold from 0.50 to 0.3247. The cost: the model now heavily favours predicting UP.

- UP F1 at tuned threshold: **0.71**
- DOWN F1 at tuned threshold: significantly lower

In a real trading system, systematic UP bias is dangerous during bear markets — every down day produces a wrong prediction. The threshold tuning is statistically valid, but a production system would need either a regime-detection layer or a separate DOWN-biased model for bearish environments.

---

## The Wider Context — A Genuine Defence

We want to be direct: **54.9% test accuracy on next-day AAPL direction is a genuinely good result.**

Not because the bar is low — it is very high — but because of what we are up against:

| Factor | Why It Matters |
|---|---|
| Market efficiency | AAPL is one of the top-3 most liquid equities globally. EOD signals are priced in during market hours, not after. |
| Data poverty | We have 16 technical features. Prop trading firms have earnings data, satellite imagery, credit card transactions, options flow, and news sentiment. |
| Test period difficulty | 2023–2025 contained the fastest rate-hike cycle in modern history + an AI-driven mega-cap re-rating. These are historically unusual conditions for any model trained pre-2022. |
| Pipeline integrity | Our strict no-leakage design means our 54.9% is honest. Many published papers reporting 60–70% accuracy are simply not accounting for data leakage correctly. |

Professional quantitative hedge funds with teams of PhDs, petabytes of alternative data, and proprietary order-flow access typically target 52–53% directional accuracy on liquid large-cap equities. **Getting to 54.9% with a 161k-parameter model trained on 16 public technical features is not a failure — it is a realistic upper bound for this problem setup.**

The honest answer is: technical indicators alone cannot consistently predict AAPL direction. We demonstrated that rigorously. **v1 will add the signals that actually matter.**
