# Challenges & How We Solved Them

Six concrete problems encountered during development, with root causes and exact fixes.

---

## Challenge 1 — Data Leakage from Rolling Features

**Problem:**
Features like `bb_width` (20-day Bollinger Band width) and `return_5d` (5-day rolling return) are computed using a rolling window over the full dataset *before* splitting. If we then use a random `train_test_split`, future rows bleed into the training set through these look-back windows.

Example: row 1000 in the test set contains `bb_width` computed from rows 981–1000. If row 990 is in the training set (random split), the training set has implicitly "seen" part of what will become a test row's features.

The visible symptom: AUC-ROC jumps from 0.500 (correct) to 0.524 (inflated) — a 0.024 gap that looks small but represents a systematic bias that makes the model appear better than it is.

**Solution:**
Use `TimeSeriesSplit(n_splits=5, gap=5)` for cross-validation. The `gap=5` parameter skips 5 rows between the training end and validation start of each fold. Since our longest rolling window is 5 days (`return_5d`), this gap guarantees no single row's rolling features can bridge the fold boundary.

```python
tscv = TimeSeriesSplit(n_splits=5, gap=5)
# gap = max rolling window length = 5 (return_5d, rsi14_lag1 etc.)
```

For the final train/test split, we use a hard temporal cut — all training data ends before all test data begins, with no overlap.

---

## Challenge 2 — Era-Recognition Bias in the DL Model

**Problem:**
If the feature set includes absolute price values — `ma5 = $143.21`, `close = $147.50`, `bb_mid = $145.00` — the model can learn to identify the *era* rather than the *pattern*. AAPL traded at $10–50 in 2010–2015 and at $150–200+ in 2020–2024. A model that learns "close > $150 → post-2020 → apply post-COVID momentum rules" is not learning a generalizable signal, it is memorising price history.

This inflates training accuracy (the model is very good at knowing what year it is) and collapses test performance (it encounters prices in a range with different characteristics than its era-matching learned).

**Solution:**
Remove all absolute price values from the DL feature set. Replace with scale-free, price-level-agnostic transforms:

| Removed | Replacement | Formula |
|---|---|---|
| `ma5` | `ma5_dist` | `(close − ma5) / ma5` |
| `ma20` | `ma20_dist` | `(close − ma20) / ma20` |
| `bb_upper`, `bb_lower`, `bb_mid` | `bb_position` | `(close − lower) / (upper − lower)` |
| `ma_cross` (binary) | `ma_cross_strength` | `ma5 − ma20` (continuous gap) |
| `close`, `open`, `high`, `low` | — | Dropped entirely |

After this change, the model cannot infer what year it is from the feature values alone — the same RSI reading of 72 and `ma5_dist` of +0.015 looks identical whether it's 2013 or 2023.

---

## Challenge 3 — Non-Stationarity Across 15 Years

**Problem:**
AAPL's trading volume grew dramatically over our 15-year dataset — from ~50M shares/day in 2010 to regularly exceeding 200M shares/day in 2022–2024. A `MinMaxScaler` fitted on 2010–2020 training data sets its `max` bound based on the highest training volume. Test values that exceed this bound are clipped to 1.0, losing all resolution above that threshold — effectively making 2022–2024 high-volume days indistinguishable from each other.

Similarly, raw OBV (On-Balance Volume) is a cumulative sum that drifts indefinitely upward, rendering it effectively non-stationary over 15 years.

**Solution:**
Use distribution-normalising transforms that are robust to long-term drift:

- **`volume_change`** — log-transform: `log(volume_t / volume_{t-1})`, clipped ±5. Returns daily percentage change, which is stationary regardless of absolute volume level.
- **`obv_change`** — 20-day rolling z-score of OBV differences, clipped ±3. Converts the non-stationary cumulative OBV into a standardised short-term momentum signal.

These features maintain meaningful variance across all eras without requiring the scaler to have seen future distribution bounds during training.

---

## Challenge 4 — Custom Keras Layer Deserialisation

**Problem:**
The `TemporalAttention` layer is implemented as a custom `tf.keras.layers.Layer` subclass. When Keras saves the model to `.keras` format, it serialises the *class name* and *config* but not the class *definition*. Loading the model in a fresh Python process (e.g. the FastAPI backend) raises:

```
TypeError: Could not locate class 'TemporalAttention'.
Make sure custom classes are decorated with @keras.saving.register_keras_serializable()
```

**Solution:**
Re-register the class *before* calling `load_model()`, using the exact same decorator the training notebook used:

```python
@tf.keras.utils.register_keras_serializable()
class TemporalAttention(Layer):
    def __init__(self, units, **kwargs): ...
    def build(self, input_shape): ...
    def call(self, inputs): ...
    def get_config(self): ...
```

This is handled in `backend/app/services/dl_service._register_custom_layers()`, which is called at the top of `_load()` before every `load_model()` invocation. The class definition must exactly match the training definition — same weight names, shapes, and `get_config()` output — for the saved weights to align correctly.

---

## Challenge 5 — Class Imbalance (54% UP / 46% DOWN)

**Problem:**
The dataset has 54.2% UP days and 45.8% DOWN days. While not severe, a naive model that always predicts UP achieves 54% accuracy and 0.54 F1 for the UP class without learning anything useful. Training without correction pushes all models toward the majority class.

**Solution:**
Apply class weights to balance gradient contribution from both classes:

**ML models:**
```python
# Random Forest
RandomForestClassifier(class_weight='balanced')

# XGBoost
XGBClassifier(scale_pos_weight=0.846)   # n_negative / n_positive

# LightGBM
LGBMClassifier(is_unbalance=True)
```

**DL model:**
```python
# Compute inverse-frequency weights
n_up   = sum(y_train == 1)   # ~1,460
n_down = sum(y_train == 0)   # ~1,240
class_weight = {1: n_down / n_up, 0: 1.0}   # {1: 0.888, 0: 1.0}

model.fit(X_train, y_train, class_weight=class_weight, ...)
```

This ensures the model receives equal total gradient signal from UP and DOWN examples regardless of their relative frequency.

---

## Challenge 6 — OBV Collapsed to Near-Zero Variance After Scaling

**Problem:**
In early experiments we included raw `obv_change` computed as `obv.pct_change()` — the daily percentage change in OBV. After `MinMaxScaler` normalisation, the entire feature array compressed to the range `[0.001, 0.002]` — essentially a constant. The model received a near-zero gradient from this feature and effectively ignored it, despite OBV being a theoretically useful volume-momentum signal.

Root cause: OBV `pct_change()` values are tiny (daily OBV changes are a small fraction of cumulative OBV), and the MinMaxScaler maps this tiny range linearly to [0, 1], where due to floating-point precision the actual encoded values all round to the same thing.

**Solution:**
Replace raw OBV change with a **20-day rolling z-score** of OBV *differences* (not percentages), clipped to ±3:

```python
obv_diff  = obv.diff()
obv_mean  = obv_diff.rolling(20).mean()
obv_std   = obv_diff.rolling(20).std()
obv_change = ((obv_diff - obv_mean) / (obv_std + 1e-9)).clip(-3, 3)
```

This produces a feature with approximately unit variance, zero mean, bounded range ±3, and meaningful signal regardless of the absolute OBV level or time period. After scaling it occupies the full [0, 1] range and contributes 6.6% gradient importance in the final model.
