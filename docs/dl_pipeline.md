# Phase 3 — Deep Learning Pipeline & Architecture

## Why Move to Deep Learning?

The ML models treat each day's features as an **independent tabular snapshot**. They do not model how patterns evolve over time — the sequential relationship between day 1 and day 30 is invisible to a Random Forest.

Real market behaviour has temporal structure:
- An RSI reading of 72 means something very different if it has been rising for 20 days vs. falling from 85
- MACD histogram divergence only makes sense as a sequence, not as a single number
- Volatility regimes (high/low ATR periods) set the context for all other signals

**A model that can learn from 30-day sequences has access to information that tabular models inherently cannot see.**

---

## Why CNN + LSTM + Attention?

| Component | Why |
|---|---|
| **CNN (Conv1D)** | Extracts local temporal patterns — short-term price momentum, recent candlestick shapes. Acts as a feature extractor before the LSTM. |
| **Causal LSTM (×2)** | Processes the sequence strictly forward in time — no lookahead into future timesteps. Two stacked layers (128 → 64 units) allow the second layer to model higher-order temporal abstractions. We chose unidirectional (causal) LSTM over Bidirectional to respect the causal constraint: in a real trading system, you cannot look at day 30 to interpret day 10. |
| **Bahdanau Attention** | Not all 30 days are equally informative. Attention learns to weight high-volatility or trend-change days more heavily than quiet drift days — letting the model ask "which past moments matter most for today's prediction?" |

---

## DL Feature Engineering (Chapter 2A)

For the DL model we **redesigned the feature set** from scratch. The 32 ML features contained absolute price values (`ma5`, `ma20`, `close`) that cause *era-recognition bias* — a model can learn "close > $200 → post-2020 → apply post-COVID momentum rules", which is not a genuine directional signal.

We replaced everything with **scale-free, price-level-agnostic features:**

| Raw Feature | DL Replacement | Why |
|---|---|---|
| `ma5` | `ma5_dist = (close - ma5) / ma5` | Ratio is scale-free; $5 gap on a $10 stock vs. $200 stock are very different |
| `ma20` | `ma20_dist` | Same |
| `bb_upper`, `bb_lower`, `bb_mid` | `bb_position = (close - lower) / (upper - lower)` | 0–1 position within the band is universal |
| `ma_cross` (binary flag) | `ma_cross_strength = ma5 - ma20` | Continuous gap gives gradient signal; binary is too coarse |
| `obv` (raw cumulative) | `obv_change` (20-day rolling z-score, clipped ±3) | Raw OBV drifts upward forever; z-score normalises across any time window |
| `close`, `open`, `high`, `low` | — | Dropped entirely — absolute values encode era, not pattern |

**Final 16 DL features:**
`volume`, `return`, `volume_change`, `rsi14`, `bb_width`, `macd`, `signal`, `ma_cross_strength`, `bb_position`, `macd_hist`, `ma5_dist`, `ma20_dist`, `atr14`, `obv_change`, `body_ratio`, `day_of_week`

**Windowing:** 30-day sliding window. Gives enough history for MACD (EMA-26) to stabilise and for momentum patterns to emerge.

```
Input shape:  (batch_size, 30, 16)
              └── 30 trading days of 16 features each
```

**Scaling:** `MinMaxScaler` fitted **only on the training set** and applied to val/test. Fitting on the full dataset would leak future distribution information into training.

---

## Model Architecture

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

---

## Training Setup

| Setting | Value | Rationale |
|---|---|---|
| Optimizer | AdamW (lr=1e-4, decay=1e-4) | Weight decay improves generalisation on noisy data |
| Gradient clipping | clipnorm=1.0 | Prevents exploding gradients in LSTM |
| Loss | BinaryCrossentropy | Standard for binary classification |
| Epochs | 150 max | Early stopped at epoch 15 |
| Early stopping patience | 25 | Generous — noisy val loss can dip for many epochs |
| LR scheduler | ReduceLROnPlateau (patience=8, factor=0.5) | Halves LR on plateau |
| Class weights | UP: 0.888, DOWN: 1.0 | Corrects for 54/46 class imbalance |

The model converged at epoch 15 out of 150. With patience=25, it had ample room to continue — it simply did not. This is consistent with the near-zero feature-to-target correlations established in Chapter 1A: the architecture is not the bottleneck; the information content of 16 technical indicators predicting next-day direction does not support deeper learning.

---

## Custom Keras Layer

The `TemporalAttention` layer is a subclassed `tf.keras.layers.Layer`. When loading a saved `.keras` model in a fresh Python process (e.g. FastAPI), Keras raises:

```
TypeError: Could not locate class 'TemporalAttention'.
```

Fix: re-register with `@tf.keras.utils.register_keras_serializable()` before every `load_model()` call. The class definition must exactly match the training definition — same weight names and `get_config()` output.

```python
@tf.keras.utils.register_keras_serializable()
class TemporalAttention(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
    def build(self, input_shape):
        self.W = self.add_weight("attn_W", shape=(input_shape[-1], self.units), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight("attn_b", shape=(self.units,), initializer="zeros", trainable=True)
        self.v = self.add_weight("attn_v", shape=(self.units,), initializer="glorot_uniform", trainable=True)
    def call(self, inputs):
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.v, axes=[[-1], [0]]), axis=1)
        return tf.reduce_sum(inputs * tf.expand_dims(attention_weights, -1), axis=1)
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config
```
