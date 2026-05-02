# The Problem — Why Predicting Stock Direction Is Hard

## The Efficient Market Hypothesis

AAPL is one of the most-traded, most-watched securities in the world. Thousands of algorithms and professional traders analyse the same public data — OHLCV, technical indicators, news — simultaneously. In a liquid market this efficient, **any predictive signal that becomes widely known gets arbitraged away almost instantly.** By the time we compute RSI-14 and MACD on EOD data, the information those indicators encode was already priced in during market hours.

This is not a failure of our model. It is the market working correctly.

---

## Non-Stationarity and Regime Shifts

Financial time series are **not stationary**. The statistical properties — mean, variance, autocorrelation structure — change over time. More critically, entire *regimes* change:

| Era | Dominant Character |
|---|---|
| 2010 – 2019 | Post-GFC recovery, near-zero interest rates, steady tech bull run |
| 2020 – 2021 | COVID crash → fastest recovery in history, meme stocks, retail explosion |
| 2022 | Fastest rate-hike cycle in 40 years, tech selloff, AAPL -27% |
| 2023 – 2025 | AI mega-boom (ChatGPT effect), AAPL as a $3T company, macro volatility |

A model trained on 2010–2020 is learning patterns from a very different world. The features that "worked" in a zero-interest-rate environment may actively mislead the model when the Fed is aggressively hiking.

This is the core reason we see val AUC = 0.55 in 2021–2022 collapse to test AUC = 0.49 in 2023–2025. The regime shifted — not our model — but the symptoms look identical. The correct fix is continual retraining on a rolling window, not a better architecture.

---

## The Signal-to-Noise Ratio Is Brutally Low

Every feature we computed has a near-zero correlation with the next-day direction:

| Feature | Pearson |r| with target |
|---|---|
| `return_20d` | 0.030 (strongest) |
| `rsi14` | ~0.020 |
| `macd_hist` | ~0.018 |
| Most others | < 0.015 |

This is not a feature engineering failure. It reflects that next-day price changes are dominated by unforeseeable news, earnings surprises, macro announcements, and institutional order flow — none of which are captured in historical OHLCV.

---

## We Are Missing the Real Signals

Technical indicators are a *reflection* of past price action. The signals that actually move AAPL tomorrow are:

- Earnings guidance revisions
- Fed rate decision surprises
- iPhone/Mac sales data leaks
- Analyst upgrades / downgrades
- Macro CPI prints
- Options market hedging flows

None of these are in our dataset. We are trying to predict an outcome driven largely by information we do not have.

**The takeaway:** Near-random performance is the *expected* and *honest* result when predicting a heavily traded stock from technicals alone. Our 54.9% test accuracy with threshold tuning is a genuinely positive result in this context, not a failure. v1 will add the signals that actually matter.
