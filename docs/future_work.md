# Future Work — v1 and Beyond

**v0 is complete.** The foundational research is done, the no-leakage pipelines are validated, the full-stack app is running, and the performance ceiling of technical-indicator-only prediction is clearly established.

**v1 will address the root cause:** we are missing the signals that actually drive AAPL direction.

---

## High Priority

| # | Improvement | Why It Matters |
|---|---|---|
| 1 | **News sentiment (FinBERT / LLM-based)** | Earnings surprises, analyst revisions, macro news are the real directional signals. A single Tim Cook statement can move AAPL more than any technical pattern. |
| 2 | **Multi-asset inputs (SPY, VIX, DXY, 10Y yield)** | AAPL does not trade in isolation. When VIX spikes, big-cap tech sells off. When DXY strengthens, international revenue concerns suppress AAPL. These are strong exogenous signals. |
| 3 | **Options market data (put/call ratio, IV surface)** | Smart money positions in the options market often precede price moves. PCR and implied volatility skew are forward-looking in a way that historical OHLCV is not. |
| 4 | **Continual / online learning** | Retrain on a rolling 52-week window weekly. Address non-stationarity at the data level rather than hoping the model generalises across regimes. |

---

## Medium Priority

| # | Improvement | Why It Matters |
|---|---|---|
| 5 | **Transformer architecture (PatchTST / TFT)** | Temporal Fusion Transformer or PatchTST can handle longer context windows (90–252 days) more efficiently than LSTM and have shown strong results in recent time-series benchmarks. |
| 6 | **Regime-conditional modelling** | Separate models for bull / bear / high-volatility regimes. The pattern that works in a trending market is actively harmful in a mean-reverting one. |
| 7 | **Conformal prediction for uncertainty** | Instead of a fixed threshold, output calibrated prediction intervals. This enables proper position sizing proportional to the model's actual uncertainty. |
| 8 | **Proper backtesting with transaction costs** | Current evaluation is purely statistical (AUC, accuracy). A vectorbt or Zipline backtest with realistic slippage and commission would show whether the statistical edge translates to P&L. |

---

## Lower Priority (but interesting)

| # | Improvement |
|---|---|
| 9 | Intraday tick data (5-min OHLCV) for shorter-horizon signals |
| 10 | Multi-stock model trained on S&P 500 universe |
| 11 | Reinforcement learning for portfolio-level position sizing |
| 12 | Adversarial training to improve regime robustness |

---

## The Honest v1 Hypothesis

The regime generalisation problem (val AUC 0.5502 → test AUC 0.4875) cannot be solved by a better architecture — it is a **data distribution problem**. Items 1–4 above directly attack the root cause:

- Items 1–3 add signals with genuine informational advantage over public technicals
- Item 4 prevents the training distribution from drifting too far from the deployment distribution

Items 5–8 are architecture and evaluation improvements that will matter *after* the data problem is addressed. Building a better Transformer on the same 16 technical features will not meaningfully shift the ceiling.
