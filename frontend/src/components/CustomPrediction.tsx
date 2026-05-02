import { useState } from 'react'
import axios from 'axios'
import { TrendingUp, TrendingDown, Minus, Loader2, AlertCircle, Info } from 'lucide-react'

interface PredResult {
  ticker: string
  last_date: string
  last_close: number
  prediction: 'UP' | 'DOWN'
  probability: number
  confidence: 'Low' | 'Medium' | 'High'
  threshold: number
}

const confColor = {
  Low:    'text-amber-400 bg-amber-500/10 border-amber-500/20',
  Medium: 'text-sky-400   bg-sky-500/10   border-sky-500/20',
  High:   'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
}

const POPULAR = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']

export default function CustomPrediction() {
  const [ticker, setTicker] = useState('AAPL')
  const [result, setResult] = useState<PredResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const predict = async () => {
    if (!ticker.trim()) return
    setLoading(true); setResult(null); setError(null)
    try {
      const { data } = await axios.post<PredResult>('/api/predict/dl', { ticker: ticker.trim().toUpperCase() })
      setResult(data)
    } catch (e: unknown) {
      if (axios.isAxiosError(e)) {
        setError(e.response?.data?.detail ?? e.message)
      } else {
        setError('Unexpected error')
      }
    } finally {
      setLoading(false)
    }
  }

  const probPct = result ? Math.round(result.probability * 100) : 0
  const isUp    = result?.prediction === 'UP'

  return (
    <section id="predict" className="py-20 px-4 bg-slate-900/40">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-sky-500/10 border border-sky-500/20 text-sky-400 text-xs font-medium uppercase tracking-widest mb-4">
            Live Prediction
          </div>
          <h2 className="section-title">Custom Prediction</h2>
          <p className="section-sub max-w-xl mx-auto">
            Enter any ticker symbol. The model fetches the last 90 trading days of data,
            computes 16 technical features, and runs the trained CNN+LSTM+Attention model to predict
            next-day price direction.
          </p>
        </div>

        {/* Input card */}
        <div className="card mb-6">
          <label className="block text-sm font-medium text-slate-300 mb-2">Ticker Symbol</label>
          <div className="flex gap-3">
            <input
              type="text"
              value={ticker}
              onChange={e => setTicker(e.target.value.toUpperCase())}
              onKeyDown={e => e.key === 'Enter' && !loading && predict()}
              placeholder="e.g. AAPL"
              maxLength={10}
              className="flex-1 bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 text-white font-mono text-lg
                         focus:outline-none focus:border-sky-500 focus:ring-1 focus:ring-sky-500/40 transition-colors uppercase"
            />
            <button
              onClick={predict}
              disabled={loading || !ticker.trim()}
              className="px-6 py-3 rounded-xl bg-sky-500 hover:bg-sky-400 disabled:bg-slate-700 disabled:text-slate-500
                         text-white font-semibold transition-colors flex items-center gap-2"
            >
              {loading ? <><Loader2 size={16} className="animate-spin" /> Predicting…</> : 'Predict'}
            </button>
          </div>

          {/* Popular tickers */}
          <div className="flex flex-wrap gap-2 mt-4">
            <span className="text-xs text-slate-500 self-center">Quick:</span>
            {POPULAR.map(t => (
              <button key={t} onClick={() => setTicker(t)}
                className={`px-3 py-1 rounded-lg text-xs font-mono font-medium transition-colors border
                  ${ticker === t
                    ? 'bg-sky-500/20 border-sky-500/40 text-sky-400'
                    : 'bg-slate-800 border-slate-700 text-slate-400 hover:border-slate-500 hover:text-slate-200'}`}>
                {t}
              </button>
            ))}
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="flex items-start gap-3 card border-red-500/30 bg-red-500/5 mb-6">
            <AlertCircle className="text-red-400 shrink-0 mt-0.5" size={16} />
            <p className="text-sm text-red-300">{error}</p>
          </div>
        )}

        {/* Result card */}
        {result && (
          <div className={`card border-2 ${isUp ? 'border-emerald-500/40' : 'border-red-500/40'}`}>
            <div className="flex items-start justify-between mb-6">
              <div>
                <div className="text-3xl font-extrabold font-mono text-white">{result.ticker}</div>
                <div className="text-sm text-slate-400 mt-1">
                  Last close <span className="text-white font-mono">${result.last_close}</span>
                  <span className="mx-2 text-slate-600">·</span>
                  <span className="font-mono">{result.last_date}</span>
                </div>
              </div>
              <div className={`flex flex-col items-center gap-1 px-5 py-3 rounded-2xl border ${isUp ? 'bg-emerald-500/10 border-emerald-500/30' : 'bg-red-500/10 border-red-500/30'}`}>
                {isUp
                  ? <TrendingUp size={28} className="text-emerald-400" />
                  : <TrendingDown size={28} className="text-red-400" />}
                <span className={`text-xl font-extrabold ${isUp ? 'text-emerald-400' : 'text-red-400'}`}>
                  {result.prediction}
                </span>
              </div>
            </div>

            {/* Probability bar */}
            <div className="mb-5">
              <div className="flex justify-between text-xs text-slate-400 mb-1.5">
                <span>DOWN ← Model Probability → UP</span>
                <span className="font-mono">{result.probability.toFixed(4)}</span>
              </div>
              <div className="h-4 bg-slate-700 rounded-full overflow-hidden relative">
                {/* Threshold marker */}
                <div className="absolute top-0 bottom-0 w-0.5 bg-amber-400/60 z-10"
                     style={{ left: `${result.threshold * 100}%` }} />
                {/* Fill */}
                <div
                  className={`h-full rounded-full transition-all ${isUp ? 'bg-gradient-to-r from-sky-500 to-emerald-500' : 'bg-gradient-to-r from-red-500 to-orange-500'}`}
                  style={{ width: `${probPct}%` }}
                />
              </div>
              <div className="flex justify-between text-xs mt-1">
                <span className="text-slate-600">0%</span>
                <span className="text-amber-400/70" style={{ marginLeft: `${result.threshold * 100 - 10}%` }}>
                  threshold ({(result.threshold * 100).toFixed(1)}%)
                </span>
                <span className="text-slate-600">100%</span>
              </div>
            </div>

            {/* Confidence badge */}
            <div className="flex items-center gap-3">
              <span className="text-sm text-slate-400">Confidence:</span>
              <span className={`px-3 py-1 rounded-full text-sm font-semibold border ${confColor[result.confidence]}`}>
                {result.confidence}
              </span>
              {result.confidence === 'High' && (
                <span className="text-xs text-emerald-400">58.3% historical accuracy at this tier</span>
              )}
              {result.confidence === 'Medium' && (
                <span className="text-xs text-sky-400">57.6% historical accuracy at this tier</span>
              )}
              {result.confidence === 'Low' && (
                <span className="text-xs text-amber-400">52.4% historical accuracy at this tier</span>
              )}
            </div>
          </div>
        )}

        {/* Disclaimer */}
        <div className="mt-6 flex items-start gap-3 p-4 rounded-xl bg-slate-800/40 border border-slate-700/40">
          <Info size={16} className="text-slate-500 shrink-0 mt-0.5" />
          <p className="text-xs text-slate-500 leading-relaxed">
            <strong className="text-slate-400">Research use only.</strong> This model is trained on AAPL technical indicators
            and has ~55% test accuracy at its best threshold. Past performance does not guarantee future results.
            The model was not trained to extrapolate to other tickers — results for non-AAPL symbols may be unreliable.
            This is not financial advice.
          </p>
        </div>
      </div>
    </section>
  )
}
