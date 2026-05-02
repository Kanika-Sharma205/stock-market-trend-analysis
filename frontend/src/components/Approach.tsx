import { Database, GitBranch, Cpu, FlaskConical } from 'lucide-react'

const pipeline = [
  {
    step: '01',
    icon: <Database size={20} />,
    title: 'Feature Extraction',
    color: 'sky',
    items: [
      'Raw OHLCV data for AAPL (2010–2025, ~3,800 trading days)',
      'MA5, MA20, MACD, Signal, RSI-14, Bollinger Bands',
      'Volume & Return signals',
      '19 base features exported to Final_Extracted_Features.csv',
    ],
  },
  {
    step: '02',
    icon: <GitBranch size={20} />,
    title: 'ML Pipeline (Chapter 1)',
    color: 'violet',
    items: [
      '32 engineered features: ATR, ROC, OBV, Stochastic, Lag features',
      'RobustScaler (COVID-outlier resistant)',
      'Temporal 80/20 split, TimeSeriesSplit(gap=5) CV',
      'RandomizedSearchCV: Random Forest, XGBoost, LightGBM',
      'Walk-forward validation + trading simulation',
    ],
  },
  {
    step: '03',
    icon: <Cpu size={20} />,
    title: 'DL Pipeline (Chapter 2)',
    color: 'emerald',
    items: [
      '16 scale-free features: MA distances, BB position, MACD hist, ATR, OBV z-score, body ratio',
      'MinMaxScaler fitted on training set only (no leakage)',
      '30-day sliding windows → (2700, 30, 16) training tensor',
      'CNN(32→64) → LSTM(128) → LSTM(64) → Bahdanau Attention(32) → Dense head',
      'AdamW + ReduceLROnPlateau + EarlyStopping(patience=25)',
    ],
  },
  {
    step: '04',
    icon: <FlaskConical size={20} />,
    title: 'Evaluation & Insights',
    color: 'amber',
    items: [
      'Threshold optimised on validation set (max-F1) → 0.325',
      'Test accuracy 54.9% vs 49.8% at default threshold',
      'High-confidence predictions: 58.3% accuracy',
      'Gradient saliency confirms MACD indicators as top features',
      'All models converge to ~0.50 AUC — confirms EMH for technicals',
    ],
  },
]

const colorMap: Record<string, string> = {
  sky:    'bg-sky-500/10 border-sky-500/30 text-sky-400',
  violet: 'bg-violet-500/10 border-violet-500/30 text-violet-400',
  emerald:'bg-emerald-500/10 border-emerald-500/30 text-emerald-400',
  amber:  'bg-amber-500/10 border-amber-500/30 text-amber-400',
}
const dotMap: Record<string, string> = {
  sky:    'bg-sky-400',
  violet: 'bg-violet-400',
  emerald:'bg-emerald-400',
  amber:  'bg-amber-400',
}

export default function Approach() {
  return (
    <section id="approach" className="py-20 px-4 bg-slate-900/40">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-14">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-violet-500/10 border border-violet-500/20 text-violet-400 text-xs font-medium uppercase tracking-widest mb-4">
            Methodology
          </div>
          <h2 className="section-title">Our Approach</h2>
          <p className="section-sub max-w-2xl mx-auto">
            A four-chapter pipeline from raw market data to explainable predictions,
            with rigorous anti-leakage design at every step.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {pipeline.map(p => (
            <div key={p.step} className="card">
              <div className="flex items-start gap-4 mb-4">
                <div className={`w-10 h-10 rounded-xl flex items-center justify-center border ${colorMap[p.color]}`}>
                  {p.icon}
                </div>
                <div>
                  <div className="text-xs font-mono text-slate-500 mb-0.5">STEP {p.step}</div>
                  <h3 className="font-semibold text-white">{p.title}</h3>
                </div>
              </div>
              <ul className="space-y-2">
                {p.items.map(item => (
                  <li key={item} className="flex items-start gap-2 text-sm text-slate-400">
                    <span className={`w-1.5 h-1.5 rounded-full mt-1.5 shrink-0 ${dotMap[p.color]}`} />
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Architecture diagram-like visual */}
        <div className="mt-10 card text-center">
          <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-widest mb-4">DL Architecture</h3>
          <div className="flex flex-wrap justify-center items-center gap-2 text-sm">
            {[
              { label: 'Input\n(30×16)', bg: 'bg-slate-700' },
              { label: 'Conv1D\n32→64', bg: 'bg-sky-500/20 text-sky-300' },
              { label: 'BatchNorm\n+MaxPool', bg: 'bg-sky-500/10 text-sky-400' },
              { label: 'LSTM\n128 units', bg: 'bg-violet-500/20 text-violet-300' },
              { label: 'LSTM\n64 units', bg: 'bg-violet-500/10 text-violet-400' },
              { label: 'Attention\n32 units', bg: 'bg-indigo-500/20 text-indigo-300' },
              { label: 'Dense\n32→16', bg: 'bg-emerald-500/10 text-emerald-400' },
              { label: 'Sigmoid\nOutput', bg: 'bg-emerald-500/20 text-emerald-300' },
            ].map((b, i, arr) => (
              <div key={b.label} className="flex items-center gap-2">
                <div className={`px-3 py-2 rounded-lg ${b.bg} font-mono text-xs whitespace-pre-line leading-tight border border-slate-700/40`}>
                  {b.label}
                </div>
                {i < arr.length - 1 && <span className="text-slate-600">→</span>}
              </div>
            ))}
          </div>
          <p className="text-xs text-slate-500 mt-4">161,121 parameters · 40 epochs trained · best checkpoint at epoch 15</p>
        </div>
      </div>
    </section>
  )
}
