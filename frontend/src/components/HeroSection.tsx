import { TrendingUp, Brain, BarChart2 } from 'lucide-react'

export default function HeroSection() {
  const badges = [
    { icon: <Brain size={14} />,      label: 'CNN + LSTM + Attention' },
    { icon: <BarChart2 size={14} />,  label: 'Random Forest · XGBoost · LightGBM' },
    { icon: <TrendingUp size={14} />, label: 'AAPL · 2010 – 2025' },
  ]

  return (
    <section className="relative overflow-hidden pt-24 pb-20 px-4">
      {/* Background glow */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-[-10%] left-[20%] w-[600px] h-[600px] bg-sky-500/5 rounded-full blur-3xl" />
        <div className="absolute top-[10%] right-[10%] w-[400px] h-[400px] bg-indigo-500/5 rounded-full blur-3xl" />
      </div>

      <div className="relative max-w-4xl mx-auto text-center">
        <div className="flex flex-wrap justify-center gap-2 mb-8">
          {badges.map(b => (
            <span key={b.label} className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-slate-800/80 border border-slate-700 text-slate-300 text-xs font-medium">
              <span className="text-sky-400">{b.icon}</span>
              {b.label}
            </span>
          ))}
        </div>

        <h1 className="text-5xl sm:text-6xl font-extrabold tracking-tight mb-4">
          <span className="bg-gradient-to-r from-sky-400 to-indigo-400 bg-clip-text text-transparent">
            MarketPulse
          </span>
        </h1>
        <p className="text-xl text-slate-400 font-medium mb-6 tracking-wide">
          Stock Market Trend Analysis
        </p>

        <p className="text-lg text-slate-400 max-w-2xl mx-auto leading-relaxed mb-10">
          A rigorous study of next-day price direction prediction for AAPL using
          traditional machine-learning ensembles and a deep learning
          CNN + LSTM + Attention architecture —
          built with strict no-leakage temporal validation.
        </p>

        <div className="flex flex-wrap justify-center gap-4">
          <a href="#performance"
             className="px-7 py-3 rounded-xl bg-sky-500 hover:bg-sky-400 text-white font-semibold transition-all shadow-lg shadow-sky-500/20">
            View Results
          </a>
          <a href="#predict"
             className="px-7 py-3 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 font-semibold transition-all">
            Run a Prediction
          </a>
        </div>

        {/* Mini stat row */}
        <div className="mt-16 grid grid-cols-2 sm:grid-cols-4 gap-4 max-w-2xl mx-auto">
          {[
            { v: '3 800+', l: 'Trading Days' },
            { v: '16',     l: 'DL Features' },
            { v: '54.9%',  l: 'Best Accuracy' },
            { v: '0.550',  l: 'Val AUC-ROC' },
          ].map(s => (
            <div key={s.l} className="card text-center py-4">
              <div className="text-xl font-bold text-sky-400 font-mono">{s.v}</div>
              <div className="text-xs text-slate-500 uppercase tracking-wider mt-1">{s.l}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
