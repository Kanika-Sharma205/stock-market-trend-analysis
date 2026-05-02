import { AlertTriangle, Target, TrendingUp, ShieldCheck } from 'lucide-react'

const challenges = [
  {
    icon: <TrendingUp className="text-sky-400" size={22} />,
    title: 'Non-Stationary Markets',
    body: 'Stock prices are non-stationary time series — their statistical properties shift over time. A model trained on 2010–2020 bull-market dynamics faces a completely different regime in 2022–2025 (rate hikes, AI boom, macro volatility).',
  },
  {
    icon: <AlertTriangle className="text-yellow-400" size={22} />,
    title: 'Efficient Market Hypothesis',
    body: 'Apple stock is one of the most heavily traded securities globally. Public technical signals are priced in nearly instantly, making it extremely hard to extract consistent predictive power from price history alone.',
  },
  {
    icon: <ShieldCheck className="text-emerald-400" size={22} />,
    title: 'Data Leakage Trap',
    body: 'Most published stock-prediction studies report inflated accuracy because they use random train/test splits. Rolling-window features computed on the full dataset leak future information. We address this with strict temporal splits and TimeSeriesSplit(gap=5) cross-validation.',
  },
  {
    icon: <Target className="text-indigo-400" size={22} />,
    title: 'Binary Classification Difficulty',
    body: 'Predicting the sign of tomorrow\'s return (UP / DOWN) is fundamentally harder than predicting magnitude. Even a small systematic edge above 50% is valuable — 54.9% accuracy with a calibrated confidence filter is a meaningful result.',
  },
]

export default function ProblemStatement() {
  return (
    <section id="problem" className="py-20 px-4">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-14">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-sky-500/10 border border-sky-500/20 text-sky-400 text-xs font-medium uppercase tracking-widest mb-4">
            Problem Statement
          </div>
          <h2 className="section-title">Why Is This Hard?</h2>
          <p className="section-sub max-w-2xl mx-auto">
            Predicting next-day stock price direction is one of the canonical hard problems in quantitative finance.
            Here's what makes it challenging — and how we tackle each issue.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {challenges.map(c => (
            <div key={c.title} className="card flex gap-4">
              <div className="shrink-0 w-10 h-10 rounded-xl bg-slate-700/60 flex items-center justify-center mt-0.5">
                {c.icon}
              </div>
              <div>
                <h3 className="font-semibold text-white mb-2">{c.title}</h3>
                <p className="text-slate-400 text-sm leading-relaxed">{c.body}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Goal box */}
        <div className="mt-8 card border-sky-500/30 bg-sky-500/5 text-center">
          <h3 className="text-lg font-semibold text-white mb-2">Our Goal</h3>
          <p className="text-slate-300 max-w-3xl mx-auto text-sm leading-relaxed">
            Build a <strong className="text-sky-400">honest, leakage-free benchmark</strong> that answers:
            "How much directional signal does 15 years of AAPL technical indicators actually contain?"
            We train parallel ML and DL pipelines, measure their true out-of-sample performance,
            and provide a live prediction interface to test the DL model on any ticker in real time.
          </p>
        </div>
      </div>
    </section>
  )
}
