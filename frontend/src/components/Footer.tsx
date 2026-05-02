export default function Footer() {
  return (
    <footer className="border-t border-slate-800 py-10 px-4">
      <div className="max-w-6xl mx-auto">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <img src="/logo.svg" alt="Logo" className="h-8 w-auto opacity-80" />
          </div>
          <div className="text-center text-sm text-slate-500">
            <p className="font-semibold text-slate-400">MarketPulse</p>
            <p>Stock Market Trend Analysis — CNN + LSTM + Attention</p>
            <p className="mt-1 text-xs">
              Built with TensorFlow · FastAPI · React · Recharts
            </p>
          </div>
          <div className="text-xs text-slate-600 text-right">
            <p>AAPL · 2010–2025</p>
            <p>Research Project</p>
          </div>
        </div>
      </div>
    </footer>
  )
}
