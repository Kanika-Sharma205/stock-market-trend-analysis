import { useEffect, useState } from 'react'
import axios from 'axios'
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend, ReferenceLine,
} from 'recharts'
import { RefreshCw, AlertCircle } from 'lucide-react'

interface MLModel {
  name: string
  auc_roc: number
  accuracy: number
  up_precision: number; up_recall: number; up_f1: number
  down_precision: number; down_recall: number; down_f1: number
  feature_importance: { feature: string; importance: number }[]
}
interface DLMetrics {
  test_auc_roc: number; test_accuracy_default: number; test_accuracy_optimal: number
  test_f1_optimal: number; final_threshold: number; val_auc_roc: number; best_epoch: number
  top3_features: string[]
  architecture: {
    window_size: number; n_features: number; feature_cols: string[]
    train_samples: number; val_samples: number; test_samples: number
    train_range: string[]; val_range: string[]; test_range: string[]
  }
  training_history: Record<string, number>[]
  feature_importance: { feature: string; importance: number }[]
}

const fmt = (n: number) => (n * 100).toFixed(1) + '%'
const fmtRaw = (n: number) => n.toFixed(4)

function MetricCard({ value, label, sub }: { value: string; label: string; sub?: string }) {
  return (
    <div className="card text-center py-5">
      <div className="metric-value">{value}</div>
      <div className="metric-label">{label}</div>
      {sub && <div className="text-xs text-slate-500 mt-1">{sub}</div>}
    </div>
  )
}

function MLTab({ models }: { models: MLModel[] }) {
  const [active, setActive] = useState(0)
  const m = models[active]

  const classData = [
    { name: 'UP',   precision: m.up_precision,   recall: m.up_recall,   f1: m.up_f1 },
    { name: 'DOWN', precision: m.down_precision, recall: m.down_recall, f1: m.down_f1 },
  ]

  return (
    <div>
      {/* Model selector */}
      <div className="flex gap-2 mb-6">
        {models.map((mo, i) => (
          <button key={mo.name} onClick={() => setActive(i)}
            className={`tab-btn ${i === active ? 'tab-active' : 'tab-inactive'}`}>
            {mo.name}
          </button>
        ))}
      </div>

      {/* Metric strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
        <MetricCard value={fmtRaw(m.auc_roc)} label="AUC-ROC" />
        <MetricCard value={fmt(m.accuracy)}   label="Accuracy" />
        <MetricCard value={fmt(m.up_f1)}      label="UP F1" />
        <MetricCard value={fmt(m.down_f1)}    label="DOWN F1" />
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Class metrics bar chart */}
        <div className="card">
          <h4 className="text-sm font-semibold text-slate-300 mb-4">Precision / Recall / F1 by Class</h4>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={classData} barSize={20}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <YAxis domain={[0, 1]} tickFormatter={v => (v * 100).toFixed(0) + '%'} tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                formatter={(v: number) => fmt(v)}
              />
              <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
              <Bar dataKey="precision" fill="#38bdf8" name="Precision" radius={[3,3,0,0]} />
              <Bar dataKey="recall"    fill="#818cf8" name="Recall"    radius={[3,3,0,0]} />
              <Bar dataKey="f1"        fill="#34d399" name="F1"        radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Feature importance */}
        <div className="card">
          <h4 className="text-sm font-semibold text-slate-300 mb-4">Top 10 Feature Importances</h4>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={m.feature_importance.slice(0, 10)} layout="vertical" barSize={12}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis type="number" tickFormatter={v => (v * 100).toFixed(1) + '%'} tick={{ fill: '#94a3b8', fontSize: 10 }} />
              <YAxis type="category" dataKey="feature" width={90} tick={{ fill: '#94a3b8', fontSize: 10 }} />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                formatter={(v: number) => (v * 100).toFixed(2) + '%'}
              />
              <Bar dataKey="importance" fill="#38bdf8" radius={[0,3,3,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="mt-4 card border-amber-500/20 bg-amber-500/5">
        <p className="text-sm text-amber-200/80">
          <strong className="text-amber-400">Key Finding:</strong> All three ML models converge to near-random
          AUC-ROC (~0.48–0.49) on the test set. Walk-forward validation confirms mean AUC = 0.53 ± 0.08,
          indicating unstable and non-replicable patterns. This is expected — the Efficient Market Hypothesis
          predicts that publicly available technical signals cannot consistently predict AAPL direction.
        </p>
      </div>
    </div>
  )
}

function DLTab({ data }: { data: DLMetrics }) {
  const historyData = data.training_history.map(r => ({
    epoch: Math.round(r.epoch) + 1,
    train_auc: r.auc,
    val_auc:   r.val_auc,
    train_acc: r.accuracy,
    val_acc:   r.val_accuracy,
    train_loss: r.loss,
    val_loss:   r.val_loss,
  }))

  const [histKey, setHistKey] = useState<'auc' | 'acc' | 'loss'>('auc')

  const chartLines: Record<string, { train: string; val: string; label: string; format: (v: number) => string }> = {
    auc:  { train: 'train_auc',  val: 'val_auc',  label: 'AUC-ROC', format: v => v.toFixed(4) },
    acc:  { train: 'train_acc',  val: 'val_acc',  label: 'Accuracy', format: v => fmt(v) },
    loss: { train: 'train_loss', val: 'val_loss', label: 'Loss',     format: v => v.toFixed(4) },
  }
  const cl = chartLines[histKey]

  return (
    <div>
      {/* Metric strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
        <MetricCard value={fmtRaw(data.val_auc_roc)}           label="Val AUC-ROC"       sub={`Best epoch ${data.best_epoch}`} />
        <MetricCard value={fmtRaw(data.test_auc_roc)}          label="Test AUC-ROC"       />
        <MetricCard value={fmt(data.test_accuracy_optimal)}    label="Test Accuracy*"     sub="threshold-tuned" />
        <MetricCard value={fmtRaw(data.final_threshold)}       label="Optimal Threshold"  sub="max-F1 on val" />
      </div>

      <div className="grid md:grid-cols-2 gap-6 mb-6">
        {/* Training curves */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-sm font-semibold text-slate-300">Training Curves</h4>
            <div className="flex gap-1">
              {(['auc', 'acc', 'loss'] as const).map(k => (
                <button key={k} onClick={() => setHistKey(k)}
                  className={`tab-btn text-xs py-1 px-2 ${histKey === k ? 'tab-active' : 'tab-inactive'}`}>
                  {k.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={historyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fill: '#64748b', fontSize: 11 }}
                tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <YAxis tickFormatter={cl.format} tick={{ fill: '#94a3b8', fontSize: 10 }} width={48} />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                formatter={(v: number) => cl.format(v)}
              />
              <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
              <ReferenceLine x={data.best_epoch} stroke="#f59e0b" strokeDasharray="4 2"
                label={{ value: `Best (${data.best_epoch})`, fill: '#f59e0b', fontSize: 10 }} />
              <Line dataKey={cl.train} name={`Train ${cl.label}`} stroke="#38bdf8" dot={false} strokeWidth={2} />
              <Line dataKey={cl.val}   name={`Val ${cl.label}`}   stroke="#818cf8" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Feature importance */}
        <div className="card">
          <h4 className="text-sm font-semibold text-slate-300 mb-4">Gradient Saliency — Feature Importance</h4>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={data.feature_importance.slice(0, 10)} layout="vertical" barSize={12}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis type="number" tickFormatter={v => (v * 100).toFixed(1) + '%'} tick={{ fill: '#94a3b8', fontSize: 10 }} />
              <YAxis type="category" dataKey="feature" width={90} tick={{ fill: '#94a3b8', fontSize: 10 }} />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                formatter={(v: number) => (v * 100).toFixed(2) + '%'}
              />
              <Bar dataKey="importance" fill="#818cf8" radius={[0,3,3,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Architecture breakdown */}
      <div className="card">
        <h4 className="text-sm font-semibold text-slate-300 mb-4">Data Split Summary</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead>
              <tr className="text-slate-500 text-xs uppercase tracking-wider border-b border-slate-700">
                <th className="pb-2 pr-4">Split</th>
                <th className="pb-2 pr-4">Date Range</th>
                <th className="pb-2 pr-4">Samples</th>
                <th className="pb-2">Purpose</th>
              </tr>
            </thead>
            <tbody className="text-slate-300 divide-y divide-slate-800">
              <tr>
                <td className="py-2 pr-4 text-sky-400 font-medium">Train</td>
                <td className="py-2 pr-4 font-mono text-xs">{data.architecture.train_range.join(' → ')}</td>
                <td className="py-2 pr-4">{data.architecture.train_samples.toLocaleString()}</td>
                <td className="py-2 text-slate-400 text-xs">Model learning + scaler fitting</td>
              </tr>
              <tr>
                <td className="py-2 pr-4 text-violet-400 font-medium">Validation</td>
                <td className="py-2 pr-4 font-mono text-xs">{data.architecture.val_range.join(' → ')}</td>
                <td className="py-2 pr-4">{data.architecture.val_samples.toLocaleString()}</td>
                <td className="py-2 text-slate-400 text-xs">Threshold tuning, early stopping</td>
              </tr>
              <tr>
                <td className="py-2 pr-4 text-emerald-400 font-medium">Test</td>
                <td className="py-2 pr-4 font-mono text-xs">{data.architecture.test_range.join(' → ')}</td>
                <td className="py-2 pr-4">{data.architecture.test_samples.toLocaleString()}</td>
                <td className="py-2 text-slate-400 text-xs">Final honest evaluation (never seen)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

export default function ModelPerformance() {
  const [tab, setTab] = useState<'ml' | 'dl'>('dl')
  const [mlData, setMlData] = useState<MLModel[] | null>(null)
  const [dlData, setDlData] = useState<DLMetrics | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchAll = async () => {
    setLoading(true); setError(null)
    try {
      const [mlRes, dlRes] = await Promise.all([
        axios.get('/api/metrics/ml'),
        axios.get('/api/metrics/dl'),
      ])
      setMlData(mlRes.data.models)
      setDlData(dlRes.data)
    } catch {
      setError('Could not connect to the backend. Make sure it is running on port 8000.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchAll() }, [])

  return (
    <section id="performance" className="py-20 px-4">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-14">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium uppercase tracking-widest mb-4">
            Model Performance
          </div>
          <h2 className="section-title">Validation &amp; Test Results</h2>
          <p className="section-sub max-w-2xl mx-auto">
            Real out-of-sample metrics — no data leakage, strict temporal splits.
          </p>
        </div>

        {/* Tab switcher */}
        <div className="flex gap-3 mb-8">
          <button onClick={() => setTab('dl')} className={`tab-btn ${tab === 'dl' ? 'tab-active' : 'tab-inactive'}`}>
            Deep Learning (CNN+LSTM+Attention)
          </button>
          <button onClick={() => setTab('ml')} className={`tab-btn ${tab === 'ml' ? 'tab-active' : 'tab-inactive'}`}>
            ML Ensemble (RF / XGB / LGB)
          </button>
          <button onClick={fetchAll} className="ml-auto p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-700 transition-colors">
            <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>

        {error && (
          <div className="flex items-center gap-3 card border-red-500/30 bg-red-500/5 mb-6">
            <AlertCircle className="text-red-400 shrink-0" size={18} />
            <p className="text-sm text-red-300">{error}</p>
          </div>
        )}

        {loading ? (
          <div className="text-center py-20 text-slate-500">
            <RefreshCw size={32} className="animate-spin mx-auto mb-4" />
            Loading metrics from backend…
          </div>
        ) : tab === 'dl' && dlData ? (
          <DLTab data={dlData} />
        ) : tab === 'ml' && mlData ? (
          <MLTab models={mlData} />
        ) : null}
      </div>
    </section>
  )
}
