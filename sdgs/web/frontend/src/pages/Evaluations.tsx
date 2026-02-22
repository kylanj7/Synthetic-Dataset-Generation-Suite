import { useEffect, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { FlaskConical } from 'lucide-react'
import { useTrainingStore } from '../store/trainingStore'
import { startEvaluation } from '../api/client'

const statusClass: Record<string, string> = {
  pending: 'badge-pending',
  running: 'badge-running',
  completed: 'badge-completed',
  failed: 'badge-failed',
  cancelled: 'badge-cancelled',
}

function metricColor(value: number | null): string {
  if (value == null) return 'var(--text-muted)'
  if (value >= 0.8) return 'var(--accent-green)'
  if (value >= 0.6) return 'var(--accent-gold)'
  return 'var(--accent-pink)'
}

function fmt(value: number | null): string {
  if (value == null) return '—'
  return (value * 100).toFixed(1) + '%'
}

export default function Evaluations() {
  const { evaluations, evalsTotal, evalsPage, loading, fetchEvaluations } = useTrainingStore()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const trainingRunId = searchParams.get('training_run_id')

  // Quick-start evaluation form
  const [showStart, setShowStart] = useState(!!trainingRunId)
  const [modelPath, setModelPath] = useState('')
  const [judgeModel, setJudgeModel] = useState('gpt-oss:120b')
  const [maxSamples, setMaxSamples] = useState(50)
  const [startError, setStartError] = useState('')
  const [starting, setStarting] = useState(false)

  useEffect(() => {
    fetchEvaluations()
  }, [])

  const totalPages = Math.ceil(evalsTotal / 20)

  const handleStartEval = async () => {
    setStartError('')
    setStarting(true)
    try {
      const run = await startEvaluation({
        training_run_id: trainingRunId ? Number(trainingRunId) : undefined,
        model_path: modelPath.trim() || undefined,
        judge_model: judgeModel,
        max_samples: maxSamples,
      })
      navigate(`/evaluations/${run.id}`)
    } catch (e) {
      setStartError(e instanceof Error ? e.message : 'Failed to start evaluation')
      setStarting(false)
    }
  }

  return (
    <div>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1>Evaluation Runs</h1>
          <p>{evalsTotal} evaluation{evalsTotal !== 1 ? 's' : ''}</p>
        </div>
        <button className="btn btn-primary" onClick={() => setShowStart(!showStart)}>
          <FlaskConical size={16} />
          New Evaluation
        </button>
      </div>

      {/* Quick-start form */}
      {showStart && (
        <div className="card" style={{ marginBottom: '16px', padding: '16px' }}>
          <div style={{ fontSize: '14px', fontWeight: 500, marginBottom: '12px' }}>
            Start Evaluation
            {trainingRunId && (
              <span style={{ fontSize: '12px', color: 'var(--text-muted)', marginLeft: '8px' }}>
                (from training run #{trainingRunId})
              </span>
            )}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px', marginBottom: '12px' }}>
            {!trainingRunId && (
              <div>
                <label style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Model/Adapter Path</label>
                <input
                  type="text"
                  placeholder="/path/to/adapter"
                  value={modelPath}
                  onChange={(e) => setModelPath(e.target.value)}
                  disabled={starting}
                  style={{ fontSize: '14px' }}
                />
              </div>
            )}
            <div>
              <label style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Judge Model</label>
              <input
                type="text"
                value={judgeModel}
                onChange={(e) => setJudgeModel(e.target.value)}
                disabled={starting}
                style={{ fontSize: '14px' }}
              />
            </div>
            <div>
              <label style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Max Samples</label>
              <input
                type="number"
                value={maxSamples}
                onChange={(e) => setMaxSamples(Math.max(1, parseInt(e.target.value) || 1))}
                min={1}
                disabled={starting}
                style={{ fontSize: '14px' }}
              />
            </div>
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              className="btn btn-primary"
              onClick={handleStartEval}
              disabled={starting || (!trainingRunId && !modelPath.trim())}
            >
              {starting ? <span className="spinner" /> : 'Start Evaluation'}
            </button>
            <button
              className="btn"
              onClick={() => { setShowStart(false); setStartError('') }}
              disabled={starting}
            >
              Cancel
            </button>
          </div>
          {startError && (
            <div style={{
              marginTop: '8px',
              fontSize: '13px',
              color: 'var(--accent-pink)',
              background: 'rgba(255, 126, 179, 0.1)',
              padding: '6px 10px',
              borderRadius: 'var(--radius-sm)',
            }}>
              {startError}
            </div>
          )}
        </div>
      )}

      {loading && evaluations.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '40px' }}>
          <div className="spinner" />
        </div>
      ) : evaluations.length === 0 ? (
        <div className="empty-state">
          <FlaskConical size={48} style={{ marginBottom: '12px', opacity: 0.3 }} />
          <h3>No evaluations yet</h3>
          <p>Evaluate a trained model to see quality metrics.</p>
        </div>
      ) : (
        <>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {evaluations.map((ev) => (
              <div
                key={ev.id}
                className="card"
                onClick={() => navigate(`/evaluations/${ev.id}`)}
                style={{ cursor: 'pointer', padding: '16px' }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '6px' }}>
                      <span style={{ fontWeight: 600, fontSize: '15px' }}>{ev.run_name}</span>
                      <span className={`badge ${statusClass[ev.status] || 'badge-pending'}`}>
                        {ev.status}
                      </span>
                    </div>
                    <div style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '8px' }}>
                      {ev.judge_model} — {ev.max_samples} samples
                      {ev.model_path && <> — {ev.model_path}</>}
                    </div>
                    {ev.status === 'completed' && (
                      <div style={{ display: 'flex', gap: '16px', fontSize: '13px' }}>
                        <span>Accuracy: <strong style={{ color: metricColor(ev.factual_accuracy) }}>{fmt(ev.factual_accuracy)}</strong></span>
                        <span>Complete: <strong style={{ color: metricColor(ev.completeness) }}>{fmt(ev.completeness)}</strong></span>
                        <span>Precision: <strong style={{ color: metricColor(ev.technical_precision) }}>{fmt(ev.technical_precision)}</strong></span>
                        <span>Overall: <strong style={{ color: metricColor(ev.overall_accuracy) }}>{fmt(ev.overall_accuracy)}</strong></span>
                        <span>Purity: <strong style={{ color: metricColor(ev.purity) }}>{fmt(ev.purity)}</strong></span>
                      </div>
                    )}
                  </div>
                  <div style={{ fontSize: '12px', color: 'var(--text-muted)', textAlign: 'right', flexShrink: 0 }}>
                    {ev.created_at && new Date(ev.created_at).toLocaleDateString()}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {totalPages > 1 && (
            <div className="pagination">
              <button
                className="btn"
                disabled={evalsPage <= 1}
                onClick={() => fetchEvaluations(evalsPage - 1)}
              >
                Previous
              </button>
              <span>Page {evalsPage} of {totalPages}</span>
              <button
                className="btn"
                disabled={evalsPage >= totalPages}
                onClick={() => fetchEvaluations(evalsPage + 1)}
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </div>
  )
}
