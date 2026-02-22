import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { ChevronDown, ChevronRight } from 'lucide-react'
import { useTrainingStore } from '../store/trainingStore'

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

export default function EvaluationDetail() {
  const { id } = useParams<{ id: string }>()
  const evalId = parseInt(id || '0')
  const { currentEval, loading, fetchEvaluation } = useTrainingStore()
  const [expandedSample, setExpandedSample] = useState<number | null>(null)
  const [showArticles, setShowArticles] = useState(false)

  useEffect(() => {
    fetchEvaluation(evalId)
  }, [evalId])

  if (loading && !currentEval) {
    return <div style={{ textAlign: 'center', padding: '40px' }}><div className="spinner" /></div>
  }

  if (!currentEval) {
    return <div className="empty-state"><h3>Evaluation not found</h3></div>
  }

  const metrics = [
    { label: 'Factual Accuracy', value: currentEval.factual_accuracy },
    { label: 'Completeness', value: currentEval.completeness },
    { label: 'Technical Precision', value: currentEval.technical_precision },
    { label: 'Overall Accuracy', value: currentEval.overall_accuracy },
    { label: 'Purity', value: currentEval.purity },
    { label: 'Entropy', value: currentEval.entropy },
  ]

  const samples = currentEval.per_sample_results as Record<string, unknown>[]
  const articles = currentEval.articles_log as Record<string, unknown>[]

  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <h1>{currentEval.run_name}</h1>
          <span className={`badge ${statusClass[currentEval.status] || 'badge-pending'}`}>
            {currentEval.status}
          </span>
        </div>
        <p>
          {currentEval.judge_model}
          {currentEval.created_at && ` — ${new Date(currentEval.created_at).toLocaleString()}`}
        </p>
      </div>

      {/* Error */}
      {currentEval.error_message && (
        <div className="card" style={{
          marginBottom: '16px',
          background: 'rgba(255, 126, 179, 0.05)',
          borderColor: 'rgba(255, 126, 179, 0.2)',
        }}>
          <h3 style={{ fontSize: '14px', fontWeight: 500, color: 'var(--accent-pink)', marginBottom: '8px' }}>Error</h3>
          <pre style={{ fontSize: '13px', color: 'var(--text-secondary)', whiteSpace: 'pre-wrap' }}>
            {currentEval.error_message}
          </pre>
        </div>
      )}

      {/* Metrics dashboard */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gap: '12px',
        marginBottom: '16px',
      }}>
        {metrics.map((m) => (
          <div key={m.label} className="card" style={{ padding: '16px', textAlign: 'center' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '6px' }}>{m.label}</div>
            <div style={{
              fontSize: '24px',
              fontWeight: 700,
              color: m.label === 'Entropy' ? 'var(--text-primary)' : metricColor(m.value),
            }}>
              {m.label === 'Entropy'
                ? (m.value != null ? m.value.toFixed(3) : '—')
                : fmt(m.value)
              }
            </div>
          </div>
        ))}
      </div>

      {/* Counts */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gap: '12px',
        marginBottom: '16px',
      }}>
        <div className="card" style={{ padding: '12px 16px' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>Scored</div>
          <div style={{ fontSize: '15px', fontWeight: 600, color: 'var(--accent-green)' }}>{currentEval.samples_scored}</div>
        </div>
        <div className="card" style={{ padding: '12px 16px' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>Skipped</div>
          <div style={{ fontSize: '15px', fontWeight: 600, color: 'var(--accent-gold)' }}>{currentEval.samples_skipped}</div>
        </div>
        <div className="card" style={{ padding: '12px 16px' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>Failed</div>
          <div style={{ fontSize: '15px', fontWeight: 600, color: 'var(--accent-pink)' }}>{currentEval.samples_failed}</div>
        </div>
      </div>

      {/* Per-sample results */}
      {samples && samples.length > 0 && (
        <div className="card" style={{ marginBottom: '16px' }}>
          <h3 style={{ fontSize: '16px', fontWeight: 500, marginBottom: '12px' }}>
            Per-Sample Results ({samples.length})
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {samples.map((sample, idx) => {
              const isExpanded = expandedSample === idx
              const question = String(sample.question || sample.instruction || `Sample ${idx + 1}`)
              const answer = String(sample.answer || sample.output || '')
              const reference = String(sample.reference || '')
              const scores = (sample.scores || sample.metrics || {}) as Record<string, number>
              const justification = String(sample.justification || sample.reasoning || '')

              return (
                <div
                  key={idx}
                  style={{
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 'var(--radius-sm)',
                    padding: '12px 16px',
                    cursor: 'pointer',
                  }}
                  onClick={() => setExpandedSample(isExpanded ? null : idx)}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: 500, fontSize: '13px' }}>
                        {question.slice(0, 200)}{question.length > 200 ? '...' : ''}
                      </div>
                      {!isExpanded && Object.keys(scores).length > 0 && (
                        <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px', display: 'flex', gap: '12px' }}>
                          {Object.entries(scores).map(([k, v]) => (
                            <span key={k}>{k}: {typeof v === 'number' ? v.toFixed(2) : String(v)}</span>
                          ))}
                        </div>
                      )}
                    </div>
                    <span style={{ fontSize: '12px', color: 'var(--text-muted)', flexShrink: 0 }}>
                      #{idx + 1}
                    </span>
                  </div>

                  {isExpanded && (
                    <div style={{ marginTop: '12px', borderTop: '1px solid var(--border-subtle)', paddingTop: '12px' }}>
                      {answer && (
                        <div style={{ marginBottom: '8px' }}>
                          <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--accent-cyan)', marginBottom: '4px' }}>
                            ANSWER
                          </div>
                          <div style={{
                            fontSize: '13px',
                            color: 'var(--text-secondary)',
                            background: 'rgba(110, 231, 216, 0.05)',
                            padding: '8px 12px',
                            borderRadius: 'var(--radius-sm)',
                            whiteSpace: 'pre-wrap',
                            maxHeight: '200px',
                            overflow: 'auto',
                          }}>
                            {answer}
                          </div>
                        </div>
                      )}
                      {reference && (
                        <div style={{ marginBottom: '8px' }}>
                          <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--accent-purple)', marginBottom: '4px' }}>
                            REFERENCE
                          </div>
                          <div style={{
                            fontSize: '13px',
                            color: 'var(--text-secondary)',
                            background: 'rgba(192, 132, 252, 0.05)',
                            padding: '8px 12px',
                            borderRadius: 'var(--radius-sm)',
                            whiteSpace: 'pre-wrap',
                            maxHeight: '200px',
                            overflow: 'auto',
                          }}>
                            {reference}
                          </div>
                        </div>
                      )}
                      {Object.keys(scores).length > 0 && (
                        <div style={{ marginBottom: '8px' }}>
                          <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--accent-gold)', marginBottom: '4px' }}>
                            SCORES
                          </div>
                          <div style={{ display: 'flex', gap: '16px', fontSize: '13px', color: 'var(--text-secondary)' }}>
                            {Object.entries(scores).map(([k, v]) => (
                              <span key={k}>
                                {k}: <strong>{typeof v === 'number' ? v.toFixed(3) : String(v)}</strong>
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                      {justification && (
                        <div>
                          <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '4px' }}>
                            JUSTIFICATION
                          </div>
                          <div style={{
                            fontSize: '13px',
                            color: 'var(--text-secondary)',
                            background: 'var(--bg-tertiary)',
                            padding: '8px 12px',
                            borderRadius: 'var(--radius-sm)',
                            whiteSpace: 'pre-wrap',
                            maxHeight: '200px',
                            overflow: 'auto',
                          }}>
                            {justification}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Articles log */}
      {articles && articles.length > 0 && (
        <div className="card">
          <button
            onClick={() => setShowArticles(!showArticles)}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-secondary)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              fontSize: '14px',
              fontWeight: 500,
              padding: 0,
            }}
          >
            {showArticles ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
            Articles Log ({articles.length})
          </button>

          {showArticles && (
            <div style={{ marginTop: '12px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {articles.map((article, idx) => (
                <div
                  key={idx}
                  style={{
                    fontSize: '13px',
                    color: 'var(--text-secondary)',
                    padding: '8px 12px',
                    background: 'var(--bg-tertiary)',
                    borderRadius: 'var(--radius-sm)',
                  }}
                >
                  <pre style={{ whiteSpace: 'pre-wrap', margin: 0 }}>
                    {JSON.stringify(article, null, 2)}
                  </pre>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
