import { useEffect, useState, useRef } from 'react'
import { useParams } from 'react-router-dom'
import { ChevronDown, ChevronRight, Wand2 } from 'lucide-react'
import { useTrainingStore } from '../store/trainingStore'
import { startCorrection } from '../api/client'

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

  // Correction agent state
  const [showCorrection, setShowCorrection] = useState(false)
  const [correctionThreshold, setCorrectionThreshold] = useState(50)
  const [correctionModel, setCorrectionModel] = useState('claude-opus-4-20250916')
  const [correctionSubmitting, setCorrectionSubmitting] = useState(false)
  const [correctionStarted, setCorrectionStarted] = useState(false)
  const [correctionError, setCorrectionError] = useState('')
  const [showCorrectionResults, setShowCorrectionResults] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    fetchEvaluation(evalId)
  }, [evalId])

  // Poll for correction results
  useEffect(() => {
    if (correctionStarted && !currentEval?.correction_results) {
      pollRef.current = setInterval(() => {
        fetchEvaluation(evalId)
      }, 10000)
    }
    if (currentEval?.correction_results && pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
      setCorrectionStarted(false)
    }
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [correctionStarted, currentEval?.correction_results, evalId])

  const handleStartCorrection = async () => {
    setCorrectionError('')
    setCorrectionSubmitting(true)
    try {
      await startCorrection(evalId, {
        score_threshold: correctionThreshold,
        model: correctionModel,
      })
      setCorrectionStarted(true)
    } catch (e) {
      setCorrectionError(e instanceof Error ? e.message : 'Failed to start correction')
    } finally {
      setCorrectionSubmitting(false)
    }
  }

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

      {/* Correction agent */}
      {currentEval.status === 'completed' && !currentEval.correction_results && (
        <div className="card" style={{ marginBottom: '16px' }}>
          <button
            onClick={() => setShowCorrection(!showCorrection)}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-secondary)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '14px',
              fontWeight: 500,
              padding: 0,
            }}
          >
            {showCorrection ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
            <Wand2 size={16} />
            Run Correction Agent
            {correctionStarted && (
              <span className="badge badge-running" style={{ marginLeft: '8px' }}>correction running</span>
            )}
          </button>

          {showCorrection && !correctionStarted && (
            <div style={{ marginTop: '12px' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '12px' }}>
                <div>
                  <label>Score Threshold</label>
                  <input
                    type="number"
                    value={correctionThreshold}
                    onChange={(e) => setCorrectionThreshold(parseFloat(e.target.value) || 50)}
                    disabled={correctionSubmitting}
                    step={5}
                  />
                </div>
                <div>
                  <label>Model</label>
                  <select
                    value={correctionModel}
                    onChange={(e) => setCorrectionModel(e.target.value)}
                    disabled={correctionSubmitting}
                  >
                    <option value="claude-opus-4-20250916">Claude Opus 4</option>
                    <option value="claude-sonnet-4-20250514">Claude Sonnet 4</option>
                  </select>
                </div>
              </div>

              {correctionError && (
                <div style={{
                  background: 'rgba(255, 126, 179, 0.1)',
                  border: '1px solid rgba(255, 126, 179, 0.3)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '8px 12px',
                  color: 'var(--accent-pink)',
                  fontSize: '13px',
                  marginBottom: '12px',
                }}>
                  {correctionError}
                </div>
              )}

              <button
                className="btn btn-primary"
                onClick={handleStartCorrection}
                disabled={correctionSubmitting}
                style={{ fontSize: '13px' }}
              >
                {correctionSubmitting ? <span className="spinner" /> : 'Start Correction'}
              </button>
            </div>
          )}

          {showCorrection && correctionStarted && (
            <div style={{ marginTop: '12px', fontSize: '13px', color: 'var(--text-muted)' }}>
              Correction is running. Polling for results every 10 seconds...
            </div>
          )}
        </div>
      )}

      {/* Correction results */}
      {currentEval.correction_results && (
        <div className="card" style={{ marginBottom: '16px' }}>
          <button
            onClick={() => setShowCorrectionResults(!showCorrectionResults)}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-secondary)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '14px',
              fontWeight: 500,
              padding: 0,
            }}
          >
            {showCorrectionResults ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
            <Wand2 size={16} />
            Correction Results
          </button>

          {showCorrectionResults && (
            <div style={{ marginTop: '12px' }}>
              {/* Summary stats */}
              <div style={{ display: 'flex', gap: '16px', marginBottom: '12px' }}>
                {currentEval.correction_results.total_corrected != null && (
                  <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                    Corrected: <strong style={{ color: 'var(--accent-cyan)' }}>
                      {String(currentEval.correction_results.total_corrected)}
                    </strong>
                  </div>
                )}
                {currentEval.correction_results.total_appended != null && (
                  <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                    Appended: <strong style={{ color: 'var(--accent-green)' }}>
                      {String(currentEval.correction_results.total_appended)}
                    </strong>
                  </div>
                )}
              </div>
              <pre style={{
                fontSize: '12px',
                color: 'var(--text-secondary)',
                background: 'var(--bg-tertiary)',
                padding: '12px',
                borderRadius: 'var(--radius-sm)',
                whiteSpace: 'pre-wrap',
                maxHeight: '400px',
                overflow: 'auto',
                margin: 0,
              }}>
                {JSON.stringify(currentEval.correction_results, null, 2)}
              </pre>
            </div>
          )}
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
