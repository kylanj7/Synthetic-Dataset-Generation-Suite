import { useEffect, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { StopCircle, FlaskConical } from 'lucide-react'
import { useTrainingStore } from '../store/trainingStore'
import { useTrainingSSE } from '../hooks/useTrainingSSE'
import { cancelTraining, getTrainingRun } from '../api/client'

const statusClass: Record<string, string> = {
  pending: 'badge-pending',
  running: 'badge-running',
  completed: 'badge-completed',
  failed: 'badge-failed',
  cancelled: 'badge-cancelled',
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  return `${h}h ${m}m`
}

export default function TrainingDetail() {
  const { id } = useParams<{ id: string }>()
  const runId = parseInt(id || '0')
  const navigate = useNavigate()
  const { currentRun, loading, fetchTrainingRun, updateTrainingRun } = useTrainingStore()
  const logViewerRef = useRef<HTMLDivElement>(null)

  const isRunning = currentRun?.status === 'pending' || currentRun?.status === 'running'
  const { logs, status: sseStatus, done } = useTrainingSSE(isRunning ? runId : null)

  useEffect(() => {
    fetchTrainingRun(runId)
  }, [runId])

  useEffect(() => {
    if (done) {
      getTrainingRun(runId).then(updateTrainingRun).catch(() => {})
    }
  }, [done, runId])

  useEffect(() => {
    if (logViewerRef.current) {
      logViewerRef.current.scrollTop = logViewerRef.current.scrollHeight
    }
  }, [logs])

  if (loading && !currentRun) {
    return <div style={{ textAlign: 'center', padding: '40px' }}><div className="spinner" /></div>
  }

  if (!currentRun) {
    return <div className="empty-state"><h3>Training run not found</h3></div>
  }

  const stats = [
    { label: 'Base Model', value: currentRun.base_model },
    { label: 'LoRA Rank', value: String(currentRun.lora_rank) },
    { label: 'Learning Rate', value: String(currentRun.learning_rate) },
    { label: 'Epochs', value: String(currentRun.num_epochs) },
    { label: 'Batch Size', value: String(currentRun.batch_size) },
    { label: 'Final Loss', value: currentRun.final_loss != null ? currentRun.final_loss.toFixed(4) : '—' },
    { label: 'Total Steps', value: currentRun.total_steps != null ? String(currentRun.total_steps) : '—' },
    { label: 'Runtime', value: currentRun.duration_seconds > 0 ? formatDuration(currentRun.duration_seconds) : '—' },
  ]

  return (
    <div>
      {/* Header */}
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <h1>{currentRun.run_name}</h1>
            <span className={`badge ${statusClass[currentRun.status] || 'badge-pending'}`}>
              {currentRun.status}
            </span>
          </div>
          <p>
            {currentRun.base_model}
            {currentRun.created_at && ` — ${new Date(currentRun.created_at).toLocaleString()}`}
          </p>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          {isRunning && (
            <button
              className="btn btn-danger"
              onClick={async () => {
                try {
                  await cancelTraining(runId)
                  fetchTrainingRun(runId)
                } catch { /* ignore */ }
              }}
            >
              <StopCircle size={16} />
              Cancel
            </button>
          )}
          {currentRun.status === 'completed' && (
            <button
              className="btn btn-primary"
              onClick={() => navigate(`/evaluations?training_run_id=${runId}`)}
            >
              <FlaskConical size={16} />
              Run Evaluation
            </button>
          )}
        </div>
      </div>

      {/* Stats cards */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: '12px',
        marginBottom: '16px',
      }}>
        {stats.map((s) => (
          <div key={s.label} className="card" style={{ padding: '12px 16px' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>{s.label}</div>
            <div style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-primary)' }}>{s.value}</div>
          </div>
        ))}
      </div>

      {/* Dataset info */}
      {currentRun.dataset_path && (
        <div className="card" style={{ marginBottom: '16px', padding: '12px 16px' }}>
          <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
            Dataset: {currentRun.dataset_path}
            {currentRun.train_samples > 0 && (
              <> — {currentRun.train_samples} train / {currentRun.val_samples} val / {currentRun.test_samples} test</>
            )}
          </span>
        </div>
      )}

      {/* Error */}
      {currentRun.error_message && (
        <div className="card" style={{
          marginBottom: '16px',
          background: 'rgba(255, 126, 179, 0.05)',
          borderColor: 'rgba(255, 126, 179, 0.2)',
        }}>
          <h3 style={{ fontSize: '14px', fontWeight: 500, color: 'var(--accent-pink)', marginBottom: '8px' }}>Error</h3>
          <pre style={{ fontSize: '13px', color: 'var(--text-secondary)', whiteSpace: 'pre-wrap' }}>
            {currentRun.error_message}
          </pre>
        </div>
      )}

      {/* Live SSE log */}
      {isRunning && (
        <div className="card" style={{ marginBottom: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <h3 style={{ fontSize: '14px', fontWeight: 500 }}>Live Progress</h3>
            {sseStatus && <span className={`badge badge-${sseStatus}`}>{sseStatus}</span>}
          </div>
          <div className="log-viewer" ref={logViewerRef} style={{ maxHeight: '400px' }}>
            {logs.map((line, i) => (
              <div key={i} className="log-line">{line}</div>
            ))}
            {logs.length === 0 && (
              <div style={{ color: 'var(--text-muted)' }}>Waiting for training to start...</div>
            )}
          </div>
        </div>
      )}

      {/* Adapter path */}
      {currentRun.adapter_path && (
        <div className="card" style={{ padding: '12px 16px' }}>
          <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
            Adapter saved to: <code style={{ color: 'var(--accent-cyan)' }}>{currentRun.adapter_path}</code>
          </span>
        </div>
      )}
    </div>
  )
}
