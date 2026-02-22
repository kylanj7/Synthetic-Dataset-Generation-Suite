import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Cpu, Plus } from 'lucide-react'
import { useTrainingStore } from '../store/trainingStore'

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

export default function Training() {
  const { trainingRuns, total, page, loading, fetchTrainingRuns } = useTrainingStore()
  const navigate = useNavigate()

  useEffect(() => {
    fetchTrainingRuns()
  }, [])

  const totalPages = Math.ceil(total / 20)

  return (
    <div>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1>Training Runs</h1>
          <p>{total} run{total !== 1 ? 's' : ''}</p>
        </div>
        <button className="btn btn-primary" onClick={() => navigate('/training/start')}>
          <Plus size={16} />
          Start Training
        </button>
      </div>

      {loading && trainingRuns.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '40px' }}>
          <div className="spinner" />
        </div>
      ) : trainingRuns.length === 0 ? (
        <div className="empty-state">
          <Cpu size={48} style={{ marginBottom: '12px', opacity: 0.3 }} />
          <h3>No training runs yet</h3>
          <p>Fine-tune a model on one of your datasets.</p>
          <button
            className="btn btn-primary"
            style={{ marginTop: '16px' }}
            onClick={() => navigate('/training/start')}
          >
            <Plus size={16} />
            Start Training
          </button>
        </div>
      ) : (
        <>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {trainingRuns.map((run) => (
              <div
                key={run.id}
                className="card"
                onClick={() => navigate(`/training/${run.id}`)}
                style={{ cursor: 'pointer', padding: '16px' }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '6px' }}>
                      <span style={{ fontWeight: 600, fontSize: '15px' }}>{run.run_name}</span>
                      <span className={`badge ${statusClass[run.status] || 'badge-pending'}`}>
                        {run.status}
                      </span>
                    </div>
                    <div style={{ fontSize: '13px', color: 'var(--text-secondary)', display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
                      <span>{run.base_model}</span>
                      <span>LoRA r={run.lora_rank}</span>
                      <span>lr={run.learning_rate}</span>
                      <span>{run.num_epochs} epoch{run.num_epochs !== 1 ? 's' : ''}</span>
                      {run.final_loss != null && <span>loss: {run.final_loss.toFixed(4)}</span>}
                      {run.duration_seconds > 0 && <span>{formatDuration(run.duration_seconds)}</span>}
                    </div>
                  </div>
                  <div style={{ fontSize: '12px', color: 'var(--text-muted)', textAlign: 'right', flexShrink: 0 }}>
                    {run.created_at && new Date(run.created_at).toLocaleDateString()}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {totalPages > 1 && (
            <div className="pagination">
              <button
                className="btn"
                disabled={page <= 1}
                onClick={() => fetchTrainingRuns(page - 1)}
              >
                Previous
              </button>
              <span>Page {page} of {totalPages}</span>
              <button
                className="btn"
                disabled={page >= totalPages}
                onClick={() => fetchTrainingRuns(page + 1)}
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
