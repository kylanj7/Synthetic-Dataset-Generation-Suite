import { Database, Clock, Zap } from 'lucide-react'
import type { Dataset } from '../../api/client'

interface Props {
  dataset: Dataset
  onClick: () => void
}

const statusClass: Record<string, string> = {
  pending: 'badge-pending',
  running: 'badge-running',
  completed: 'badge-completed',
  failed: 'badge-failed',
  cancelled: 'badge-cancelled',
}

export default function DatasetCard({ dataset, onClick }: Props) {
  const date = dataset.created_at ? new Date(dataset.created_at).toLocaleDateString() : ''

  return (
    <div
      className="card"
      onClick={onClick}
      style={{ cursor: 'pointer', padding: '16px 20px' }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flex: 1 }}>
          <Database size={20} style={{ color: 'var(--accent-cyan)', flexShrink: 0 }} />
          <div>
            <div style={{ fontWeight: 500, marginBottom: '4px' }}>{dataset.topic}</div>
            <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              {dataset.provider || 'default'}{dataset.model ? ` / ${dataset.model}` : ''} &middot; {date}
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px', fontSize: '13px', color: 'var(--text-secondary)' }}>
            {dataset.actual_size > 0 && (
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <Database size={14} /> {dataset.actual_size} pairs
              </span>
            )}
            {dataset.total_tokens > 0 && (
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <Zap size={14} /> {dataset.total_tokens.toLocaleString()} tokens
              </span>
            )}
            {dataset.duration_seconds > 0 && (
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <Clock size={14} /> {Math.round(dataset.duration_seconds)}s
              </span>
            )}
          </div>
          <span className={`badge ${statusClass[dataset.status] || 'badge-pending'}`}>
            {dataset.status}
          </span>
        </div>
      </div>
    </div>
  )
}
