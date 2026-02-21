import type { Dataset } from '../../api/client'

interface Props {
  dataset: Dataset
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`
  const mins = Math.floor(seconds / 60)
  const secs = Math.round(seconds % 60)
  return `${mins}m ${secs}s`
}

export default function StatsCards({ dataset }: Props) {
  return (
    <div className="grid-4" style={{ marginBottom: '16px' }}>
      <div className="card stat-card">
        <div className="stat-value">{dataset.valid_count}</div>
        <div className="stat-label">Valid Pairs</div>
        {(dataset.invalid_count > 0 || dataset.healed_count > 0) && (
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
            {dataset.invalid_count} invalid &middot; {dataset.healed_count} healed
          </div>
        )}
      </div>

      <div className="card stat-card">
        <div className="stat-value">{dataset.total_tokens.toLocaleString()}</div>
        <div className="stat-label">Total Tokens</div>
        <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
          {dataset.prompt_tokens.toLocaleString()} prompt &middot; {dataset.completion_tokens.toLocaleString()} completion
        </div>
      </div>

      <div className="card stat-card">
        <div className="stat-value">
          {dataset.gpu_kwh > 0 ? `${dataset.gpu_kwh.toFixed(4)}` : '—'}
        </div>
        <div className="stat-label">GPU kWh</div>
      </div>

      <div className="card stat-card">
        <div className="stat-value">
          {dataset.duration_seconds > 0 ? formatDuration(dataset.duration_seconds) : '—'}
        </div>
        <div className="stat-label">Duration</div>
      </div>
    </div>
  )
}
