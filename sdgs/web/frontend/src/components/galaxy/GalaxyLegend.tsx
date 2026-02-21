import { ClusterInfo } from '../../api/client'

interface Props {
  clusters: ClusterInfo[]
  activeCluster: number | null
  onToggle: (id: number | null) => void
}

export default function GalaxyLegend({ clusters, activeCluster, onToggle }: Props) {
  return (
    <div className="card" style={{ padding: '12px', maxWidth: '280px' }}>
      <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '8px' }}>
        Clusters
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
        <button
          className="btn"
          onClick={() => onToggle(null)}
          style={{
            fontSize: '12px', padding: '4px 10px',
            background: activeCluster === null ? 'rgba(126, 184, 255, 0.15)' : undefined,
          }}
        >
          All
        </button>
        {clusters.map(c => (
          <button
            key={c.id}
            className="btn"
            onClick={() => onToggle(activeCluster === c.id ? null : c.id)}
            style={{
              fontSize: '12px', padding: '4px 10px',
              background: activeCluster === c.id ? `${c.color}22` : undefined,
              borderColor: activeCluster === c.id ? `${c.color}66` : undefined,
            }}
          >
            <span style={{
              display: 'inline-block', width: '8px', height: '8px',
              borderRadius: '50%', background: c.color, marginRight: '4px',
            }} />
            {c.label} ({c.paper_count})
          </button>
        ))}
      </div>
    </div>
  )
}
