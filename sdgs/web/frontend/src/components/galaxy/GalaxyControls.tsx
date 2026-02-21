import { Search, Eye, EyeOff } from 'lucide-react'

interface Props {
  showQA: boolean
  searchQuery: string
  onToggleQA: () => void
  onSearch: (q: string) => void
}

export default function GalaxyControls({ showQA, searchQuery, onToggleQA, onSearch }: Props) {
  return (
    <div className="card" style={{ padding: '12px', width: '280px' }}>
      {/* Search */}
      <div style={{ position: 'relative', marginBottom: '8px' }}>
        <input
          value={searchQuery}
          onChange={e => onSearch(e.target.value)}
          placeholder="Search nodes..."
          style={{ paddingLeft: '32px', fontSize: '13px' }}
        />
        <Search size={14} style={{
          position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)',
          color: 'var(--text-muted)',
        }} />
      </div>

      {/* QA toggle */}
      <button className="btn" onClick={onToggleQA} style={{ width: '100%', justifyContent: 'center', fontSize: '13px' }}>
        {showQA ? <Eye size={14} /> : <EyeOff size={14} />}
        {showQA ? 'Hide Q&A Nodes' : 'Show Q&A Nodes'}
      </button>
    </div>
  )
}
