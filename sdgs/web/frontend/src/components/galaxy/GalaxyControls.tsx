import { Search } from 'lucide-react'

interface Props {
  searchQuery: string
  onSearch: (q: string) => void
}

export default function GalaxyControls({ searchQuery, onSearch }: Props) {
  return (
    <div className="card" style={{ padding: '12px', width: '280px' }}>
      <div style={{ position: 'relative' }}>
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
    </div>
  )
}
