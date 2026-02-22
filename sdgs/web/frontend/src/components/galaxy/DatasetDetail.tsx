import { X, Database } from 'lucide-react'

interface Props {
  node: any
  papers: any[]
  onPaperClick: (node: any) => void
  onClose: () => void
}

export default function DatasetDetailPanel({ node, papers, onClose, onPaperClick }: Props) {
  return (
    <div style={{
      position: 'absolute', top: 0, right: 0, bottom: 0, width: '420px',
      background: 'var(--bg-secondary)',
      borderLeft: '1px solid var(--border-subtle)',
      overflowY: 'auto',
      zIndex: 20,
      boxShadow: '-4px 0 20px rgba(0, 0, 0, 0.3)',
    }}>
      {/* Header */}
      <div style={{
        padding: '16px 20px',
        borderBottom: '1px solid var(--border-subtle)',
        display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
      }}>
        <div style={{ flex: 1 }}>
          <div style={{
            fontSize: '11px', color: 'var(--text-muted)', marginBottom: '6px',
            display: 'flex', alignItems: 'center', gap: '4px', textTransform: 'uppercase', letterSpacing: '0.5px',
          }}>
            <Database size={12} /> Dataset
          </div>
          <h2 style={{ fontSize: '16px', lineHeight: 1.4 }}>{node.label}</h2>
        </div>
        <button onClick={onClose} style={{
          background: 'none', border: 'none', cursor: 'pointer',
          color: 'var(--text-secondary)', padding: '4px',
        }}>
          <X size={18} />
        </button>
      </div>

      <div style={{ padding: '16px 20px' }}>
        {/* Stats */}
        {node.abstract && (
          <div style={{
            fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '16px',
            background: 'rgba(0, 0, 0, 0.2)', padding: '10px 12px',
            borderRadius: 'var(--radius-sm)',
          }}>
            {node.abstract}
          </div>
        )}

        {/* Papers list */}
        <div style={{
          fontSize: '12px', color: 'var(--text-muted)', marginBottom: '12px',
        }}>
          {papers.length} Papers
        </div>

        {papers.map((paper: any) => (
          <div
            key={paper.id}
            onClick={() => onPaperClick(paper)}
            className="card"
            style={{
              marginBottom: '8px', padding: '12px', cursor: 'pointer',
            }}
          >
            <div style={{ fontSize: '13px', fontWeight: 500, marginBottom: '4px' }}>
              {paper.label}
            </div>
            <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              {paper.year && `${paper.year} · `}
              {paper.citation_count || 0} citations
              {paper.qa_pair_count ? ` · ${paper.qa_pair_count} Q&A` : ''}
            </div>
          </div>
        ))}

        {papers.length === 0 && (
          <div style={{ fontSize: '13px', color: 'var(--text-muted)', fontStyle: 'italic' }}>
            No papers in this dataset
          </div>
        )}
      </div>
    </div>
  )
}
