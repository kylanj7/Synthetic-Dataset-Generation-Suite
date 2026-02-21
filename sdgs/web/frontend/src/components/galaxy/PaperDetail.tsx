import { X, ExternalLink, BookOpen } from 'lucide-react'
import { PaperDetail } from '../../api/client'

interface Props {
  paper: PaperDetail
  onClose: () => void
}

export default function PaperDetailPanel({ paper, onClose }: Props) {
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
          <h2 style={{ fontSize: '16px', lineHeight: 1.4, marginBottom: '4px' }}>
            {paper.title}
          </h2>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            {paper.year && `${paper.year} · `}
            {paper.citation_count} citations
          </div>
        </div>
        <button onClick={onClose} style={{
          background: 'none', border: 'none', cursor: 'pointer',
          color: 'var(--text-secondary)', padding: '4px',
        }}>
          <X size={18} />
        </button>
      </div>

      <div style={{ padding: '16px 20px' }}>
        {/* Authors */}
        {paper.authors.length > 0 && (
          <div style={{ marginBottom: '16px' }}>
            <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '4px' }}>Authors</div>
            <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
              {paper.authors.join(', ')}
            </div>
          </div>
        )}

        {/* URL */}
        {paper.url && (
          <a href={paper.url} target="_blank" rel="noopener noreferrer"
            style={{
              display: 'inline-flex', alignItems: 'center', gap: '4px',
              fontSize: '13px', marginBottom: '16px',
            }}>
            <ExternalLink size={14} /> View Paper
          </a>
        )}

        {/* Abstract */}
        {paper.abstract && (
          <div style={{ marginBottom: '20px' }}>
            <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '4px' }}>Abstract</div>
            <div style={{
              fontSize: '13px', lineHeight: 1.6, color: 'var(--text-secondary)',
              background: 'rgba(0, 0, 0, 0.2)', padding: '12px',
              borderRadius: 'var(--radius-sm)',
            }}>
              {paper.abstract}
            </div>
          </div>
        )}

        {/* Q&A Pairs */}
        <div>
          <div style={{
            display: 'flex', alignItems: 'center', gap: '6px',
            fontSize: '12px', color: 'var(--text-muted)', marginBottom: '12px',
          }}>
            <BookOpen size={14} />
            {paper.qa_pairs.length} Q&A Pairs
          </div>

          {paper.qa_pairs.map((qa, idx) => (
            <div key={idx} className="card" style={{ marginBottom: '8px', padding: '12px' }}>
              <div style={{ fontSize: '13px', fontWeight: 500, marginBottom: '8px' }}>
                {qa.instruction.length > 150
                  ? qa.instruction.slice(0, 150) + '...'
                  : qa.instruction}
              </div>
              {qa.answer_text && (
                <div style={{
                  fontSize: '12px', color: 'var(--text-secondary)',
                  background: 'rgba(110, 231, 216, 0.05)',
                  padding: '8px', borderRadius: 'var(--radius-sm)',
                  maxHeight: '120px', overflow: 'auto',
                }}>
                  {qa.answer_text}
                </div>
              )}
              <div style={{ display: 'flex', gap: '6px', marginTop: '6px' }}>
                {qa.is_valid && (
                  <span className="badge badge-completed" style={{ fontSize: '11px' }}>valid</span>
                )}
                {qa.was_healed && (
                  <span className="badge badge-cancelled" style={{ fontSize: '11px' }}>healed</span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
