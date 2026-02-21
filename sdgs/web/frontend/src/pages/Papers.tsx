import { useState, useEffect } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { Search, ExternalLink, Download, ChevronLeft, ChevronRight, HardDrive, Sparkles } from 'lucide-react'
import { getPapers, getPaperTopics, PaperInfo } from '../api/client'

export default function Papers() {
  const [searchParams, setSearchParams] = useSearchParams()
  const navigate = useNavigate()
  const [papers, setPapers] = useState<PaperInfo[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState(searchParams.get('search') || '')
  const [loading, setLoading] = useState(true)
  const [topics, setTopics] = useState<string[]>([])
  const [selectedTopic, setSelectedTopic] = useState('')
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set())

  const datasetId = searchParams.get('dataset_id')
    ? Number(searchParams.get('dataset_id'))
    : undefined

  useEffect(() => {
    getPaperTopics().then(setTopics).catch(() => {})
  }, [])

  useEffect(() => {
    setLoading(true)
    getPapers(page, search || undefined, datasetId, selectedTopic || undefined)
      .then((res) => {
        setPapers(res.papers)
        setTotal(res.total)
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [page, search, datasetId, selectedTopic])

  const totalPages = Math.ceil(total / 50)

  const handleSearch = (val: string) => {
    setSearch(val)
    setPage(1)
    setSelectedIds(new Set())
    const params = new URLSearchParams(searchParams)
    if (val) params.set('search', val)
    else params.delete('search')
    setSearchParams(params, { replace: true })
  }

  const handleTopicChange = (val: string) => {
    setSelectedTopic(val)
    setPage(1)
    setSelectedIds(new Set())
  }

  const toggleSelect = (id: number) => {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const toggleSelectAll = () => {
    const currentPageIds = papers.map((p) => p.id)
    const allSelected = currentPageIds.every((id) => selectedIds.has(id))
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (allSelected) {
        currentPageIds.forEach((id) => next.delete(id))
      } else {
        currentPageIds.forEach((id) => next.add(id))
      }
      return next
    })
  }

  const handleGenerateFromSelected = () => {
    const ids = Array.from(selectedIds).join(',')
    navigate(`/create?from_papers=${ids}`)
  }

  const allOnPageSelected = papers.length > 0 && papers.every((p) => selectedIds.has(p.id))

  return (
    <div>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1>Papers</h1>
          <p>All scholarly papers used in dataset generation ({total} total)</p>
        </div>
        {selectedIds.size > 0 && (
          <button className="btn btn-primary" onClick={handleGenerateFromSelected}>
            <Sparkles size={16} />
            Generate from {selectedIds.size} Selected
          </button>
        )}
      </div>

      {/* Search + Topic Filter */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
        <div style={{ position: 'relative', flex: 1 }}>
          <Search
            size={16}
            style={{
              position: 'absolute',
              left: '12px',
              top: '50%',
              transform: 'translateY(-50%)',
              color: 'var(--text-muted)',
            }}
          />
          <input
            type="text"
            placeholder="Search papers by title..."
            value={search}
            onChange={(e) => handleSearch(e.target.value)}
            style={{ paddingLeft: '36px', fontSize: '14px', width: '100%' }}
          />
        </div>
        {topics.length > 0 && (
          <select
            value={selectedTopic}
            onChange={(e) => handleTopicChange(e.target.value)}
            style={{ width: '220px', fontSize: '14px' }}
          >
            <option value="">All Topics</option>
            {topics.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        )}
      </div>

      {datasetId && (
        <div style={{
          marginBottom: '16px',
          padding: '8px 12px',
          background: 'rgba(126, 184, 255, 0.1)',
          border: '1px solid rgba(126, 184, 255, 0.2)',
          borderRadius: 'var(--radius-sm)',
          fontSize: '13px',
          color: 'var(--text-secondary)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}>
          <span>Showing papers for dataset #{datasetId}</span>
          <button
            onClick={() => {
              const params = new URLSearchParams(searchParams)
              params.delete('dataset_id')
              setSearchParams(params, { replace: true })
            }}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--accent-blue)',
              cursor: 'pointer',
              fontSize: '13px',
            }}
          >
            Show all
          </button>
        </div>
      )}

      {/* Table */}
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
              <th style={{ ...thStyle, width: '40px', textAlign: 'center' }}>
                <input
                  type="checkbox"
                  checked={allOnPageSelected}
                  onChange={toggleSelectAll}
                  style={{ cursor: 'pointer' }}
                />
              </th>
              <th style={thStyle}>Title</th>
              <th style={{ ...thStyle, width: '180px' }}>Authors</th>
              <th style={{ ...thStyle, width: '60px', textAlign: 'center' }}>Year</th>
              <th style={{ ...thStyle, width: '70px', textAlign: 'center' }}>Source</th>
              <th style={{ ...thStyle, width: '80px', textAlign: 'right' }}>Citations</th>
              <th style={{ ...thStyle, width: '60px', textAlign: 'right' }}>QA</th>
              <th style={{ ...thStyle, width: '40px' }}></th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={8} style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
                  <span className="spinner" />
                </td>
              </tr>
            ) : papers.length === 0 ? (
              <tr>
                <td colSpan={8} style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
                  {search ? 'No papers match your search' : 'No papers yet. Generate a dataset to see papers here.'}
                </td>
              </tr>
            ) : (
              papers.map((p) => (
                <tr key={p.id} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>
                    <input
                      type="checkbox"
                      checked={selectedIds.has(p.id)}
                      onChange={() => toggleSelect(p.id)}
                      style={{ cursor: 'pointer' }}
                    />
                  </td>
                  <td style={tdStyle}>
                    <div style={{ fontWeight: 500, color: 'var(--text-primary)', lineHeight: 1.4, display: 'flex', alignItems: 'center', gap: '6px' }}>
                      {p.pdf_path ? (
                        <a
                          href={`/api/papers/${p.id}/pdf`}
                          target="_blank"
                          rel="noopener noreferrer"
                          style={{ color: 'var(--accent-blue)', textDecoration: 'none' }}
                          title="View PDF"
                        >
                          {p.title}
                        </a>
                      ) : (
                        p.title
                      )}
                      {p.pdf_path && (
                        <span title="PDF saved locally" style={{ display: 'flex', flexShrink: 0 }}>
                          <HardDrive size={12} style={{ color: 'var(--accent-green)' }} />
                        </span>
                      )}
                    </div>
                  </td>
                  <td style={{ ...tdStyle, color: 'var(--text-secondary)' }}>
                    {p.authors.length > 0
                      ? p.authors.length <= 2
                        ? p.authors.join(', ')
                        : `${p.authors[0]} et al.`
                      : <span style={{ color: 'var(--text-muted)' }}>—</span>
                    }
                  </td>
                  <td style={{ ...tdStyle, textAlign: 'center', color: 'var(--text-secondary)' }}>
                    {p.year || '—'}
                  </td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>
                    {p.source ? (
                      <span style={{
                        fontSize: '11px',
                        padding: '2px 6px',
                        borderRadius: '3px',
                        background: p.source === 'arxiv' ? 'rgba(126, 184, 255, 0.15)' : 'rgba(126, 217, 195, 0.15)',
                        color: p.source === 'arxiv' ? 'var(--accent-blue)' : 'var(--accent-green)',
                      }}>
                        {p.source === 'semantic_scholar' ? 'S2' : p.source}
                      </span>
                    ) : '—'}
                  </td>
                  <td style={{ ...tdStyle, textAlign: 'right', color: 'var(--text-secondary)' }}>
                    {p.citation_count > 0 ? p.citation_count.toLocaleString() : '—'}
                  </td>
                  <td style={{ ...tdStyle, textAlign: 'right', color: 'var(--text-secondary)' }}>
                    {p.qa_pair_count > 0 ? p.qa_pair_count : '—'}
                  </td>
                  <td style={tdStyle}>
                    {p.pdf_path ? (
                      <a
                        href={`/api/papers/${p.id}/pdf`}
                        download
                        style={{ color: 'var(--accent-blue)', display: 'flex' }}
                        title="Download PDF"
                      >
                        <Download size={14} />
                      </a>
                    ) : p.url ? (
                      <a
                        href={p.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{ color: 'var(--text-muted)', display: 'flex' }}
                        title="Open paper"
                      >
                        <ExternalLink size={14} />
                      </a>
                    ) : null}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          gap: '12px',
          marginTop: '16px',
          fontSize: '13px',
          color: 'var(--text-secondary)',
        }}>
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1}
            style={paginationBtnStyle}
          >
            <ChevronLeft size={16} />
          </button>
          <span>Page {page} of {totalPages}</span>
          <button
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page >= totalPages}
            style={paginationBtnStyle}
          >
            <ChevronRight size={16} />
          </button>
        </div>
      )}
    </div>
  )
}

const thStyle: React.CSSProperties = {
  padding: '10px 12px',
  textAlign: 'left',
  fontWeight: 500,
  color: 'var(--text-muted)',
  fontSize: '12px',
  textTransform: 'uppercase',
  letterSpacing: '0.5px',
}

const tdStyle: React.CSSProperties = {
  padding: '10px 12px',
  verticalAlign: 'top',
}

const paginationBtnStyle: React.CSSProperties = {
  background: 'var(--bg-tertiary)',
  border: '1px solid var(--border-primary)',
  borderRadius: 'var(--radius-sm)',
  color: 'var(--text-secondary)',
  cursor: 'pointer',
  padding: '4px 8px',
  display: 'flex',
  alignItems: 'center',
}
