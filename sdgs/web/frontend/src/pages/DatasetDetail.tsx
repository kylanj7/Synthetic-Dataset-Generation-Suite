import { useEffect, useState, useRef } from 'react'
import { useParams } from 'react-router-dom'
import { Search, Upload } from 'lucide-react'
import { useDatasetStore } from '../store/datasetStore'
import { useSSE } from '../hooks/useSSE'
import { getDataset } from '../api/client'
import StatsCards from '../components/datasets/StatsCards'
import HFPushModal from '../components/datasets/HFPushModal'

const statusClass: Record<string, string> = {
  pending: 'badge-pending',
  running: 'badge-running',
  completed: 'badge-completed',
  failed: 'badge-failed',
  cancelled: 'badge-cancelled',
}

export default function DatasetDetail() {
  const { id } = useParams<{ id: string }>()
  const datasetId = parseInt(id || '0')
  const {
    currentDataset, samples, samplesTotal, samplesPage,
    loading, fetchDataset, fetchSamples, updateDataset,
  } = useDatasetStore()
  const [search, setSearch] = useState('')
  const [showHFModal, setShowHFModal] = useState(false)
  const [expandedQA, setExpandedQA] = useState<number | null>(null)
  const logViewerRef = useRef<HTMLDivElement>(null)

  // SSE for running datasets
  const isRunning = currentDataset?.status === 'pending' || currentDataset?.status === 'running'
  const { logs, status: sseStatus, done } = useSSE(isRunning ? datasetId : null)

  useEffect(() => {
    fetchDataset(datasetId)
    fetchSamples(datasetId)
  }, [datasetId])

  // Refresh dataset after SSE completes
  useEffect(() => {
    if (done) {
      getDataset(datasetId).then(updateDataset).catch(() => {})
      fetchSamples(datasetId)
    }
  }, [done, datasetId])

  useEffect(() => {
    if (logViewerRef.current) {
      logViewerRef.current.scrollTop = logViewerRef.current.scrollHeight
    }
  }, [logs])

  const handleSearch = () => {
    fetchSamples(datasetId, 1, search || undefined)
  }

  const totalPages = Math.ceil(samplesTotal / 20)

  if (loading && !currentDataset) {
    return <div style={{ textAlign: 'center', padding: '40px' }}><div className="spinner" /></div>
  }

  if (!currentDataset) {
    return <div className="empty-state"><h3>Dataset not found</h3></div>
  }

  return (
    <div>
      {/* Header */}
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <h1>{currentDataset.topic}</h1>
            <span className={`badge ${statusClass[currentDataset.status] || 'badge-pending'}`}>
              {currentDataset.status}
            </span>
          </div>
          <p>
            {currentDataset.provider || 'default'}{currentDataset.model ? ` / ${currentDataset.model}` : ''}
            {currentDataset.created_at && ` — ${new Date(currentDataset.created_at).toLocaleString()}`}
          </p>
        </div>

        {currentDataset.status === 'completed' && (
          <button className="btn btn-primary" onClick={() => setShowHFModal(true)}>
            <Upload size={16} />
            Push to HuggingFace
          </button>
        )}
      </div>

      {/* HF Repo link */}
      {currentDataset.hf_repo && (
        <div className="card" style={{ marginBottom: '16px', padding: '12px 16px' }}>
          <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
            Published to HuggingFace:{' '}
            <a
              href={`https://huggingface.co/datasets/${currentDataset.hf_repo}`}
              target="_blank"
              rel="noopener noreferrer"
            >
              {currentDataset.hf_repo}
            </a>
          </span>
        </div>
      )}

      {/* Stats */}
      {currentDataset.status !== 'pending' && <StatsCards dataset={currentDataset} />}

      {/* Error */}
      {currentDataset.error_message && (
        <div className="card" style={{
          marginBottom: '16px',
          background: 'rgba(255, 126, 179, 0.05)',
          borderColor: 'rgba(255, 126, 179, 0.2)',
        }}>
          <h3 style={{ fontSize: '14px', fontWeight: 500, color: 'var(--accent-pink)', marginBottom: '8px' }}>Error</h3>
          <pre style={{ fontSize: '13px', color: 'var(--text-secondary)', whiteSpace: 'pre-wrap' }}>
            {currentDataset.error_message}
          </pre>
        </div>
      )}

      {/* Live SSE log for running datasets */}
      {isRunning && (
        <div className="card" style={{ marginBottom: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <h3 style={{ fontSize: '14px', fontWeight: 500 }}>Live Progress</h3>
            {sseStatus && <span className={`badge badge-${sseStatus}`}>{sseStatus}</span>}
          </div>
          <div className="log-viewer" ref={logViewerRef} style={{ maxHeight: '300px' }}>
            {logs.map((line, i) => (
              <div key={i} className="log-line">{line}</div>
            ))}
            {logs.length === 0 && (
              <div style={{ color: 'var(--text-muted)' }}>Waiting for pipeline to start...</div>
            )}
          </div>
        </div>
      )}

      {/* Q&A Pair Browser */}
      {(currentDataset.status === 'completed' || samples.length > 0) && (
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <h3 style={{ fontSize: '16px', fontWeight: 500 }}>Q&A Pairs ({samplesTotal})</h3>
            <div style={{ display: 'flex', gap: '8px' }}>
              <input
                type="text"
                placeholder="Search pairs..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                style={{ width: '250px' }}
              />
              <button className="btn" onClick={handleSearch}>
                <Search size={14} />
              </button>
            </div>
          </div>

          {samples.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '20px', color: 'var(--text-muted)' }}>
              No Q&A pairs found
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {samples.map((qa) => {
                const isExpanded = expandedQA === qa.id
                return (
                  <div
                    key={qa.id}
                    style={{
                      border: '1px solid var(--border-subtle)',
                      borderRadius: 'var(--radius-sm)',
                      padding: '12px 16px',
                      cursor: 'pointer',
                    }}
                    onClick={() => setExpandedQA(isExpanded ? null : qa.id)}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '12px' }}>
                      <div style={{ flex: 1 }}>
                        <div style={{ fontWeight: 500, fontSize: '13px', marginBottom: '4px' }}>
                          {qa.instruction.slice(0, 200)}{qa.instruction.length > 200 ? '...' : ''}
                        </div>
                        {!isExpanded && (
                          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                            {(qa.answer_text || qa.output).slice(0, 100)}...
                          </div>
                        )}
                      </div>
                      <div style={{ display: 'flex', gap: '6px', flexShrink: 0 }}>
                        {qa.was_healed && (
                          <span className="badge" style={{ background: 'rgba(255, 214, 102, 0.2)', color: 'var(--accent-gold)', fontSize: '11px' }}>
                            healed
                          </span>
                        )}
                        <span className={`badge ${qa.is_valid ? 'badge-completed' : 'badge-failed'}`} style={{ fontSize: '11px' }}>
                          {qa.is_valid ? 'valid' : 'invalid'}
                        </span>
                      </div>
                    </div>

                    {isExpanded && (
                      <div style={{ marginTop: '12px', borderTop: '1px solid var(--border-subtle)', paddingTop: '12px' }}>
                        {qa.source_title && (
                          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '8px' }}>
                            Source: {qa.source_title}
                          </div>
                        )}
                        {qa.think_text && (
                          <div style={{ marginBottom: '8px' }}>
                            <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--accent-purple)', marginBottom: '4px' }}>
                              THINK
                            </div>
                            <div style={{
                              fontSize: '13px',
                              color: 'var(--text-secondary)',
                              background: 'rgba(192, 132, 252, 0.05)',
                              padding: '8px 12px',
                              borderRadius: 'var(--radius-sm)',
                              whiteSpace: 'pre-wrap',
                              maxHeight: '200px',
                              overflow: 'auto',
                            }}>
                              {qa.think_text}
                            </div>
                          </div>
                        )}
                        <div>
                          <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--accent-cyan)', marginBottom: '4px' }}>
                            ANSWER
                          </div>
                          <div style={{
                            fontSize: '13px',
                            color: 'var(--text-secondary)',
                            background: 'rgba(110, 231, 216, 0.05)',
                            padding: '8px 12px',
                            borderRadius: 'var(--radius-sm)',
                            whiteSpace: 'pre-wrap',
                            maxHeight: '300px',
                            overflow: 'auto',
                          }}>
                            {qa.answer_text || qa.output}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}

          {totalPages > 1 && (
            <div className="pagination">
              <button
                className="btn"
                disabled={samplesPage <= 1}
                onClick={() => fetchSamples(datasetId, samplesPage - 1, search || undefined)}
              >
                Previous
              </button>
              <span>Page {samplesPage} of {totalPages}</span>
              <button
                className="btn"
                disabled={samplesPage >= totalPages}
                onClick={() => fetchSamples(datasetId, samplesPage + 1, search || undefined)}
              >
                Next
              </button>
            </div>
          )}
        </div>
      )}

      {/* HF Push Modal */}
      {showHFModal && (
        <HFPushModal
          datasetId={datasetId}
          onClose={() => setShowHFModal(false)}
          onSuccess={(hfRepo) => {
            updateDataset({ ...currentDataset, hf_repo: hfRepo })
            setShowHFModal(false)
          }}
        />
      )}
    </div>
  )
}
