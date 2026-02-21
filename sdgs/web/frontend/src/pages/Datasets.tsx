import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Database, Plus, Download } from 'lucide-react'
import { useDatasetStore } from '../store/datasetStore'
import { importFromHuggingFace } from '../api/client'
import DatasetCard from '../components/datasets/DatasetCard'

export default function Datasets() {
  const { datasets, total, page, loading, fetchDatasets } = useDatasetStore()
  const navigate = useNavigate()
  const [showImport, setShowImport] = useState(false)
  const [repoId, setRepoId] = useState('')
  const [split, setSplit] = useState('')
  const [importing, setImporting] = useState(false)
  const [importError, setImportError] = useState('')

  useEffect(() => {
    fetchDatasets()
  }, [])

  const totalPages = Math.ceil(total / 20)

  const handleImport = async () => {
    if (!repoId.trim()) return
    setImportError('')
    setImporting(true)
    try {
      const ds = await importFromHuggingFace({
        repo_id: repoId.trim(),
        split: split.trim() || undefined,
      })
      setShowImport(false)
      setRepoId('')
      setSplit('')
      navigate(`/datasets/${ds.id}`)
    } catch (e) {
      setImportError(e instanceof Error ? e.message : 'Failed to import dataset')
    } finally {
      setImporting(false)
    }
  }

  return (
    <div>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1>Your Datasets</h1>
          <p>{total} dataset{total !== 1 ? 's' : ''}</p>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            className="btn"
            onClick={() => setShowImport(!showImport)}
            style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
          >
            <Download size={16} />
            Import from HF
          </button>
          <button className="btn btn-primary" onClick={() => navigate('/create')}>
            <Plus size={16} />
            Create New Dataset
          </button>
        </div>
      </div>

      {/* Import inline form */}
      {showImport && (
        <div className="card" style={{ marginBottom: '16px', padding: '16px' }}>
          <div style={{ fontSize: '14px', fontWeight: 500, marginBottom: '12px' }}>
            Import from HuggingFace
          </div>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'flex-end' }}>
            <div style={{ flex: 1 }}>
              <label style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Repository ID</label>
              <input
                type="text"
                placeholder="e.g. tatsu-lab/alpaca"
                value={repoId}
                onChange={(e) => setRepoId(e.target.value)}
                disabled={importing}
                onKeyDown={(e) => e.key === 'Enter' && handleImport()}
                style={{ fontSize: '14px' }}
              />
            </div>
            <div style={{ width: '140px' }}>
              <label style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Split (optional)</label>
              <input
                type="text"
                placeholder="train"
                value={split}
                onChange={(e) => setSplit(e.target.value)}
                disabled={importing}
                style={{ fontSize: '14px' }}
              />
            </div>
            <button
              className="btn btn-primary"
              onClick={handleImport}
              disabled={importing || !repoId.trim()}
              style={{ whiteSpace: 'nowrap' }}
            >
              {importing ? <span className="spinner" /> : 'Import'}
            </button>
            <button
              className="btn"
              onClick={() => { setShowImport(false); setImportError('') }}
              disabled={importing}
            >
              Cancel
            </button>
          </div>
          {importError && (
            <div style={{
              marginTop: '8px',
              fontSize: '13px',
              color: 'var(--accent-pink)',
              background: 'rgba(255, 126, 179, 0.1)',
              padding: '6px 10px',
              borderRadius: 'var(--radius-sm)',
            }}>
              {importError}
            </div>
          )}
        </div>
      )}

      {loading && datasets.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '40px' }}>
          <div className="spinner" />
        </div>
      ) : datasets.length === 0 ? (
        <div className="empty-state">
          <Database size={48} style={{ marginBottom: '12px', opacity: 0.3 }} />
          <h3>No datasets yet</h3>
          <p>Create your first synthetic Q&A dataset from academic papers.</p>
          <button
            className="btn btn-primary"
            style={{ marginTop: '16px' }}
            onClick={() => navigate('/create')}
          >
            <Plus size={16} />
            Create Dataset
          </button>
        </div>
      ) : (
        <>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {datasets.map((ds) => (
              <DatasetCard
                key={ds.id}
                dataset={ds}
                onClick={() => navigate(`/datasets/${ds.id}`)}
              />
            ))}
          </div>

          {totalPages > 1 && (
            <div className="pagination">
              <button
                className="btn"
                disabled={page <= 1}
                onClick={() => fetchDatasets(page - 1)}
              >
                Previous
              </button>
              <span>Page {page} of {totalPages}</span>
              <button
                className="btn"
                disabled={page >= totalPages}
                onClick={() => fetchDatasets(page + 1)}
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
