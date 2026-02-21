import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Database, Plus } from 'lucide-react'
import { useDatasetStore } from '../store/datasetStore'
import DatasetCard from '../components/datasets/DatasetCard'

export default function Datasets() {
  const { datasets, total, page, loading, fetchDatasets } = useDatasetStore()
  const navigate = useNavigate()

  useEffect(() => {
    fetchDatasets()
  }, [])

  const totalPages = Math.ceil(total / 20)

  return (
    <div>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1>Your Datasets</h1>
          <p>{total} dataset{total !== 1 ? 's' : ''}</p>
        </div>
        <button className="btn btn-primary" onClick={() => navigate('/create')}>
          <Plus size={16} />
          Create New Dataset
        </button>
      </div>

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
