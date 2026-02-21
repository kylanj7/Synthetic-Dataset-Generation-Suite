import { useState } from 'react'
import { X } from 'lucide-react'
import { pushToHuggingFace } from '../../api/client'

interface Props {
  datasetId: number
  onClose: () => void
  onSuccess: (hfRepo: string) => void
}

export default function HFPushModal({ datasetId, onClose, onSuccess }: Props) {
  const [repoName, setRepoName] = useState('')
  const [description, setDescription] = useState('')
  const [isPrivate, setIsPrivate] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handlePush = async () => {
    if (!repoName.trim()) return
    setError('')
    setLoading(true)

    try {
      const res = await pushToHuggingFace(datasetId, {
        repo_name: repoName.trim(),
        description: description.trim(),
        private: isPrivate,
      })
      onSuccess(res.hf_repo)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Push failed')
      setLoading(false)
    }
  }

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      background: 'rgba(0, 0, 0, 0.7)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
    }} onClick={onClose}>
      <div
        className="card"
        style={{ width: '100%', maxWidth: '480px' }}
        onClick={(e) => e.stopPropagation()}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <h2 style={{ fontSize: '18px', fontWeight: 600 }}>Push to HuggingFace</h2>
          <button
            onClick={onClose}
            style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}
          >
            <X size={20} />
          </button>
        </div>

        {error && (
          <div style={{
            background: 'rgba(255, 126, 179, 0.1)',
            border: '1px solid rgba(255, 126, 179, 0.3)',
            borderRadius: 'var(--radius-sm)',
            padding: '8px 12px',
            color: 'var(--accent-pink)',
            fontSize: '13px',
            marginBottom: '16px',
          }}>
            {error}
          </div>
        )}

        <div style={{ marginBottom: '16px' }}>
          <label>Repository Name</label>
          <input
            type="text"
            placeholder="username/my-dataset"
            value={repoName}
            onChange={(e) => setRepoName(e.target.value)}
            disabled={loading}
            autoFocus
          />
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
            Format: username/dataset-name or organization/dataset-name
          </div>
        </div>

        <div style={{ marginBottom: '16px' }}>
          <label>Description (optional)</label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            disabled={loading}
            rows={2}
            placeholder="A brief description of this dataset"
          />
        </div>

        <div style={{ marginBottom: '20px' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={isPrivate}
              onChange={(e) => setIsPrivate(e.target.checked)}
              disabled={loading}
              style={{ width: 'auto' }}
            />
            Private repository
          </label>
        </div>

        <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
          <button className="btn" onClick={onClose} disabled={loading}>Cancel</button>
          <button
            className="btn btn-primary"
            onClick={handlePush}
            disabled={loading || !repoName.trim()}
          >
            {loading ? <span className="spinner" /> : 'Push to HuggingFace'}
          </button>
        </div>
      </div>
    </div>
  )
}
