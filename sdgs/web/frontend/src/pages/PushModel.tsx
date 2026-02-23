import { useState, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { ChevronDown, ChevronRight } from 'lucide-react'
import { pushModel, getArtifacts, ArtifactEntry } from '../api/client'

export default function PushModel() {
  const [searchParams] = useSearchParams()
  const prefilledGguf = searchParams.get('gguf_path') || ''

  const [ggufFiles, setGgufFiles] = useState<ArtifactEntry[]>([])
  const [adapters, setAdapters] = useState<ArtifactEntry[]>([])
  const [sourceType, setSourceType] = useState<'gguf' | 'merged'>(prefilledGguf ? 'gguf' : 'gguf')
  const [path, setPath] = useState(prefilledGguf)
  const [customPath, setCustomPath] = useState(!!prefilledGguf)
  const [repoId, setRepoId] = useState('')

  useEffect(() => {
    getArtifacts().then((res) => {
      setGgufFiles(res.gguf_files)
      setAdapters(res.adapters)
      // If prefilled path matches a known artifact, don't use custom mode
      if (prefilledGguf && res.gguf_files.some((f) => f.path === prefilledGguf)) {
        setCustomPath(false)
      }
    }).catch(() => {})
  }, [])
  const [isPrivate, setIsPrivate] = useState(true)

  // Metadata
  const [showMeta, setShowMeta] = useState(false)
  const [baseModel, setBaseModel] = useState('')
  const [description, setDescription] = useState('')
  const [dataset, setDataset] = useState('')
  const [author, setAuthor] = useState('')

  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const [repoUrl, setRepoUrl] = useState<string | null>(null)

  const handleSubmit = async () => {
    setError('')
    setSubmitting(true)
    try {
      const res = await pushModel({
        repo_id: repoId.trim(),
        gguf_path: sourceType === 'gguf' ? path.trim() : undefined,
        merged_model_dir: sourceType === 'merged' ? path.trim() : undefined,
        private: isPrivate,
        base_model: baseModel.trim() || undefined,
        description: description.trim() || undefined,
        dataset: dataset.trim() || undefined,
        author: author.trim() || undefined,
      })
      setRepoUrl(res.repo_url)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Push failed')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div style={{ maxWidth: '700px' }}>
      <div className="page-header">
        <h1>Push Model</h1>
        <p>Upload a model to HuggingFace Hub</p>
      </div>

      {!repoUrl ? (
        <div className="card">
          {/* Source type toggle */}
          <div style={{ marginBottom: '16px' }}>
            <label style={{ fontSize: '15px', fontWeight: 500, color: 'var(--text-primary)', marginBottom: '8px' }}>
              Source Type
            </label>
            <div style={{ display: 'flex', gap: '8px', marginTop: '8px' }}>
              <button
                onClick={() => { setSourceType('gguf'); setCustomPath(false); setPath('') }}
                disabled={submitting}
                style={{
                  background: sourceType === 'gguf' ? 'var(--accent-blue)' : 'var(--bg-tertiary)',
                  border: '1px solid ' + (sourceType === 'gguf' ? 'var(--accent-blue)' : 'var(--border-primary)'),
                  color: sourceType === 'gguf' ? '#fff' : 'var(--text-secondary)',
                  cursor: 'pointer',
                  fontSize: '13px',
                  padding: '6px 12px',
                  borderRadius: 'var(--radius-sm)',
                }}
              >
                GGUF File
              </button>
              <button
                onClick={() => { setSourceType('merged'); setCustomPath(false); setPath('') }}
                disabled={submitting}
                style={{
                  background: sourceType === 'merged' ? 'var(--accent-blue)' : 'var(--bg-tertiary)',
                  border: '1px solid ' + (sourceType === 'merged' ? 'var(--accent-blue)' : 'var(--border-primary)'),
                  color: sourceType === 'merged' ? '#fff' : 'var(--text-secondary)',
                  cursor: 'pointer',
                  fontSize: '13px',
                  padding: '6px 12px',
                  borderRadius: 'var(--radius-sm)',
                }}
              >
                Merged Model Dir
              </button>
            </div>
          </div>

          <div style={{ marginBottom: '16px' }}>
            <label>{sourceType === 'gguf' ? 'GGUF Path' : 'Merged Model Directory'}</label>
            {!customPath ? (
              <select
                value={path}
                onChange={(e) => {
                  if (e.target.value === '__custom__') {
                    setCustomPath(true)
                    setPath('')
                  } else {
                    setPath(e.target.value)
                  }
                }}
                disabled={submitting}
              >
                <option value="">Select {sourceType === 'gguf' ? 'a GGUF file' : 'a model'}...</option>
                {(sourceType === 'gguf' ? ggufFiles : adapters).map((a) => (
                  <option key={a.path} value={a.path}>{a.label}</option>
                ))}
                <option value="__custom__">Custom path...</option>
              </select>
            ) : (
              <>
                <input
                  type="text"
                  placeholder={sourceType === 'gguf' ? '/path/to/model.gguf' : '/path/to/merged-model/'}
                  value={path}
                  onChange={(e) => setPath(e.target.value)}
                  disabled={submitting}
                />
                <button
                  onClick={() => { setCustomPath(false); setPath('') }}
                  style={{
                    background: 'none', border: 'none', color: 'var(--accent-blue)',
                    cursor: 'pointer', fontSize: '12px', padding: '4px 0', marginTop: '4px',
                  }}
                >
                  Back to dropdown
                </button>
              </>
            )}
          </div>

          <div style={{ marginBottom: '16px' }}>
            <label>Repo ID *</label>
            <input
              type="text"
              placeholder="username/my-model"
              value={repoId}
              onChange={(e) => setRepoId(e.target.value)}
              disabled={submitting}
            />
          </div>

          <div style={{ marginBottom: '16px' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={isPrivate}
                onChange={(e) => setIsPrivate(e.target.checked)}
                disabled={submitting}
                style={{ width: 'auto' }}
              />
              Private repository
            </label>
          </div>

          {/* Metadata collapsible */}
          <div style={{ marginBottom: '20px' }}>
            <button
              onClick={() => setShowMeta(!showMeta)}
              style={{
                background: 'none',
                border: 'none',
                color: 'var(--text-secondary)',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                fontSize: '13px',
                padding: 0,
              }}
            >
              {showMeta ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
              Metadata
            </button>

            {showMeta && (
              <div style={{ marginTop: '12px', paddingLeft: '18px' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '12px' }}>
                  <div>
                    <label>Base Model</label>
                    <input
                      type="text"
                      placeholder="Qwen/Qwen2.5-14B-Instruct"
                      value={baseModel}
                      onChange={(e) => setBaseModel(e.target.value)}
                      disabled={submitting}
                    />
                  </div>
                  <div>
                    <label>Author</label>
                    <input
                      type="text"
                      value={author}
                      onChange={(e) => setAuthor(e.target.value)}
                      disabled={submitting}
                    />
                  </div>
                </div>
                <div style={{ marginBottom: '12px' }}>
                  <label>Dataset</label>
                  <input
                    type="text"
                    value={dataset}
                    onChange={(e) => setDataset(e.target.value)}
                    disabled={submitting}
                  />
                </div>
                <div>
                  <label>Description</label>
                  <textarea
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    disabled={submitting}
                    rows={3}
                    style={{
                      width: '100%',
                      resize: 'vertical',
                      background: 'var(--bg-tertiary)',
                      border: '1px solid var(--border-primary)',
                      borderRadius: 'var(--radius-sm)',
                      color: 'var(--text-primary)',
                      padding: '8px 12px',
                      fontSize: '13px',
                      fontFamily: 'inherit',
                    }}
                  />
                </div>
              </div>
            )}
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

          <button
            className="btn btn-primary"
            onClick={handleSubmit}
            disabled={submitting || !repoId.trim() || !path.trim()}
            style={{ width: '100%', justifyContent: 'center', padding: '10px 20px', fontSize: '15px' }}
          >
            {submitting ? <span className="spinner" /> : 'Push to HuggingFace'}
          </button>
        </div>
      ) : (
        <div className="card">
          <div style={{ fontSize: '14px', fontWeight: 500, color: 'var(--accent-green)', marginBottom: '8px' }}>
            Push Complete
          </div>
          <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
            Repository:{' '}
            <a
              href={repoUrl}
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: 'var(--accent-cyan)' }}
            >
              {repoUrl}
            </a>
          </div>
        </div>
      )}
    </div>
  )
}
