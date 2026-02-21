import { useState, useEffect, useRef } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { ChevronDown, ChevronRight, CheckCircle, XCircle, StopCircle, Plus, Trash2, Layers, FileText } from 'lucide-react'
import { getProviders, cancelDataset, createBatchDatasets, createDatasetFromPapers, ProviderInfo } from '../api/client'
import { useDatasetStore } from '../store/datasetStore'
import { useSSE } from '../hooks/useSSE'

interface BatchRow {
  id: number
  topic: string
  targetSize: number
}

let nextRowId = 1

export default function CreateDataset() {
  const [searchParams] = useSearchParams()
  const fromPapersParam = searchParams.get('from_papers')
  const paperIds = fromPapersParam
    ? fromPapersParam.split(',').map(Number).filter((n) => !isNaN(n) && n > 0)
    : []
  const isFromPapers = paperIds.length > 0

  const [topic, setTopic] = useState('')
  const [targetSize, setTargetSize] = useState(100)
  const [provider, setProvider] = useState<string>('ollama')
  const [model, setModel] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [systemPrompt, setSystemPrompt] = useState('')
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(4096)
  const [providers, setProviders] = useState<ProviderInfo[]>([])
  const [generating, setGenerating] = useState(false)
  const [datasetId, setDatasetId] = useState<number | null>(null)
  const [error, setError] = useState('')
  const logViewerRef = useRef<HTMLDivElement>(null)

  // Batch mode state
  const [batchMode, setBatchMode] = useState(false)
  const [batchRows, setBatchRows] = useState<BatchRow[]>([
    { id: nextRowId++, topic: '', targetSize: 100 },
    { id: nextRowId++, topic: '', targetSize: 100 },
  ])

  const { createDataset } = useDatasetStore()
  const { logs, status, done } = useSSE(datasetId)
  const navigate = useNavigate()

  useEffect(() => {
    getProviders().then(setProviders).catch(() => {})
  }, [])

  useEffect(() => {
    if (logViewerRef.current) {
      logViewerRef.current.scrollTop = logViewerRef.current.scrollHeight
    }
  }, [logs])

  useEffect(() => {
    if (done && status === 'completed' && datasetId) {
      setTimeout(() => navigate(`/datasets/${datasetId}`), 1000)
    }
  }, [done, status, datasetId, navigate])

  const selectedProvider = providers.find((p) => p.name === provider)

  const handleGenerate = async () => {
    if (isFromPapers) {
      setError('')
      setGenerating(true)
      try {
        const ds = await createDatasetFromPapers({
          paper_ids: paperIds,
          provider: provider || undefined,
          model: model || undefined,
          system_prompt: systemPrompt || undefined,
          temperature,
          max_tokens: maxTokens,
        })
        setDatasetId(ds.id)
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to create dataset from papers')
        setGenerating(false)
      }
      return
    }

    if (!topic.trim()) return
    setError('')
    setGenerating(true)

    try {
      const ds = await createDataset({
        topic: topic.trim(),
        provider: provider || undefined,
        model: model || undefined,
        target_size: targetSize,
        system_prompt: systemPrompt || undefined,
        temperature,
        max_tokens: maxTokens,
      })
      setDatasetId(ds.id)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to create dataset')
      setGenerating(false)
    }
  }

  const handleBatchGenerate = async () => {
    const validRows = batchRows.filter((r) => r.topic.trim())
    if (validRows.length === 0) return
    setError('')
    setGenerating(true)

    try {
      await createBatchDatasets(
        validRows.map((r) => ({
          topic: r.topic.trim(),
          provider: provider || undefined,
          model: model || undefined,
          target_size: r.targetSize,
          system_prompt: systemPrompt || undefined,
          temperature,
          max_tokens: maxTokens,
        }))
      )
      navigate('/datasets')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to create batch datasets')
      setGenerating(false)
    }
  }

  const updateBatchRow = (id: number, field: keyof Omit<BatchRow, 'id'>, value: string | number) => {
    setBatchRows((rows) =>
      rows.map((r) => (r.id === id ? { ...r, [field]: value } : r))
    )
  }

  const addBatchRow = () => {
    setBatchRows((rows) => [...rows, { id: nextRowId++, topic: '', targetSize: 100 }])
  }

  const removeBatchRow = (id: number) => {
    setBatchRows((rows) => rows.length > 1 ? rows.filter((r) => r.id !== id) : rows)
  }

  const validBatchCount = batchRows.filter((r) => r.topic.trim()).length

  return (
    <div style={{ maxWidth: '700px' }}>
      <div className="page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <h1>{isFromPapers ? 'Generate from Papers' : 'Create Dataset'}</h1>
            <p>{isFromPapers ? 'Generate Q&A pairs from selected papers' : 'Generate a synthetic Q&A dataset from academic papers'}</p>
          </div>
          {!isFromPapers && (
            <button
              onClick={() => setBatchMode(!batchMode)}
              disabled={generating}
              style={{
                background: batchMode ? 'var(--accent-blue)' : 'var(--bg-tertiary)',
                border: '1px solid ' + (batchMode ? 'var(--accent-blue)' : 'var(--border-primary)'),
                color: batchMode ? '#fff' : 'var(--text-secondary)',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                fontSize: '13px',
                padding: '6px 12px',
                borderRadius: 'var(--radius-sm)',
                whiteSpace: 'nowrap',
              }}
            >
              <Layers size={14} />
              Batch Mode
            </button>
          )}
        </div>
      </div>

      <div className="card" style={{ marginBottom: '20px' }}>
        {/* Paper-based generation banner */}
        {isFromPapers && (
          <div style={{
            marginBottom: '16px',
            padding: '10px 14px',
            background: 'rgba(126, 184, 255, 0.1)',
            border: '1px solid rgba(126, 184, 255, 0.2)',
            borderRadius: 'var(--radius-sm)',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '13px',
            color: 'var(--text-secondary)',
          }}>
            <FileText size={16} style={{ color: 'var(--accent-blue)' }} />
            Generating from {paperIds.length} selected paper{paperIds.length !== 1 ? 's' : ''}
          </div>
        )}

        {/* Single mode: Topic */}
        {!batchMode && !isFromPapers && (
          <div style={{ marginBottom: '20px' }}>
            <label style={{ fontSize: '15px', fontWeight: 500, color: 'var(--text-primary)', marginBottom: '8px' }}>
              What should this dataset be about?
            </label>
            <input
              type="text"
              placeholder='e.g. "quantum computing error correction"'
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              disabled={generating}
              autoFocus
              style={{ fontSize: '15px', padding: '12px 16px' }}
            />
          </div>
        )}

        {/* Batch mode: Dataset rows */}
        {batchMode && (
          <div style={{ marginBottom: '20px' }}>
            <label style={{ fontSize: '15px', fontWeight: 500, color: 'var(--text-primary)', marginBottom: '12px', display: 'block' }}>
              Datasets to generate
            </label>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {batchRows.map((row, idx) => (
                <div key={row.id} style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <span style={{ fontSize: '12px', color: 'var(--text-muted)', width: '20px', textAlign: 'right', flexShrink: 0 }}>
                    {idx + 1}.
                  </span>
                  <input
                    type="text"
                    placeholder="Topic (e.g. drug discovery for cancer research)"
                    value={row.topic}
                    onChange={(e) => updateBatchRow(row.id, 'topic', e.target.value)}
                    disabled={generating}
                    style={{ flex: 1, fontSize: '14px', padding: '8px 12px' }}
                  />
                  <div style={{ display: 'flex', alignItems: 'center', gap: '4px', flexShrink: 0 }}>
                    <input
                      type="number"
                      value={row.targetSize}
                      onChange={(e) => updateBatchRow(row.id, 'targetSize', Math.max(10, parseInt(e.target.value) || 10))}
                      min={10}
                      disabled={generating}
                      style={{ width: '80px', fontSize: '14px', padding: '8px', textAlign: 'right' }}
                    />
                    <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>pairs</span>
                  </div>
                  <button
                    onClick={() => removeBatchRow(row.id)}
                    disabled={generating || batchRows.length <= 1}
                    style={{
                      background: 'none',
                      border: 'none',
                      color: batchRows.length <= 1 ? 'var(--text-muted)' : 'var(--accent-pink)',
                      cursor: batchRows.length <= 1 ? 'default' : 'pointer',
                      padding: '4px',
                      flexShrink: 0,
                      display: 'flex',
                      alignItems: 'center',
                    }}
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}
            </div>
            <button
              onClick={addBatchRow}
              disabled={generating}
              style={{
                background: 'none',
                border: '1px dashed var(--border-primary)',
                color: 'var(--text-secondary)',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '6px',
                fontSize: '13px',
                padding: '8px',
                borderRadius: 'var(--radius-sm)',
                width: '100%',
                marginTop: '8px',
              }}
            >
              <Plus size={14} />
              Add Dataset
            </button>
          </div>
        )}

        {/* Single/papers mode: Provider, Model (+ Target Size for non-paper mode) */}
        {!batchMode && (
          isFromPapers ? (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px' }}>
              <div>
                <label>Provider</label>
                <select
                  value={provider}
                  onChange={(e) => setProvider(e.target.value)}
                  disabled={generating}
                >
                  {providers.map((p) => (
                    <option key={p.name} value={p.name}>{p.name}</option>
                  ))}
                </select>
              </div>
              <div>
                <label>Model</label>
                <input
                  type="text"
                  placeholder={selectedProvider?.default_model || '(default)'}
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  disabled={generating}
                />
              </div>
            </div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '16px', marginBottom: '20px' }}>
              <div>
                <label>Target Size</label>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <input
                    type="number"
                    value={targetSize}
                    onChange={(e) => setTargetSize(Math.max(10, parseInt(e.target.value) || 10))}
                    min={10}
                    disabled={generating}
                    style={{ width: '100px' }}
                  />
                  <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>pairs</span>
                </div>
              </div>
              <div>
                <label>Provider</label>
                <select
                  value={provider}
                  onChange={(e) => setProvider(e.target.value)}
                  disabled={generating}
                >
                  {providers.map((p) => (
                    <option key={p.name} value={p.name}>{p.name}</option>
                  ))}
                </select>
              </div>
              <div>
                <label>Model</label>
                <input
                  type="text"
                  placeholder={selectedProvider?.default_model || '(default)'}
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  disabled={generating}
                />
              </div>
            </div>
          )
        )}

        {/* Batch mode: Shared Provider/Model */}
        {batchMode && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px' }}>
            <div>
              <label>Provider (shared)</label>
              <select
                value={provider}
                onChange={(e) => setProvider(e.target.value)}
                disabled={generating}
              >
                {providers.map((p) => (
                  <option key={p.name} value={p.name}>{p.name}</option>
                ))}
              </select>
            </div>
            <div>
              <label>Model (shared)</label>
              <input
                type="text"
                placeholder={selectedProvider?.default_model || '(default)'}
                value={model}
                onChange={(e) => setModel(e.target.value)}
                disabled={generating}
              />
            </div>
          </div>
        )}

        {/* API Key status */}
        {provider && (
          <div style={{
            fontSize: '13px',
            marginBottom: '16px',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}>
            {selectedProvider?.has_key ? (
              <>
                <CheckCircle size={14} style={{ color: 'var(--accent-green)' }} />
                <span style={{ color: 'var(--text-secondary)' }}>API Key: Configured (from settings)</span>
              </>
            ) : selectedProvider?.api_key_env ? (
              <>
                <XCircle size={14} style={{ color: 'var(--accent-gold)' }} />
                <span style={{ color: 'var(--text-secondary)' }}>
                  No API key stored. Will use env var ({selectedProvider.api_key_env}) or add key in{' '}
                  <a href="/settings" style={{ color: 'var(--accent-blue)' }}>Settings</a>
                </span>
              </>
            ) : (
              <>
                <CheckCircle size={14} style={{ color: 'var(--text-muted)' }} />
                <span style={{ color: 'var(--text-secondary)' }}>No API key required</span>
              </>
            )}
          </div>
        )}

        {/* Advanced Options */}
        <div style={{ marginBottom: '20px' }}>
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
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
            {showAdvanced ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            Advanced Options
          </button>

          {showAdvanced && (
            <div style={{ marginTop: '12px', paddingLeft: '18px' }}>
              <div style={{ marginBottom: '12px' }}>
                <label>System Prompt Override</label>
                <textarea
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                  disabled={generating}
                  rows={3}
                  placeholder="Leave empty to use default"
                />
              </div>
              <div>
                <label>Temperature: {temperature}</label>
                <input
                  type="range"
                  min={0}
                  max={1.5}
                  step={0.1}
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  disabled={generating}
                  style={{ width: '200px' }}
                />
              </div>
              <div style={{ marginTop: '12px' }}>
                <label>Max Tokens: {maxTokens}</label>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <input
                    type="range"
                    min={256}
                    max={8192}
                    step={256}
                    value={maxTokens}
                    onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                    disabled={generating}
                    style={{ width: '200px' }}
                  />
                  <input
                    type="number"
                    value={maxTokens}
                    onChange={(e) => setMaxTokens(Math.max(256, Math.min(8192, parseInt(e.target.value) || 256)))}
                    min={256}
                    max={8192}
                    disabled={generating}
                    style={{ width: '80px', fontSize: '13px', padding: '4px 8px' }}
                  />
                </div>
                <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                  Higher values produce longer, more detailed answers (default: 4096)
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Error */}
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

        {/* Generate button */}
        {!batchMode ? (
          <button
            className="btn btn-primary"
            onClick={handleGenerate}
            disabled={generating || (!isFromPapers && !topic.trim())}
            style={{ width: '100%', justifyContent: 'center', padding: '10px 20px', fontSize: '15px' }}
          >
            {generating ? <span className="spinner" /> : isFromPapers ? `Generate from ${paperIds.length} Papers` : 'Generate Dataset'}
          </button>
        ) : (
          <button
            className="btn btn-primary"
            onClick={handleBatchGenerate}
            disabled={generating || validBatchCount === 0}
            style={{ width: '100%', justifyContent: 'center', padding: '10px 20px', fontSize: '15px' }}
          >
            {generating ? <span className="spinner" /> : `Generate All (${validBatchCount} dataset${validBatchCount !== 1 ? 's' : ''})`}
          </button>
        )}
      </div>

      {/* Progress log (single mode only) */}
      {datasetId && !batchMode && (
        <div className="card">
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '12px',
          }}>
            <h3 style={{ fontSize: '14px', fontWeight: 500 }}>Progress</h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              {status && (
                <span className={`badge badge-${status}`}>
                  {status}
                </span>
              )}
              {generating && !done && (
                <button
                  className="btn btn-danger"
                  style={{ padding: '4px 10px', fontSize: '12px' }}
                  onClick={async () => {
                    if (datasetId) {
                      try {
                        await cancelDataset(datasetId)
                        setGenerating(false)
                      } catch { /* ignore */ }
                    }
                  }}
                >
                  <StopCircle size={14} />
                  Cancel
                </button>
              )}
            </div>
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
    </div>
  )
}
