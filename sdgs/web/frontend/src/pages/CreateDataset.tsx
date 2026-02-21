import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { ChevronDown, ChevronRight, CheckCircle, XCircle } from 'lucide-react'
import { getProviders, ProviderInfo } from '../api/client'
import { useDatasetStore } from '../store/datasetStore'
import { useSSE } from '../hooks/useSSE'

export default function CreateDataset() {
  const [topic, setTopic] = useState('')
  const [targetSize, setTargetSize] = useState(100)
  const [provider, setProvider] = useState<string>('')
  const [model, setModel] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [systemPrompt, setSystemPrompt] = useState('')
  const [temperature, setTemperature] = useState(0.7)
  const [providers, setProviders] = useState<ProviderInfo[]>([])
  const [generating, setGenerating] = useState(false)
  const [datasetId, setDatasetId] = useState<number | null>(null)
  const [error, setError] = useState('')
  const logViewerRef = useRef<HTMLDivElement>(null)

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
      })
      setDatasetId(ds.id)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to create dataset')
      setGenerating(false)
    }
  }

  return (
    <div style={{ maxWidth: '700px' }}>
      <div className="page-header">
        <h1>Create Dataset</h1>
        <p>Generate a synthetic Q&A dataset from academic papers</p>
      </div>

      <div className="card" style={{ marginBottom: '20px' }}>
        {/* Topic */}
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

        {/* Target size, Provider, Model */}
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
              <option value="">(default)</option>
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
        <button
          className="btn btn-primary"
          onClick={handleGenerate}
          disabled={generating || !topic.trim()}
          style={{ width: '100%', justifyContent: 'center', padding: '10px 20px', fontSize: '15px' }}
        >
          {generating ? <span className="spinner" /> : 'Generate Dataset'}
        </button>
      </div>

      {/* Progress log */}
      {datasetId && (
        <div className="card">
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '12px',
          }}>
            <h3 style={{ fontSize: '14px', fontWeight: 500 }}>Progress</h3>
            {status && (
              <span className={`badge badge-${status}`}>
                {status}
              </span>
            )}
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
