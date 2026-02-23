import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { mergeConvert } from '../api/client'

const QUANT_METHODS = ['q2_k', 'q3_k_m', 'q4_k_s', 'q4_k_m', 'q5_k_m', 'q6_k', 'q8_0']

export default function MergeConvert() {
  const navigate = useNavigate()
  const [adapterPath, setAdapterPath] = useState('')
  const [baseModel, setBaseModel] = useState('')
  const [quantMethod, setQuantMethod] = useState('q4_k_m')
  const [outputName, setOutputName] = useState('')
  const [keepMerged, setKeepMerged] = useState(false)

  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const [ggufPath, setGgufPath] = useState<string | null>(null)

  const handleSubmit = async () => {
    setError('')
    setSubmitting(true)
    try {
      const res = await mergeConvert({
        adapter_path: adapterPath.trim(),
        base_model: baseModel.trim() || undefined,
        quant_method: quantMethod,
        output_name: outputName.trim() || undefined,
        keep_merged: keepMerged,
      })
      setGgufPath(res.gguf_path)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Merge/convert failed')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div style={{ maxWidth: '700px' }}>
      <div className="page-header">
        <h1>Merge & Convert</h1>
        <p>Merge LoRA adapter into base model and convert to GGUF</p>
      </div>

      {!ggufPath ? (
        <div className="card">
          <div style={{ marginBottom: '16px' }}>
            <label>Adapter Path *</label>
            <input
              type="text"
              placeholder="/path/to/lora-adapter"
              value={adapterPath}
              onChange={(e) => setAdapterPath(e.target.value)}
              disabled={submitting}
            />
          </div>

          <div style={{ marginBottom: '16px' }}>
            <label>Base Model</label>
            <input
              type="text"
              placeholder="Auto-detected from adapter"
              value={baseModel}
              onChange={(e) => setBaseModel(e.target.value)}
              disabled={submitting}
            />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
            <div>
              <label>Quant Method</label>
              <select
                value={quantMethod}
                onChange={(e) => setQuantMethod(e.target.value)}
                disabled={submitting}
              >
                {QUANT_METHODS.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </div>
            <div>
              <label>Output Name</label>
              <input
                type="text"
                placeholder="Optional"
                value={outputName}
                onChange={(e) => setOutputName(e.target.value)}
                disabled={submitting}
              />
            </div>
          </div>

          <div style={{ marginBottom: '20px' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={keepMerged}
                onChange={(e) => setKeepMerged(e.target.checked)}
                disabled={submitting}
                style={{ width: 'auto' }}
              />
              Keep merged model directory
            </label>
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
            disabled={submitting || !adapterPath.trim()}
            style={{ width: '100%', justifyContent: 'center', padding: '10px 20px', fontSize: '15px' }}
          >
            {submitting ? <span className="spinner" /> : 'Start Merge & Convert'}
          </button>
        </div>
      ) : (
        <div className="card">
          <div style={{ marginBottom: '16px' }}>
            <div style={{ fontSize: '14px', fontWeight: 500, color: 'var(--accent-green)', marginBottom: '8px' }}>
              Conversion Complete
            </div>
            <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
              GGUF saved to: <code style={{ color: 'var(--accent-cyan)' }}>{ggufPath}</code>
            </div>
          </div>
          <button
            className="btn btn-primary"
            onClick={() => navigate(`/training/push?gguf_path=${encodeURIComponent(ggufPath)}`)}
            style={{ fontSize: '14px' }}
          >
            Push to HuggingFace
          </button>
        </div>
      )}
    </div>
  )
}
