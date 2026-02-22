import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { ChevronDown, ChevronRight, StopCircle } from 'lucide-react'
import { getDatasets, startTraining, cancelTraining, Dataset } from '../api/client'
import { useTrainingSSE } from '../hooks/useTrainingSSE'

export default function StartTraining() {
  const navigate = useNavigate()
  const logViewerRef = useRef<HTMLDivElement>(null)

  // Dataset selection
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [datasetSource, setDatasetSource] = useState<'dataset' | 'path'>('dataset')
  const [datasetId, setDatasetId] = useState<number | null>(null)
  const [datasetPath, setDatasetPath] = useState('')

  // Model config
  const [baseModel, setBaseModel] = useState('Qwen/Qwen2.5-14B-Instruct')
  const [modelSize, setModelSize] = useState('14B')

  // LoRA config
  const [loraRank, setLoraRank] = useState(16)
  const [loraAlpha, setLoraAlpha] = useState(16)

  // Training config
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [learningRate, setLearningRate] = useState(0.00005)
  const [numEpochs, setNumEpochs] = useState(1)
  const [batchSize, setBatchSize] = useState(4)
  const [gradAccumSteps, setGradAccumSteps] = useState(4)
  const [maxSteps, setMaxSteps] = useState(-1)

  // State
  const [submitting, setSubmitting] = useState(false)
  const [runId, setRunId] = useState<number | null>(null)
  const [error, setError] = useState('')

  const { logs, status, done } = useTrainingSSE(runId)

  useEffect(() => {
    getDatasets(1).then((res) => {
      setDatasets(res.datasets.filter((d) => d.status === 'completed'))
    }).catch(() => {})
  }, [])

  useEffect(() => {
    if (logViewerRef.current) {
      logViewerRef.current.scrollTop = logViewerRef.current.scrollHeight
    }
  }, [logs])

  useEffect(() => {
    if (done && status === 'completed' && runId) {
      setTimeout(() => navigate(`/training/${runId}`), 1000)
    }
  }, [done, status, runId, navigate])

  const handleStart = async () => {
    setError('')
    setSubmitting(true)
    try {
      const run = await startTraining({
        dataset_id: datasetSource === 'dataset' ? datasetId : undefined,
        dataset_path: datasetSource === 'path' ? datasetPath.trim() || undefined : undefined,
        base_model: baseModel,
        model_size: modelSize,
        lora_rank: loraRank,
        lora_alpha: loraAlpha,
        learning_rate: learningRate,
        num_epochs: numEpochs,
        batch_size: batchSize,
        gradient_accumulation_steps: gradAccumSteps,
        max_steps: maxSteps,
      })
      setRunId(run.id)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to start training')
      setSubmitting(false)
    }
  }

  const canSubmit = !submitting && (
    (datasetSource === 'dataset' && datasetId != null) ||
    (datasetSource === 'path' && datasetPath.trim())
  )

  return (
    <div style={{ maxWidth: '700px' }}>
      <div className="page-header">
        <h1>Start Training</h1>
        <p>Configure and launch a fine-tuning run</p>
      </div>

      <div className="card" style={{ marginBottom: '20px' }}>
        {/* Dataset source */}
        <div style={{ marginBottom: '20px' }}>
          <label style={{ fontSize: '15px', fontWeight: 500, color: 'var(--text-primary)', marginBottom: '8px' }}>
            Dataset Source
          </label>
          <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
            <button
              onClick={() => setDatasetSource('dataset')}
              disabled={submitting}
              style={{
                background: datasetSource === 'dataset' ? 'var(--accent-blue)' : 'var(--bg-tertiary)',
                border: '1px solid ' + (datasetSource === 'dataset' ? 'var(--accent-blue)' : 'var(--border-primary)'),
                color: datasetSource === 'dataset' ? '#fff' : 'var(--text-secondary)',
                cursor: 'pointer',
                fontSize: '13px',
                padding: '6px 12px',
                borderRadius: 'var(--radius-sm)',
              }}
            >
              From Dataset
            </button>
            <button
              onClick={() => setDatasetSource('path')}
              disabled={submitting}
              style={{
                background: datasetSource === 'path' ? 'var(--accent-blue)' : 'var(--bg-tertiary)',
                border: '1px solid ' + (datasetSource === 'path' ? 'var(--accent-blue)' : 'var(--border-primary)'),
                color: datasetSource === 'path' ? '#fff' : 'var(--text-secondary)',
                cursor: 'pointer',
                fontSize: '13px',
                padding: '6px 12px',
                borderRadius: 'var(--radius-sm)',
              }}
            >
              Manual Path
            </button>
          </div>

          {datasetSource === 'dataset' ? (
            <select
              value={datasetId ?? ''}
              onChange={(e) => setDatasetId(e.target.value ? Number(e.target.value) : null)}
              disabled={submitting}
            >
              <option value="">Select a completed dataset...</option>
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name || d.topic} ({d.actual_size} samples)
                </option>
              ))}
            </select>
          ) : (
            <input
              type="text"
              placeholder="/path/to/train.jsonl"
              value={datasetPath}
              onChange={(e) => setDatasetPath(e.target.value)}
              disabled={submitting}
              style={{ fontSize: '14px' }}
            />
          )}
        </div>

        {/* Model config */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px' }}>
          <div>
            <label>Base Model</label>
            <input
              type="text"
              value={baseModel}
              onChange={(e) => setBaseModel(e.target.value)}
              disabled={submitting}
            />
          </div>
          <div>
            <label>Model Size</label>
            <input
              type="text"
              value={modelSize}
              onChange={(e) => setModelSize(e.target.value)}
              disabled={submitting}
            />
          </div>
        </div>

        {/* LoRA config */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px' }}>
          <div>
            <label>LoRA Rank</label>
            <input
              type="number"
              value={loraRank}
              onChange={(e) => setLoraRank(Math.max(1, parseInt(e.target.value) || 1))}
              min={1}
              disabled={submitting}
            />
          </div>
          <div>
            <label>LoRA Alpha</label>
            <input
              type="number"
              value={loraAlpha}
              onChange={(e) => setLoraAlpha(Math.max(1, parseInt(e.target.value) || 1))}
              min={1}
              disabled={submitting}
            />
          </div>
        </div>

        {/* Advanced training config */}
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
            Training Parameters
          </button>

          {showAdvanced && (
            <div style={{ marginTop: '12px', paddingLeft: '18px' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                <div>
                  <label>Learning Rate</label>
                  <input
                    type="number"
                    value={learningRate}
                    onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.00005)}
                    step={0.00001}
                    disabled={submitting}
                  />
                </div>
                <div>
                  <label>Epochs</label>
                  <input
                    type="number"
                    value={numEpochs}
                    onChange={(e) => setNumEpochs(Math.max(1, parseInt(e.target.value) || 1))}
                    min={1}
                    disabled={submitting}
                  />
                </div>
                <div>
                  <label>Batch Size</label>
                  <input
                    type="number"
                    value={batchSize}
                    onChange={(e) => setBatchSize(Math.max(1, parseInt(e.target.value) || 1))}
                    min={1}
                    disabled={submitting}
                  />
                </div>
                <div>
                  <label>Gradient Accumulation Steps</label>
                  <input
                    type="number"
                    value={gradAccumSteps}
                    onChange={(e) => setGradAccumSteps(Math.max(1, parseInt(e.target.value) || 1))}
                    min={1}
                    disabled={submitting}
                  />
                </div>
              </div>
              <div style={{ marginTop: '12px' }}>
                <label>Max Steps (-1 for unlimited)</label>
                <input
                  type="number"
                  value={maxSteps}
                  onChange={(e) => setMaxSteps(parseInt(e.target.value) || -1)}
                  disabled={submitting}
                  style={{ width: '120px' }}
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

        {/* Submit */}
        <button
          className="btn btn-primary"
          onClick={handleStart}
          disabled={!canSubmit}
          style={{ width: '100%', justifyContent: 'center', padding: '10px 20px', fontSize: '15px' }}
        >
          {submitting ? <span className="spinner" /> : 'Start Training'}
        </button>
      </div>

      {/* Progress log */}
      {runId && (
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
              {submitting && !done && (
                <button
                  className="btn btn-danger"
                  style={{ padding: '4px 10px', fontSize: '12px' }}
                  onClick={async () => {
                    try {
                      await cancelTraining(runId)
                      setSubmitting(false)
                    } catch { /* ignore */ }
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
              <div style={{ color: 'var(--text-muted)' }}>Waiting for training to start...</div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
