import { useEffect, useState } from 'react'
import { Key, Trash2, Plus, Check } from 'lucide-react'
import {
  getProviders, getApiKeys, saveApiKey, deleteApiKey,
  getHFTokenStatus, saveHFToken, deleteHFToken,
  getS2TokenStatus, saveS2Token, deleteS2Token,
  ProviderInfo, ApiKeyInfo,
} from '../api/client'

export default function Settings() {
  const [providers, setProviders] = useState<ProviderInfo[]>([])
  const [apiKeys, setApiKeys] = useState<ApiKeyInfo[]>([])
  const [hfConfigured, setHfConfigured] = useState(false)
  const [editingProvider, setEditingProvider] = useState<string | null>(null)
  const [editingKey, setEditingKey] = useState('')
  const [editingHF, setEditingHF] = useState(false)
  const [hfToken, setHfToken] = useState('')
  const [s2Configured, setS2Configured] = useState(false)
  const [editingS2, setEditingS2] = useState(false)
  const [s2Token, setS2Token] = useState('')
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState('')

  const refresh = async () => {
    try {
      const [p, k, hf, s2] = await Promise.all([
        getProviders(),
        getApiKeys(),
        getHFTokenStatus(),
        getS2TokenStatus(),
      ])
      setProviders(p)
      setApiKeys(k)
      setHfConfigured(hf.configured)
      setS2Configured(s2.configured)
    } catch {
      // ignore
    }
  }

  useEffect(() => {
    refresh()
  }, [])

  const handleSaveKey = async (provider: string) => {
    if (!editingKey.trim()) return
    setSaving(true)
    setMessage('')
    try {
      await saveApiKey(provider, editingKey.trim())
      setEditingProvider(null)
      setEditingKey('')
      setMessage(`API key for ${provider} saved`)
      await refresh()
    } catch {
      setMessage('Failed to save API key')
    }
    setSaving(false)
  }

  const handleDeleteKey = async (provider: string) => {
    setSaving(true)
    try {
      await deleteApiKey(provider)
      setMessage(`API key for ${provider} removed`)
      await refresh()
    } catch {
      setMessage('Failed to delete API key')
    }
    setSaving(false)
  }

  const handleSaveHFToken = async () => {
    if (!hfToken.trim()) return
    setSaving(true)
    setMessage('')
    try {
      await saveHFToken(hfToken.trim())
      setEditingHF(false)
      setHfToken('')
      setMessage('HuggingFace token saved')
      await refresh()
    } catch {
      setMessage('Failed to save HuggingFace token')
    }
    setSaving(false)
  }

  const handleDeleteHFToken = async () => {
    setSaving(true)
    try {
      await deleteHFToken()
      setMessage('HuggingFace token removed')
      await refresh()
    } catch {
      setMessage('Failed to remove HuggingFace token')
    }
    setSaving(false)
  }

  const handleSaveS2Token = async () => {
    if (!s2Token.trim()) return
    setSaving(true)
    setMessage('')
    try {
      await saveS2Token(s2Token.trim())
      setEditingS2(false)
      setS2Token('')
      setMessage('Semantic Scholar API key saved')
      await refresh()
    } catch {
      setMessage('Failed to save Semantic Scholar API key')
    }
    setSaving(false)
  }

  const handleDeleteS2Token = async () => {
    setSaving(true)
    try {
      await deleteS2Token()
      setMessage('Semantic Scholar API key removed')
      await refresh()
    } catch {
      setMessage('Failed to remove Semantic Scholar API key')
    }
    setSaving(false)
  }

  const keyMap = Object.fromEntries(apiKeys.map((k) => [k.provider_name, k]))

  return (
    <div style={{ maxWidth: '700px' }}>
      <div className="page-header">
        <h1>Settings</h1>
        <p>Manage API keys and integrations</p>
      </div>

      {message && (
        <div style={{
          background: 'rgba(110, 231, 216, 0.1)',
          border: '1px solid rgba(110, 231, 216, 0.3)',
          borderRadius: 'var(--radius-sm)',
          padding: '8px 12px',
          color: 'var(--accent-cyan)',
          fontSize: '13px',
          marginBottom: '16px',
        }}>
          {message}
        </div>
      )}

      {/* Provider API Keys */}
      <div className="card" style={{ marginBottom: '20px' }}>
        <h2 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Key size={18} />
          Provider API Keys
        </h2>
        <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '16px' }}>
          API keys are encrypted and stored securely. Only the last 4 characters are visible.
        </p>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {providers.map((p) => {
            const stored = keyMap[p.name]
            const isEditing = editingProvider === p.name

            return (
              <div
                key={p.name}
                style={{
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '12px 16px',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ fontWeight: 500, fontSize: '14px' }}>{p.name}</div>
                    <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                      Default model: {p.default_model}
                      {p.api_key_env && ` — env: ${p.api_key_env}`}
                    </div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    {stored ? (
                      <>
                        <span style={{ fontSize: '13px', color: 'var(--accent-green)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                          <Check size={14} /> {stored.masked_key}
                        </span>
                        <button
                          className="btn"
                          style={{ padding: '4px 8px' }}
                          onClick={() => { setEditingProvider(p.name); setEditingKey('') }}
                        >
                          Edit
                        </button>
                        <button
                          className="btn btn-danger"
                          style={{ padding: '4px 8px' }}
                          onClick={() => handleDeleteKey(p.name)}
                          disabled={saving}
                        >
                          <Trash2 size={14} />
                        </button>
                      </>
                    ) : (
                      <button
                        className="btn"
                        style={{ padding: '4px 8px' }}
                        onClick={() => { setEditingProvider(p.name); setEditingKey('') }}
                      >
                        <Plus size={14} /> Add Key
                      </button>
                    )}
                  </div>
                </div>

                {isEditing && (
                  <div style={{ marginTop: '12px', display: 'flex', gap: '8px' }}>
                    <input
                      type="password"
                      placeholder="Enter API key"
                      value={editingKey}
                      onChange={(e) => setEditingKey(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleSaveKey(p.name)}
                      autoFocus
                    />
                    <button
                      className="btn btn-primary"
                      onClick={() => handleSaveKey(p.name)}
                      disabled={saving || !editingKey.trim()}
                    >
                      Save
                    </button>
                    <button className="btn" onClick={() => setEditingProvider(null)}>
                      Cancel
                    </button>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* HuggingFace Token */}
      <div className="card">
        <h2 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px' }}>
          HuggingFace Token
        </h2>
        <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '16px' }}>
          Required to push datasets to HuggingFace Hub. Get your token from{' '}
          <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener noreferrer">
            huggingface.co/settings/tokens
          </a>
        </p>

        {hfConfigured && !editingHF ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span style={{ fontSize: '13px', color: 'var(--accent-green)', display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Check size={14} /> Token configured
            </span>
            <button className="btn" style={{ padding: '4px 8px' }} onClick={() => setEditingHF(true)}>
              Update
            </button>
            <button
              className="btn btn-danger"
              style={{ padding: '4px 8px' }}
              onClick={handleDeleteHFToken}
              disabled={saving}
            >
              <Trash2 size={14} /> Remove
            </button>
          </div>
        ) : (
          <div style={{ display: 'flex', gap: '8px' }}>
            <input
              type="password"
              placeholder="hf_..."
              value={hfToken}
              onChange={(e) => setHfToken(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSaveHFToken()}
              autoFocus={editingHF}
            />
            <button
              className="btn btn-primary"
              onClick={handleSaveHFToken}
              disabled={saving || !hfToken.trim()}
            >
              Save
            </button>
            {editingHF && (
              <button className="btn" onClick={() => setEditingHF(false)}>
                Cancel
              </button>
            )}
          </div>
        )}
      </div>

      {/* Semantic Scholar API Key */}
      <div className="card" style={{ marginTop: '20px' }}>
        <h2 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px' }}>
          Semantic Scholar API Key
        </h2>
        <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '16px' }}>
          Improves paper search with higher rate limits. Get your key from{' '}
          <a href="https://www.semanticscholar.org/product/api#api-key-form" target="_blank" rel="noopener noreferrer">
            semanticscholar.org/product/api
          </a>
        </p>

        {s2Configured && !editingS2 ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span style={{ fontSize: '13px', color: 'var(--accent-green)', display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Check size={14} /> API key configured
            </span>
            <button className="btn" style={{ padding: '4px 8px' }} onClick={() => setEditingS2(true)}>
              Update
            </button>
            <button
              className="btn btn-danger"
              style={{ padding: '4px 8px' }}
              onClick={handleDeleteS2Token}
              disabled={saving}
            >
              <Trash2 size={14} /> Remove
            </button>
          </div>
        ) : (
          <div style={{ display: 'flex', gap: '8px' }}>
            <input
              type="password"
              placeholder="Enter Semantic Scholar API key"
              value={s2Token}
              onChange={(e) => setS2Token(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSaveS2Token()}
              autoFocus={editingS2}
            />
            <button
              className="btn btn-primary"
              onClick={handleSaveS2Token}
              disabled={saving || !s2Token.trim()}
            >
              Save
            </button>
            {editingS2 && (
              <button className="btn" onClick={() => setEditingS2(false)}>
                Cancel
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
