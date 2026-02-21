const BASE_URL = '/api'

class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

function getAuthHeader(): Record<string, string> {
  const token = localStorage.getItem('access_token')
  return token ? { Authorization: `Bearer ${token}` } : {}
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeader(),
      ...options?.headers,
    },
    ...options,
  })
  if (!res.ok) {
    if (res.status === 401) {
      localStorage.removeItem('access_token')
      localStorage.removeItem('refresh_token')
      window.location.href = '/login'
    }
    const body = await res.text()
    throw new ApiError(body, res.status)
  }
  return res.json()
}

// Auth
export interface TokenResponse {
  access_token: string
  refresh_token: string
  token_type: string
}

export const register = (username: string, password: string) =>
  request<TokenResponse>('/auth/register', {
    method: 'POST',
    body: JSON.stringify({ username, password }),
  })

export const login = (username: string, password: string) =>
  request<TokenResponse>('/auth/login', {
    method: 'POST',
    body: JSON.stringify({ username, password }),
  })

export const refreshToken = (refresh_token: string) =>
  request<TokenResponse>('/auth/refresh', {
    method: 'POST',
    body: JSON.stringify({ refresh_token }),
  })

// Datasets
export interface Dataset {
  id: number
  name: string
  topic: string
  status: string
  provider: string | null
  model: string | null
  target_size: number
  actual_size: number
  valid_count: number
  invalid_count: number
  healed_count: number
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  gpu_kwh: number
  duration_seconds: number
  output_path: string | null
  system_prompt: string | null
  temperature: number
  hf_repo: string | null
  created_at: string | null
  started_at: string | null
  completed_at: string | null
  error_message: string | null
}

export interface DatasetListResponse {
  datasets: Dataset[]
  total: number
  page: number
  per_page: number
}

export interface CreateDatasetRequest {
  topic: string
  provider?: string | null
  model?: string | null
  target_size?: number
  system_prompt?: string | null
  temperature?: number
}

export const createDataset = (data: CreateDatasetRequest) =>
  request<Dataset>('/datasets', { method: 'POST', body: JSON.stringify(data) })

export const getDatasets = (page = 1) =>
  request<DatasetListResponse>(`/datasets?page=${page}`)

export const getDataset = (id: number) =>
  request<Dataset>(`/datasets/${id}`)

export const cancelDataset = (id: number) =>
  request<{ status: string }>(`/datasets/${id}/cancel`, { method: 'POST' })

export interface QAPair {
  id: number | null
  instruction: string
  output: string
  is_valid: boolean
  was_healed: boolean
  source_title: string | null
  think_text: string | null
  answer_text: string | null
}

export interface DatasetSamplesResponse {
  samples: QAPair[]
  total: number
  page: number
  per_page: number
}

export const getDatasetSamples = (id: number, page = 1, search?: string) => {
  const params = new URLSearchParams({ page: String(page) })
  if (search) params.set('search', search)
  return request<DatasetSamplesResponse>(`/datasets/${id}/samples?${params}`)
}

// HuggingFace Push
export interface HFPushResponse {
  repo_url: string
  hf_repo: string
}

export const pushToHuggingFace = (id: number, data: {
  repo_name: string
  description?: string
  private?: boolean
}) =>
  request<HFPushResponse>(`/datasets/${id}/push-hf`, {
    method: 'POST',
    body: JSON.stringify(data),
  })

// Providers
export interface ProviderInfo {
  name: string
  default_model: string
  api_key_env: string | null
  has_key: boolean
}

export const getProviders = () => request<ProviderInfo[]>('/providers')

// Settings - API Keys
export interface ApiKeyInfo {
  provider_name: string
  masked_key: string
  updated_at: string | null
}

export const getApiKeys = () => request<ApiKeyInfo[]>('/settings/keys')

export const saveApiKey = (provider: string, api_key: string) =>
  request<{ status: string }>(`/settings/keys/${provider}`, {
    method: 'PUT',
    body: JSON.stringify({ api_key }),
  })

export const deleteApiKey = (provider: string) =>
  request<{ status: string }>(`/settings/keys/${provider}`, { method: 'DELETE' })

// Settings - HF Token
export interface HFTokenStatus {
  configured: boolean
}

export const getHFTokenStatus = () => request<HFTokenStatus>('/settings/hf-token')

export const saveHFToken = (token: string) =>
  request<{ status: string }>('/settings/hf-token', {
    method: 'PUT',
    body: JSON.stringify({ token }),
  })

export const deleteHFToken = () =>
  request<{ status: string }>('/settings/hf-token', { method: 'DELETE' })

// Galaxy
export interface GalaxyNode {
  id: string
  type: string
  label: string
  size: number
  color: string
  cluster: number
  year?: number | null
  citation_count?: number | null
  abstract?: string | null
  authors?: string[] | null
  url?: string | null
  instruction?: string | null
  output_preview?: string | null
}

export interface GalaxyLink {
  source: string
  target: string
  weight: number
  type: string
}

export interface ClusterInfo {
  id: number
  label: string
  color: string
  paper_count: number
}

export interface GalaxyData {
  nodes: GalaxyNode[]
  links: GalaxyLink[]
  clusters: ClusterInfo[]
}

export interface PaperDetail {
  paper_id: string
  title: string
  authors: string[]
  abstract: string | null
  year: number | null
  citation_count: number
  url: string | null
  qa_pairs: QAPair[]
}

export const getGalaxyData = () => request<GalaxyData>('/galaxy/data')
export const getPaperDetail = (paperId: number) =>
  request<PaperDetail>(`/galaxy/paper/${paperId}`)

// Health
export const getHealth = () => request<{ status: string }>('/health')
