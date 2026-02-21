import { create } from 'zustand'
import { login as apiLogin, register as apiRegister } from '../api/client'

interface AuthStore {
  isAuthenticated: boolean
  username: string | null
  loading: boolean
  error: string | null

  login: (username: string, password: string) => Promise<void>
  register: (username: string, password: string) => Promise<void>
  logout: () => void
  checkAuth: () => void
}

function parseJwt(token: string): Record<string, unknown> | null {
  try {
    const base64Url = token.split('.')[1]
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/')
    return JSON.parse(atob(base64))
  } catch {
    return null
  }
}

export const useAuthStore = create<AuthStore>((set) => ({
  isAuthenticated: !!localStorage.getItem('access_token'),
  username: (() => {
    const token = localStorage.getItem('access_token')
    if (!token) return null
    const payload = parseJwt(token)
    return (payload?.username as string) || null
  })(),
  loading: false,
  error: null,

  login: async (username, password) => {
    set({ loading: true, error: null })
    try {
      const res = await apiLogin(username, password)
      localStorage.setItem('access_token', res.access_token)
      localStorage.setItem('refresh_token', res.refresh_token)
      set({ isAuthenticated: true, username, loading: false })
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Login failed'
      set({ error: msg, loading: false })
      throw e
    }
  },

  register: async (username, password) => {
    set({ loading: true, error: null })
    try {
      const res = await apiRegister(username, password)
      localStorage.setItem('access_token', res.access_token)
      localStorage.setItem('refresh_token', res.refresh_token)
      set({ isAuthenticated: true, username, loading: false })
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Registration failed'
      set({ error: msg, loading: false })
      throw e
    }
  },

  logout: () => {
    localStorage.removeItem('access_token')
    localStorage.removeItem('refresh_token')
    set({ isAuthenticated: false, username: null })
    window.location.href = '/login'
  },

  checkAuth: () => {
    const token = localStorage.getItem('access_token')
    if (!token) {
      set({ isAuthenticated: false, username: null })
      return
    }
    const payload = parseJwt(token)
    if (!payload) {
      localStorage.removeItem('access_token')
      set({ isAuthenticated: false, username: null })
      return
    }
    // Check expiry
    const exp = payload.exp as number
    if (exp && exp * 1000 < Date.now()) {
      localStorage.removeItem('access_token')
      set({ isAuthenticated: false, username: null })
      return
    }
    set({ isAuthenticated: true, username: (payload.username as string) || null })
  },
}))
