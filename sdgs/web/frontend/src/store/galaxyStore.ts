import { create } from 'zustand'
import { getGalaxyData, getPaperDetail, GalaxyData, PaperDetail } from '../api/client'

interface GalaxyStore {
  data: GalaxyData | null
  selectedPaper: PaperDetail | null
  loading: boolean
  error: string | null
  showQA: boolean
  searchQuery: string
  activeCluster: number | null

  fetchData: () => Promise<void>
  selectPaper: (paperId: number) => Promise<void>
  clearSelection: () => void
  setShowQA: (show: boolean) => void
  setSearchQuery: (q: string) => void
  setActiveCluster: (id: number | null) => void
}

export const useGalaxyStore = create<GalaxyStore>((set) => ({
  data: null,
  selectedPaper: null,
  loading: false,
  error: null,
  showQA: true,
  searchQuery: '',
  activeCluster: null,

  fetchData: async () => {
    set({ loading: true, error: null })
    try {
      const data = await getGalaxyData()
      set({ data, loading: false })
    } catch (e) {
      set({ error: String(e), loading: false })
    }
  },

  selectPaper: async (paperId: number) => {
    try {
      const detail = await getPaperDetail(paperId)
      set({ selectedPaper: detail })
    } catch (e) {
      set({ error: String(e) })
    }
  },

  clearSelection: () => set({ selectedPaper: null }),
  setShowQA: (show) => set({ showQA: show }),
  setSearchQuery: (q) => set({ searchQuery: q }),
  setActiveCluster: (id) => set({ activeCluster: id }),
}))
