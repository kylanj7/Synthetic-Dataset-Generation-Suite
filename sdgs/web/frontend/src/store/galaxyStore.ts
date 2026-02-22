import { create } from 'zustand'
import { getGalaxyData, getPaperDetail, GalaxyData, PaperDetail } from '../api/client'

interface GalaxyStore {
  data: GalaxyData | null
  selectedPaper: PaperDetail | null
  expandedPaperGraphId: string | null
  selectedDatasetNode: any | null
  loading: boolean
  error: string | null
  searchQuery: string
  activeCluster: number | null

  fetchData: () => Promise<void>
  selectPaper: (paperId: number, graphId: string) => Promise<void>
  selectDatasetNode: (node: any) => void
  clearSelection: () => void
  setSearchQuery: (q: string) => void
  setActiveCluster: (id: number | null) => void
}

export const useGalaxyStore = create<GalaxyStore>((set) => ({
  data: null,
  selectedPaper: null,
  expandedPaperGraphId: null,
  selectedDatasetNode: null,
  loading: false,
  error: null,
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

  selectPaper: async (paperId: number, graphId: string) => {
    set({ expandedPaperGraphId: graphId, selectedDatasetNode: null })
    try {
      const detail = await getPaperDetail(paperId)
      set({ selectedPaper: detail })
    } catch (e) {
      set({ error: String(e) })
    }
  },

  selectDatasetNode: (node) => set({
    selectedDatasetNode: node,
    selectedPaper: null,
    expandedPaperGraphId: null,
  }),

  clearSelection: () => set({
    selectedPaper: null,
    expandedPaperGraphId: null,
    selectedDatasetNode: null,
  }),
  setSearchQuery: (q) => set({ searchQuery: q }),
  setActiveCluster: (id) => set({ activeCluster: id }),
}))
