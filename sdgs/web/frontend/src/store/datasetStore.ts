import { create } from 'zustand'
import {
  getDatasets, getDataset, getDatasetSamples, createDataset,
  Dataset, QAPair, CreateDatasetRequest,
} from '../api/client'

interface DatasetStore {
  datasets: Dataset[]
  total: number
  page: number
  currentDataset: Dataset | null
  samples: QAPair[]
  samplesTotal: number
  samplesPage: number
  loading: boolean
  error: string | null

  fetchDatasets: (page?: number) => Promise<void>
  fetchDataset: (id: number) => Promise<void>
  createDataset: (data: CreateDatasetRequest) => Promise<Dataset>
  fetchSamples: (id: number, page?: number, search?: string) => Promise<void>
  updateDataset: (dataset: Dataset) => void
}

export const useDatasetStore = create<DatasetStore>((set) => ({
  datasets: [],
  total: 0,
  page: 1,
  currentDataset: null,
  samples: [],
  samplesTotal: 0,
  samplesPage: 1,
  loading: false,
  error: null,

  fetchDatasets: async (page = 1) => {
    set({ loading: true, error: null })
    try {
      const res = await getDatasets(page)
      set({ datasets: res.datasets, total: res.total, page: res.page, loading: false })
    } catch (e) {
      set({ error: String(e), loading: false })
    }
  },

  fetchDataset: async (id: number) => {
    set({ loading: true, error: null })
    try {
      const dataset = await getDataset(id)
      set({ currentDataset: dataset, loading: false })
    } catch (e) {
      set({ error: String(e), loading: false })
    }
  },

  createDataset: async (data: CreateDatasetRequest) => {
    set({ loading: true, error: null })
    try {
      const dataset = await createDataset(data)
      set((state) => ({
        datasets: [dataset, ...state.datasets],
        loading: false,
      }))
      return dataset
    } catch (e) {
      set({ error: String(e), loading: false })
      throw e
    }
  },

  fetchSamples: async (id: number, page = 1, search?: string) => {
    set({ loading: true, error: null })
    try {
      const res = await getDatasetSamples(id, page, search)
      set({
        samples: res.samples,
        samplesTotal: res.total,
        samplesPage: res.page,
        loading: false,
      })
    } catch (e) {
      set({ error: String(e), loading: false })
    }
  },

  updateDataset: (dataset: Dataset) => {
    set((state) => ({
      currentDataset: dataset,
      datasets: state.datasets.map((d) => (d.id === dataset.id ? dataset : d)),
    }))
  },
}))
