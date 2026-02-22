import { create } from 'zustand'
import {
  getTrainingRuns, getTrainingRun,
  getEvaluations, getEvaluationDetail,
  TrainingRun, EvaluationRun, EvaluationDetail,
} from '../api/client'

interface TrainingStore {
  trainingRuns: TrainingRun[]
  total: number
  page: number
  currentRun: TrainingRun | null
  evaluations: EvaluationRun[]
  evalsTotal: number
  evalsPage: number
  currentEval: EvaluationDetail | null
  loading: boolean
  error: string | null

  fetchTrainingRuns: (page?: number) => Promise<void>
  fetchTrainingRun: (id: number) => Promise<void>
  updateTrainingRun: (run: TrainingRun) => void
  fetchEvaluations: (page?: number) => Promise<void>
  fetchEvaluation: (id: number) => Promise<void>
}

export const useTrainingStore = create<TrainingStore>((set) => ({
  trainingRuns: [],
  total: 0,
  page: 1,
  currentRun: null,
  evaluations: [],
  evalsTotal: 0,
  evalsPage: 1,
  currentEval: null,
  loading: false,
  error: null,

  fetchTrainingRuns: async (page = 1) => {
    set({ loading: true, error: null })
    try {
      const res = await getTrainingRuns(page)
      set({ trainingRuns: res.runs, total: res.total, page: res.page, loading: false })
    } catch (e) {
      set({ error: String(e), loading: false })
    }
  },

  fetchTrainingRun: async (id: number) => {
    set({ loading: true, error: null })
    try {
      const run = await getTrainingRun(id)
      set({ currentRun: run, loading: false })
    } catch (e) {
      set({ error: String(e), loading: false })
    }
  },

  updateTrainingRun: (run: TrainingRun) => {
    set((state) => ({
      currentRun: run,
      trainingRuns: state.trainingRuns.map((r) => (r.id === run.id ? run : r)),
    }))
  },

  fetchEvaluations: async (page = 1) => {
    set({ loading: true, error: null })
    try {
      const res = await getEvaluations(page)
      set({ evaluations: res.evaluations, evalsTotal: res.total, evalsPage: res.page, loading: false })
    } catch (e) {
      set({ error: String(e), loading: false })
    }
  },

  fetchEvaluation: async (id: number) => {
    set({ loading: true, error: null })
    try {
      const detail = await getEvaluationDetail(id)
      set({ currentEval: detail, loading: false })
    } catch (e) {
      set({ error: String(e), loading: false })
    }
  },
}))
