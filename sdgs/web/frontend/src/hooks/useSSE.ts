import { useEffect, useRef, useState, useCallback } from 'react'
import { createDatasetSSE, SSEMessage } from '../api/sse'

export function useSSE(datasetId: number | null) {
  const [logs, setLogs] = useState<string[]>([])
  const [status, setStatus] = useState<string | null>(null)
  const [done, setDone] = useState(false)
  const closeRef = useRef<(() => void) | null>(null)

  useEffect(() => {
    if (!datasetId) return

    setLogs([])
    setStatus(null)
    setDone(false)

    const close = createDatasetSSE(
      datasetId,
      (msg: SSEMessage) => {
        if (msg.type === 'log') {
          setLogs(prev => [...prev, msg.data])
        } else if (msg.type === 'status') {
          setStatus(msg.data)
        } else if (msg.type === 'error') {
          setLogs(prev => [...prev, `ERROR: ${msg.data}`])
        }
      },
      () => setDone(true),
    )
    closeRef.current = close

    return () => close()
  }, [datasetId])

  const clear = useCallback(() => {
    setLogs([])
    setStatus(null)
    setDone(false)
  }, [])

  return { logs, status, done, clear }
}
