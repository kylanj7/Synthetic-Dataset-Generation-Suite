import { useEffect, useRef, useState, useCallback } from 'react'
import { createDatasetSSE, SSEMessage } from '../api/sse'

export function useSSE(datasetId: number | null) {
  const [logs, setLogs] = useState<string[]>([])
  const [status, setStatus] = useState<string | null>(null)
  const [done, setDone] = useState(false)
  const closeRef = useRef<(() => void) | null>(null)
  const prevIdRef = useRef<number | null>(null)

  useEffect(() => {
    if (!datasetId) return

    // Only clear logs when connecting to a different dataset
    if (prevIdRef.current !== datasetId) {
      setLogs([])
      setStatus(null)
      setDone(false)
      prevIdRef.current = datasetId
    }

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
