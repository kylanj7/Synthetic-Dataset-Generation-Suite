import { useEffect, useRef, useState, useCallback } from 'react'
import { createTrainingSSE, SSEMessage } from '../api/sse'

export function useTrainingSSE(runId: number | null) {
  const [logs, setLogs] = useState<string[]>([])
  const [status, setStatus] = useState<string | null>(null)
  const [done, setDone] = useState(false)
  const closeRef = useRef<(() => void) | null>(null)
  const prevIdRef = useRef<number | null>(null)

  useEffect(() => {
    if (!runId) return

    if (prevIdRef.current !== runId) {
      setLogs([])
      setStatus(null)
      setDone(false)
      prevIdRef.current = runId
    }

    const close = createTrainingSSE(
      runId,
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
  }, [runId])

  const clear = useCallback(() => {
    setLogs([])
    setStatus(null)
    setDone(false)
  }, [])

  return { logs, status, done, clear }
}
