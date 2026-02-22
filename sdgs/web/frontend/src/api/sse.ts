export interface SSEMessage {
  type: 'log' | 'status' | 'error' | 'done'
  data: string
}

export function createDatasetSSE(
  datasetId: number,
  onMessage: (msg: SSEMessage) => void,
  onDone?: () => void,
): () => void {
  let lastEventId = 0
  let reconnectTimer: number | null = null
  let eventSource: EventSource | null = null
  let closed = false

  function connect(fromId: number) {
    if (closed) return

    const url = `/api/events/datasets/${datasetId}?last_id=${fromId}`
    eventSource = new EventSource(url)

    eventSource.onmessage = (event) => {
      if (event.lastEventId) {
        lastEventId = parseInt(event.lastEventId, 10) + 1
      }
      try {
        const msg: SSEMessage = JSON.parse(event.data)
        onMessage(msg)
        if (msg.type === 'done') {
          cleanup()
          onDone?.()
        }
      } catch {
        // ignore parse errors
      }
    }

    eventSource.onerror = () => {
      if (closed) return
      eventSource?.close()
      // Reconnect after 2 seconds from where we left off
      reconnectTimer = window.setTimeout(() => {
        connect(lastEventId)
      }, 2000)
    }
  }

  function cleanup() {
    closed = true
    if (reconnectTimer !== null) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
    eventSource?.close()
    eventSource = null
  }

  connect(0)

  return cleanup
}
