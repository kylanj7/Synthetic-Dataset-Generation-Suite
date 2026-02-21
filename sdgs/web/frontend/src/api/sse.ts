export interface SSEMessage {
  type: 'log' | 'status' | 'error' | 'done'
  data: string
}

export function createDatasetSSE(
  datasetId: number,
  onMessage: (msg: SSEMessage) => void,
  onDone?: () => void,
): () => void {
  const url = `/api/events/datasets/${datasetId}`
  const eventSource = new EventSource(url)

  eventSource.onmessage = (event) => {
    try {
      const msg: SSEMessage = JSON.parse(event.data)
      onMessage(msg)
      if (msg.type === 'done') {
        eventSource.close()
        onDone?.()
      }
    } catch {
      // ignore parse errors
    }
  }

  eventSource.onerror = () => {
    eventSource.close()
    onDone?.()
  }

  return () => eventSource.close()
}
