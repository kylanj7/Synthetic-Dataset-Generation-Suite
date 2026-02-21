import { useMemo } from 'react'
import { useGalaxyStore } from '../store/galaxyStore'

export function useGalaxyGraph() {
  const { data, showQA, searchQuery, activeCluster } = useGalaxyStore()

  return useMemo(() => {
    if (!data) return { nodes: [], links: [] }

    let nodes = [...data.nodes]
    let links = [...data.links]

    // Filter by cluster
    if (activeCluster !== null) {
      const clusterNodeIds = new Set(
        nodes.filter(n => n.cluster === activeCluster).map(n => n.id)
      )
      nodes = nodes.filter(n => clusterNodeIds.has(n.id))
      links = links.filter(l => {
        const src = typeof l.source === 'string' ? l.source : (l.source as any).id
        const tgt = typeof l.target === 'string' ? l.target : (l.target as any).id
        return clusterNodeIds.has(src) && clusterNodeIds.has(tgt)
      })
    }

    // Filter QA visibility
    if (!showQA) {
      const keepIds = new Set(nodes.filter(n => n.type === 'paper' || n.type === 'dataset').map(n => n.id))
      nodes = nodes.filter(n => n.type !== 'qa')
      links = links.filter(l => {
        const src = typeof l.source === 'string' ? l.source : (l.source as any).id
        const tgt = typeof l.target === 'string' ? l.target : (l.target as any).id
        return keepIds.has(src) && keepIds.has(tgt)
      })
    }

    // Search highlighting (add a `_match` flag)
    if (searchQuery) {
      const q = searchQuery.toLowerCase()
      nodes = nodes.map(n => ({
        ...n,
        _match: n.label.toLowerCase().includes(q) ||
          (n.instruction?.toLowerCase().includes(q) ?? false),
      }))
    }

    return { nodes, links }
  }, [data, showQA, searchQuery, activeCluster])
}
