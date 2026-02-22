import { useMemo } from 'react'
import { useGalaxyStore } from '../store/galaxyStore'

export function useGalaxyGraph() {
  const { data, searchQuery, activeCluster, selectedPaper, expandedPaperGraphId } = useGalaxyStore()

  return useMemo(() => {
    if (!data) return { nodes: [], links: [] }

    let nodes = [...data.nodes]
    let links = [...data.links]

    // Filter by cluster (includes QA nodes in that cluster)
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

    // Expand QA nodes for selected paper
    if (selectedPaper && expandedPaperGraphId) {
      const paperNode = nodes.find(n => n.id === expandedPaperGraphId)
      if (paperNode) {
        ;(paperNode as any)._expanded = true

        // Remove sample QAs already linked to this paper (backend sends 2)
        const sampleQaIds = new Set<string>()
        for (const l of links) {
          const src = typeof l.source === 'string' ? l.source : (l.source as any).id
          const tgt = typeof l.target === 'string' ? l.target : (l.target as any).id
          if (src === expandedPaperGraphId && l.type === 'paper_qa') {
            sampleQaIds.add(tgt)
          }
        }
        nodes = nodes.filter(n => !sampleQaIds.has(n.id))
        links = links.filter(l => {
          const tgt = typeof l.target === 'string' ? l.target : (l.target as any).id
          return !sampleQaIds.has(tgt)
        })

        // Inject all QAs from the fetched paper detail
        const qas = selectedPaper.qa_pairs.slice(0, 30)
        for (let i = 0; i < qas.length; i++) {
          const qa = qas[i]
          nodes.push({
            id: `qa-exp-${i}`,
            type: 'qa',
            label: (qa.instruction || '').slice(0, 50),
            size: 2.5,
            color: paperNode.color,
            cluster: paperNode.cluster,
            instruction: qa.instruction,
            answer_text: qa.answer_text || qa.output || '',
            is_valid: qa.is_valid,
          } as any)
          links.push({
            source: expandedPaperGraphId,
            target: `qa-exp-${i}`,
            weight: 0.3,
            type: 'paper_qa',
          })
        }
      }
    }

    // Search highlighting
    if (searchQuery) {
      const q = searchQuery.toLowerCase()
      nodes = nodes.map(n => ({
        ...n,
        _match: n.label.toLowerCase().includes(q) ||
          (n.type === 'qa' && (n as any).instruction?.toLowerCase().includes(q)),
      }))
    }

    return { nodes, links }
  }, [data, searchQuery, activeCluster, selectedPaper, expandedPaperGraphId])
}
