import { useEffect, useCallback, useMemo } from 'react'
import { useGalaxyStore } from '../store/galaxyStore'
import { useGalaxyGraph } from '../hooks/useGalaxyGraph'
import GalaxyCanvas from '../components/galaxy/GalaxyCanvas'
import GalaxyControls from '../components/galaxy/GalaxyControls'
import GalaxyLegend from '../components/galaxy/GalaxyLegend'
import PaperDetailPanel from '../components/galaxy/PaperDetail'
import DatasetDetailPanel from '../components/galaxy/DatasetDetail'

export default function Galaxy() {
  const {
    data, selectedPaper, selectedDatasetNode, loading, error, searchQuery, activeCluster,
    fetchData, selectPaper, selectDatasetNode, clearSelection,
    setSearchQuery, setActiveCluster,
  } = useGalaxyStore()

  const graphData = useGalaxyGraph()

  useEffect(() => {
    fetchData()
  }, [])

  // Papers in the selected dataset's cluster (for the dataset panel)
  const datasetPapers = useMemo(() => {
    if (!selectedDatasetNode) return []
    return graphData.nodes.filter(
      (n: any) => n.type === 'paper' && n.cluster === selectedDatasetNode.cluster,
    )
  }, [selectedDatasetNode, graphData.nodes])

  const handleNodeClick = useCallback((node: any) => {
    if (node.type === 'paper') {
      const paperId = parseInt(node.id.replace('paper-', ''))
      selectPaper(paperId, node.id)
    } else if (node.type === 'dataset') {
      if (selectedDatasetNode?.id === node.id) {
        // Click same dataset again → deselect
        clearSelection()
        setActiveCluster(null)
      } else {
        selectDatasetNode(node)
        setActiveCluster(node.cluster)
      }
    }
  }, [selectPaper, selectDatasetNode, clearSelection, setActiveCluster, selectedDatasetNode])

  const handleDatasetPaperClick = useCallback((paperNode: any) => {
    const paperId = parseInt(paperNode.id.replace('paper-', ''))
    selectPaper(paperId, paperNode.id)
  }, [selectPaper])

  const handleClosePanel = useCallback(() => {
    clearSelection()
    if (selectedDatasetNode) setActiveCluster(null)
  }, [clearSelection, setActiveCluster, selectedDatasetNode])

  if (loading) {
    return <div style={{ textAlign: 'center', padding: '40px' }}><div className="spinner" /></div>
  }

  if (error) {
    return (
      <div style={{ textAlign: 'center', padding: '40px', color: 'var(--accent-red, #ff6b6b)' }}>
        <h3>Galaxy Error</h3>
        <pre style={{ whiteSpace: 'pre-wrap', fontSize: '13px', maxWidth: '600px', margin: '0 auto' }}>{error}</pre>
      </div>
    )
  }

  return (
    <div style={{ position: 'relative', height: 'calc(100vh - 48px)' }}>
      {/* Controls overlay */}
      <div style={{
        position: 'absolute', top: '16px', left: '16px', zIndex: 10,
        display: 'flex', flexDirection: 'column', gap: '8px',
      }}>
        <GalaxyControls
          searchQuery={searchQuery}
          onSearch={setSearchQuery}
        />
      </div>

      {/* Legend overlay */}
      {data && data.clusters.length > 0 && (
        <div style={{
          position: 'absolute', bottom: '16px', left: '16px', zIndex: 10,
        }}>
          <GalaxyLegend
            clusters={data.clusters}
            activeCluster={activeCluster}
            onToggle={setActiveCluster}
          />
        </div>
      )}

      {/* Graph */}
      <GalaxyCanvas
        nodes={graphData.nodes}
        links={graphData.links}
        searchQuery={searchQuery}
        onNodeClick={handleNodeClick}
      />

      {/* Paper detail panel */}
      {selectedPaper && (
        <PaperDetailPanel paper={selectedPaper} onClose={handleClosePanel} />
      )}

      {/* Dataset detail panel */}
      {selectedDatasetNode && !selectedPaper && (
        <DatasetDetailPanel
          node={selectedDatasetNode}
          papers={datasetPapers}
          onPaperClick={handleDatasetPaperClick}
          onClose={handleClosePanel}
        />
      )}
    </div>
  )
}
