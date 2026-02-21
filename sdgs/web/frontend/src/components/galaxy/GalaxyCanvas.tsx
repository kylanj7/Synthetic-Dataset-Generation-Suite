import { useCallback, useRef, useEffect, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'

interface Props {
  nodes: any[]
  links: any[]
  searchQuery: string
  onNodeClick: (node: any) => void
}

export default function GalaxyCanvas({ nodes, links, searchQuery, onNodeClick }: Props) {
  const fgRef = useRef<any>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 })

  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        })
      }
    }
    updateSize()
    window.addEventListener('resize', updateSize)
    return () => window.removeEventListener('resize', updateSize)
  }, [])

  // Custom node painter with cosmic glow effect
  const paintNode = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const size = node.size || 4
    const isMatch = searchQuery && node._match
    const alpha = searchQuery && !node._match ? 0.15 : 1

    ctx.save()
    ctx.globalAlpha = alpha

    // Outer glow
    const gradient = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, size * 2.5)
    gradient.addColorStop(0, node.color + 'cc')
    gradient.addColorStop(0.4, node.color + '44')
    gradient.addColorStop(1, 'transparent')
    ctx.fillStyle = gradient
    ctx.beginPath()
    ctx.arc(node.x, node.y, size * 2.5, 0, 2 * Math.PI)
    ctx.fill()

    // Core
    ctx.fillStyle = isMatch ? '#ffffff' : node.color
    ctx.beginPath()
    ctx.arc(node.x, node.y, size, 0, 2 * Math.PI)
    ctx.fill()

    // Label (papers only, on zoom)
    if (node.type === 'paper' && globalScale > 0.7) {
      const label = node.label.length > 40 ? node.label.slice(0, 40) + '...' : node.label
      const fontSize = Math.max(10 / globalScale, 3)
      ctx.font = `${fontSize}px Inter, sans-serif`
      ctx.fillStyle = 'rgba(232, 232, 240, 0.8)'
      ctx.textAlign = 'center'
      ctx.fillText(label, node.x, node.y + size + fontSize + 2)
    }

    ctx.restore()
  }, [searchQuery])

  // Tooltip
  const getNodeTooltip = useCallback((node: any) => {
    if (node.type === 'paper') {
      return `${node.label}\nYear: ${node.year || 'N/A'} | Citations: ${node.citation_count || 0}`
    }
    return node.instruction?.slice(0, 100) || node.label
  }, [])

  return (
    <div ref={containerRef} style={{
      width: '100%', height: '100%',
      background: 'radial-gradient(ellipse at center, #0a0a1a 0%, #06060f 100%)',
    }}>
      {nodes.length === 0 ? (
        <div className="empty-state" style={{ paddingTop: '200px' }}>
          <h3>No data in Galaxy</h3>
          <p>Run a scrape job to populate the Galaxy viewer</p>
        </div>
      ) : (
        <ForceGraph2D
          ref={fgRef}
          width={dimensions.width}
          height={dimensions.height}
          graphData={{ nodes, links }}
          nodeCanvasObject={paintNode}
          nodePointerAreaPaint={(node: any, color: string, ctx: CanvasRenderingContext2D) => {
            ctx.fillStyle = color
            ctx.beginPath()
            ctx.arc(node.x, node.y, (node.size || 4) * 2, 0, 2 * Math.PI)
            ctx.fill()
          }}
          linkColor={() => 'rgba(126, 184, 255, 0.08)'}
          linkWidth={(link: any) => link.weight * 2}
          linkDirectionalParticles={(link: any) => link.type === 'paper_qa' ? 2 : 0}
          linkDirectionalParticleWidth={1.5}
          linkDirectionalParticleColor={() => 'rgba(110, 231, 216, 0.6)'}
          onNodeClick={onNodeClick}
          nodeLabel={getNodeTooltip}
          d3AlphaDecay={0.02}
          d3VelocityDecay={0.3}
          cooldownTicks={100}
          backgroundColor="transparent"
        />
      )}
    </div>
  )
}
