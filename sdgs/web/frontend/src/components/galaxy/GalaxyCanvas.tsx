import { useCallback, useRef, useEffect, useState } from 'react'
import ForceGraph3D from 'react-force-graph-3d'
import * as THREE from 'three'

// --- Shared geometries (created once, reused by all nodes) ---
const SHARED_GEO = {
  qa: new THREE.OctahedronGeometry(1, 0),
  paper: new THREE.SphereGeometry(1, 8, 6),
  dataset: new THREE.IcosahedronGeometry(1, 1),
  datasetWire: new THREE.IcosahedronGeometry(1.15, 1),
}

// --- Material cache ---
const _matCache = new Map<string, THREE.Material>()

function getMaterial(
  kind: 'basic' | 'phong',
  color: string,
  opacity: number,
  emissiveIntensity = 0,
  extra: Record<string, any> = {},
): THREE.Material {
  const key = `${kind}|${color}|${opacity}|${emissiveIntensity}|${JSON.stringify(extra)}`
  let mat = _matCache.get(key)
  if (!mat) {
    if (kind === 'basic') {
      mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity, ...extra })
    } else {
      mat = new THREE.MeshPhongMaterial({
        color, transparent: true, opacity,
        emissive: color, emissiveIntensity, ...extra,
      })
    }
    _matCache.set(key, mat)
  }
  return mat
}

const PERF_THRESHOLD = 500

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
  const isLarge = nodes.length > PERF_THRESHOLD

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

  // Auto-rotate
  useEffect(() => {
    const fg = fgRef.current
    if (!fg) return
    const controls = fg.controls()
    if (controls) {
      controls.autoRotate = true
      controls.autoRotateSpeed = 0.5
    }
  }, [nodes])

  // Custom 3D node objects
  const nodeThreeObject = useCallback((node: any) => {
    const color = node.color || '#7eb8ff'
    const size = node.size || 4
    const isMatch = searchQuery && node._match
    const dimmed = searchQuery && !node._match

    const nodeColor = isMatch ? '#ffffff' : color
    const opacity = dimmed ? 0.15 : 1

    if (node.type === 'dataset') {
      const mat = getMaterial('phong', nodeColor, opacity, 0.6, { shininess: 100 })
      const mesh = new THREE.Mesh(SHARED_GEO.dataset, mat)
      mesh.scale.setScalar(size)

      // Wireframe overlay
      const wireMat = getMaterial('basic', '#ffffff', opacity * 0.25, 0, { wireframe: true })
      const wire = new THREE.Mesh(SHARED_GEO.datasetWire, wireMat)
      wire.scale.setScalar(1)  // already scaled relative to parent
      mesh.add(wire)

      if (!isLarge) {
        const spriteMat = new THREE.SpriteMaterial({
          map: createGlowTexture(color),
          transparent: true,
          opacity: opacity * 0.4,
          blending: THREE.AdditiveBlending,
        })
        const sprite = new THREE.Sprite(spriteMat)
        sprite.scale.set(6, 6, 1)
        mesh.add(sprite)
      }

      return mesh
    }

    if (node.type === 'paper') {
      const mat = getMaterial('phong', nodeColor, opacity, 0.3)
      const mesh = new THREE.Mesh(SHARED_GEO.paper, mat)
      mesh.scale.setScalar(size)

      if (!isLarge) {
        const spriteMat = new THREE.SpriteMaterial({
          map: createGlowTexture(color),
          transparent: true,
          opacity: opacity * 0.3,
          blending: THREE.AdditiveBlending,
        })
        const sprite = new THREE.Sprite(spriteMat)
        sprite.scale.set(5, 5, 1)
        mesh.add(sprite)
      }

      return mesh
    }

    // QA nodes: flat material — no lighting needed for small abundant nodes
    const mat = getMaterial('basic', nodeColor, opacity)
    const mesh = new THREE.Mesh(SHARED_GEO.qa, mat)
    mesh.scale.setScalar(size)
    return mesh
  }, [searchQuery, isLarge])

  // Tooltip
  const getNodeLabel = useCallback((node: any) => {
    if (node.type === 'dataset') {
      return `<div style="background:rgba(10,10,30,0.9);padding:8px 12px;border-radius:6px;border:1px solid rgba(126,184,255,0.3);max-width:300px">
        <div style="color:#7eb8ff;font-weight:600;margin-bottom:4px">Dataset</div>
        <div style="color:#e8e8f0">${node.label}</div>
        <div style="color:#888;font-size:12px;margin-top:4px">${node.abstract || ''}</div>
      </div>`
    }
    if (node.type === 'paper') {
      return `<div style="background:rgba(10,10,30,0.9);padding:8px 12px;border-radius:6px;border:1px solid rgba(126,184,255,0.3);max-width:300px">
        <div style="color:#6ee7d8;font-weight:600;margin-bottom:4px">Paper</div>
        <div style="color:#e8e8f0">${node.label}</div>
        <div style="color:#888;font-size:12px;margin-top:4px">Year: ${node.year || 'N/A'} | Citations: ${node.citation_count || 0}</div>
      </div>`
    }
    return `<div style="background:rgba(10,10,30,0.9);padding:6px 10px;border-radius:4px;border:1px solid rgba(110,231,216,0.3);max-width:280px">
      <div style="color:#ffd666;font-size:12px;margin-bottom:2px">Q&A</div>
      <div style="color:#e8e8f0;font-size:13px">${(node.instruction || node.label || '').slice(0, 120)}</div>
    </div>`
  }, [])

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%', background: '#06060f' }}>
      {nodes.length === 0 ? (
        <div className="empty-state" style={{ paddingTop: '200px' }}>
          <h3>No data in Galaxy</h3>
          <p>Generate a dataset to populate the Galaxy viewer</p>
        </div>
      ) : (
        <ForceGraph3D
          ref={fgRef}
          width={dimensions.width}
          height={dimensions.height}
          graphData={{ nodes, links }}
          nodeThreeObject={nodeThreeObject}
          nodeThreeObjectExtend={false}
          linkColor={(link: any) => {
            if (link.type === 'dataset_paper') return 'rgba(126, 184, 255, 0.2)'
            if (link.type === 'paper_qa' || link.type === 'dataset_qa') return 'rgba(110, 231, 216, 0.15)'
            if (link.type === 'keyword') return 'rgba(255, 214, 102, 0.12)'
            return 'rgba(126, 184, 255, 0.08)'
          }}
          linkWidth={isLarge ? 1 : ((link: any) => link.weight * 1.5)}
          linkOpacity={0.6}
          linkDirectionalParticles={isLarge ? 0 : (link: any) =>
            ['paper_qa', 'dataset_paper', 'dataset_qa'].includes(link.type) ? 2 : 0
          }
          linkDirectionalParticleWidth={1.5}
          linkDirectionalParticleColor={(link: any) => {
            if (link.type === 'dataset_paper') return '#7eb8ff'
            return '#6ee7d8'
          }}
          onNodeClick={onNodeClick}
          nodeLabel={getNodeLabel}
          warmupTicks={isLarge ? 60 : 0}
          d3AlphaDecay={isLarge ? 0.05 : 0.02}
          d3VelocityDecay={isLarge ? 0.4 : 0.3}
          cooldownTicks={isLarge ? 60 : 100}
          backgroundColor="#06060f"
          showNavInfo={false}
        />
      )}
    </div>
  )
}

// Generate a radial glow texture for sprites
const glowTextureCache = new Map<string, THREE.Texture>()

function createGlowTexture(color: string): THREE.Texture {
  if (glowTextureCache.has(color)) return glowTextureCache.get(color)!

  const size = 128
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')!

  const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2)
  gradient.addColorStop(0, color)
  gradient.addColorStop(0.4, 'rgba(126, 184, 255, 0.15)')
  gradient.addColorStop(1, 'rgba(0, 0, 0, 0)')
  ctx.fillStyle = gradient
  ctx.fillRect(0, 0, size, size)

  const texture = new THREE.CanvasTexture(canvas)
  glowTextureCache.set(color, texture)
  return texture
}
