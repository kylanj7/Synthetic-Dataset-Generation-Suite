import { useCallback, useRef, useEffect, useState, useMemo } from 'react'
import ForceGraph3D from 'react-force-graph-3d'
import * as THREE from 'three'

// ── Performance threshold ────────────────────────────────────────────
// Above this count, QA nodes switch from individual Mesh → single Points object.
// Papers/datasets stay as Mesh (they're few and need custom geometry + interaction).
const PERF_THRESHOLD = 500

// ── GLSL shaders for QA Points overlay ───────────────────────────────
// One draw call renders ALL QA nodes as screen-space diamond shapes
// with per-point color, size, and opacity via vertex attributes.
const QA_VERT = `
attribute float aSize;
attribute float aOpacity;
varying vec3 vColor;
varying float vOpacity;

void main() {
  vColor = color;
  vOpacity = aOpacity;
  vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
  // Size attenuation: larger when close, smaller when far
  gl_PointSize = aSize * (300.0 / -mvPos.z);
  gl_PointSize = clamp(gl_PointSize, 1.0, 48.0);
  gl_Position = projectionMatrix * mvPos;
}
`

const QA_FRAG = `
varying vec3 vColor;
varying float vOpacity;

void main() {
  // Diamond shape via Manhattan distance on the point sprite
  vec2 p = 2.0 * gl_PointCoord - 1.0;
  float d = abs(p.x) + abs(p.y);
  if (d > 1.0) discard;
  // Soft anti-aliased edge
  float alpha = 1.0 - smoothstep(0.75, 1.0, d);
  gl_FragColor = vec4(vColor, vOpacity * alpha);
}
`

// ── Shared geometries for paper/dataset Mesh nodes ───────────────────
const SHARED_GEO = {
  qa: new THREE.OctahedronGeometry(1, 0),
  paper: new THREE.SphereGeometry(1, 8, 6),
  dataset: new THREE.IcosahedronGeometry(1, 1),
  datasetWire: new THREE.IcosahedronGeometry(1.15, 1),
}

// ── Material cache ───────────────────────────────────────────────────
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

// ── Reusable empty Object3D for invisible QA placeholders ────────────
// Returned by nodeThreeObject for QA nodes in large mode.
// The library positions it (for force sim), but nothing is drawn (0 draw calls).
const _emptyObj = new THREE.Object3D()

// ── Glow texture cache ───────────────────────────────────────────────
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


// ══════════════════════════════════════════════════════════════════════
// Component
// ══════════════════════════════════════════════════════════════════════

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

  // Refs for the QA Points overlay (only used when isLarge)
  const qaPointsRef = useRef<THREE.Points | null>(null)
  const qaMatRef = useRef<THREE.ShaderMaterial | null>(null)
  const qaGeoRef = useRef<THREE.BufferGeometry | null>(null)

  // Index QA nodes so we can map node objects to buffer indices
  const qaNodes = useMemo(
    () => (isLarge ? nodes.filter((n: any) => n.type === 'qa') : []),
    [nodes, isLarge],
  )

  // ── Resize ─────────────────────────────────────────────────────────
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

  // ── Auto-rotate ────────────────────────────────────────────────────
  useEffect(() => {
    const fg = fgRef.current
    if (!fg) return
    const controls = fg.controls()
    if (controls) {
      controls.autoRotate = true
      controls.autoRotateSpeed = 0.5
    }
  }, [nodes])

  // ── Create / destroy QA Points overlay ─────────────────────────────
  // Runs when qaNodes changes (new data loaded).
  useEffect(() => {
    const fg = fgRef.current
    if (!fg || qaNodes.length === 0) return

    const scene = fg.scene()

    // Tear down previous Points if any
    if (qaPointsRef.current) {
      scene.remove(qaPointsRef.current)
      qaGeoRef.current?.dispose()
      qaMatRef.current?.dispose()
      qaPointsRef.current = null
    }

    const count = qaNodes.length
    const positions = new Float32Array(count * 3)
    const colors = new Float32Array(count * 3)
    const sizes = new Float32Array(count)
    const opacities = new Float32Array(count)

    const tmpColor = new THREE.Color()

    for (let i = 0; i < count; i++) {
      const node = qaNodes[i]

      // Initial positions (may be 0 if warmup hasn't run yet — synced later)
      positions[i * 3] = node.x || 0
      positions[i * 3 + 1] = node.y || 0
      positions[i * 3 + 2] = node.z || 0

      tmpColor.set(node.color || '#6ee7d8')
      colors[i * 3] = tmpColor.r
      colors[i * 3 + 1] = tmpColor.g
      colors[i * 3 + 2] = tmpColor.b

      sizes[i] = (node.size || 2) * 3
      opacities[i] = 1.0
    }

    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    geometry.setAttribute('aSize', new THREE.BufferAttribute(sizes, 1))
    geometry.setAttribute('aOpacity', new THREE.BufferAttribute(opacities, 1))

    const material = new THREE.ShaderMaterial({
      vertexShader: QA_VERT,
      fragmentShader: QA_FRAG,
      vertexColors: true,
      transparent: true,
      depthWrite: false,
    })

    const points = new THREE.Points(geometry, material)
    scene.add(points)

    qaPointsRef.current = points
    qaGeoRef.current = geometry
    qaMatRef.current = material

    return () => {
      scene.remove(points)
      geometry.dispose()
      material.dispose()
      qaPointsRef.current = null
      qaGeoRef.current = null
      qaMatRef.current = null
    }
  }, [qaNodes])

  // ── Sync QA Point positions from force simulation ──────────────────
  // Called on every engine tick and once when engine stops.
  const syncQAPositions = useCallback(() => {
    const geo = qaGeoRef.current
    if (!geo || qaNodes.length === 0) return

    const posAttr = geo.getAttribute('position') as THREE.BufferAttribute
    const arr = posAttr.array as Float32Array

    for (let i = 0; i < qaNodes.length; i++) {
      const n = qaNodes[i]
      arr[i * 3] = n.x || 0
      arr[i * 3 + 1] = n.y || 0
      arr[i * 3 + 2] = n.z || 0
    }
    posAttr.needsUpdate = true
  }, [qaNodes])

  // ── Update QA colors/opacity when searchQuery changes ──────────────
  useEffect(() => {
    const geo = qaGeoRef.current
    if (!geo || qaNodes.length === 0) return

    const colorAttr = geo.getAttribute('color') as THREE.BufferAttribute
    const opacityAttr = geo.getAttribute('aOpacity') as THREE.BufferAttribute
    const cArr = colorAttr.array as Float32Array
    const oArr = opacityAttr.array as Float32Array
    const tmpColor = new THREE.Color()

    for (let i = 0; i < qaNodes.length; i++) {
      const node = qaNodes[i]
      const isMatch = searchQuery && node._match
      const dimmed = searchQuery && !node._match

      tmpColor.set(isMatch ? '#ffffff' : (node.color || '#6ee7d8'))
      cArr[i * 3] = tmpColor.r
      cArr[i * 3 + 1] = tmpColor.g
      cArr[i * 3 + 2] = tmpColor.b

      oArr[i] = dimmed ? 0.15 : 1.0
    }
    colorAttr.needsUpdate = true
    opacityAttr.needsUpdate = true
  }, [searchQuery, qaNodes])

  // ── nodeThreeObject ────────────────────────────────────────────────
  // In large mode: QA → empty Object3D (0 draw calls, positioned by library)
  //                Paper/Dataset → custom Mesh (few draw calls)
  // In normal mode: all nodes get individual Mesh objects.
  const nodeThreeObject = useCallback((node: any) => {
    // Large mode: QA rendered by Points overlay, return invisible placeholder
    if (isLarge && node.type === 'qa') {
      return _emptyObj.clone()
    }

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

      const wireMat = getMaterial('basic', '#ffffff', opacity * 0.25, 0, { wireframe: true })
      const wire = new THREE.Mesh(SHARED_GEO.datasetWire, wireMat)
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

    // QA nodes in normal mode (<= PERF_THRESHOLD): individual Mesh
    const mat = getMaterial('basic', nodeColor, opacity)
    const mesh = new THREE.Mesh(SHARED_GEO.qa, mat)
    mesh.scale.setScalar(size)
    return mesh
  }, [searchQuery, isLarge])

  // ── Tooltip ────────────────────────────────────────────────────────
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

  // ── Render ─────────────────────────────────────────────────────────
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
          // linkWidth 0 → THREE.Line (1 draw call each, no cylinder geometry)
          // positive value → THREE.Mesh cylinder (expensive per link)
          linkWidth={isLarge ? 0 : ((link: any) => link.weight * 1.5)}
          linkOpacity={0.6}
          linkDirectionalParticles={isLarge ? 0 : ((link: any) =>
            ['paper_qa', 'dataset_paper', 'dataset_qa'].includes(link.type) ? 2 : 0
          )}
          linkDirectionalParticleWidth={1.5}
          linkDirectionalParticleColor={(link: any) => {
            if (link.type === 'dataset_paper') return '#7eb8ff'
            return '#6ee7d8'
          }}
          onNodeClick={onNodeClick}
          nodeLabel={getNodeLabel}
          // Force sim tuning: front-load computation, converge fast
          warmupTicks={isLarge ? 100 : 0}
          cooldownTicks={isLarge ? 40 : 100}
          d3AlphaDecay={isLarge ? 0.06 : 0.02}
          d3VelocityDecay={isLarge ? 0.4 : 0.3}
          d3AlphaMin={0.001}
          // Sync QA Points positions from force simulation
          onEngineTick={isLarge ? syncQAPositions : undefined}
          onEngineStop={isLarge ? syncQAPositions : undefined}
          backgroundColor="#06060f"
          showNavInfo={false}
        />
      )}
    </div>
  )
}
