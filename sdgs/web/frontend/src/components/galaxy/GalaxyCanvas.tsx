import { useCallback, useRef, useEffect, useState, useMemo } from 'react'
import ForceGraph3D from 'react-force-graph-3d'
import * as THREE from 'three'

// ── Shared geometries ────────────────────────────────────────────────
const SHARED_GEO = {
  qa: new THREE.OctahedronGeometry(1, 0),
  paper: new THREE.SphereGeometry(1, 8, 6),
  dataset: new THREE.IcosahedronGeometry(1, 1),
  datasetWire: new THREE.IcosahedronGeometry(1.15, 1),
}

// ── Material cache (includes sprites) ────────────────────────────────
const _matCache = new Map<string, THREE.Material>()

function getMaterial(
  kind: 'basic' | 'phong' | 'sprite',
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
    } else if (kind === 'sprite') {
      mat = new THREE.SpriteMaterial({ transparent: true, opacity, ...extra })
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

// ── Glow texture cache ───────────────────────────────────────────────
const glowTextureCache = new Map<string, THREE.Texture>()

function createGlowTexture(color: string): THREE.Texture {
  if (glowTextureCache.has(color)) return glowTextureCache.get(color)!
  const size = 128
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')!
  const gradient = ctx.createRadialGradient(64, 64, 0, 64, 64, 64)
  gradient.addColorStop(0, color)
  gradient.addColorStop(0.4, 'rgba(126, 184, 255, 0.15)')
  gradient.addColorStop(1, 'rgba(0, 0, 0, 0)')
  ctx.fillStyle = gradient
  ctx.fillRect(0, 0, size, size)
  const texture = new THREE.CanvasTexture(canvas)
  glowTextureCache.set(color, texture)
  return texture
}

// ── QA cloud shaders (all orbit math on GPU) ─────────────────────────
const CLOUD_VERT = `
uniform float uTime;
attribute vec3 aCenter;
attribute float aRadius;
attribute float aSpeed;
attribute vec3 aPhase;
varying vec3 vColor;

void main() {
  vColor = color;
  vec3 offset = vec3(
    cos(uTime * aSpeed + aPhase.x) * aRadius,
    sin(uTime * aSpeed + aPhase.y) * aRadius * 0.7,
    cos(uTime * aSpeed * 0.8 + aPhase.z) * aRadius
  );
  vec4 mvPos = modelViewMatrix * vec4(aCenter + offset, 1.0);
  gl_PointSize = 4.0 * (200.0 / -mvPos.z);
  gl_PointSize = clamp(gl_PointSize, 0.5, 12.0);
  gl_Position = projectionMatrix * mvPos;
}
`

const CLOUD_FRAG = `
varying vec3 vColor;
void main() {
  float d = length(gl_PointCoord - 0.5) * 2.0;
  if (d > 1.0) discard;
  float alpha = (1.0 - smoothstep(0.3, 1.0, d)) * 0.55;
  gl_FragColor = vec4(vColor, alpha);
}
`


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

  const starfieldRef = useRef<THREE.Points | null>(null)
  const qaCloudRef = useRef<THREE.Points | null>(null)
  const nodesRef = useRef(nodes)
  nodesRef.current = nodes

  // Nodes eligible for decorative QA cloud (papers + datasets with QA pairs)
  const cloudNodes = useMemo(
    () => nodes.filter((n: any) =>
      (n.type === 'paper' || n.type === 'dataset') && (n.qa_pair_count || 0) > 0 && !n._expanded
    ),
    [nodes],
  )

  // Stable mapping: particle index → paper node ID (for center sync)
  const cloudMapRef = useRef<string[]>([])

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

  // ── Starfield (static positions, rotation via onBeforeRender) ──────
  useEffect(() => {
    const fg = fgRef.current
    if (!fg || nodes.length === 0) return
    const scene = fg.scene()

    if (starfieldRef.current) {
      scene.remove(starfieldRef.current)
      starfieldRef.current.geometry.dispose()
      ;(starfieldRef.current.material as THREE.Material).dispose()
    }

    const count = 2500
    const positions = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      const theta = Math.random() * Math.PI * 2
      const phi = Math.acos(2 * Math.random() - 1)
      const r = 500 + Math.random() * 1000
      positions[i * 3] = r * Math.sin(phi) * Math.cos(theta)
      positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta)
      positions[i * 3 + 2] = r * Math.cos(phi)
    }

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    const mat = new THREE.PointsMaterial({
      color: '#c8d4ff',
      size: 1.2,
      transparent: true,
      opacity: 0.5,
      sizeAttenuation: false,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    })
    const stars = new THREE.Points(geo, mat)
    // Rotation driven by Three.js render loop — zero CPU cost
    stars.onBeforeRender = () => {
      const t = performance.now() * 0.001
      stars.rotation.y = t * 0.008
      stars.rotation.x = Math.sin(t * 0.004) * 0.08
    }
    scene.add(stars)
    starfieldRef.current = stars

    return () => {
      scene.remove(stars)
      geo.dispose()
      mat.dispose()
      starfieldRef.current = null
    }
  }, [nodes.length > 0]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── QA cloud (GPU-driven orbits via ShaderMaterial) ────────────────
  useEffect(() => {
    const fg = fgRef.current
    if (!fg) return
    const scene = fg.scene()

    if (qaCloudRef.current) {
      scene.remove(qaCloudRef.current)
      qaCloudRef.current.geometry.dispose()
      ;(qaCloudRef.current.material as THREE.Material).dispose()
      qaCloudRef.current = null
    }

    if (cloudNodes.length === 0) {
      cloudMapRef.current = []
      return
    }

    const tmpColor = new THREE.Color()
    const nodeIds: string[] = []
    const centers: number[] = []
    const radii: number[] = []
    const speeds: number[] = []
    const phases: number[] = []
    const colors: number[] = []

    for (const paper of cloudNodes) {
      const qaCount = paper.qa_pair_count || 0
      const particleCount = Math.min(qaCount, 8)
      const baseRadius = (paper.size || 4) * 1.2 + Math.log2(qaCount + 1) * 2
      tmpColor.set(paper.color || '#6ee7d8')

      for (let i = 0; i < particleCount; i++) {
        nodeIds.push(paper.id)
        centers.push(paper.x || 0, paper.y || 0, paper.z || 0)
        radii.push(baseRadius + Math.random() * 5)
        speeds.push(0.2 + Math.random() * 0.5)
        phases.push(
          Math.random() * Math.PI * 2,
          Math.random() * Math.PI * 2,
          Math.random() * Math.PI * 2,
        )
        colors.push(tmpColor.r, tmpColor.g, tmpColor.b)
      }
    }

    cloudMapRef.current = nodeIds
    const total = nodeIds.length

    const geo = new THREE.BufferGeometry()
    // position is required by Three.js but unused — shader reads aCenter
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(total * 3), 3))
    geo.setAttribute('aCenter', new THREE.BufferAttribute(new Float32Array(centers), 3))
    geo.setAttribute('aRadius', new THREE.BufferAttribute(new Float32Array(radii), 1))
    geo.setAttribute('aSpeed', new THREE.BufferAttribute(new Float32Array(speeds), 1))
    geo.setAttribute('aPhase', new THREE.BufferAttribute(new Float32Array(phases), 3))
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(colors), 3))

    const mat = new THREE.ShaderMaterial({
      vertexShader: CLOUD_VERT,
      fragmentShader: CLOUD_FRAG,
      uniforms: { uTime: { value: 0 } },
      vertexColors: true,
      transparent: true,
      depthWrite: false,
    })

    const points = new THREE.Points(geo, mat)
    // Time uniform updated by Three.js render loop — 1 float, zero overhead
    points.onBeforeRender = () => {
      mat.uniforms.uTime.value = performance.now() * 0.001
    }
    scene.add(points)
    qaCloudRef.current = points

    return () => {
      scene.remove(points)
      geo.dispose()
      mat.dispose()
      qaCloudRef.current = null
      cloudMapRef.current = []
    }
  }, [cloudNodes])

  // ── Sync cloud center positions from force sim (CPU, only during sim) ──
  const syncCloudCenters = useCallback(() => {
    const cloud = qaCloudRef.current
    const map = cloudMapRef.current
    if (!cloud || map.length === 0) return

    const centerAttr = cloud.geometry.getAttribute('aCenter') as THREE.BufferAttribute
    const arr = centerAttr.array as Float32Array

    // Build lookup from current nodes
    const currentNodes = nodesRef.current
    const nodeMap = new Map<string, any>()
    for (const n of currentNodes) nodeMap.set(n.id, n)

    for (let i = 0; i < map.length; i++) {
      const paper = nodeMap.get(map[i])
      if (!paper) continue
      arr[i * 3] = paper.x || 0
      arr[i * 3 + 1] = paper.y || 0
      arr[i * 3 + 2] = paper.z || 0
    }
    centerAttr.needsUpdate = true
  }, [])

  // ── nodeThreeObject ────────────────────────────────────────────────
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

      const wireMat = getMaterial('basic', '#ffffff', opacity * 0.25, 0, { wireframe: true })
      const wire = new THREE.Mesh(SHARED_GEO.datasetWire, wireMat)
      mesh.add(wire)

      const spriteMat = getMaterial('sprite', color, opacity * 0.4, 0, {
        map: createGlowTexture(color),
        blending: THREE.AdditiveBlending,
      }) as THREE.SpriteMaterial
      const sprite = new THREE.Sprite(spriteMat)
      sprite.scale.set(6, 6, 1)
      mesh.add(sprite)
      return mesh
    }

    if (node.type === 'qa') {
      const mat = getMaterial('phong', nodeColor, opacity * 0.9, 0.4)
      const mesh = new THREE.Mesh(SHARED_GEO.qa, mat)
      mesh.scale.setScalar(size)

      const spriteMat = getMaterial('sprite', color, opacity * 0.2, 0, {
        map: createGlowTexture(color),
        blending: THREE.AdditiveBlending,
      }) as THREE.SpriteMaterial
      const sprite = new THREE.Sprite(spriteMat)
      sprite.scale.set(3, 3, 1)
      mesh.add(sprite)
      return mesh
    }

    // Paper nodes
    const mat = getMaterial('phong', nodeColor, opacity, 0.3)
    const mesh = new THREE.Mesh(SHARED_GEO.paper, mat)
    mesh.scale.setScalar(size)

    const spriteMat = getMaterial('sprite', color, opacity * 0.3, 0, {
      map: createGlowTexture(color),
      blending: THREE.AdditiveBlending,
    }) as THREE.SpriteMaterial
    const sprite = new THREE.Sprite(spriteMat)
    sprite.scale.set(5, 5, 1)
    mesh.add(sprite)
    return mesh
  }, [searchQuery])

  // ── Tooltip ────────────────────────────────────────────────────────
  const getNodeLabel = useCallback((node: any) => {
    if (node.type === 'dataset') {
      return `<div style="background:rgba(10,10,30,0.9);padding:8px 12px;border-radius:6px;border:1px solid rgba(126,184,255,0.3);max-width:300px">
        <div style="color:#7eb8ff;font-weight:600;margin-bottom:4px">Dataset</div>
        <div style="color:#e8e8f0">${node.label}</div>
        <div style="color:#888;font-size:12px;margin-top:4px">${node.abstract || ''}</div>
      </div>`
    }
    if (node.type === 'qa') {
      const instruction = node.instruction || node.label || ''
      const answer = node.answer_text || ''
      return `<div style="background:rgba(10,10,30,0.95);padding:10px 14px;border-radius:6px;border:1px solid rgba(110,231,216,0.3);max-width:350px">
        <div style="color:#ffd666;font-weight:600;font-size:11px;margin-bottom:6px">Q&A</div>
        <div style="color:#e8e8f0;font-size:13px;margin-bottom:8px">${instruction.length > 200 ? instruction.slice(0, 200) + '...' : instruction}</div>
        ${answer ? `<div style="color:#a8a8b8;font-size:12px;border-top:1px solid rgba(255,255,255,0.1);padding-top:6px">${answer.length > 150 ? answer.slice(0, 150) + '...' : answer}</div>` : ''}
        ${node.is_valid ? '<div style="margin-top:4px"><span style="background:rgba(110,231,216,0.15);color:#6ee7d8;font-size:10px;padding:2px 6px;border-radius:3px">valid</span></div>' : ''}
      </div>`
    }
    const qaCount = node.qa_pair_count != null ? node.qa_pair_count : 0
    return `<div style="background:rgba(10,10,30,0.9);padding:8px 12px;border-radius:6px;border:1px solid rgba(126,184,255,0.3);max-width:300px">
      <div style="color:#6ee7d8;font-weight:600;margin-bottom:4px">Paper</div>
      <div style="color:#e8e8f0">${node.label}</div>
      <div style="color:#888;font-size:12px;margin-top:4px">Year: ${node.year || 'N/A'} | Citations: ${node.citation_count || 0} | Q&A pairs: ${qaCount}</div>
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
            if (link.type === 'dataset_paper') return 'rgba(126, 184, 255, 0.25)'
            if (link.type === 'paper_qa' || link.type === 'dataset_qa') return 'rgba(110, 231, 216, 0.2)'
            if (link.type === 'keyword') return 'rgba(255, 214, 102, 0.15)'
            return 'rgba(126, 184, 255, 0.1)'
          }}
          linkWidth={(link: any) =>
            link.type === 'paper_qa' || link.type === 'dataset_qa' ? 0.8 : link.weight * 2
          }
          linkOpacity={0.7}
          linkDirectionalParticles={(link: any) =>
            link.type === 'dataset_paper' ? 3
              : link.type === 'paper_qa' || link.type === 'dataset_qa' ? 2
              : link.type === 'keyword' ? 2
              : 0
          }
          linkDirectionalParticleWidth={(link: any) =>
            link.type === 'paper_qa' || link.type === 'dataset_qa' ? 1.2 : 1.8
          }
          linkDirectionalParticleSpeed={0.006}
          linkDirectionalParticleColor={(link: any) => {
            if (link.type === 'dataset_paper') return '#7eb8ff'
            if (link.type === 'paper_qa' || link.type === 'dataset_qa') return '#6ee7d8'
            return '#ffd666'
          }}
          onNodeClick={onNodeClick}
          nodeLabel={getNodeLabel}
          onEngineTick={syncCloudCenters}
          onEngineStop={syncCloudCenters}
          cooldownTicks={100}
          d3AlphaDecay={0.02}
          d3VelocityDecay={0.3}
          d3AlphaMin={0.001}
          backgroundColor="#06060f"
          showNavInfo={false}
        />
      )}
    </div>
  )
}
