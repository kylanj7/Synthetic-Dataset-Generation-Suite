import { NavLink } from 'react-router-dom'
import { Database, FileText, Orbit, Settings, LogOut } from 'lucide-react'
import { useAuthStore } from '../../store/authStore'

const navItems = [
  { to: '/', icon: Database, label: 'Datasets' },
  { to: '/papers', icon: FileText, label: 'Papers' },
  { to: '/galaxy', icon: Orbit, label: 'Galaxy' },
  { to: '/settings', icon: Settings, label: 'Settings' },
]

export default function Sidebar() {
  const { username, logout } = useAuthStore()

  return (
    <aside style={{
      position: 'fixed',
      left: 0,
      top: 0,
      bottom: 0,
      width: 'var(--sidebar-width)',
      background: 'var(--bg-secondary)',
      borderRight: '1px solid var(--border-subtle)',
      display: 'flex',
      flexDirection: 'column',
      zIndex: 100,
    }}>
      <div style={{
        padding: '20px',
        borderBottom: '1px solid var(--border-subtle)',
      }}>
        <h1 style={{
          fontSize: '18px',
          fontWeight: 700,
          background: 'linear-gradient(135deg, var(--accent-blue), var(--accent-cyan))',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
        }}>
          SDGS Web
        </h1>
        <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
          Synthetic Dataset Generation
        </p>
      </div>

      <nav style={{ flex: 1, padding: '12px' }}>
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            style={({ isActive }) => ({
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              padding: '10px 12px',
              borderRadius: 'var(--radius-sm)',
              color: isActive ? 'var(--accent-blue)' : 'var(--text-secondary)',
              background: isActive ? 'rgba(126, 184, 255, 0.1)' : 'transparent',
              textDecoration: 'none',
              fontSize: '14px',
              marginBottom: '2px',
              transition: 'all 0.15s',
            })}
          >
            <Icon size={18} />
            {label}
          </NavLink>
        ))}
      </nav>

      <div style={{
        padding: '16px 20px',
        borderTop: '1px solid var(--border-subtle)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <div>
          <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>{username}</div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>v0.2.0</div>
        </div>
        <button
          onClick={logout}
          title="Sign out"
          style={{
            background: 'none',
            border: 'none',
            color: 'var(--text-muted)',
            cursor: 'pointer',
            padding: '4px',
          }}
        >
          <LogOut size={16} />
        </button>
      </div>
    </aside>
  )
}
