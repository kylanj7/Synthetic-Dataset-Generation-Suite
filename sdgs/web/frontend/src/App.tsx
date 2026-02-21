import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/common/Layout'
import AuthGuard from './components/common/AuthGuard'
import Login from './pages/Login'
import Register from './pages/Register'
import Datasets from './pages/Datasets'
import CreateDataset from './pages/CreateDataset'
import DatasetDetail from './pages/DatasetDetail'
import Papers from './pages/Papers'
import Galaxy from './pages/Galaxy'
import Settings from './pages/Settings'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public routes */}
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />

        {/* Protected routes */}
        <Route element={<AuthGuard />}>
          <Route element={<Layout />}>
            <Route path="/" element={<Datasets />} />
            <Route path="/create" element={<CreateDataset />} />
            <Route path="/datasets/:id" element={<DatasetDetail />} />
            <Route path="/papers" element={<Papers />} />
            <Route path="/galaxy" element={<Galaxy />} />
            <Route path="/settings" element={<Settings />} />
          </Route>
        </Route>

        {/* Catch-all redirect */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
