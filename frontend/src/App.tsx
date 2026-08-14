import { Routes, Route, Navigate } from 'react-router-dom';
import AppLayout from './components/AppLayout';
import MonitorPage from './pages/MonitorPage';
import FaceLibraryPage from './pages/FaceLibraryPage';
import EventLogPage from './pages/EventLogPage';
import SettingsPage from './pages/SettingsPage';

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route path="/" element={<Navigate to="/monitor" replace />} />
        <Route path="/monitor" element={<MonitorPage />} />
        <Route path="/faces" element={<FaceLibraryPage />} />
        <Route path="/events" element={<EventLogPage />} />
        <Route path="/settings" element={<SettingsPage />} />
      </Route>
    </Routes>
  );
}
