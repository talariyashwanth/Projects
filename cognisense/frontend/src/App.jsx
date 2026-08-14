import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Dashboard from './components/Dashboard';
import LiveSession from './components/LiveSession';
import SessionAnalytics from './components/SessionAnalytics';
import ModelInsights from './components/ModelInsights';
import { fetchHealth, predictCognitiveLoad } from './api';

// Preset behavioural profiles used by the dashboard buttons and the initial load.
// These are INPUT feature vectors; every score/probability shown in the UI comes
// back from the real model via the API, never hardcoded.
const PRESETS = {
  low: {
    typing_speed_wpm: 68.0,
    avg_keystroke_interval_ms: 175.0,
    backspace_ratio: 0.03,
    pause_frequency_per_min: 2,
    mouse_velocity_px_s: 430.0,
    mouse_distance_px: 1100.0,
    click_frequency_per_min: 14.0,
    idle_time_seconds: 3.5,
    error_rate: 0.03,
    response_time_seconds: 4.2,
    retry_count: 0,
    context_switches_per_min: 1
  },
  medium: {
    typing_speed_wpm: 46.0,
    avg_keystroke_interval_ms: 260.0,
    backspace_ratio: 0.11,
    pause_frequency_per_min: 6,
    mouse_velocity_px_s: 280.0,
    mouse_distance_px: 1750.0,
    click_frequency_per_min: 22.0,
    idle_time_seconds: 7.5,
    error_rate: 0.10,
    response_time_seconds: 8.5,
    retry_count: 2,
    context_switches_per_min: 4
  },
  high: {
    typing_speed_wpm: 24.0,
    avg_keystroke_interval_ms: 420.0,
    backspace_ratio: 0.24,
    pause_frequency_per_min: 13,
    mouse_velocity_px_s: 160.0,
    mouse_distance_px: 2600.0,
    click_frequency_per_min: 33.0,
    idle_time_seconds: 14.5,
    error_rate: 0.25,
    response_time_seconds: 16.0,
    retry_count: 5,
    context_switches_per_min: 9
  }
};

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [healthData, setHealthData] = useState(null);
  const [dashboardData, setDashboardData] = useState(null);
  const [error, setError] = useState(null);

  // On mount, check the API and score a real starting profile through the model.
  // Nothing is displayed until the backend actually responds, so the dashboard
  // never shows invented numbers.
  useEffect(() => {
    async function initialise() {
      const health = await fetchHealth();
      setHealthData(health);

      if (health.status !== 'online') {
        setError('Backend offline. Start it with: uvicorn backend.main:app --reload');
        return;
      }

      try {
        setDashboardData(await predictCognitiveLoad(PRESETS.high));
      } catch (err) {
        setError(err.message);
      }
    }
    initialise();
  }, []);

  const handleSimulatePreset = async (presetType) => {
    try {
      setDashboardData(await predictCognitiveLoad(PRESETS[presetType] || PRESETS.high));
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg-dark)' }}>
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} healthData={healthData} />

      <main style={{ maxWidth: '1280px', margin: '0 auto', padding: '32px 24px' }}>
        {activeTab === 'dashboard' && (
          error ? (
            <div className="glass-card" style={{ padding: '48px', textAlign: 'center' }}>
              <div style={{ color: '#fb7185', fontWeight: 700, marginBottom: '8px' }}>
                Cannot reach the Cognisense API
              </div>
              <div style={{ color: 'var(--text-secondary)', fontSize: '0.88rem' }}>{error}</div>
            </div>
          ) : !dashboardData ? (
            <div className="glass-card" style={{ padding: '60px', textAlign: 'center', color: 'var(--text-secondary)' }}>
              Scoring initial behavioural profile through the model...
            </div>
          ) : (
            <Dashboard currentData={dashboardData} onSimulatePreset={handleSimulatePreset} />
          )
        )}

        {activeTab === 'playground' && (
          <LiveSession onUpdateDashboard={(res) => setDashboardData(res)} />
        )}

        {activeTab === 'analytics' && (
          <SessionAnalytics />
        )}

        {activeTab === 'model' && (
          <ModelInsights />
        )}
      </main>

      <footer style={{
        borderTop: '1px solid var(--border-subtle)',
        padding: '24px',
        textAlign: 'center',
        color: 'var(--text-muted)',
        fontSize: '0.8rem',
        marginTop: '40px'
      }}>
        Cognisense — Experimental Behavioral Cognitive Load Detector ML Prototype • Grounded in HCI Research
      </footer>
    </div>
  );
}
