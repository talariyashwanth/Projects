import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Dashboard from './components/Dashboard';
import LiveSession from './components/LiveSession';
import SessionAnalytics from './components/SessionAnalytics';
import ModelInsights from './components/ModelInsights';
import { fetchHealth, predictCognitiveLoad } from './api';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [healthData, setHealthData] = useState(null);

  // Default sample cognitive load state
  const [dashboardData, setDashboardData] = useState({
    predicted_state: 'High',
    cognitive_load_score: 78,
    probabilities: { Low: 0.05, Medium: 0.21, High: 0.74 },
    top_contributing_signals: [
      { display_name: 'Error Rate', value: '21.0%', is_elevating_load: true, z_score: 1.8 },
      { display_name: 'Pause Frequency', value: '11 / min', is_elevating_load: true, z_score: 1.5 },
      { display_name: 'Context Switches', value: '8 / min', is_elevating_load: true, z_score: 1.3 },
      { display_name: 'Typing Speed', value: '34 WPM', is_elevating_load: true, z_score: -1.2 }
    ]
  });

  useEffect(() => {
    async function checkSystem() {
      const data = await fetchHealth();
      setHealthData(data);
    }
    checkSystem();
  }, []);

  const handleSimulatePreset = async (presetType) => {
    let presetFeats = {};
    if (presetType === 'low') {
      presetFeats = {
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
      };
    } else if (presetType === 'medium') {
      presetFeats = {
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
      };
    } else {
      presetFeats = {
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
      };
    }

    try {
      const res = await predictCognitiveLoad(presetFeats);
      setDashboardData(res);
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg-dark)' }}>
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} healthData={healthData} />

      <main style={{ maxWidth: '1280px', margin: '0 auto', padding: '32px 24px' }}>
        {activeTab === 'dashboard' && (
          <Dashboard currentData={dashboardData} onSimulatePreset={handleSimulatePreset} />
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
