import React from 'react';
import { AlertTriangle, CheckCircle2, Flame, ArrowUpRight, ArrowDownRight, Info, HelpCircle } from 'lucide-react';

export default function Dashboard({ currentData, onSimulatePreset }) {
  if (!currentData) return null;

  const { predicted_state, cognitive_load_score, probabilities, top_contributing_signals } = currentData;

  // Determine badge styling based on state
  let stateBadgeClass = 'badge-low';
  let stateColor = '#10b981';
  let stateIcon = CheckCircle2;
  let recommendation = "User is operating comfortably with low hesitation and stable typing cadence.";

  if (predicted_state === 'Medium') {
    stateBadgeClass = 'badge-med';
    stateColor = '#f59e0b';
    stateIcon = Info;
    recommendation = "Moderate mental effort detected. Increased pause frequency and correction rates observed.";
  } else if (predicted_state === 'High') {
    stateBadgeClass = 'badge-high';
    stateColor = '#f43f5e';
    stateIcon = AlertTriangle;
    recommendation = "High cognitive overload detected. Long pauses, high error correction, and frequent context switches.";
  }

  const StateIcon = stateIcon;

  // Key Signal Display List
  const signals = [
    { label: 'Typing Speed', value: '34 WPM', change: '-28%', isElevating: true, desc: 'Words typed per min' },
    { label: 'Backspace Ratio', value: '18.4%', change: '+14%', isElevating: true, desc: 'Correction ratio' },
    { label: 'Pause Frequency', value: '11 / min', change: '+85%', isElevating: true, desc: 'Pauses > 2.0s' },
    { label: 'Mouse Velocity', value: '185 px/s', change: '-32%', isElevating: true, desc: 'Cursor speed' },
    { label: 'Error Rate', value: '22.0%', change: '+18%', isElevating: true, desc: 'Failed attempts' },
    { label: 'Context Switches', value: '8 / min', change: '+120%', isElevating: true, desc: 'Task window switches' },
    { label: 'Response Time', value: '14.2s', change: '+75%', isElevating: true, desc: 'Avg task action delay' },
    { label: 'Idle Time', value: '12.5s', change: '+50%', isElevating: true, desc: 'Inactivity duration' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Top Banner & Preset Quick-Switches */}
      <div className="glass-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#fff', marginBottom: '4px' }}>
            Current Cognitive State Monitor
          </h2>
          <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)' }}>
            Real-time behavioral ML estimation based on interaction dynamics
          </p>
        </div>

        <div style={{ display: 'flex', gap: '10px' }}>
          <button className="btn-secondary" onClick={() => onSimulatePreset('low')} style={{ fontSize: '0.8rem' }}>
            🟢 Relaxed Flow (Low)
          </button>
          <button className="btn-secondary" onClick={() => onSimulatePreset('medium')} style={{ fontSize: '0.8rem' }}>
            🟡 Debugging (Med)
          </button>
          <button className="btn-secondary" onClick={() => onSimulatePreset('high')} style={{ fontSize: '0.8rem' }}>
            🔴 Overload Stress (High)
          </button>
        </div>
      </div>

      {/* Main Grid: Gauge Dial & Probability Distribution */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '24px' }}>
        
        {/* Left Card: 0-100 Score Gauge Dial */}
        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center', position: 'relative' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 700, letterSpacing: '0.1em', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '16px' }}>
            ESTIMATED COGNITIVE LOAD SCORE
          </span>

          <div className="gauge-container">
            {/* SVG Circular Gauge Ring */}
            <svg width="200" height="200" viewBox="0 0 200 200" style={{ transform: 'rotate(-90deg)' }}>
              {/* Background Ring */}
              <circle cx="100" cy="100" r="80" stroke="rgba(255, 255, 255, 0.08)" strokeWidth="16" fill="transparent" />
              {/* Progress Ring */}
              <circle
                cx="100"
                cy="100"
                r="80"
                stroke={stateColor}
                strokeWidth="16"
                fill="transparent"
                strokeDasharray={2 * Math.PI * 80}
                strokeDashoffset={2 * Math.PI * 80 * (1 - cognitive_load_score / 100)}
                strokeLinecap="round"
                style={{ transition: 'stroke-dashoffset 0.8s ease, stroke 0.4s ease', filter: `drop-shadow(0 0 8px ${stateColor})` }}
              />
            </svg>

            {/* Gauge Dial Center Content */}
            <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', textAlign: 'center' }}>
              <div className="gauge-score" style={{ color: stateColor }}>
                {cognitive_load_score}
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>
                OUT OF 100
              </div>
            </div>
          </div>

          <div style={{ marginTop: '20px' }}>
            <span className={stateBadgeClass}>
              <StateIcon size={14} />
              ESTIMATED LOAD: {predicted_state.toUpperCase()}
            </span>
          </div>

          <p style={{ marginTop: '16px', fontSize: '0.85rem', color: 'var(--text-secondary)', maxWidth: '280px' }}>
            {recommendation}
          </p>
        </div>

        {/* Right Card: Class Probability Distribution & Explanation */}
        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#fff', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Flame size={18} color="#38bdf8" />
              ML Classification Probabilities
            </h3>

            {/* Probability Bars */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              {[
                { label: 'Low Cognitive Load (🟢)', key: 'Low', color: '#10b981', val: probabilities.Low },
                { label: 'Medium Cognitive Load (🟡)', key: 'Medium', color: '#f59e0b', val: probabilities.Medium },
                { label: 'High Cognitive Load (🔴)', key: 'High', color: '#f43f5e', val: probabilities.High },
              ].map((item) => (
                <div key={item.key}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '6px' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>{item.label}</span>
                    <span style={{ fontWeight: 700, color: '#fff' }}>{(item.val * 100).toFixed(1)}%</span>
                  </div>
                  <div style={{ height: '8px', width: '100%', background: 'rgba(255, 255, 255, 0.08)', borderRadius: '4px', overflow: 'hidden' }}>
                    <div style={{
                      height: '100%',
                      width: `${item.val * 100}%`,
                      background: item.color,
                      borderRadius: '4px',
                      transition: 'width 0.6s ease'
                    }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Model Note */}
          <div style={{
            background: 'rgba(255, 255, 255, 0.03)',
            padding: '12px 16px',
            borderRadius: '10px',
            border: '1px solid var(--border-subtle)',
            fontSize: '0.78rem',
            color: 'var(--text-muted)',
            marginTop: '20px'
          }}>
            ℹ️ Probabilities are computed from multi-dimensional behavioral feature scaling. Scores strictly describe estimated workload states, not medical condition.
          </div>
        </div>
      </div>

      {/* Explainability Engine: WHY? Section */}
      <div className="glass-card">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
          <div>
            <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#fff', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <HelpCircle size={18} color="#a855f7" />
              Explainability Engine — Why did the model predict {predicted_state.toUpperCase()}?
            </h3>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
              Top behavioral interaction signals contributing to current workload estimate
            </p>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '16px' }}>
          {top_contributing_signals.map((sig, idx) => (
            <div key={idx} style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '12px',
              padding: '16px'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#fff' }}>
                  {sig.display_name}
                </span>
                <span style={{
                  fontSize: '0.75rem',
                  fontWeight: 700,
                  color: sig.is_elevating_load ? '#f43f5e' : '#10b981',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '2px'
                }}>
                  {sig.is_elevating_load ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
                  {sig.is_elevating_load ? '+Load Impact' : '-Load Impact'}
                </span>
              </div>
              <div style={{ fontSize: '1.25rem', fontWeight: 800, color: '#f8fafc' }}>
                {sig.value}
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                Z-score deviation: {sig.z_score > 0 ? `+${sig.z_score}` : sig.z_score} σ
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
