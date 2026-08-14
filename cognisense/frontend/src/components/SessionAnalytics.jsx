import React, { useState, useEffect } from 'react';
import { BarChart3, TrendingUp, RefreshCw, Zap, Coffee, ShieldAlert } from 'lucide-react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { simulateSession } from '../api';

export default function SessionAnalytics() {
  const [sessionData, setSessionData] = useState(null);
  const [loading, setLoading] = useState(true);

  const loadSimulatedSession = async () => {
    setLoading(true);
    try {
      const data = await simulateSession(15, "coding_challenge");
      setSessionData(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSimulatedSession();
  }, []);

  if (loading || !sessionData) {
    return (
      <div className="glass-card" style={{ padding: '60px', textAlign: 'center', color: 'var(--text-secondary)' }}>
        Loading session analytics timeline...
      </div>
    );
  }

  const { average_score, peak_score, time_in_states, recovery_analysis, timeline } = sessionData;

  // Custom Chart Tooltip
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const stateColor = data.predicted_state === 'High' ? '#f43f5e' : data.predicted_state === 'Medium' ? '#f59e0b' : '#10b981';
      return (
        <div style={{
          background: '#0d1320',
          border: '1px solid var(--border-subtle)',
          borderRadius: '10px',
          padding: '12px 16px',
          boxShadow: '0 8px 24px rgba(0,0,0,0.5)'
        }}>
          <div style={{ fontWeight: 700, color: '#fff', fontSize: '0.88rem' }}>Minute {data.minute}</div>
          <div style={{ color: stateColor, fontWeight: 800, fontSize: '1.25rem' }}>
            Score: {data.cognitive_load_score} / 100 ({data.predicted_state})
          </div>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
            WPM: {data.wpm} | Errors: {data.error_rate}% | Pauses: {data.pause_freq}/min
          </div>
          {data.is_break && (
            <div style={{ color: '#38bdf8', fontSize: '0.75rem', fontWeight: 700, marginTop: '4px' }}>
              ☕ Rest Break Taken
            </div>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Session Header */}
      <div className="glass-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#fff', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <BarChart3 size={22} color="#38bdf8" />
            Session Analytics & Load Timeline
          </h2>
          <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)' }}>
            Temporal cognitive load progression over a 15-minute monitored coding session
          </p>
        </div>

        <button className="btn-secondary" onClick={loadSimulatedSession} style={{ fontSize: '0.85rem' }}>
          <RefreshCw size={16} /> Regenerate Session Scenario
        </button>
      </div>

      {/* Top Metric Summary Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px' }}>
        <div className="glass-card" style={{ padding: '20px' }}>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600 }}>SESSION DURATION</span>
          <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#fff', marginTop: '4px' }}>15m 00s</div>
          <div style={{ fontSize: '0.75rem', color: '#10b981', marginTop: '4px' }}>Monitored session</div>
        </div>

        <div className="glass-card" style={{ padding: '20px' }}>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600 }}>AVERAGE WORKLOAD</span>
          <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#f59e0b', marginTop: '4px' }}>{average_score} / 100</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>Medium aggregate load</div>
        </div>

        <div className="glass-card" style={{ padding: '20px' }}>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600 }}>PEAK COGNITIVE LOAD</span>
          <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#f43f5e', marginTop: '4px' }}>{peak_score} / 100</div>
          <div style={{ fontSize: '0.75rem', color: '#f43f5e', marginTop: '4px' }}>High load spike (Min 10-11)</div>
        </div>

        <div className="glass-card" style={{ padding: '20px' }}>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600 }}>RECOVERY IMPACT</span>
          <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#38bdf8', marginTop: '4px' }}>
            -{recovery_analysis.recovery_points} Pts
          </div>
          <div style={{ fontSize: '0.75rem', color: '#38bdf8', marginTop: '4px' }}>Post-break rest delta</div>
        </div>
      </div>

      {/* Main Graph: 15-Minute Load Curve */}
      <div className="glass-card">
        <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#fff', marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <TrendingUp size={18} color="#38bdf8" />
          Estimated Cognitive Load Progression Throughout Task Session
        </h3>

        <div style={{ width: '100%', height: '320px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={timeline} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="loadGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#38bdf8" stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.06)" />
              <XAxis dataKey="minute" stroke="#64748b" tickFormatter={(m) => `${m}m`} />
              <YAxis domain={[0, 100]} stroke="#64748b" />
              <Tooltip content={<CustomTooltip />} />
              <Area type="monotone" dataKey="cognitive_load_score" stroke="#38bdf8" strokeWidth={3} fillOpacity={1} fill="url(#loadGradient)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recovery Feature & Time in States Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '24px' }}>
        
        {/* Recovery Feature Card */}
        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#fff', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Coffee size={18} color="#10b981" />
              Recovery Detection Feature
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Detects whether estimated behavioral load decreases post-rest break.
            </p>

            <div style={{ background: '#0d1320', padding: '16px', borderRadius: '12px', border: '1px solid var(--border-subtle)', margin: '16px 0' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Pre-Break Load (Min 11)</span>
                <span style={{ fontWeight: 700, color: '#f43f5e' }}>{recovery_analysis.pre_break_score} / 100</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Post-Break Load (Min 13)</span>
                <span style={{ fontWeight: 700, color: '#10b981' }}>{recovery_analysis.post_break_score} / 100</span>
              </div>
              <hr style={{ borderColor: 'var(--border-subtle)', margin: '8px 0' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 700 }}>
                <span style={{ color: '#38bdf8' }}>Observed Recovery Delta</span>
                <span style={{ color: '#38bdf8' }}>-{recovery_analysis.recovery_points} Points</span>
              </div>
            </div>
          </div>

          <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', background: 'rgba(255,255,255,0.02)', padding: '10px', borderRadius: '8px' }}>
            💡 A decrease in model score post-break reflects observed behavioral normalization, not proof of complete psychological recovery.
          </div>
        </div>

        {/* Time in States Card */}
        <div className="glass-card">
          <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#fff', marginBottom: '16px' }}>
            Time Spent in Cognitive States
          </h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
                <span style={{ color: 'var(--text-secondary)' }}>🟢 Low Load</span>
                <span style={{ fontWeight: 700, color: '#10b981' }}>{time_in_states.Low}%</span>
              </div>
              <div style={{ height: '8px', background: 'rgba(255,255,255,0.08)', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${time_in_states.Low}%`, background: '#10b981' }} />
              </div>
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
                <span style={{ color: 'var(--text-secondary)' }}>🟡 Medium Load</span>
                <span style={{ fontWeight: 700, color: '#f59e0b' }}>{time_in_states.Medium}%</span>
              </div>
              <div style={{ height: '8px', background: 'rgba(255,255,255,0.08)', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${time_in_states.Medium}%`, background: '#f59e0b' }} />
              </div>
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
                <span style={{ color: 'var(--text-secondary)' }}>🔴 High Load (Overload)</span>
                <span style={{ fontWeight: 700, color: '#f43f5e' }}>{time_in_states.High}%</span>
              </div>
              <div style={{ height: '8px', background: 'rgba(255,255,255,0.08)', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${time_in_states.High}%`, background: '#f43f5e' }} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
