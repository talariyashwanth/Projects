import React, { useState, useEffect, useRef } from 'react';
import { Terminal, Play, Pause, RotateCcw, Zap, Sparkles, Activity, AlertCircle } from 'lucide-react';
import { predictRawEvents, predictCognitiveLoad } from '../api';

export default function LiveSession({ onUpdateDashboard }) {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [textInput, setTextInput] = useState('');
  const [livePrediction, setLivePrediction] = useState(null);
  const [eventStats, setEventStats] = useState({
    keystrokeCount: 0,
    backspaceCount: 0,
    wpm: 0,
    backspaceRatio: 0,
    pauseCount: 0,
    mouseDistance: 0,
    mouseSpeed: 0,
    clickCount: 0,
    contextSwitches: 0
  });

  // Event Tracking Refs
  const timestampsRef = useRef([]);
  const backspacesRef = useRef(0);
  const keystrokesRef = useRef(0);
  const wordsRef = useRef(0);
  const mousePosRef = useRef([]);
  const clicksRef = useRef(0);
  const lastKeyTimeRef = useRef(Date.now());
  const idlePeriodsRef = useRef([]);
  const contextSwitchesRef = useRef(0);
  const sessionStartTimeRef = useRef(Date.now());

  // Handle Keystroke event
  const handleKeyDown = (e) => {
    if (!isMonitoring) return;

    const now = Date.now();
    const dt = (now - lastKeyTimeRef.current) / 1000.0;
    lastKeyTimeRef.current = now;

    if (dt >= 2.0) {
      idlePeriodsRef.current.push(dt);
    }

    keystrokesRef.current += 1;
    timestampsRef.current.push(now / 1000.0);

    if (e.key === 'Backspace') {
      backspacesRef.current += 1;
    }
  };

  // Handle Mouse movement
  const handleMouseMove = (e) => {
    if (!isMonitoring) return;
    const now = Date.now() / 1000.0;
    mousePosRef.current.push({ x: e.clientX, y: e.clientY, t: now });
    if (mousePosRef.current.length > 100) {
      mousePosRef.current.shift();
    }
  };

  // Handle Clicks
  const handleClick = () => {
    if (!isMonitoring) return;
    clicksRef.current += 1;
  };

  // Start / Stop monitoring session
  const toggleMonitoring = () => {
    if (!isMonitoring) {
      // Start
      setIsMonitoring(true);
      sessionStartTimeRef.current = Date.now();
      timestampsRef.current = [];
      backspacesRef.current = 0;
      keystrokesRef.current = 0;
      wordsRef.current = 0;
      mousePosRef.current = [];
      clicksRef.current = 0;
      idlePeriodsRef.current = [];
      contextSwitchesRef.current = 0;
      setTextInput('');
    } else {
      setIsMonitoring(false);
    }
  };

  // Live Timer for evaluating feature vector every 3 seconds
  useEffect(() => {
    if (!isMonitoring) return;

    const interval = setInterval(async () => {
      const durationSec = Math.max((Date.now() - sessionStartTimeRef.current) / 1000.0, 1.0);
      const wordCount = textInput.trim() ? textInput.trim().split(/\s+/).length : 0;

      const rawPayload = {
        duration_seconds: durationSec,
        keystroke_timestamps: timestampsRef.current,
        backspace_count: backspacesRef.current,
        total_keystrokes: keystrokesRef.current,
        words_typed: wordCount,
        mouse_positions: mousePosRef.current,
        click_count: clicksRef.current,
        idle_periods_seconds: idlePeriodsRef.current,
        error_count: backspacesRef.current,
        attempt_count: keystrokesRef.current,
        response_time_sec: durationSec > 0 ? durationSec / Math.max(wordCount, 1) : 4.0,
        retry_count: backspacesRef.current > 5 ? 2 : 0,
        context_switches: contextSwitchesRef.current
      };

      try {
        const res = await predictRawEvents(rawPayload);
        setLivePrediction(res);
        if (onUpdateDashboard) onUpdateDashboard(res);

        // Update display stats
        const feats = res.extracted_features;
        setEventStats({
          keystrokeCount: keystrokesRef.current,
          backspaceCount: backspacesRef.current,
          wpm: feats.typing_speed_wpm,
          backspaceRatio: (feats.backspace_ratio * 100).toFixed(1),
          pauseCount: feats.pause_frequency_per_min,
          mouseDistance: Math.round(feats.mouse_distance_px),
          mouseSpeed: Math.round(feats.mouse_velocity_px_s),
          clickCount: clicksRef.current,
          contextSwitches: contextSwitchesRef.current
        });
      } catch (err) {
        console.error("Live predict error:", err);
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [isMonitoring, textInput]);

  // Handle Preset Simulation (Stress test vs Relaxed test)
  const triggerPreset = async (type) => {
    let presetFeats = {};
    if (type === 'relaxed') {
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
    } else if (type === 'debugging') {
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
      // Stress / High load
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
      setLivePrediction(res);
      if (onUpdateDashboard) onUpdateDashboard(res);
    } catch (e) {
      console.error(e);
    }
  };

  const state = livePrediction?.predicted_state || 'Low';
  const score = livePrediction?.cognitive_load_score || 25;
  const stateColor = state === 'High' ? '#f43f5e' : state === 'Medium' ? '#f59e0b' : '#10b981';

  return (
    <div
      onMouseMove={handleMouseMove}
      onClick={handleClick}
      style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}
    >
      {/* Task Playground Header */}
      <div className="glass-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#fff', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Terminal size={22} color="#38bdf8" />
            Interactive Live Task Playground
          </h2>
          <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)' }}>
            Real-time keyboard & mouse dynamics feature extraction right in your browser
          </p>
        </div>

        <button
          className={isMonitoring ? 'btn-secondary' : 'btn-primary'}
          onClick={toggleMonitoring}
          style={{ padding: '12px 24px', fontSize: '0.95rem' }}
        >
          {isMonitoring ? <Pause size={18} /> : <Play size={18} />}
          {isMonitoring ? 'Stop Monitoring Session' : 'Start Live Monitoring'}
        </button>
      </div>

      {/* Main Interactive Workspace */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '24px' }}>
        
        {/* Interactive Code / Typing Challenge Box */}
        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.9rem', fontWeight: 700, color: '#fff' }}>
              Interactive Task: Solve Code Challenge / Typing Test
            </span>
            <span style={{ fontSize: '0.78rem', color: isMonitoring ? '#10b981' : 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div className="pulse-dot" style={{ background: isMonitoring ? '#10b981' : '#64748b' }} />
              {isMonitoring ? 'RECORDING BEHAVIOR' : 'PAUSED'}
            </span>
          </div>

          <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
            Try typing comfortable code vs. making intentional typos/backspaces and taking pauses to see live workload score reaction!
          </p>

          <textarea
            className="custom-input font-mono"
            rows={10}
            placeholder={
              isMonitoring
                ? "Start typing code or solution here... (Keystrokes, WPM, pauses, and backspaces are extracted live!)"
                : "Click 'Start Live Monitoring' above to enable keystroke feature extraction!"
            }
            disabled={!isMonitoring}
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            onKeyDown={handleKeyDown}
            style={{
              resize: 'vertical',
              fontSize: '0.9rem',
              lineHeight: '1.6',
              padding: '14px',
              background: '#0a0f1d',
              border: isMonitoring ? '1px solid rgba(56, 189, 248, 0.4)' : '1px solid var(--border-subtle)'
            }}
          />

          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            <button
              className="btn-secondary"
              onClick={() => { contextSwitchesRef.current += 1; }}
              disabled={!isMonitoring}
              style={{ fontSize: '0.78rem' }}
            >
              🔄 Simulate Window Switch
            </button>
            <button
              className="btn-secondary"
              onClick={() => triggerPreset('high')}
              style={{ fontSize: '0.78rem' }}
            >
              ⚡ Instant Stress Preset (High)
            </button>
            <button
              className="btn-secondary"
              onClick={() => triggerPreset('relaxed')}
              style={{ fontSize: '0.78rem' }}
            >
              🌱 Instant Flow Preset (Low)
            </button>
          </div>
        </div>

        {/* Live Metrics & Predictions Monitor */}
        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#fff' }}>
                Real-Time Behavioral Prediction
              </h3>
              <span style={{
                background: `${stateColor}20`,
                color: stateColor,
                border: `1px solid ${stateColor}40`,
                padding: '4px 10px',
                borderRadius: '9999px',
                fontSize: '0.78rem',
                fontWeight: 700
              }}>
                LOAD: {state.toUpperCase()}
              </span>
            </div>

            {/* Score Display Bar */}
            <div style={{ background: '#0a0f1d', padding: '16px', borderRadius: '12px', border: '1px solid var(--border-subtle)', marginBottom: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Cognitive Workload Score</span>
                <span style={{ fontSize: '2rem', fontWeight: 800, color: stateColor }}>{score} / 100</span>
              </div>
              <div style={{ height: '8px', width: '100%', background: 'rgba(255,255,255,0.08)', borderRadius: '4px', marginTop: '8px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${score}%`, background: stateColor, transition: 'width 0.4s ease' }} />
              </div>
            </div>

            {/* Real-time Extracted Behavioral Features Grid */}
            <h4 style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '12px' }}>
              Live Extracted Behavioral Features
            </h4>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '10px 12px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Typing Speed</span>
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#fff' }}>{eventStats.wpm} WPM</div>
              </div>
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '10px 12px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Backspace Ratio</span>
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#fff' }}>{eventStats.backspaceRatio}%</div>
              </div>
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '10px 12px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Pause Count (&gt;2s)</span>
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#fff' }}>{eventStats.pauseCount} / min</div>
              </div>
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '10px 12px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Context Switches</span>
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#fff' }}>{eventStats.contextSwitches} / min</div>
              </div>
            </div>
          </div>

          <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', background: 'rgba(255,255,255,0.02)', padding: '10px', borderRadius: '8px', marginTop: '16px' }}>
            ⚡ Feature vectors evaluate key interaction signals: WPM, Keystroke Intervals, Corrections, Pauses & Context Switches.
          </div>
        </div>
      </div>
    </div>
  );
}
