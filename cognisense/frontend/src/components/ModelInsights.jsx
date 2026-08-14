import React, { useState, useEffect } from 'react';
import { Cpu, Award, BarChart2, ShieldAlert, CheckCircle2 } from 'lucide-react';
import { fetchModelInsights } from '../api';

export default function ModelInsights() {
  const [insights, setInsights] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const data = await fetchModelInsights();
        setInsights(data);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading || !insights) {
    return (
      <div className="glass-card" style={{ padding: '60px', textAlign: 'center', color: 'var(--text-secondary)' }}>
        Loading ML model metrics & evaluation report...
      </div>
    );
  }

  const { comparison, feature_importance, importance_method } = insights;
  const bestModel = comparison.best_model;
  const modelEntries = Object.entries(comparison.models);

  const bestModelData = comparison.models[bestModel];
  const cm = bestModelData?.confusion_matrix || [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const modelNames = modelEntries.map(([name]) => name).join(', ');

  // Largest importance value, used to normalise bar widths so the chart scales
  // to the data instead of relying on a hardcoded multiplier.
  const maxImportance = Math.max(...feature_importance.map((f) => f.importance), 0.0001);

  const cmTotal = cm.flat().reduce((a, b) => a + b, 0);
  const severeErrors = (cm[0]?.[2] || 0) + (cm[2]?.[0] || 0);
  const adjacentErrors =
    (cm[0]?.[1] || 0) + (cm[1]?.[0] || 0) + (cm[1]?.[2] || 0) + (cm[2]?.[1] || 0);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Model Insights Header */}
      <div className="glass-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#fff', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Cpu size={22} color="#a855f7" />
            ML Pipeline & Model Insights Report
          </h2>
          <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)' }}>
            Empirical comparison of {modelNames}
          </p>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            {comparison.split_strategy} · {comparison.cv_strategy}
          </p>
        </div>

        <div style={{
          background: 'rgba(168, 85, 247, 0.15)',
          border: '1px solid rgba(168, 85, 247, 0.3)',
          color: '#c084fc',
          padding: '8px 16px',
          borderRadius: '10px',
          fontWeight: 700,
          fontSize: '0.85rem',
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}>
          <Award size={18} />
          SELECTED MODEL: {bestModel.toUpperCase()}
        </div>
      </div>

      {/* Model Architecture Comparison Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
        {modelEntries.map(([name, data]) => {
          const isSelected = name === bestModel;
          return (
            <div
              key={name}
              className="glass-card"
              style={{
                borderColor: isSelected ? 'rgba(56, 189, 248, 0.5)' : 'var(--border-subtle)',
                background: isSelected ? 'rgba(16, 23, 38, 0.9)' : 'rgba(16, 23, 38, 0.6)'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#fff' }}>{name}</h3>
                {isSelected && (
                  <span style={{ fontSize: '0.72rem', fontWeight: 700, padding: '2px 8px', borderRadius: '4px', background: '#38bdf820', color: '#38bdf8', border: '1px solid #38bdf840' }}>
                    WINNER
                  </span>
                )}
              </div>

              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '16px', minHeight: '36px' }}>
                {data.description}
              </p>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                <div style={{ background: '#0a0f1d', padding: '10px', borderRadius: '8px' }}>
                  <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Test Accuracy</span>
                  <div style={{ fontSize: '1.2rem', fontWeight: 800, color: '#10b981' }}>
                    {(data.test_accuracy * 100).toFixed(1)}%
                  </div>
                </div>

                <div style={{ background: '#0a0f1d', padding: '10px', borderRadius: '8px' }}>
                  <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>F1-Score (Macro)</span>
                  <div style={{ fontSize: '1.2rem', fontWeight: 800, color: '#38bdf8' }}>
                    {(data.f1_macro * 100).toFixed(1)}%
                  </div>
                </div>

                <div style={{ background: '#0a0f1d', padding: '10px', borderRadius: '8px' }}>
                  <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Precision</span>
                  <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc' }}>
                    {(data.precision_macro * 100).toFixed(1)}%
                  </div>
                </div>

                <div style={{ background: '#0a0f1d', padding: '10px', borderRadius: '8px' }}>
                  <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Recall</span>
                  <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc' }}>
                    {(data.recall_macro * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              <div style={{ marginTop: '12px', fontSize: '0.75rem', color: 'var(--text-muted)', textAlign: 'right' }}>
                CV macro-F1: {(data.cv_f1_mean * 100).toFixed(1)}%
                {data.cv_f1_std != null && ` ± ${(data.cv_f1_std * 100).toFixed(1)}%`}
                {isSelected && ' — selection metric'}
              </div>
            </div>
          );
        })}
      </div>

      {/* Feature Importance & Confusion Matrix Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '24px' }}>
        
        {/* Feature Importance Bar Chart */}
        <div className="glass-card">
          <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#fff', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <BarChart2 size={18} color="#38bdf8" />
            Behavioral Feature Importance
          </h3>

          <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '16px', lineHeight: '1.5' }}>
            {importance_method || 'Permutation importance measured on held-out data.'}
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {feature_importance.map((item, idx) => (
              <div key={item.feature}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '4px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>
                    {idx + 1}. {item.feature.replace(/_/g, ' ')}
                  </span>
                  <span style={{ fontWeight: 700, color: '#fff' }}>
                    {(item.importance * 100).toFixed(1)}%
                  </span>
                </div>
                <div style={{ height: '6px', width: '100%', background: 'rgba(255, 255, 255, 0.08)', borderRadius: '3px', overflow: 'hidden' }}>
                  <div style={{
                    height: '100%',
                    width: `${(item.importance / maxImportance) * 100}%`,
                    background: 'linear-gradient(90deg, #3b82f6 0%, #38bdf8 100%)',
                    borderRadius: '3px'
                  }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 3x3 Confusion Matrix Heatmap */}
        <div className="glass-card">
          <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#fff', marginBottom: '16px' }}>
            Confusion Matrix ({bestModel})
          </h3>

          <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
            Actual vs predicted classes on held-out subjects never seen during training:
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: '80px repeat(3, 1fr)', gap: '8px', textAlign: 'center', fontSize: '0.82rem' }}>
            <div></div>
            <div style={{ fontWeight: 700, color: '#10b981' }}>Pred Low</div>
            <div style={{ fontWeight: 700, color: '#f59e0b' }}>Pred Med</div>
            <div style={{ fontWeight: 700, color: '#f43f5e' }}>Pred High</div>

            {['Actual Low', 'Actual Med', 'Actual High'].map((rowLabel, rIdx) => (
              <React.Fragment key={rowLabel}>
                <div style={{ fontWeight: 600, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center' }}>
                  {rowLabel}
                </div>
                {cm[rIdx].map((cellVal, cIdx) => {
                  const isDiag = rIdx === cIdx;
                  return (
                    <div
                      key={cIdx}
                      style={{
                        background: isDiag ? 'rgba(56, 189, 248, 0.15)' : 'rgba(255,255,255,0.03)',
                        border: isDiag ? '1px solid rgba(56, 189, 248, 0.4)' : '1px solid var(--border-subtle)',
                        borderRadius: '8px',
                        padding: '16px 8px',
                        fontWeight: 800,
                        fontSize: '1.1rem',
                        color: isDiag ? '#38bdf8' : 'var(--text-muted)'
                      }}
                    >
                      {cellVal}
                    </div>
                  );
                })}
              </React.Fragment>
            ))}
          </div>

          <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '20px', lineHeight: '1.6' }}>
            <div>
              <strong style={{ color: 'var(--text-secondary)' }}>Severe errors</strong> (Low↔High):{' '}
              {severeErrors} ({cmTotal ? ((severeErrors / cmTotal) * 100).toFixed(1) : '0.0'}%)
            </div>
            <div>
              <strong style={{ color: 'var(--text-secondary)' }}>Adjacent errors</strong> (off-by-one):{' '}
              {adjacentErrors} ({cmTotal ? ((adjacentErrors / cmTotal) * 100).toFixed(1) : '0.0'}%)
            </div>
            <div style={{ marginTop: '8px' }}>
              Errors concentrate on <em>adjacent</em> states — the expected failure mode when
              discretising a continuum, since "Medium" absorbs ambiguity from both boundaries.
            </div>
          </div>
        </div>
      </div>

      {/* Scientific & Ethical Limitations Disclaimer (PRD Section 30) */}
      <div className="glass-card" style={{ borderColor: 'rgba(245, 158, 11, 0.3)', background: 'rgba(245, 158, 11, 0.05)' }}>
        <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#fbbf24', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
          <ShieldAlert size={18} />
          Scientific and Ethical Limitations Disclaimer
        </h3>
        <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
          Cognisense estimates cognitive-load states from observable behavioral interaction patterns. The system is an experimental ML prototype and is not a clinical or psychological diagnostic tool. Monitoring should be transparent, consent-based, and minimize sensitive data collection.
        </p>
        {comparison.data_provenance && (
          <p style={{ fontSize: '0.8rem', color: '#fbbf24', lineHeight: '1.6', marginTop: '10px', fontWeight: 600 }}>
            Data provenance: {comparison.data_provenance}
          </p>
        )}
      </div>
    </div>
  );
}
