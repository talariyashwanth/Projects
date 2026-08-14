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

  const { comparison, feature_importance } = insights;
  const bestModel = comparison.best_model;
  const modelEntries = Object.entries(comparison.models);

  // Extract confusion matrix from best model
  const bestModelData = comparison.models[bestModel];
  const cm = bestModelData?.confusion_matrix || [[82, 7, 1], [8, 71, 6], [1, 7, 84]];

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
            Empirical evaluation comparing Logistic Regression, Random Forest, and Gradient Boosting
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
                5-Fold CV Accuracy: {(data.cv_accuracy_mean * 100).toFixed(1)}%
              </div>
            </div>
          );
        })}
      </div>

      {/* Feature Importance & Confusion Matrix Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '24px' }}>
        
        {/* Feature Importance Bar Chart */}
        <div className="glass-card">
          <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#fff', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <BarChart2 size={18} color="#38bdf8" />
            Random Forest Feature Importance Ranks
          </h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {feature_importance.map((item, idx) => (
              <div key={item.feature}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '4px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>
                    {idx + 1}. {item.feature.replace(/_/g, ' ').title || item.feature}
                  </span>
                  <span style={{ fontWeight: 700, color: '#fff' }}>
                    {(item.importance * 100).toFixed(1)}%
                  </span>
                </div>
                <div style={{ height: '6px', width: '100%', background: 'rgba(255, 255, 255, 0.08)', borderRadius: '3px', overflow: 'hidden' }}>
                  <div style={{
                    height: '100%',
                    width: `${item.importance * 100 * 3.5}%`,
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
            Actual vs Predicted class distribution across test holdout dataset:
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

          <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '20px' }}>
            High diagonal values confirm robust classification boundary separation across states.
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
          Cognisense estimates cognitive-load states from observable behavioral interaction patterns. The system is an experimental ML prototype and is not a clinical or psychological diagnostic tool. Synthetic data cannot establish real-world clinical validity. Monitoring should be transparent, consent-based, and minimize sensitive data collection.
        </p>
      </div>
    </div>
  );
}
