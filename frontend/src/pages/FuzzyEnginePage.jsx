import React from 'react';
import {
  Sliders
} from 'lucide-react';
import MembershipChart from '../components/MembershipChart';
import AggregatedOutputChart from '../components/AggregatedOutputChart';

export default function FuzzyEnginePage({
  fuzzyResult,
  metadata,
  onNavigate,
  filename = 'Photograph',
}) {
  if (!fuzzyResult || !fuzzyResult.fuzzy) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '56px 20px' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '6px' }}>
          No Fuzzy Evaluation Loaded
        </h2>
        <p style={{ color: 'var(--text-secondary)', maxWidth: '480px', margin: '0 auto 20px', fontSize: '0.9rem' }}>
          To view fuzzification curves, membership degrees, activated rules, and centroid defuzzification, analyse a photograph or use the playground.
        </p>
        <div style={{ display: 'flex', justifyContent: 'center', gap: '10px', flexWrap: 'wrap' }}>
          <button className="btn btn-primary" onClick={() => onNavigate('analysis')}>
            <span>Upload Photograph</span>
          </button>
          <button className="btn btn-secondary" onClick={() => onNavigate('playground')}>
            <span>Open Fuzzy Inference Analysis</span>
          </button>
        </div>
      </div>
    );
  }

  const { fuzzy, decision, category } = fuzzyResult;
  const inputValues = fuzzy.input_values || {};
  const memberships = fuzzy.memberships || {};
  const activatedRules = fuzzy.activated_rules || [];
  const aggregatedCurve = fuzzy.aggregated_curve || [];
  const defuzzScore = fuzzy.defuzzified_score || 50.0;
  const recommendation = decision?.recommendation || 'KEEP';
  const variablesMeta = metadata?.variables || {};

  const inputVars = [
    { key: 'blur', label: 'Blur', unit: 'pts' },
    { key: 'brightness', label: 'Brightness', unit: 'pts' },
    { key: 'contrast', label: 'Contrast', unit: 'pts' },
    { key: 'noise', label: 'Noise', unit: 'pts' },
    { key: 'face_visibility', label: 'Face Visibility', unit: 'pts' },
    { key: 'composition', label: 'Composition', unit: 'pts' },
  ];

  return (
    <div>
      {/* Page Header */}
      <div className="page-header-row" style={{ marginBottom: '20px' }}>
        <div>
          <h1 style={{ fontSize: '1.65rem', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.02em', marginBottom: '4px' }}>
            Fuzzy Inference Diagnostics
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem' }}>
            Target: <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{filename}</span> &bull; Category: <span className="badge badge-neutral">{category}</span>
          </p>
        </div>
        <div className="page-header-actions">
          <button className="btn btn-secondary" onClick={() => onNavigate('results')} style={{ fontSize: '0.84rem' }}>
            <span>View Final Report</span>
          </button>
          <button className="btn btn-primary" onClick={() => onNavigate('playground')} style={{ fontSize: '0.84rem' }}>
            <Sliders size={14} />
            <span>Fuzzy Inference Analysis</span>
          </button>
        </div>
      </div>

      {/* STEP 1: Input Fuzzification & Membership Curves */}
      <div className="card" style={{ marginBottom: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', flexWrap: 'wrap', gap: '8px' }}>
          <div>
            <h2 className="card-title">
              Step 1: Input Fuzzification & Membership Value Assignment
            </h2>
            <p className="card-subtitle" style={{ marginBottom: 0 }}>
              Crisp numerical inputs x &isin; [0, 100] are mapped to linguistic fuzzy sets via triangular and trapezoidal membership functions &mu;<sub>A</sub>(x) &isin; [0, 1].
            </p>
          </div>
          <span className="badge badge-neutral">6 Input Variables</span>
        </div>

        <div className="grid-3 charts-grid">
          {inputVars.map((v) => {
            const val = inputValues[v.key] ?? 50.0;
            const mems = memberships[v.key] || {};
            const varMeta = variablesMeta[v.key] || {};
            const points = varMeta.curve_points || [];

            return (
              <MembershipChart
                key={v.key}
                title={v.label}
                variableName={v.key}
                curvePoints={points}
                currentValue={val}
                memberships={mems}
                unit={v.unit}
                height={165}
              />
            );
          })}
        </div>
      </div>

      {/* STEP 2: Mamdani Rule Evaluation & Firing Strengths */}
      <div className="card" style={{ marginBottom: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', flexWrap: 'wrap', gap: '8px' }}>
          <div>
            <h2 className="card-title">
              Step 2: Fuzzy Rule Evaluation & Min-Conjunction
            </h2>
            <p className="card-subtitle" style={{ marginBottom: 0 }}>
              Antecedents evaluated via Min-conjunction: &alpha; = min(&mu;<sub>1</sub>, &mu;<sub>2</sub>, ...). Consequent fuzzy sets are clipped at firing strength &alpha;.
            </p>
          </div>
          <span style={{ fontSize: '0.84rem', color: 'var(--text-secondary)' }}>
            Activated: <strong style={{ color: 'var(--accent-deep)' }}>{activatedRules.length}</strong> rules
          </span>
        </div>

        <div className="table-container">
          <table className="custom-table" style={{ minWidth: '680px' }}>
            <thead>
              <tr>
                <th style={{ width: '56px' }}>ID</th>
                <th style={{ width: '180px' }}>Rule Name</th>
                <th>Antecedent Conditions (AND = min)</th>
                <th style={{ width: '100px' }}>Consequent</th>
                <th style={{ width: '130px' }}>Firing Strength (&alpha;)</th>
                <th style={{ width: '200px' }}>Significance</th>
              </tr>
            </thead>
            <tbody>
              {activatedRules.length === 0 ? (
                <tr>
                  <td colSpan={6} style={{ textAlign: 'center', padding: '20px', color: 'var(--text-secondary)' }}>
                    No rules activated for the current feature set.
                  </td>
                </tr>
              ) : (
                activatedRules.map((rule) => (
                  <tr key={rule.rule_id}>
                    <td>
                      <span className="mono" style={{ fontWeight: 700, color: 'var(--accent-deep)' }}>{rule.rule_id}</span>
                    </td>
                    <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{rule.name}</td>
                    <td>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                        {rule.antecedents.map(([v, t], idx) => {
                          const deg = memberships[v]?.[t] ?? 0;
                          return (
                            <span
                              key={idx}
                              style={{
                                fontSize: '0.74rem',
                                background: 'var(--bg-subtle-gray)',
                                border: '1px solid var(--border)',
                                padding: '2px 6px',
                                borderRadius: '3px',
                                color: 'var(--text-primary)',
                              }}
                            >
                              <strong>{v}</strong> is <em>{t}</em> (&mu;={deg.toFixed(2)})
                              {idx < rule.antecedents.length - 1 && <span style={{ color: 'var(--accent-primary)', marginLeft: '3px' }}>&and;</span>}
                            </span>
                          );
                        })}
                      </div>
                    </td>
                    <td>
                      <span
                        className="badge"
                        style={{
                          background: rule.consequent[1] === 'Excellent' || rule.consequent[1] === 'Good'
                            ? 'var(--keep-bg)'
                            : rule.consequent[1] === 'Moderate'
                            ? 'var(--recover-bg)'
                            : 'var(--retake-bg)',
                          color: rule.consequent[1] === 'Excellent' || rule.consequent[1] === 'Good'
                            ? 'var(--keep-color)'
                            : rule.consequent[1] === 'Moderate'
                            ? 'var(--recover-color)'
                            : 'var(--retake-color)',
                          border: `1px solid ${
                            rule.consequent[1] === 'Excellent' || rule.consequent[1] === 'Good'
                              ? 'var(--keep-border)'
                              : rule.consequent[1] === 'Moderate'
                              ? 'var(--recover-border)'
                              : 'var(--retake-border)'
                          }`,
                        }}
                      >
                        {rule.consequent[1]}
                      </span>
                    </td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <div style={{ flex: 1, height: '5px', background: 'var(--border)', borderRadius: '3px', overflow: 'hidden' }}>
                          <div
                            style={{
                              width: `${rule.firing_strength * 100}%`,
                              height: '100%',
                              background: 'var(--accent-primary)',
                              borderRadius: '3px',
                            }}
                          />
                        </div>
                        <span className="mono" style={{ fontWeight: 600, fontSize: '0.82rem', color: 'var(--text-primary)' }}>
                          {rule.firing_strength.toFixed(2)}
                        </span>
                      </div>
                    </td>
                    <td style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                      {rule.explanation}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* STEP 3: Aggregation & Centroid Defuzzification */}
      <div className="card">
        <h2 className="card-title">
          Step 3: Output Aggregation & Centroid Defuzzification
        </h2>
        <p className="card-subtitle">
          Max-operator aggregates all truncated consequent sets into composite shape &mu;<sub>out</sub>(y).
          Centroid calculation (Center of Gravity) determines the final crisp quality score z*.
        </p>

        <AggregatedOutputChart
          aggregatedCurve={aggregatedCurve}
          defuzzifiedScore={defuzzScore}
          recommendation={recommendation}
          height={240}
        />
      </div>
    </div>
  );
}
