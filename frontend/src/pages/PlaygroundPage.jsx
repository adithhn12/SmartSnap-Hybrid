import React, { useState, useEffect, useCallback } from 'react';
import {
  RotateCcw
} from 'lucide-react';
import { simulateFuzzy } from '../services/api';
import MembershipChart from '../components/MembershipChart';
import AggregatedOutputChart from '../components/AggregatedOutputChart';

const PRESETS = [
  {
    name: 'Ideal Sharp Portrait',
    category: 'Portrait',
    inputs: { blur: 10, brightness: 52, contrast: 80, noise: 8, face_visibility: 92, composition: 85 },
  },
  {
    name: 'Blurry Portrait (Defocus)',
    category: 'Portrait',
    inputs: { blur: 88, brightness: 50, contrast: 55, noise: 15, face_visibility: 30, composition: 60 },
  },
  {
    name: 'Missed Focus (Blurry Subject, Sharp BG)',
    category: 'Portrait',
    inputs: { blur: 82, brightness: 52, contrast: 60, noise: 8, face_visibility: 75, composition: 80 },
  },
  {
    name: 'Severe Motion-Blurred Portrait',
    category: 'Portrait',
    inputs: { blur: 92, brightness: 48, contrast: 40, noise: 7, face_visibility: 68, composition: 70 },
  },
  {
    name: 'Scenic Landscape (0 Faces)',
    category: 'Landscape',
    inputs: { blur: 12, brightness: 54, contrast: 85, noise: 10, face_visibility: 0, composition: 85 },
  },
  {
    name: 'Dark & Low Contrast',
    category: 'General Photography',
    inputs: { blur: 18, brightness: 12, contrast: 20, noise: 35, face_visibility: 25, composition: 50 },
  },
  {
    name: 'Blown Overexposure',
    category: 'General Photography',
    inputs: { blur: 15, brightness: 92, contrast: 25, noise: 8, face_visibility: 40, composition: 50 },
  },
  {
    name: 'High Sensor Grain/Noise',
    category: 'General Photography',
    inputs: { blur: 20, brightness: 50, contrast: 70, noise: 85, face_visibility: 65, composition: 75 },
  },
  {
    name: 'Borderline Recoverable',
    category: 'General Photography',
    inputs: { blur: 48, brightness: 45, contrast: 50, noise: 35, face_visibility: 55, composition: 55 },
  },
];

export default function PlaygroundPage({ metadata }) {
  const [category, setCategory] = useState('General Photography');
  const [inputs, setInputs] = useState({
    blur: 25.0,
    brightness: 52.0,
    contrast: 65.0,
    noise: 15.0,
    face_visibility: 75.0,
    composition: 70.0,
  });

  const [simResult, setSimResult] = useState(null);

  // Debounced API call to backend /api/simulate
  const runSimulation = useCallback(async (currentInputs, currentCategory) => {
    try {
      const res = await simulateFuzzy(currentInputs, currentCategory);
      setSimResult(res);
    } catch (e) {
      console.error('Simulation error:', e);
    }
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => {
      runSimulation(inputs, category);
    }, 120);
    return () => clearTimeout(timer);
  }, [inputs, category, runSimulation]);

  const handleSliderChange = (key, val) => {
    setInputs((prev) => ({
      ...prev,
      [key]: parseFloat(val),
    }));
  };

  const applyPreset = (preset) => {
    setCategory(preset.category);
    setInputs(preset.inputs);
  };

  const resetDefaults = () => {
    setCategory('General Photography');
    setInputs({
      blur: 25.0,
      brightness: 50.0,
      contrast: 60.0,
      noise: 15.0,
      face_visibility: 75.0,
      composition: 70.0,
    });
  };

  const fuzzy = simResult?.fuzzy || {};
  const decision = simResult?.decision || {};
  const memberships = fuzzy.memberships || {};
  const activatedRules = fuzzy.activated_rules || [];
  const aggregatedCurve = fuzzy.aggregated_curve || [];
  const defuzzScore = fuzzy.defuzzified_score ?? 50.0;
  const recommendation = decision.recommendation || 'KEEP';
  const variablesMeta = metadata?.variables || {};

  const sliders = [
    { key: 'blur', label: 'Blur Level', min: 0, max: 100, step: 1, desc: '0 = Sharp, 100 = Severe Blur' },
    { key: 'brightness', label: 'Brightness', min: 0, max: 100, step: 1, desc: '0 = Dark, 50 = Normal, 100 = Overexposed' },
    { key: 'contrast', label: 'Contrast', min: 0, max: 100, step: 1, desc: '0 = Washed, 100 = Dynamic Range' },
    { key: 'noise', label: 'Noise / Grain', min: 0, max: 100, step: 1, desc: '0 = Clean, 100 = Heavy Grain' },
    { key: 'face_visibility', label: 'Face Visibility', min: 0, max: 100, step: 1, desc: '0 = None, 100 = Prominent' },
    { key: 'composition', label: 'Composition', min: 0, max: 100, step: 1, desc: '0 = Poor, 100 = Rule of Thirds' },
  ];

  return (
    <div>
      {/* Title */}
      <div className="page-header-row" style={{ marginBottom: '18px' }}>
        <div>
          <h1 style={{ fontSize: '1.65rem', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.02em', marginBottom: '4px' }}>
            Fuzzy Inference Analysis
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem' }}>
            Adjust photographic inputs in real time to observe Mamdani rule firing and centroid defuzzification.
          </p>
        </div>

        <button className="btn btn-secondary" onClick={resetDefaults} style={{ fontSize: '0.84rem' }}>
          <RotateCcw size={14} />
          <span>Reset Sliders</span>
        </button>
      </div>

      {/* Preset Buttons */}
      <div className="card" style={{ marginBottom: '18px', padding: '14px 18px' }}>
        <div style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
          Test Scenarios
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
          {PRESETS.map((p, i) => (
            <button
              key={i}
              type="button"
              className="btn btn-secondary preset-btn"
              onClick={() => applyPreset(p)}
            >
              {p.name}
            </button>
          ))}
        </div>
      </div>

      <div className="grid-2 analysis-grid" style={{ marginBottom: '20px' }}>
        {/* Left: Interactive Sliders & Category */}
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', flexWrap: 'wrap', gap: '8px' }}>
            <h2 className="card-title">Input Variables</h2>
            <div style={{ width: '170px' }}>
              <select
                className="form-select"
                value={category}
                onChange={(e) => setCategory(e.target.value)}
                style={{ padding: '6px 10px', fontSize: '0.82rem' }}
              >
                <option value="General Photography">General Photography</option>
                <option value="Portrait">Portrait</option>
                <option value="Group Photograph">Group Photograph</option>
                <option value="Landscape">Landscape</option>
                <option value="Document">Document</option>
              </select>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            {sliders.map((s) => {
              const val = inputs[s.key] ?? 50;
              return (
                <div key={s.key} className="slider-container" style={{ marginBottom: '4px' }}>
                  <div className="slider-header">
                    <div>
                      <span className="slider-name">{s.label}</span>
                      <span style={{ fontSize: '0.74rem', color: 'var(--text-secondary)', marginLeft: '6px' }}>({s.desc})</span>
                    </div>
                    <span className="slider-val">{val.toFixed(1)}</span>
                  </div>
                  <input
                    type="range"
                    min={s.min}
                    max={s.max}
                    step={s.step}
                    value={val}
                    onChange={(e) => handleSliderChange(s.key, e.target.value)}
                    className="custom-range"
                  />
                </div>
              );
            })}
          </div>
        </div>

        {/* Right: Live Defuzzification & Decision */}
        <div className="card">
          <h2 className="card-title">Centroid Defuzzification Surface</h2>
          <p className="card-subtitle">
            Calculated in real-time from the aggregated output surface
          </p>

          <AggregatedOutputChart
            aggregatedCurve={aggregatedCurve}
            defuzzifiedScore={defuzzScore}
            recommendation={recommendation}
            height={220}
          />

          <div style={{ marginTop: '14px', backgroundColor: 'var(--bg-subtle-gray)', padding: '12px', borderRadius: '6px', border: '1px solid var(--border)' }}>
            <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '3px' }}>
              FIS Explanation:
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: '1.5', whiteSpace: 'pre-line', wordBreak: 'break-word' }}>
              {decision.explanation || 'Calculating...'}
            </div>
          </div>
        </div>
      </div>

      {/* Activated Rules & Firing Strengths in Playground */}
      <div className="card" style={{ marginBottom: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px', flexWrap: 'wrap', gap: '8px' }}>
          <h2 className="card-title">
            Active Rules & Firing Strengths (&alpha;)
          </h2>
          <span className="badge badge-neutral">{activatedRules.length} Rules Fired</span>
        </div>

        <div className="table-container">
          <table className="custom-table" style={{ minWidth: '600px' }}>
            <thead>
              <tr>
                <th style={{ width: '56px' }}>ID</th>
                <th>Rule Name</th>
                <th>Antecedent Conditions</th>
                <th style={{ width: '90px' }}>Consequent</th>
                <th style={{ width: '130px' }}>Firing Strength (&alpha;)</th>
              </tr>
            </thead>
            <tbody>
              {activatedRules.map((r) => (
                <tr key={r.rule_id}>
                  <td className="mono" style={{ color: 'var(--accent-deep)', fontWeight: 700 }}>{r.rule_id}</td>
                  <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{r.name}</td>
                  <td>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                      {r.antecedents.map(([v, t], idx) => (
                        <span key={idx} style={{ fontSize: '0.74rem', background: 'var(--bg-subtle-gray)', border: '1px solid var(--border)', padding: '2px 6px', borderRadius: '3px', color: 'var(--text-primary)' }}>
                          {v} is <em>{t}</em> (&mu;={memberships[v]?.[t]?.toFixed(2) ?? '0.00'})
                          {idx < r.antecedents.length - 1 && <span style={{ color: 'var(--accent-primary)', marginLeft: '3px' }}>&and;</span>}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td>
                    <span
                      className="badge"
                      style={{
                        background: r.consequent[1] === 'Excellent' || r.consequent[1] === 'Good'
                          ? 'var(--keep-bg)'
                          : r.consequent[1] === 'Moderate'
                          ? 'var(--recover-bg)'
                          : 'var(--retake-bg)',
                        color: r.consequent[1] === 'Excellent' || r.consequent[1] === 'Good'
                          ? 'var(--keep-color)'
                          : r.consequent[1] === 'Moderate'
                          ? 'var(--recover-color)'
                          : 'var(--retake-color)',
                        border: `1px solid ${
                          r.consequent[1] === 'Excellent' || r.consequent[1] === 'Good'
                            ? 'var(--keep-border)'
                            : r.consequent[1] === 'Moderate'
                            ? 'var(--recover-border)'
                            : 'var(--retake-border)'
                        }`,
                      }}
                    >
                      {r.consequent[1]}
                    </span>
                  </td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div style={{ flex: 1, height: '5px', background: 'var(--border)', borderRadius: '3px', overflow: 'hidden' }}>
                        <div
                          style={{
                            width: `${r.firing_strength * 100}%`,
                            height: '100%',
                            background: 'var(--accent-primary)',
                          }}
                        />
                      </div>
                      <span className="mono" style={{ fontWeight: 600, fontSize: '0.82rem', color: 'var(--text-primary)' }}>
                        {r.firing_strength.toFixed(2)}
                      </span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Membership Curves Grid for Playground */}
      <div className="card">
        <h2 className="card-title">Dynamic Input Membership Curves</h2>
        <p className="card-subtitle">
          Real-time red marker tracking across overlapping fuzzy sets as inputs change
        </p>

        <div className="grid-3 charts-grid">
          {sliders.map((s) => {
            const val = inputs[s.key] ?? 50;
            const mems = memberships[s.key] || {};
            const varMeta = variablesMeta[s.key] || {};
            const points = varMeta.curve_points || [];

            return (
              <MembershipChart
                key={s.key}
                title={s.label}
                variableName={s.key}
                curvePoints={points}
                currentValue={val}
                memberships={mems}
                unit="pts"
                height={155}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}
