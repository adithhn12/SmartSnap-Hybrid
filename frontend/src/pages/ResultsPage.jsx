import React from 'react';
import {
  ArrowUpRight,
  CheckCircle2,
  CircleHelp,
  Cpu,
  Image as ImageIcon,
  Lightbulb,
  RotateCcw,
  Sparkles,
} from 'lucide-react';
import {
  buildMetricPresentations,
  buildPersonalizedFeedback,
  getRecommendationPresentation,
} from './resultsPresentation.js';

const technicalLabels = {
  blur: 'Focus and clarity',
  brightness: 'Lighting',
  contrast: 'Image contrast',
  noise: 'Image grain',
  face_visibility: 'Face detection / visibility',
  object_visibility: 'Primary object visibility',
  composition: 'Framing',
};

function TechnicalValue({ value }) {
  if (value === null || value === undefined) return <span>Not available</span>;
  if (typeof value === 'boolean') return <span>{value ? 'Yes' : 'No'}</span>;
  if (typeof value === 'number') return <span className="mono">{Number.isInteger(value) ? value : value.toFixed(3)}</span>;
  if (Array.isArray(value)) return <span>{value.length ? JSON.stringify(value) : 'None'}</span>;
  if (typeof value === 'object') return <span>{JSON.stringify(value)}</span>;
  return <span>{String(value)}</span>;
}

function TechnicalFeature({ featureKey, item }) {
  const rawFields = Object.entries(item).filter(([key, value]) =>
    !['fuzzy_input_name', 'description', 'explanation', 'raw_stat', 'method', 'faces', 'applicable_to_quality'].includes(key) &&
    (typeof value === 'number' || typeof value === 'boolean' || value === null),
  );
  return (
    <section className="advanced-feature">
      <div className="advanced-feature-heading">
        <h4>{technicalLabels[featureKey] ?? item.fuzzy_input_name ?? featureKey}</h4>
        {item.raw_stat && <span>{item.raw_stat}</span>}
      </div>
      {item.description && <p>{item.description}</p>}
      {item.explanation && <p>{item.explanation}</p>}
      {item.method && <p className="advanced-method">{item.method}</p>}
      {rawFields.length > 0 && (
        <dl className="technical-values">
          {rawFields.map(([name, value]) => (
            <div key={name}>
              <dt>{name.replaceAll('_', ' ')}</dt>
              <dd><TechnicalValue value={value} /></dd>
            </div>
          ))}
        </dl>
      )}
      {featureKey === 'face_visibility' && (
        <>
          <p className="advanced-method">
            Face confidence: {item.confidence ?? 'Not available'}; faces detected: {item.face_count ?? 'Not available'}.
          </p>
          {item.faces?.length > 0 && (
            <p className="advanced-method">Verified face regions: <TechnicalValue value={item.faces} /></p>
          )}
        </>
      )}
      {featureKey === 'blur' && item.faces?.length > 0 && (
        <p className="advanced-method">Verified face regions: <TechnicalValue value={item.faces} /></p>
      )}
    </section>
  );
}

export default function ResultsPage({ record, onNavigate }) {
  if (!record) {
    return (
      <div className="card results-empty">
        <div className="results-empty-icon"><ImageIcon size={22} /></div>
        <h2>No photo results yet</h2>
        <p>Upload a photograph to see a simple quality summary and helpful next steps.</p>
        <button className="btn btn-primary" onClick={() => onNavigate('analysis')}>
          <span>Upload a photo</span>
        </button>
      </div>
    );
  }

  const {
    filename = 'Your photo',
    image_url,
    category = 'General Photography',
    timestamp,
    dimensions = {},
    features = {},
    fuzzy = {},
    decision = {},
    random_forest: randomForest = null,
    diagnostics = {},
  } = record;
  const score = Number.isFinite(Number(fuzzy.defuzzified_score))
    ? Number(fuzzy.defuzzified_score)
    : Number(decision.quality_score) || 0;
  const recommendation = decision.recommendation ?? 'RECOVER';
  const recommendationPresentation = getRecommendationPresentation(recommendation);
  const metrics = buildMetricPresentations(features, fuzzy.memberships, category);
  const feedback = buildPersonalizedFeedback(metrics, recommendation);
  const recommendationClass = recommendationPresentation.className;
  const technicalFeatureEntries = Object.entries(features).filter(([, item]) => item && typeof item === 'object');
  const faceData = features.face_visibility ?? {};
  const faceAssessmentAvailable = category.toLowerCase() === 'portrait photography';
  const categoryAssessment = {
    'Portrait Photography': 'Individual face detail, face exposure and visibility, subject/background focus, and portrait framing are emphasized.',
    'Item Photography': 'Primary-object detail, visibility, exposure and framing are emphasized; background softness is separated and face analysis is not used.',
    'Landscape Photography': 'Regional scene sharpness, exposure balance, and landscape composition are emphasized.',
    'General Photography': 'Overall image focus, lighting, contrast, noise, and composition are assessed.',
  }[category];
  const activatedRules = fuzzy.activated_rules ?? [];
  const fuzzyMemberships = fuzzy.memberships ?? {};
  const fuzzyInputs = fuzzy.input_values ?? {};
  const scoreAngle = `${Math.max(0, Math.min(100, score)) * 3.6}deg`;

  return (
    <div className="results-page">
      <section className={`results-hero results-${recommendationClass}`}>
        <div className="results-hero-copy">
          <div className="results-eyebrow">
            <span className={`results-status-dot ${recommendationClass}`} />
            PHOTO QUALITY RESULT
            {category && <span className="results-category">{category}</span>}
          </div>
          <div className="results-recommendation">
            <div>
              <span className={`results-recommendation-label ${recommendationClass}`}>{recommendation}</span>
              <h1>{recommendationPresentation.title}</h1>
              <p>{recommendation === 'KEEP'
                ? 'Your photo looks good.'
                : recommendation === 'RECOVER'
                  ? 'Good photo with some areas that could be improved.'
                  : 'This photo would benefit from being taken again.'}
              </p>
            </div>
            <div
              className={`quality-score-ring ${recommendationClass}`}
              style={{ '--score-angle': scoreAngle }}
              role="progressbar"
              aria-label="Overall photo quality score"
              aria-valuenow={Math.round(score)}
              aria-valuemin={0}
              aria-valuemax={100}
            >
              <div className="quality-score-center">
                <strong>{score.toFixed(0)}</strong>
                <span>out of 100</span>
              </div>
            </div>
          </div>
          <div className="results-score-caption">
            <Sparkles size={16} aria-hidden="true" />
            <span>Overall photo quality</span>
          </div>
          {categoryAssessment && <p className="results-category-explanation">{categoryAssessment}</p>}
          {timestamp && <div className="results-file-caption">{filename} <span>·</span> {timestamp}</div>}
        </div>

        <figure className="results-photo">
          {image_url ? (
            <img src={image_url} alt={filename} />
          ) : (
            <div className="results-photo-placeholder"><ImageIcon size={32} /><span>Photo preview unavailable</span></div>
          )}
          <figcaption>
            <span>{filename}</span>
            {dimensions.width && dimensions.height && <span>{dimensions.width} × {dimensions.height} px</span>}
          </figcaption>
        </figure>
      </section>

      <section className="results-section results-feedback-section" aria-labelledby="feedback-heading">
        <div className="results-section-heading">
          <div>
            <span className="results-section-kicker">A QUICK READ</span>
            <h2 id="feedback-heading">What stands out</h2>
          </div>
          <p>Based on the details found in this photo</p>
        </div>
        {feedback.length > 0 ? (
          <div className="feedback-grid">
            {feedback.map((item) => (
              <article className={`feedback-card feedback-${item.tone}`} key={item.key}>
                <span className="feedback-icon">
                  {item.tone === 'positive' ? <CheckCircle2 size={18} /> : <Lightbulb size={18} />}
                </span>
                <div>
                  <h3>{item.title}</h3>
                  <p>{item.description}</p>
                </div>
              </article>
            ))}
          </div>
        ) : (
          <div className="feedback-card feedback-neutral">
            <span className="feedback-icon"><CircleHelp size={18} /></span>
            <div><h3>Review the photo</h3><p>Some image details could not be summarized from this saved result.</p></div>
          </div>
        )}
      </section>

      <section className="results-section" aria-labelledby="details-heading">
        <div className="results-section-heading">
          <div>
            <span className="results-section-kicker">IMAGE DETAILS</span>
            <h2 id="details-heading">How the photo looks</h2>
          </div>
          <p>Simple notes on the qualities we checked</p>
        </div>
        <div className="quality-metric-grid">
          {metrics.map((metric) => (
            <article className={`quality-metric-card status-${metric.statusClass}`} key={metric.key}>
              <div className="quality-metric-topline">
                <h3>{metric.label}</h3>
                <span className={`quality-status ${metric.statusClass}`}>{metric.status}</span>
              </div>
              <p>{metric.explanation}</p>
            </article>
          ))}
        </div>
      </section>

      <details className="advanced-analysis">
        <summary>
          <span>
            <span className="results-section-kicker">FOR A CLOSER LOOK</span>
            <strong>Advanced Analysis</strong>
            <small>Model output, measurements, and fuzzy inference details</small>
          </span>
          <span className="advanced-summary-action">Show technical details <ArrowUpRight size={16} /></span>
        </summary>

        <div className="advanced-content">
          <div className="advanced-score-grid">
            <article className="advanced-score-card">
              <span>Random Forest predicted MOS</span>
              {randomForest ? (
                <>
                  <strong>{Number(randomForest.predicted_mos).toFixed(2)} / 5</strong>
                  <p>Normalized score: {Number(randomForest.normalized_quality_score).toFixed(1)} / 100</p>
                  <p>Model version: {randomForest.model_version}</p>
                </>
              ) : <p>This saved result does not include a model prediction.</p>}
            </article>
            <article className="advanced-score-card">
              <span>Mamdani fuzzy score</span>
              <strong>{score.toFixed(2)} / 100</strong>
              <p>Defuzzification: {fuzzy.defuzz_method ?? 'Not available'}</p>
              <p>Recommendation: {recommendation}</p>
            </article>
          </div>

          <section className="advanced-subsection">
            <h3>Computer vision measurements</h3>
            <div className="advanced-feature-grid">
              {technicalFeatureEntries.map(([key, item]) => (
                <TechnicalFeature key={key} featureKey={key} item={item} />
              ))}
            </div>
            {diagnostics && Object.keys(diagnostics).length > 0 && (
              <details className="advanced-nested">
                <summary>Blur and focus diagnostics</summary>
                <dl className="technical-values">
                  {Object.entries(diagnostics).map(([key, value]) => (
                    <div key={key}><dt>{key.replaceAll('_', ' ')}</dt><dd><TechnicalValue value={value} /></dd></div>
                  ))}
                </dl>
              </details>
            )}
          </section>

          {randomForest?.feature_values && (
            <section className="advanced-subsection">
              <h3>Random Forest input features</h3>
              <dl className="technical-values">
                {Object.entries(randomForest.feature_values).map(([key, value]) => (
                  <div key={key}><dt>{key.replaceAll('_', ' ')}</dt><dd><TechnicalValue value={value} /></dd></div>
                ))}
              </dl>
            </section>
          )}

          <section className="advanced-subsection">
            <h3>Fuzzy inference</h3>
            <p className="advanced-method">
              Pipeline: {fuzzy.pipeline ?? 'legacy'} · Category: {fuzzy.category ?? category} ·
              {' '}Activated rules: {fuzzy.activated_count ?? activatedRules.length} / {fuzzy.all_rules_count ?? '—'}
            </p>
            {Object.keys(fuzzyInputs).length > 0 && (
              <>
                <h4>Fuzzy inputs</h4>
                <dl className="technical-values">
                  {Object.entries(fuzzyInputs).map(([key, value]) => (
                    <div key={key}><dt>{key.replaceAll('_', ' ')}</dt><dd><TechnicalValue value={value} /></dd></div>
                  ))}
                </dl>
              </>
            )}
            {Object.keys(fuzzyMemberships).length > 0 && (
              <div className="advanced-memberships">
                {Object.entries(fuzzyMemberships).map(([variable, terms]) => (
                  <article key={variable}>
                    <h4>{variable.replaceAll('_', ' ')}</h4>
                    <dl className="technical-values">
                      {Object.entries(terms).map(([term, value]) => (
                        <div key={term}><dt>{term}</dt><dd><TechnicalValue value={value} /></dd></div>
                      ))}
                    </dl>
                  </article>
                ))}
              </div>
            )}
            {activatedRules.length > 0 && (
              <div className="advanced-rules">
                <h4>Activated rules</h4>
                {activatedRules.map((rule) => (
                  <article key={rule.rule_id}>
                    <div><strong>{rule.rule_id}: {rule.name}</strong><span>α = {Number(rule.firing_strength).toFixed(3)}</span></div>
                    <p>{rule.explanation}</p>
                    <small>IF {rule.antecedents.map(([variable, term]) => `${variable} is ${term}`).join(' AND ')} THEN quality is {rule.consequent?.[1]}</small>
                  </article>
                ))}
              </div>
            )}
            {Array.isArray(fuzzy.aggregated_curve) && fuzzy.aggregated_curve.length > 0 && (
              <details className="advanced-nested">
                <summary>Aggregated output curve ({fuzzy.aggregated_curve.length} samples)</summary>
                <pre>{JSON.stringify(fuzzy.aggregated_curve, null, 2)}</pre>
              </details>
            )}
          </section>

          {(decision.problems?.length > 0 || decision.corrections?.length > 0) && (
            <section className="advanced-subsection advanced-problems-grid">
              {decision.problems?.length > 0 && (
                <div><h3>Detailed findings</h3><ul>{decision.problems.map((problem, index) => <li key={`${index}-${problem}`}>{problem}</li>)}</ul></div>
              )}
              {decision.corrections?.length > 0 && (
                <div><h3>Suggested adjustments</h3><ul>{decision.corrections.map((correction, index) => <li key={`${index}-${correction}`}>{correction}</li>)}</ul></div>
              )}
            </section>
          )}

          {decision.critical_flags?.length > 0 && (
            <section className="advanced-subsection">
              <h3>Critical technical flags</h3>
              <ul>{decision.critical_flags.map((flag, index) => <li key={`${index}-${flag}`}>{flag}</li>)}</ul>
            </section>
          )}

          {decision.explanation && (
            <details className="advanced-nested">
              <summary>Full decision explanation</summary>
              <p className="advanced-explanation">{decision.explanation}</p>
            </details>
          )}

          <div className="advanced-actions">
            <button className="btn btn-secondary" onClick={() => onNavigate('fuzzy')}>
              <Cpu size={16} /> Inspect fuzzy engine
            </button>
            <button className="btn btn-secondary" onClick={() => onNavigate('analysis')}>
              <RotateCcw size={16} /> Analyze another photo
            </button>
          </div>
          {faceAssessmentAvailable && (
            <p className="advanced-method">
              Face detector confidence: {faceData.confidence ?? 'Not available'} ·
              {' '}Detected faces: {faceData.face_count ?? 'Not available'}
            </p>
          )}
        </div>
      </details>
    </div>
  );
}
