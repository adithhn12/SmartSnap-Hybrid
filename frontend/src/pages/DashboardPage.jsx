import React from 'react';
import { Camera, ArrowRight } from 'lucide-react';

export default function DashboardPage({ stats, onNavigate, onSelectRecord }) {
  const total = stats?.total_analyzed || 0;
  const keep = stats?.keep_count || 0;
  const recover = stats?.recover_count || 0;
  const retake = stats?.retake_count || 0;
  const recent = stats?.recent || [];

  return (
    <div>
      {/* Hero Welcome Section */}
      <div className="card hero-card" style={{ marginBottom: '20px' }}>
        <div className="hero-content">
          <div className="hero-text">
            <h1 className="hero-heading">
              SmartSnap Image Quality & Retake Advisor
            </h1>
            <p className="hero-subtext">
              Evaluate your photograph's quality before keeping it.
            </p>
          </div>
          <div className="hero-actions">
            <button
              className="btn btn-primary hero-btn"
              onClick={() => onNavigate('analysis')}
            >
              <span>Analyse Photograph</span>
            </button>
            <button
              className="btn btn-secondary hero-btn"
              onClick={() => onNavigate('playground')}
            >
              <span>Fuzzy Inference Analysis</span>
            </button>
          </div>
        </div>
      </div>

      {/* Statistics Section (2x2 on mobile, 4-col on desktop) */}
      <div className="stats-grid" style={{ marginBottom: '20px' }}>
        <div className="stat-card">
          <div className="stat-info">
            <div className="stat-val">{total}</div>
            <div className="stat-lbl">Photos Analysed</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-info">
            <div className="stat-val" style={{ color: 'var(--keep-color)' }}>{keep}</div>
            <div className="stat-lbl" style={{ color: 'var(--keep-color)' }}>KEEP</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-info">
            <div className="stat-val" style={{ color: 'var(--recover-color)' }}>{recover}</div>
            <div className="stat-lbl" style={{ color: 'var(--recover-color)' }}>RECOVER</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-info">
            <div className="stat-val" style={{ color: 'var(--retake-color)' }}>{retake}</div>
            <div className="stat-lbl" style={{ color: 'var(--retake-color)' }}>RETAKE</div>
          </div>
        </div>
      </div>

      {/* Recent Evaluations Section */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', flexWrap: 'wrap', gap: '8px' }}>
          <div>
            <h2 className="card-title" style={{ marginBottom: '2px' }}>
              Recent Evaluations
            </h2>
            <p className="card-subtitle" style={{ marginBottom: 0 }}>
              Photographs processed through feature extraction and centroid defuzzification
            </p>
          </div>
          {recent.length > 0 && (
            <button
              className="btn btn-secondary"
              onClick={() => onNavigate('history')}
              style={{ fontSize: '0.8rem', padding: '6px 12px' }}
            >
              <span>View All</span>
              <ArrowRight size={13} />
            </button>
          )}
        </div>

        {recent.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '36px 16px', color: 'var(--text-secondary)' }}>
            <p style={{ fontSize: '0.92rem', fontWeight: 500, color: 'var(--text-primary)', marginBottom: '4px' }}>
              No photographs analysed yet
            </p>
            <p style={{ fontSize: '0.84rem', marginBottom: '16px', color: 'var(--text-secondary)' }}>
              Upload a photograph to view quality metrics and fuzzy recommendation.
            </p>
            <button className="btn btn-primary" onClick={() => onNavigate('analysis')} style={{ fontSize: '0.85rem' }}>
              <Camera size={14} />
              <span>Analyse Photograph</span>
            </button>
          </div>
        ) : (
          <>
            {/* Desktop Table View */}
            <div className="table-container desktop-only">
              <table className="custom-table">
                <thead>
                  <tr>
                    <th style={{ width: '60px' }}>Thumbnail</th>
                    <th>Filename</th>
                    <th>Category</th>
                    <th>Quality Score</th>
                    <th>Recommendation</th>
                    <th>Analysis Date</th>
                    <th style={{ width: '90px' }}>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {recent.map((item) => (
                    <tr key={item.id}>
                      <td>
                        <img
                          src={item.image_url}
                          alt={item.filename}
                          style={{
                            width: '40px',
                            height: '40px',
                            objectFit: 'cover',
                            borderRadius: '4px',
                            border: '1px solid var(--border)',
                            display: 'block'
                          }}
                        />
                      </td>
                      <td style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{item.filename}</td>
                      <td>
                        <span className="badge badge-neutral">{item.category}</span>
                      </td>
                      <td>
                        <span className="mono" style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--text-primary)' }}>
                          {Number(item.score).toFixed(1)} / 100
                        </span>
                      </td>
                      <td>
                        <span className={`badge badge-${item.recommendation.toLowerCase()}`}>
                          {item.recommendation}
                        </span>
                      </td>
                      <td style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>{item.timestamp}</td>
                      <td>
                        <button
                          className="btn btn-secondary"
                          style={{ padding: '5px 10px', fontSize: '0.78rem' }}
                          onClick={() => onSelectRecord(item.id)}
                        >
                          Inspect
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Mobile Card List View */}
            <div className="mobile-card-list mobile-only">
              {recent.map((item) => (
                <div key={item.id} className="mobile-record-card">
                  <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                    <img
                      src={item.image_url}
                      alt={item.filename}
                      style={{
                        width: '48px',
                        height: '48px',
                        objectFit: 'cover',
                        borderRadius: '4px',
                        border: '1px solid var(--border)',
                        flexShrink: 0
                      }}
                    />
                    <div style={{ minWidth: 0, flex: 1 }}>
                      <div style={{ fontWeight: 600, fontSize: '0.88rem', color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {item.filename}
                      </div>
                      <div style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                        {item.category} &bull; {item.timestamp}
                      </div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '10px', paddingTop: '8px', borderTop: '1px solid var(--border)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span className="mono" style={{ fontWeight: 700, fontSize: '0.92rem', color: 'var(--text-primary)' }}>
                        {Number(item.score).toFixed(1)}
                      </span>
                      <span className={`badge badge-${item.recommendation.toLowerCase()}`}>
                        {item.recommendation}
                      </span>
                    </div>
                    <button
                      className="btn btn-secondary"
                      style={{ padding: '5px 12px', fontSize: '0.78rem' }}
                      onClick={() => onSelectRecord(item.id)}
                    >
                      Inspect
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
