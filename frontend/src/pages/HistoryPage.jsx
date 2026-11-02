import React, { useState } from 'react';
import {
  Trash2,
  Search,
  Eye
} from 'lucide-react';
import { clearHistory } from '../services/api';

export default function HistoryPage({ history = [], onSelectRecord, onRefreshHistory, onNavigate }) {
  const [searchTerm, setSearchTerm] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('All');
  const [recFilter, setRecFilter] = useState('All');
  const [isClearing, setIsClearing] = useState(false);

  const handleClear = async () => {
    if (!window.confirm('Are you sure you want to clear all analysis history?')) return;
    setIsClearing(true);
    try {
      await clearHistory();
      onRefreshHistory();
    } catch (e) {
      alert('Failed to clear history: ' + e.message);
    } finally {
      setIsClearing(false);
    }
  };

  const filtered = history.filter((item) => {
    const matchesSearch = item.filename.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCat = categoryFilter === 'All' || item.category === categoryFilter;
    const matchesRec = recFilter === 'All' || item.decision?.recommendation === recFilter;
    return matchesSearch && matchesCat && matchesRec;
  });

  return (
    <div>
      <div className="page-header-row" style={{ marginBottom: '20px' }}>
        <div>
          <h1 style={{ fontSize: '1.65rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '4px', letterSpacing: '-0.02em' }}>
            Analysis History
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem' }}>
            Archive of all photographs evaluated by the Mamdani Fuzzy Inference Engine
          </p>
        </div>
        <div className="page-header-actions">
          <button className="btn btn-primary" onClick={() => onNavigate('analysis')} style={{ fontSize: '0.84rem' }}>
            <span>Analyse New</span>
          </button>
          {history.length > 0 && (
            <button
              className="btn btn-danger"
              onClick={handleClear}
              disabled={isClearing}
              style={{ fontSize: '0.84rem' }}
            >
              <Trash2 size={14} />
              <span>Clear History</span>
            </button>
          )}
        </div>
      </div>

      {/* Filters Bar */}
      <div className="card" style={{ marginBottom: '18px', padding: '14px 16px' }}>
        <div className="history-filters-grid">
          <div style={{ position: 'relative', flex: '1 1 200px' }}>
            <Search size={15} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
            <input
              type="text"
              placeholder="Search by filename..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="form-input"
              style={{ paddingLeft: '34px' }}
            />
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flex: '1 1 180px' }}>
            <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', fontWeight: 500, whiteSpace: 'nowrap' }}>Category:</span>
            <select
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(e.target.value)}
              className="form-select"
            >
              <option value="All">All Categories</option>
              <option value="Portrait Photography">Portrait Photography</option>
              <option value="Item Photography">Item Photography</option>
              <option value="Landscape Photography">Landscape Photography</option>
              <option value="General Photography">General Photography</option>
            </select>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flex: '1 1 160px' }}>
            <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', fontWeight: 500, whiteSpace: 'nowrap' }}>Decision:</span>
            <select
              value={recFilter}
              onChange={(e) => setRecFilter(e.target.value)}
              className="form-select"
            >
              <option value="All">All Decisions</option>
              <option value="KEEP">KEEP</option>
              <option value="RECOVER">RECOVER</option>
              <option value="RETAKE">RETAKE</option>
            </select>
          </div>
        </div>
      </div>

      {/* Table & Cards */}
      <div className="card">
        {filtered.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '40px 16px', color: 'var(--text-secondary)' }}>
            <p style={{ fontSize: '0.92rem', fontWeight: 500, color: 'var(--text-primary)', marginBottom: '4px' }}>No records match your filters</p>
            <p style={{ fontSize: '0.84rem' }}>Upload photographs or adjust your search filter above.</p>
          </div>
        ) : (
          <>
            {/* Desktop Table View */}
            <div className="table-container desktop-only">
              <table className="custom-table">
                <thead>
                  <tr>
                    <th style={{ width: '56px' }}>Thumbnail</th>
                    <th>Filename</th>
                    <th>Category</th>
                    <th>Quality Score</th>
                    <th>Recommendation</th>
                    <th>Analysis Date</th>
                    <th style={{ width: '90px' }}>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((item) => (
                    <tr key={item.id}>
                      <td>
                        <img
                          src={item.image_url}
                          alt={item.filename}
                          style={{ width: '40px', height: '40px', objectFit: 'cover', borderRadius: '4px', border: '1px solid var(--border)', display: 'block' }}
                        />
                      </td>
                      <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{item.filename}</td>
                      <td>
                        <span className="badge badge-neutral">
                          {item.category}
                        </span>
                      </td>
                      <td>
                        <span className="mono" style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--text-primary)' }}>
                          {Number(item.fuzzy?.defuzzified_score ?? 0).toFixed(1)} / 100
                        </span>
                      </td>
                      <td>
                        <span className={`badge badge-${item.decision?.recommendation?.toLowerCase()}`}>
                          {item.decision?.recommendation}
                        </span>
                      </td>
                      <td style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>{item.timestamp}</td>
                      <td>
                        <button
                          className="btn btn-secondary"
                          style={{ padding: '5px 10px', fontSize: '0.78rem' }}
                          onClick={() => onSelectRecord(item.id)}
                        >
                          <Eye size={13} />
                          <span>Inspect</span>
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Mobile Card List View */}
            <div className="mobile-card-list mobile-only">
              {filtered.map((item) => (
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
                        {Number(item.fuzzy?.defuzzified_score ?? 0).toFixed(1)}
                      </span>
                      <span className={`badge badge-${item.decision?.recommendation?.toLowerCase()}`}>
                        {item.decision?.recommendation}
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
