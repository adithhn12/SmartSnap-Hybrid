/**
 * API client for SmartSnap backend.
 */

const API_BASE = '/api';

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

export async function fetchFuzzyMetadata() {
  const res = await fetch(`${API_BASE}/fuzzy/metadata`);
  if (!res.ok) throw new Error('Failed to fetch fuzzy metadata');
  return res.json();
}

export async function fetchDashboard() {
  const res = await fetch(`${API_BASE}/dashboard`);
  if (!res.ok) throw new Error('Failed to fetch dashboard data');
  return res.json();
}

export async function fetchHistory() {
  const res = await fetch(`${API_BASE}/history`);
  if (!res.ok) throw new Error('Failed to fetch history');
  return res.json();
}

export async function fetchRecord(recordId) {
  const res = await fetch(`${API_BASE}/history/${recordId}`);
  if (!res.ok) throw new Error('Failed to fetch record');
  return res.json();
}

export async function analyzePhoto(file, category) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('category', category);

  const res = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Analysis failed' }));
    throw new Error(err.detail || 'Analysis failed');
  }

  return res.json();
}

export async function simulateFuzzy(inputs, category) {
  const res = await fetch(`${API_BASE}/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ inputs, category }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Simulation failed' }));
    throw new Error(err.detail || 'Simulation failed');
  }

  return res.json();
}

export async function clearHistory() {
  const res = await fetch(`${API_BASE}/history`, { method: 'DELETE' });
  return res.json();
}

