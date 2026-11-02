import React, { useState, useEffect, useCallback } from 'react';
import Navbar from './components/Navbar';
import DashboardPage from './pages/DashboardPage';
import AnalysisPage from './pages/AnalysisPage';
import FuzzyEnginePage from './pages/FuzzyEnginePage';
import ResultsPage from './pages/ResultsPage';
import HistoryPage from './pages/HistoryPage';
import PlaygroundPage from './pages/PlaygroundPage';
import {
  fetchHealth,
  fetchFuzzyMetadata,
  fetchDashboard,
  fetchHistory,
  fetchRecord
} from './services/api';

export default function App() {
  const [activePage, setActivePage] = useState('dashboard');
  const [metadata, setMetadata] = useState(null);
  const [dashboardStats, setDashboardStats] = useState(null);
  const [historyList, setHistoryList] = useState([]);
  const [currentRecord, setCurrentRecord] = useState(null);

  // Theme Management: light or dark
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('smartsnap-theme');
    if (saved === 'dark' || saved === 'light') return saved;
    // Default to OS system preference
    if (typeof window !== 'undefined' && window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark';
    }
    return 'light';
  });

  // Apply theme to DOM documentElement and save to localStorage
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('smartsnap-theme', theme);
  }, [theme]);

  // Listen to OS system preference changes if user hasn't explicitly set a preference
  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleSystemChange = (e) => {
      const explicit = localStorage.getItem('smartsnap-theme-explicit');
      if (!explicit) {
        setTheme(e.matches ? 'dark' : 'light');
      }
    };
    mediaQuery.addEventListener('change', handleSystemChange);
    return () => mediaQuery.removeEventListener('change', handleSystemChange);
  }, []);

  const toggleTheme = () => {
    setTheme((prev) => {
      const next = prev === 'dark' ? 'light' : 'dark';
      localStorage.setItem('smartsnap-theme-explicit', 'true');
      return next;
    });
  };

  const loadInitialData = useCallback(async () => {
    try {
      await fetchHealth();

      const meta = await fetchFuzzyMetadata();
      setMetadata(meta);

      const stats = await fetchDashboard();
      setDashboardStats(stats);

      const hist = await fetchHistory();
      setHistoryList(hist);

      if (hist && hist.length > 0) {
        setCurrentRecord(hist[0]);
      }
    } catch (e) {
      console.warn('Backend not reachable yet:', e);
    }
  }, []);

  useEffect(() => {
    loadInitialData();
  }, [loadInitialData]);

  const refreshHistory = async () => {
    try {
      const hist = await fetchHistory();
      setHistoryList(hist);
      const stats = await fetchDashboard();
      setDashboardStats(stats);
    } catch (e) {
      console.error('Error refreshing history:', e);
    }
  };

  const handleAnalysisComplete = (newRecord) => {
    setCurrentRecord(newRecord);
    refreshHistory();
    // Navigate straight to Results
    setActivePage('results');
  };

  const handleSelectRecordById = async (recordId) => {
    try {
      const rec = await fetchRecord(recordId);
      setCurrentRecord(rec);
      setActivePage('results');
    } catch (e) {
      console.error('Failed to load record:', e);
    }
  };

  return (
    <div className="app-container">
      <Navbar
        activePage={activePage}
        setActivePage={setActivePage}
        theme={theme}
        toggleTheme={toggleTheme}
      />

      <main className="main-content">
        {activePage === 'dashboard' && (
          <DashboardPage
            stats={dashboardStats}
            onNavigate={setActivePage}
            onSelectRecord={handleSelectRecordById}
          />
        )}

        {activePage === 'analysis' && (
          <AnalysisPage
            onAnalysisComplete={handleAnalysisComplete}
          />
        )}

        {activePage === 'fuzzy' && (
          <FuzzyEnginePage
            fuzzyResult={currentRecord}
            metadata={metadata}
            onNavigate={setActivePage}
            filename={currentRecord?.filename || 'Uploaded Photo'}
          />
        )}

        {activePage === 'results' && (
          <ResultsPage
            record={currentRecord}
            onNavigate={setActivePage}
          />
        )}

        {activePage === 'history' && (
          <HistoryPage
            history={historyList}
            onSelectRecord={handleSelectRecordById}
            onRefreshHistory={refreshHistory}
            onNavigate={setActivePage}
          />
        )}

        {activePage === 'playground' && (
          <PlaygroundPage
            metadata={metadata}
          />
        )}
      </main>
    </div>
  );
}
