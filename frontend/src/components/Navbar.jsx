import React, { useState } from 'react';
import {
  Camera,
  LayoutDashboard,
  Cpu,
  CheckCircle2,
  History,
  Sliders,
  Menu,
  X,
  Sun,
  Moon
} from 'lucide-react';

export default function Navbar({ activePage, setActivePage, theme = 'light', toggleTheme }) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'analysis', label: 'Image Analysis', icon: Camera },
    { id: 'fuzzy', label: 'Fuzzy Engine', icon: Cpu },
    { id: 'results', label: 'Results', icon: CheckCircle2 },
    { id: 'history', label: 'History', icon: History },
    { id: 'playground', label: 'Fuzzy Inference Analysis', icon: Sliders },
  ];

  const handleNavClick = (id) => {
    setActivePage(id);
    setMobileMenuOpen(false);
  };

  const isDark = theme === 'dark';

  return (
    <nav className="navbar">
      <div className="navbar-inner">
        {/* Brand */}
        <div
          className="brand"
          style={{ cursor: 'pointer' }}
          onClick={() => handleNavClick('dashboard')}
        >
          <div className="brand-icon">
            <Camera size={18} strokeWidth={2.2} />
          </div>
          <span>SmartSnap</span>
        </div>

        {/* Right side wrapper: Nav Links + Theme Toggle + Mobile Menu */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {/* Desktop Navigation */}
          <ul className="nav-links desktop-only">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activePage === item.id;
              return (
                <li key={item.id}>
                  <button
                    className={`nav-link ${isActive ? 'active' : ''}`}
                    onClick={() => handleNavClick(item.id)}
                  >
                    <Icon size={15} />
                    <span>{item.label}</span>
                  </button>
                </li>
              );
            })}
          </ul>

          {/* Theme Toggle Button (Always accessible) */}
          <button
            className="theme-toggle-btn"
            onClick={toggleTheme}
            aria-label={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            title={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
          >
            {isDark ? <Sun size={17} /> : <Moon size={17} />}
          </button>

          {/* Mobile Hamburger Toggle Button */}
          <button
            className="mobile-menu-btn"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            aria-label={mobileMenuOpen ? 'Close menu' : 'Open menu'}
            aria-expanded={mobileMenuOpen}
          >
            {mobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
      </div>

      {/* Mobile Navigation Drawer */}
      {mobileMenuOpen && (
        <div className="mobile-nav-drawer">
          <ul className="mobile-nav-list">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activePage === item.id;
              return (
                <li key={item.id}>
                  <button
                    className={`mobile-nav-link ${isActive ? 'active' : ''}`}
                    onClick={() => handleNavClick(item.id)}
                  >
                    <Icon size={18} />
                    <span>{item.label}</span>
                  </button>
                </li>
              );
            })}
            {/* Quick theme toggle row in mobile drawer */}
            <li style={{ borderTop: '1px solid var(--border)', marginTop: '6px', paddingTop: '6px' }}>
              <button
                className="mobile-nav-link"
                onClick={toggleTheme}
                style={{ color: 'var(--text-secondary)' }}
              >
                {isDark ? <Sun size={18} color="var(--accent-primary)" /> : <Moon size={18} color="var(--accent-primary)" />}
                <span>{isDark ? 'Switch to Light Theme' : 'Switch to Dark Theme'}</span>
              </button>
            </li>
          </ul>
        </div>
      )}
    </nav>
  );
}
