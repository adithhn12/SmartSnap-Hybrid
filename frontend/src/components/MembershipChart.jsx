import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';

const TERM_COLORS = {
  Low: '#3B82F6',
  Dark: '#3B82F6',
  Poor: '#DC5252',
  Medium: '#D97706',
  Normal: '#16A34A',
  Acceptable: '#3B82F6',
  Average: '#D97706',
  High: '#60A5FA',
  Overexposed: '#DC5252',
  Clear: '#16A34A',
  Good: '#16A34A',
  'Very Poor': '#DC5252',
  Moderate: '#D97706',
  Excellent: '#16A34A',
};

export default function MembershipChart({
  title,
  variableName,
  curvePoints = [],
  currentValue = null,
  memberships = {},
  unit = 'pts',
  height = 170,
}) {
  const terms = curvePoints.length > 0 ? Object.keys(curvePoints[0]).filter(k => k !== 'x') : [];

  return (
    <div style={{ background: 'var(--bg-surface)', borderRadius: '6px', padding: '14px', border: '1px solid var(--border)', boxShadow: 'var(--shadow-sm)', minWidth: 0, overflow: 'hidden', transition: 'background-color 0.2s ease, border-color 0.2s ease' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px', flexWrap: 'wrap', gap: '4px' }}>
        <span style={{ fontWeight: 600, fontSize: '0.88rem', color: 'var(--text-primary)' }}>
          {title || variableName}
        </span>
        {currentValue !== null && (
          <span className="mono" style={{ fontSize: '0.78rem', color: 'var(--accent-deep)', background: 'var(--accent-light)', border: '1px solid var(--accent-border)', padding: '2px 6px', borderRadius: '4px', fontWeight: 600 }}>
            {Number(currentValue).toFixed(1)} {unit}
          </span>
        )}
      </div>

      <div style={{ width: '100%', height, minWidth: 0 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={curvePoints} margin={{ top: 8, right: 8, left: -25, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
            <XAxis dataKey="x" domain={[0, 100]} stroke="var(--chart-axis)" fontSize={10} tickLine={false} />
            <YAxis domain={[0, 1.05]} ticks={[0, 0.5, 1.0]} stroke="var(--chart-axis)" fontSize={10} tickLine={false} />
            <Tooltip
              contentStyle={{ background: 'var(--chart-tooltip-bg)', border: '1px solid var(--chart-tooltip-border)', borderRadius: '4px', fontSize: '11px', color: 'var(--chart-tooltip-text)', boxShadow: 'var(--shadow-sm)' }}
              labelFormatter={(val) => `Input x = ${val}`}
            />
            {currentValue !== null && (
              <ReferenceLine
                x={currentValue}
                stroke="#DC5252"
                strokeWidth={1.8}
                strokeDasharray="3 3"
                label={{ value: `${currentValue.toFixed(1)}`, position: 'top', fill: '#DC5252', fontSize: 10, fontWeight: 600 }}
              />
            )}
            {terms.map((term) => (
              <Line
                key={term}
                type="monotone"
                dataKey={term}
                stroke={TERM_COLORS[term] || '#3B82F6'}
                strokeWidth={1.8}
                dot={false}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Evaluated Membership Degrees */}
      {memberships && Object.keys(memberships).length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: '8px', paddingTop: '6px', borderTop: '1px solid var(--border)' }}>
          {Object.entries(memberships).map(([term, deg]) => {
            const isActive = deg > 0;
            return (
              <div
                key={term}
                style={{
                  fontSize: '0.74rem',
                  padding: '2px 6px',
                  borderRadius: '3px',
                  background: isActive ? 'var(--accent-light)' : 'var(--bg-subtle-gray)',
                  border: isActive ? '1px solid var(--accent-border)' : '1px solid var(--border)',
                  color: isActive ? 'var(--accent-deep)' : 'var(--text-muted)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                }}
              >
                <span>{term}:</span>
                <span className="mono" style={{ fontWeight: isActive ? 700 : 500 }}>
                  {deg.toFixed(2)}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
