import React from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';

export default function AggregatedOutputChart({
  aggregatedCurve = [],
  defuzzifiedScore = 50.0,
  recommendation = 'KEEP',
  height = 240,
}) {
  return (
    <div style={{ background: 'var(--bg-surface)', borderRadius: '6px', padding: '16px', border: '1px solid var(--border)', boxShadow: 'var(--shadow-sm)', minWidth: 0, overflow: 'hidden', transition: 'background-color 0.2s ease, border-color 0.2s ease' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px', flexWrap: 'wrap', gap: '10px' }}>
        <div>
          <h4 style={{ fontSize: '0.94rem', fontWeight: 600, color: 'var(--text-primary)' }}>
            Aggregated Fuzzy Output &mu;<sub>out</sub>(y)
          </h4>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            Combined output surface across active rules (Mamdani Max-Union)
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
          <div style={{ textAlign: 'right' }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', display: 'block', textTransform: 'uppercase', letterSpacing: '0.03em', fontWeight: 500 }}>
              Centroid Score (z*)
            </span>
            <span className="mono" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--accent-primary)' }}>
              {Number(defuzzifiedScore).toFixed(1)} / 100
            </span>
          </div>
          <span
            className={`badge badge-${recommendation.toLowerCase()}`}
            style={{ fontSize: '0.82rem', padding: '5px 12px' }}
          >
            {recommendation}
          </span>
        </div>
      </div>

      <div style={{ width: '100%', height, minWidth: 0 }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={aggregatedCurve} margin={{ top: 12, right: 12, left: -25, bottom: 0 }}>
            <defs>
              <linearGradient id="aggLightGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.35} />
                <stop offset="95%" stopColor="#3B82F6" stopOpacity={0.03} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
            <XAxis
              dataKey="quality"
              domain={[0, 100]}
              stroke="var(--chart-axis)"
              fontSize={11}
              tickLine={false}
              label={{ value: 'Image Quality Universe [0, 100]', position: 'insideBottom', offset: -2, fill: 'var(--text-secondary)', fontSize: 10 }}
            />
            <YAxis domain={[0, 1.05]} ticks={[0, 0.5, 1.0]} stroke="var(--chart-axis)" fontSize={11} tickLine={false} />
            <Tooltip
              contentStyle={{ background: 'var(--chart-tooltip-bg)', border: '1px solid var(--chart-tooltip-border)', borderRadius: '4px', fontSize: '11px', color: 'var(--chart-tooltip-text)', boxShadow: 'var(--shadow-sm)' }}
              formatter={(val) => [`${Number(val).toFixed(3)}`, 'Fuzzy Degree μ']}
              labelFormatter={(val) => `Quality y = ${val}`}
            />
            <ReferenceLine
              x={defuzzifiedScore}
              stroke="var(--accent-primary)"
              strokeWidth={2}
              strokeDasharray="4 4"
              label={{
                value: `Centroid z* = ${defuzzifiedScore.toFixed(1)}`,
                position: 'top',
                fill: 'var(--accent-primary)',
                fontSize: 11,
                fontWeight: 600,
              }}
            />
            <Area
              type="monotone"
              dataKey="membership"
              stroke="#3B82F6"
              strokeWidth={2}
              fillOpacity={1}
              fill="url(#aggLightGradient)"
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
