'use client';

import React, { useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
} from 'recharts';

interface SHAPValue {
  feature: string;
  value: number;
  value_formatted?: string;
}

interface SHAPWaterfallProps {
  shapValues: SHAPValue[];
  baseValue: number;
  prediction: number;
  title?: string;
  height?: number | string;
}

const SHAPWaterfall: React.FC<SHAPWaterfallProps> = ({
  shapValues,
  baseValue,
  prediction,
  title = 'SHAP Waterfall Plot',
  height = 500,
}) => {
  const toFiniteNumber = (value: unknown, fallback = 0) => {
    const n = typeof value === 'number' ? value : Number(value);
    return Number.isFinite(n) ? n : fallback;
  };

  const formatNumber = (value: unknown, digits = 4) => {
    const n = toFiniteNumber(value, 0);
    return n.toFixed(digits);
  };

  const safeBaseValue = toFiniteNumber(baseValue, 0);
  const safePrediction = toFiniteNumber(prediction, 0);
  const impact = safePrediction - safeBaseValue;

  // Sort by absolute SHAP value descending, take top 15
  const chartData = useMemo(() => {
    return [...shapValues]
      .sort((a, b) => Math.abs(toFiniteNumber(b.value)) - Math.abs(toFiniteNumber(a.value)))
      .slice(0, 15)
      .map((shap) => ({
        feature: shap.feature,
        value: toFiniteNumber(shap.value, 0),
      }));
  }, [shapValues]);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const shapValue = toFiniteNumber(payload[0]?.value, 0);
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="text-sm font-semibold text-gray-900">{label}</p>
          <p className="text-sm">
            SHAP value:{' '}
            <span className={shapValue >= 0 ? 'text-green-600' : 'text-red-600'}>
              {shapValue >= 0 ? '+' : ''}{formatNumber(shapValue, 4)}
            </span>
          </p>
        </div>
      );
    }
    return null;
  };

  if (!chartData.length) {
    return (
      <div className="w-full bg-white rounded-lg border border-gray-200 p-8 text-center text-gray-500">
        No SHAP values to display.
      </div>
    );
  }

  return (
    <div className="w-full bg-white rounded-lg border border-gray-200 p-4">
      <h3 className="text-lg font-semibold mb-2">{title}</h3>

      <div className="flex items-center gap-4 text-sm text-gray-600 mb-4">
        <div>
          <span className="font-medium">Base value:</span>{' '}
          <span className="font-mono">{formatNumber(safeBaseValue, 4)}</span>
        </div>
        <div>
          <span className="font-medium">Prediction:</span>{' '}
          <span className="font-mono">{formatNumber(safePrediction, 4)}</span>
        </div>
        <div>
          <span className="font-medium">Impact:</span>{' '}
          <span className={`font-mono ${impact >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {formatNumber(impact, 4)}
          </span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 5, right: 60, left: 140, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" horizontal={false} vertical={true} />

          {/* Zero reference line */}
          <ReferenceLine x={0} stroke="#374151" strokeWidth={1.5} />

          <XAxis
            type="number"
            tick={{ fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: '#e5e7eb' }}
            tickFormatter={(v) => formatNumber(v, 3)}
            domain={['auto', 'auto']}
          />

          <YAxis
            type="category"
            dataKey="feature"
            tick={{ fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: '#e5e7eb' }}
            width={130}
            interval={0}
          />

          <Tooltip content={<CustomTooltip />} />

          {/* Single bar with per-cell color: green = positive, red = negative */}
          <Bar dataKey="value" barSize={18} radius={[0, 4, 4, 0]}>
            {chartData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.value >= 0 ? '#10b981' : '#ef4444'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <div className="flex items-center justify-center gap-6 mt-4 text-xs text-gray-600">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded bg-green-500"></div>
          <span>Increases prediction</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded bg-red-500"></div>
          <span>Decreases prediction</span>
        </div>
      </div>
    </div>
  );
};

export default SHAPWaterfall;
