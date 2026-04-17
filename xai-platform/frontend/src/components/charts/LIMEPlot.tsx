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

interface LIMEWeight {
  feature: string;
  weight: number;
  value?: any;
}

interface LIMEPlotProps {
  data: LIMEWeight[];
  intercept?: number;
  localPred?: number;
  explainedClass?: string;
  title?: string;
  height?: number;
}

const LIMEPlot: React.FC<LIMEPlotProps> = ({
  data,
  intercept,
  localPred,
  explainedClass,
  title = 'LIME Feature Contributions',
  height = 500,
}) => {
  // Sort by absolute weight descending, show up to 15 features
  const chartData = useMemo(() => {
    return [...data]
      .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
      .slice(0, 15)
      .map(item => ({
        feature: item.feature,
        weight: typeof item.weight === 'number' ? item.weight : 0,
      }));
  }, [data]);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const val: number = payload[0].value ?? 0;
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="text-sm font-semibold text-gray-900">{label}</p>
          <p className="text-sm">
            Weight:{' '}
            <span className={val >= 0 ? 'text-purple-600 font-mono' : 'text-orange-600 font-mono'}>
              {val >= 0 ? '+' : ''}{val.toFixed(4)}
            </span>
          </p>
          <p className="text-xs text-gray-500 mt-1">
            {val >= 0 ? '↑ Increases prediction' : '↓ Decreases prediction'}
          </p>
        </div>
      );
    }
    return null;
  };

  if (!chartData.length) {
    return (
      <div className="w-full bg-white rounded-lg border border-gray-200 p-8 text-center text-gray-500">
        No LIME contribution data available.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="w-full bg-white rounded-lg border border-gray-200 p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">{title}</h3>
          {explainedClass && (
            <div className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-xs font-bold border border-green-200 shadow-sm animate-pulse-subtle">
              Target Outcome: {explainedClass}
            </div>
          )}
        </div>
        <p className="text-xs text-gray-500 mb-4">
          Purple bars increase the prediction · Orange bars decrease it
        </p>

        <ResponsiveContainer width="100%" height={height}>
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 5, right: 80, left: 150, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" horizontal={false} />
            <ReferenceLine x={0} stroke="#6b7280" strokeWidth={1.5} />
            <XAxis
              type="number"
              tick={{ fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: '#e5e7eb' }}
              tickFormatter={(v) => v.toFixed(3)}
            />
            <YAxis
              type="category"
              dataKey="feature"
              tick={{ fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: '#e5e7eb' }}
              width={140}
              interval={0}
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="weight" barSize={18} radius={[0, 4, 4, 0]}>
              {chartData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.weight >= 0 ? '#8b5cf6' : '#f97316'}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* LIME metadata */}
      <div className="bg-gray-50 rounded-lg p-4 text-sm">
        <div className="grid grid-cols-2 gap-4 mb-2">
          {intercept !== undefined && (
            <div>
              <span className="font-medium text-gray-600">Intercept:</span>
              <span className="ml-2 font-mono text-gray-900">{intercept.toFixed(4)}</span>
            </div>
          )}
          {localPred !== undefined && (
            <div>
              <span className="font-medium text-gray-600">Local Prediction:</span>
              <span className="ml-2 font-mono text-gray-900">{localPred.toFixed(4)}</span>
            </div>
          )}
        </div>
        <div className="flex gap-6 text-xs text-gray-600 mt-2">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-purple-500" />
            <span>Positive contribution</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-orange-500" />
            <span>Negative contribution</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LIMEPlot;
