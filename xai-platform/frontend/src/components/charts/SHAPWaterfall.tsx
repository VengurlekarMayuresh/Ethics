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
  // Transform SHAP values into chart data: positive and negative contributions
  const chartData = useMemo(() => {
    return shapValues.map((shap) => ({
      feature: shap.feature,
      value: shap.value,
      // Split into positive and negative for stacked effect
      positive: shap.value > 0 ? shap.value : 0,
      negative: shap.value < 0 ? shap.value : 0,
    }));
  }, [shapValues]);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const shap = shapValues.find(s => s.feature === label);
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="text-sm font-semibold text-gray-900">{label}</p>
          <p className="text-sm">
            SHAP value:{' '}
            <span className={shap.value >= 0 ? 'text-green-600' : 'text-red-600'}>
              {shap.value >= 0 ? '+' : ''}{shap.value.toFixed(4)}
            </span>
          </p>
          {shap.value_formatted && (
            <p className="text-xs text-gray-500">{shap.value_formatted}</p>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full bg-white rounded-lg border border-gray-200 p-4">
      <h3 className="text-lg font-semibold mb-2">{title}</h3>

      <div className="flex items-center gap-4 text-sm text-gray-600 mb-4">
        <div>
          <span className="font-medium">Base value:</span>{' '}
          <span className="font-mono">{baseValue.toFixed(4)}</span>
        </div>
        <div>
          <span className="font-medium">Prediction:</span>{' '}
          <span className="font-mono">{prediction.toFixed(4)}</span>
        </div>
        <div>
          <span className="font-medium">Impact:</span>{' '}
          <span className={`font-mono ${prediction - baseValue >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {(prediction - baseValue).toFixed(4)}
          </span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{
            top: 5,
            right: 30,
            left: 120,
            bottom: 5,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" horizontal={true} vertical={false} />

          {/* Base value reference line */}
          <ReferenceLine
            x={baseValue}
            stroke="#6b7280"
            strokeDasharray="3 3"
            label={{ value: 'Base', position: 'top', fill: '#6b7280', fontSize: 11 }}
          />

          {/* Prediction reference line */}
          <ReferenceLine
            x={prediction}
            stroke="#1d4ed8"
            strokeDasharray="3 3"
            label={{ value: 'Prediction', position: 'top', fill: '#1d4ed8', fontSize: 11 }}
          />

          <XAxis
            type="number"
            tick={{ fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: '#e5e7eb' }}
            tickFormatter={(value) => value.toFixed(3)}
          />

          <YAxis
            type="category"
            dataKey="feature"
            tick={{ fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: '#e5e7eb' }}
            width={110}
            interval={0}
          />

          <Tooltip content={<CustomTooltip />} />

          {/* Positive contribution bars */}
          <Bar
            dataKey="positive"
            name="Positive"
            fill="#10b981"
            stackId="x"
            radius={[0, 4, 4, 0]}
            barSize={20}
          />

          {/* Negative contribution bars */}
          <Bar
            dataKey="negative"
            name="Negative"
            fill="#ef4444"
            stackId="x"
            radius={[0, 4, 4, 0]}
            barSize={20}
          />
        </BarChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 mt-4 text-xs text-gray-600">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded bg-green-500"></div>
          <span>Increases prediction</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded bg-red-500"></div>
          <span>Decreases prediction</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded bg-gray-500"></div>
          <span>Base</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded bg-blue-600"></div>
          <span>Prediction</span>
        </div>
      </div>
    </div>
  );
};

export default SHAPWaterfall;
