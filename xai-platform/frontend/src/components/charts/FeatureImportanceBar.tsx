'use client';

import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

interface FeatureImportance {
  feature: string;
  importance: number;
}

interface FeatureImportanceBarProps {
  data: FeatureImportance[];
  title?: string;
  height?: number;
  color?: string;
}

const FeatureImportanceBar: React.FC<FeatureImportanceBarProps> = ({
  data,
  title = 'Feature Importance',
  height = 400,
  color = '#3b82f6',
}) => {
  // Sort data by importance descending
  const sortedData = [...data]
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 15); // Show top 15 features

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="text-sm font-semibold text-gray-900">{label}</p>
          <p className="text-sm text-gray-600">
            Importance: <span className="font-mono">{payload[0].value.toFixed(4)}</span>
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full bg-white rounded-lg border border-gray-200 p-4">
      <h3 className="text-lg font-semibold mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart
          data={sortedData}
          layout="vertical"
          margin={{
            top: 5,
            right: 30,
            left: 100,
            bottom: 5,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            type="number"
            tick={{ fontSize: 12 }}
            tickLine={false}
            axisLine={{ stroke: '#e5e7eb' }}
            tickFormatter={(value) => value.toFixed(3)}
          />
          <YAxis
            type="category"
            dataKey="feature"
            tick={{ fontSize: 12 }}
            tickLine={false}
            axisLine={{ stroke: '#e5e7eb' }}
            width={90}
          />
          <Tooltip content={<CustomTooltip />} />
          <Bar
            dataKey="importance"
            fill={color}
            radius={[0, 4, 4, 0]}
            barSize={20}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default FeatureImportanceBar;
