import React from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ZAxis,
  Cell
} from 'recharts';

interface SHAPDependenceProps {
  featureName: string;
  xValues: number[];
  shapValues: number[];
  width?: number;
  height?: number;
}

const SHAPDependence: React.FC<SHAPDependenceProps> = ({
  featureName,
  xValues,
  shapValues,
  width = 600,
  height = 400
}) => {
  // Combine data
  const data = xValues.map((x, i) => ({
    x,
    y: shapValues[i]
  }));

  if (data.length === 0) {
    return <div className="p-4 text-gray-500">No data available for dependence plot</div>;
  }

  // Calculate axis ranges
  const xExtent = [Math.min(...xValues), Math.max(...xValues)];
  const yExtent = [Math.min(...shapValues), Math.max(...shapValues)];

  // Color points by SHAP value magnitude
  const maxAbsShap = Math.max(...shapValues.map(Math.abs));

  const getPointColor = (value: number) => {
    if (value > 0) return `rgba(239, 68, 68, 0.7)`; // red-500 with opacity
    return `rgba(59, 130, 246, 0.7)`; // blue-500 with opacity
  };

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-2 border border-gray-300 rounded shadow">
          <p className="text-sm">{`${featureName}: ${payload[0].payload.x.toFixed(4)}`}</p>
          <p className="text-sm">{`SHAP Value: ${payload[0].payload.y.toFixed(4)}`}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div style={{ width, height }}>
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 20, right: 20, bottom: 40, left: 60 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            type="number"
            dataKey="x"
            name={featureName}
            label={{ value: featureName, position: 'bottom', offset: 0 }}
            tick={{ fontSize: 12 }}
            domain={xExtent}
          />
          <YAxis
            type="number"
            dataKey="y"
            name="SHAP Value"
            label={{ value: 'SHAP Value', angle: -90, position: 'left' }}
            tick={{ fontSize: 12 }}
            domain={['auto', 'auto']}
          />
          <Tooltip content={<CustomTooltip />} />
          <Scatter
            name="SHAP Values"
            data={data}
            fill="#8884d8"
            shape="circle"
            isAnimationActive={false}
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getPointColor(entry.y)} />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
};

export default SHAPDependence;
