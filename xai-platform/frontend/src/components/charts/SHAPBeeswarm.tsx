'use client';

import React, { useMemo } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts';

interface SHAPBeeswarmProps {
  shapValues: number[][]; // Matrix: [samples x features]
  featureNames: string[];
  sampleValues?: number[][]; // Optional: feature values for coloring
  title?: string;
  height?: number | string;
  maxPointsPerFeature?: number; // Limit for performance
}

interface BeePoint {
  feature: string;
  shap: number;
  jitter: number;
  featureValue?: number;
  index: number;
}

const SHAPBeeswarm: React.FC<SHAPBeeswarmProps> = ({
  shapValues,
  featureNames,
  sampleValues,
  title = 'SHAP Beeswarm Plot',
  height = 600,
  maxPointsPerFeature = 500,
}) => {
  const chartData = useMemo(() => {
    if (!shapValues || shapValues.length === 0 || !featureNames || featureNames.length === 0) {
      return [];
    }

    const points: BeePoint[] = [];
    const numFeatures = featureNames.length;
    const numSamples = shapValues.length;

    // For each feature, collect SHAP values from all samples
    featureNames.forEach((feature, featureIdx) => {
      // Get SHAP values for this feature across all samples
      const featureShapValues = shapValues.map((sample, sampleIdx) => {
        // shapValues can be in different formats depending on model type
        let shap;
        if (Array.isArray(sample)) {
          shap = sample[featureIdx];
        } else if (typeof sample === 'object') {
          shap = sample[featureIdx];
        } else {
          shap = sample;
        }
        return {
          shap: shap || 0,
          sampleIdx,
          featureValue: sampleValues?.[sampleIdx]?.[featureIdx],
        };
      });

      // Sort by SHAP value for better jitter distribution
      featureShapValues.sort((a, b) => a.shap - b.shap);

      // Sample if too many points
      const step = featureShapValues.length > maxPointsPerFeature
        ? Math.ceil(featureShapValues.length / maxPointsPerFeature)
        : 1;
      const sampled = featureShapValues.filter((_, idx) => idx % step === 0);

      // Generate jittered y positions centered on feature index
      sampled.forEach((item, idx) => {
        // Calculate jitter within [-0.4, 0.4] range around the feature position
        const jitter = (Math.random() - 0.5) * 0.8;

        points.push({
          feature,
          shap: item.shap,
          jitter: featureIdx + jitter,
          featureValue: item.featureValue,
          index: item.sampleIdx,
        });
      });
    });

    return points;
  }, [shapValues, featureNames, sampleValues, maxPointsPerFeature]);

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="text-sm font-semibold text-gray-900">{data.feature}</p>
          <p className="text-sm">
            SHAP value:{' '}
            <span className={data.shap >= 0 ? 'text-green-600' : 'text-red-600'}>
              {data.shap >= 0 ? '+' : ''}{data.shap.toFixed(4)}
            </span>
          </p>
          {data.featureValue !== undefined && (
            <p className="text-xs text-gray-500 mt-1">
              Feature value: {typeof data.featureValue === 'number' ? data.featureValue.toFixed(4) : data.featureValue}
            </p>
          )}
          <p className="text-xs text-gray-500">
            Sample #{data.index}
          </p>
        </div>
      );
    }
    return null;
  };

  // Calculate y-axis domain
  const yAxisDomain = useMemo(() => {
    if (!featureNames.length) return [0, 1];
    return [0, featureNames.length - 1];
  }, [featureNames]);

  // Get point color based on SHAP value and optional feature value
  const getPointColor = (shap: number, featureValue?: number) => {
    if (featureValue !== undefined) {
      // Color by feature value using a simple blue gradient
      // Normalize feature value to 0-1 range (simplified)
      const intensity = Math.min(1, Math.max(0, (Number(featureValue) + 10) / 20)); // assuming roughly -10 to 10 range
      const blue = Math.round(100 + 155 * intensity);
      return `rgb(59, 130, ${blue})`;
    }
    // Color by SHAP value direction
    return shap >= 0 ? '#10b981' : '#ef4444';
  };

  return (
    <div className="w-full bg-white rounded-lg border border-gray-200 p-4">
      <h3 className="text-lg font-semibold mb-2">{title}</h3>
      <p className="text-sm text-gray-600 mb-4">
        Each point represents a sample's SHAP value for a feature.
        <span className="text-green-600 font-medium"> Green = increases prediction</span>
        <span className="text-red-600 font-medium ml-2"> Red = decreases prediction</span>
      </p>

      <ResponsiveContainer width="100%" height={height}>
        <ScatterChart
          margin={{
            top: 20,
            right: 40,
            left: 120,
            bottom: 20,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />

          {/* Vertical line at x=0 */}
          <ReferenceLine x={0} stroke="#9ca3af" strokeDasharray="3 3" />

          <XAxis
            type="number"
            dataKey="shap"
            name="SHAP value"
            tick={{ fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: '#e5e7eb' }}
            tickFormatter={(value) => value.toFixed(2)}
          />

          <YAxis
            type="number"
            dataKey="jitter"
            name="Feature"
            tick={{ fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: '#e5e7eb' }}
            tickFormatter={(value, index) => {
              const featureIdx = Math.round(value);
              return featureIdx >= 0 && featureIdx < featureNames.length
                ? featureNames[featureIdx].length > 15
                  ? featureNames[featureIdx].substring(0, 15) + '...'
                  : featureNames[featureIdx]
                : '';
            }}
            domain={yAxisDomain}
            interval={0}
            ticks={featureNames.map((_, idx) => idx)}
            width={110}
          />

          <Tooltip content={<CustomTooltip />} />

          <Scatter
            name="SHAP values"
            data={chartData}
            fill="#3b82f6"
            shape={(props: any) => {
              const { cx, cy, payload } = props;
              const color = getPointColor(payload.shap, payload.featureValue);
              return (
                <circle
                  cx={cx}
                  cy={cy}
                  r={3}
                  fill={color}
                  fillOpacity={0.6}
                  stroke="none"
                />
              );
            }}
          />
        </ScatterChart>
      </ResponsiveContainer>

      {/* Stats summary */}
      <div className="mt-4 text-sm text-gray-600">
        {chartData.length > 0 && (
          <div className="flex items-center justify-center gap-6">
            <div>
              <span className="font-medium">Total points:</span> {chartData.length}
            </div>
            <div>
              <span className="font-medium">Features:</span> {featureNames.length}
            </div>
            <div>
              <span className="font-medium">Positive SHAP:</span>{' '}
              {chartData.filter(p => p.shap > 0).length}
            </div>
            <div>
              <span className="font-medium">Negative SHAP:</span>{' '}
              {chartData.filter(p => p.shap < 0).length}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SHAPBeeswarm;
