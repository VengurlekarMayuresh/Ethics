'use client';

import React from 'react';
import FeatureImportanceBar from './FeatureImportanceBar';

interface LIMEWeight {
  feature: string;
  weight: number;
  value?: any;
}

interface LIMEPlotProps {
  data: LIMEWeight[];
  intercept?: number;
  localPred?: number;
  title?: string;
  height?: number | string;
}

const LIMEPlot: React.FC<LIMEPlotProps> = ({
  data,
  intercept,
  localPred,
  title = 'LIME Feature Contributions',
  height = 400,
}) => {
  // Transform LIME weights to FeatureImportance format
  const transformedData = data.map(item => ({
    feature: item.feature,
    importance: Math.abs(item.weight), // Use absolute value for importance bar height
  }));

  return (
    <div className="space-y-4">
      <FeatureImportanceBar
        data={transformedData}
        title={title}
        height={height}
        color="#8b5cf6" // Purple for LIME
      />

      {/* LIME-specific info */}
      <div className="bg-gray-50 rounded-lg p-4 text-sm">
        <div className="grid grid-cols-2 gap-4">
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
        <p className="text-xs text-gray-500 mt-2">
          LIME creates a local interpretable model around the prediction.
          Positive weights (blue) increase the prediction, negative weights (red) decrease it.
        </p>
      </div>
    </div>
  );
};

export default LIMEPlot;
