'use client';

import React, { useMemo } from 'react';

interface SHAPValue {
  feature: string;
  value: number;
}

interface SHAPForcePlotProps {
  shapValues: SHAPValue[];
  baseValue: number;
  prediction: number;
  title?: string;
}

const SVG_W = 860;
const SVG_H = 170;
const PLOT_L = 40;
const PLOT_R = SVG_W - 40;
const BAR_H = 36;
const BAR_Y = 68;
const TIP_W = 14; // chevron tip width

const SHAPForcePlot: React.FC<SHAPForcePlotProps> = ({
  shapValues,
  baseValue,
  prediction,
  title = 'SHAP Force Plot',
}) => {
  const fmt = (n: number, d = 4) => (Number.isFinite(n) ? n.toFixed(d) : '0.0000');

  // Build axis scale that covers all cumulative segment positions
  const { toX, axisMin, axisMax, axisTicks } = useMemo(() => {
    const positions: number[] = [baseValue, prediction];
    let r = baseValue;
    [...shapValues].filter(s => s.value > 0).sort((a, b) => b.value - a.value).forEach(s => {
      r += s.value; positions.push(r);
    });
    let l = baseValue;
    [...shapValues].filter(s => s.value < 0).sort((a, b) => a.value - b.value).forEach(s => {
      l += s.value; positions.push(l);
    });
    const dMin = Math.min(...positions);
    const dMax = Math.max(...positions);
    const range = dMax - dMin || 1;
    const pad = range * 0.18;
    const axisMin = dMin - pad;
    const axisMax = dMax + pad;
    const toX = (v: number) => PLOT_L + ((v - axisMin) / (axisMax - axisMin)) * (PLOT_R - PLOT_L);

    // Compute nice axis ticks
    const tickCount = 7;
    const step = (axisMax - axisMin) / (tickCount - 1);
    const axisTicks = Array.from({ length: tickCount }, (_, i) => axisMin + i * step);

    return { toX, axisMin, axisMax, axisTicks };
  }, [shapValues, baseValue, prediction]);

  // Compute chevron segments
  const { posSegs, negSegs } = useMemo(() => {
    const positives = shapValues.filter(s => s.value > 0).sort((a, b) => b.value - a.value);
    const negatives = shapValues.filter(s => s.value < 0).sort((a, b) => a.value - b.value);

    type Seg = { x1: number; x2: number; feature: string; value: number; w: number };

    const posSegs: Seg[] = [];
    let rightEdge = baseValue;
    for (const sv of positives) {
      const x1 = toX(rightEdge);
      const x2 = toX(rightEdge + sv.value);
      posSegs.push({ x1, x2, feature: sv.feature, value: sv.value, w: x2 - x1 });
      rightEdge += sv.value;
    }

    const negSegs: Seg[] = [];
    let leftEdge = baseValue;
    for (const sv of negatives) {
      const x1 = toX(leftEdge + sv.value);
      const x2 = toX(leftEdge);
      negSegs.push({ x1, x2, feature: sv.feature, value: sv.value, w: x2 - x1 });
      leftEdge += sv.value;
    }

    return { posSegs, negSegs };
  }, [shapValues, baseValue, toX]);

  // SVG path for right-pointing chevron (positive)
  const rightChevron = (x1: number, x2: number) => {
    const tip = Math.min(TIP_W, (x2 - x1) * 0.45);
    const t = BAR_Y;
    const b = BAR_Y + BAR_H;
    const m = BAR_Y + BAR_H / 2;
    return `M ${x1},${t} L ${x2 - tip},${t} L ${x2},${m} L ${x2 - tip},${b} L ${x1},${b} Z`;
  };

  // SVG path for left-pointing chevron (negative)
  const leftChevron = (x1: number, x2: number) => {
    const tip = Math.min(TIP_W, (x2 - x1) * 0.45);
    const t = BAR_Y;
    const b = BAR_Y + BAR_H;
    const m = BAR_Y + BAR_H / 2;
    return `M ${x2},${t} L ${x2},${b} L ${x1 + tip},${b} L ${x1},${m} L ${x1 + tip},${t} Z`;
  };

  const baseX = toX(baseValue);
  const predX = toX(prediction);
  const axisY = BAR_Y + BAR_H + 8;

  if (!shapValues.length) {
    return (
      <div className="w-full bg-white rounded-lg border border-gray-200 p-8 text-center text-gray-500">
        No SHAP values to display.
      </div>
    );
  }

  return (
    <div className="w-full bg-white rounded-lg border border-gray-200 p-4">
      <h3 className="text-lg font-semibold mb-1">{title}</h3>

      {/* Summary stats */}
      <div className="flex gap-6 text-sm text-gray-600 mb-3">
        <span><span className="font-medium">Base value:</span> <span className="font-mono">{fmt(baseValue)}</span></span>
        <span><span className="font-medium">f(x):</span> <span className="font-mono font-bold text-blue-700">{fmt(prediction)}</span></span>
        <span><span className="font-medium">Net impact:</span> <span className={`font-mono ${prediction - baseValue >= 0 ? 'text-red-600' : 'text-blue-600'}`}>{prediction - baseValue >= 0 ? '+' : ''}{fmt(prediction - baseValue)}</span></span>
      </div>

      <svg
        width="100%"
        viewBox={`0 0 ${SVG_W} ${SVG_H}`}
        className="overflow-visible"
        style={{ fontFamily: 'inherit' }}
      >
        {/* ── "higher" / "lower" direction labels ── */}
        <text x={PLOT_L + 4} y={BAR_Y - 6} fontSize={10} fill="#dc2626" fontWeight={500}>← lower</text>
        <text x={PLOT_R - 4} y={BAR_Y - 6} fontSize={10} fill="#dc2626" fontWeight={500} textAnchor="end">higher →</text>

        {/* ── Negative segments (blue, left-pointing) ── */}
        {negSegs.map((seg, i) => (
          <g key={`neg-${i}`}>
            <path d={leftChevron(seg.x1, seg.x2)} fill={`hsl(213,80%,${65 - i * 4}%)`} stroke="white" strokeWidth={0.8} opacity={0.9} />
            {seg.w > 38 && (
              <text x={(seg.x1 + seg.x2) / 2} y={BAR_Y + BAR_H / 2 + 4} textAnchor="middle" fontSize={9} fill="white" fontWeight={600}>
                {seg.feature}
              </text>
            )}
          </g>
        ))}

        {/* ── Positive segments (red, right-pointing) ── */}
        {posSegs.map((seg, i) => (
          <g key={`pos-${i}`}>
            <path d={rightChevron(seg.x1, seg.x2)} fill={`hsl(0,78%,${62 - i * 4}%)`} stroke="white" strokeWidth={0.8} opacity={0.9} />
            {seg.w > 38 && (
              <text x={(seg.x1 + seg.x2) / 2} y={BAR_Y + BAR_H / 2 + 4} textAnchor="middle" fontSize={9} fill="white" fontWeight={600}>
                {seg.feature}
              </text>
            )}
          </g>
        ))}

        {/* ── Axis line ── */}
        <line x1={PLOT_L} y1={axisY} x2={PLOT_R} y2={axisY} stroke="#9ca3af" strokeWidth={1} />

        {/* ── Axis ticks & labels ── */}
        {axisTicks.map((tick, i) => {
          const tx = toX(tick);
          return (
            <g key={i}>
              <line x1={tx} y1={axisY} x2={tx} y2={axisY + 4} stroke="#9ca3af" strokeWidth={1} />
              <text x={tx} y={axisY + 14} textAnchor="middle" fontSize={8} fill="#6b7280">
                {tick.toFixed(2)}
              </text>
            </g>
          );
        })}

        {/* ── Base value marker ── */}
        <line x1={baseX} y1={BAR_Y - 12} x2={baseX} y2={axisY + 4} stroke="#6b7280" strokeWidth={1.5} strokeDasharray="4,3" />
        <text x={baseX} y={BAR_Y - 16} textAnchor="middle" fontSize={8} fill="#6b7280">base value</text>
        <text x={baseX} y={BAR_Y - 25} textAnchor="middle" fontSize={8} fill="#6b7280">{fmt(baseValue)}</text>

        {/* ── f(x) prediction marker ── */}
        <line x1={predX} y1={BAR_Y - 16} x2={predX} y2={axisY + 4} stroke="#1d4ed8" strokeWidth={2} />
        <polygon
          points={`${predX - 5},${BAR_Y - 16} ${predX + 5},${BAR_Y - 16} ${predX},${BAR_Y - 8}`}
          fill="#1d4ed8"
        />
        <text x={predX} y={BAR_Y - 30} textAnchor="middle" fontSize={10} fontWeight="bold" fill="#1d4ed8">
          f(x) = {fmt(prediction)}
        </text>
      </svg>

      {/* ── Feature legend ── */}
      <div className="mt-3 pt-3 border-t border-gray-100">
        <p className="text-xs text-gray-500 mb-2 font-medium">Feature contributions</p>
        <div className="flex flex-wrap gap-x-5 gap-y-1.5">
          {[...posSegs, ...negSegs]
            .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
            .map((seg, i) => (
              <div key={i} className="flex items-center gap-1.5 text-xs">
                <div
                  className="w-2.5 h-2.5 rounded-sm flex-shrink-0"
                  style={{ backgroundColor: seg.value >= 0 ? '#f87171' : '#60a5fa' }}
                />
                <span className="text-gray-700">{seg.feature}</span>
                <span className={`font-mono font-semibold ${seg.value >= 0 ? 'text-red-600' : 'text-blue-600'}`}>
                  {seg.value > 0 ? '+' : ''}{fmt(seg.value)}
                </span>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
};

export default SHAPForcePlot;
