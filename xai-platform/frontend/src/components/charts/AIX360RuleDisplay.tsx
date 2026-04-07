'use client';

import React from 'react';
import { AlertCircle, BookOpen, CheckCircle2 } from 'lucide-react';

interface AIX360Rule {
  rule: string;
  prediction?: string;
  confidence?: number;
  support?: number;
}

interface AIX360ExplanationData {
  rules?: AIX360Rule[];
  feature_importance?: Array<{ feature: string; importance: number }>;
  // Error / unavailable cases
  status?: string;
  error?: string;
}

interface AIX360RuleDisplayProps {
  explanationData: AIX360ExplanationData;
  title?: string;
}

const AIX360RuleDisplay: React.FC<AIX360RuleDisplayProps> = ({
  explanationData,
  title = 'AIX360 — Boolean Rule Explanation',
}) => {
  // ── Framework unavailable ────────────────────────────────────────────────
  if (
    explanationData?.status === 'failed' ||
    explanationData?.error
  ) {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 p-6">
        <div className="flex items-start gap-4">
          <div className="flex-shrink-0 mt-0.5 h-10 w-10 rounded-full bg-red-100 flex items-center justify-center">
            <AlertCircle className="h-6 w-6 text-red-500" />
          </div>
          <div className="flex-1">
            <h3 className="text-base font-semibold text-red-900">
              Framework Unavailable
            </h3>
            <p className="mt-1 text-sm text-red-700">
              AIX360 could not generate an explanation. This is usually caused by
              missing or incompatible dependencies.
            </p>

            {explanationData.error && (
              <pre className="mt-3 text-xs bg-red-100 text-red-900 rounded-lg p-3 overflow-auto max-h-40 whitespace-pre-wrap font-mono">
                {explanationData.error}
              </pre>
            )}

            <div className="mt-4 rounded-lg bg-white border border-red-200 p-4 text-sm text-red-800">
              <p className="font-medium mb-2">To fix this, install the correct versions:</p>
              <pre className="font-mono text-xs bg-red-50 p-2 rounded">
                pip install aix360==0.3.0 scikit-learn==1.3.2 numpy&lt;2.0
              </pre>
              <p className="mt-2 text-xs text-red-600">
                See <code className="bg-red-100 px-1 rounded">notebooks/requirements-xai-frameworks.txt</code> for
                full dependency details.
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const rules: AIX360Rule[] = explanationData?.rules ?? [];

  // ── No data yet ──────────────────────────────────────────────────────────
  if (rules.length === 0) {
    return (
      <div className="rounded-xl border border-gray-200 bg-gray-50 p-8 text-center text-gray-500">
        No AIX360 rule explanation data available.
      </div>
    );
  }

  // ── Render rule list ─────────────────────────────────────────────────────
  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-gray-200 bg-white p-6">
        <div className="flex items-center gap-3 mb-1">
          <BookOpen className="h-5 w-5 text-amber-500" />
          <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
        </div>
        <p className="text-sm text-gray-500 mb-5">
          Boolean rules derived from the model. Each rule captures a set of
          conditions that strongly associate with a particular prediction outcome.
        </p>

        <ol className="space-y-3">
          {rules.map((item, idx) => (
            <li key={idx}>
              <div className="rounded-xl border border-amber-200 bg-gradient-to-r from-amber-50 to-yellow-50 p-4">
                {/* Rule number + text */}
                <div className="flex items-start gap-3">
                  <span className="flex-shrink-0 h-7 w-7 rounded-full bg-amber-200 text-amber-900 text-xs font-bold flex items-center justify-center mt-0.5">
                    {idx + 1}
                  </span>
                  <div className="flex-1">
                    {/* Parse and highlight the rule segments */}
                    <div className="font-mono text-sm text-gray-800 leading-relaxed">
                      {parseAndHighlightRule(item.rule)}
                    </div>

                    {/* Prediction outcome */}
                    {item.prediction && (
                      <div className="flex items-center gap-2 mt-2">
                        <CheckCircle2 className="h-4 w-4 text-amber-600 flex-shrink-0" />
                        <span className="text-sm font-semibold text-amber-800">
                          → {item.prediction}
                        </span>
                      </div>
                    )}

                    {/* Confidence / support */}
                    {(item.confidence !== undefined || item.support !== undefined) && (
                      <div className="mt-3 flex flex-wrap gap-4 text-xs text-gray-600">
                        {item.confidence !== undefined && (
                          <span>
                            <span className="font-medium">Confidence:</span>{' '}
                            <span className="font-mono text-amber-700">
                              {(item.confidence * 100).toFixed(1)}%
                            </span>
                          </span>
                        )}
                        {item.support !== undefined && (
                          <span>
                            <span className="font-medium">Support:</span>{' '}
                            <span className="font-mono text-amber-700">
                              {(item.support * 100).toFixed(1)}%
                            </span>
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </li>
          ))}
        </ol>
      </div>

      {/* Legend */}
      <div className="rounded-lg border border-gray-100 bg-gray-50 p-4 text-xs text-gray-500">
        <p>
          <span className="font-medium text-gray-700">About AIX360 rules:</span>{' '}
          Boolean Rule Column Generation (BRCG) produces minimal, human-readable
          rules that can be audited without a data science background. Each rule
          is a conjunction of simple feature conditions.
        </p>
      </div>
    </div>
  );
};

/** Highlight AND / OR keywords and inequality operators in a rule string. */
function parseAndHighlightRule(rule: string): React.ReactNode {
  if (!rule) return null;

  // Split on AND / OR keywords (case-insensitive), keeping the delimiter
  const parts = rule.split(/(\bAND\b|\bOR\b)/gi);

  return (
    <span>
      {parts.map((part, i) => {
        const upper = part.trim().toUpperCase();
        if (upper === 'AND') {
          return (
            <span key={i} className="mx-1 font-bold text-amber-700">
              AND
            </span>
          );
        }
        if (upper === 'OR') {
          return (
            <span key={i} className="mx-1 font-bold text-orange-600">
              OR
            </span>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </span>
  );
}

export default AIX360RuleDisplay;
