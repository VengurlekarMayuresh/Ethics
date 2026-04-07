'use client';

import React from 'react';
import { CheckCircle, AlertCircle, Info } from 'lucide-react';
import FeatureImportanceBar from './FeatureImportanceBar';

interface AnchorRule {
  rule?: string;
  conditions?: string[];
  precision?: number;
  coverage?: number;
  prediction?: string;
}

interface AlibiExplanationData {
  // Anchor / rule-based output
  anchor?: AnchorRule;
  rules?: AnchorRule[];
  // KernelShap / feature importance fallback
  feature_importance?: Array<{ feature: string; importance: number }>;
  shap_values?: number[];
  // Error case
  status?: string;
  error?: string;
}

interface AlibiRuleDisplayProps {
  explanationData: AlibiExplanationData;
  title?: string;
}

const AlibiRuleDisplay: React.FC<AlibiRuleDisplayProps> = ({
  explanationData,
  title = 'Alibi Explain — Anchor Rules',
}) => {
  // ── Error / unavailable ──────────────────────────────────────────────────
  if (
    explanationData?.status === 'failed' ||
    explanationData?.error
  ) {
    return (
      <div className="rounded-xl border border-amber-200 bg-amber-50 p-6">
        <div className="flex items-start gap-3">
          <AlertCircle className="h-6 w-6 text-amber-500 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="font-semibold text-amber-900">Framework Unavailable</h3>
            <p className="mt-1 text-sm text-amber-800">
              Alibi Explain could not generate an explanation for this prediction.
            </p>
            {explanationData.error && (
              <pre className="mt-3 text-xs bg-amber-100 text-amber-900 rounded p-3 overflow-auto max-h-40 whitespace-pre-wrap">
                {explanationData.error}
              </pre>
            )}
            <p className="mt-3 text-xs text-amber-700">
              Ensure <code className="bg-amber-100 px-1 rounded">alibi[all]</code> and its
              dependencies are installed in the backend environment.
            </p>
          </div>
        </div>
      </div>
    );
  }

  // ── Collect rules to display ──────────────────────────────────────────────
  const rules: AnchorRule[] = [];
  if (explanationData?.anchor) rules.push(explanationData.anchor);
  if (explanationData?.rules?.length) rules.push(...explanationData.rules);

  // ── Feature importance fallback (KernelShap path) ─────────────────────────
  if (rules.length === 0 && explanationData?.feature_importance?.length) {
    return (
      <div className="space-y-4">
        <div className="rounded-xl border border-sky-200 bg-sky-50 p-4 flex items-start gap-3">
          <Info className="h-5 w-5 text-sky-500 flex-shrink-0 mt-0.5" />
          <p className="text-sm text-sky-800">
            This explanation was computed via{' '}
            <strong>KernelShap (Alibi)</strong> — showing feature importances
            rather than anchor rules.
          </p>
        </div>
        <FeatureImportanceBar
          data={explanationData.feature_importance}
          title={title}
          color="#0ea5e9"
          height={350}
        />
      </div>
    );
  }

  // ── No data yet ──────────────────────────────────────────────────────────
  if (rules.length === 0) {
    return (
      <div className="rounded-xl border border-gray-200 bg-gray-50 p-8 text-center text-gray-500">
        No Alibi explanation data available.
      </div>
    );
  }

  // ── Render anchor rules ──────────────────────────────────────────────────
  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-gray-200 bg-white p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-1">{title}</h3>
        <p className="text-sm text-gray-500 mb-5">
          Anchor rules are the minimal set of conditions sufficient to make the
          model predict a given outcome with high precision.
        </p>

        <div className="space-y-4">
          {rules.map((rule, idx) => {
            // Build the rule text from either `rule` string or `conditions` array
            const conditions: string[] =
              rule.conditions?.length
                ? rule.conditions
                : rule.rule
                ? rule.rule.split(/\s+AND\s+/i)
                : [];

            const ruleText =
              rule.rule || conditions.join(' AND ') || '(no conditions)';

            return (
              <div
                key={idx}
                className="rounded-xl border border-sky-200 bg-gradient-to-br from-sky-50 to-indigo-50 p-5"
              >
                {/* Conditions as badge pills */}
                {conditions.length > 0 ? (
                  <div className="flex flex-wrap gap-2 mb-3">
                    {conditions.map((cond, ci) => (
                      <React.Fragment key={ci}>
                        <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-white border border-sky-300 text-sky-800 shadow-sm">
                          {cond.trim()}
                        </span>
                        {ci < conditions.length - 1 && (
                          <span className="inline-flex items-center text-xs font-bold text-gray-400 uppercase tracking-wider self-center">
                            AND
                          </span>
                        )}
                      </React.Fragment>
                    ))}
                  </div>
                ) : (
                  <p className="font-mono text-sm text-gray-800 mb-3">{ruleText}</p>
                )}

                {/* Prediction outcome */}
                {rule.prediction && (
                  <div className="flex items-center gap-2 mb-3">
                    <CheckCircle className="h-4 w-4 text-green-500 flex-shrink-0" />
                    <span className="text-sm font-semibold text-gray-700">
                      Prediction:{' '}
                      <span className="text-green-700">{rule.prediction}</span>
                    </span>
                  </div>
                )}

                {/* Precision & Coverage metrics */}
                {(rule.precision !== undefined || rule.coverage !== undefined) && (
                  <div className="mt-3 pt-3 border-t border-sky-200 flex flex-wrap gap-6">
                    {rule.precision !== undefined && (
                      <div>
                        <p className="text-xs text-gray-500 uppercase tracking-wide font-medium mb-1">
                          Precision
                        </p>
                        <div className="flex items-center gap-2">
                          <div className="h-2 w-24 bg-gray-200 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-green-500 rounded-full"
                              style={{ width: `${Math.min(rule.precision * 100, 100)}%` }}
                            />
                          </div>
                          <span className="text-sm font-bold text-green-700">
                            {(rule.precision * 100).toFixed(1)}%
                          </span>
                        </div>
                        <p className="text-xs text-gray-400 mt-1">
                          How often this rule correctly predicts the outcome
                        </p>
                      </div>
                    )}

                    {rule.coverage !== undefined && (
                      <div>
                        <p className="text-xs text-gray-500 uppercase tracking-wide font-medium mb-1">
                          Coverage
                        </p>
                        <div className="flex items-center gap-2">
                          <div className="h-2 w-24 bg-gray-200 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-sky-500 rounded-full"
                              style={{ width: `${Math.min(rule.coverage * 100, 100)}%` }}
                            />
                          </div>
                          <span className="text-sm font-bold text-sky-700">
                            {(rule.coverage * 100).toFixed(1)}%
                          </span>
                        </div>
                        <p className="text-xs text-gray-400 mt-1">
                          Fraction of data points this rule applies to
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default AlibiRuleDisplay;
