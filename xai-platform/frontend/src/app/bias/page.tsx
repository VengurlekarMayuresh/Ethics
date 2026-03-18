'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { useState, useMemo } from 'react';
import {
  Upload,
  Loader2,
  AlertCircle,
  CheckCircle,
  BarChart3,
  Shield,
  RefreshCw,
  ChevronDown,
  FileText,
} from 'lucide-react';
import { format } from 'date-fns';

interface BiasReport {
  _id: string;
  model_id: string;
  model_name?: string;
  protected_attribute: string;
  sensitive_attribute: string;
  demographic_parity_diff: number;
  equal_opportunity_diff: number;
  disparate_impact_ratio: number;
  group_metrics: Record<string, any>;
  dataset_size: number;
  created_at: string;
}

interface Model {
  _id: string;
  name: string;
  framework: string;
  task_type: string;
}

interface AnalysisResult {
  bias_id: string;
  metrics: {
    demographic_parity_diff: number;
    equal_opportunity_diff: number;
    disparate_impact_ratio: number;
    group_metrics: Record<string, any>;
  };
  dataset_size: number;
}

export default function BiasDashboard() {
  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const [protectedAttr, setProtectedAttr] = useState<string>('');
  const [sensitiveAttr, setSensitiveAttr] = useState<string>('');
  const [datasetFile, setDatasetFile] = useState<File | null>(null);
  const [showHistory, setShowHistory] = useState(false);

  const queryClient = useQueryClient();

  // Fetch user's models
  const { data: models, isLoading: modelsLoading } = useQuery<Model[]>({
    queryKey: ['models'],
    queryFn: async () => {
      const { data } = await api.get('/models/');
      return data;
    },
  });

  // Fetch bias reports for selected model
  const { data: reports, isLoading: reportsLoading } = useQuery<BiasReport[]>({
    queryKey: ['biasReports', selectedModelId],
    queryFn: async () => {
      if (!selectedModelId) return [];
      const { data } = await api.get(`/bias/reports/${selectedModelId}`);
      return data;
    },
    enabled: !!selectedModelId && showHistory,
  });

  // Run bias analysis mutation
  const analyzeMutation = useMutation<AnalysisResult, Error>({
    mutationFn: async () => {
      if (!selectedModelId) throw new Error('Please select a model');
      if (!datasetFile) throw new Error('Please upload a dataset');
      if (!protectedAttr || !sensitiveAttr) throw new Error('Please specify protected and sensitive attributes');

      const formData = new FormData();
      formData.append('file', datasetFile);

      const { data } = await api.post('/bias/analyze', formData, {
        params: {
          model_id: selectedModelId,
          protected_attribute: protectedAttr,
          sensitive_attribute: sensitiveAttr,
        },
      });
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['biasReports', selectedModelId] });
      // Reset file
      setDatasetFile(null);
      // Show history
      setShowHistory(true);
    },
  });

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setDatasetFile(e.target.files[0]);
    }
  };

  const handleAnalyze = () => {
    analyzeMutation.mutate();
  };

  const selectedModel = useMemo(() => {
    return models?.find((m) => m._id === selectedModelId);
  }, [models, selectedModelId]);

  // Helper to determine risk level based on metrics
  const getRiskLevel = (report: BiasReport) => {
    const { disparate_impact_ratio, demographic_parity_diff, equal_opportunity_diff } = report;
    let score = 0;
    if (disparate_impact_ratio < 0.8) score += 2;
    if (disparate_impact_ratio < 0.9) score += 1;
    if (demographic_parity_diff > 0.1) score += 1;
    if (equal_opportunity_diff > 0.1) score += 1;

    if (score >= 3) return { level: 'High', color: 'text-red-600 bg-red-100' };
    if (score >= 2) return { level: 'Medium', color: 'text-yellow-600 bg-yellow-100' };
    return { level: 'Low', color: 'text-green-600 bg-green-100' };
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Bias Analysis Dashboard</h1>
        <p className="mt-1 text-sm text-gray-500">
          Evaluate model fairness across different demographic groups.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Analysis Panel */}
        <div className="lg:col-span-1 space-y-6">
          <div className="rounded-lg border border-gray-200 bg-white p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Run Analysis</h2>

            <div className="space-y-4">
              {/* Model Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Select Model
                </label>
                <select
                  value={selectedModelId}
                  onChange={(e) => setSelectedModelId(e.target.value)}
                  disabled={modelsLoading}
                  className="block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 sm:text-sm disabled:bg-gray-50"
                >
                  <option value="">Choose a model...</option>
                  {models?.map((model) => (
                    <option key={model._id} value={model._id}>
                      {model.name} ({model.framework})
                    </option>
                  ))}
                </select>
              </div>

              {/* Attribute Inputs */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Protected Attribute
                  </label>
                  <input
                    type="text"
                    value={protectedAttr}
                    onChange={(e) => setProtectedAttr(e.target.value)}
                    placeholder="e.g., gender, age_group"
                    className="block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 sm:text-sm"
                  />
                  <p className="mt-1 text-xs text-gray-500">Column name in dataset</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Sensitive Attribute
                  </label>
                  <input
                    type="text"
                    value={sensitiveAttr}
                    onChange={(e) => setSensitiveAttr(e.target.value)}
                    placeholder="e.g., outcome, label"
                    className="block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 sm:text-sm"
                  />
                  <p className="mt-1 text-xs text-gray-500">Usually the ground truth or outcome column</p>
                </div>
              </div>

              {/* Dataset Upload */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Evaluation Dataset (CSV)
                </label>
                <div className="mt-1 flex justify-center rounded-md border-2 border-dashed border-gray-300 px-6 pt-5 pb-6 hover:bg-gray-50">
                  <div className="space-y-1 text-center">
                    <Upload className="mx-auto h-8 w-8 text-gray-400" />
                    <div className="flex text-sm text-gray-600 justify-center">
                      <label htmlFor="dataset-upload" className="relative cursor-pointer rounded-md font-medium text-indigo-600 hover:text-indigo-500">
                        <span>Upload a file</span>
                        <input
                          id="dataset-upload"
                          type="file"
                          accept=".csv"
                          className="sr-only"
                          onChange={handleFileChange}
                          disabled={analyzeMutation.isPending}
                        />
                      </label>
                      <p className="pl-1">or drag and drop</p>
                    </div>
                    <p className="text-xs text-gray-500">CSV up to 50MB</p>
                    {datasetFile && (
                      <p className="text-sm font-medium text-indigo-600">{datasetFile.name}</p>
                    )}
                  </div>
                </div>
              </div>

              {/* Error message */}
              {analyzeMutation.isError && (
                <div className="rounded-md bg-red-50 p-3 border border-red-200">
                  <p className="text-sm text-red-800">
                    {(analyzeMutation.error as Error)?.message || 'Analysis failed'}
                  </p>
                </div>
              )}

              {/* Submit button */}
              <button
                onClick={handleAnalyze}
                disabled={analyzeMutation.isPending || !selectedModelId || !datasetFile || !protectedAttr || !sensitiveAttr}
                className="w-full flex justify-center items-center rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 transition"
              >
                {analyzeMutation.isPending ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Shield className="mr-2 h-4 w-4" />
                    Run Bias Analysis
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Model Info */}
          {selectedModel && (
            <div className="rounded-lg border border-gray-200 bg-white p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-3">Selected Model</h3>
              <div className="space-y-3 text-sm">
                <div>
                  <p className="text-gray-500">Name</p>
                  <p className="font-medium text-gray-900">{selectedModel.name}</p>
                </div>
                <div>
                  <p className="text-gray-500">Framework</p>
                  <p className="font-medium text-gray-900 capitalize">{selectedModel.framework}</p>
                </div>
                <div>
                  <p className="text-gray-500">Task Type</p>
                  <p className="font-medium text-gray-900">{selectedModel.task_type}</p>
                </div>
              </div>
            </div>
          )}

          {/* Recent Reports */}
          <div className="rounded-lg border border-gray-200 bg-white p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Recent Analyses</h3>
              <button
                onClick={() => setShowHistory(!showHistory)}
                className="text-sm text-indigo-600 hover:text-indigo-800 flex items-center"
              >
                {showHistory ? 'Hide' : 'Show'}
                <ChevronDown className={`ml-1 h-4 w-4 transform ${showHistory ? 'rotate-180' : ''}`} />
              </button>
            </div>

            {reportsLoading ? (
              <div className="flex justify-center py-4">
                <Loader2 className="h-6 w-6 animate-spin text-indigo-600" />
              </div>
            ) : reports && reports.length > 0 ? (
              <div className="space-y-3">
                {reports.slice(0, 5).map((report) => {
                  const risk = getRiskLevel(report);
                  return (
                    <div
                      key={report._id}
                      className="rounded-lg border border-gray-200 p-3 hover:bg-gray-50 cursor-pointer"
                      onClick={() => {
                        // Could expand details
                      }}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm font-medium text-gray-900">
                          {report.protected_attribute} vs {report.sensitive_attribute}
                        </span>
                        <span className={`text-xs px-2 py-0.5 rounded-full ${risk.color}`}>
                          {risk.level} Risk
                        </span>
                      </div>
                      <p className="text-xs text-gray-500">
                        {format(new Date(report.created_at), 'MMM d, yyyy HH:mm')} • {report.dataset_size} records
                      </p>
                      <div className="mt-2 text-xs">
                        <div className="flex justify-between">
                          <span className="text-gray-600">Disparate Impact:</span>
                          <span className={`font-mono ${report.disparate_impact_ratio < 0.8 ? 'text-red-600' : 'text-green-600'}`}>
                            {report.disparate_impact_ratio.toFixed(3)}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">DP Diff:</span>
                          <span className="font-mono">{report.demographic_parity_diff.toFixed(3)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">EO Diff:</span>
                          <span className="font-mono">{report.equal_opportunity_diff.toFixed(3)}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="text-sm text-gray-500 text-center py-4">No analyses yet</p>
            )}
          </div>
        </div>

        {/* Main Content - Analysis Result */}
        <div className="lg:col-span-2">
          {analyzeMutation.isSuccess && (
            <div className="space-y-6">
              <div className="rounded-lg border border-green-200 bg-green-50 p-4 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <CheckCircle className="h-5 w-5 text-green-600" />
                  <div>
                    <p className="text-sm font-medium text-green-800">Analysis complete</p>
                    <p className="text-xs text-green-700">
                      Protected: {protectedAttr} | Sensitive: {sensitiveAttr}
                    </p>
                  </div>
                </div>
                <div className="text-sm font-mono text-green-700">
                  ID: {analyzeMutation.data.bias_id.slice(0, 8)}...
                </div>
              </div>

              {/* Key Metrics Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <MetricCard
                  title="Disparate Impact Ratio"
                  value={analyzeMutation.data.metrics.disparate_impact_ratio}
                  threshold={0.8}
                  tooltip="Ratio of positive outcome rates between groups. Values >= 0.8 indicate fairness."
                  format={(v) => v.toFixed(3)}
                />
                <MetricCard
                  title="Demographic Parity Difference"
                  value={analyzeMutation.data.metrics.demographic_parity_diff}
                  threshold={0.1}
                  tooltip="Difference in positive outcome rates between groups. Lower is better."
                  format={(v) => v.toFixed(3)}
                  inverse={false}
                />
                <MetricCard
                  title="Equal Opportunity Difference"
                  value={analyzeMutation.data.metrics.equal_opportunity_diff}
                  threshold={0.1}
                  tooltip="Difference in true positive rates between groups. Lower is better."
                  format={(v) => v.toFixed(3)}
                  inverse={false}
                />
              </div>

              {/* Group Metrics Table */}
              <div className="rounded-lg border border-gray-200 bg-white p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Group-wise Metrics</h3>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                          Group
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                          Positive Rate
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                          True Positive Rate
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                          False Positive Rate
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                          Accuracy
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200 bg-white">
                      {Object.entries(analyzeMutation.data.metrics.group_metrics).map(([group, metrics]: [string, any]) => (
                        <tr key={group}>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                            {group}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 font-mono">
                            {(metrics.positive_rate * 100).toFixed(1)}%
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 font-mono">
                            {(metrics.true_positive_rate * 100).toFixed(1)}%
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 font-mono">
                            {(metrics.false_positive_rate * 100).toFixed(1)}%
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 font-mono">
                            {(metrics.accuracy * 100).toFixed(1)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Interpretation */}
              <div className="rounded-lg border border-gray-200 bg-white p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-3">Interpretation</h3>
                <div className="space-y-3 text-sm text-gray-700">
                  <p>
                    <strong>Disparate Impact Ratio (DIR):</strong> {' '}
                    {analyzeMutation.data.metrics.disparate_impact_ratio >= 0.8 ? (
                      <span className="text-green-600">This model meets the 80% fairness threshold (DIR &gt;= 0.8).</span>
                    ) : (
                      <span className="text-red-600">Warning: This model may have adverse impact (DIR &lt; 0.8). Consider mitigation.</span>
                    )}
                  </p>
                  <p>
                    <strong>Demographic Parity Difference:</strong> {' '}
                    {analyzeMutation.data.metrics.demographic_parity_diff <= 0.1 ? (
                      <span className="text-green-600">The difference in positive outcome rates is small ({analyzeMutation.data.metrics.demographic_parity_diff.toFixed(3)}), indicating fairness.</span>
                    ) : (
                      <span className="text-yellow-600">The difference ({analyzeMutation.data.metrics.demographic_parity_diff.toFixed(3)}) may be significant. Review your model and data.</span>
                    )}
                  </p>
                  <p>
                    <strong>Equal Opportunity Difference:</strong> {' '}
                    {analyzeMutation.data.metrics.equal_opportunity_diff <= 0.1 ? (
                      <span className="text-green-600">True positive rates are similar across groups.</span>
                    ) : (
                      <span className="text-yellow-600">There is a notable difference in true positive rates. This may indicate unequal opportunity.</span>
                    )}
                  </p>
                </div>
              </div>
            </div>
          )}

          {!analyzeMutation.isSuccess && !analyzeMutation.isPending && (
            <div className="rounded-lg border border-gray-200 bg-white p-12 text-center">
              <Shield className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-4 text-lg font-semibold text-gray-900">No Analysis Yet</h3>
              <p className="mt-2 text-sm text-gray-600 max-w-md mx-auto">
                Select a model, upload an evaluation dataset with protected and sensitive attribute columns, and run the analysis to see fairness metrics.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

interface MetricCardProps {
  title: string;
  value: number;
  threshold: number;
  tooltip: string;
  format: (v: number) => string;
  inverse?: boolean; // If true, lower values are better (default false means higher is better)
}

function MetricCard({ title, value, threshold, tooltip, format, inverse = false }: MetricCardProps) {
  const isGood = inverse ? value <= threshold : value >= threshold;
  const colorClass = isGood ? 'text-green-600' : 'text-red-600';
  const bgClass = isGood ? 'bg-green-50' : 'bg-red-50';
  const borderClass = isGood ? 'border-green-200' : 'border-red-200';

  return (
    <div className={`rounded-lg border ${borderClass} ${bgClass} p-4`}>
      <h4 className="text-sm font-medium text-gray-600 mb-1">{title}</h4>
      <p className={`text-2xl font-bold ${colorClass}`}>{format(value)}</p>
      <p className="text-xs text-gray-500 mt-1">{tooltip}</p>
    </div>
  );
}
