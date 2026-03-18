'use client';

import { useQuery, useMutation } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { useState, useMemo } from 'react';
import {
  Upload,
  Loader2,
  AlertCircle,
  BarChart3,
  CheckSquare,
  Square,
  FileText,
  Table,
} from 'lucide-react';
import Link from 'next/link';

interface Model {
  _id: string;
  name: string;
  framework: string;
  task_type: string;
}

interface ComparisonResult {
  models: Array<{
    model_id: string;
    model_name: string;
    task_type: string;
    framework: string;
    feature_importance: Array<{ feature: string; importance: number }>;
    predictions: number[];
    probabilities?: number[][];
  }>;
  global_importance: Array<{ feature: string; importance: number }>;
  prediction_comparison: Array<{ row_index: number; predictions: number[] }>;
  dataset_size: number;
}

export default function ComparePage() {
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>([]);
  const [datasetFile, setDatasetFile] = useState<File | null>(null);

  // Fetch user's models
  const { data: models, isLoading: modelsLoading } = useQuery<Model[]>({
    queryKey: ['models'],
    queryFn: async () => {
      const { data } = await api.get('/models/');
      return data;
    },
  });

  // Comparison mutation
  const compareMutation = useMutation<ComparisonResult, Error>({
    mutationFn: async () => {
      if (selectedModelIds.length < 2) {
        throw new Error('Please select at least 2 models to compare');
      }
      if (!datasetFile) {
        throw new Error('Please upload a dataset');
      }

      const formData = new FormData();
      formData.append('file', datasetFile);
      // backend expects model_ids as query params? Actually POST /compare/ with model_ids as query param? Let's check: in API it's `model_ids: List[str] = Query(...)`, but file is also in form-data. That's tricky because mixing query params and file upload in multipart might be okay? Usually query params are separate. I'll send as query parameters.
      // Since it's FormData, I can't include list in form. Need to send as query string. Let's adjust: call with params: { model_ids: selectedModelIds } and body: formData.
      const { data } = await api.post('/compare/', formData, {
        params: { model_ids: selectedModelIds },
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return data;
    },
  });

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setDatasetFile(e.target.files[0]);
    }
  };

  const toggleModel = (modelId: string) => {
    setSelectedModelIds(prev =>
      prev.includes(modelId)
        ? prev.filter(id => id !== modelId)
        : [...prev, modelId]
    );
  };

  const handleCompare = () => {
    compareMutation.mutate();
  };

  const result = compareMutation.data;

  // Prepare data for feature importance comparison chart
  const allFeatures = useMemo(() => {
    if (!result) return new Set<string>();
    const features = new Set<string>();
    result.models.forEach(m => {
      m.feature_importance.forEach(fi => features.add(fi.feature));
    });
    return features;
  }, [result]);

  const featureImportanceComparison = useMemo(() => {
    if (!result) return [];
    return Array.from(allFeatures).map(feature => {
      const entry: any = { feature };
      result.models.forEach(model => {
        const fi = model.feature_importance.find(f => f.feature === feature);
        entry[model.model_id] = fi ? fi.importance : 0;
      });
      return entry;
    }).sort((a, b) => {
      // Sort by total importance sum across models descending
      const sumA = selectedModelIds.reduce((acc, id) => acc + (a[id] || 0), 0);
      const sumB = selectedModelIds.reduce((acc, id) => acc + (b[id] || 0), 0);
      return sumB - sumA;
    });
  }, [result, allFeatures, selectedModelIds]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Model Comparison</h1>
        <p className="mt-1 text-sm text-gray-500">
          Compare feature importance and predictions across multiple models using the same evaluation dataset.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Configuration Panel */}
        <div className="lg:col-span-1 space-y-6">
          <div className="rounded-lg border border-gray-200 bg-white p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Select Models</h2>

            {modelsLoading ? (
              <div className="flex justify-center py-4">
                <Loader2 className="h-6 w-6 animate-spin text-indigo-600" />
              </div>
            ) : models && models.length > 0 ? (
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {models.map(model => (
                  <div
                    key={model._id}
                    className={`flex items-center justify-between rounded-lg border p-3 cursor-pointer transition ${
                      selectedModelIds.includes(model._id)
                        ? 'border-indigo-300 bg-indigo-50'
                        : 'border-gray-200 hover:bg-gray-50'
                    }`}
                    onClick={() => toggleModel(model._id)}
                  >
                    <div className="flex items-center gap-3">
                      {selectedModelIds.includes(model._id) ? (
                        <CheckSquare className="h-5 w-5 text-indigo-600" />
                      ) : (
                        <Square className="h-5 w-5 text-gray-400" />
                      )}
                      <div>
                        <p className="font-medium text-gray-900 text-sm">{model.name}</p>
                        <p className="text-xs text-gray-500 capitalize">{model.framework} • {model.task_type}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-gray-500 text-center py-4">No models found. Upload some models first.</p>
            )}

            <div className="mt-4 text-sm text-gray-600">
              Selected: {selectedModelIds.length} / {models?.length || 0}
            </div>
          </div>

          <div className="rounded-lg border border-gray-200 bg-white p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Dataset</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Evaluation Dataset (CSV)
                </label>
                <div className="mt-1 flex justify-center rounded-md border-2 border-dashed border-gray-300 px-6 pt-5 pb-6 hover:bg-gray-50">
                  <div className="space-y-1 text-center">
                    <Upload className="mx-auto h-8 w-8 text-gray-400" />
                    <div className="flex text-sm text-gray-600 justify-center">
                      <label htmlFor="compare-dataset-upload" className="relative cursor-pointer rounded-md font-medium text-indigo-600 hover:text-indigo-500">
                        <span>Upload a file</span>
                        <input
                          id="compare-dataset-upload"
                          type="file"
                          accept=".csv"
                          className="sr-only"
                          onChange={handleFileChange}
                          disabled={compareMutation.isPending}
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

              {compareMutation.isError && (
                <div className="rounded-md bg-red-50 p-3 border border-red-200">
                  <p className="text-sm text-red-800">
                    {(compareMutation.error as Error)?.message || 'Comparison failed'}
                  </p>
                </div>
              )}

              <button
                onClick={handleCompare}
                disabled={compareMutation.isPending || selectedModelIds.length < 2 || !datasetFile}
                className="w-full flex justify-center items-center rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 transition"
              >
                {compareMutation.isPending ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Comparing...
                  </>
                ) : (
                  <>
                    <BarChart3 className="mr-2 h-4 w-4" />
                    Compare Models
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Results Panel */}
        <div className="lg:col-span-2">
          {compareMutation.isSuccess && result && (
            <div className="space-y-6">
              {/* Summary */}
              <div className="rounded-lg border border-green-200 bg-green-50 p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <CheckSquare className="h-5 w-5 text-green-600" />
                    <div>
                      <p className="text-sm font-medium text-green-800">Comparison complete</p>
                      <p className="text-xs text-green-700">
                        {result.dataset_size} samples • {result.models.length} models
                      </p>
                    </div>
                  </div>
                  <div className="text-sm font-mono text-green-700">
                    Global features: {Object.keys(allFeatures).length}
                  </div>
                </div>
              </div>

              {/* Feature Importance Comparison */}
              <div className="rounded-lg border border-gray-200 bg-white p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Feature Importance Comparison</h3>
                {featureImportanceComparison.length > 0 ? (
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                            Feature
                          </th>
                          {result.models.map(model => (
                            <th key={model.model_id} className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                              {model.model_name}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-200 bg-white">
                        {featureImportanceComparison.slice(0, 10).map((row: any) => (
                          <tr key={row.feature}>
                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                              {row.feature}
                            </td>
                            {result.models.map(model => (
                              <td key={model.model_id} className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 font-mono">
                                {(row[model.model_id] * 100).toFixed(1)}%
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="text-sm text-gray-500 text-center py-8">
                    No feature importance data available for these models.
                  </p>
                )}
              </div>

              {/* Prediction Comparison */}
              <div className="rounded-lg border border-gray-200 bg-white p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Prediction Comparison</h3>
                {result.prediction_comparison && result.prediction_comparison.length > 0 ? (
                  <div>
                    <div className="overflow-x-auto max-h-96 overflow-y-auto">
                      <table className="min-w-full divide-y divide-gray-200">
                        <thead className="bg-gray-50 sticky top-0">
                          <tr>
                            <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                              Row
                            </th>
                            {result.models.map(model => (
                              <th key={model.model_id} className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                                {model.model_name}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200 bg-white">
                          {result.prediction_comparison.slice(0, 100).map((row: any) => (
                            <tr key={row.row_index}>
                              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                {row.row_index + 1}
                              </td>
                              {result.models.map(model => (
                                <td key={model.model_id} className="px-6 py-4 whitespace-nowrap text-sm font-mono text-gray-900">
                                  {row.predictions[result.models.indexOf(model)] !== undefined
                                    ? Number(row.predictions[result.models.indexOf(model)]).toFixed(4)
                                    : '-'}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    {result.prediction_comparison.length > 100 && (
                      <p className="text-center text-sm text-gray-500 mt-2">
                        Showing first 100 rows of {result.prediction_comparison.length}
                      </p>
                    )}
                  </div>
                ) : (
                  <p className="text-sm text-gray-500 text-center py-8">No predictions to compare.</p>
                )}
              </div>

              {/* Individual Model Details */}
              <div className="grid gap-6 md:grid-cols-2">
                {result.models.map(model => (
                  <div key={model.model_id} className="rounded-lg border border-gray-200 bg-white p-6">
                    <h4 className="text-lg font-semibold text-gray-900 mb-3">{model.model_name}</h4>
                    <div className="space-y-3 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-500">Framework</span>
                        <span className="font-medium text-gray-900">{model.framework}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Task Type</span>
                        <span className="font-medium text-gray-900">{model.task_type}</span>
                      </div>
                      <div>
                        <span className="text-gray-500 block mb-1">Top Features</span>
                        {model.feature_importance && model.feature_importance.length > 0 ? (
                          <ul className="space-y-1">
                            {model.feature_importance.slice(0, 5).map((fi, idx) => (
                              <li key={idx} className="flex justify-between text-xs">
                                <span className="text-gray-700">{fi.feature}</span>
                                <span className="font-mono text-gray-900">{(fi.importance * 100).toFixed(1)}%</span>
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <p className="text-xs text-gray-500">No feature importance available</p>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {!compareMutation.isSuccess && !compareMutation.isPending && (
            <div className="rounded-lg border border-gray-200 bg-white p-12 text-center">
              <BarChart3 className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-4 text-lg font-semibold text-gray-900">No Comparison Yet</h3>
              <p className="mt-2 text-sm text-gray-600 max-w-md mx-auto">
                Select at least 2 models from your library, upload an evaluation dataset, and compare their predictions and feature importance.
              </p>
            </div>
          )}

          {compareMutation.isPending && (
            <div className="rounded-lg border border-gray-200 bg-white p-12 text-center">
              <Loader2 className="mx-auto h-12 w-12 animate-spin text-indigo-600" />
              <h3 className="mt-4 text-lg font-semibold text-gray-900">Comparing Models</h3>
              <p className="mt-2 text-sm text-gray-600">
                Running predictions on all selected models and aggregating feature importance...
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
