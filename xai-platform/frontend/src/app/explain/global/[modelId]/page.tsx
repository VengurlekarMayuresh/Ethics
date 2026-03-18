'use client';

import { useQuery, useMutation } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { useParams, useRouter } from 'next/navigation';
import { useState } from 'react';
import {
  ArrowLeft,
  Upload,
  Loader2,
  BarChart3,
  AlertCircle,
  CheckCircle,
  FileText,
  ChevronDown,
  ScatterChart,
} from 'lucide-react';
import Link from 'next/link';
import FeatureImportanceBar from '@/components/charts/FeatureImportanceBar';
import SHAPBeeswarm from '@/components/charts/SHAPBeeswarm';
import SHAPDependence from '@/components/charts/SHAPDependence';
import { format } from 'date-fns';

interface GlobalExplanation {
  _id: string;
  method: 'shap' | 'lime';
  explanation_type: 'global';
  shap_values?: number[][];
  expected_value?: number | number[];
  feature_names: string[];
  global_importance?: Array<{ feature: string; importance: number }>;
  lime_global_importance?: Array<{ feature: string; importance: number }>;
  created_at: string;
}

interface Model {
  _id: string;
  name: string;
  description: string;
  framework: string;
  task_type: string;
  feature_schema: Array<{ name: string; type: string; options?: string[] }>;
}

export default function GlobalExplanationPage() {
  const params = useParams();
  const router = useRouter();
  const modelId = params.modelId as string;

  const [backgroundFile, setBackgroundFile] = useState<File | null>(null);
  const [requestMethod, setRequestMethod] = useState<'shap' | 'lime'>('shap');
  const [selectedFeature, setSelectedFeature] = useState<string>('');
  const [dependenceData, setDependenceData] = useState<{ x_values: number[]; shap_values: number[] } | null>(null);
  const [dependenceFeatureFile, setDependenceFeatureFile] = useState<File | null>(null);
  const [isLoadingDependence, setIsLoadingDependence] = useState(false);

  // Fetch model
  const { data: model, isLoading: modelLoading } = useQuery<Model>({
    queryKey: ['model', modelId],
    queryFn: async () => {
      const { data } = await api.get(`/models/${modelId}`);
      return data;
    },
    enabled: !!modelId,
  });

  // Fetch latest global explanation
  const {
    data: explanation,
    isLoading: explanationLoading,
    error: explanationError,
    refetch: refetchExplanation,
  } = useQuery<GlobalExplanation>({
    queryKey: ['globalExplanation', modelId, requestMethod],
    queryFn: async () => {
      const endpoint = requestMethod === 'shap'
        ? `/explain/global/${modelId}/latest`
        : `/explain/lime/global/${modelId}/latest`;
      const { data } = await api.get(endpoint);
      return data;
    },
    enabled: !!modelId && !!requestMethod,
    retry: false,
  });

  // Request global explanation mutation
  const requestGlobal = useMutation({
    mutationFn: async () => {
      if (!backgroundFile) {
        throw new Error('Background data file is required for global explanation');
      }
      const formData = new FormData();
      formData.append('background_data', backgroundFile);
      if (requestMethod === 'lime') {
        // LIME also uses background data? According to backend, it uses same parameter name
        formData.append('num_features', '10');
      }
      const endpoint = requestMethod === 'shap'
        ? `/explain/global/${modelId}`
        : `/explain/lime/global/${modelId}`;
      const { data } = await api.post(endpoint, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return data;
    },
    onSuccess: () => {
      // Refetch after a delay
      setTimeout(() => {
        refetchExplanation();
      }, 3000);
    },
  });

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setBackgroundFile(e.target.files[0]);
    }
  };

  const handleRequest = () => {
    requestGlobal.mutate();
  };

  const handleFeatureFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setDependenceFeatureFile(e.target.files[0]);
    }
  };

  const loadDependencePlot = async () => {
    if (!selectedFeature || !dependenceFeatureFile) return;

    setIsLoadingDependence(true);
    try {
      const formData = new FormData();
      formData.append('background_data', dependenceFeatureFile);

      const { data } = await api.post(`/explain/dependence/${modelId}?feature=${encodeURIComponent(selectedFeature)}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setDependenceData({
        x_values: data.x_values,
        shap_values: data.shap_values,
      });
    } catch (error: any) {
      console.error('Failed to fetch dependence data:', error);
      alert('Failed to load dependence plot: ' + (error.response?.data?.detail || error.message));
    } finally {
      setIsLoadingDependence(false);
    }
  };

  const isLoading = modelLoading || explanationLoading || requestGlobal.isPending;

  if (modelLoading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-indigo-600" />
      </div>
    );
  }

  if (!model) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-8 text-center">
        <AlertCircle className="mx-auto h-12 w-12 text-red-400" />
        <h2 className="mt-4 text-lg font-semibold text-red-900">Model not found</h2>
        <p className="mt-2 text-red-700">The model you're looking for doesn't exist.</p>
        <Link href="/models" className="mt-4 inline-flex items-center text-red-600 hover:text-red-800">
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to models
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link
            href={`/models/${modelId}`}
            className="rounded-lg border border-gray-200 bg-white p-2 hover:bg-gray-50 transition"
          >
            <ArrowLeft className="h-5 w-5 text-gray-600" />
          </Link>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Global Explanation</h1>
            <p className="mt-1 text-sm text-gray-500">
              Model: {model.name}
            </p>
          </div>
        </div>

        {/* Method selector */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setRequestMethod('shap')}
            className={`inline-flex items-center rounded-lg px-4 py-2 text-sm font-medium transition ${
              requestMethod === 'shap'
                ? 'bg-indigo-600 text-white'
                : 'bg-white border border-gray-300 text-gray-700 hover:bg-gray-50'
            }`}
          >
            <BarChart3 className="mr-2 h-4 w-4" />
            SHAP
          </button>
          <button
            onClick={() => setRequestMethod('lime')}
            className={`inline-flex items-center rounded-lg px-4 py-2 text-sm font-medium transition ${
              requestMethod === 'lime'
                ? 'bg-purple-600 text-white'
                : 'bg-white border border-gray-300 text-gray-700 hover:bg-gray-50'
            }`}
          >
            <FileText className="mr-2 h-4 w-4" />
            LIME
          </button>
        </div>
      </div>

      {/* Upload section if no explanation */}
      {!explanation && (
        <div className="rounded-lg border-2 border-dashed border-gray-300 bg-gray-50 p-8">
          <div className="text-center">
            <Upload className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-4 text-lg font-semibold text-gray-900">
              Request Global {requestMethod.toUpperCase()} Explanation
            </h3>
            <p className="mt-2 text-sm text-gray-600 max-w-lg mx-auto">
              Upload a background dataset (CSV) to compute feature importance across the model.
              This dataset should be representative of the data your model was trained on or typical inference data.
            </p>
            <div className="mt-6 flex justify-center">
              <label className="cursor-pointer">
                <span className="inline-flex items-center rounded-lg bg-white px-4 py-2 text-sm font-medium text-gray-700 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50">
                  <Upload className="mr-2 h-4 w-4" />
                  {backgroundFile ? backgroundFile.name : 'Choose CSV file'}
                </span>
                <input
                  type="file"
                  accept=".csv"
                  className="hidden"
                  onChange={handleFileChange}
                  disabled={requestGlobal.isPending}
                />
              </label>
            </div>
            {backgroundFile && (
              <div className="mt-4">
                <button
                  onClick={handleRequest}
                  disabled={requestGlobal.isPending}
                  className="inline-flex items-center rounded-lg bg-indigo-600 px-6 py-2.5 text-sm font-medium text-white hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {requestGlobal.isPending ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Computing...
                    </>
                  ) : (
                    'Compute Explanation'
                  )}
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Error state */}
      {explanationError && !requestGlobal.isPending && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 flex items-start">
          <AlertCircle className="h-5 w-5 text-red-400 mr-3 mt-0.5 flex-shrink-0" />
          <div>
            <h4 className="text-sm font-medium text-red-800">Failed to load explanation</h4>
            <p className="mt-1 text-sm text-red-700">
              {explanationError.response?.data?.detail || 'An error occurred while fetching the explanation.'}
            </p>
            {requestMethod === 'shap' ? (
              <button
                onClick={() => refetchExplanation()}
                className="mt-2 text-sm font-medium text-red-600 hover:text-red-500"
              >
                Retry
              </button>
            ) : (
              <p className="mt-2 text-sm text-red-600">
                Please upload background data again and request a new computation.
              </p>
            )}
          </div>
        </div>
      )}

      {/* Explanation display */}
      {explanation && (
        <div className="space-y-6">
          {/* Info banner */}
          <div className="rounded-lg border border-green-200 bg-green-50 p-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <CheckCircle className="h-5 w-5 text-green-600" />
              <div>
                <p className="text-sm font-medium text-green-800">
                  {requestMethod.toUpperCase()} explanation ready
                </p>
                <p className="text-xs text-green-700">
                  Generated: {format(new Date(explanation.created_at), 'MMM d, yyyy HH:mm')}
                </p>
              </div>
            </div>
            <div className="text-sm font-mono text-green-700">
              ID: {explanation._id.slice(0, 8)}...
            </div>
          </div>

          {/* Feature Importance Bar Chart */}
          {explanation.global_importance && explanation.global_importance.length > 0 && (
            <FeatureImportanceBar
              data={explanation.global_importance}
              title={`Global Feature Importance (${requestMethod.toUpperCase()})`}
              height={400}
              color={requestMethod === 'shap' ? '#3b82f6' : '#8b5cf6'}
            />
          )}

          {explanation.lime_global_importance && explanation.lime_global_importance.length > 0 && (
            <FeatureImportanceBar
              data={explanation.lime_global_importance}
              title={`Global Feature Importance (LIME)`}
              height={400}
              color="#8b5cf6"
            />
          )}

          {/* Beeswarm plot for SHAP */}
          {requestMethod === 'shap' && explanation.shap_values && (
            <SHAPBeeswarm
              shapValues={explanation.shap_values}
              featureNames={explanation.feature_names}
              title="SHAP Beeswarm Plot (Global Distribution)"
              height={500}
            />
          )}

          {/* SHAP Dependence Plot */}
          {requestMethod === 'shap' && explanation.feature_names && (
            <div className="rounded-lg border border-gray-200 bg-white p-6">
              <div className="mb-4">
                <h3 className="text-lg font-semibold text-gray-900 mb餐2">SHAP Dependence Plots</h3>
                <p className="text-sm text-gray-600 mb-4">
                  Visualize how a specific feature influences model predictions. Upload a background dataset and select a feature to see the relationship.
                </p>

                <div className="flex flex-wrap items-end gap-4">
                  <div className="min-w-64">
                    <label className="block text-sm font-medium text-gray-700 mb-2">Select Feature</label>
                    <div className="relative">
                      <select
                        value={selectedFeature}
                        onChange={(e) => setSelectedFeature(e.target.value)}
                        className="w-full px-3 py-2 pr-10 border border-gray-300 rounded-lg appearance-none bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                        disabled={isLoadingDependence}
                      >
                        <option value="">-- Choose a feature --</option>
                        {explanation.feature_names.map((feature: string) => (
                          <option key={feature} value={feature}>{feature}</option>
                        ))}
                      </select>
                      <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-500 pointer-events-none" />
                    </div>
                  </div>

                  <div className="flex-1 min-w-80">
                    <label className="block text-sm font-medium text-gray-700 mb-2">Background Dataset (CSV)</label>
                    <label className="cursor-pointer inline-flex items-center px-4 py-2 bg-white border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50">
                      <Upload className="mr-2 h-4 w-4" />
                      {dependenceFeatureFile ? dependenceFeatureFile.name : 'Upload CSV'}
                      <input
                        type="file"
                        accept=".csv"
                        className="hidden"
                        onChange={handleFeatureFileChange}
                        disabled={isLoadingDependence}
                      />
                    </label>
                  </div>

                  <button
                    onClick={loadDependencePlot}
                    disabled={!selectedFeature || !dependenceFeatureFile || isLoadingDependence}
                    className="inline-flex items-center px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isLoadingDependence ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Loading...
                      </>
                    ) : (
                      <>
                        <ScatterChart className="mr-2 h-4 w-4" />
                        Generate Plot
                      </>
                    )}
                  </button>
                </div>
              </div>

              {dependenceData && (
                <div className="mt-6">
                  <SHAPDependence
                    featureName={selectedFeature}
                    xValues={dependenceData.x_values}
                    shapValues={dependenceData.shap_values}
                    height={400}
                  />
                  <p className="text-xs text-gray-500 mt-2 text-center">
                    Each point represents a sample from the background dataset.
                    Red points indicate positive SHAP values (increasing prediction),
                    blue points indicate negative SHAP values (decreasing prediction).
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Background data info */}
          <div className="rounded-lg border border-gray-200 bg-white p-4">
            <h3 className="text-sm font-semibold text-gray-900 mb-2">About this analysis</h3>
            <p className="text-sm text-gray-600">
              This {requestMethod} global explanation was computed on a background dataset of samples.
              The importance scores represent the average absolute impact of each feature on the model's predictions across the dataset.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
