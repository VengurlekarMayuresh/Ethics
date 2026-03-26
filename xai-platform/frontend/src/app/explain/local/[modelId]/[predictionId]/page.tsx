'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { useParams, useRouter } from 'next/navigation';
import { useState, useRef, useEffect } from 'react';
import {
  ArrowLeft,
  Loader2,
  AlertCircle,
  CheckCircle,
  BarChart3,
  Brain,
  FileText,
  Upload,
} from 'lucide-react';
import Link from 'next/link';
import SHAPForcePlot from '@/components/charts/SHAPForcePlot';
import LIMEPlot from '@/components/charts/LIMEPlot';
import { format } from 'date-fns';
import ReactMarkdown from 'react-markdown';

interface Prediction {
  _id: string;
  model_id: string;
  input_data: Record<string, any>;
  prediction: any;
  probability?: any;
  created_at: string;
}

interface GlobalExplanation {
  _id: string;
  method: 'shap';
  explanation_type: 'global';
  shap_values?: number[][];
  expected_value?: number | number[];
  feature_names: string[];
  global_importance?: Array<{ feature: string; importance: number }>;
  created_at?: string;
}

interface LocalExplanation {
  _id?: string;
  method: 'shap' | 'lime';
  explanation_type: 'local';
  shap_values?: number[][];
  expected_value?: number;
  feature_names?: string[];
  lime_weights?: any[];
  lime_intercept?: number;
  lime_local_pred?: number;
  created_at?: string;
  status?: 'pending' | 'complete' | 'failed';
  error?: string;
}

type TabId = 'shap-force' | 'lime' | 'explanation';

// ─── ExplanationTab ──────────────────────────────────────────────────────────
function ExplanationTab({
  localShap, localLime, predictionValue, predictionLabel, explanationReady,
}: {
  localShap: any; localLime: any; predictionValue: number; predictionLabel: string; explanationReady: boolean;
}) {
  const [nlText, setNlText] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [source, setSource] = useState<'openrouter' | 'template' | null>(null);
  const [error, setError] = useState<string | null>(null);

  const generate = async () => {
    setLoading(true);
    setError(null);
    try {
      // Build flat SHAP values
      let shapValues: number[] | undefined;
      let shapFeatureNames: string[] | undefined;
      if (localShap?.shap_values && localShap.feature_names) {
        const raw: number[][] | number[] = localShap.shap_values;
        shapValues = Array.isArray(raw[0]) ? (raw[0] as number[]) : (raw as number[]);
        shapFeatureNames = localShap.feature_names;
      }
      const payload = {
        prediction_label: predictionLabel,
        prediction_value: predictionValue,
        shap_feature_names: shapFeatureNames,
        shap_values: shapValues,
        shap_base_value: localShap?.expected_value ?? undefined,
        lime_weights: localLime?.lime_weights ?? undefined,
        lime_local_pred: localLime?.lime_local_pred ?? undefined,
      };
      const { data } = await api.post('/explain/nl-generate', payload);
      setNlText(data.explanation);
      setSource(data.source);
    } catch (e: any) {
      setError(e?.response?.data?.detail || 'Failed to generate explanation.');
    } finally {
      setLoading(false);
    }
  };

  if (!explanationReady) {
    return (
      <div className="rounded-lg border-2 border-dashed border-gray-300 bg-gray-50 p-12 text-center">
        <FileText className="mx-auto h-16 w-16 text-gray-400" />
        <h3 className="mt-4 text-xl font-semibold text-gray-900">AI Explanation</h3>
        <p className="mt-2 text-gray-600 max-w-lg mx-auto">
          Generate SHAP and/or LIME explanations first, then come back here for a plain-English interpretation.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header card */}
      <div className="rounded-xl border border-indigo-200 bg-gradient-to-br from-indigo-50 to-purple-50 p-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-xl font-bold text-indigo-900">AI-Powered Explanation</h2>
            <p className="mt-1 text-sm text-indigo-700">
              A plain-English interpretation of SHAP and LIME results for this prediction.
            </p>
          </div>
          <button
            onClick={generate}
            disabled={loading}
            className="flex-shrink-0 inline-flex items-center gap-2 px-5 py-2.5 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 disabled:opacity-60 transition"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <FileText className="h-4 w-4" />}
            {loading ? 'Generating...' : nlText ? 'Regenerate' : 'Generate Explanation'}
          </button>
        </div>

        {/* Data availability badges */}
        <div className="flex gap-3 mt-4">
          {localShap?.shap_values ? (
            <span className="inline-flex items-center gap-1 text-xs font-medium px-2.5 py-1 rounded-full bg-green-100 text-green-800">
              <CheckCircle className="h-3 w-3" /> SHAP ready
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 text-xs font-medium px-2.5 py-1 rounded-full bg-gray-100 text-gray-500">
              SHAP not available
            </span>
          )}
          {localLime?.lime_weights?.length ? (
            <span className="inline-flex items-center gap-1 text-xs font-medium px-2.5 py-1 rounded-full bg-purple-100 text-purple-800">
              <CheckCircle className="h-3 w-3" /> LIME ready
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 text-xs font-medium px-2.5 py-1 rounded-full bg-gray-100 text-gray-500">
              LIME not available
            </span>
          )}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700 flex items-center gap-2">
          <AlertCircle className="h-4 w-4 flex-shrink-0" />
          {error}
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="flex items-center justify-center py-16">
          <div className="text-center">
            <Loader2 className="mx-auto h-10 w-10 animate-spin text-indigo-500" />
            <p className="mt-3 text-gray-600 font-medium">Analyzing predictions...</p>
            <p className="text-sm text-gray-500">This usually takes a few seconds.</p>
          </div>
        </div>
      )}

      {/* Result */}
      {nlText && !loading && (
        <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-gray-900 flex items-center gap-2">
              <Brain className="h-5 w-5 text-indigo-500" />
              Explanation
            </h3>
            {source && (
              <span className={`text-xs px-2 py-1 rounded-full font-medium ${
                source === 'openrouter'
                  ? 'bg-green-100 text-green-700'
                  : 'bg-yellow-100 text-yellow-700'
              }`}>
                {source === 'openrouter' ? '✨ AI Generated' : '📋 Template'}
              </span>
            )}
          </div>
          <div className="prose prose-sm max-w-none text-gray-800 leading-relaxed">
            <ReactMarkdown>{nlText}</ReactMarkdown>
          </div>
        </div>
      )}

      {/* Prompt to generate */}
      {!nlText && !loading && !error && (
        <div className="flex items-center justify-center py-12 border-2 border-dashed border-gray-200 rounded-xl text-gray-500">
          <div className="text-center">
            <FileText className="mx-auto h-10 w-10 text-gray-300 mb-3" />
            <p className="font-medium">Click "Generate Explanation" to get started</p>
            <p className="text-sm mt-1">SHAP + LIME data will be sent to the AI for interpretation</p>
          </div>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────

export default function UnifiedExplanationPage() {
  const params = useParams();
  const router = useRouter();
  const queryClient = useQueryClient();

  const modelId = params.modelId as string;
  const predictionId = params.predictionId as string;

  const [activeTab, setActiveTab] = useState<TabId>('shap-force');
  const [autoTriggered, setAutoTriggered] = useState(false);

  // For global SHAP upload
  const [globalShapBackgroundFile, setGlobalShapBackgroundFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Fetch prediction
  const {
    data: prediction,
    isLoading: predictionLoading,
    error: predictionError,
  } = useQuery<Prediction>({
    queryKey: ['prediction', predictionId],
    queryFn: async () => {
      const { data } = await api.get(`/predict/${predictionId}`);
      return data;
    },
    enabled: !!predictionId,
  });

  const {
    data: globalShap,
    isLoading: globalShapLoading,
    error: globalShapError,
    refetch: refetchGlobalShap,
  } = useQuery<GlobalExplanation>({
    queryKey: ['globalExplanation', modelId],
    queryFn: async () => {
      try {
        const { data } = await api.get(`/explain/global/${modelId}/latest`);
        return data;
      } catch (error: any) {
        if (error.response?.status === 404) return null;
        throw error;
      }
    },
    enabled: false,
    retry: false,
  });
  // Fetch local SHAP explanation — always enabled, polls every 5s until data arrives
  const {
    data: localShap,
    isLoading: localShapLoading,
    error: localShapError,
  } = useQuery<LocalExplanation>({
    queryKey: ['localExplanation', predictionId, 'shap'],
    queryFn: async () => {
      try {
        const { data } = await api.get(`/explain/prediction/${predictionId}?method=shap`);
        return data;
      } catch (error: any) {
        if (error.response?.status === 404) {
          return null;
        }
        throw error;
      }
    },
    // Always enabled (not gated on activeTab) so polling continues in background
    enabled: !!predictionId,
    retry: false,
    // Poll every 5 seconds until explanation data is present
    refetchInterval: (query) => {
      const d = query.state.data as LocalExplanation | null | undefined;
      if (d && d.shap_values && d.shap_values.length > 0) return false; // stop polling
      return 5000;
    },
  });

  // Fetch local LIME explanation — always enabled, polls every 5s until data arrives
  const {
    data: localLime,
    isLoading: localLimeLoading,
    error: localLimeError,
  } = useQuery<LocalExplanation>({
    queryKey: ['localExplanation', predictionId, 'lime'],
    queryFn: async () => {
      try {
        const { data } = await api.get(`/explain/prediction/${predictionId}?method=lime`);
        return data;
      } catch (error: any) {
        if (error.response?.status === 404) {
          return null;
        }
        throw error;
      }
    },
    // Always enabled (not gated on activeTab) so polling continues in background
    enabled: !!predictionId,
    retry: false,
    // Poll every 5 seconds until lime_weights data is present
    refetchInterval: (query) => {
      const d = query.state.data as LocalExplanation | null | undefined;
      if (d && d.lime_weights && (d.lime_weights as any[]).length > 0) return false; // stop polling
      return 5000;
    },
  });

  // Mutations to request explanations
  // (No onSuccess refetch needed — refetchInterval handles continuous polling)
  const requestShapMutation = useMutation({
    mutationFn: async () => {
      const { data } = await api.post(`/explain/local/${modelId}`, null, {
        params: { prediction_id: predictionId },
      });
      return data;
    },
  });

  const requestLimeMutation = useMutation({
    mutationFn: async () => {
      const { data } = await api.post(`/explain/lime/${modelId}`, null, {
        params: { prediction_id: predictionId },
      });
      return data;
    },
  });

  const requestGlobalShapMutation = useMutation({
    mutationFn: async () => {
      if (!globalShapBackgroundFile) {
        throw new Error('Background data file is required');
      }
      const formData = new FormData();
      formData.append('background_data', globalShapBackgroundFile);
      const { data } = await api.post(`/explain/global/${modelId}`, formData);
      return data;
    },
    onSuccess: () => {
      // Refetch global SHAP after a delay
      setTimeout(() => {
        refetchGlobalShap();
      }, 3000);
    },
  });

  const handleGenerateShap = () => {
    requestShapMutation.mutate();
  };

  const handleGenerateLime = () => {
    requestLimeMutation.mutate();
  };

  const handleGenerateGlobalShap = () => {
    requestGlobalShapMutation.mutate();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setGlobalShapBackgroundFile(e.target.files[0]);
    }
  };

  const tabs = [
    { id: 'shap-force' as TabId, label: 'SHAP Force Plot', icon: BarChart3 },
    { id: 'lime' as TabId, label: 'LIME', icon: Brain },
    { id: 'explanation' as TabId, label: 'AI Explanation', icon: FileText },
  ];

  const isTabLoading = (tab: TabId) => {
    switch (tab) {
      case 'shap-force':
        return localShapLoading || localShap?.status === 'pending';
      case 'lime':
        return localLimeLoading || localLime?.status === 'pending';
      default:
        return false;
    }
  };

  const getTabData = (tab: TabId) => {
    switch (tab) {
      case 'shap-force': return localShap;
      case 'lime': return localLime;
      default: return null;
    }
  };

  const getTabError = (tab: TabId) => {
    switch (tab) {
      case 'shap-force': return localShapError;
      case 'lime': return localLimeError;
      default: return null;
    }
  };

  const isGeneratingGlobal = requestGlobalShapMutation.isPending;
  const globalShapHasData = globalShap && (globalShap.global_importance?.length > 0 || globalShap.shap_values?.length > 0);

  // Auto-trigger SHAP and LIME once on page load if no completed explanation exists.
  // refetchInterval handles polling; this just ensures computation is kicked off.
  useEffect(() => {
    if (autoTriggered || !predictionId) return;
    // Only fire once — don't wait for localShap/localLime to load first
    // (they may not be loaded yet on first render)
    setAutoTriggered(true);

    const shapComplete = localShap && localShap.shap_values && (localShap.shap_values as number[][]).length > 0;
    if (!shapComplete && !requestShapMutation.isPending) {
      requestShapMutation.mutate();
    }

    const limeComplete = localLime && localLime.lime_weights && (localLime.lime_weights as any[]).length > 0;
    if (!limeComplete && !requestLimeMutation.isPending) {
      requestLimeMutation.mutate();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [predictionId]);

  if (predictionLoading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-indigo-600" />
      </div>
    );
  }

  if (predictionError || !prediction) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-8 text-center">
        <AlertCircle className="mx-auto h-12 w-12 text-red-400" />
        <h2 className="mt-4 text-lg font-semibold text-red-900">Prediction not found</h2>
        <p className="mt-2 text-red-700">The prediction you're looking for doesn't exist or you don't have access.</p>
        <Link href="/predict/history" className="mt-4 inline-flex items-center text-red-600 hover:text-red-800">
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to prediction history
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
            href={`/predict/${modelId}`}
            className="rounded-lg border border-gray-200 bg-white p-2 hover:bg-gray-50 transition"
          >
            <ArrowLeft className="h-5 w-5 text-gray-600" />
          </Link>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Explanation Dashboard</h1>
            <p className="mt-1 text-sm text-gray-500">
              Model: {modelId.slice(0, 8)}... | Prediction: {predictionId.slice(0, 8)}...
            </p>
          </div>
        </div>
      </div>

      {/* Prediction summary */}
      <div className="rounded-lg border border-gray-200 bg-white p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Prediction Details</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-sm font-medium text-gray-500 mb-2">Input Features</h3>
            <div className="space-y-2">
              {Object.entries(prediction.input_data).map(([key, value]) => (
                <div key={key} className="flex justify-between text-sm">
                  <span className="text-gray-600">{key}:</span>
                  <span className="font-mono text-gray-900">{String(value)}</span>
                </div>
              ))}
            </div>
          </div>
          <div>
            <h3 className="text-sm font-medium text-gray-500 mb-2">Result</h3>
            <div className="rounded-lg bg-indigo-50 p-4">
              <p className="text-2xl font-bold text-indigo-700">
                {typeof prediction.prediction === 'number'
                  ? prediction.prediction.toFixed(4)
                  : prediction.prediction}
              </p>
              {prediction.probability && (
                <p className="text-sm text-indigo-600 mt-1">
                  Probability: {(prediction.probability * 100).toFixed(1)}%
                </p>
              )}
            </div>
            <p className="text-xs text-gray-500 mt-2">
              Generated: {prediction.created_at ? format(new Date(prediction.created_at), 'MMM d, yyyy HH:mm') : 'N/A'}
            </p>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="flex space-x-0.5 overflow-x-auto" aria-label="Tabs">
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            const isLoading = isTabLoading(tab.id);
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`group relative min-w-fit py-4 px-6 border-b-2 font-medium text-sm transition-colors ${
                  isActive
                    ? 'border-indigo-600 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <div className="flex items-center gap-2">
                  <Icon className="h-4 w-4" />
                  <span>{tab.label}</span>
                  {isLoading && <Loader2 className="h-4 w-4 animate-spin" />}
                  {!isLoading && isActive && <CheckCircle className="h-4 w-4" />}
                </div>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Tab content */}
      <div className="min-h-[400px]">
        {tabs.map((tab) => {
          const isActive = activeTab === tab.id;
          const isLoading = isTabLoading(tab.id);
          const data = getTabData(tab.id);
          const error = getTabError(tab.id);

          if (!isActive) return null;

          // Loading state
          if (isLoading) {
            return (
              <div key={tab.id} className="flex h-64 items-center justify-center border-2 border-dashed border-gray-300 rounded-lg bg-gray-50">
                <div className="text-center">
                  <Loader2 className="mx-auto h-10 w-10 animate-spin text-indigo-600" />
                  <h3 className="mt-4 text-lg font-semibold text-gray-900">Loading...</h3>
                  <p className="mt-2 text-gray-600">Please wait while we load the explanation.</p>
                </div>
              </div>
            );
          }

          // Error state (non-404 errors)
          if (error && error.response?.status !== 404) {
            return (
              <div key={tab.id} className="rounded-lg border border-red-200 bg-red-50 p-8 text-center">
                <AlertCircle className="mx-auto h-12 w-12 text-red-400" />
                <h3 className="mt-4 text-lg font-semibold text-red-800">Failed to load explanation</h3>
                <p className="mt-2 text-red-700">
                  {(error as any)?.response?.data?.detail || 'An error occurred.'}
                </p>
                <button
                  onClick={() => window.location.reload()}
                  className="mt-4 inline-flex items-center px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
                >
                  Retry
                </button>
              </div>
            );
          }

          // Tab specific content
          switch (tab.id) {
            case 'shap-bar':
              if (!globalShapHasData) {
                return (
                  <div key={tab.id} className="rounded-lg border-2 border-dashed border-gray-300 bg-gray-50 p-12 text-center">
                    <BarChart3 className="mx-auto h-16 w-16 text-gray-400" />
                    <h3 className="mt-4 text-xl font-semibold text-gray-900">SHAP Summary Bar Plot</h3>
                    <p className="mt-2 text-gray-600 max-w-lg mx-auto">
                      This chart shows global feature importance based on SHAP values. Upload a background dataset to generate it.
                    </p>
                    <div className="mt-6">
                      <input
                        type="file"
                        accept=".csv"
                        onChange={handleFileChange}
                        className="hidden"
                        ref={fileInputRef}
                        disabled={isGeneratingGlobal}
                      />
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        disabled={isGeneratingGlobal}
                        className="inline-flex items-center px-6 py-3 bg-white border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 disabled:opacity-50"
                      >
                        <Upload className="mr-2 h-5 w-5" />
                        {globalShapBackgroundFile ? globalShapBackgroundFile.name : 'Choose CSV file'}
                      </button>
                      {globalShapBackgroundFile && (
                        <button
                          onClick={handleGenerateGlobalShap}
                          disabled={isGeneratingGlobal}
                          className="ml-4 inline-flex items-center px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50"
                        >
                          {isGeneratingGlobal ? (
                            <>
                              <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                              Computing...
                            </>
                          ) : (
                            'Compute SHAP'
                          )}
                        </button>
                      )}
                    </div>
                  </div>
                );
              }
              if (globalShap && globalShap.global_importance) {
                return (
                  <FeatureImportanceBar
                    key={tab.id}
                    data={globalShap.global_importance}
                    title="SHAP Feature Importance (Global)"
                    height={400}
                    color="#3b82f6"
                  />
                );
              }
              return null;

            case 'shap-beeswarm':
              if (!globalShapHasData) {
                return (
                  <div key={tab.id} className="rounded-lg border-2 border-dashed border-gray-300 bg-gray-50 p-12 text-center">
                    <BarChart3 className="mx-auto h-16 w-16 text-gray-400" />
                    <h3 className="mt-4 text-xl font-semibold text-gray-900">SHAP Beeswarm Plot</h3>
                    <p className="mt-2 text-gray-600 max-w-lg mx-auto">
                      This plot shows the distribution of SHAP values across all features. Upload a background dataset to generate it.
                    </p>
                    <div className="mt-6">
                      <input
                        type="file"
                        accept=".csv"
                        onChange={handleFileChange}
                        className="hidden"
                        ref={fileInputRef}
                        disabled={isGeneratingGlobal}
                      />
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        disabled={isGeneratingGlobal}
                        className="inline-flex items-center px-6 py-3 bg-white border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 disabled:opacity-50"
                      >
                        <Upload className="mr-2 h-5 w-5" />
                        {globalShapBackgroundFile ? globalShapBackgroundFile.name : 'Choose CSV file'}
                      </button>
                      {globalShapBackgroundFile && (
                        <button
                          onClick={handleGenerateGlobalShap}
                          disabled={isGeneratingGlobal}
                          className="ml-4 inline-flex items-center px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50"
                        >
                          {isGeneratingGlobal ? (
                            <>
                              <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                              Computing...
                            </>
                          ) : (
                            'Compute SHAP'
                          )}
                        </button>
                      )}
                    </div>
                  </div>
                );
              }
              if (globalShap && globalShap.shap_values && globalShap.feature_names) {
                return (
                  <SHAPBeeswarm
                    key={tab.id}
                    shapValues={globalShap.shap_values}
                    featureNames={globalShap.feature_names}
                    title="SHAP Beeswarm Plot (Global Distribution)"
                    height={500}
                  />
                );
              }
              return null;

            case 'shap-force':
              if (!localShap || !localShap.shap_values) {
                if (localShap?.status === 'pending' || localShapLoading) {
                  return (
                    <div key={tab.id} className="flex h-64 items-center justify-center border-2 border-dashed border-gray-300 rounded-lg bg-gray-50">
                      <div className="text-center">
                        <Loader2 className="mx-auto h-10 w-10 animate-spin text-yellow-500" />
                        <h3 className="mt-4 text-lg font-semibold text-yellow-800">Computing SHAP...</h3>
                        <p className="mt-2 text-yellow-700">This may take a few moments.</p>
                      </div>
                    </div>
                  );
                }
                return (
                  <div key={tab.id} className="rounded-lg border-2 border-dashed border-gray-300 bg-gray-50 p-12 text-center">
                    <BarChart3 className="mx-auto h-16 w-16 text-gray-400" />
                    <h3 className="mt-4 text-xl font-semibold text-gray-900">SHAP Force Plot</h3>
                    <p className="mt-2 text-gray-600 max-w-lg mx-auto">
                      This waterfall plot shows how each feature contributed to push the prediction from the base value to the final result.
                    </p>
                    <div className="mt-6">
                      <button
                        onClick={handleGenerateShap}
                        disabled={requestShapMutation.isPending}
                        className="inline-flex items-center px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50"
                      >
                        {requestShapMutation.isPending ? (
                          <>
                            <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                            Generating...
                          </>
                        ) : (
                          'Generate SHAP Explanation'
                        )}
                      </button>
                    </div>
                  </div>
                );
              }

              const baseValue = localShap.expected_value ?? 0;
              const predValue = typeof prediction.prediction === 'number' ? prediction.prediction : 0;
              const shapValuesRaw = Array.isArray(localShap.shap_values) ? localShap.shap_values : [];
              // Handle both [[v1,v2,...]] (nested) and [v1,v2,...] (flat) from backend
              const firstRow: number[] = Array.isArray(shapValuesRaw[0])
                ? (shapValuesRaw[0] as number[])
                : (shapValuesRaw as number[]);
              const featureNames = localShap.feature_names || [];

              const shapValuesFormatted = firstRow.map((val: number, idx: number) => ({
                feature: featureNames[idx] || `Feature ${idx}`,
                value: val,
              }));

              return (
                <div key={tab.id}>
                  <SHAPForcePlot
                    shapValues={shapValuesFormatted}
                    baseValue={baseValue}
                    prediction={predValue}
                    title="SHAP Force Plot (Local Explanation)"
                  />
                </div>
              );

            case 'lime':
              if (!localLime || !localLime.lime_weights) {
                if (localLime?.status === 'pending' || localLimeLoading) {
                  return (
                    <div key={tab.id} className="flex h-64 items-center justify-center border-2 border-dashed border-gray-300 rounded-lg bg-gray-50">
                      <div className="text-center">
                        <Loader2 className="mx-auto h-10 w-10 animate-spin text-purple-500" />
                        <h3 className="mt-4 text-lg font-semibold text-purple-800">Computing LIME...</h3>
                        <p className="mt-2 text-purple-700">This may take a few moments.</p>
                      </div>
                    </div>
                  );
                }
                return (
                  <div key={tab.id} className="rounded-lg border-2 border-dashed border-gray-300 bg-gray-50 p-12 text-center">
                    <Brain className="mx-auto h-16 w-16 text-gray-400" />
                    <h3 className="mt-4 text-xl font-semibold text-gray-900">LIME Local Explanation</h3>
                    <p className="mt-2 text-gray-600 max-w-lg mx-auto">
                      LIME explains the prediction by approximating the model locally with an interpretable model.
                    </p>
                    <div className="mt-6">
                      <button
                        onClick={handleGenerateLime}
                        disabled={requestLimeMutation.isPending}
                        className="inline-flex items-center px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50"
                      >
                        {requestLimeMutation.isPending ? (
                          <>
                            <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                            Generating...
                          </>
                        ) : (
                          'Generate LIME Explanation'
                        )}
                      </button>
                    </div>
                  </div>
                );
              }

              const limeWeights = localLime.lime_weights || [];
              return (
                <div key={tab.id}>
                  <LIMEPlot
                    data={limeWeights}
                    intercept={localLime.lime_intercept}
                    localPred={localLime.lime_local_pred}
                    title="LIME Feature Contributions"
                    height={400}
                  />
                </div>
              );

            case 'explanation': {
              // Gather SHAP data
              const hasShap = !!(localShap?.shap_values && localShap.feature_names);
              const hasLime = !!(localLime?.lime_weights && (localLime.lime_weights as any[]).length > 0);
              const explanationReady = hasShap || hasLime;

              return (
                <ExplanationTab
                  key={tab.id}
                  localShap={localShap}
                  localLime={localLime}
                  predictionValue={typeof prediction.prediction === 'number' ? prediction.prediction : 0}
                  predictionLabel={String(prediction.prediction)}
                  explanationReady={explanationReady}
                />
              );
            }

            default:
              return null;
          }
        })}
      </div>
    </div>
  );
}
