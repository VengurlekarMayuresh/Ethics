'use client';

import { useQuery, useMutation } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { useParams } from 'next/navigation';
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
  Cpu,
  Anchor,
  BookOpen,
} from 'lucide-react';
import Link from 'next/link';
import FeatureImportanceBar from '@/components/charts/FeatureImportanceBar';
import SHAPBeeswarm from '@/components/charts/SHAPBeeswarm';
import SHAPDependence from '@/components/charts/SHAPDependence';
import AlibiRuleDisplay from '@/components/charts/AlibiRuleDisplay';
import AIX360RuleDisplay from '@/components/charts/AIX360RuleDisplay';
import { format } from 'date-fns';

type FrameworkMethod = 'shap' | 'lime' | 'interpretml' | 'alibi' | 'aix360';

interface GlobalExplanation {
  _id: string;
  method: FrameworkMethod;
  explanation_type: 'global';
  // SHAP fields
  shap_values?: number[][];
  expected_value?: number | number[];
  // Shared
  feature_names: string[];
  global_importance?: Array<{ feature: string; importance: number }>;
  lime_global_importance?: Array<{ feature: string; importance: number }>;
  // New frameworks
  explanation_data?: {
    feature_importance?: Array<{ feature: string; importance: number }>;
    rules?: Array<{ rule: string; prediction?: string; confidence?: number; support?: number }>;
    anchor?: { rule?: string; conditions?: string[]; precision?: number; coverage?: number; prediction?: string };
    status?: string;
    error?: string;
  };
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

// ── Per-framework config ────────────────────────────────────────────────────
const FRAMEWORK_CONFIG: Record<
  FrameworkMethod,
  { label: string; color: string; activeClass: string; inactiveClass: string; icon: React.ElementType; postEndpoint: (id: string) => string; getEndpoint: (id: string) => string }
> = {
  shap: {
    label: 'SHAP',
    color: '#3b82f6',
    activeClass: 'bg-indigo-600 text-white',
    inactiveClass: 'bg-white border border-gray-300 text-gray-700 hover:bg-gray-50',
    icon: BarChart3,
    postEndpoint: (id) => `/explain/global/${id}`,
    getEndpoint: (id) => `/explain/global/${id}/latest`,
  },
  lime: {
    label: 'LIME',
    color: '#8b5cf6',
    activeClass: 'bg-purple-600 text-white',
    inactiveClass: 'bg-white border border-gray-300 text-gray-700 hover:bg-gray-50',
    icon: FileText,
    postEndpoint: (id) => `/explain/lime/global/${id}`,
    getEndpoint: (id) => `/explain/lime/global/${id}/latest`,
  },
  interpretml: {
    label: 'InterpretML',
    color: '#0d9488',
    activeClass: 'bg-teal-600 text-white',
    inactiveClass: 'bg-white border border-gray-300 text-gray-700 hover:bg-gray-50',
    icon: Cpu,
    postEndpoint: (id) => `/explain/interpretml/global/${id}`,
    getEndpoint: (id) => `/explain/interpretml/global/${id}/latest`,
  },
  alibi: {
    label: 'Alibi',
    color: '#0ea5e9',
    activeClass: 'bg-sky-600 text-white',
    inactiveClass: 'bg-white border border-gray-300 text-gray-700 hover:bg-gray-50',
    icon: Anchor,
    postEndpoint: (id) => `/explain/alibi/global/${id}`,
    getEndpoint: (id) => `/explain/alibi/global/${id}/latest`,
  },
  aix360: {
    label: 'AIX360',
    color: '#d97706',
    activeClass: 'bg-amber-600 text-white',
    inactiveClass: 'bg-white border border-gray-300 text-gray-700 hover:bg-gray-50',
    icon: BookOpen,
    postEndpoint: (id) => `/explain/aix360/global/${id}`,
    getEndpoint: (id) => `/explain/aix360/global/${id}/latest`,
  },
};

export default function GlobalExplanationPage() {
  const params = useParams();
  const modelId = params.modelId as string;

  const [backgroundFile, setBackgroundFile] = useState<File | null>(null);
  const [requestMethod, setRequestMethod] = useState<FrameworkMethod>('shap');
  const [selectedFeature, setSelectedFeature] = useState<string>('');
  const [dependenceData, setDependenceData] = useState<{ x_values: number[]; shap_values: number[] } | null>(null);
  const [dependenceFeatureFile, setDependenceFeatureFile] = useState<File | null>(null);
  const [isLoadingDependence, setIsLoadingDependence] = useState(false);

  const cfg = FRAMEWORK_CONFIG[requestMethod];

  // ── Switch framework — clear stale result immediately ────────────────────
  const handleSwitchMethod = (method: FrameworkMethod) => {
    if (method === requestMethod) return;
    setRequestMethod(method);
    // Clear uploaded file so the upload UI reappears for the new framework
    setBackgroundFile(null);
    setDependenceData(null);
    setSelectedFeature('');
  };

  // ── Fetch model ──────────────────────────────────────────────────────────
  const { data: model, isLoading: modelLoading } = useQuery<Model>({
    queryKey: ['model', modelId],
    queryFn: async () => {
      const { data } = await api.get(`/models/${modelId}`);
      return data;
    },
    enabled: !!modelId,
  });

  // ── Fetch latest global explanation (per-framework key clears stale data)
  const {
    data: explanation,
    isLoading: explanationLoading,
    error: explanationError,
    refetch: refetchExplanation,
  } = useQuery<GlobalExplanation>({
    // Key includes requestMethod — switching method gives a fresh cache entry
    queryKey: ['globalExplanation', modelId, requestMethod],
    queryFn: async () => {
      const { data } = await api.get(cfg.getEndpoint(modelId));
      return data;
    },
    enabled: !!modelId,
    retry: false,
    refetchInterval: (query) => {
      if (!query.state.data && query.state.fetchStatus !== 'fetching') {
        const error = query.state.error as any;
        const is404 = error?.response?.status === 404;
        return is404 ? 3000 : false;
      }
      return false;
    },
  });

  // ── Request global explanation mutation ─────────────────────────────────
  const requestGlobal = useMutation({
    mutationFn: async () => {
      if (!backgroundFile) throw new Error('Background data file is required');
      const formData = new FormData();
      formData.append('background_data', backgroundFile);
      if (requestMethod === 'lime') formData.append('num_features', '10');
      const { data } = await api.post(cfg.postEndpoint(modelId), formData);
      return data;
    },
    onSuccess: () => {
      setTimeout(() => refetchExplanation(), 3000);
    },
  });

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) setBackgroundFile(e.target.files[0]);
  };

  const handleFeatureFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) setDependenceFeatureFile(e.target.files[0]);
  };

  const loadDependencePlot = async () => {
    if (!selectedFeature || !dependenceFeatureFile) return;
    setIsLoadingDependence(true);
    try {
      const formData = new FormData();
      formData.append('background_data', dependenceFeatureFile);
      const { data } = await api.post(
        `/explain/dependence/${modelId}?feature=${encodeURIComponent(selectedFeature)}`,
        formData,
      );
      setDependenceData({ x_values: data.x_values, shap_values: data.shap_values });
    } catch (error: any) {
      alert('Failed to load dependence plot: ' + (error.response?.data?.detail || error.message));
    } finally {
      setIsLoadingDependence(false);
    }
  };

  // ── Derive display data ──────────────────────────────────────────────────
  const featureImportanceData = (() => {
    if (!explanation) return null;
    if (explanation.global_importance?.length) return explanation.global_importance;
    if (explanation.lime_global_importance?.length) return explanation.lime_global_importance;
    if (explanation.explanation_data?.feature_importance?.length)
      return explanation.explanation_data.feature_importance;
    return null;
  })();

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
        <Link href="/models" className="mt-4 inline-flex items-center text-red-600 hover:text-red-800">
          <ArrowLeft className="mr-2 h-4 w-4" /> Back to models
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div className="flex items-center gap-4">
          <Link
            href={`/models/${modelId}`}
            className="rounded-lg border border-gray-200 bg-white p-2 hover:bg-gray-50 transition"
          >
            <ArrowLeft className="h-5 w-5 text-gray-600" />
          </Link>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Global Explanation</h1>
            <p className="mt-1 text-sm text-gray-500">Model: {model.name}</p>
          </div>
        </div>

        {/* Framework selector — single-select, 5 options */}
        <div className="flex flex-wrap items-center gap-2">
          {(Object.entries(FRAMEWORK_CONFIG) as [FrameworkMethod, typeof cfg][]).map(([method, c]) => {
            const Icon = c.icon;
            const isActive = requestMethod === method;
            return (
              <button
                key={method}
                onClick={() => handleSwitchMethod(method)}
                className={`inline-flex items-center rounded-lg px-3 py-2 text-sm font-medium transition ${
                  isActive ? c.activeClass : c.inactiveClass
                }`}
              >
                <Icon className="mr-1.5 h-4 w-4" />
                {c.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Upload section — shown when no explanation exists for this framework */}
      {!explanation && (
        <div className="rounded-lg border-2 border-dashed border-gray-300 bg-gray-50 p-8">
          <div className="text-center">
            <Upload className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-4 text-lg font-semibold text-gray-900">
              Request Global {cfg.label} Explanation
            </h3>
            <p className="mt-2 text-sm text-gray-600 max-w-lg mx-auto">
              Upload a background CSV dataset representative of your training data.
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
                  onClick={() => requestGlobal.mutate()}
                  disabled={requestGlobal.isPending}
                  className="inline-flex items-center rounded-lg bg-indigo-600 px-6 py-2.5 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
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

      {/* Polling / loading state after triggering */}
      {!explanation && explanationLoading && (
        <div className="flex h-40 items-center justify-center border-2 border-dashed border-gray-200 rounded-lg bg-gray-50">
          <div className="text-center">
            <Loader2 className="mx-auto h-8 w-8 animate-spin text-indigo-500" />
            <p className="mt-3 text-sm text-gray-600">Computing {cfg.label} explanation…</p>
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
              {(explanationError as any).response?.data?.detail ||
                'An error occurred while fetching the explanation.'}
            </p>
            <button
              onClick={() => refetchExplanation()}
              className="mt-2 text-sm font-medium text-red-600 hover:text-red-500"
            >
              Retry
            </button>
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
                  {cfg.label} explanation ready
                </p>
                <p className="text-xs text-green-700">
                  Generated: {format(new Date(explanation.created_at), 'MMM d, yyyy HH:mm')}
                </p>
              </div>
            </div>
            <div className="text-sm font-mono text-green-700">
              ID: {explanation._id.slice(0, 8)}…
            </div>
          </div>

          {/* ── Feature Importance Bar (SHAP / LIME / InterpretML) ─────────── */}
          {featureImportanceData && featureImportanceData.length > 0 && (
            <FeatureImportanceBar
              data={featureImportanceData}
              title={`Global Feature Importance (${cfg.label})`}
              height={400}
              color={cfg.color}
            />
          )}

          {/* ── SHAP-only: Beeswarm ───────────────────────────────────────── */}
          {requestMethod === 'shap' && explanation.shap_values && (
            <SHAPBeeswarm
              shapValues={explanation.shap_values}
              featureNames={explanation.feature_names}
              title="SHAP Beeswarm Plot (Global Distribution)"
              height={500}
            />
          )}

          {/* ── SHAP-only: Dependence plots ───────────────────────────────── */}
          {requestMethod === 'shap' && explanation.feature_names && (
            <div className="rounded-lg border border-gray-200 bg-white p-6">
              <div className="mb-4">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">SHAP Dependence Plots</h3>
                <p className="text-sm text-gray-600 mb-4">
                  Visualize how a specific feature influences model predictions.
                </p>
                <div className="flex flex-wrap items-end gap-4">
                  <div className="min-w-64">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Select Feature
                    </label>
                    <div className="relative">
                      <select
                        value={selectedFeature}
                        onChange={(e) => setSelectedFeature(e.target.value)}
                        className="w-full px-3 py-2 pr-10 border border-gray-300 rounded-lg appearance-none bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                        disabled={isLoadingDependence}
                      >
                        <option value="">-- Choose a feature --</option>
                        {explanation.feature_names.map((f: string) => (
                          <option key={f} value={f}>{f}</option>
                        ))}
                      </select>
                      <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-500 pointer-events-none" />
                    </div>
                  </div>

                  <div className="flex-1 min-w-80">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Background Dataset (CSV)
                    </label>
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
                        Loading…
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
                  </p>
                </div>
              )}
            </div>
          )}

          {/* ── Alibi ─────────────────────────────────────────────────────── */}
          {requestMethod === 'alibi' && explanation.explanation_data && (
            <AlibiRuleDisplay
              explanationData={explanation.explanation_data as any}
              title="Alibi Explain — Global Anchor/ALE"
            />
          )}

          {/* ── AIX360 ───────────────────────────────────────────────────── */}
          {requestMethod === 'aix360' && explanation.explanation_data && (
            <AIX360RuleDisplay
              explanationData={explanation.explanation_data as any}
              title="AIX360 — Global Boolean Rules"
            />
          )}

          {/* About panel */}
          <div className="rounded-lg border border-gray-200 bg-white p-4">
            <h3 className="text-sm font-semibold text-gray-900 mb-2">About this analysis</h3>
            <p className="text-sm text-gray-600">
              This {cfg.label} global explanation shows the overall importance of each feature
              across the background dataset. Features with higher scores have a greater influence
              on the model's predictions on average.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
