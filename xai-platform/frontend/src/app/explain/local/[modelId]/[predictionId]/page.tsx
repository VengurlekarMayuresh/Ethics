'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { useParams } from 'next/navigation';
import { useState, useRef, useEffect } from 'react';
import {
  ArrowLeft,
  Loader2,
  AlertCircle,
  CheckCircle,
  BarChart3,
  Brain,
  FileText,
  Cpu,
  Anchor,
  BookOpen,
  Sparkles,
} from 'lucide-react';
import Link from 'next/link';
import SHAPForcePlot from '@/components/charts/SHAPForcePlot';
import LIMEPlot from '@/components/charts/LIMEPlot';
import FeatureImportanceBar from '@/components/charts/FeatureImportanceBar';
import AIX360RuleDisplay from '@/components/charts/AIX360RuleDisplay';
import { format } from 'date-fns';
import ReactMarkdown from 'react-markdown';

// ── Types ────────────────────────────────────────────────────────────────────

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
  method: string;
  explanation_type: 'local';
  // SHAP
  shap_values?: number[][];
  expected_value?: number;
  feature_names?: string[];
  // LIME
  lime_weights?: Array<{ feature: string; weight: number }> | any[];
  lime_intercept?: number | Record<string, number>;
  lime_local_pred?: number | number[];
  explained_class?: string;
  // New frameworks
  explanation_data?: Record<string, any>;
  created_at?: string;
  status?: 'pending' | 'complete' | 'failed';
  error?: string;
}

type TabId = 'shap-force' | 'lime' | 'interpretml' | 'aix360' | 'explanation';

// ── ExplanationTab (AI NL generation) ────────────────────────────────────────

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
      let shapValues: number[] | undefined;
      let shapFeatureNames: string[] | undefined;
      if (localShap?.shap_values && localShap.feature_names) {
        const raw: number[][] | number[] = localShap.shap_values;
        shapValues = Array.isArray(raw[0]) ? (raw[0] as number[]) : (raw as number[]);
        shapFeatureNames = localShap.feature_names;
      }
      let baseVal: number | undefined;
      if (typeof localShap?.expected_value === 'number') baseVal = localShap.expected_value;
      else if (Array.isArray(localShap?.expected_value)) baseVal = Number(localShap.expected_value[0]);

      let limeW: { feature: string; weight: number }[] | undefined;
      if (localLime?.lime_weights) {
        limeW = localLime.lime_weights.map((item: any) => {
          if (typeof item === 'object' && !Array.isArray(item) && 'feature' in item && 'weight' in item) {
            return { feature: String(item.feature), weight: Number(item.weight) };
          }
          if (Array.isArray(item)) {
            return { feature: String(item[0]), weight: Number(item[1]) };
          }
          return { feature: String(item), weight: 0 };
        });
      }

      let limePred: number | undefined;
      if (typeof localLime?.lime_local_pred === 'number') limePred = localLime.lime_local_pred;
      else if (Array.isArray(localLime?.lime_local_pred)) limePred = Number(localLime.lime_local_pred[0]);

      const payload = {
        prediction_label: predictionLabel,
        prediction_value: predictionValue,
        shap_feature_names: shapFeatureNames,
        shap_values: shapValues,
        shap_base_value: baseVal,
        lime_weights: limeW,
        lime_local_pred: limePred,
      };
      const { data } = await api.post('/explain/nl-generate', payload);
      setNlText(data.explanation);
      setSource(data.source);
    } catch (e: any) {
      let errMsg = e?.response?.data?.detail;
      if (!errMsg) {
        errMsg = e.message || 'Failed to generate explanation.';
      } else if (typeof errMsg !== 'string') {
        errMsg = JSON.stringify(errMsg);
      }
      setError(errMsg);
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
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
            {loading ? 'Generating…' : nlText ? 'Regenerate' : 'Generate Explanation'}
          </button>
        </div>
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

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700 flex items-center gap-2">
          <AlertCircle className="h-4 w-4 flex-shrink-0" />{error}
        </div>
      )}

      {loading && (
        <div className="flex items-center justify-center py-16">
          <div className="text-center">
            <Loader2 className="mx-auto h-10 w-10 animate-spin text-indigo-500" />
            <p className="mt-3 text-gray-600 font-medium">Analysing predictions…</p>
            <p className="text-sm text-gray-500">This usually takes a few seconds.</p>
          </div>
        </div>
      )}

      {nlText && !loading && (
        <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-gray-900 flex items-center gap-2">
              <Brain className="h-5 w-5 text-indigo-500" /> Explanation
            </h3>
            {source && (
              <span className={`text-xs px-2 py-1 rounded-full font-medium ${
                source === 'openrouter' ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'
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

// ── EmptyTabPrompt — shown for new frameworks when no data exists yet ─────────

function EmptyTabPrompt({
  label,
  description,
  color,
  icon: Icon,
  onGenerate,
  isPending,
}: {
  label: string;
  description: string;
  color: string;
  icon: React.ElementType;
  onGenerate: () => void;
  isPending: boolean;
}) {
  return (
    <div className="rounded-lg border-2 border-dashed border-gray-300 bg-gray-50 p-12 text-center">
      <Icon className="mx-auto h-16 w-16 text-gray-400" />
      <h3 className="mt-4 text-xl font-semibold text-gray-900">{label} Explanation</h3>
      <p className="mt-2 text-gray-600 max-w-lg mx-auto">{description}</p>
      <div className="mt-6">
        <button
          onClick={onGenerate}
          disabled={isPending}
          style={isPending ? {} : { backgroundColor: color }}
          className={`inline-flex items-center px-6 py-3 text-white rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition`}
        >
          {isPending ? (
            <>
              <Loader2 className="mr-2 h-5 w-5 animate-spin" />
              Generating…
            </>
          ) : (
            `Generate ${label} Explanation`
          )}
        </button>
      </div>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function UnifiedExplanationPage() {
  const params = useParams();
  const queryClient = useQueryClient();

  const modelId = params.modelId as string;
  const predictionId = params.predictionId as string;

  const [activeTab, setActiveTab] = useState<TabId>('shap-force');
  const [autoTriggered, setAutoTriggered] = useState(false);
  const [globalShapBackgroundFile, setGlobalShapBackgroundFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── Fetch prediction ─────────────────────────────────────────────────────
  const { data: prediction, isLoading: predictionLoading, error: predictionError } =
    useQuery<Prediction>({
      queryKey: ['prediction', predictionId],
      queryFn: async () => {
        const { data } = await api.get(`/predict/${predictionId}`);
        return data;
      },
      enabled: !!predictionId,
    });

  // ── Global SHAP (for optional context) ───────────────────────────────────
  const { data: globalShap, refetch: refetchGlobalShap } = useQuery<GlobalExplanation>({
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

  // ── SHAP local — auto-poll until done ────────────────────────────────────
  const { data: localShap, isLoading: localShapLoading, error: localShapError } =
    useQuery<LocalExplanation>({
      queryKey: ['localExplanation', predictionId, 'shap'],
      queryFn: async () => {
        try {
          const { data } = await api.get(`/explain/prediction/${predictionId}?method=shap`);
          return data;
        } catch (error: any) {
          if (error.response?.status === 404) return null;
          throw error;
        }
      },
      enabled: !!predictionId,
      retry: false,
      refetchInterval: (query) => {
        const d = query.state.data as LocalExplanation | null | undefined;
        if (d?.shap_values && d.shap_values.length > 0) return false;
        return 5000;
      },
    });

  // ── LIME local — auto-poll until done ────────────────────────────────────
  const { data: localLime, isLoading: localLimeLoading, error: localLimeError } =
    useQuery<LocalExplanation>({
      queryKey: ['localExplanation', predictionId, 'lime'],
      queryFn: async () => {
        try {
          const { data } = await api.get(`/explain/prediction/${predictionId}?method=lime`);
          return data;
        } catch (error: any) {
          if (error.response?.status === 404) return null;
          throw error;
        }
      },
      enabled: !!predictionId,
      retry: false,
      refetchInterval: (query) => {
        const d = query.state.data as LocalExplanation | null | undefined;
        if (d?.lime_weights && (d.lime_weights as any[]).length > 0) return false;
        return 5000;
      },
    });

  // ── NEW: InterpretML local — manual trigger, polls until done ────────────
  const { data: localInterpretml, isLoading: localInterpretmlLoading } =
    useQuery<LocalExplanation>({
      queryKey: ['localExplanation', predictionId, 'interpretml'],
      queryFn: async () => {
        try {
          const { data } = await api.get(`/explain/prediction/${predictionId}?method=interpretml`);
          return data;
        } catch (error: any) {
          if (error.response?.status === 404) return null;
          throw error;
        }
      },
      enabled: activeTab === 'interpretml' && !!predictionId,
      retry: false,
      refetchInterval: (query) => {
        const d = query.state.data as LocalExplanation | null | undefined;
        if (d?.explanation_data) return false;
        if (activeTab !== 'interpretml') return false;
        return 5000;
      },
    });

  // ── NEW: AIX360 local ────────────────────────────────────────────────────
  const { data: localAix360, isLoading: localAix360Loading } =
    useQuery<LocalExplanation>({
      queryKey: ['localExplanation', predictionId, 'aix360'],
      queryFn: async () => {
        try {
          const { data } = await api.get(`/explain/prediction/${predictionId}?method=aix360`);
          return data;
        } catch (error: any) {
          if (error.response?.status === 404) return null;
          throw error;
        }
      },
      enabled: activeTab === 'aix360' && !!predictionId,
      retry: false,
      refetchInterval: (query) => {
        const d = query.state.data as LocalExplanation | null | undefined;
        if (d?.explanation_data) return false;
        if (activeTab !== 'aix360') return false;
        return 5000;
      },
    });

  // ── Mutations ─────────────────────────────────────────────────────────────
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

  const requestInterpretmlMutation = useMutation({
    mutationFn: async () => {
      const { data } = await api.post(`/explain/interpretml/${modelId}`, null, {
        params: { prediction_id: predictionId },
      });
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['localExplanation', predictionId, 'interpretml'] });
    },
  });

  const requestAix360Mutation = useMutation({
    mutationFn: async () => {
      const { data } = await api.post(`/explain/aix360/${modelId}`, null, {
        params: { prediction_id: predictionId },
      });
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['localExplanation', predictionId, 'aix360'] });
    },
  });

  const requestGlobalShapMutation = useMutation({
    mutationFn: async () => {
      if (!globalShapBackgroundFile) throw new Error('File required');
      const formData = new FormData();
      formData.append('background_data', globalShapBackgroundFile);
      const { data } = await api.post(`/explain/global/${modelId}`, formData);
      return data;
    },
    onSuccess: () => {
      setTimeout(() => refetchGlobalShap(), 3000);
    },
  });

  // ── Auto-trigger SHAP + LIME once on load ────────────────────────────────
  useEffect(() => {
    if (autoTriggered || !predictionId) return;
    setAutoTriggered(true);

    const shapComplete = localShap?.shap_values && (localShap.shap_values as number[][]).length > 0;
    if (!shapComplete && !requestShapMutation.isPending) requestShapMutation.mutate();

    const limeComplete = localLime?.lime_weights && (localLime.lime_weights as any[]).length > 0;
    if (!limeComplete && !requestLimeMutation.isPending) requestLimeMutation.mutate();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [predictionId]);

  // ── Tabs config ───────────────────────────────────────────────────────────
  const tabs: Array<{ id: TabId; label: string; icon: React.ElementType }> = [
    { id: 'shap-force', label: 'SHAP Force Plot', icon: BarChart3 },
    { id: 'lime', label: 'LIME', icon: Brain },
    { id: 'interpretml', label: 'InterpretML', icon: Cpu },
    { id: 'aix360', label: 'AIX360', icon: BookOpen },
    { id: 'explanation', label: 'AI Explanation', icon: FileText },
  ];

  const isTabLoading = (tab: TabId) => {
    switch (tab) {
      case 'shap-force': return localShapLoading || localShap?.status === 'pending';
      case 'lime': return localLimeLoading || localLime?.status === 'pending';
      case 'interpretml': return localInterpretmlLoading || requestInterpretmlMutation.isPending;
      case 'aix360': return localAix360Loading || requestAix360Mutation.isPending;
      default: return false;
    }
  };

  const getTabError = (tab: TabId) => {
    switch (tab) {
      case 'shap-force': return localShapError;
      case 'lime': return localLimeError;
      default: return null;
    }
  };

  // ── Guards ────────────────────────────────────────────────────────────────
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
        <p className="mt-2 text-red-700">
          The prediction you're looking for doesn't exist or you don't have access.
        </p>
        <Link href="/predict/history" className="mt-4 inline-flex items-center text-red-600 hover:text-red-800">
          <ArrowLeft className="mr-2 h-4 w-4" /> Back to prediction history
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
              Model: {modelId.slice(0, 8)}… | Prediction: {predictionId.slice(0, 8)}…
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
              {/* Use prediction_confidence first (sanitized by backend), fallback to probability[0] or confidence calc */}
              {(prediction.prediction_confidence !== undefined || prediction.probability !== undefined) && (
                <p className="text-sm text-indigo-600 mt-1">
                  Probability: {(() => {
                    const conf = prediction.prediction_confidence ?? 
                                (Array.isArray(prediction.probability) 
                                 ? Math.max(...prediction.probability.map(Number)) 
                                 : Number(prediction.probability));
                    return !isNaN(conf) ? (conf * 100).toFixed(1) + '%' : 'N/A';
                  })()}
                </p>
              )}
            </div>
            <p className="text-xs text-gray-500 mt-2">
              Generated: {prediction.created_at
                ? format(new Date(prediction.created_at), 'MMM d, yyyy HH:mm')
                : 'N/A'}
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
                className={`group relative min-w-fit py-4 px-5 border-b-2 font-medium text-sm transition-colors whitespace-nowrap ${
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
          if (activeTab !== tab.id) return null;

          const isLoading = isTabLoading(tab.id);
          const error = getTabError(tab.id);

          // Shared loading spinner
          if (isLoading && tab.id !== 'interpretml' && tab.id !== 'alibi' && tab.id !== 'aix360') {
            return (
              <div key={tab.id} className="flex h-64 items-center justify-center border-2 border-dashed border-gray-300 rounded-lg bg-gray-50">
                <div className="text-center">
                  <Loader2 className="mx-auto h-10 w-10 animate-spin text-indigo-600" />
                  <h3 className="mt-4 text-lg font-semibold text-gray-900">Loading…</h3>
                  <p className="mt-2 text-gray-600">Please wait while we load the explanation.</p>
                </div>
              </div>
            );
          }

          // Shared error state (non-404)
          if (error && (error as any).response?.status !== 404) {
            return (
              <div key={tab.id} className="rounded-lg border border-red-200 bg-red-50 p-8 text-center">
                <AlertCircle className="mx-auto h-12 w-12 text-red-400" />
                <h3 className="mt-4 text-lg font-semibold text-red-800">Failed to load explanation</h3>
                <p className="mt-2 text-red-700">{(error as any)?.response?.data?.detail || 'An error occurred.'}</p>
                <button
                  onClick={() => window.location.reload()}
                  className="mt-4 inline-flex items-center px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
                >
                  Retry
                </button>
              </div>
            );
          }

          // ── Tab-specific content ─────────────────────────────────────────
          switch (tab.id) {
            // ── SHAP Force Plot ──────────────────────────────────────────────
            case 'shap-force': {
              if (!localShap?.shap_values) {
                if (localShap?.status === 'pending' || localShapLoading) {
                  return (
                    <div key={tab.id} className="flex h-64 items-center justify-center border-2 border-dashed border-gray-300 rounded-lg bg-gray-50">
                      <div className="text-center">
                        <Loader2 className="mx-auto h-10 w-10 animate-spin text-yellow-500" />
                        <h3 className="mt-4 text-lg font-semibold text-yellow-800">Computing SHAP…</h3>
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
                      Shows how each feature contributed to push the prediction from the base value.
                    </p>
                    <div className="mt-6">
                      <button
                        onClick={() => requestShapMutation.mutate()}
                        disabled={requestShapMutation.isPending}
                        className="inline-flex items-center px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50"
                      >
                        {requestShapMutation.isPending ? (
                          <><Loader2 className="mr-2 h-5 w-5 animate-spin" />Generating…</>
                        ) : 'Generate SHAP Explanation'}
                      </button>
                    </div>
                  </div>
                );
              }

              const baseValue = localShap.expected_value ?? 0;
              const predValue = typeof prediction.prediction === 'number' ? prediction.prediction : 0;
              const shapValuesRaw = Array.isArray(localShap.shap_values) ? localShap.shap_values : [];
              const firstRow: number[] = Array.isArray(shapValuesRaw[0])
                ? (shapValuesRaw[0] as number[])
                : (shapValuesRaw as unknown as number[]);
              const featureNames = localShap.feature_names || [];
              const shapValuesFormatted = firstRow.map((val, idx) => ({
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
            }

            // ── LIME ────────────────────────────────────────────────────────
            case 'lime': {
              if (!localLime?.lime_weights) {
                if (localLime?.status === 'pending' || localLimeLoading) {
                  return (
                    <div key={tab.id} className="flex h-64 items-center justify-center border-2 border-dashed border-gray-300 rounded-lg bg-gray-50">
                      <div className="text-center">
                        <Loader2 className="mx-auto h-10 w-10 animate-spin text-purple-500" />
                        <h3 className="mt-4 text-lg font-semibold text-purple-800">Computing LIME…</h3>
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
                      LIME explains the prediction by approximating the model locally.
                    </p>
                    <div className="mt-6">
                      <button
                        onClick={() => requestLimeMutation.mutate()}
                        disabled={requestLimeMutation.isPending}
                        className="inline-flex items-center px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50"
                      >
                        {requestLimeMutation.isPending ? (
                          <><Loader2 className="mr-2 h-5 w-5 animate-spin" />Generating…</>
                        ) : 'Generate LIME Explanation'}
                      </button>
                    </div>
                  </div>
                );
              }

              // Normalize lime_weights: backend now returns [{feature, weight}]
              const limeWeights = (localLime.lime_weights || []).map((item: any) => {
                if (typeof item === 'object' && 'feature' in item && 'weight' in item) {
                  return { feature: String(item.feature), weight: Number(item.weight) };
                }
                // Legacy format: [feature_str, weight_num] tuple
                if (Array.isArray(item)) {
                  return { feature: String(item[0]), weight: Number(item[1]) };
                }
                return { feature: String(item), weight: 0 };
              });

              // Normalize intercept: backend returns {"1": 0.42} → grab first value
              const rawIntercept = localLime.lime_intercept;
              const interceptNum: number | undefined =
                typeof rawIntercept === 'number'
                  ? rawIntercept
                  : rawIntercept && typeof rawIntercept === 'object'
                    ? Number(Object.values(rawIntercept)[0])
                    : undefined;

              // Normalize localPred: backend returns [0.37] → grab first value
              const rawLocalPred = localLime.lime_local_pred;
              const localPredNum: number | undefined =
                typeof rawLocalPred === 'number'
                  ? rawLocalPred
                  : Array.isArray(rawLocalPred) && rawLocalPred.length > 0
                    ? Number(rawLocalPred[0])
                    : undefined;

              return (
                <div key={tab.id}>
                  <LIMEPlot
                    data={limeWeights}
                    intercept={interceptNum}
                    localPred={localPredNum}
                    explainedClass={localLime.explained_class}
                    title="LIME Feature Contributions"
                    height={400}
                  />
                </div>
              );
            }

            // ── InterpretML ──────────────────────────────────────────────────
            case 'interpretml': {
              if (localInterpretmlLoading || requestInterpretmlMutation.isPending) {
                return (
                  <div key={tab.id} className="flex h-64 items-center justify-center border-2 border-dashed border-gray-300 rounded-lg bg-gray-50">
                    <div className="text-center">
                      <Loader2 className="mx-auto h-10 w-10 animate-spin text-teal-500" />
                      <h3 className="mt-4 text-lg font-semibold text-teal-800">Computing InterpretML…</h3>
                      <p className="mt-2 text-teal-700">This may take a few moments.</p>
                    </div>
                  </div>
                );
              }

              const impData = localInterpretml?.explanation_data?.feature_importance;
              if (!impData?.length) {
                return (
                  <EmptyTabPrompt
                    key={tab.id}
                    label="InterpretML"
                    description="InterpretML uses Explainable Boosting Machines (EBM) to identify each feature's contribution as a clear, quantified importance score."
                    color="#0d9488"
                    icon={Cpu}
                    onGenerate={() => requestInterpretmlMutation.mutate()}
                    isPending={requestInterpretmlMutation.isPending}
                  />
                );
              }

              return (
                <div key={tab.id}>
                  <FeatureImportanceBar
                    data={impData}
                    title="InterpretML — Feature Importance"
                    color="#0d9488"
                    height={400}
                  />
                </div>
              );
            }

            // ── AIX360 ───────────────────────────────────────────────────────
            case 'aix360': {
              if (localAix360Loading || requestAix360Mutation.isPending) {
                return (
                  <div key={tab.id} className="flex h-64 items-center justify-center border-2 border-dashed border-gray-300 rounded-lg bg-gray-50">
                    <div className="text-center">
                      <Loader2 className="mx-auto h-10 w-10 animate-spin text-amber-500" />
                      <h3 className="mt-4 text-lg font-semibold text-amber-800">Computing AIX360…</h3>
                      <p className="mt-2 text-amber-700">This may take a few moments.</p>
                    </div>
                  </div>
                );
              }

              if (!localAix360?.explanation_data) {
                return (
                  <EmptyTabPrompt
                    key={tab.id}
                    label="AIX360"
                    description="AIX360 generates human-readable Boolean rules that describe why the model made this prediction. Rules can be audited without a data science background."
                    color="#d97706"
                    icon={BookOpen}
                    onGenerate={() => requestAix360Mutation.mutate()}
                    isPending={requestAix360Mutation.isPending}
                  />
                );
              }

              return (
                <div key={tab.id}>
                  <AIX360RuleDisplay
                    explanationData={localAix360.explanation_data as any}
                    title="AIX360 — Boolean Rules (Local)"
                  />
                </div>
              );
            }

            // ── AI Explanation (NL) ──────────────────────────────────────────
            case 'explanation': {
              const hasShap = !!(localShap?.shap_values && localShap.feature_names);
              const hasLime = !!(localLime?.lime_weights && (localLime.lime_weights as any[]).length > 0);
              return (
                <ExplanationTab
                  key={tab.id}
                  localShap={localShap}
                  localLime={localLime}
                  predictionValue={typeof prediction.prediction === 'number' ? prediction.prediction : 0}
                  predictionLabel={String(prediction.prediction)}
                  explanationReady={hasShap || hasLime}
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
