'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { useParams, useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import {
  ArrowLeft,
  Loader2,
  AlertCircle,
  CheckCircle,
  BarChart3,
  Brain,
} from 'lucide-react';
import Link from 'next/link';
import SHAPWaterfall from '@/components/charts/SHAPWaterfall';
import LIMEPlot from '@/components/charts/LIMEPlot';
import { format } from 'date-fns';

interface Explanation {
  _id?: string;
  status?: string;
  method: 'shap' | 'lime';
  explanation_type?: 'local' | 'global';
  shap_values?: number[][];
  lime_weights?: any[];
  expected_value?: number;
  lime_intercept?: number;
  lime_local_pred?: number;
  feature_names?: string[];
  nl_explanation?: string;
  created_at?: string;
  error?: string;
}

interface Prediction {
  _id: string;
  model_id: string;
  input_data: Record<string, any>;
  prediction: any;
  probability?: any;
  created_at: string;
}

export default function LocalExplanationPage() {
  const params = useParams();
  const router = useRouter();
  const queryClient = useQueryClient();

  const modelId = params.modelId as string;
  const predictionId = params.predictionId as string;

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

  // Fetch explanation for this prediction
  const {
    data: explanation,
    isLoading: explanationLoading,
    error: explanationError,
    refetch: refetchExplanation,
  } = useQuery<Explanation>({
    queryKey: ['explanation', predictionId],
    queryFn: async () => {
      const { data } = await api.get(`/explain/prediction/${predictionId}`);
      return data;
    },
    enabled: !!predictionId,
    retry: false, // Don't retry on 404 (no explanation yet)
    refetchInterval: (query) => {
      // Poll every 3 seconds if status is pending
      return query.state.data?.status === 'pending' ? 3000 : false;
    },
  });

  // Request explanation mutation
  const requestExplanation = useMutation({
    mutationFn: async (method: 'shap' | 'lime') => {
      const endpoint = method === 'shap' ? `/explain/local/${modelId}` : `/explain/lime/${modelId}`;
      const { data } = await api.post(endpoint, null, {
        params: { prediction_id: predictionId },
      });
      return { ...data, method };
    },
    onSuccess: (data) => {
      // Poll for explanation after a delay
      setTimeout(() => {
        refetchExplanation();
      }, 2000);
    },
  });

  const handleRequestExplanation = (method: 'shap' | 'lime') => {
    requestExplanation.mutate(method);
  };

  // Check if explanation exists
  const hasShap = explanation?.method === 'shap';
  const hasLime = explanation?.method === 'lime';

  // Prepare SHAP data
  const shapData = hasShap ? {
    shapValues: explanation.shap_values?.[0] || [], // For single prediction, first row
    baseValue: explanation.expected_value || 0,
    prediction: typeof prediction?.prediction === 'number' ? prediction.prediction : 0,
    featureNames: explanation.feature_names || [],
  } : null;

  // Prepare LIME data
  const limeData = hasLime ? {
    weights: explanation.lime_weights || [],
    intercept: explanation.lime_intercept,
    localPred: explanation.lime_local_pred,
    featureNames: explanation.feature_names || [],
  } : null;

  // Determine what to show
  const isPending = explanation?.status === 'pending' || requestExplanation.isPending;
  const isFailed = explanation?.status === 'failed';
  const isLoading = predictionLoading || explanationLoading || isPending;

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
            <h1 className="text-2xl font-bold text-gray-900">Explanation</h1>
            <p className="mt-1 text-sm text-gray-500">
              Model: {modelId.slice(0, 8)}... | Prediction: {predictionId.slice(0, 8)}...
            </p>
          </div>
        </div>

        {/* Tabs for SHAP/LIME */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => handleRequestExplanation('shap')}
            disabled={requestExplanation.isPending}
            className={`inline-flex items-center rounded-lg px-4 py-2 text-sm font-medium transition ${
              hasShap && !hasLime
                ? 'bg-indigo-600 text-white'
                : 'bg-white border border-gray-300 text-gray-700 hover:bg-gray-50'
            }`}
          >
            <BarChart3 className="mr-2 h-4 w-4" />
            SHAP
            {!hasShap && !hasLime && requestExplanation.isPending && (
              <Loader2 className="ml-2 h-4 w-4 animate-spin" />
            )}
            {hasShap && <CheckCircle className="ml-2 h-4 w-4" />}
          </button>
          <button
            onClick={() => handleRequestExplanation('lime')}
            disabled={requestExplanation.isPending}
            className={`inline-flex items-center rounded-lg px-4 py-2 text-sm font-medium transition ${
              hasLime && !hasShap
                ? 'bg-purple-600 text-white'
                : 'bg-white border border-gray-300 text-gray-700 hover:bg-gray-50'
            }`}
          >
            <Brain className="mr-2 h-4 w-4" />
            LIME
            {hasLime && <CheckCircle className="ml-2 h-4 w-4" />}
          </button>
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

      {/* Explanation content */}
      {(isLoading || explanationError || isFailed) && (
        <div className="rounded-lg border border-yellow-200 bg-yellow-50 p-8 text-center">
          {isPending ? (
            <>
              <Loader2 className="mx-auto h-10 w-10 animate-spin text-yellow-500" />
              <h3 className="mt-4 text-lg font-semibold text-yellow-800">
                Computing explanation...
              </h3>
              <p className="mt-2 text-yellow-700">
                This may take a few moments depending on model complexity.
              </p>
            </>
          ) : (explanationError || isFailed) ? (
            <>
              <AlertCircle className="mx-auto h-10 w-10 text-red-400" />
              <h3 className="mt-4 text-lg font-semibold text-red-800">Explanation error</h3>
              <p className="mt-2 text-red-700">
                {explanation?.error || explanationError?.response?.data?.detail || 'Failed to compute explanation.'}
              </p>
            </>
          ) : !explanation ? (
            <>
              <Brain className="mx-auto h-10 w-10 text-gray-400" />
              <h3 className="mt-4 text-lg font-semibold text-gray-900">No explanation yet</h3>
              <p className="mt-2 text-gray-600">
                Click SHAP or LIME button above to generate an explanation for this prediction.
              </p>
            </>
          ) : null}
        </div>
      )}

      {explanation && explanation.status !== 'pending' && (
        <div className="space-y-6">
          {/* Natural language explanation */}
          {explanation.nl_explanation && (
            <div className="rounded-lg border border-blue-200 bg-blue-50 p-6">
              <h3 className="text-lg font-semibold text-blue-900 mb-2">Plain English Explanation</h3>
              <p className="text-blue-800 leading-relaxed">{explanation.nl_explanation}</p>
              <p className="text-xs text-blue-600 mt-2">
                Generated by AI • May require review for accuracy
              </p>
            </div>
          )}

          {/* Charts */}
          {hasShap && shapData && (
            <SHAPWaterfall
              shapValues={shapData.shapValues.map((value, idx) => ({
                feature: shapData.featureNames[idx] || `Feature ${idx}`,
                value,
              }))}
              baseValue={shapData.baseValue}
              prediction={shapData.prediction}
              title="SHAP Waterfall Plot"
              height={500}
            />
          )}

          {hasLime && limeData && (
            <LIMEPlot
              data={limeData.weights}
              intercept={limeData.intercept}
              localPred={limeData.localPred}
              title="LIME Feature Contributions"
              height={400}
            />
          )}

          {/* Explanation metadata */}
          <div className="rounded-lg border border-gray-200 bg-white p-4">
            <div className="flex items-center justify-between text-sm text-gray-600">
              <div>
                <span className="font-medium">Method:</span>{' '}
                <span className="capitalize">{explanation.method}</span>
              </div>
              <div>
                <span className="font-medium">Generated:</span>{' '}
                {explanation.created_at ? format(new Date(explanation.created_at), 'MMM d, yyyy HH:mm') : 'N/A'}
              </div>
              <div className="text-xs text-gray-500">
                ID: {explanation._id?.slice(0, 8) || 'N/A'}...
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
