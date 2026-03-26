'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { useParams, useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import {
  ArrowLeft,
  Tag,
  Calendar,
  Database,
  Activity,
  BarChart3,
  Shield,
  GitCompare,
  Trash2,
  Play,
  FileText,
} from 'lucide-react';
import Link from 'next/link';
import { format } from 'date-fns';

interface FeatureSchema {
  name: string;
  type: 'numeric' | 'categorical';
  options?: string[];
  description?: string;
  min?: number | null;
  max?: number | null;
  mean?: number | null;
}

interface Model {
  _id: string;
  name: string;
  description: string;
  framework: string;
  task_type: string;
  feature_schema: FeatureSchema[];
  created_at: string;
  background_data_path?: string;
  model_type?: string;  // e.g., "RandomForestRegressor"
  model_family?: string; // "tree", "linear", "svm", etc.
  is_tree_based?: boolean;
}

export default function ModelDetailPage() {
  const params = useParams();
  const router = useRouter();
  const queryClient = useQueryClient();
  const modelId = params.id as string;

  const { data: model, isLoading, error } = useQuery<Model>({
    queryKey: ['model', modelId],
    queryFn: async () => {
      const { data } = await api.get(`/models/${modelId}`);
      return data;
    },
    enabled: !!modelId,
  });

  const deleteMutation = useMutation({
    mutationFn: async () => {
      await api.delete(`/models/${modelId}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] });
      router.push('/models');
    },
  });

  if (isLoading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  if (error || !model) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-8 text-center">
        <h2 className="text-lg font-semibold text-red-900">Model not found</h2>
        <p className="mt-2 text-red-700">The model you're looking for doesn't exist or you don't have access.</p>
        <Link href="/models" className="mt-4 inline-flex items-center text-red-600 hover:text-red-800">
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to models
        </Link>
      </div>
    );
  }

  const frameworkColors: Record<string, string> = {
    sklearn: 'bg-blue-100 text-blue-800',
    xgboost: 'bg-orange-100 text-orange-800',
    keras: 'bg-red-100 text-red-800',
    onnx: 'bg-purple-100 text-purple-800',
    lightgbm: 'bg-green-100 text-green-800',
  };

  const taskTypeLabels: Record<string, string> = {
    regression: 'Regression',
    classification: 'Classification',
    binary_classification: 'Binary Classification',
    multiclass_classification: 'Multi-class Classification',
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-4">
          <Link
            href="/models"
            className="rounded-lg border border-gray-200 bg-white p-2 hover:bg-gray-50 transition"
          >
            <ArrowLeft className="h-5 w-5 text-gray-600" />
          </Link>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">{model.name}</h1>
            <div className="mt-2 flex items-center gap-3">
              <span className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-medium ${frameworkColors[model.framework] || 'bg-gray-100 text-gray-800'}`}>
                {model.framework}
              </span>
              <span className="inline-flex items-center rounded-full bg-indigo-50 px-2.5 py-1 text-xs font-medium text-indigo-700">
                {taskTypeLabels[model.task_type] || model.task_type}
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Link
            href={`/predict/${modelId}`}
            className="inline-flex items-center rounded-lg bg-green-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-green-700 transition"
          >
            <Play className="mr-2 h-4 w-4" />
            Predict
          </Link>
          <Link
            href={`/explain/global/${modelId}`}
            className="inline-flex items-center rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 transition"
          >
            <BarChart3 className="mr-2 h-4 w-4" />
            Explain
          </Link>
          {model.background_data_path && (
            <Link
              href={`/bias?model=${modelId}`}
              className="inline-flex items-center rounded-lg bg-orange-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-orange-700 transition"
            >
              <Shield className="mr-2 h-4 w-4" />
              Analyze Bias
            </Link>
          )}
          <button
            onClick={() => deleteMutation.mutate()}
            disabled={deleteMutation.isPending}
            className="rounded-lg border border-red-200 bg-white px-4 py-2 text-sm font-medium text-red-600 hover:bg-red-50 transition disabled:opacity-50"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Model Info */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2 space-y-6">
          {/* Description */}
          <div className="rounded-lg border border-gray-200 bg-white p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-3">Description</h2>
            <p className="text-gray-700 whitespace-pre-wrap">
              {model.description || 'No description provided.'}
            </p>
          </div>

          {/* Feature Schema */}
          <div className="rounded-lg border border-gray-200 bg-white p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-3">Input Features</h2>
            {model.feature_schema && model.feature_schema.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                        Feature
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                        Type
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                        Description
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200 bg-white">
                    {model.feature_schema.map((feature) => (
                      <tr key={feature.name}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {feature.name}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          <span className="inline-flex rounded-full bg-gray-100 px-2 py-1 text-xs font-medium text-gray-800">
                            {feature.type}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-500">
                          {feature.description || '-'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-gray-500 italic">No feature schema defined.</p>
            )}
          </div>
        </div>

        <div className="space-y-6">
          {/* Metadata */}
          <div className="rounded-lg border border-gray-200 bg-white p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Metadata</h3>
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <Database className="h-5 w-5 text-gray-400" />
                <div>
                  <p className="text-sm text-gray-500">Framework</p>
                  <p className="font-medium text-gray-900 capitalize">{model.framework}</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <Activity className="h-5 w-5 text-gray-400" />
                <div>
                  <p className="text-sm text-gray-500">Task Type</p>
                  <p className="font-medium text-gray-900">{taskTypeLabels[model.task_type] || model.task_type}</p>
                </div>
              </div>
              {model.model_type && (
                <div className="flex items-center gap-3">
                  <div className="h-5 w-5 flex items-center justify-center">
                    <svg className="h-4 w-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                    </svg>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Algorithm</p>
                    <p className="font-medium text-gray-900">{model.model_type}</p>
                  </div>
                </div>
              )}
              {model.model_family && (
                <div className="flex items-center gap-3">
                  <div className="h-5 w-5 flex items-center justify-center">
                    <svg className="h-4 w-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                    </svg>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Model Family</p>
                    <p className="inline-flex items-center rounded-full bg-purple-50 px-2 py-1 text-xs font-medium text-purple-700 capitalize">
                      {model.model_family}
                    </p>
                  </div>
                </div>
              )}
              <div className="flex items-center gap-3">
                <Calendar className="h-5 w-5 text-gray-400" />
                <div>
                  <p className="text-sm text-gray-500">Created</p>
                  <p className="font-medium text-gray-900">
                    {format(new Date(model.created_at), 'MMM d, yyyy HH:mm')}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <Tag className="h-5 w-5 text-gray-400" />
                <div>
                  <p className="text-sm text-gray-500">ID</p>
                  <p className="font-mono text-sm text-gray-900">{model._id}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="rounded-lg border border-gray-200 bg-white p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
            <div className="space-y-3">
              <Link
                href={`/predict/${modelId}`}
                className="flex items-center justify-between rounded-lg border border-gray-200 p-4 hover:bg-gray-50 transition group"
              >
                <div className="flex items-center gap-3">
                  <div className="rounded-md bg-green-100 p-2">
                    <Play className="h-5 w-5 text-green-600" />
                  </div>
                  <div>
                    <p className="font-medium text-gray-900">Make Prediction</p>
                    <p className="text-xs text-gray-500">Test the model with input data</p>
                  </div>
                </div>
              </Link>

              <Link
                href={`/explain/local/${modelId}/latest`}
                className="flex items-center justify-between rounded-lg border border-gray-200 p-4 hover:bg-gray-50 transition group"
              >
                <div className="flex items-center gap-3">
                  <div className="rounded-md bg-indigo-100 p-2">
                    <BarChart3 className="h-5 w-5 text-indigo-600" />
                  </div>
                  <div>
                    <p className="font-medium text-gray-900">Local Explanation</p>
                    <p className="text-xs text-gray-500">SHAP/LIME for a single prediction</p>
                  </div>
                </div>
              </Link>

              <Link
                href={`/explain/global/${modelId}`}
                className="flex items-center justify-between rounded-lg border border-gray-200 p-4 hover:bg-gray-50 transition group"
              >
                <div className="flex items-center gap-3">
                  <div className="rounded-md bg-purple-100 p-2">
                    <BarChart3 className="h-5 w-5 text-purple-600" />
                  </div>
                  <div>
                    <p className="font-medium text-gray-900">Global Explanation</p>
                    <p className="text-xs text-gray-500">Overall feature importance</p>
                  </div>
                </div>
              </Link>

              <Link
                href={`/bias?model=${modelId}`}
                className="flex items-center justify-between rounded-lg border border-gray-200 p-4 hover:bg-gray-50 transition group"
              >
                <div className="flex items-center gap-3">
                  <div className="rounded-md bg-orange-100 p-2">
                    <Shield className="h-5 w-5 text-orange-600" />
                  </div>
                  <div>
                    <p className="font-medium text-gray-900">Bias Analysis</p>
                    <p className="text-xs text-gray-500">Check fairness metrics</p>
                  </div>
                </div>
              </Link>

              <Link
                href={`/compare?models=${modelId}`}
                className="flex items-center justify-between rounded-lg border border-gray-200 p-4 hover:bg-gray-50 transition group"
              >
                <div className="flex items-center gap-3">
                  <div className="rounded-md bg-blue-100 p-2">
                    <GitCompare className="h-5 w-5 text-blue-600" />
                  </div>
                  <div>
                    <p className="font-medium text-gray-900">Compare Models</p>
                    <p className="text-xs text-gray-500">Side-by-side comparison</p>
                  </div>
                </div>
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
