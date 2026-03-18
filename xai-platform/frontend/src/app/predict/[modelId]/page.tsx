'use client';

import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { useParams } from 'next/navigation';
import { useEffect, useState } from 'react';
import { ArrowLeft, History } from 'lucide-react';
import Link from 'next/link';
import PredictionForm from '@/components/forms/PredictionForm';
import { format } from 'date-fns';

interface FeatureSchema {
  name: string;
  type: 'numeric' | 'categorical';
  options?: string[];
  description?: string;
}

interface Model {
  _id: string;
  name: string;
  description: string;
  framework: string;
  task_type: string;
  feature_schema: FeatureSchema[];
  created_at: string;
}

export default function PredictPage() {
  const params = useParams();
  const modelId = params.modelId as string;

  const { data: model, isLoading, error } = useQuery<Model>({
    queryKey: ['model', modelId],
    queryFn: async () => {
      const { data } = await api.get(`/models/${modelId}`);
      return data;
    },
    enabled: !!modelId,
  });

  const featureSchema: FeatureSchema[] = model?.feature_schema || [];

  const handlePredictionSuccess = (predictionId: string) => {
    console.log('Prediction created:', predictionId);
    // Optionally redirect or show more details
  };

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
        <p className="mt-2 text-red-700">
          The model you're trying to predict with doesn't exist or you don't have access.
        </p>
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
            <h1 className="text-2xl font-bold text-gray-900">Predict with {model.name}</h1>
            <p className="mt-1 text-sm text-gray-500">
              Enter values for the model features to get a prediction.
            </p>
          </div>
        </div>

        <Link
          href="/predict/history"
          className="inline-flex items-center rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 transition"
        >
          <History className="mr-2 h-4 w-4" />
          View History
        </Link>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Prediction Form */}
        <div className="lg:col-span-2">
          <div className="rounded-lg border border-gray-200 bg-white p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Input Data</h2>
            <PredictionForm
              modelId={modelId}
              featureSchema={featureSchema}
              onSuccess={handlePredictionSuccess}
            />
          </div>
        </div>

        {/* Model Info Sidebar */}
        <div className="space-y-6">
          <div className="rounded-lg border border-gray-200 bg-white p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Details</h3>
            <div className="space-y-4">
              <div>
                <p className="text-sm text-gray-500">Framework</p>
                <p className="font-medium text-gray-900 capitalize">{model.framework}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Task Type</p>
                <p className="font-medium text-gray-900">{model.task_type}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Features</p>
                <p className="font-medium text-gray-900">{featureSchema.length}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Description</p>
                <p className="text-sm text-gray-700">{model.description || 'No description'}</p>
              </div>
            </div>
          </div>

          <div className="rounded-lg border border-gray-200 bg-white p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Tips</h3>
            <ul className="space-y-2 text-sm text-gray-600">
              <li className="flex items-start gap-2">
                <span className="mt-1 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-indigo-600"></span>
                <span>Enter values that match the expected data types (numeric or categorical).</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-1 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-indigo-600"></span>
                <span>For categorical features, select from the dropdown options.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-1 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-indigo-600"></span>
                <span>After prediction, you can view SHAP or LIME explanations.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-1 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-indigo-600"></span>
                <span>All predictions are saved to your history.</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
