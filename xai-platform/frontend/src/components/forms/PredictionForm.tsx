'use client';

import React, { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { Loader2, Play } from 'lucide-react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';

interface FeatureSchema {
  name: string;
  type: 'numeric' | 'categorical';
  options?: string[];
  description?: string;
  min?: number | null;
  max?: number | null;
  mean?: number | null;
}

interface PredictionFormProps {
  modelId: string;
  featureSchema: FeatureSchema[];
  onSuccess?: (predictionId: string) => void;
}

export default function PredictionForm({
  modelId,
  featureSchema,
  onSuccess,
}: PredictionFormProps) {
  const router = useRouter();
  const queryClient = useQueryClient();
  const [formData, setFormData] = useState<Record<string, any>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});

  const predictMutation = useMutation({
    mutationFn: async (data: Record<string, any>) => {
      const formData = new FormData();
      formData.append('input_data', JSON.stringify(data));
      const response = await api.post(`/predict/${modelId}`, formData);
      return response.data;
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['predictions'] });
      onSuccess?.(data.prediction_id);
    },
    onError: (error: any) => {
      const message = error.response?.data?.detail || 'Prediction failed';
      setErrors({ submit: message });
    },
  });

  const handleInputChange = (featureName: string, value: string, type: string) => {
    setFormData(prev => ({
      ...prev,
      [featureName]: type === 'numeric' ? (value === '' ? '' : parseFloat(value)) : value,
    }));
    // Clear error when user modifies field
    if (errors[featureName]) {
      setErrors(prev => {
        const newErrors = { ...prev };
        delete newErrors[featureName];
        return newErrors;
      });
    }
  };

  const validate = () => {
    const newErrors: Record<string, string> = {};

    featureSchema.forEach(feature => {
      const value = formData[feature.name];

      if (value === undefined || value === '') {
        newErrors[feature.name] = `${feature.name} is required`;
        return;
      }

      if (feature.type === 'numeric') {
        const numValue = Number(value);
        if (isNaN(numValue)) {
          newErrors[feature.name] = `${feature.name} must be a number`;
        } else {
          if (feature.min !== null && feature.min !== undefined && numValue < feature.min) {
            newErrors[feature.name] = `${feature.name} must be at least ${feature.min}`;
          }
          if (feature.max !== null && feature.max !== undefined && numValue > feature.max) {
            newErrors[feature.name] = `${feature.name} must be at most ${feature.max}`;
          }
        }
      } else if (feature.type === 'categorical' && feature.options && !feature.options.includes(value)) {
        newErrors[feature.name] = `${feature.name} must be one of the allowed options`;
      }
    });

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (validate()) {
      predictMutation.mutate(formData);
    }
  };

  const resetForm = () => {
    setFormData({});
    setErrors({});
  };

  if (featureSchema.length === 0) {
    return (
      <div className="rounded-lg border border-yellow-200 bg-yellow-50 p-6 text-center">
        <p className="text-yellow-800">
          This model does not have a defined feature schema. Please upload a model with proper metadata.
        </p>
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="grid grid-cols-1 gap-6">
        {featureSchema.map((feature) => (
          <div key={feature.name} className="space-y-2">
            <label htmlFor={feature.name} className="block text-sm font-medium text-gray-700">
              {feature.name}
              {feature.type === 'numeric' && (
                <span className="ml-1 text-gray-400">(number)</span>
              )}
            </label>

            {feature.description && (
              <p className="text-xs text-gray-500">{feature.description}</p>
            )}

            {feature.type === 'categorical' && feature.options && (
              <select
                id={feature.name}
                value={formData[feature.name] ?? ''}
                onChange={(e) => handleInputChange(feature.name, e.target.value, 'categorical')}
                className={`block w-full rounded-lg border ${
                  errors[feature.name] ? 'border-red-300' : 'border-gray-300'
                } px-3 py-2 shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 sm:text-sm`}
              >
                <option value="">Select an option</option>
                {feature.options.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            )}

            {feature.type === 'numeric' && (
              <div>
                <input
                  type="number"
                  id={feature.name}
                  step="any"
                  value={formData[feature.name] ?? ''}
                  onChange={(e) => handleInputChange(feature.name, e.target.value, 'numeric')}
                  min={feature.min !== null && feature.min !== undefined ? feature.min : undefined}
                  max={feature.max !== null && feature.max !== undefined ? feature.max : undefined}
                  className={`block w-full rounded-lg border ${
                    errors[feature.name] ? 'border-red-300' : 'border-gray-300'
                  } px-3 py-2 shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 sm:text-sm`}
                  placeholder="Enter a number"
                />
                {feature.min !== null && feature.max !== null && (
                  <p className="mt-1 text-xs text-gray-500">
                    Range: {feature.min} to {feature.max}
                  </p>
                )}
              </div>
            )}

            {errors[feature.name] && (
              <p className="text-sm text-red-600">{errors[feature.name]}</p>
            )}
          </div>
        ))}
      </div>

      {errors.submit && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4">
          <p className="text-sm text-red-800">{errors.submit}</p>
        </div>
      )}

      <div className="flex items-center gap-4">
        <button
          type="submit"
          disabled={predictMutation.isPending}
          className="inline-flex items-center rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 transition"
        >
          {predictMutation.isPending ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Predicting...
            </>
          ) : (
            <>
              <Play className="mr-2 h-4 w-4" />
              Get Prediction
            </>
          )}
        </button>

        <button
          type="button"
          onClick={resetForm}
          className="rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition"
        >
          Reset
        </button>
      </div>

      {predictMutation.isSuccess && (
        <div className="rounded-lg border border-green-200 bg-green-50 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-green-800">Prediction successful!</p>
              <p className="mt-1 text-sm text-green-700">
                Prediction ID: {predictMutation.data.prediction_id}
              </p>
            </div>
            <div className="flex gap-2">
              <Link
                href={`/explain/local/${modelId}/${predictMutation.data.prediction_id}`}
                className="inline-flex items-center rounded-md bg-green-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-green-700 transition"
              >
                Get SHAP Explanation
              </Link>
              <Link
                href={`/predict/history`}
                className="inline-flex items-center rounded-md border border-green-300 bg-white px-3 py-1.5 text-xs font-medium text-green-700 hover:bg-green-50 transition"
              >
                View History
              </Link>
            </div>
          </div>
        </div>
      )}
    </form>
  );
}
