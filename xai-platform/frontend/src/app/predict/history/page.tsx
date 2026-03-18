'use client';

import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { format } from 'date-fns';
import { Activity, ArrowLeft, Eye } from 'lucide-react';
import Link from 'next/link';

interface Prediction {
  _id: string;
  model_id: string;
  input_data: Record<string, any>;
  prediction: any;
  probability?: any;
  created_at: string;
}

export default function PredictionHistoryPage() {
  const { data: predictions, isLoading, error } = useQuery<Prediction[]>({
    queryKey: ['predictionHistory'],
    queryFn: async () => {
      const { data } = await api.get('/predict/history');
      return data;
    },
  });

  if (isLoading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  if (error || !predictions) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-8 text-center">
        <h2 className="text-lg font-semibold text-red-900">Failed to load predictions</h2>
        <p className="mt-2 text-red-700">An error occurred while fetching your prediction history.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link
            href="/"
            className="rounded-lg border border-gray-200 bg-white p-2 hover:bg-gray-50 transition"
          >
            <ArrowLeft className="h-5 w-5 text-gray-600" />
          </Link>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Prediction History</h1>
            <p className="mt-1 text-sm text-gray-500">
              Review your past predictions and explanations.
            </p>
          </div>
        </div>
      </div>

      {predictions.length === 0 ? (
        <div className="rounded-lg border border-gray-200 bg-white p-12 text-center">
          <Activity className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-4 text-lg font-semibold text-gray-900">No predictions yet</h3>
          <p className="mt-2 text-sm text-gray-600">
            When you make predictions, they will appear here.
          </p>
          <Link
            href="/models"
            className="mt-6 inline-flex items-center rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 shadow-sm transition"
          >
            Go to Models
          </Link>
        </div>
      ) : (
        <div className="rounded-lg border border-gray-200 bg-white overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                  Date
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                  Model
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                  Input
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                  Prediction
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                  Probability
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium uppercase tracking-wider text-gray-500">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 bg-white">
              {predictions.map((pred) => (
                <tr key={pred._id}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {format(new Date(pred.created_at), 'MMM d, yyyy HH:mm')}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 font-mono">
                    {pred.model_id.slice(0, 8)}...
                  </td>
                  <td className="px-6 py-4">
                    <div className="max-w-xs truncate text-sm text-gray-600">
                      {JSON.stringify(pred.input_data, null, 2)}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {typeof pred.prediction === 'number' ? pred.prediction.toFixed(4) : String(pred.prediction)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 font-mono">
                    {pred.probability
                      ? `${(pred.probability * 100).toFixed(1)}%`
                      : '-'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right">
                    <Link
                      href={`/explain/local/${pred.model_id}/${pred._id}`}
                      className="inline-flex items-center text-sm text-indigo-600 hover:text-indigo-800"
                    >
                      <Eye className="mr-1 h-4 w-4" />
                      Explain
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
