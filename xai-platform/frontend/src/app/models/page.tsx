'use client';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { Box, Plus, Trash2, Tag, Calendar, Database } from 'lucide-react';
import Link from 'next/link';
import { format } from 'date-fns';

interface Model {
  _id: string;
  name: string;
  description: string;
  framework: string;
  task_type: string;
  created_at: string;
}

export default function ModelsPage() {
  const { data: models, isLoading, refetch } = useQuery<Model[]>({
    queryKey: ['models'],
    queryFn: async () => {
      const { data } = await api.get('/models/');
      return data;
    }
  });

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.preventDefault();
    if (confirm('Are you sure you want to delete this model?')) {
      await api.delete(`/models/${id}`);
      refetch();
    }
  };

  const frameworkColors: Record<string, string> = {
    sklearn: 'bg-blue-100 text-blue-800',
    xgboost: 'bg-orange-100 text-orange-800',
    keras: 'bg-red-100 text-red-800',
    onnx: 'bg-purple-100 text-purple-800',
  };

  return (
    <div className="space-y-6">
      <div className="sm:flex sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Your Models</h1>
          <p className="mt-1 text-sm text-gray-500">Manage uploaded machine learning models and connect APIs.</p>
        </div>
        <div className="mt-4 sm:ml-16 sm:mt-0 sm:flex-none">
          <Link
            href="/models/upload"
            className="flex items-center rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 transition"
          >
            <Plus className="mr-2 h-4 w-4" />
            Upload New Model
          </Link>
        </div>
      </div>

      {isLoading ? (
        <div className="flex h-64 items-center justify-center">
          <div className="h-8 w-8 animate-spin rounded-full border-b-2 border-indigo-600"></div>
        </div>
      ) : models?.length === 0 ? (
        <div className="rounded-xl border border-gray-200 bg-white p-12 text-center shadow-sm">
          <Box className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">No models</h3>
          <p className="mt-1 text-sm text-gray-500">Get started by uploading a trained model file.</p>
          <div className="mt-6">
            <Link
              href="/models/upload"
              className="inline-flex items-center rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 shadow-sm transition"
            >
              <Plus className="-ml-0.5 mr-1.5 h-5 w-5" />
              Upload Model
            </Link>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {models?.map((model) => (
            <Link key={model._id} href={`/models/${model._id}`} className="group flex flex-col justify-between overflow-hidden rounded-xl border border-gray-200 bg-white shadow-sm transition hover:shadow-md hover:border-indigo-300">
              <div className="p-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-gray-900 group-hover:text-indigo-600 transition-colors line-clamp-1">{model.name}</h3>
                  <span className={`inline-flex items-center rounded-full px-2 py-1 text-xs font-medium ${frameworkColors[model.framework] || 'bg-gray-100 text-gray-800'}`}>
                    {model.framework}
                  </span>
                </div>
                <p className="mt-2 text-sm text-gray-500 line-clamp-2">
                  {model.description || 'No description provided.'}
                </p>
                <div className="mt-4 flex flex-col space-y-2">
                  <div className="flex items-center text-sm text-gray-500">
                    <Tag className="mr-1.5 h-4 w-4 flex-shrink-0 text-gray-400" />
                    <span className="capitalize">{model.task_type}</span>
                  </div>
                  <div className="flex items-center text-sm text-gray-500">
                    <Calendar className="mr-1.5 h-4 w-4 flex-shrink-0 text-gray-400" />
                    {format(new Date(model.created_at), 'MMM d, yyyy')}
                  </div>
                </div>
              </div>
              <div className="bg-gray-50 px-6 py-3 border-t border-gray-100 flex justify-between items-center text-sm">
                <span className="text-indigo-600 font-medium">View details &rarr;</span>
                <button 
                  onClick={(e) => handleDelete(model._id, e)}
                  title="Delete Model"
                  className="text-gray-400 hover:text-red-500 transition-colors"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
