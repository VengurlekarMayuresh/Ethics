'use client';
import { useStore } from '@/lib/store';
import { Box, Activity, Clock, Plus } from 'lucide-react';
import Link from 'next/link';

export default function Dashboard() {
  const user = useStore((state) => state.user);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Welcome back, {user?.name || 'User'}</h1>
          <p className="mt-1 text-sm text-gray-500">Here's an overview of your Explainable AI workspace.</p>
        </div>
        <Link 
          href="/models/upload" 
          className="flex items-center rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 transition"
        >
          <Plus className="mr-2 h-4 w-4" />
          New Model
        </Link>
      </div>

      <div className="grid grid-cols-1 gap-5 sm:grid-cols-3">
        <div className="overflow-hidden rounded-xl bg-white p-6 shadow-sm border border-gray-100">
          <div className="flex items-center">
            <div className="flex flex-shrink-0 rounded-lg bg-indigo-50 p-3">
              <Box className="h-6 w-6 text-indigo-600" />
            </div>
            <div className="ml-5 w-0 flex-1">
              <dl>
                <dt className="truncate text-sm font-medium text-gray-500">Active Models</dt>
                <dd className="mt-1 text-2xl font-semibold tracking-tight text-gray-900">0</dd>
              </dl>
            </div>
          </div>
        </div>

        <div className="overflow-hidden rounded-xl bg-white p-6 shadow-sm border border-gray-100">
          <div className="flex items-center">
            <div className="flex flex-shrink-0 rounded-lg bg-green-50 p-3">
              <Activity className="h-6 w-6 text-green-600" />
            </div>
            <div className="ml-5 w-0 flex-1">
              <dl>
                <dt className="truncate text-sm font-medium text-gray-500">Total Predictions</dt>
                <dd className="mt-1 text-2xl font-semibold tracking-tight text-gray-900">0</dd>
              </dl>
            </div>
          </div>
        </div>

        <div className="overflow-hidden rounded-xl bg-white p-6 shadow-sm border border-gray-100">
          <div className="flex items-center">
            <div className="flex flex-shrink-0 rounded-lg bg-blue-50 p-3">
              <Clock className="h-6 w-6 text-blue-600" />
            </div>
            <div className="ml-5 w-0 flex-1">
              <dl>
                <dt className="truncate text-sm font-medium text-gray-500">Recent Explanations</dt>
                <dd className="mt-1 text-2xl font-semibold tracking-tight text-gray-900">0</dd>
              </dl>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-8">
        <h2 className="text-lg font-medium text-gray-900">Recent Activity</h2>
        <div className="mt-4 rounded-xl border border-gray-200 bg-white shadow-sm">
          <div className="p-10 text-center">
            <Activity className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">No activity yet</h3>
            <p className="mt-1 text-sm text-gray-500">Upload your first model to get started with explainable AI.</p>
            <div className="mt-6">
              <Link
                href="/models/upload"
                className="inline-flex items-center rounded-md bg-indigo-600 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
              >
                <Plus className="-ml-0.5 mr-1.5 h-5 w-5" aria-hidden="true" />
                Upload Model
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
