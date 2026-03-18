'use client';

import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { format } from 'date-fns';

interface AuditLog {
  _id: string;
  user_id: string;
  action: string;
  resource_type: string;
  resource_id?: string;
  details?: Record<string, any>;
  ip_address?: string;
  user_agent?: string;
  created_at: string;
}

const actionColors: Record<string, string> = {
  'model.upload': 'bg-blue-100 text-blue-800',
  'model.delete': 'bg-red-100 text-red-800',
  'prediction.create': 'bg-green-100 text-green-800',
  'explanation.create': 'bg-purple-100 text-purple-800',
  'bias.analyze': 'bg-yellow-100 text-yellow-800',
  'api_key.create': 'bg-indigo-100 text-indigo-800',
  'api_key.delete': 'bg-red-100 text-red-800',
  'user.login': 'bg-gray-100 text-gray-800',
  'user.register': 'bg-teal-100 text-teal-800',
};

export default function AuditPage() {
  const [filters, setFilters] = useState({
    action: '',
    startDate: '',
    endDate: ''
  });

  const { data: logs = [], isLoading, refetch } = useQuery<AuditLog[]>({
    queryKey: ['audit-logs', filters],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (filters.action) params.append('action', filters.action);
      if (filters.startDate) params.append('start_date', filters.startDate);
      if (filters.endDate) params.append('end_date', filters.endDate);
      params.append('limit', '100');

      const token = localStorage.getItem('token');
      const response = await fetch(`/api/v1/audit/my?${params.toString()}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (!response.ok) throw new Error('Failed to fetch audit logs');
      return response.json();
    }
  });

  const getActionBadgeClass = (action: string) => {
    return actionColors[action] || 'bg-gray-100 text-gray-800';
  };

  const getActionDisplayName = (action: string) => {
    return action.replace(/\./g, ' → ');
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">Audit Log</h1>

        {/* Filters */}
        <div className="bg-white p-6 rounded-lg shadow-sm mb-6">
          <h2 className="text-lg font-semibold mb-4">Filters</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Action
              </label>
              <select
                value={filters.action}
                onChange={(e) => setFilters({ ...filters, action: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">All Actions</option>
                <option value="model.upload">Model Upload</option>
                <option value="model.delete">Model Delete</option>
                <option value="prediction.create">Prediction</option>
                <option value="explanation.create">Explanation</option>
                <option value="bias.analyze">Bias Analysis</option>
                <option value="api_key.create">API Key Created</option>
                <option value="api_key.delete">API Key Deleted</option>
                <option value="user.login">User Login</option>
                <option value="user.register">User Register</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Start Date
              </label>
              <input
                type="datetime-local"
                value={filters.startDate}
                onChange={(e) => setFilters({ ...filters, startDate: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                End Date
              </label>
              <input
                type="datetime-local"
                value={filters.endDate}
                onChange={(e) => setFilters({ ...filters, endDate: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>

          <div className="mt-4 flex gap-2">
            <button
              onClick={() => refetch()}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
            >
              Apply Filters
            </button>
            <button
              onClick={() => setFilters({ action: '', startDate: '', endDate: '' })}
              className="px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300"
            >
              Clear
            </button>
          </div>
        </div>

        {/* Audit Logs Table */}
        <div className="bg-white rounded-lg shadow-sm overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-semibold">Activity Log</h2>
            <p className="text-sm text-gray-500 mt-1">
              Showing {logs.length} entries
            </p>
          </div>

          {isLoading ? (
            <div className="p-8 text-center text-gray-500">
              Loading audit logs...
            </div>
          ) : logs.length === 0 ? (
            <div className="p-8 text-center text-gray-500">
              No audit logs found
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Timestamp
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Action
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Resource
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Details
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      IP Address
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {logs.map((log) => (
                    <tr key={log._id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {format(new Date(log.created_at), 'PPpp')}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${getActionBadgeClass(log.action)}`}>
                          {getActionDisplayName(log.action)}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        <span className="font-medium">{log.resource_type}</span>
                        {log.resource_id && (
                          <span className="text-gray-500 ml-2">
                            ({log.resource_id.slice(-6)})
                          </span>
                        )}
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-500 max-w-xs truncate">
                        {log.details && Object.keys(log.details).length > 0 ? (
                          <pre className="text-xs">
                            {JSON.stringify(log.details, null, 2)}
                          </pre>
                        ) : (
                          <span className="text-gray-400">No details</span>
                        )}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {log.ip_address || '-'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
