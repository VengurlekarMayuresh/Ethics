'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { useState } from 'react';
import {
  Key,
  Plus,
  Trash2,
  Copy,
  Check,
  Loader2,
  AlertCircle,
  Eye,
  EyeOff,
  Clock,
  Shield,
} from 'lucide-react';
import { format } from 'date-fns';

interface APIKey {
  id: string;
  name: string;
  scopes: string[];
  created_at: string;
  expires_at?: string;
  last_used_at?: string;
}

interface NewKeyForm {
  name: string;
  scopes: string[];
  expires_in_days?: number;
}

const defaultScopes = [
  { id: 'read', label: 'Read access' },
  { id: 'predict', label: 'Make predictions' },
  { id: 'explain', label: 'Get explanations' },
];

export default function APIKeysPage() {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newKey, setNewKey] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [showKey, setShowKey] = useState(false);

  const queryClient = useQueryClient();

  // Fetch API keys
  const { data: apiKeys, isLoading, refetch } = useQuery<APIKey[]>({
    queryKey: ['apiKeys'],
    queryFn: async () => {
      const { data } = await api.get('/api-keys/');
      return data;
    },
  });

  // Create key mutation
  const createMutation = useMutation({
    mutationFn: async (keyData: NewKeyForm) => {
      const { data } = await api.post('/api-keys/', keyData);
      return data;
    },
    onSuccess: (data) => {
      setNewKey(data.key);
      refetch();
    },
  });

  // Delete key mutation
  const deleteMutation = useMutation({
    mutationFn: async (keyId: string) => {
      await api.delete(`/api-keys/${keyId}`);
    },
    onSuccess: () => {
      refetch();
    },
  });

  const handleCreate = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const name = formData.get('name') as string;
    const scopes = formData.getAll('scopes') as string[];
    const expires_in_days = formData.get('expires_in_days') as string;

    createMutation.mutate({
      name,
      scopes,
      expires_in_days: expires_in_days ? parseInt(expires_in_days, 10) : undefined,
    });
  };

  const handleCopy = () => {
    if (newKey) {
      navigator.clipboard.writeText(newKey);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleCloseCreate = () => {
    setShowCreateModal(false);
    setNewKey(null);
  };

  const getScopeLabel = (scopeId: string) => {
    const scope = defaultScopes.find(s => s.id === scopeId);
    return scope ? scope.label : scopeId;
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">API Keys</h1>
          <p className="mt-1 text-sm text-gray-500">
            Manage API keys for external integrations and automation.
          </p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          disabled={createMutation.isPending}
          className="inline-flex items-center rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 transition"
        >
          <Plus className="mr-2 h-4 w-4" />
          Create API Key
        </button>
      </div>

      {isLoading ? (
        <div className="flex h-64 items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-indigo-600" />
        </div>
      ) : apiKeys && apiKeys.length > 0 ? (
        <div className="rounded-lg border border-gray-200 bg-white overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                  Name
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                  Scopes
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                  Created
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                  Last Used
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium uppercase tracking-wider text-gray-500">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 bg-white">
              {apiKeys.map((key) => (
                <tr key={key.id}>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <Key className="mr-3 h-5 w-5 text-gray-400" />
                      <span className="text-sm font-medium text-gray-900">{key.name}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex flex-wrap gap-1">
                      {key.scopes.map(scope => (
                        <span
                          key={scope}
                          className="inline-flex items-center rounded-full bg-blue-50 px-2 py-1 text-xs font-medium text-blue-700"
                        >
                          <Shield className="mr-1 h-3 w-3" />
                          {getScopeLabel(scope)}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {format(new Date(key.created_at), 'MMM d, yyyy HH:mm')}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {key.last_used_at ? (
                      format(new Date(key.last_used_at), 'MMM d, yyyy HH:mm')
                    ) : (
                      <span className="text-gray-400">Never</span>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right">
                    <button
                      onClick={() => deleteMutation.mutate(key.id)}
                      disabled={deleteMutation.isPending}
                      className="text-gray-400 hover:text-red-500 transition-colors disabled:opacity-50"
                      title="Revoke API key"
                    >
                      <Trash2 className="h-5 w-5" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="rounded-lg border border-gray-200 bg-white p-12 text-center">
          <Key className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-4 text-lg font-semibold text-gray-900">No API keys</h3>
          <p className="mt-2 text-sm text-gray-600">
            Create an API key to allow external tools and scripts to access the platform programmatically.
          </p>
        </div>
      )}

      {/* Create API Key Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-lg w-full p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Create API Key</h2>

            {newKey ? (
              <div className="space-y-4">
                <div className="rounded-lg bg-yellow-50 p-4 border border-yellow-200">
                  <div className="flex items-start gap-3">
                    <AlertCircle className="h-5 w-5 text-yellow-600 mt-0.5" />
                    <div>
                      <h4 className="text-sm font-medium text-yellow-800">Important: Make a note of your API key</h4>
                      <p className="text-xs text-yellow-700 mt-1">
                        This is the only time your API key will be shown. Save it securely; you won't be able to see it again.
                      </p>
                    </div>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">API Key</label>
                  <div className="flex items-center gap-2">
                    <input
                      type={showKey ? 'text' : 'password'}
                      value={newKey}
                      readOnly
                      className="flex-1 rounded-md border border-gray-300 bg-gray-50 px-3 py-2 text-sm font-mono text-gray-700"
                    />
                    <button
                      type="button"
                      onClick={() => setShowKey(!showKey)}
                      className="p-2 text-gray-400 hover:text-gray-600"
                      title={showKey ? 'Hide' : 'Show'}
                    >
                      {showKey ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                    </button>
                    <button
                      type="button"
                      onClick={handleCopy}
                      className="p-2 text-gray-400 hover:text-gray-600"
                      title="Copy to clipboard"
                    >
                      {copied ? <Check className="h-5 w-5 text-green-600" /> : <Copy className="h-5 w-5" />}
                    </button>
                  </div>
                </div>

                <div className="flex justify-end">
                  <button
                    type="button"
                    onClick={handleCloseCreate}
                    className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 transition"
                  >
                    Done
                  </button>
                </div>
              </div>
            ) : (
              <form onSubmit={handleCreate} className="space-y-4">
                <div>
                  <label htmlFor="name" className="block text-sm font-medium text-gray-700">
                    Key Name
                  </label>
                  <input
                    type="text"
                    name="name"
                    id="name"
                    required
                    placeholder="e.g., CI/CD Pipeline, Data Sync"
                    className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 sm:text-sm"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Scopes (Permissions)
                  </label>
                  <div className="space-y-2">
                    {defaultScopes.map(scope => (
                      <label key={scope.id} className="flex items-center">
                        <input
                          type="checkbox"
                          name="scopes"
                          value={scope.id}
                          defaultChecked
                          className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                        />
                        <span className="ml-2 text-sm text-gray-700">{scope.label}</span>
                      </label>
                    ))}
                  </div>
                </div>

                <div>
                  <label htmlFor="expires_in_days" className="block text-sm font-medium text-gray-700">
                    Expires in (days, optional)
                  </label>
                  <input
                    type="number"
                    name="expires_in_days"
                    id="expires_in_days"
                    min="1"
                    placeholder="Leave empty for no expiration"
                    className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 sm:text-sm"
                  />
                </div>

                {createMutation.isError && (
                  <div className="rounded-md bg-red-50 p-3 border border-red-200">
                    <p className="text-sm text-red-800">
                      {(createMutation.error as Error)?.message || 'Failed to create API key'}
                    </p>
                  </div>
                )}

                <div className="flex justify-end gap-3 pt-4 border-t border-gray-100">
                  <button
                    type="button"
                    onClick={() => setShowCreateModal(false)}
                    className="rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 transition"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={createMutation.isPending}
                    className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 transition"
                  >
                    {createMutation.isPending ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin inline" />
                        Creating...
                      </>
                    ) : (
                      'Create Key'
                    )}
                  </button>
                </div>
              </form>
            )}
          </div>
        </div>
      )}

      {/* Info Banner */}
      <div className="rounded-lg border border-blue-200 bg-blue-50 p-4">
        <div className="flex items-start gap-3">
          <Shield className="h-5 w-5 text-blue-600 mt-0.5" />
          <div>
            <h4 className="text-sm font-medium text-blue-800">About API Keys</h4>
            <p className="mt-1 text-sm text-blue-700">
              API keys allow external tools, scripts, or CI/CD pipelines to access the XAI Platform API.
              Include the key in requests as an <code className="bg-blue-100 px-1 rounded text-xs">Authorization: Bearer &lt;your-key&gt;</code> header.
              Keep your keys secure and rotate them regularly.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
