import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../../services/api';
import {
  Plus,
  Trash2,
  Edit2,
  Play,
  CheckCircle,
  AlertCircle,
  Loader2,
  Server,
  Eye,
  EyeOff,
  ToggleLeft,
  ToggleRight,
  X,
} from 'lucide-react';
import type {
  OpenAICompatibleEndpoint,
  CreateOpenAICompatibleRequest,
  TestEndpointResponse,
} from '../../types';

function EndpointCard({
  endpoint,
  onEdit,
  onDelete,
  onTest,
  onToggle,
}: {
  endpoint: OpenAICompatibleEndpoint;
  onEdit: () => void;
  onDelete: () => void;
  onTest: () => void;
  onToggle: () => void;
}) {
  const [showKey, setShowKey] = useState(false);

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <Server className="h-5 w-5 text-gray-500" />
          <div>
            <h4 className="font-medium text-gray-900">{endpoint.name}</h4>
            <p className="text-sm text-gray-500 truncate max-w-xs">
              {endpoint.base_url}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`px-2 py-1 text-xs font-medium rounded ${
              endpoint.enabled
                ? 'bg-green-100 text-green-700'
                : 'bg-gray-100 text-gray-500'
            }`}
          >
            {endpoint.enabled ? 'Enabled' : 'Disabled'}
          </span>
        </div>
      </div>

      <div className="space-y-2 mb-4">
        {endpoint.api_key && (
          <div className="flex items-center gap-2 text-sm">
            <span className="text-gray-500">API Key:</span>
            <span className="font-mono text-gray-700">
              {showKey ? endpoint.api_key : '••••••••'}
            </span>
            <button
              onClick={() => setShowKey(!showKey)}
              className="text-gray-400 hover:text-gray-600"
            >
              {showKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </button>
          </div>
        )}
        {endpoint.models && (
          <div className="text-sm">
            <span className="text-gray-500">Models: </span>
            <span className="text-gray-700">{endpoint.models}</span>
          </div>
        )}
        <div className="flex items-center gap-2 text-sm">
          <span className="text-gray-500">Streaming:</span>
          <span className={endpoint.streaming ? 'text-green-600' : 'text-gray-400'}>
            {endpoint.streaming ? 'Enabled' : 'Disabled'}
          </span>
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        <button
          onClick={onTest}
          className="flex items-center gap-1 px-3 py-1.5 text-sm bg-blue-50 text-blue-700 rounded-lg hover:bg-blue-100"
        >
          <Play className="h-3 w-3" />
          Test
        </button>
        <button
          onClick={onToggle}
          className="flex items-center gap-1 px-3 py-1.5 text-sm bg-gray-50 text-gray-700 rounded-lg hover:bg-gray-100"
        >
          {endpoint.enabled ? (
            <>
              <ToggleRight className="h-4 w-4" />
              Disable
            </>
          ) : (
            <>
              <ToggleLeft className="h-4 w-4" />
              Enable
            </>
          )}
        </button>
        <button
          onClick={onEdit}
          className="flex items-center gap-1 px-3 py-1.5 text-sm bg-gray-50 text-gray-700 rounded-lg hover:bg-gray-100"
        >
          <Edit2 className="h-3 w-3" />
          Edit
        </button>
        <button
          onClick={onDelete}
          className="flex items-center gap-1 px-3 py-1.5 text-sm bg-red-50 text-red-700 rounded-lg hover:bg-red-100"
        >
          <Trash2 className="h-3 w-3" />
          Delete
        </button>
      </div>
    </div>
  );
}

function EndpointForm({
  endpoint,
  onSave,
  onCancel,
}: {
  endpoint?: OpenAICompatibleEndpoint;
  onSave: (data: CreateOpenAICompatibleRequest) => void;
  onCancel: () => void;
}) {
  const [name, setName] = useState(endpoint?.name || '');
  const [baseUrl, setBaseUrl] = useState(endpoint?.base_url || '');
  const [apiKey, setApiKey] = useState(endpoint?.api_key || '');
  const [models, setModels] = useState(endpoint?.models || '');
  const [streaming, setStreaming] = useState(endpoint?.streaming ?? true);
  const [enabled, setEnabled] = useState(endpoint?.enabled ?? true);
  const [showKey, setShowKey] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSave({
      name: name || baseUrl.split('/').slice(0, 3).join('/'),
      base_url: baseUrl,
      api_key: apiKey || null,
      models,
      streaming,
      enabled,
    });
  };

  return (
    <form onSubmit={handleSubmit} className="bg-white rounded-lg border border-gray-200 p-4">
      <h4 className="font-medium text-gray-900 mb-4">
        {endpoint ? 'Edit Endpoint' : 'Add New Endpoint'}
      </h4>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Base URL <span className="text-red-500">*</span>
          </label>
          <input
            type="text"
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
            placeholder="https://api.example.com/v1"
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            required
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Name (optional)
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="My API"
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            API Key (optional)
          </label>
          <div className="relative">
            <input
              type={showKey ? 'text' : 'password'}
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="sk-..."
              className="w-full px-3 py-2 pr-10 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <button
              type="button"
              onClick={() => setShowKey(!showKey)}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
            >
              {showKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </button>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Default Models (optional)
          </label>
          <input
            type="text"
            value={models}
            onChange={(e) => setModels(e.target.value)}
            placeholder="model1,model2"
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <p className="text-xs text-gray-500 mt-1">
            Comma-separated list of models. Leave empty to auto-discover.
          </p>
        </div>

        <div className="flex items-center gap-6">
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={streaming}
              onChange={(e) => setStreaming(e.target.checked)}
              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span className="text-sm text-gray-700">Enable Streaming</span>
          </label>

          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={enabled}
              onChange={(e) => setEnabled(e.target.checked)}
              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span className="text-sm text-gray-700">Enabled</span>
          </label>
        </div>
      </div>

      <div className="flex justify-end gap-2 mt-6">
        <button
          type="button"
          onClick={onCancel}
          className="px-4 py-2 text-sm text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200"
        >
          Cancel
        </button>
        <button
          type="submit"
          className="px-4 py-2 text-sm text-white bg-blue-600 rounded-lg hover:bg-blue-700"
        >
          {endpoint ? 'Save Changes' : 'Add Endpoint'}
        </button>
      </div>
    </form>
  );
}

export function OpenAICompatible() {
  const queryClient = useQueryClient();
  const [editingEndpoint, setEditingEndpoint] = useState<OpenAICompatibleEndpoint | null>(null);
  const [isAdding, setIsAdding] = useState(false);
  const [notification, setNotification] = useState<{
    type: 'success' | 'error';
    message: string;
  } | null>(null);
  const [testResult, setTestResult] = useState<TestEndpointResponse | null>(null);

  const showNotification = (type: 'success' | 'error', message: string) => {
    setNotification({ type, message });
    setTimeout(() => setNotification(null), 4000);
  };

  const { data: endpoints, isLoading, error, refetch } = useQuery({
    queryKey: ['openaiCompatibleEndpoints'],
    queryFn: api.getOpenAICompatibleEndpoints,
  });

  const createMutation = useMutation({
    mutationFn: api.createOpenAICompatibleEndpoint,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['openaiCompatibleEndpoints'] });
      setIsAdding(false);
      showNotification('success', data.message);
    },
    onError: (err: Error) => {
      showNotification('error', err.message);
    },
  });

  const updateMutation = useMutation({
    mutationFn: ({ id, endpoint }: { id: string; endpoint: CreateOpenAICompatibleRequest }) =>
      api.updateOpenAICompatibleEndpoint(id, endpoint),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['openaiCompatibleEndpoints'] });
      setEditingEndpoint(null);
      showNotification('success', data.message);
    },
    onError: (err: Error) => {
      showNotification('error', err.message);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: api.deleteOpenAICompatibleEndpoint,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['openaiCompatibleEndpoints'] });
      showNotification('success', data.message);
    },
    onError: (err: Error) => {
      showNotification('error', err.message);
    },
  });

  const toggleMutation = useMutation({
    mutationFn: async ({ id, endpoint }: { id: string; endpoint: OpenAICompatibleEndpoint }) => {
      return api.updateOpenAICompatibleEndpoint(id, {
        name: endpoint.name,
        base_url: endpoint.base_url,
        api_key: endpoint.api_key,
        models: endpoint.models,
        streaming: endpoint.streaming,
        enabled: !endpoint.enabled,
      });
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['openaiCompatibleEndpoints'] });
      showNotification('success', data.message);
    },
    onError: (err: Error) => {
      showNotification('error', err.message);
    },
  });

  const testMutation = useMutation({
    mutationFn: api.testOpenAICompatibleEndpoint,
    onSuccess: (data) => {
      setTestResult(data);
    },
    onError: (err: Error) => {
      showNotification('error', err.message);
    },
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-32">
        <Loader2 className="h-6 w-6 animate-spin text-blue-600" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center gap-3">
        <AlertCircle className="h-5 w-5 text-red-600" />
        <span className="text-red-700">Failed to load endpoints</span>
        <button
          onClick={() => refetch()}
          className="ml-auto text-sm text-red-600 hover:text-red-800 underline"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">OpenAI Compatible Endpoints</h3>
          <p className="text-sm text-gray-500">
            Configure custom OpenAI-compatible API endpoints. Each endpoint can be used as a provider.
          </p>
        </div>
        <button
          onClick={() => setIsAdding(true)}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          <Plus className="h-4 w-4" />
          Add Endpoint
        </button>
      </div>

      {notification && (
        <div
          className={`flex items-center gap-2 p-4 rounded-lg ${
            notification.type === 'success'
              ? 'bg-green-50 text-green-700 border border-green-200'
              : 'bg-red-50 text-red-700 border border-red-200'
          }`}
        >
          {notification.type === 'success' ? (
            <CheckCircle className="h-5 w-5" />
          ) : (
            <AlertCircle className="h-5 w-5" />
          )}
          {notification.message}
        </div>
      )}

      {testResult && (
        <div
          className={`flex items-start gap-2 p-4 rounded-lg ${
            testResult.success
              ? 'bg-green-50 text-green-700 border border-green-200'
              : 'bg-red-50 text-red-700 border border-red-200'
          }`}
        >
          {testResult.success ? (
            <CheckCircle className="h-5 w-5 mt-0.5" />
          ) : (
            <AlertCircle className="h-5 w-5 mt-0.5" />
          )}
          <div className="flex-1">
            <p className="font-medium">{testResult.message}</p>
            {testResult.models && testResult.models.length > 0 && (
              <p className="text-sm mt-1">
                Models: {testResult.models.join(', ')}
              </p>
            )}
          </div>
          <button onClick={() => setTestResult(null)} className="text-gray-400 hover:text-gray-600">
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      {(isAdding || editingEndpoint) && (
        <EndpointForm
          endpoint={editingEndpoint || undefined}
          onSave={(data) => {
            if (editingEndpoint) {
              updateMutation.mutate({ id: editingEndpoint.id, endpoint: data });
            } else {
              createMutation.mutate(data);
            }
          }}
          onCancel={() => {
            setIsAdding(false);
            setEditingEndpoint(null);
          }}
        />
      )}

      {endpoints && endpoints.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {endpoints.map((endpoint) => (
            <EndpointCard
              key={endpoint.id}
              endpoint={endpoint}
              onEdit={() => setEditingEndpoint(endpoint)}
              onDelete={() => {
                if (confirm(`Delete endpoint "${endpoint.name}"?`)) {
                  deleteMutation.mutate(endpoint.id);
                }
              }}
              onTest={() => testMutation.mutate(endpoint.id)}
              onToggle={() => toggleMutation.mutate({ id: endpoint.id, endpoint })}
            />
          ))}
        </div>
      ) : (
        <div className="text-center py-12 bg-gray-50 rounded-lg border border-gray-200">
          <Server className="h-12 w-12 text-gray-400 mx-auto mb-3" />
          <p className="text-gray-500">No OpenAI-compatible endpoints configured</p>
          <button
            onClick={() => setIsAdding(true)}
            className="mt-2 text-blue-600 hover:text-blue-700"
          >
            Add your first endpoint
          </button>
        </div>
      )}

      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
        <p className="text-sm text-yellow-700">
          <strong>Note:</strong> After adding, updating, or removing endpoints, restart the router for changes to take effect.
        </p>
      </div>
    </div>
  );
}
