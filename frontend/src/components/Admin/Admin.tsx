import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../../services/api';
import {
  Trash2,
  RotateCcw,
  RefreshCw,
  CheckCircle,
  AlertCircle,
} from 'lucide-react';
import { ApiKeys } from './ApiKeys';

interface ActionButtonProps {
  onClick: () => void;
  loading: boolean;
  icon: React.ReactNode;
  label: string;
  description: string;
  variant?: 'danger' | 'warning' | 'info';
}

function ActionButton({
  onClick,
  loading,
  icon,
  label,
  description,
  variant = 'info',
}: ActionButtonProps) {
  const colors = {
    danger: 'border-red-200 hover:bg-red-50 text-red-700',
    warning: 'border-yellow-200 hover:bg-yellow-50 text-yellow-700',
    info: 'border-blue-200 hover:bg-blue-50 text-blue-700',
  };

  return (
    <button
      onClick={onClick}
      disabled={loading}
      className={`flex items-start gap-4 p-4 rounded-lg border ${colors[variant]} transition-colors w-full text-left disabled:opacity-50`}
    >
      <div className="flex-shrink-0">{icon}</div>
      <div>
        <p className="font-medium">{label}</p>
        <p className="text-sm opacity-75">{description}</p>
      </div>
      {loading && (
        <RefreshCw className="h-5 w-5 animate-spin ml-auto flex-shrink-0" />
      )}
    </button>
  );
}

export function Admin() {
  const queryClient = useQueryClient();
  const [notification, setNotification] = useState<{
    type: 'success' | 'error';
    message: string;
  } | null>(null);

  const showNotification = (type: 'success' | 'error', message: string) => {
    setNotification({ type, message });
    setTimeout(() => setNotification(null), 3000);
  };

  const clearCacheMutation = useMutation({
    mutationFn: api.clearCache,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['providers'] });
      queryClient.invalidateQueries({ queryKey: ['stats'] });
      showNotification('success', 'Cache cleared successfully');
    },
    onError: (error: Error) => {
      showNotification('error', error.message);
    },
  });

  const resetQuotasMutation = useMutation({
    mutationFn: api.resetQuotas,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['providers'] });
      queryClient.invalidateQueries({ queryKey: ['stats'] });
      showNotification('success', 'Quotas reset successfully');
    },
    onError: (error: Error) => {
      showNotification('error', error.message);
    },
  });

  const refreshModelsMutation = useMutation({
    mutationFn: api.refreshModels,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] });
      showNotification('success', 'Models refreshed successfully');
    },
    onError: (error: Error) => {
      showNotification('error', error.message);
    },
  });

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-900">Admin</h2>

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

      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Cache Management
        </h3>
        <p className="text-gray-600 mb-4">
          Clear the response cache to force fresh requests to providers.
        </p>
        <ActionButton
          onClick={() => clearCacheMutation.mutate()}
          loading={clearCacheMutation.isPending}
          icon={<Trash2 className="h-5 w-5" />}
          label="Clear Cache"
          description="Remove all cached responses"
          variant="danger"
        />
      </div>

      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Quota Management
        </h3>
        <p className="text-gray-600 mb-4">
          Reset all provider quota counters. This is useful for testing or
          recovering from rate limits.
        </p>
        <ActionButton
          onClick={() => resetQuotasMutation.mutate()}
          loading={resetQuotasMutation.isPending}
          icon={<RotateCcw className="h-5 w-5" />}
          label="Reset Quotas"
          description="Reset RPM/RPD counters for all providers"
          variant="warning"
        />
      </div>

      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Model Discovery
        </h3>
        <p className="text-gray-600 mb-4">
          Force a refresh of model capabilities for all providers.
        </p>
        <ActionButton
          onClick={() => refreshModelsMutation.mutate()}
          loading={refreshModelsMutation.isPending}
          icon={<RefreshCw className="h-5 w-5" />}
          label="Refresh Models"
          description="Update model capabilities from all providers"
          variant="info"
        />
      </div>

      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <ApiKeys />
      </div>
    </div>
  );
}
