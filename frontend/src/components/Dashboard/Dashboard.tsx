import { useQuery } from '@tanstack/react-query';
import { api } from '../../services/api';
import { CheckCircle, XCircle, AlertCircle, RefreshCw } from 'lucide-react';

function ProviderCard({
  name,
  available,
}: {
  name: string;
  available: boolean;
}) {
  return (
    <div
      className={`p-4 rounded-lg border ${
        available
          ? 'bg-white border-green-200 shadow-sm'
          : 'bg-gray-50 border-gray-200'
      }`}
    >
      <div className="flex items-center gap-3">
        {available ? (
          <CheckCircle className="h-5 w-5 text-green-600" />
        ) : (
          <XCircle className="h-5 w-5 text-gray-400" />
        )}
        <span
          className={`font-medium capitalize ${
            available ? 'text-gray-900' : 'text-gray-500'
          }`}
        >
          {name}
        </span>
      </div>
    </div>
  );
}

export function Dashboard() {
  const { data: health, isLoading, refetch, error } = useQuery({
    queryKey: ['health'],
    queryFn: api.getHealth,
    refetchInterval: 30000,
  });

  const { data: providers } = useQuery({
    queryKey: ['providers'],
    queryFn: api.getProviders,
    refetchInterval: 30000,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 animate-spin text-blue-600" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center gap-3">
        <AlertCircle className="h-5 w-5 text-red-600" />
        <span className="text-red-700">Failed to connect to router</span>
      </div>
    );
  }

  const allProviders = providers
    ? Object.keys(providers.providers).sort()
    : [];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">Dashboard</h2>
        <button
          onClick={() => refetch()}
          className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
        >
          <RefreshCw className="h-5 w-5 text-gray-600" />
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
          <p className="text-sm text-gray-500">Status</p>
          <p
            className={`text-2xl font-semibold ${
              health?.status === 'healthy' ? 'text-green-600' : 'text-yellow-600'
            }`}
          >
            {health?.status === 'healthy' ? 'Healthy' : 'Degraded'}
          </p>
        </div>
        <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
          <p className="text-sm text-gray-500">Providers Available</p>
          <p className="text-2xl font-semibold text-gray-900">
            {health?.providers_available.length || 0}
          </p>
        </div>
        <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
          <p className="text-sm text-gray-500">Total Providers</p>
          <p className="text-2xl font-semibold text-gray-900">
            {health?.providers_total || 0}
          </p>
        </div>
      </div>

      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Provider Health
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
          {allProviders.map((provider) => (
            <ProviderCard
              key={provider}
              name={provider}
              available={
                health?.providers_available.includes(provider) || false
              }
            />
          ))}
        </div>
      </div>

      {providers?.cache && (
        <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Cache Overview
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-sm text-gray-500">Hits</p>
              <p className="text-xl font-semibold text-gray-900">
                {providers.cache.hits}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Misses</p>
              <p className="text-xl font-semibold text-gray-900">
                {providers.cache.misses}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Size</p>
              <p className="text-xl font-semibold text-gray-900">
                {(providers.cache.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Entries</p>
              <p className="text-xl font-semibold text-gray-900">
                {providers.cache.length}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
