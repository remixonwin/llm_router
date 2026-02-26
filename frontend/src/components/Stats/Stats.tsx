import { useQuery } from '@tanstack/react-query';
import { api } from '../../services/api';
import { RefreshCw, TrendingUp, Database, Server } from 'lucide-react';
import type { ProviderStats } from '../../types';

function UsageBar({
  used,
  limit,
  label,
}: {
  used: number;
  limit: number;
  label: string;
}) {
  const percentage = limit > 0 ? (used / limit) * 100 : 0;
  const color =
    percentage > 90
      ? 'bg-red-500'
      : percentage > 70
        ? 'bg-yellow-500'
        : 'bg-blue-500';

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm">
        <span className="text-gray-600">{label}</span>
        <span className="text-gray-900 font-medium">
          {used.toLocaleString()} / {limit.toLocaleString()}
        </span>
      </div>
      <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
        <div
          className={`h-full ${color} transition-all`}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
      </div>
    </div>
  );
}

function ProviderRow({
  name,
  stats,
}: {
  name: string;
  stats: ProviderStats;
}) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold text-gray-900 capitalize">{name}</h3>
        <span
          className={`px-2 py-1 text-xs font-medium rounded ${
            stats.available
              ? 'bg-green-100 text-green-700'
              : 'bg-red-100 text-red-700'
          }`}
        >
          {stats.available ? 'Available' : 'Unavailable'}
        </span>
      </div>
      <div className="space-y-3">
        <UsageBar used={stats.rpm_used} limit={stats.rpm_limit} label="RPM" />
        <UsageBar used={stats.rpd_used} limit={stats.rpd_limit} label="RPD" />
      </div>
      {stats.error_count > 0 && (
        <p className="mt-2 text-sm text-red-600">
          Errors: {stats.error_count}
        </p>
      )}
    </div>
  );
}

export function Stats() {
  const { data, isLoading, refetch } = useQuery({
    queryKey: ['stats'],
    queryFn: api.getStats,
    refetchInterval: 30000,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 animate-spin text-blue-600" />
      </div>
    );
  }

  const providers = data?.quota ? Object.entries(data.quota) : [];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">Statistics</h2>
        <button
          onClick={() => refetch()}
          className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
        >
          <RefreshCw className="h-5 w-5 text-gray-600" />
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
          <div className="flex items-center gap-3">
            <Server className="h-8 w-8 text-blue-600" />
            <div>
              <p className="text-sm text-gray-500">Total Models</p>
              <p className="text-2xl font-semibold text-gray-900">
                {data?.total_models || 0}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
          <div className="flex items-center gap-3">
            <Database className="h-8 w-8 text-green-600" />
            <div>
              <p className="text-sm text-gray-500">Cache Hits</p>
              <p className="text-2xl font-semibold text-gray-900">
                {data?.cache.hits || 0}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
          <div className="flex items-center gap-3">
            <TrendingUp className="h-8 w-8 text-purple-600" />
            <div>
              <p className="text-sm text-gray-500">Cache Misses</p>
              <p className="text-2xl font-semibold text-gray-900">
                {data?.cache.misses || 0}
              </p>
            </div>
          </div>
        </div>
      </div>

      {data?.models && Object.keys(data.models).length > 0 && (
        <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Models by Provider
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            {Object.entries(data.models).map(([provider, count]) => (
              <div key={provider} className="text-center">
                <p className="text-2xl font-bold text-gray-900">{count}</p>
                <p className="text-sm text-gray-500 capitalize">{provider}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Provider Quotas
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {providers.map(([name, stats]) => (
            <ProviderRow key={name} name={name} stats={stats} />
          ))}
        </div>
      </div>
    </div>
  );
}
