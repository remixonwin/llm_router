import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../../services/api';
import { Search, Filter, RefreshCw } from 'lucide-react';
import { Model } from '../../types';

function CapabilityBadge({ capability }: { capability: string }) {
  const colors: Record<string, string> = {
    chat: 'bg-blue-100 text-blue-700',
    text: 'bg-gray-100 text-gray-700',
    vision: 'bg-purple-100 text-purple-700',
    embedding: 'bg-green-100 text-green-700',
    function_calling: 'bg-yellow-100 text-yellow-700',
  };
  
  const color = colors[capability] || 'bg-gray-100 text-gray-700';
  
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${color}`}>
      {capability.replace('_', ' ')}
    </span>
  );
}

export function Models() {
  const [search, setSearch] = useState('');
  const [providerFilter, setProviderFilter] = useState<string>('');
  const [capabilityFilter, setCapabilityFilter] = useState<string>('');

  const { data, isLoading, refetch } = useQuery({
    queryKey: ['models'],
    queryFn: api.getModels,
    refetchInterval: 60000,
  });

  const providers = useMemo(() => {
    if (!data) return [];
    const provs = new Set(data.data.map((m) => m.owned_by));
    return Array.from(provs).sort();
  }, [data]);

  const capabilities = useMemo(() => {
    if (!data) return [];
    const caps = new Set<string>();
    data.data.forEach((m) => m.capabilities.forEach((c) => caps.add(c)));
    return Array.from(caps).sort();
  }, [data]);

  const filteredModels = useMemo(() => {
    if (!data) return [];
    
    return data.data.filter((model: Model) => {
      const matchesSearch =
        !search ||
        model.id.toLowerCase().includes(search.toLowerCase());
      const matchesProvider =
        !providerFilter || model.owned_by === providerFilter;
      const matchesCapability =
        !capabilityFilter || model.capabilities.includes(capabilityFilter);
      return matchesSearch && matchesProvider && matchesCapability;
    });
  }, [data, search, providerFilter, capabilityFilter]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 animate-spin text-blue-600" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">Models</h2>
        <button
          onClick={() => refetch()}
          className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
        >
          <RefreshCw className="h-5 w-5 text-gray-600" />
        </button>
      </div>

      <div className="flex flex-wrap gap-4">
        <div className="relative flex-1 min-w-[200px]">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search models..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>

        <div className="flex items-center gap-2">
          <Filter className="h-5 w-5 text-gray-400" />
          <select
            value={providerFilter}
            onChange={(e) => setProviderFilter(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="">All Providers</option>
            {providers.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>

          <select
            value={capabilityFilter}
            onChange={(e) => setCapabilityFilter(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="">All Capabilities</option>
            {capabilities.map((c) => (
              <option key={c} value={c}>
                {c.replace('_', ' ')}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Model
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Provider
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Context
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Capabilities
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Free
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {filteredModels.map((model: Model) => (
                <tr key={model.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3">
                    <span className="font-mono text-sm text-gray-900">
                      {model.id}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <span className="capitalize text-sm text-gray-600">
                      {model.owned_by}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-600">
                    {model.context_window?.toLocaleString() || '-'}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex flex-wrap gap-1">
                      {model.capabilities.map((cap) => (
                        <CapabilityBadge key={cap} capability={cap} />
                      ))}
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    {model.is_free ? (
                      <span className="px-2 py-1 text-xs font-medium bg-green-100 text-green-700 rounded">
                        Free
                      </span>
                    ) : (
                      <span className="text-gray-400">-</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <p className="text-sm text-gray-500">
        Showing {filteredModels.length} of {data?.data.length || 0} models
      </p>
    </div>
  );
}
