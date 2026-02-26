import type {
  HealthStatus,
  ProvidersResponse,
  ModelsResponse,
  ProviderModelsResponse,
  StatsResponse,
  AdminResponse,
} from '../types';

const API_BASE = import.meta.env.VITE_API_BASE || '';

async function fetchApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

export const api = {
  getHealth: () => fetchApi<HealthStatus>('/health'),

  getProviders: () => fetchApi<ProvidersResponse>('/providers'),

  getProvider: (provider: string) =>
    fetchApi<ProvidersResponse['providers']>(`/providers/${provider}`),

  getModels: () => fetchApi<ModelsResponse>('/v1/models'),

  getProviderModels: (provider: string) =>
    fetchApi<ProviderModelsResponse>(`/v1/models/${provider}`),

  getStats: () => fetchApi<StatsResponse>('/stats'),

  clearCache: () =>
    fetchApi<AdminResponse>('/admin/cache/clear', { method: 'POST' }),

  resetQuotas: () =>
    fetchApi<AdminResponse>('/admin/quotas/reset', { method: 'POST' }),

  refreshModels: () =>
    fetchApi<AdminResponse>('/admin/refresh', { method: 'POST' }),
};
