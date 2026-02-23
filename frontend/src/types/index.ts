export interface HealthStatus {
  status: 'healthy' | 'degraded';
  providers_available: string[];
  providers_total: number;
}

export interface ProviderStats {
  available: boolean;
  rpm_limit: number;
  rpm_used: number;
  rpd_limit: number;
  rpd_used: number;
  error_count: number;
  consecutive_errors: number;
  circuit_open: boolean;
  cooldown_until: string | null;
}

export interface ProvidersResponse {
  providers: Record<string, ProviderStats>;
  cache: CacheStats;
}

export interface CacheStats {
  hits: number;
  misses: number;
  size: number;
  length: number;
}

export interface Model {
  id: string;
  object: string;
  owned_by: string;
  capabilities: string[];
  context_window: number;
  is_free: boolean;
}

export interface ModelsResponse {
  object: string;
  data: Model[];
}

export interface ProviderModelsResponse {
  provider: string;
  count: number;
  models: {
    id: string;
    capabilities: string[];
    context_window: number;
  }[];
}

export interface StatsResponse {
  quota: Record<string, ProviderStats>;
  cache: CacheStats;
  models: Record<string, number>;
  total_models: number;
}

export interface AdminResponse {
  status: 'ok' | 'error';
  message: string;
}
