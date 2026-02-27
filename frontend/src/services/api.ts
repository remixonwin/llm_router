import type {
  HealthStatus,
  ProvidersResponse,
  ModelsResponse,
  ProviderModelsResponse,
  StatsResponse,
  AdminResponse,
  ApiKeyStatus,
  SetApiKeyRequest,
  OpenAICompatibleEndpoint,
  CreateOpenAICompatibleRequest,
  TestEndpointResponse,
} from "../types";

// API_BASE is empty because the LLM Router backend serves API at root paths
// (e.g., /health, /providers, /v1/models) alongside the frontend static files
const API_BASE = "";

const ROUTER_API_KEY = "remixonwin";

function getAuthHeaders() {
  return {
    Authorization: `Bearer ${ROUTER_API_KEY}`,
  };
}

async function fetchApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...getAuthHeaders(),
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

export const api = {
  getHealth: () => fetchApi<HealthStatus>("/health"),

  getProviders: () => fetchApi<ProvidersResponse>("/providers"),

  getProvider: (provider: string) =>
    fetchApi<ProvidersResponse["providers"]>(`/providers/${provider}`),

  getModels: () => fetchApi<ModelsResponse>("/v1/models"),

  getProviderModels: (provider: string) =>
    fetchApi<ProviderModelsResponse>(`/v1/models/${provider}`),

  getStats: () => fetchApi<StatsResponse>("/stats"),

  clearCache: () => fetchApi<AdminResponse>("/admin/cache/clear", { method: "POST" }),

  resetQuotas: () => fetchApi<AdminResponse>("/admin/quotas/reset", { method: "POST" }),

  refreshModels: () => fetchApi<AdminResponse>("/admin/refresh", { method: "POST" }),

  getApiKeys: () => fetchApi<ApiKeyStatus[]>("/admin/api-keys"),

  setApiKey: (provider: string, apiKey: string) =>
    fetchApi<AdminResponse>(`/admin/api-keys/${provider}`, {
      method: "POST",
      body: JSON.stringify({ api_key: apiKey } as SetApiKeyRequest),
    }),

  deleteApiKey: (provider: string) =>
    fetchApi<AdminResponse>(`/admin/api-keys/${provider}`, {
      method: "DELETE",
    }),

  getOpenAICompatibleEndpoints: () =>
    fetchApi<OpenAICompatibleEndpoint[]>("/admin/api-keys/openai-compatible"),

  createOpenAICompatibleEndpoint: (endpoint: CreateOpenAICompatibleRequest) =>
    fetchApi<AdminResponse & { id?: string }>("/admin/api-keys/openai-compatible", {
      method: "POST",
      body: JSON.stringify(endpoint),
    }),

  updateOpenAICompatibleEndpoint: (id: string, endpoint: CreateOpenAICompatibleRequest) =>
    fetchApi<AdminResponse>(`/admin/api-keys/openai-compatible/${id}`, {
      method: "PUT",
      body: JSON.stringify(endpoint),
    }),

  deleteOpenAICompatibleEndpoint: (id: string) =>
    fetchApi<AdminResponse>(`/admin/api-keys/openai-compatible/${id}`, {
      method: "DELETE",
    }),

  testOpenAICompatibleEndpoint: (id: string) =>
    fetchApi<TestEndpointResponse>(`/admin/api-keys/openai-compatible/${id}/test`, {
      method: "POST",
    }),
};
