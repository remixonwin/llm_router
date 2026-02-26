import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../../services/api";
import { Key, Eye, EyeOff, Trash2, Save, CheckCircle, AlertCircle, Loader2 } from "lucide-react";
import type { ApiKeyStatus } from "../../types";

const PROVIDER_DISPLAY_NAMES: Record<string, string> = {
  groq: "Groq",
  gemini: "Google Gemini",
  mistral: "Mistral",
  openrouter: "OpenRouter",
  together: "Together AI",
  huggingface: "HuggingFace",
  cohere: "Cohere",
  deepseek: "DeepSeek",
  dashscope: "DashScope",
  xai: "xAI",
  openai: "OpenAI",
  anthropic: "Anthropic",
  openai_compatible: "OpenAI Compatible",
};

function ProviderKeyCard({ providerKey }: { providerKey: ApiKeyStatus }) {
  const queryClient = useQueryClient();
  const [apiKey, setApiKey] = useState("");
  const [showKey, setShowKey] = useState(false);

  const setKeyMutation = useMutation({
    mutationFn: () => api.setApiKey(providerKey.provider, apiKey),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["apiKeys"] });
      setApiKey("");
      showNotification(
        "success",
        `API key for ${PROVIDER_DISPLAY_NAMES[providerKey.provider] || providerKey.provider} updated`
      );
    },
    onError: (error: Error) => {
      showNotification("error", error.message);
    },
  });

  const deleteKeyMutation = useMutation({
    mutationFn: () => api.deleteApiKey(providerKey.provider),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["apiKeys"] });
      showNotification(
        "success",
        `API key for ${PROVIDER_DISPLAY_NAMES[providerKey.provider] || providerKey.provider} removed`
      );
    },
    onError: (error: Error) => {
      showNotification("error", error.message);
    },
  });

  const [notification, setNotification] = useState<{
    type: "success" | "error";
    message: string;
  } | null>(null);

  const showNotification = (type: "success" | "error", message: string) => {
    setNotification({ type, message });
    setTimeout(() => setNotification(null), 3000);
  };

  const displayName = PROVIDER_DISPLAY_NAMES[providerKey.provider] || providerKey.provider;

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      {notification && (
        <div
          className={`flex items-center gap-2 p-2 rounded-md mb-3 text-sm ${
            notification.type === "success"
              ? "bg-green-50 text-green-700"
              : "bg-red-50 text-red-700"
          }`}
        >
          {notification.type === "success" ? (
            <CheckCircle className="h-4 w-4" />
          ) : (
            <AlertCircle className="h-4 w-4" />
          )}
          {notification.message}
        </div>
      )}

      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Key className="h-4 w-4 text-gray-500" />
          <span className="font-medium text-gray-900">{displayName}</span>
        </div>
        <span
          className={`px-2 py-1 text-xs font-medium rounded ${
            providerKey.has_key ? "bg-green-100 text-green-700" : "bg-gray-100 text-gray-500"
          }`}
        >
          {providerKey.has_key ? "Configured" : "Not configured"}
        </span>
      </div>

      {providerKey.has_key && providerKey.key_masked && (
        <p className="text-sm text-gray-500 mb-3">Current: {providerKey.key_masked}</p>
      )}

      <div className="flex gap-2">
        <div className="relative flex-1">
          <input
            type={showKey ? "text" : "password"}
            placeholder={providerKey.has_key ? "Enter new key to replace" : "Enter API key"}
            value={apiKey}
            onChange={e => setApiKey(e.target.value)}
            className="w-full px-3 py-2 pr-10 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
          />
          <button
            type="button"
            onClick={() => setShowKey(!showKey)}
            className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
          >
            {showKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
          </button>
        </div>

        {apiKey && (
          <button
            onClick={() => setKeyMutation.mutate()}
            disabled={setKeyMutation.isPending}
            className="px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center gap-1 text-sm"
          >
            {setKeyMutation.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Save className="h-4 w-4" />
            )}
            Save
          </button>
        )}

        {providerKey.has_key && (
          <button
            onClick={() => deleteKeyMutation.mutate()}
            disabled={deleteKeyMutation.isPending}
            className="px-3 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 flex items-center gap-1 text-sm"
          >
            {deleteKeyMutation.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Trash2 className="h-4 w-4" />
            )}
            Remove
          </button>
        )}
      </div>
    </div>
  );
}

export function ApiKeys() {
  const {
    data: apiKeys,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ["apiKeys"],
    queryFn: api.getApiKeys,
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
        <span className="text-red-700">Failed to load API keys</span>
        <button
          onClick={() => refetch()}
          className="ml-auto text-sm text-red-600 hover:text-red-800 underline"
        >
          Retry
        </button>
      </div>
    );
  }

  const configured = apiKeys?.filter(k => k.has_key) || [];
  const unconfigured = apiKeys?.filter(k => !k.has_key) || [];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">API Keys</h3>
          <p className="text-sm text-gray-500">
            Configure API keys for each provider. Keys are saved to .env file.
          </p>
        </div>
        <div className="text-sm text-gray-500">
          {configured.length} / {apiKeys?.length || 0} configured
        </div>
      </div>

      {configured.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-3">Configured</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {configured.map(pk => (
              <ProviderKeyCard key={pk.provider} providerKey={pk} />
            ))}
          </div>
        </div>
      )}

      {unconfigured.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-3">Not Configured</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {unconfigured.map(pk => (
              <ProviderKeyCard key={pk.provider} providerKey={pk} />
            ))}
          </div>
        </div>
      )}

      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
        <p className="text-sm text-yellow-700">
          <strong>Note:</strong> After adding or removing API keys, restart the router for changes
          to take effect.
        </p>
      </div>
    </div>
  );
}
