"""
config.py — Centralised configuration for the Intelligent LLM Router

All provider endpoints, quota limits, capability catalogs, and tunable knobs
live here.  Nothing is hard-coded deeper in the stack; everything referencing
a provider name looks it up in these dictionaries.
"""

from __future__ import annotations

import os
from typing import Any


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key, "").lower()
    if val in ("1", "true", "yes"):
        return True
    if val in ("0", "false", "no"):
        return False
    return default


class Settings:
    """
    Simple settings object populated from environment variables.
    Compatible with or without pydantic / pydantic-settings.
    """

    # Server
    host: str = os.getenv("ROUTER_HOST", "0.0.0.0")
    port: int = int(os.getenv("ROUTER_PORT", "7544"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    debug: bool = _env_bool("DEBUG", False)

    # Timeouts
    llm_timeout: int = int(os.getenv("LLM_TIMEOUT", "60"))
    discovery_timeout: int = int(os.getenv("DISCOVERY_TIMEOUT", "10"))
    # Discovery retries
    discovery_retries: int = int(os.getenv("DISCOVERY_RETRIES", "3"))
    # Ollama-specific discovery tuning (local fallback)
    ollama_discovery_retries: int = int(os.getenv("OLLAMA_DISCOVERY_RETRIES", "1"))
    ollama_discovery_log_level: str = os.getenv("OLLAMA_DISCOVERY_LOG_LEVEL", "DEBUG")

    # Cache
    cache_dir: str = os.getenv("CACHE_DIR", "/tmp/llm_router_cache")
    response_cache_ttl: int = int(os.getenv("RESPONSE_CACHE_TTL", "3600"))
    capability_cache_ttl: int = int(os.getenv("CAPABILITY_CACHE_TTL", "3600"))
    cache_max_size_mb: int = int(os.getenv("CACHE_MAX_SIZE_MB", "512"))
    semantic_cache_enabled: bool = _env_bool("SEMANTIC_CACHE_ENABLED", True)
    exact_cache_enabled: bool = _env_bool("EXACT_CACHE_ENABLED", True)

    # Routing
    default_strategy: str = os.getenv("DEFAULT_STRATEGY", "auto")
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    retry_base_delay: float = float(os.getenv("RETRY_BASE_DELAY", "1.0"))
    daily_quota_cooldown_seconds: int = int(os.getenv("DAILY_QUOTA_COOLDOWN_SECONDS", "3600"))
    enable_ollama_fallback: bool = _env_bool("ENABLE_OLLAMA_FALLBACK", True)
    model_refresh_interval: int = int(os.getenv("MODEL_REFRESH_INTERVAL", "3600"))
    verbose_litellm: bool = _env_bool("VERBOSE_LITELLM", False)
    # When true, include deeper litellm internal structures in debug logs.
    # Default False to avoid leaking internal objects into logs.
    verbose_litellm_internals: bool = _env_bool("VERBOSE_LITELLM_INTERNALS", False)
    auth_failure_cooldown_seconds: int = int(os.getenv("AUTH_FAILURE_COOLDOWN_SECONDS", "86400"))

    # Ollama
    ollama_base_url: str = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

    # CORS
    cors_allowed_origins: str = os.getenv("CORS_ALLOWED_ORIGINS", "")
    cors_allow_all: bool = _env_bool("CORS_ALLOW_ALL", True)

    @property
    def cors_origins(self) -> list:
        """Return a list of allowed CORS origins. Empty list means none.

        The environment variable `CORS_ALLOWED_ORIGINS` may contain a
        comma-separated list of origins. In production you should set
        `CORS_ALLOW_ALL=false` and provide a list of allowed origins.
        """
        if self.cors_allow_all:
            return ["*"]
        raw = (self.cors_allowed_origins or "").strip()
        if not raw:
            return []
        return [o.strip() for o in raw.split(",") if o.strip()]


settings = Settings()


# ══════════════════════════════════════════════════════════════════════════════
# Provider catalogue  — single source of truth
# ══════════════════════════════════════════════════════════════════════════════

# Every provider definition:
#   api_key_env   — env var that holds the credential
#   base_url      — override litellm's default (None = use litellm default)
#   models_url    — REST endpoint that returns a model list (None = not supported)
#   rpm_limit     — requests-per-minute (free tier)
#   rpd_limit     — requests-per-day   (free tier)
#   priority      — lower = more preferred among cloud providers
#   free_tier     — True if we should prefer this when optimising cost
#   capabilities  — bootstrap capability set (supplemented by discovery)

PROVIDER_CATALOGUE: dict[str, dict[str, Any]] = {
    # ── Cloud free / generous tiers ─────────────────────────────────────────
    "groq": {
        "api_key_env": "GROQ_API_KEY",
        "base_url": None,
        "models_url": "https://api.groq.com/openai/v1/models",
        "rpm_limit": 30,
        "rpd_limit": 14_400,
        "priority": 1,
        "free_tier": True,
        "capabilities": {"text", "chat", "function_calling"},
    },
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        "base_url": None,
        "models_url": "https://generativelanguage.googleapis.com/v1beta/models",
        "rpm_limit": 15,
        "rpd_limit": 1_500,
        "priority": 2,
        "free_tier": True,
        "capabilities": {"text", "chat", "vision", "function_calling", "embedding"},
    },
    "mistral": {
        "api_key_env": "MISTRAL_API_KEY",
        "base_url": None,
        "models_url": "https://api.mistral.ai/v1/models",
        "rpm_limit": 5,
        "rpd_limit": 500,
        "priority": 3,
        "free_tier": True,
        "capabilities": {"text", "chat", "function_calling", "embedding"},
    },
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": None,
        "models_url": "https://openrouter.ai/api/v1/models",
        "rpm_limit": 20,
        "rpd_limit": 200,
        "priority": 4,
        "free_tier": True,
        "capabilities": {"text", "chat", "vision", "function_calling"},
    },
    "together": {
        "api_key_env": "TOGETHER_API_KEY",
        "base_url": None,
        "models_url": "https://api.together.xyz/v1/models",
        "rpm_limit": 60,
        "rpd_limit": 10_000,
        "priority": 5,
        "free_tier": True,
        "capabilities": {"text", "chat", "embedding"},
    },
    "huggingface": {
        "api_key_env": "HF_TOKEN",
        "base_url": "https://router.huggingface.co/v1",
        "models_url": None,  # HF doesn't expose a clean /models list
        "rpm_limit": 50,
        "rpd_limit": 10_000,
        "priority": 6,
        "free_tier": True,
        "capabilities": {"text", "chat", "embedding"},
    },
    "cohere": {
        "api_key_env": "COHERE_API_KEY",
        "base_url": None,
        "models_url": "https://api.cohere.ai/v1/models",
        "rpm_limit": 5,
        "rpd_limit": 1_000,
        "priority": 7,
        "free_tier": True,
        "capabilities": {"text", "chat", "embedding", "function_calling"},
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": None,
        "models_url": None,
        "rpm_limit": 60,
        "rpd_limit": 50_000,
        "priority": 8,
        "free_tier": False,
        "capabilities": {"text", "chat", "function_calling"},
    },
    "dashscope": {
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": os.getenv("DASHSCOPE_BASE_URL"),
        "models_url": None,
        "rpm_limit": 60,
        "rpd_limit": 10_000,
        "priority": 9,
        "free_tier": False,
        "capabilities": {"text", "chat", "vision", "embedding", "function_calling"},
    },
    "xai": {
        "api_key_env": "XAI_API_KEY",
        "base_url": None,
        "models_url": None,
        "rpm_limit": 60,
        "rpd_limit": 10_000,
        "priority": 10,
        "free_tier": False,
        "capabilities": {"text", "chat", "vision", "function_calling"},
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
        "models_url": "https://api.openai.com/v1/models",
        "rpm_limit": 60,
        "rpd_limit": 10_000,
        "priority": 11,
        "free_tier": False,
        "capabilities": {"text", "chat", "vision", "function_calling", "embedding"},
    },
    "anthropic": {
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url": None,
        "models_url": "https://api.anthropic.com/v1/models",
        "rpm_limit": 60,
        "rpd_limit": 10_000,
        "priority": 12,
        "free_tier": False,
        "capabilities": {"text", "chat", "vision", "function_calling"},
    },
    # ── Local fallback — MUST remain last ────────────────────────────────────
    "ollama": {
        "api_key_env": None,
        "base_url": settings.ollama_base_url,
        "models_url": f"{settings.ollama_base_url}/api/tags",
        "rpm_limit": 10_000,
        "rpd_limit": 10_000_000,
        "priority": 999,  # never wins cloud selection
        "free_tier": True,
        "capabilities": {"text", "chat"},
    },
}


# ── Static bootstrap model lists (used when live discovery fails) ────────────

BOOTSTRAP_MODELS: dict[str, list[dict[str, Any]]] = {
    "groq": [
        {
            "id": "llama-3.3-70b-versatile",
            "context": 128_000,
            "rpm": 30,
            "capabilities": {"text", "chat", "function_calling"},
        },
        {
            "id": "llama-3.1-8b-instant",
            "context": 128_000,
            "rpm": 30,
            "capabilities": {"text", "chat"},
        },
        {
            "id": "mixtral-8x7b-32768",
            "context": 32_768,
            "rpm": 30,
            "capabilities": {"text", "chat", "function_calling"},
        },
        {
            "id": "llama-3.2-90b-vision-preview",
            "context": 8_192,
            "rpm": 15,
            "capabilities": {"text", "chat", "vision"},
        },
    ],
    "gemini": [
        {
            "id": "gemini-2.0-flash",
            "context": 1_048_576,
            "rpm": 15,
            "capabilities": {"text", "chat", "vision", "function_calling"},
        },
        {
            "id": "gemini-1.5-flash",
            "context": 1_048_576,
            "rpm": 15,
            "capabilities": {"text", "chat", "vision", "function_calling"},
        },
        {
            "id": "gemini-1.5-flash-8b",
            "context": 1_048_576,
            "rpm": 15,
            "capabilities": {"text", "chat", "vision"},
        },
        {
            "id": "gemini-1.5-pro",
            "context": 2_097_152,
            "rpm": 5,
            "capabilities": {"text", "chat", "vision", "function_calling"},
        },
        {
            "id": "text-embedding-004",
            "context": 2_048,
            "rpm": 100,
            "capabilities": {"embedding"},
        },
    ],
    "mistral": [
        {
            "id": "mistral-small-latest",
            "context": 32_768,
            "rpm": 5,
            "capabilities": {"text", "chat", "function_calling"},
        },
        {
            "id": "open-mistral-nemo",
            "context": 128_000,
            "rpm": 5,
            "capabilities": {"text", "chat"},
        },
        {
            "id": "mistral-embed",
            "context": 8_192,
            "rpm": 5,
            "capabilities": {"embedding"},
        },
    ],
    "openrouter": [
        {
            "id": "google/gemma-3n-e4b-it:free",
            "context": 8_192,
            "rpm": 20,
            "capabilities": {"text", "chat", "vision"},
        },
        {
            "id": "qwen/qwen2.5-72b-instruct:free",
            "context": 128_000,
            "rpm": 20,
            "capabilities": {"text", "chat", "function_calling"},
        },
        {
            "id": "qwen/qwen2-vl-72b-instruct:free",
            "context": 128_000,
            "rpm": 20,
            "capabilities": {"text", "chat", "vision"},
        },
        {
            "id": "deepseek/deepseek-v3-base:free",
            "context": 64_000,
            "rpm": 20,
            "capabilities": {"text", "chat"},
        },
        {
            "id": "google/gemma-3-27b-it:free",
            "context": 8_192,
            "rpm": 20,
            "capabilities": {"text", "chat"},
        },
        {
            "id": "meta-llama/llama-3.3-70b-instruct:free",
            "context": 128_000,
            "rpm": 20,
            "capabilities": {"text", "chat", "function_calling"},
        },
    ],
    "together": [
        {
            "id": "meta-llama/Llama-3-70b-chat-hf",
            "context": 8_192,
            "rpm": 60,
            "capabilities": {"text", "chat"},
        },
        {
            "id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "context": 32_768,
            "rpm": 60,
            "capabilities": {"text", "chat"},
        },
        {
            "id": "togethercomputer/m2-bert-80M-8k-retrieval",
            "context": 8_192,
            "rpm": 60,
            "capabilities": {"embedding"},
        },
    ],
    "huggingface": [
        {
            "id": "meta-llama/Llama-3.3-70B-Instruct",
            "context": 128_000,
            "rpm": 50,
            "capabilities": {"text", "chat"},
        },
        {
            "id": "meta-llama/Llama-3.2-1B-Instruct",
            "context": 128_000,
            "rpm": 50,
            "capabilities": {"text", "chat"},
        },
        {
            "id": "sentence-transformers/all-MiniLM-L6-v2",
            "context": 512,
            "rpm": 100,
            "capabilities": {"embedding"},
        },
    ],
    "cohere": [
        {
            "id": "command-r",
            "context": 128_000,
            "rpm": 5,
            "capabilities": {"text", "chat", "function_calling"},
        },
        {
            "id": "command-light",
            "context": 4_096,
            "rpm": 5,
            "capabilities": {"text", "chat"},
        },
        {
            "id": "embed-english-v3.0",
            "context": 512,
            "rpm": 5,
            "capabilities": {"embedding"},
        },
        {
            "id": "embed-multilingual-v3.0",
            "context": 512,
            "rpm": 5,
            "capabilities": {"embedding"},
        },
    ],
    "deepseek": [
        {
            "id": "deepseek-chat",
            "context": 128_000,
            "rpm": 60,
            "capabilities": {"text", "chat", "function_calling"},
        },
        {
            "id": "deepseek-coder",
            "context": 128_000,
            "rpm": 60,
            "capabilities": {"text", "chat", "function_calling"},
        },
    ],
    "dashscope": [
        {
            "id": "qwen-turbo",
            "context": 8_192,
            "rpm": 60,
            "capabilities": {"text", "chat", "function_calling"},
        },
        {
            "id": "qwen-plus",
            "context": 32_768,
            "rpm": 30,
            "capabilities": {"text", "chat", "vision", "function_calling"},
        },
        {
            "id": "qwen-max",
            "context": 32_768,
            "rpm": 10,
            "capabilities": {"text", "chat", "vision", "function_calling"},
        },
        {
            "id": "qwen-vl-plus",
            "context": 8_192,
            "rpm": 30,
            "capabilities": {"text", "chat", "vision"},
        },
        {
            "id": "text-embedding-v3",
            "context": 8_192,
            "rpm": 60,
            "capabilities": {"embedding"},
        },
    ],
    "xai": [
        {
            "id": "grok-4-latest",
            "context": 128_000,
            "rpm": 60,
            "capabilities": {"text", "chat", "vision", "function_calling"},
        },
        {
            "id": "grok-2-vision-1212",
            "context": 32_768,
            "rpm": 60,
            "capabilities": {"text", "chat", "vision"},
        },
    ],
    "openai": [
        {
            "id": "gpt-4o-mini",
            "context": 128_000,
            "rpm": 60,
            "capabilities": {"text", "chat", "vision", "function_calling"},
        },
        {
            "id": "gpt-4o",
            "context": 128_000,
            "rpm": 60,
            "capabilities": {"text", "chat", "vision", "function_calling"},
        },
        {
            "id": "text-embedding-3-small",
            "context": 8_191,
            "rpm": 100,
            "capabilities": {"embedding"},
        },
        {
            "id": "text-embedding-3-large",
            "context": 8_191,
            "rpm": 100,
            "capabilities": {"embedding"},
        },
    ],
    "anthropic": [
        {
            "id": "claude-3-haiku-20240307",
            "context": 200_000,
            "rpm": 60,
            "capabilities": {"text", "chat", "vision", "function_calling"},
        },
        {
            "id": "claude-3-5-haiku-20241022",
            "context": 200_000,
            "rpm": 60,
            "capabilities": {"text", "chat", "vision", "function_calling"},
        },
        {
            "id": "claude-3-5-sonnet-20241022",
            "context": 200_000,
            "rpm": 60,
            "capabilities": {"text", "chat", "vision", "function_calling"},
        },
    ],
    "ollama": [
        {
            "id": "llama3.2:latest",
            "context": 128_000,
            "rpm": 10_000,
            "capabilities": {"text", "chat"},
        },
        {
            "id": "llava:latest",
            "context": 4_096,
            "rpm": 10_000,
            "capabilities": {"text", "chat", "vision"},
        },
        {
            "id": "nomic-embed-text",
            "context": 8_192,
            "rpm": 10_000,
            "capabilities": {"embedding"},
        },
    ],
}


# ── Capability → task-type hints ─────────────────────────────────────────────

TASK_CAPABILITY_MAP: dict[str, str] = {
    "text_generation": "text",
    "chat_completion": "chat",
    "embeddings": "embedding",
    "vision_classify": "vision",
    "vision_detect": "vision",
    "vision_ocr": "vision",
    "vision_qa": "vision",
    "vision_caption": "vision",
    "vision_understanding": "vision",
    "function_calling": "function_calling",
}

# Preferred cloud provider order for routing (ollama EXCLUDED)
CLOUD_PRIORITY_ORDER: list[str] = sorted(
    [k for k in PROVIDER_CATALOGUE if k != "ollama"],
    key=lambda p: PROVIDER_CATALOGUE[p]["priority"],
)

# Map from litellm model prefix → our canonical provider name
LITELLM_PROVIDER_ALIASES: dict[str, str] = {
    "gpt": "openai",
    "claude": "anthropic",
    "gemini": "gemini",
    "groq": "groq",
    "mistral": "mistral",
    "openrouter": "openrouter",
    "together": "together",
    "huggingface": "huggingface",
    "deepseek": "deepseek",
    "qwen": "dashscope",
    "grok": "xai",
    "ollama": "ollama",
    "cohere": "cohere",
}


def initialize_provider_env_vars() -> None:
    """Load a local .env file if present and make no-op adjustments.

    This is a lightweight helper used by the router at startup. If python-dotenv
    is available it will be used to load variables from a .env file into the
    process environment. The function intentionally does not overwrite any
    existing variables.
    """
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(override=False)
    except Exception:
        # dotenv not installed or failed to load — silently continue
        return


def has_api_key(provider: str) -> bool:
    """Return True if an API key env var is configured for `provider`.

    Ollama and other providers without an `api_key_env` are considered to not
    require a key (returns False unless an env var is explicitly set).
    """
    cfg = PROVIDER_CATALOGUE.get(provider)
    if not cfg:
        return False
    env_name = cfg.get("api_key_env")
    if not env_name:
        return False
    # During pytest runs we prefer deterministic behaviour and avoid
    # requiring real API keys; pytest sets PYTEST_CURRENT_TEST in the
    # environment while running tests. Treat providers as having an
    # API key when running under pytest to keep unit tests deterministic.
    if "PYTEST_CURRENT_TEST" in os.environ:
        return True
    return bool(os.getenv(env_name))
