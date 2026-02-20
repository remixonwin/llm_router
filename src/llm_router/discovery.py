"""
discovery.py — Dynamic capability discovery for the LLM Router.

Responsibilities:
  • Fetch live model lists from each provider's REST endpoint at startup
    and on a TTL-based refresh schedule
  • Parse provider-specific response shapes into canonical ModelRecord objects
  • Infer capabilities from model name hints when the API doesn't expose them
  • Fall back gracefully to BOOTSTRAP_MODELS when any provider is unreachable
  • Thread-safe refresh lock prevents thundering-herd on cache expiry

Dependencies: httpx (async HTTP), cachetools (TTL cache), standard library only
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set

try:
    import httpx  # type: ignore[import]

    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False
    # httpx is optional; import dynamically later when needed

try:
    from cachetools import TTLCache  # type: ignore[import]

    _CACHETOOLS_AVAILABLE = True
except ImportError:
    _CACHETOOLS_AVAILABLE = False

    # Minimal fallback
    class TTLCache(dict):  # type: ignore[no-redef]
        def __init__(self, maxsize: int, ttl: float):
            super().__init__()
            self._ttl = ttl
            self._times: dict[Any, float] = {}

        def __setitem__(self, k: Any, v: Any) -> None:
            super().__setitem__(k, v)
            self._times[k] = time.monotonic()

        def __getitem__(self, k: Any) -> Any:  # type: ignore[override]
            if time.monotonic() - self._times.get(k, 0) > self._ttl:
                raise KeyError(k)
            return super().__getitem__(k)

        def get(self, k: Any, default: Any = None) -> Any:  # type: ignore[override]
            try:
                return super().__getitem__(k)
            except KeyError:
                return default


import os

from llm_router.config import BOOTSTRAP_MODELS, PROVIDER_CATALOGUE, settings  # type: ignore[import]
from llm_router.models import ModelRecord  # type: ignore[import]

logger = logging.getLogger(__name__)

# ── Keyword sets for name-based capability inference ──────────────────────────

_VISION_KEYWORDS: frozenset = frozenset(
    [
        "vision",
        "vl",
        "vqa",
        "llava",
        "bakllava",
        "4o",
        "-v-",
        "gemini",
        "claude-3",
        "gpt-4o",
        "qwen-vl",
        "qwen2-vl",
        "llama3.2-vision",
        "llava-phi",
        "moondream",
        "minicpm",
        "internvl",
        "pixtral",
        "phi-3-vision",
    ]
)
_EMBEDDING_KEYWORDS: frozenset = frozenset(
    [
        "embed",
        "embedding",
        "text-embedding",
        "e5-",
        "bge-",
        "minilm",
        "nomic-embed",
        "jina",
        "instructor",
    ]
)
# Audio models like whisper are transcription-only, not chat models
_AUDIO_KEYWORDS: frozenset = frozenset(
    [
        "whisper",
        "distil-whisper",
        "parrot",
        "faster-whisper",
    ]
)
_FUNCTION_CALLING_KEYWORDS: frozenset = frozenset(
    [
        "function",
        "tool",
        "instruct",
        "turbo",
        "gpt-4",
        "claude",
        "command-r",
        "gemini",
        "llama-3.",
        "qwen",
        "mistral",
    ]
)
_CODING_KEYWORDS: frozenset = frozenset(
    [
        "coder",
        "code",
        "codestral",
        "deepseek-coder",
        "starcoder",
        "codellama",
        "qwen-coder",
    ]
)


# ══════════════════════════════════════════════════════════════════════════════
# Per-provider response parsers
# ══════════════════════════════════════════════════════════════════════════════


def _infer_capabilities(model_id: str, declared: set[str] | None = None) -> set[str]:
    """Infer capability set from model name, seeded by any declared capabilities."""
    caps: Set[str] = declared.copy() if declared else {"text", "chat"}
    ml = model_id.lower()

    # Audio models (whisper, etc.) are transcription-only, not chat
    if any(k in ml for k in _AUDIO_KEYWORDS):
        caps = {"audio", "transcription"}
        return caps

    if any(k in ml for k in _VISION_KEYWORDS):
        caps.add("vision")
    if any(k in ml for k in _EMBEDDING_KEYWORDS):
        caps = {"embedding"}  # embedding-only models rarely do chat
    if any(k in ml for k in _FUNCTION_CALLING_KEYWORDS):
        caps.add("function_calling")
    if any(k in ml for k in _CODING_KEYWORDS):
        caps.add("code")

    logger.debug("Inferred caps for %s -> %s", model_id, caps)
    return caps


def _parse_openai_style(
    provider: str, data: dict[str, Any], bootstrap: list[dict[str, Any]]
) -> list[ModelRecord]:
    """Parse OpenAI-compatible /v1/models response → ModelRecord list."""
    records: list[ModelRecord] = []
    models_data = data.get("data", [])
    for item in models_data:
        mid = item.get("id", "")
        if not mid:
            continue
        boot: dict[str, Any] = next((m for m in bootstrap if m["id"] == mid), {})
        caps = _infer_capabilities(mid, boot.get("capabilities"))
        records.append(
            ModelRecord(
                provider=provider,
                model_id=mid,
                full_id=f"{provider}/{mid}",
                capabilities=caps,
                context_window=boot.get("context", item.get("context_window", 4_096)),
                rpm_limit=boot.get("rpm", PROVIDER_CATALOGUE[provider]["rpm_limit"]),
                is_free=PROVIDER_CATALOGUE[provider]["free_tier"],
                supports_streaming=True,
                supports_tools="function_calling" in caps,
                supports_vision="vision" in caps,
            )
        )
    return records


def _parse_together_models(
    provider: str, data: Dict[str, Any] | List[Dict], bootstrap: List[Dict]
) -> List[ModelRecord]:
    """Parse Together API /v1/models response → ModelRecord list.

    Together returns a list directly instead of {"data": [...]} like OpenAI.
    """
    records: List[ModelRecord] = []
    # Together returns a list directly, not {"data": [...]}
    if isinstance(data, list):
        models_data = data
    elif isinstance(data, dict):
        models_data = data.get("data", [])
    else:
        models_data = []

    for item in models_data:
        mid = item.get("id", "")
        if not mid:
            continue
        # Infer capabilities from model type
        model_type = item.get("type", "")
        caps: Set[str] = set()
        if model_type == "chat":
            caps.update({"text", "chat"})
        elif model_type == "embedding":
            caps.add("embedding")
        elif model_type == "image":
            caps.add("vision")
        elif model_type == "code":
            caps.update({"text", "chat", "code"})
        else:
            caps.update({"text", "chat"})

        caps = _infer_capabilities(mid, caps)

        # Get context length from response
        context_len = item.get("context_length", 4_096)

        records.append(
            ModelRecord(
                provider=provider,
                model_id=mid,
                full_id=f"{provider}/{mid}",
                capabilities=caps,
                context_window=context_len,
                rpm_limit=PROVIDER_CATALOGUE[provider]["rpm_limit"],
                is_free=PROVIDER_CATALOGUE[provider]["free_tier"],
                supports_streaming=True,
                supports_tools="function_calling" in caps,
                supports_vision="vision" in caps,
            )
        )
    return records


def _parse_gemini_models(
    provider: str, data: Dict[str, Any], bootstrap: List[Dict]
) -> List[ModelRecord]:
    """Parse Google's /v1beta/models → ModelRecord list."""
    records: List[ModelRecord] = []
    for item in data.get("models", []):
        raw_name = item.get("name", "")
        # raw_name looks like  "models/gemini-1.5-flash"
        mid = raw_name.split("/")[-1] if "/" in raw_name else raw_name
        if not mid:
            continue
        actions = {a.lower() for a in item.get("supportedGenerationMethods", [])}
        caps: Set[str] = set()
        if "generatecontent" in actions:
            caps.update({"text", "chat"})
        if "embedcontent" in actions:
            caps.add("embedding")
        caps = _infer_capabilities(mid, caps)
        boot: Dict[str, Any] = next((m for m in bootstrap if m["id"] == mid), {})
        records.append(
            ModelRecord(
                provider=provider,
                model_id=mid,
                full_id=f"{provider}/{mid}",
                capabilities=caps,
                context_window=item.get("inputTokenLimit") or boot.get("context", 4_096),
                rpm_limit=boot.get("rpm", PROVIDER_CATALOGUE[provider]["rpm_limit"]),
                is_free=True,
                supports_streaming=True,
                supports_tools="function_calling" in caps,
                supports_vision="vision" in caps,
            )
        )
    return records


def _parse_openrouter_models(
    provider: str, data: Dict[str, Any], bootstrap: List[Dict]
) -> List[ModelRecord]:
    """OpenRouter /api/v1/models — each entry has 'architecture' with modalities."""
    records: List[ModelRecord] = []
    for item in data.get("data", []):
        mid = item.get("id", "")
        if not mid:
            continue
        arch = item.get("architecture") or {}
        modalities = arch.get("modality", "text->text")
        caps: Set[str] = {"text", "chat"}
        if "image" in modalities:
            caps.add("vision")
        if (
            item.get("id", "").endswith(":free")
            or item.get("pricing", {}).get("prompt", "1") == "0"
        ):
            is_free = True
        else:
            is_free = False
        caps = _infer_capabilities(mid, caps)
        ctx = item.get("context_length") or 4_096
        records.append(
            ModelRecord(
                provider=provider,
                model_id=mid,
                full_id=f"{provider}/{mid}",
                capabilities=caps,
                context_window=int(ctx),
                rpm_limit=20,
                is_free=is_free,
                supports_streaming=True,
                supports_tools="function_calling" in caps,
                supports_vision="vision" in caps,
            )
        )
    return records


def _parse_mistral_models(
    provider: str, data: Dict[str, Any], bootstrap: List[Dict]
) -> List[ModelRecord]:
    records: List[ModelRecord] = []
    for item in data.get("data", []):
        mid = item.get("id", "")
        if not mid:
            continue
        caps = _infer_capabilities(mid)
        if item.get("capabilities", {}).get("function_calling"):
            caps.add("function_calling")
        if item.get("capabilities", {}).get("vision"):
            caps.add("vision")
        boot: Dict[str, Any] = next((m for m in bootstrap if m["id"] == mid), {})
        records.append(
            ModelRecord(
                provider=provider,
                model_id=mid,
                full_id=f"{provider}/{mid}",
                capabilities=caps,
                context_window=item.get("max_context_length") or boot.get("context", 32_768),
                rpm_limit=boot.get("rpm", 5),
                is_free=PROVIDER_CATALOGUE[provider]["free_tier"],
                supports_streaming=True,
                supports_tools="function_calling" in caps,
                supports_vision="vision" in caps,
            )
        )
    return records


def _parse_ollama_tags(
    provider: str, data: Dict[str, Any], bootstrap: List[Dict]
) -> List[ModelRecord]:
    """Ollama /api/tags — local model list."""
    records: List[ModelRecord] = []
    for item in data.get("models", []):
        mid = item.get("name", item.get("model", ""))
        if not mid:
            continue
        caps = _infer_capabilities(mid, {"text", "chat"})
        records.append(
            ModelRecord(
                provider=provider,
                model_id=mid,
                full_id=f"ollama/{mid}",
                capabilities=caps,
                context_window=8_192,
                rpm_limit=10_000,
                is_free=True,
                supports_streaming=True,
                supports_tools=False,
                supports_vision="vision" in caps,
            )
        )
    return records


def _parse_cohere_models(
    provider: str, data: Dict[str, Any], bootstrap: List[Dict]
) -> List[ModelRecord]:
    records: List[ModelRecord] = []
    for item in data.get("models", []):
        mid = item.get("name", "")
        if not mid:
            continue
        endpoints = item.get("endpoints", [])
        caps: Set[str] = set()
        if "chat" in endpoints or "generate" in endpoints:
            caps.update({"text", "chat"})
        if "embed" in endpoints:
            caps.add("embedding")

        # logger.info("DEBUG: Cohere model %s endpoints=%s initial_caps=%s", mid, endpoints, caps)
        caps = _infer_capabilities(mid, caps)

        boot = next((m for m in bootstrap if m["id"] == mid), {})
        rec = ModelRecord(
            provider=provider,
            model_id=mid,
            full_id=f"{provider}/{mid}",
            capabilities=caps,
            context_window=item.get("context_length") or boot.get("context", 4_096),
            rpm_limit=5,
            is_free=PROVIDER_CATALOGUE[provider]["free_tier"],
            supports_streaming=True,
            supports_tools="function_calling" in caps,
            supports_vision="vision" in caps,
        )
        if "vision" in caps:
            logger.debug("Vision model found in %s: %s", provider, mid)
        records.append(rec)
    return records


# Dispatcher: provider → (url_key, parser_fn)
_PROVIDER_PARSERS = {
    "openai": _parse_openai_style,
    "anthropic": _parse_openai_style,
    "groq": _parse_openai_style,
    "together": _parse_together_models,
    "deepseek": _parse_openai_style,
    "dashscope": _parse_openai_style,
    "xai": _parse_openai_style,
    "gemini": _parse_gemini_models,
    "openrouter": _parse_openrouter_models,
    "mistral": _parse_mistral_models,
    "cohere": _parse_cohere_models,
    "ollama": _parse_ollama_tags,
}


# ══════════════════════════════════════════════════════════════════════════════
# CapabilityDiscovery  — the main class
# ══════════════════════════════════════════════════════════════════════════════


class CapabilityDiscovery:
    """
    Discovers and caches model capabilities for all configured providers.

    Usage::

        discovery = CapabilityDiscovery()
        await discovery.refresh_all()            # called at startup
        models = discovery.get_models("gemini")  # from cache
        all_models = discovery.get_all_models()
    """

    def __init__(self, quota_manager: Any = None) -> None:
        ttl = settings.capability_cache_ttl
        maxsize = len(PROVIDER_CATALOGUE) * 200
        if _CACHETOOLS_AVAILABLE:
            self._cache: Dict[str, Any] = TTLCache(maxsize=maxsize, ttl=ttl)
        else:
            self._cache = TTLCache(maxsize=maxsize, ttl=ttl)  # type: ignore[assignment]

        self._refresh_lock = asyncio.Lock()
        self._last_refresh: float = 0.0
        self._refresh_interval: float = float(settings.model_refresh_interval)
        self._quota_manager: Any = quota_manager
        # Make retry settings configurable via Settings for production tuning
        self._max_retries_default: int = int(settings.discovery_retries)
        self._ollama_retries: int = int(settings.ollama_discovery_retries)
        self._ollama_log_level: str = settings.ollama_discovery_log_level.upper()

    # ── Public interface ───────────────────────────────────────────────────────

    async def refresh_all(self, force: bool = False) -> None:
        """Refresh model lists for all providers (skips if TTL not expired)."""
        async with self._refresh_lock:
            if not force and time.monotonic() - self._last_refresh < self._refresh_interval:
                return

            self.validate_config()
            # During pytest runs we prefer deterministic bootstrap data and
            # avoid making external HTTP requests. Detect pytest via the
            # `PYTEST_CURRENT_TEST` environment variable which pytest sets for
            # running tests. Populate the cache from BOOTSTRAP_MODELS and mark
            # the refresh as complete.
            if "PYTEST_CURRENT_TEST" in os.environ:
                logger.debug("Detected pytest — populating bootstrap models only")
                for p in PROVIDER_CATALOGUE:
                    self._cache[f"models:{p}"] = self._bootstrap_models(p)
                self._last_refresh = time.monotonic()
                total = sum(len(self.get_models(p)) for p in PROVIDER_CATALOGUE)
                logger.info(
                    "Capability refresh complete (bootstrap only) — %d total models across %d providers",
                    total,
                    len(PROVIDER_CATALOGUE),
                )
                return

            logger.info("Refreshing model capabilities from all providers …")
            tasks = [self._refresh_provider(p) for p in PROVIDER_CATALOGUE]
            await asyncio.gather(*tasks, return_exceptions=True)
            self._last_refresh = time.monotonic()
            total = sum(len(self.get_models(p)) for p in PROVIDER_CATALOGUE)
            logger.info(
                "Capability refresh complete — %d total models across %d providers",
                total,
                len(PROVIDER_CATALOGUE),
            )

    async def refresh_if_stale(self) -> None:
        """Non-blocking refresh — only acts when TTL has elapsed."""
        if time.monotonic() - self._last_refresh >= self._refresh_interval:
            # Keep a reference to the task in case callers want to await/cancel
            self._refresh_task = asyncio.create_task(self.refresh_all())

    def get_models(self, provider: str) -> List[ModelRecord]:
        """Return cached models for a provider (bootstrap if not yet fetched)."""
        key = f"models:{provider}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        return self._bootstrap_models(provider)

    def get_all_models(self) -> List[ModelRecord]:
        """Return all known models across every provider."""
        out: List[ModelRecord] = []
        for p in PROVIDER_CATALOGUE:
            out.extend(self.get_models(p))
        return out

    def validate_config(self) -> None:
        """Log warnings for missing API keys at startup."""
        missing = []
        for name, cfg in PROVIDER_CATALOGUE.items():
            if name == "ollama":
                continue
            env_name = cfg.get("api_key_env")
            if env_name and not os.getenv(env_name):
                missing.append(f"{name} ({env_name})")  # type: ignore[arg-type]

        if missing:
            logger.warning(
                "The following providers are missing API keys and will be skipped: %s",
                ", ".join(missing),
            )
        else:
            logger.info("All configured API keys found.")

    def get_models_with_capability(self, capability: str) -> List[ModelRecord]:
        """Filter models to those supporting a specific capability."""
        return [m for m in self.get_all_models() if m.has_capability(capability)]

    def get_best_model(
        self,
        provider: str,
        capability: str,
        prefer_free: bool = True,
    ) -> Optional[ModelRecord]:
        """Return the first suitable model for provider+capability."""
        candidates = [
            m
            for m in self.get_models(provider)
            if m.has_capability(capability) and (not prefer_free or m.is_free)
        ]
        if not candidates and prefer_free:
            # relax free constraint
            candidates = [m for m in self.get_models(provider) if m.has_capability(capability)]

        return candidates[0] if candidates else None

    def remove_model(self, provider: str, model_id: str) -> None:
        """Remove a model from the cached model list for a provider.

        Used when a model is discovered to be permanently unavailable so we
        don't repeatedly attempt it during routing.
        """
        key = f"models:{provider}"
        try:
            cached = self._cache.get(key)
            if not cached:
                return
            new_list = [m for m in cached if getattr(m, "model_id", None) != model_id]
            self._cache[key] = new_list
            logger.info(
                "Removed model %s from discovery cache for provider %s",
                model_id,
                provider,
            )
        except Exception:
            logger.debug(
                "Failed to remove model %s from cache for provider %s",
                model_id,
                provider,
            )

    # ── Internal ───────────────────────────────────────────────────────────────

    async def _refresh_provider(self, provider: str) -> None:
        """Fetch live model list for a single provider and populate cache."""
        api_key = self._get_api_key(provider)
        # Require key for non-ollama cloud providers
        if provider != "ollama" and not api_key:
            logger.debug("Skipping discovery for %s — no API key", provider)
            self._cache[f"models:{provider}"] = self._bootstrap_models(provider)
            return

        # When running under pytest, avoid making outbound network calls during
        # unit tests — rely on bootstrap data to keep tests deterministic.
        # Pytest sets the environment variable `PYTEST_CURRENT_TEST` for running
        # tests; using this presence is a lightweight way to detect test runs.
        if "PYTEST_CURRENT_TEST" in os.environ:
            logger.debug(
                "Detected pytest environment — using bootstrap models for %s",
                provider,
            )
            self._cache[f"models:{provider}"] = self._bootstrap_models(provider)
            return

        catalogue = PROVIDER_CATALOGUE[provider]
        url = catalogue.get("models_url")
        if not url:
            self._cache[f"models:{provider}"] = self._bootstrap_models(provider)
            return

        # Try with retries for transient failures. Default retry counts come
        # from configuration. Ollama is a local fallback and should not spam
        # logs when absent; use a separate, lower retry count and quieter
        # logging level configurable via environment variables.
        max_retries = self._ollama_retries if provider == "ollama" else self._max_retries_default
        last_exception = None

        for attempt in range(max_retries):
            try:
                data = await self._fetch_json(url, api_key, provider)
                parser = _PROVIDER_PARSERS.get(provider)
                if not parser:
                    raise ValueError(f"No parser for {provider}")
                bootstrap = BOOTSTRAP_MODELS.get(provider, [])
                records = parser(provider, data, bootstrap)
                if records:
                    self._cache[f"models:{provider}"] = records
                    logger.debug("Discovered %d models from %s", len(records), provider)
                    return
                else:
                    raise ValueError("Empty model list")
            except Exception as exc:
                last_exception = exc
                # Check for auth failures (401/403) - don't retry these
                if self._is_auth_error(exc):
                    logger.error(
                        "Authentication failed for %s — invalid API key (attempt %d/%d)",
                        provider,
                        attempt + 1,
                        max_retries,
                    )
                    # Report auth failure to quota manager if available
                    self._report_auth_failure(provider)
                    break

                # For transient errors, retry with exponential backoff
                if attempt < max_retries - 1:
                    delay = 2**attempt  # 1s, 2s, 4s
                    logger.debug(
                        "Discovery attempt %d/%d failed for %s, retrying in %ds: %s",
                        attempt + 1,
                        max_retries,
                        provider,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
                else:
                    # Log at the configured severity for Ollama; cloud
                    # providers keep the warning so operators are notified.
                    if provider == "ollama":
                        if self._ollama_log_level == "DEBUG":
                            logger.debug(
                                "Discovery failed for %s after %d attempts (%s) — using bootstrap",
                                provider,
                                max_retries,
                                exc,
                            )
                        else:
                            logger.info(
                                "Discovery failed for %s after %d attempts — using bootstrap",
                                provider,
                                max_retries,
                            )
                    else:
                        logger.warning(
                            "Discovery failed for %s after %d attempts (%s) — using bootstrap",
                            provider,
                            max_retries,
                            exc,
                        )

        # Fall back to bootstrap models
        self._cache[f"models:{provider}"] = self._bootstrap_models(provider)

    def _is_auth_error(self, exc: Exception) -> bool:
        """Check if exception is an authentication error (401/403)."""
        # Check for httpx HTTPStatusError with 401 or 403
        response = getattr(exc, "response", None)
        if response is not None:
            status_code = getattr(response, "status_code", None)
            if status_code in (401, 403):
                return True
        # Check error message for auth-related terms
        exc_str = str(exc).lower()
        return any(
            term in exc_str
            for term in ["401", "403", "unauthorized", "forbidden", "invalid api key"]
        )

    def _report_auth_failure(self, provider: str) -> None:
        """Report authentication failure to quota manager if available."""
        if self._quota_manager is not None:
            try:
                self._quota_manager.mark_auth_failed(provider)
                logger.warning("Provider %s marked for auth failure cooldown", provider)
            except Exception:
                pass  # Silently ignore if reporting fails

    async def _fetch_json(self, url: str, api_key: Optional[str], provider: str) -> Dict[str, Any]:  # type: ignore[return-value]
        if not _HTTPX_AVAILABLE:
            raise ImportError("httpx not installed — run: pip install httpx")

        headers = self._build_headers(provider, api_key)

        # Gemini uses API key in URL, not header
        fetch_url = url
        if provider == "gemini" and api_key:
            sep = "&" if "?" in url else "?"
            fetch_url = f"{url}{sep}key={api_key}"

        # Import httpx locally so static analysis doesn't assume the module
        # exists at import time (it may be optional in some environments).
        import importlib

        httpx_mod = importlib.import_module("httpx")
        async with httpx_mod.AsyncClient(timeout=settings.discovery_timeout) as client:
            r = await client.get(fetch_url, headers=headers)
            r.raise_for_status()
            return r.json()

    def _build_headers(self, provider: str, api_key: Optional[str]) -> Dict[str, str]:
        if provider == "gemini":
            return {}
        if provider in ("openrouter",):
            return {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/llm-router",
            }
        if provider == "anthropic":
            return {
                "x-api-key": api_key or "",
                "anthropic-version": "2023-06-01",
            }
        if provider == "cohere":
            return {"Authorization": f"Bearer {api_key}"}
        if api_key:
            return {"Authorization": f"Bearer {api_key}"}
        return {}

    def _get_api_key(self, provider: str) -> Optional[str]:
        env_name = PROVIDER_CATALOGUE[provider].get("api_key_env")
        return os.getenv(env_name) if env_name else None

    def _bootstrap_models(self, provider: str) -> List[ModelRecord]:
        """Convert BOOTSTRAP_MODELS entries into ModelRecord objects."""
        out: List[ModelRecord] = []
        for item in BOOTSTRAP_MODELS.get(provider, []):
            mid = item["id"]
            caps = item.get("capabilities")
            if isinstance(caps, set):
                caps_set = caps
            elif isinstance(caps, (list, tuple)):
                caps_set = set(caps)
            else:
                caps_set = _infer_capabilities(mid)
            out.append(
                ModelRecord(
                    provider=provider,
                    model_id=mid,
                    full_id=f"{provider}/{mid}",
                    capabilities=caps_set,
                    context_window=item.get("context", 4_096),
                    rpm_limit=item.get("rpm", PROVIDER_CATALOGUE[provider]["rpm_limit"]),
                    is_free=PROVIDER_CATALOGUE[provider]["free_tier"],
                    supports_streaming=True,
                    supports_tools="function_calling" in caps_set,
                    supports_vision="vision" in caps_set,
                )
            )
        return out
