"""
router.py — Intelligent LLM Router (core routing engine).

Combines:
  • CapabilityDiscovery  — live model/capability data
  • QuotaManager         — rate-limit, scoring, cooldown
  • ResponseCache        — exact + semantic deduplication

Decision flow for every request:

  1.  Check exact cache  → serve instantly if hit
  2.  Detect request type (text / vision / embedding / function-calling)
  3.  Collect eligible providers:
        - have the required capability
        - have remaining RPD quota
        - not in cooldown / circuit-broken
        - Ollama explicitly EXCLUDED from this list
  4.  Score providers by strategy weights; weighted-random selection
      prevents starvation of lower-scored providers
  5.  Attempt the call.  On failure:
        a. Rate-limit  → retry with exponential back-off (up to MAX_RETRIES)
                         try next model on same provider before giving up
        b. Daily limit → put provider in long cooldown; advance to next
        c. Network err → short cooldown; advance to next provider
  6.  After exhausting ALL cloud providers, fall back to Ollama
      ONLY if settings.enable_ollama_fallback is True
  7.  Populate cache on success

Dependencies: litellm, tenacity
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import time
from typing import Any

from .cache import ResponseCache
from .config import (
    CLOUD_PRIORITY_ORDER,
    PROVIDER_CATALOGUE,
    initialize_provider_env_vars,
    settings,
)
from .discovery import CapabilityDiscovery
from .models import (
    CachePolicy,
    ModelRecord,
    ProviderState,
    RouteDecision,
    RoutingOptions,
    RoutingStrategy,
    TaskType,
)
from .quota import QuotaManager

# litellm symbols may not exist in every environment; pre-declare for mypy
litellm: Any = None
acompletion: Any = None
LiteLLMRateLimit: Any
LiteLLMTimeout: Any
LiteLLMConnectionError: Any
_LITELLM_AVAILABLE: bool = False

try:
    import litellm
    from litellm import APIConnectionError as LiteLLMConnectionError
    from litellm import RateLimitError as LiteLLMRateLimit
    from litellm import Timeout as LiteLLMTimeout
    from litellm import acompletion

    _LITELLM_AVAILABLE = True
except ImportError:
    _LITELLM_AVAILABLE = False
    litellm = None  # type: ignore[assignment]
    acompletion = None  # type: ignore[assignment]
    LiteLLMRateLimit = Exception  # fallback
    LiteLLMTimeout = Exception  # fallback
    LiteLLMConnectionError = Exception  # fallback

logger = logging.getLogger(__name__)

# Keyword patterns for task-type detection
_VISION_CONTENT_INDICATORS: frozenset = frozenset(["image_url", "data:image"])
_EMBEDDING_TASK_TYPES: frozenset = frozenset(["embeddings", "embedding"])

# Strategy weight presets  (w_quota, w_latency, w_quality, w_errors)
_STRATEGY_WEIGHTS: dict[str, dict[str, float]] = {
    RoutingStrategy.AUTO: {"w_quota": 0.50, "w_latency": 0.25, "w_quality": 0.15, "w_errors": 0.10},
    RoutingStrategy.COST_OPTIMIZED: {
        "w_quota": 0.80,
        "w_latency": 0.10,
        "w_quality": 0.05,
        "w_errors": 0.05,
    },
    RoutingStrategy.QUALITY_FIRST: {
        "w_quota": 0.10,
        "w_latency": 0.20,
        "w_quality": 0.60,
        "w_errors": 0.10,
    },
    RoutingStrategy.LATENCY_FIRST: {
        "w_quota": 0.20,
        "w_latency": 0.60,
        "w_quality": 0.10,
        "w_errors": 0.10,
    },
    RoutingStrategy.ROUND_ROBIN: {
        "w_quota": 0.25,
        "w_latency": 0.25,
        "w_quality": 0.25,
        "w_errors": 0.25,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Helper: weighted random choice
# ══════════════════════════════════════════════════════════════════════════════


def _weighted_choice(scores: dict[str, float]) -> str | None:
    """Probabilistic selection weighted by score — spreads load across providers."""
    if not scores:
        return None
    total = sum(max(v, 0.0) for v in scores.values())
    if total <= 0:
        return max(scores, key=lambda k: scores[k])
    r = random.uniform(0, total)
    cumulative = 0.0
    for name, score in scores.items():
        cumulative += max(score, 0.0)
        if cumulative >= r:
            return name
    return list(scores)[-1]


# ══════════════════════════════════════════════════════════════════════════════
# IntelligentRouter
# ══════════════════════════════════════════════════════════════════════════════


class IntelligentRouter:
    """
    Top-level router. Instantiate once and reuse across the application lifetime.

    Entry point: ``await router.route(request_data)``
    """

    def __init__(self) -> None:
        self.discovery = CapabilityDiscovery()
        self.quota = QuotaManager()
        self.cache = ResponseCache()
        self._initialised = False

    # ── Lifecycle ───────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Call once at application startup (FastAPI lifespan)."""
        await self.quota.start()
        await self.discovery.refresh_all()
        self._configure_litellm()
        self._initialised = True
        logger.info("IntelligentRouter ready — %d providers", len(PROVIDER_CATALOGUE))

    async def stop(self) -> None:
        await self.quota.stop()

    def _configure_litellm(self) -> None:
        """Push environment API keys into litellm and suppress its verbose logs."""
        if not _LITELLM_AVAILABLE:
            logger.error("litellm is not installed. Run: pip install litellm")
            return

        os.environ.setdefault("LITELLM_LOG", "ERROR")

        # Initialize provider environment variables from .env file
        initialize_provider_env_vars()

        # Configure API keys for each provider
        configured_providers = []
        missing_providers = []

        for provider, cfg in PROVIDER_CATALOGUE.items():
            env_name = cfg.get("api_key_env")
            if env_name:
                val = os.getenv(env_name)
                if val:
                    os.environ[env_name] = val
                    configured_providers.append(provider)
                elif provider != "ollama":
                    missing_providers.append((provider, env_name))

            # Tell litellm about custom base URLs (dashscope, ollama,
            # huggingface, etc.)
            base_url = cfg.get("base_url")
            if base_url:
                os.environ[f"{provider.upper()}_BASE_URL"] = base_url
                os.environ[f"{provider.upper()}_API_BASE"] = base_url

        # Configure OpenAI-compatible custom providers (multiple endpoints)
        oai_endpoints = settings.get_enabled_openai_compatible_endpoints()
        for endpoint in oai_endpoints:
            endpoint_id = endpoint.get("id", "default")
            provider_name = f"openai_compatible_{endpoint_id}"

            base_url = endpoint.get("base_url", "")
            api_key = endpoint.get("api_key")

            if base_url:
                os.environ[f"OPENAI_COMPATIBLE_{endpoint_id.upper()}_API_BASE"] = base_url
                if api_key:
                    os.environ[f"OPENAI_COMPATIBLE_{endpoint_id.upper()}_API_KEY"] = api_key
                configured_providers.append(provider_name)
                logger.info(
                    "Configured OpenAI-compatible endpoint '%s': %s",
                    endpoint.get("name", endpoint_id),
                    base_url,
                )

        # Legacy single OAI endpoint support
        if settings.openai_compatible_api_base and not oai_endpoints:
            os.environ["OPENAI_COMPATIBLE_API_BASE"] = settings.openai_compatible_api_base
            api_key = os.getenv("ROUTER_OPENAI_COMPATIBLE_API_KEY", "")
            if api_key:
                os.environ["OPENAI_COMPATIBLE_API_KEY"] = api_key
                configured_providers.append("openai_compatible")
            else:
                missing_providers.append(("openai_compatible", "ROUTER_OPENAI_COMPATIBLE_API_KEY"))
            logger.info(
                "Configured OpenAI-compatible provider: %s",
                settings.openai_compatible_api_base,
            )

        # Log configuration status
        if configured_providers:
            logger.info(f"Configured API keys for providers: {', '.join(configured_providers)}")
        else:
            logger.warning("No API keys configured for any cloud providers")

        if missing_providers:
            for provider, env_name in missing_providers:
                logger.warning(
                    f"Provider '{provider}' requires API key but {env_name} is not set. "
                    f"Add {env_name} to your .env file."
                )

    # ══════════════════════════════════════════════════════════════════════════
    # Primary routing entry point
    # ══════════════════════════════════════════════════════════════════════════

    async def route(
        self,
        request_data: dict[str, Any],
        routing_options: RoutingOptions | None = None,
    ) -> dict[str, Any]:
        """
        Route a request to the optimal provider.

        ``request_data`` must contain at minimum one of:
          • ``messages`` — for chat/text
          • ``input``    — for embeddings
          • ``image_url`` or ``image_base64`` — for vision
        """
        await self.discovery.refresh_if_stale()

        opts = routing_options or RoutingOptions()
        task_type = self._detect_task_type(request_data)

        # ── Cache lookup ─────────────────────────────────────────────────────
        if opts.cache_policy == CachePolicy.ENABLED:
            messages = request_data.get("messages", [])
            # Construct cache keys in a backwards-compatible way: some callers
            # (and tests) create keys without task_type. First try the
            # task_type-agnostic key, then fall back to the task_type-specific
            # key if present.
            task_type_str = (
                task_type.value
                if isinstance(task_type, TaskType)
                else str(task_type)
                if task_type
                else None
            )
            # Try key without task_type first for compatibility
            cache_key = self.cache.make_key(
                messages, request_data.get("model"), request_data.get("temperature"), task_type=None
            )
            cached = self.cache.get_exact(cache_key)
            if cached is not None:
                cached["routing_metadata"]["cache_hit"] = True
                return cached

            # If not found, try including the detected task_type
            cache_key = self.cache.make_key(
                messages,
                request_data.get("model"),
                request_data.get("temperature"),
                task_type=task_type_str,
            )
            cached = self.cache.get_exact(cache_key)
            if cached is not None:
                cached["routing_metadata"]["cache_hit"] = True
                return cached
        else:
            cache_key = ""  # type: ignore[assignment]

        # ── Dispatch ─────────────────────────────────────────────────────────
        if task_type in (TaskType.EMBEDDINGS,):
            result = await self._route_embedding(request_data, opts)
        elif task_type in (
            TaskType.VISION_CLASSIFY,
            TaskType.VISION_DETECT,
            TaskType.VISION_OCR,
            TaskType.VISION_QA,
            TaskType.VISION_CAPTION,
            TaskType.VISION_UNDERSTANDING,
        ):
            result = await self._route_vision(request_data, opts, task_type)
        else:
            result = await self._route_text(request_data, opts)

        # ── Cache store ──────────────────────────────────────────────────────
        # Skip caching for streaming requests - the response is an async iterable
        # that cannot be pickled. But cache if it's a dict (collected streaming response).
        is_streaming = request_data.get("stream", False)
        is_dict_response = isinstance(result, dict)

        if opts.cache_policy != CachePolicy.DISABLED and cache_key and is_dict_response:
            self.cache.set(cache_key, result, prompt_text=self._extract_prompt_text(request_data))

        return result

    # ══════════════════════════════════════════════════════════════════════════
    # Text / chat routing
    # ══════════════════════════════════════════════════════════════════════════

    async def _route_text(
        self, request_data: dict[str, Any], opts: RoutingOptions
    ) -> dict[str, Any]:
        messages = request_data.get("messages", [])
        has_tools = bool(request_data.get("tools"))
        capability = "function_calling" if has_tools else "chat"

        decision = self._select(capability, opts)
        if decision is None:
            raise RuntimeError("No provider available for text/chat completion")

        return await self._attempt_with_fallback(
            primary=decision,
            messages=messages,
            capability=capability,
            opts=opts,
            extra_params={
                "temperature": request_data.get("temperature"),
                "max_tokens": request_data.get("max_tokens"),
                "top_p": request_data.get("top_p"),
                "stream": request_data.get("stream", False),
                "tools": request_data.get("tools"),
                "tool_choice": request_data.get("tool_choice"),
            },
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Vision routing
    # ══════════════════════════════════════════════════════════════════════════

    async def _route_vision(
        self, request_data: dict[str, Any], opts: RoutingOptions, task_type: TaskType
    ) -> dict[str, Any]:
        prompt = self._build_vision_prompt(request_data, task_type)
        image_content = self._build_image_content(request_data)

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, *image_content],
            }
        ]
        decision = self._select("vision", opts)
        if decision is None:
            raise RuntimeError("No vision-capable provider available")

        return await self._attempt_with_fallback(
            primary=decision,
            messages=messages,
            capability="vision",
            opts=opts,
            extra_params={"temperature": request_data.get("temperature", 0.2)},
        )

    def _build_image_content(self, request_data: dict[str, Any]) -> list[dict[str, Any]]:
        if request_data.get("image_url"):
            return [{"type": "image_url", "image_url": {"url": request_data["image_url"]}}]
        if request_data.get("image_base64"):
            mime = request_data.get("mime_type", "image/jpeg")
            b64 = request_data["image_base64"]
            data_url = b64 if b64.startswith("data:") else f"data:{mime};base64,{b64}"
            return [{"type": "image_url", "image_url": {"url": data_url}}]
        # Check OpenAI-style messages for embedded images
        for msg in request_data.get("messages", []):
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            if isinstance(content, list):
                return [p for p in content if isinstance(p, dict) and p.get("type") == "image_url"]
        return []

    def _build_vision_prompt(self, request_data: dict[str, Any], task_type: TaskType) -> str:
        prompts = {
            TaskType.VISION_CLASSIFY: (
                "Classify this image. Return the top categories with confidence scores."
            ),
            TaskType.VISION_DETECT: (
                "Detect all objects in this image. Provide bounding box descriptions and labels."
            ),
            TaskType.VISION_OCR: (
                "Extract all text from this image. Preserve the layout as closely as possible."
            ),
            TaskType.VISION_QA: (
                request_data.get("question") or "Answer any question about this image."
            ),
            TaskType.VISION_CAPTION: "Write a detailed, descriptive caption for this image.",
            TaskType.VISION_UNDERSTANDING: "Analyse and describe this image in detail.",
        }
        return prompts.get(task_type, "Describe this image.")

    # ══════════════════════════════════════════════════════════════════════════
    # Embedding routing
    # ══════════════════════════════════════════════════════════════════════════

    async def _route_embedding(
        self, request_data: dict[str, Any], opts: RoutingOptions
    ) -> dict[str, Any]:
        text_input = request_data.get("input", "")
        decision = self._select("embedding", opts)
        if decision is None:
            raise RuntimeError("No embedding provider available")

        start = time.monotonic()
        for provider, model in self._fallback_chain(decision, "embedding", opts):
            try:
                self.quota.consume(provider)
                result = await self._call_embedding(model.litellm_id, text_input)
                latency = (time.monotonic() - start) * 1000
                self.quota.record_latency(provider, latency)
                return {
                    "object": "list",
                    "data": [
                        {"object": "embedding", "embedding": e, "index": i}
                        for i, e in enumerate(result)
                    ],
                    "model": model.litellm_id,
                    "routing_metadata": {
                        "provider": provider,
                        "model": model.model_id,
                        "latency_ms": round(float(latency), 1),
                        "strategy": opts.strategy,
                        "cache_hit": False,
                    },
                }
            except Exception as exc:
                logger.warning("Embedding %s/%s failed: %s", provider, model.model_id, exc)
                self.quota.mark_error(provider)

        raise RuntimeError("All embedding providers failed")

    async def _call_embedding(self, model_id: str, text: Any) -> list[list[float]]:
        if not _LITELLM_AVAILABLE:
            raise ImportError("litellm required: pip install litellm")
        # Narrow types for the static checker
        assert litellm is not None and hasattr(litellm, "aembedding")
        texts = [text] if isinstance(text, str) else text
        response = await litellm.aembedding(model=model_id, input=texts)
        return [item["embedding"] for item in response["data"]]

    # ══════════════════════════════════════════════════════════════════════════
    # Attempt + fallback logic
    # ══════════════════════════════════════════════════════════════════════════

    async def _attempt_with_fallback(
        self,
        primary: RouteDecision,
        messages: list[dict[str, Any]],
        capability: str,
        opts: RoutingOptions,
        extra_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Try primary provider, fall back through cloud chain, then Ollama."""
        start = time.monotonic()
        original_provider = primary.provider

        for provider, model in self._fallback_chain(primary, capability, opts):
            result = await self._try_provider(
                provider=provider,
                model=model,
                messages=messages,
                extra_params=extra_params,
                start_time=start,
                original_provider=original_provider,
                strategy=opts.strategy,
            )
            if result is not None:
                return result

        raise RuntimeError(f"All providers exhausted for capability={capability}")

    def _sanitize_tools_for_provider(
        self, provider: str, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Remove provider-incompatible fields from tool definitions."""
        if not tools:
            return tools

        sanitized = []
        for tool in tools:
            if not isinstance(tool, dict):
                sanitized.append(tool)
                continue

            func = tool.get("function", {})
            if not isinstance(func, dict):
                sanitized.append(tool)
                continue

            params = func.get("parameters")
            if isinstance(params, dict):
                sanitized_params = self._sanitize_json_schema(params, provider)
                func = dict(func, parameters=sanitized_params)

                if provider == "cohere":
                    func.pop("strict", None)

            sanitized.append(dict(tool, function=func))

        return sanitized

    def _sanitize_json_schema(self, schema: dict[str, Any], provider: str) -> dict[str, Any]:
        """Ensure JSON schema is compatible with the target provider."""
        if not isinstance(schema, dict):
            return schema

        schema = dict(schema)

        if provider == "groq":
            self._ensure_additional_properties_false(schema)

        return schema

    def _ensure_additional_properties_false(self, obj: dict[str, Any]) -> None:
        """Recursively ensure all objects have additionalProperties: false."""
        if not isinstance(obj, dict):
            return

        if obj.get("type") == "object":
            if "properties" in obj and "additionalProperties" not in obj:
                obj["additionalProperties"] = False

        for value in obj.values():
            if isinstance(value, dict):
                self._ensure_additional_properties_false(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        self._ensure_additional_properties_false(item)

    async def _try_provider(
        self,
        provider: str,
        model: ModelRecord,
        messages: list[dict[str, Any]],
        extra_params: dict[str, Any],
        start_time: float,
        original_provider: str,
        strategy: str,
    ) -> dict[str, Any] | Any | None:
        """
        Attempt a single provider+model.  Handles:
          - RPM rate limits (exponential back-off, up to MAX_RETRIES)
          - Daily limits (long cooldown, return None immediately)
          - Network errors (short cooldown, return None)
        Returns a response dict on success, None to signal "try next".
        For streaming requests (stream=True), returns an async iterable.
        """
        if not self.quota.can_accept(provider):
            logger.debug("Skipping %s — quota gate", provider)
            return None

        params = {k: v for k, v in extra_params.items() if v is not None}

        if params.get("tools"):
            params["tools"] = self._sanitize_tools_for_provider(provider, params["tools"])

        client_requested_streaming = params.get("stream", False)
        # Check if streaming is forced by config for this provider
        provider_forced_streaming = (
            provider == "openai_compatible" and settings.openai_compatible_streaming
        )
        # Use streaming only if client requested it OR provider requires it
        params["stream"] = client_requested_streaming or provider_forced_streaming
        is_streaming = params["stream"]

        for attempt in range(1, settings.max_retries + 1):
            try:
                self.quota.consume(provider)
                response = await self._litellm_call(model.litellm_id, messages, params)
                latency: float = (time.monotonic() - start_time) * 1000.0
                self.quota.record_latency(provider, latency)

                # Record successful request (increments rpd_used, resets error counters)
                state = self.quota.states.get(provider)
                if state:
                    state.record_success(latency)

                # Handle streaming responses - return async iterable directly
                if is_streaming and hasattr(response, "__aiter__"):
                    # If client didn't request streaming but provider forced it,
                    # collect all chunks and return as single response
                    if not client_requested_streaming:
                        return await self._collect_streaming_response(
                            response, provider, model, latency, strategy, original_provider
                        )
                    return response

                return self._format_response(
                    response, provider, model, latency, strategy, original_provider
                )

            except LiteLLMRateLimit as e:
                # Some environments set LiteLLMRateLimit to a broad Exception
                # class or the upstream error may contain a permanent-model-removed
                # message. Detect that pattern early so we can prune the model
                # from discovery rather than treating it as a transient rate
                # limit.
                try:
                    msg = str(e).lower()
                    if "removed" in msg and self.discovery is not None:
                        try:
                            self.discovery.remove_model(provider, model.model_id)
                        except Exception:
                            pass
                except Exception:
                    pass
                delay = self._parse_retry_delay(e) or (settings.retry_base_delay * 2**attempt)
                if self._is_daily_limit(e):
                    self.quota.mark_daily_exhausted(provider)
                    logger.warning("Daily limit on %s — advancing to next provider", provider)
                    return None
                self.quota.mark_rate_limited(provider, delay)
                if attempt < settings.max_retries:
                    jitter = random.uniform(0, delay * 0.2)
                    await asyncio.sleep(delay + jitter)
                else:
                    return None

            except (LiteLLMTimeout, LiteLLMConnectionError) as e:
                # See note above in the rate-limit handler: if the underlying
                # exception message indicates the model was removed, ensure we
                # remove it from discovery cache.
                try:
                    msg = str(e).lower()
                    if "removed" in msg and self.discovery is not None:
                        try:
                            self.discovery.remove_model(provider, model.model_id)
                        except Exception:
                            pass
                except Exception:
                    pass
                self.quota.mark_error(provider)
                logger.warning("%s: connection/timeout — %s", provider, e)
                return None

            except Exception as e:
                msg = str(e).lower()
                # If upstream reports a model was removed, prune it from discovery
                # so we don't repeatedly attempt the same missing model.
                try:
                    if "removed" in msg and self.discovery is not None:
                        # discovery.remove_model expects (provider, model_id)
                        try:
                            self.discovery.remove_model(provider, model.model_id)
                        except Exception:
                            # best-effort only; do not surface removal failures
                            pass
                except Exception:
                    pass
                # 404 model-not-found → skip this model permanently
                if (
                    "404" in msg
                    or "model_not_found" in msg
                    or "does not exist" in msg
                    or "not a valid model" in msg
                ):
                    logger.warning("%s/%s: 404 — model not found", provider, model.model_id)
                    try:
                        self.discovery.remove_model(provider, model.model_id)
                    except Exception:
                        pass
                    return None
                self.quota.mark_error(provider)
                logger.warning("%s: unexpected error — %s", provider, e)
                return None

        return None

    async def _litellm_call(self, *args: Any, **kwargs: Any) -> Any:
        """Flexible wrapper around litellm.acompletion.

        Supports both internal callers that pass (model_id, messages, params)
        and tests which call (provider, model, messages, params). It also
        accepts keyword args. The function awaits and returns whatever
        litellm.acompletion returns.
        """
        # Allow tests to patch `acompletion` directly even if litellm isn't
        # importable in the environment. Only raise if litellm isn't available
        # and no `acompletion` function has been provided.
        if not _LITELLM_AVAILABLE and acompletion is None:
            raise ImportError("litellm not installed — run: pip install litellm")
        # Help the static analyzer understand `acompletion` is callable here
        assert acompletion is not None and callable(acompletion)

        model_to_call: str | None = None
        messages: list[Any] = kwargs.get("messages", [])
        params: dict[str, Any] = {}
        provider: str | None = None

        # Positional dispatch
        if len(args) == 3:
            # (model_id, messages, params)
            model_to_call = args[0]
            messages = args[1] or []
            params = dict(args[2] or {})
        elif len(args) == 4:
            # (provider, model, messages, params) — tests use this form
            provider = args[0]
            model = args[1]
            messages = args[2] or []
            params = dict(args[3] or {})
            model_to_call = f"{provider}/{model}"
        elif "provider" in kwargs and "model" in kwargs:
            provider = kwargs.get("provider")
            model_to_call = f"{kwargs.get('provider')}/{kwargs.get('model')}"
            messages = kwargs.get("messages", messages)
            params = dict(kwargs.get("params", {}))
        elif "model" in kwargs:
            model_to_call = kwargs.get("model")
            messages = kwargs.get("messages", messages)
            params = dict(kwargs.get("params", {}))
        else:
            # Fallback: try first positional arg as model id
            if args:
                model_to_call = args[0]
                if len(args) > 1:
                    messages = args[1] or messages
                if len(args) > 2:
                    params = dict(args[2] or {})

        # Merge any remaining kwargs into params (exclude reserved names)
        for k, v in kwargs.items():
            if k in ("model", "provider", "messages", "params"):
                continue
            params[k] = v

        if model_to_call is None:
            raise TypeError("_litellm_call missing model identifier")

        # Infer provider from model ID if not explicitly provided
        if provider is None and model_to_call and "/" in model_to_call:
            possible_provider = model_to_call.split("/", 1)[0]
            if possible_provider in PROVIDER_CATALOGUE:
                provider = possible_provider

        # Handle OpenAI-compatible custom providers
        if provider and provider.startswith("openai_compatible"):
            cfg = PROVIDER_CATALOGUE.get("openai_compatible", {})
            litellm_provider = cfg.get("litellm_provider", "custom_openai")

            # Extract endpoint ID and actual model name
            if provider == "openai_compatible":
                # Legacy single endpoint
                endpoint_id = None
                actual_model = model_to_call.split("/", 1)[-1] if model_to_call else ""
            else:
                # New format: openai_compatible_{endpoint_id}
                parts = provider.split("_", 2)
                endpoint_id = parts[2] if len(parts) > 2 else None
                actual_model = model_to_call.split("/", 1)[-1] if model_to_call else ""

            # Construct model ID with the litellm provider prefix
            model_to_call = f"{litellm_provider}/{actual_model}"

            # Get endpoint-specific configuration
            if endpoint_id:
                endpoint = settings.get_openai_compatible_endpoint(endpoint_id)
                if endpoint:
                    params["api_base"] = endpoint.get("base_url")
                    if endpoint.get("api_key"):
                        params["api_key"] = endpoint.get("api_key")
                    if endpoint.get("streaming", True):
                        params["stream"] = True
            elif settings.openai_compatible_api_base:
                # Legacy single endpoint
                params["api_base"] = settings.openai_compatible_api_base
                if settings.openai_compatible_api_key:
                    params["api_key"] = settings.openai_compatible_api_key
                if settings.openai_compatible_streaming:
                    params["stream"] = True

        # If the model seems unqualified, leave callers to provide proper ids.
        # The tests expect provider/model concatenation when provider is
        # supplied explicitly (handled above).
        res = acompletion(
            model=model_to_call, messages=messages, timeout=settings.llm_timeout, **params
        )
        # Some litellm shims may return an awaitable or a plain object; handle both.
        if asyncio.iscoroutine(res) or hasattr(res, "__await__"):
            return await res
        return res

    async def _collect_streaming_response(
        self,
        async_iterable: Any,
        provider: str,
        model: Any,
        latency: float,
        strategy: str,
        original_provider: str | None,
    ) -> dict[str, Any]:
        """Collect streaming response chunks into a single response."""
        chunks = []
        async for chunk in async_iterable:
            chunks.append(chunk)

        if not chunks:
            return {"error": "No response from provider"}

        combined_content = ""
        first_chunk = chunks[0]
        last_chunk = chunks[-1]
        completion_tokens = 0
        finish_reason = "stop"

        for chunk in chunks:
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta
                if delta and hasattr(delta, "content") and delta.content:
                    combined_content += delta.content
                # Try to get finish_reason from chunk
                if hasattr(choice, "finish_reason") and choice.finish_reason:
                    finish_reason = choice.finish_reason
                # Try to get usage info if available
                if hasattr(chunk, "usage") and chunk.usage:
                    completion_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0

        # Get model name
        model_name = ""
        if hasattr(first_chunk, "model"):
            model_name = first_chunk.model
        elif hasattr(model, "model_id"):
            model_name = model.model_id
        else:
            model_name = "unknown"

        # Get created timestamp
        created = 0
        if hasattr(first_chunk, "created"):
            created = first_chunk.created

        result = {
            "id": f"chatcmpl-{getattr(first_chunk, 'id', 'internal')}",
            "created": created,
            "model": model_name,
            "object": "chat.completion",
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": combined_content,
                    },
                }
            ],
            "usage": {
                "completion_tokens": completion_tokens,
                "prompt_tokens": 0,
                "total_tokens": completion_tokens,
            },
        }

        return self._format_response(result, provider, model, latency, strategy, original_provider)

    # ══════════════════════════════════════════════════════════════════════════
    # Fallback chain construction
    # ══════════════════════════════════════════════════════════════════════════

    def _fallback_chain(
        self,
        primary: RouteDecision,
        capability: str,
        opts: RoutingOptions,
    ):
        """
        Generator yielding (provider, ModelRecord) pairs in order:
          1. Primary selection
          2. Remaining cloud providers sorted by score
          3. Ollama (only if enabled and all cloud exhausted)
        """
        yielded: set[str] = set()

        # Primary
        yield primary.provider, primary.model
        yielded.add(primary.provider)

        # Cloud fallbacks
        excluded = set(opts.excluded_providers) | {"ollama"}
        scored = self.quota.scored_providers(available_only=True, exclude=list(excluded))
        for provider, _ in scored:
            if provider in yielded:
                continue
            model = self.discovery.get_best_model(provider, capability)
            if model:
                yield provider, model
                yielded.add(provider)

        # Ollama — absolute last resort
        if settings.enable_ollama_fallback and "ollama" not in opts.excluded_providers:
            model = self.discovery.get_best_model("ollama", capability)
            if model:
                logger.info("All cloud providers exhausted — falling back to Ollama")
                yield "ollama", model

    # ══════════════════════════════════════════════════════════════════════════
    # Provider selection
    # ══════════════════════════════════════════════════════════════════════════

    def _select(self, capability: str, opts: RoutingOptions) -> RouteDecision | None:
        """Select the highest priority available provider (deterministic by CLOUD_PRIORITY_ORDER)."""
        excluded = set(opts.excluded_providers) | {"ollama"}

        for provider in CLOUD_PRIORITY_ORDER:
            if provider in excluded:
                continue
            if opts.preferred_providers and provider not in opts.preferred_providers:
                continue
            state = self.quota.states.get(provider)
            if not state or not state.is_available():
                continue
            model = self.discovery.get_best_model(
                provider, capability, prefer_free=opts.free_tier_only
            )
            if model is None:
                continue
            # Found the highest priority available provider
            return RouteDecision(
                provider=provider,
                model=model,
                strategy=opts.strategy,
                score=1.0,
            )

        return None

    # ══════════════════════════════════════════════════════════════════════════
    # Task-type detection
    # ══════════════════════════════════════════════════════════════════════════

    def _detect_task_type(self, request_data: dict[str, Any]) -> TaskType:
        raw = request_data.get("task_type", "").lower()
        if raw:
            for tt in TaskType:
                if tt.value == raw:
                    return tt

        if "input" in request_data and "messages" not in request_data:
            return TaskType.EMBEDDINGS

        if "image_url" in request_data or "image_base64" in request_data:
            return TaskType.VISION_UNDERSTANDING

        for msg in request_data.get("messages", []):
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        return TaskType.VISION_UNDERSTANDING
            elif isinstance(content, str) and "data:image" in content:
                return TaskType.VISION_UNDERSTANDING

        if request_data.get("tools"):
            return TaskType.FUNCTION_CALLING

        return TaskType.CHAT_COMPLETION

    # ══════════════════════════════════════════════════════════════════════════
    # Utilities
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _format_response(
        raw: Any,
        provider: str,
        model: ModelRecord,
        latency_ms: float,
        strategy: str,
        original_provider: str,
    ) -> dict[str, Any]:
        """Normalise a litellm response into a standard dict."""
        # Ensure we always return a plain `dict` (avoid MappingProxyType / readonly maps)
        if isinstance(raw, dict):
            resp = dict(raw)
        else:
            resp = dict(raw.model_dump() if hasattr(raw, "model_dump") else vars(raw))

        resp["routing_metadata"] = {
            "provider": provider,
            "model": model.model_id,
            "strategy": strategy,
            "latency_ms": round(float(latency_ms), 1),
            "cache_hit": False,
            "cost_usd": 0.0,
            "fallback": provider != original_provider,
            "original_provider": original_provider,
        }
        return resp

    @staticmethod
    def _is_daily_limit(exc: Exception) -> bool:
        """Detect if error indicates quota exhaustion or account balance issues."""
        msg = str(exc).lower()
        # Comprehensive list of quota/balance exhaustion indicators
        exhaustion_indicators = (
            "tokens per day",
            "tpd",
            "credit balance",
            "depleted",
            "daily limit",
            "daily quota",
            "no credits",
            "key limit exceeded",
            "insufficient balance",
            "insufficient funds",
            "not have permission",
            "permission",
            "permission_denied",
            "payment required",
            "quota exceeded",
            "quota exhausted",
            "rate limit",
            "billing",
            "account balance",
            "doesn't have any credits",
            "doesn't have any licenses",
            "newly created team",
            "purchase those on",
            "upgrade your plan",
            "plan limits",
            "monthly limit",
            "usage limit",
            "max requests",
            "threshold exceeded",
        )
        return any(k in msg for k in exhaustion_indicators)

    @staticmethod
    def _parse_retry_delay(exc: Exception) -> float | None:
        for attr in ("retry_info", "retry_delay", "retry_after"):
            try:
                val = getattr(exc, attr, None)
                if isinstance(val, (int, float)):
                    return float(val)
            except Exception:
                pass
        try:
            headers = getattr(getattr(exc, "response", None), "headers", None) or {}
            ra = headers.get("Retry-After") or headers.get("retry-after")
            if ra:
                return float(ra)
        except Exception:
            pass
        m = re.search(r"(?:retry[- ]?after|retry in)\D{0,10}(\d+)", str(exc), re.I)
        if m:
            return float(m.group(1))
        return None

    @staticmethod
    def _extract_prompt_text(request_data: dict[str, Any]) -> str:
        msgs = request_data.get("messages", [])
        if msgs:
            last = msgs[-1]
            content = last.get("content", "") if isinstance(last, dict) else ""
            return str(content)[:200]
        return str(request_data.get("input", ""))[:200]

    # ══════════════════════════════════════════════════════════════════════════
    # Observability
    # ══════════════════════════════════════════════════════════════════════════

    def get_stats(self) -> dict[str, Any]:
        return {
            "providers": self.quota.get_stats(),
            "cache": self.cache.stats,
            "models_per_provider": {
                p: len(self.discovery.get_models(p)) for p in PROVIDER_CATALOGUE
            },
        }
