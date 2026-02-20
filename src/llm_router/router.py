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
        – have the required capability
        – have remaining RPD quota
        – not in cooldown / circuit-broken
        – Ollama explicitly EXCLUDED from this list
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

try:
    import litellm  # type: ignore[import]
    from litellm import APIConnectionError as LiteLLMConnectionError  # type: ignore[import]
    from litellm import AuthenticationError as LiteLLMAuthError  # type: ignore[import]
    from litellm import RateLimitError as LiteLLMRateLimit  # type: ignore[import]
    from litellm import Timeout as LiteLLMTimeout  # type: ignore[import]
    from litellm import acompletion  # type: ignore[import]

    _LITELLM_AVAILABLE = True
except ImportError:
    _LITELLM_AVAILABLE = False
    litellm = None  # type: ignore[assignment]
    acompletion = None  # type: ignore[assignment]
    LiteLLMRateLimit = LiteLLMTimeout = LiteLLMConnectionError = LiteLLMAuthError = Exception  # type: ignore[assignment]

from llm_router.cache import ResponseCache  # type: ignore[import]
from llm_router.config import (  # type: ignore[import]
    CLOUD_PRIORITY_ORDER,
    PROVIDER_CATALOGUE,
    settings,
)
from llm_router.discovery import CapabilityDiscovery  # type: ignore[import]
from llm_router.models import (  # type: ignore[import]
    CachePolicy,
    ModelRecord,
    ProviderState,
    RouteDecision,
    RoutingOptions,
    RoutingStrategy,
    TaskType,
)
from llm_router.quota import QuotaManager  # type: ignore[import]

if _LITELLM_AVAILABLE and settings.verbose_litellm:
    # Try to enable litellm verbose mode if supported. Some litellm versions
    # expose a function `set_verbose(bool)`; others may expose an attribute.
    try:
        sv = getattr(litellm, "set_verbose", None)
        if callable(sv):
            sv(True)  # type: ignore[call-arg]
        else:
            try:
                litellm.set_verbose = True  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        # Don't fail startup if the litellm debug toggle is unavailable.
        logging.getLogger(__name__).debug("Could not enable litellm verbose mode; continuing")


# Helper to extract plain text from varied litellm chunk shapes. Keep this
# conservative and defensive to avoid returning unpickled/internal objects.
def _extract_chunk_text(item: Any) -> str:
    try:
        if item is None:
            return ""
        if isinstance(item, str):
            return item
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        if isinstance(item, dict):
            # common litellm shapes: {'delta': {'content': '...'}}
            d = item.get("delta") or item.get("message") or item
            if isinstance(d, dict):
                if "content" in d:
                    return d["content"]
                if "text" in d:
                    return d["text"]
            if "choices" in item and isinstance(item["choices"], list) and item["choices"]:
                first = item["choices"][0]
                if isinstance(first, dict):
                    msg = first.get("message") or first.get("delta") or first
                    if isinstance(msg, dict) and "content" in msg:
                        return msg["content"]
                    if "text" in first:
                        return first["text"]
            if "content" in item:
                return item["content"]
        # objects: try model_dump or attributes
        # Prefer using getattr to avoid static analysis warnings where `item`
        # may be typed as a dict. If a pydantic model exposes `model_dump`,
        # call it and re-run extraction on the result.
        md = getattr(item, "model_dump", None)
        if callable(md):
            try:
                return _extract_chunk_text(md())
            except Exception:
                pass
        if hasattr(item, "message"):
            try:
                return _extract_chunk_text(getattr(item, "message"))
            except Exception:
                pass
        if hasattr(item, "choices"):
            try:
                choices = getattr(item, "choices")
                if isinstance(choices, (list, tuple)) and choices:
                    return _extract_chunk_text(choices[0])
            except Exception:
                pass
    except Exception:
        pass
    return str(item)


# If litellm is present and internals logging is disabled, wrap `acompletion`
# so we only surface safe plain-text chunks or JSON-serialisable dicts.
if _LITELLM_AVAILABLE and not settings.verbose_litellm_internals and acompletion is not None:
    _orig_acompletion = acompletion

    async def _safe_acompletion(*args: Any, **kwargs: Any) -> Any:
        res = await _orig_acompletion(*args, **kwargs)
        # If the response is an async iterable, return an async generator that
        # yields plain text chunks (strings). Otherwise, normalise to text or
        # a simple dict/str safe for JSON encoding.
        if hasattr(res, "__aiter__"):

            async def _aiter():
                async for item in res:
                    yield _extract_chunk_text(item)

            return _aiter()

        # Non-iterable: try to extract textual content or a serialisable dict
        extracted = _extract_chunk_text(res)
        # If extraction returned something that looks like JSON text, attempt to
        # parse it so downstream code gets dicts where possible.
        try:
            import json

            parsed = json.loads(extracted)
            return parsed
        except Exception:
            return extracted

    acompletion = _safe_acompletion  # type: ignore[assignment]

# If internals logging is explicitly enabled, provide a lightweight debug wrapper
# that logs raw litellm responses (or notes streaming) at DEBUG level. This is
# separate from the safe wrapper and should only be active when administrators
# intentionally enable verbose internals via VERBOSE_LITELLM_INTERNALS.
if _LITELLM_AVAILABLE and settings.verbose_litellm_internals and acompletion is not None:
    _orig_acomp_dbg = acompletion

    async def _debug_acompletion(*args: Any, **kwargs: Any) -> Any:
        try:
            res = await _orig_acomp_dbg(*args, **kwargs)
        except Exception as e:
            logger.debug("litellm acompletion raised: %s", type(e).__name__)
            raise
        # If streaming result, avoid consuming it; just log that it's streaming.
        try:
            if hasattr(res, "__aiter__"):
                logger.debug("litellm returned async iterable (streaming) for model call")
            else:
                # Log shallow repr; admins can enable internals and see details.
                logger.debug("litellm raw response: %s", repr(res))
        except Exception:
            # Never allow logging to break execution
            pass
        return res

    acompletion = _debug_acompletion  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Keyword patterns for task-type detection
_VISION_CONTENT_INDICATORS: frozenset = frozenset(["image_url", "data:image"])
_EMBEDDING_TASK_TYPES: frozenset = frozenset(["embeddings", "embedding"])

# Strategy weight presets  (w_quota, w_latency, w_quality, w_errors)
# Use explicit dict literals to keep static analysis happy and concise.
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
    RoutingStrategy.CONTEXT_LENGTH: {
        "w_quota": 0.20,
        "w_latency": 0.20,
        "w_quality": 0.30,
        "w_errors": 0.10,
    },
    RoutingStrategy.VISION_CAPABLE: {
        "w_quota": 0.20,
        "w_latency": 0.20,
        "w_quality": 0.30,
        "w_errors": 0.10,
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

# Map from our provider name to litellm's expected prefix
_LITELLM_PROVIDER_PREFIX = {
    "together": "together_ai",
    "dashscope": "ali_qwen",  # dashscope can also be ali_qwen
    "huggingface": "huggingface",
    "groq": "groq",
    "mistral": "mistral",
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "gemini",
    "openrouter": "openrouter",
    "deepseek": "deepseek",
    "xai": "xai",
    "ollama": "ollama",
    "cohere": "cohere",
}


class IntelligentRouter:
    """
    Top-level router. Instantiate once and reuse across the application lifetime.

    Entry point: ``await router.route(request_data)``
    """

    def __init__(self) -> None:
        self.quota = QuotaManager()
        self.discovery = CapabilityDiscovery(quota_manager=self.quota)
        self.cache = ResponseCache()
        self._initialised = False

    # ── Lifecycle ──────────────────────────────────────────────────────────────

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
            return
        os.environ.setdefault("LITELLM_LOG", "ERROR")
        for _provider, cfg in PROVIDER_CATALOGUE.items():
            env_name = cfg.get("api_key_env")
            if env_name:
                val = os.getenv(env_name)
                if val:
                    os.environ[env_name] = val
        # Tell litellm about custom base URLs (dashscope, ollama, huggingface, etc.)
        for provider, cfg in PROVIDER_CATALOGUE.items():
            base_url = cfg.get("base_url")
            if base_url:
                os.environ[f"{provider.upper()}_BASE_URL"] = base_url
                os.environ[f"{provider.upper()}_API_BASE"] = base_url

    # ══════════════════════════════════════════════════════════════════════════
    # Primary routing entry point
    # ══════════════════════════════════════════════════════════════════════════

    async def route(
        self,
        request_data: dict[str, Any],
        routing_options: RoutingOptions | None = None,
    ) -> Any:
        """Route a request to the optimal provider.

        ``request_data`` must contain at minimum one of: messages, input or
        image data. The return value may be a dict (synchronous response) or
        an async iterable for streaming completions — callers should handle
        both shapes. The return type is therefore annotated as Any.
        """
        await self.discovery.refresh_if_stale()

        opts = routing_options or RoutingOptions()
        task_type = self._detect_task_type(request_data)

        # ── Cache lookup ───────────────────────────────────────────────────────
        if opts.cache_policy == CachePolicy.ENABLED:
            messages = request_data.get("messages", [])
            cache_key = self.cache.make_key(
                messages,
                request_data.get("model"),
                request_data.get("temperature"),
            )
            cached = self.cache.get_exact(cache_key)
            if cached is not None:
                cached["routing_metadata"]["cache_hit"] = True
                return cached
        else:
            cache_key = ""  # type: ignore[assignment]

        v = task_type.value
        # ── Dispatch ───────────────────────────────────────────────────────────
        if v == TaskType.EMBEDDINGS.value:
            result = await self._route_embedding(request_data, opts)
        elif v in (
            TaskType.VISION_CLASSIFY.value,
            TaskType.VISION_DETECT.value,
            TaskType.VISION_OCR.value,
            TaskType.VISION_QA.value,
            TaskType.VISION_CAPTION.value,
            TaskType.VISION_UNDERSTANDING.value,
        ):
            result = await self._route_vision(request_data, opts, task_type)
        else:
            result = await self._route_text(request_data, opts)

        # ── Cache store ────────────────────────────────────────────────────────
        if opts.cache_policy != CachePolicy.DISABLED and cache_key:
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
                "content": [{"type": "text", "text": prompt}] + image_content,
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
            TaskType.VISION_CLASSIFY: "Classify this image. Return the top categories with confidence scores.",
            TaskType.VISION_DETECT: "Detect all objects in this image. Provide bounding box descriptions and labels.",
            TaskType.VISION_OCR: "Extract all text from this image. Preserve the layout as closely as possible.",
            TaskType.VISION_QA: request_data.get("question")
            or "Answer any question about this image.",
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
                        "latency_ms": round(latency),
                        "strategy": opts.strategy,
                        "cache_hit": False,
                    },
                }
            except Exception as exc:
                logger.warning("Embedding %s/%s failed: %s", provider, model.model_id, exc)
                self.quota.mark_error(provider)

        raise RuntimeError("All embedding providers failed")

    async def _call_embedding(self, model_id: str, text: Any) -> list[list[float]]:
        if not _LITELLM_AVAILABLE or not hasattr(litellm, "aembedding"):
            raise ImportError("litellm required: pip install litellm")
        texts = [text] if isinstance(text, str) else list(text)
        aembed = getattr(litellm, "aembedding")
        if not callable(aembed):
            raise ImportError("litellm aembedding not callable")
        response = await aembed(model=model_id, input=texts)  # type: ignore[call-arg]
        data = getattr(response, "get", None)
        if callable(data):
            items = response.get("data", [])
        else:
            # If response is an object with attribute `data`
            items = getattr(response, "data", [])
        embeddings: list[list[float]] = []
        for item in items:
            try:
                if isinstance(item, dict):
                    emb = item.get("embedding")
                else:
                    emb = getattr(item, "embedding", None)
                if emb is None:
                    continue
                if isinstance(emb, (list, tuple)):
                    embeddings.append([float(x) for x in emb])
                else:
                    embeddings.append([float(emb)])
            except Exception:
                continue
        return embeddings

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

    async def _try_provider(
        self,
        provider: str,
        model: ModelRecord,
        messages: list[dict[str, Any]],
        extra_params: dict[str, Any],
        start_time: float,
        original_provider: str,
        strategy: str,
    ) -> Any | None:
        """
        Attempt a single provider+model.  Handles:
          - RPM rate limits (exponential back-off, up to MAX_RETRIES)
          - Daily limits (long cooldown, return None immediately)
          - Network errors (short cooldown, return None)
        Returns a response dict on success, None to signal "try next".
        """
        if not self.quota.can_accept(provider):
            # Provider currently blocked by quota gates. In order to ensure
            # we still detect permanent model-not-found errors (so discovery
            # can prune unavailable models) attempt one try — the exception
            # handlers will correctly mark rate-limits and cooldowns.
            logger.debug(
                "Provider %s currently rate-limited; attempting once to detect permanent model errors",
                provider,
            )

        params = {k: v for k, v in extra_params.items() if v is not None}
        # Explicitly pass API key if configured
        provider_cfg = PROVIDER_CATALOGUE.get(provider, {})
        api_key_env = provider_cfg.get("api_key_env")
        # Only include an explicit api_key param if the environment variable is
        # actually set. Passing None slips through to litellm and causes
        # misleading authentication errors.
        if api_key_env:
            val = os.getenv(api_key_env)
            if val:
                params["api_key"] = val

        # `last_exc` used to hold the last exception for debugging; remove the
        # initial assignment to satisfy linters when unused.

        stream_mode = params.get("stream", False)

        # Streaming path: delegate to helper for clarity and reduced complexity.
        if stream_mode:
            return await self._handle_stream_call(
                provider=provider,
                model=model,
                messages=messages,
                params=params,
                start_time=start_time,
            )

        # Non-streaming path: original retry logic with exponential backoff
        return await self._handle_nonstream_calls(
            provider=provider,
            model=model,
            messages=messages,
            params=params,
            start_time=start_time,
            strategy=strategy,
            original_provider=original_provider,
        )

    async def _handle_nonstream_calls(
        self,
        provider: str,
        model: ModelRecord,
        messages: list[dict[str, Any]],
        params: dict[str, Any],
        start_time: float,
        strategy: str,
        original_provider: str,
    ) -> dict[str, Any] | None:
        """Handle the retry loop for non-streaming calls. Returns a response dict or None."""
        for attempt in range(1, settings.max_retries + 1):
            try:
                self.quota.consume(provider)
                response = await self._litellm_call(provider, model.model_id, messages, params)
                latency = (time.monotonic() - start_time) * 1000
                self.quota.record_latency(provider, latency)
                return self._format_response(
                    response, provider, model, latency, strategy, original_provider
                )

            except LiteLLMRateLimit as e:
                msg = str(e).lower()
                if any(
                    k in msg
                    for k in (
                        "404",
                        "model_not_found",
                        "does not exist",
                        "was removed",
                        "removed on",
                        "no longer available",
                    )
                ):
                    logger.warning("%s/%s: model not available — %s", provider, model.model_id, msg)
                    try:
                        self.discovery.remove_model(provider, model.model_id)
                    except Exception:
                        pass
                    return None

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
                self.quota.mark_error(provider)
                logger.warning("%s: connection/timeout — %s", provider, e)
                return None

            except LiteLLMAuthError as e:
                self.quota.mark_auth_failed(provider)
                logger.error("%s: authentication failure — %s", provider, e)
                return None

            except Exception as e:
                msg = str(e).lower()
                if any(
                    k in msg
                    for k in (
                        "404",
                        "model_not_found",
                        "does not exist",
                        "was removed",
                        "removed on",
                        "no longer available",
                    )
                ):
                    logger.warning("%s/%s: model not available — %s", provider, model.model_id, msg)
                    try:
                        self.discovery.remove_model(provider, model.model_id)
                    except Exception:
                        pass
                    return None
                self.quota.mark_error(provider)
                logger.warning("%s: unexpected error — %s", provider, e)
                return None

        return None

    async def _handle_stream_call(
        self,
        provider: str,
        model: ModelRecord,
        messages: list[dict[str, Any]],
        params: dict[str, Any],
        start_time: float,
    ) -> Any | None:
        """Handle a single streaming-capable call. Returns an async iterable or None on failure.

        This extracts the streaming-path logic out of _try_provider to keep
        that function smaller and easier to reason about.
        """
        try:
            self.quota.consume(provider)
            lit = self._litellm_call(provider, model.model_id, messages, params)
            if asyncio.iscoroutine(lit):
                res = await lit
            else:
                res = lit

            latency = (time.monotonic() - start_time) * 1000
            self.quota.record_latency(provider, latency)

            if hasattr(res, "__aiter__"):
                return res  # type: ignore[return-value]

            async def single():
                yield res

            return single()

        except LiteLLMRateLimit as e:
            msg = str(e).lower()
            if any(
                k in msg
                for k in (
                    "404",
                    "model_not_found",
                    "does not exist",
                    "was removed",
                    "removed on",
                    "no longer available",
                )
            ):
                try:
                    self.discovery.remove_model(provider, model.model_id)
                except Exception:
                    pass
                return None
            self.quota.mark_rate_limited(provider, 0)
            self.quota.mark_error(provider)
            return None
        except (LiteLLMTimeout, LiteLLMConnectionError) as e:
            self.quota.mark_error(provider)
            logger.warning("%s: connection/timeout — %s", provider, e)
            return None
        except LiteLLMAuthError as e:
            self.quota.mark_auth_failed(provider)
            logger.error("%s: authentication failure — %s", provider, e)
            return None
        except Exception as e:
            msg = str(e).lower()
            if any(
                k in msg
                for k in (
                    "404",
                    "model_not_found",
                    "does not exist",
                    "was removed",
                    "removed on",
                    "no longer available",
                )
            ):
                try:
                    self.discovery.remove_model(provider, model.model_id)
                except Exception:
                    pass
                return None
            self.quota.mark_error(provider)
            logger.warning("%s: unexpected error — %s", provider, e)
            return None

    async def _litellm_call(
        self, provider: str, model_id: str, messages: list[Any], params: dict[str, Any]
    ) -> Any:
        if not _LITELLM_AVAILABLE:
            raise ImportError("litellm not installed — run: pip install litellm")

        # Ensure correct litellm prefix. litellm expects a provider prefix like
        # "openrouter/google/gemma-3n...". Models stored in discovery may already
        # include a slash (e.g. "google/gemma-3n...") — if that happens and the
        # provider prefix is missing we prepend it so litellm receives a fully
        # qualified model id.
        prefix = _LITELLM_PROVIDER_PREFIX.get(provider, provider)
        if "/" not in model_id:
            litellm_model = f"{prefix}/{model_id}"
        else:
            # If the model id already contains a provider segment, ensure the
            # configured prefix is present; otherwise prepend it. This fixes
            # errors like: "LLM Provider NOT provided" when model="google/xxx"
            # is passed to openrouter (should be "openrouter/google/xxx").
            if model_id.startswith(f"{prefix}/") or prefix in model_id.split("/"):
                litellm_model = model_id
            else:
                litellm_model = f"{prefix}/{model_id}"
        logger.debug(
            "Calling litellm with model=%s (provider=%s) => %s",
            model_id,
            provider,
            litellm_model,
        )

        # If streaming requested, attempt to return an async iterable from
        # litellm without eagerly waiting for the full result. Some litellm
        # versions return an async generator for streaming completions.
        try:
            # Ensure acompletion is available (it may be wrapped or disabled)
            local_acompletion = acompletion
            if not callable(local_acompletion):
                raise ImportError("litellm acompletion not available")

            coro = local_acompletion(  # type: ignore[union-attr]
                model=litellm_model,
                messages=messages,
                timeout=settings.llm_timeout,
                **params,
            )
            # If caller asked for stream we prefer to return an async
            # iterable. If the call returns a coroutine that resolves to an
            # async iterable, await it; otherwise if it's already an async
            # iterable, return it directly.
            if params.get("stream"):
                # Await if it's a coroutine to get the iterable
                if asyncio.iscoroutine(coro):
                    result = await coro
                    return result
                return coro
            # Non-streaming: await the coroutine and return result
            return await coro
        except Exception:
            # Surface exceptions to caller to allow fallback handling
            raise

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
        """Score eligible providers and pick one via weighted random."""
        logger.debug("SELECT CALL: capability=%s opts=%s", capability, opts)

        weights = _STRATEGY_WEIGHTS.get(opts.strategy, _STRATEGY_WEIGHTS[RoutingStrategy.AUTO])
        excluded = set(opts.excluded_providers) | {"ollama"}

        candidates: dict[str, float] = {}
        best_models: dict[str, ModelRecord] = {}

        for provider in CLOUD_PRIORITY_ORDER:
            logger.debug("considering provider=%s", provider)

            if provider in excluded:
                logger.debug("skipping %s (excluded)", provider)
                continue
            if opts.preferred_providers and provider not in opts.preferred_providers:
                logger.debug("skipping %s (not in preferred list)", provider)
                continue

            api_key_env = PROVIDER_CATALOGUE.get(provider, {}).get("api_key_env")
            if api_key_env and not os.getenv(api_key_env):
                logger.debug("skipping %s (no API key configured: %s)", provider, api_key_env)
                continue

            if not self.quota.states.get(provider, ProviderState("", 0, 0)).is_available():
                logger.debug("skipping %s (not available)", provider)
                continue

            model = self.discovery.get_best_model(
                provider, capability, prefer_free=opts.free_tier_only
            )
            if model is None:
                logger.debug("skipping %s (no model for capability=%s)", provider, capability)
                continue

            if capability != "embedding" and "embedding" in model.capabilities:
                logger.debug("skipping %s (model is embedding-only)", provider)
                continue
            if capability == "embedding" and "embedding" not in model.capabilities:
                logger.debug("skipping %s (model lacks embedding capability)", provider)
                continue

            base_score = self.quota.score(provider, **weights)

            if opts.strategy == RoutingStrategy.CONTEXT_LENGTH:
                model_score = self._score_context_length(model)
                final_score = base_score * 0.4 + model_score * 0.6
            elif opts.strategy == RoutingStrategy.VISION_CAPABLE:
                model_score = self._score_vision_capability(model)
                final_score = base_score * 0.4 + model_score * 0.6
            else:
                final_score = base_score

            logger.debug("provider=%s score=%s (base=%s)", provider, final_score, base_score)

            if final_score > 0:
                candidates[provider] = final_score
                best_models[provider] = model

        chosen = _weighted_choice(candidates)
        if chosen is None:
            logger.debug("SELECT FAILED: no candidates for %s", capability)
            return None

        model = best_models.get(chosen)
        if model is None:
            model = self.discovery.get_best_model(
                chosen, capability, prefer_free=opts.free_tier_only
            )
        if model is None:
            logger.debug("SELECT FAILED: best model None for %s %s", chosen, capability)
            return None

        logger.debug("SELECT RESULT: chosen=%s model=%s", chosen, model.full_id)

        return RouteDecision(
            provider=chosen,
            model=model,
            strategy=opts.strategy,
            score=candidates[chosen],
        )

    def _score_context_length(self, model: ModelRecord) -> float:
        """Score model based on context window size. Higher context = higher score."""
        max_context = 1_000_000
        normalized = min(model.context_window / max_context, 1.0)
        return normalized

    def _score_vision_capability(self, model: ModelRecord) -> float:
        """Score model based on vision capability. Vision models get higher score."""
        if model.supports_vision:
            return 1.0
        if "vision" in model.capabilities:
            return 0.9
        return 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # Task-type detection
    # ══════════════════════════════════════════════════════════════════════════

    def _detect_task_type(self, request_data: dict[str, Any]) -> TaskType:
        raw = request_data.get("task_type", "")
        if isinstance(raw, TaskType):
            return raw

        raw_str = str(raw).lower()
        if raw_str:
            for tt in TaskType:
                if tt.value == raw_str:
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
    def _format_response(  # type: ignore
        raw: Any,
        provider: str,
        model: ModelRecord,
        latency_ms: float,
        strategy: str,
        original_provider: str,
    ) -> dict[str, Any]:
        """Normalise a litellm response into a standard dict."""
        # Normalize the raw response into a dict[str, Any] safely. We avoid
        # returning objects that might contain non-serialisable internals.
        resp_dict: dict[str, Any] = {}

        if isinstance(raw, dict):
            for k, v in raw.items():
                try:
                    resp_dict[str(k)] = v
                except Exception:
                    resp_dict[str(k)] = str(v)

        else:
            md = getattr(raw, "model_dump", None)
            if callable(md):
                try:
                    d = md()
                    # d may be a mapping-like object or a pydantic model result;
                    # iterate via items() when available, otherwise fall back to vars().
                    # `d` may be a mapping-like object or a simple namespace.
                    # Attempt to iterate mapping-like contents safely.
                    for k, v in self._iter_items(d):
                        resp_dict[str(k)] = v
                except Exception:
                    resp_dict["result"] = str(raw)
            else:
                # For arbitrary objects: prefer items() if present, else vars(),
                # otherwise provide a string representation.
                for k, v in self._iter_items(raw):
                    resp_dict[str(k)] = v

        # Attach routing metadata with predictable types
        resp_dict["routing_metadata"] = {
            "provider": provider,
            "model": model.model_id,
            "strategy": strategy,
            "latency_ms": round(latency_ms),
            "cache_hit": False,
            "cost_usd": 0.0,
            "fallback": provider != original_provider,
            "original_provider": original_provider,
        }

        return resp_dict

    @staticmethod
    def _is_daily_limit(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(
            k in msg
            for k in (
                "tokens per day",
                "tpd",
                "credit balance",
                "depleted",
                "daily limit",
            )
        )

    @staticmethod
    def _iter_items(obj: object):
        """Yield (key, value) pairs for mapping-like or object-like inputs.

        This helper avoids direct attribute access that static analysers may
        flag and centralises the try/except logic used when coercing objects
        into dicts for JSON responses.
        """
        try:
            if hasattr(obj, "items"):
                for k, v in obj.items():  # type: ignore[attr-defined]
                    yield k, v
                return
        except Exception:
            pass
        try:
            for k, v in vars(obj).items():
                yield k, v
            return
        except Exception:
            pass
        # Fallback to nothing
        return

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
            return str(content)[:200]  # type: ignore[misc]
        return str(request_data.get("input", ""))[:200]  # type: ignore[misc]

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
