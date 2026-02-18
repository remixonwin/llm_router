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
import types
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple

try:
    import litellm  # type: ignore[import]
    from litellm import acompletion  # type: ignore[import]
    from litellm import APIConnectionError as LiteLLMConnectionError  # type: ignore[import]
    from litellm import RateLimitError as LiteLLMRateLimit  # type: ignore[import]
    from litellm import Timeout as LiteLLMTimeout  # type: ignore[import]
    from litellm import AuthenticationError as LiteLLMAuthError  # type: ignore[import]
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
    TASK_CAPABILITY_MAP,
    settings,
)
from llm_router.discovery import CapabilityDiscovery  # type: ignore[import]
from llm_router.models import (  # type: ignore[import]
    CacheKey,
    CachePolicy,
    ModelRecord,
    ProviderState,
    RouteDecision,
    RoutingMetadata,
    RoutingOptions,
    RoutingStrategy,
    TaskType,
)
from llm_router.quota import QuotaManager  # type: ignore[import]
 
if _LITELLM_AVAILABLE and settings.verbose_litellm:
    litellm.set_verbose = True  # type: ignore[union-attr]

logger = logging.getLogger(__name__)

# Keyword patterns for task-type detection
_VISION_CONTENT_INDICATORS: frozenset = frozenset(["image_url", "data:image"])
_EMBEDDING_TASK_TYPES: frozenset = frozenset(["embeddings", "embedding"])

# Strategy weight presets  (w_quota, w_latency, w_quality, w_errors)
_STRATEGY_WEIGHTS: Dict[str, Dict[str, float]] = {
    RoutingStrategy.AUTO:          dict(w_quota=0.50, w_latency=0.25, w_quality=0.15, w_errors=0.10),
    RoutingStrategy.COST_OPTIMIZED:dict(w_quota=0.80, w_latency=0.10, w_quality=0.05, w_errors=0.05),
    RoutingStrategy.QUALITY_FIRST: dict(w_quota=0.10, w_latency=0.20, w_quality=0.60, w_errors=0.10),
    RoutingStrategy.LATENCY_FIRST: dict(w_quota=0.20, w_latency=0.60, w_quality=0.10, w_errors=0.10),
    RoutingStrategy.ROUND_ROBIN:   dict(w_quota=0.25, w_latency=0.25, w_quality=0.25, w_errors=0.25),
}


# ══════════════════════════════════════════════════════════════════════════════
# Helper: weighted random choice
# ══════════════════════════════════════════════════════════════════════════════

def _weighted_choice(scores: Dict[str, float]) -> Optional[str]:
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
    "dashscope": "ali_qwen", # dashscope can also be ali_qwen
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
        self.discovery = CapabilityDiscovery()
        self.quota = QuotaManager()
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
        for provider, cfg in PROVIDER_CATALOGUE.items():
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
        request_data: Dict[str, Any],
        routing_options: Optional[RoutingOptions] = None,
    ) -> Dict[str, Any]:
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
            TaskType.VISION_CLASSIFY.value, TaskType.VISION_DETECT.value, TaskType.VISION_OCR.value,
            TaskType.VISION_QA.value, TaskType.VISION_CAPTION.value, TaskType.VISION_UNDERSTANDING.value,
        ):
            result = await self._route_vision(request_data, opts, task_type)
        else:
            result = await self._route_text(request_data, opts)

        # ── Cache store ────────────────────────────────────────────────────────
        if opts.cache_policy != CachePolicy.DISABLED and cache_key:
            self.cache.set(cache_key, result,
                           prompt_text=self._extract_prompt_text(request_data))

        return result

    # ══════════════════════════════════════════════════════════════════════════
    # Text / chat routing
    # ══════════════════════════════════════════════════════════════════════════

    async def _route_text(
        self, request_data: Dict[str, Any], opts: RoutingOptions
    ) -> Dict[str, Any]:
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
                "max_tokens":  request_data.get("max_tokens"),
                "top_p":       request_data.get("top_p"),
                "stream":      request_data.get("stream", False),
                "tools":       request_data.get("tools"),
                "tool_choice": request_data.get("tool_choice"),
            },
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Vision routing
    # ══════════════════════════════════════════════════════════════════════════

    async def _route_vision(
        self, request_data: Dict[str, Any], opts: RoutingOptions, task_type: TaskType
    ) -> Dict[str, Any]:
        prompt = self._build_vision_prompt(request_data, task_type)
        image_content = self._build_image_content(request_data)

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}] + image_content}]
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

    def _build_image_content(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
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

    def _build_vision_prompt(self, request_data: Dict[str, Any], task_type: TaskType) -> str:
        prompts = {
            TaskType.VISION_CLASSIFY:    "Classify this image. Return the top categories with confidence scores.",
            TaskType.VISION_DETECT:      "Detect all objects in this image. Provide bounding box descriptions and labels.",
            TaskType.VISION_OCR:         "Extract all text from this image. Preserve the layout as closely as possible.",
            TaskType.VISION_QA:          request_data.get("question") or "Answer any question about this image.",
            TaskType.VISION_CAPTION:     "Write a detailed, descriptive caption for this image.",
            TaskType.VISION_UNDERSTANDING:"Analyse and describe this image in detail.",
        }
        return prompts.get(task_type, "Describe this image.")

    # ══════════════════════════════════════════════════════════════════════════
    # Embedding routing
    # ══════════════════════════════════════════════════════════════════════════

    async def _route_embedding(
        self, request_data: Dict[str, Any], opts: RoutingOptions
    ) -> Dict[str, Any]:
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
                    "data": [{"object": "embedding", "embedding": e, "index": i}
                             for i, e in enumerate(result)],
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

    async def _call_embedding(self, model_id: str, text: Any) -> List[List[float]]:
        if not _LITELLM_AVAILABLE:
            raise ImportError("litellm required: pip install litellm")
        texts = [text] if isinstance(text, str) else text
        response = await litellm.aembedding(model=model_id, input=texts)  # type: ignore[union-attr]
        return [item["embedding"] for item in response["data"]]

    # ══════════════════════════════════════════════════════════════════════════
    # Attempt + fallback logic
    # ══════════════════════════════════════════════════════════════════════════

    async def _attempt_with_fallback(
        self,
        primary: RouteDecision,
        messages: List[Dict[str, Any]],
        capability: str,
        opts: RoutingOptions,
        extra_params: Dict[str, Any],
    ) -> Dict[str, Any]:
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
        messages: List[Dict[str, Any]],
        extra_params: Dict[str, Any],
        start_time: float,
        original_provider: str,
        strategy: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt a single provider+model.  Handles:
          - RPM rate limits (exponential back-off, up to MAX_RETRIES)
          - Daily limits (long cooldown, return None immediately)
          - Network errors (short cooldown, return None)
        Returns a response dict on success, None to signal "try next".
        """
        if not self.quota.can_accept(provider):
            logger.debug("Skipping %s — quota gate", provider)
            return None

        params = {k: v for k, v in extra_params.items() if v is not None}
        # Explicitly pass API key if configured
        provider_cfg = PROVIDER_CATALOGUE.get(provider, {})
        api_key_env = provider_cfg.get("api_key_env")
        if api_key_env:
            params["api_key"] = os.getenv(api_key_env)

        last_exc: Optional[Exception] = None

        for attempt in range(1, settings.max_retries + 1):
            try:
                self.quota.consume(provider)
                response = await self._litellm_call(provider, model.model_id, messages, params)
                latency = (time.monotonic() - start_time) * 1000
                self.quota.record_latency(provider, latency)
                return self._format_response(response, provider, model, latency, strategy,
                                             original_provider)

            except LiteLLMRateLimit as e:
                last_exc = e
                delay = self._parse_retry_delay(e) or (settings.retry_base_delay * 2 ** attempt)
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
                last_exc = e
                self.quota.mark_error(provider)
                logger.warning("%s: connection/timeout — %s", provider, e)
                return None

            except LiteLLMAuthError as e:
                last_exc = e
                self.quota.mark_auth_failed(provider)
                logger.error("%s: authentication failure — %s", provider, e)
                return None

            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                # 404 model-not-found → skip this model permanently
                if "404" in msg or "model_not_found" in msg or "does not exist" in msg:
                    logger.warning("%s/%s: 404 — model not found", provider, model.model_id)
                    return None
                self.quota.mark_error(provider)
                logger.warning("%s: unexpected error — %s", provider, e)
                return None

        return None

    async def _litellm_call(
        self, provider: str, model_id: str, messages: List[Any], params: Dict[str, Any]
    ) -> Any:
        if not _LITELLM_AVAILABLE:
            raise ImportError("litellm not installed — run: pip install litellm")
        
        # Ensure correct litellm prefix
        prefix = _LITELLM_PROVIDER_PREFIX.get(provider, provider)
        litellm_model = f"{prefix}/{model_id}" if "/" not in model_id else model_id
        if "/" in model_id and prefix not in model_id:
             # handle cases like openrouter/google/gemma
             pass

        return await acompletion(  # type: ignore[union-attr]
            model=litellm_model,
            messages=messages,
            timeout=settings.llm_timeout,
            **params,
        )

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
        yielded: Set[str] = set()

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

    def _select(self, capability: str, opts: RoutingOptions) -> Optional[RouteDecision]:
        """Score eligible providers and pick one via weighted random."""
        with open("/tmp/ROUTER_TRACE.log", "a") as f:
            f.write(f"SELECT CALL: capability={capability} opts={opts}\n")
        
        weights = _STRATEGY_WEIGHTS.get(opts.strategy, _STRATEGY_WEIGHTS[RoutingStrategy.AUTO])
        excluded = set(opts.excluded_providers) | {"ollama"}

        candidates: Dict[str, float] = {}
        for provider in CLOUD_PRIORITY_ORDER:
            if provider in excluded:
                continue
            if opts.preferred_providers and provider not in opts.preferred_providers:
                continue
            
            # Skip providers without API keys to avoid auth failures and long cooldowns
            api_key_env = PROVIDER_CATALOGUE.get(provider, {}).get("api_key_env")
            if api_key_env and not os.getenv(api_key_env):
                logger.debug("Skipping %s — no API key configured", provider)
                continue
            
            if not self.quota.states.get(provider, ProviderState("", 0, 0)).is_available():
                continue
            model = self.discovery.get_best_model(
                provider, capability, prefer_free=opts.free_tier_only
            )
            if model is None:
                continue
            
            # CRITICAL: Prevent embedding models from being used for chat/vision
            if capability != "embedding" and "embedding" in model.capabilities:
                continue
            if capability == "embedding" and "embedding" not in model.capabilities:
                continue

            score = self.quota.score(provider, **weights)
            if score > 0:
                candidates[provider] = score  # type: ignore[index]

        chosen = _weighted_choice(candidates)
        if chosen is None:
            return None

        model = self.discovery.get_best_model(chosen, capability, prefer_free=opts.free_tier_only)
        if model is None:
            return None

        return RouteDecision(
            provider=chosen,
            model=model,
            strategy=opts.strategy,
            score=candidates[chosen],
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Task-type detection
    # ══════════════════════════════════════════════════════════════════════════

    def _detect_task_type(self, request_data: Dict[str, Any]) -> TaskType:
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
    def _format_response(
        raw: Any,
        provider: str,
        model: ModelRecord,
        latency_ms: float,
        strategy: str,
        original_provider: str,
    ) -> Dict[str, Any]:
        """Normalise a litellm response into a standard dict."""
        if isinstance(raw, dict):
            resp: Dict[str, Any] = raw
        elif hasattr(raw, "model_dump"):
            resp = dict(raw.model_dump())
        else:
            resp = dict(vars(raw))

        resp["routing_metadata"] = {  # type: ignore[index]
            "provider":          provider,
            "model":             model.model_id,
            "strategy":          strategy,
            "latency_ms":        round(latency_ms),
            "cache_hit":         False,
            "cost_usd":          0.0,
            "fallback":          provider != original_provider,
            "original_provider": original_provider,
        }
        return resp

    @staticmethod
    def _is_daily_limit(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(k in msg for k in ("tokens per day", "tpd", "credit balance", "depleted", "daily limit"))

    @staticmethod
    def _parse_retry_delay(exc: Exception) -> Optional[float]:
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
    def _extract_prompt_text(request_data: Dict[str, Any]) -> str:
        msgs = request_data.get("messages", [])
        if msgs:
            last = msgs[-1]
            content = last.get("content", "") if isinstance(last, dict) else ""
            return str(content)[:200]  # type: ignore[misc]
        return str(request_data.get("input", ""))[:200]  # type: ignore[misc]

    # ══════════════════════════════════════════════════════════════════════════
    # Observability
    # ══════════════════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict[str, Any]:
        return {
            "providers": self.quota.get_stats(),
            "cache":     self.cache.stats,
            "models_per_provider": {
                p: len(self.discovery.get_models(p)) for p in PROVIDER_CATALOGUE
            },
        }
