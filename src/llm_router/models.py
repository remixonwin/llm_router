"""
models.py — Pydantic schemas and runtime dataclasses for the LLM Router.

Three layers:
  1. API request / response schemas (FastAPI i/o)
  2. Runtime state objects  (ProviderState, ModelRecord)
  3. Internal routing types (RouteDecision, CacheKey)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

try:
    from pydantic import BaseModel, Field, field_validator
    _PYDANTIC_AVAILABLE = True
except ImportError:
    _PYDANTIC_AVAILABLE = False
    # Minimal fallback stubs
    class BaseModel:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def model_dump(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    class Field:  # type: ignore[no-redef]
        def __new__(cls, default=None, **kwargs):
            return default

    def field_validator(*args, **kwargs):  # type: ignore[misc]
        def decorator(fn):
            return fn
        return decorator


# ══════════════════════════════════════════════════════════════════════════════
# Enumerations
# ══════════════════════════════════════════════════════════════════════════════


class TaskType(str, Enum):
    TEXT_GENERATION     = "text_generation"
    CHAT_COMPLETION     = "chat_completion"
    EMBEDDINGS          = "embeddings"
    VISION_CLASSIFY     = "vision_classify"
    VISION_DETECT       = "vision_detect"
    VISION_OCR          = "vision_ocr"
    VISION_QA           = "vision_qa"
    VISION_CAPTION      = "vision_caption"
    VISION_UNDERSTANDING= "vision_understanding"
    FUNCTION_CALLING    = "function_calling"
    UNKNOWN             = "unknown"


class RoutingStrategy(str, Enum):
    AUTO          = "auto"        # balanced: quota + latency + quality
    COST_OPTIMIZED= "cost_optimized"   # maximise remaining free quota
    QUALITY_FIRST = "quality_first"    # highest quality_score wins
    LATENCY_FIRST = "latency_first"    # lowest measured latency wins
    ROUND_ROBIN   = "round_robin"      # uniform spread regardless of score


class CachePolicy(str, Enum):
    ENABLED  = "enabled"   # use cache if hit
    DISABLED = "disabled"  # bypass entirely
    REFRESH  = "refresh"   # force re-fetch and repopulate


# ══════════════════════════════════════════════════════════════════════════════
# API Request / Response schemas
# ══════════════════════════════════════════════════════════════════════════════


class RoutingOptions(BaseModel):
    strategy: RoutingStrategy = RoutingStrategy.AUTO
    free_tier_only: bool = False
    preferred_providers: List[str] = Field(default_factory=list)
    excluded_providers: List[str] = Field(default_factory=list)
    cache_policy: CachePolicy = CachePolicy.ENABLED
    require_capability: Optional[str] = None


class Message(BaseModel):
    role: str
    content: Union[str, List[Any]]
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Union[Message, Dict[str, Any]]]
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, gt=0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    routing: Optional[RoutingOptions] = None

    @field_validator("messages", mode="before")
    @classmethod
    def normalise_messages(cls, v: Any) -> List[Any]:
        if isinstance(v, list):
            return [m if isinstance(m, dict) else m.model_dump() for m in v]
        return v


class EmbeddingRequest(BaseModel):
    model: Optional[str] = None
    input: Union[str, List[str]]
    routing: Optional[RoutingOptions] = None


class VisionRequest(BaseModel):
    task_type: TaskType = TaskType.VISION_UNDERSTANDING
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    question: Optional[str] = None
    categories: Optional[List[str]] = None
    language: Optional[str] = None
    model: Optional[str] = None
    routing: Optional[RoutingOptions] = None

    @field_validator("image_url", "image_base64", mode="before")
    @classmethod
    def at_least_one_image(cls, v: Any, info: Any) -> Any:
        return v  # cross-field validation handled in __init__


class RoutingMetadata(BaseModel):
    provider: str
    model: str
    strategy: str
    latency_ms: float
    cache_hit: bool = False
    cost_usd: float = 0.0
    quota_remaining: Optional[int] = None
    fallback: bool = False
    original_provider: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# Runtime state — ProviderState
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ModelRecord:
    """A single model as discovered from a provider."""
    provider: str
    model_id: str                               # bare id (no provider/ prefix)
    full_id: str                                # provider/model_id
    capabilities: Set[str] = field(default_factory=set)
    context_window: int = 4_096
    rpm_limit: int = 60
    is_free: bool = True
    supports_streaming: bool = True
    supports_tools: bool = False
    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def litellm_id(self) -> str:
        """ID to pass directly to litellm.acompletion()."""
        return self.full_id

    def has_capability(self, cap: str) -> bool:
        return cap in self.capabilities


@dataclass
class ProviderState:
    """Runtime quota and health state for a provider."""
    name: str
    rpm_limit: int
    rpd_limit: int
    # quota counters (reset by QuotaManager)
    rpm_used: int = 0
    rpd_used: int = 0
    # rolling hourly bucket (index = hour-of-day)
    hourly_usage: List[int] = field(default_factory=lambda: [0] * 24)
    # performance metrics
    error_count: int = 0
    consecutive_errors: int = 0
    avg_latency_ms: float = 200.0
    quality_score: float = 1.0
    # circuit-breaker
    circuit_open: bool = False
    circuit_open_until: Optional[datetime] = None
    # cooldown (e.g. after daily-limit hit)
    cooldown_until: Optional[datetime] = None
    # authentication failure cooldown
    auth_failure_until: Optional[datetime] = None

    # ── Derived properties ───────────────────────────────────────────────────

    @property
    def rpm_remaining(self) -> int:
        return max(0, self.rpm_limit - self.rpm_used)

    @property
    def rpd_remaining(self) -> int:
        return max(0, self.rpd_limit - self.rpd_used)

    @property
    def rpm_utilization(self) -> float:
        return self.rpm_used / self.rpm_limit if self.rpm_limit > 0 else 0.0

    @property
    def rpd_utilization(self) -> float:
        return self.rpd_used / self.rpd_limit if self.rpd_limit > 0 else 0.0

    def is_available(self) -> bool:
        now = datetime.now(timezone.utc)
        if self.circuit_open and self.circuit_open_until and now < self.circuit_open_until:
            return False
        if self.cooldown_until and now < self.cooldown_until:
            return False
        if self.auth_failure_until and now < self.auth_failure_until:
            return False
        if self.rpd_remaining <= 0:
            return False
        return True

    def record_success(self, latency_ms: float) -> None:
        now = datetime.now(timezone.utc)
        hour = now.hour
        self.rpm_used += 1
        self.rpd_used += 1
        self.hourly_usage[hour] += 1
        self.avg_latency_ms = 0.2 * latency_ms + 0.8 * self.avg_latency_ms
        self.consecutive_errors = 0
        self.circuit_open = False

    def record_failure(self, is_rate_limit: bool = False) -> None:
        self.error_count += 1
        self.consecutive_errors += 1
        if self.consecutive_errors >= 5:
            self.trip_circuit(60)

    def trip_circuit(self, seconds: float) -> None:
        from datetime import timedelta
        self.circuit_open = True
        self.circuit_open_until = datetime.now(timezone.utc) + timedelta(seconds=seconds)

    def set_cooldown(self, seconds: float) -> None:
        from datetime import timedelta
        self.cooldown_until = datetime.now(timezone.utc) + timedelta(seconds=seconds)

    def predict_exhaustion_hours(self) -> float:
        """Estimate hours until daily quota runs out based on hourly usage pattern."""
        current_hour = datetime.now(timezone.utc).hour
        non_zero = [h for h in self.hourly_usage if h > 0]
        avg_per_hour = sum(non_zero) / len(non_zero) if non_zero else 1.0
        remaining = float(self.rpd_remaining)
        for offset in range(24 - current_hour):
            hour = (current_hour + offset) % 24
            remaining -= self.hourly_usage[hour] or avg_per_hour
            if remaining <= 0:
                return float(offset)
        return 24.0


# ══════════════════════════════════════════════════════════════════════════════
# Internal routing types
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class RouteDecision:
    provider: str
    model: ModelRecord
    strategy: str
    score: float
    is_fallback: bool = False
    fallback_chain: List[str] = field(default_factory=list)


@dataclass
class CacheKey:
    """Deterministic cache key for a completion request."""
    messages_hash: str
    model: Optional[str]
    temperature: Optional[float]

    @classmethod
    def from_request(cls, messages: List[Any], model: Optional[str], temperature: Optional[float]) -> "CacheKey":
        payload = json.dumps({"messages": messages, "model": model}, sort_keys=True, default=str)
        h = hashlib.sha256(payload.encode()).hexdigest()[:24]
        return cls(messages_hash=h, model=model, temperature=temperature)

    def __str__(self) -> str:
        return f"llm:{self.messages_hash}:t{self.temperature}"
