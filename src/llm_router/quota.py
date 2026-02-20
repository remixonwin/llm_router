"""
quota.py — Provider quota management for the Intelligent LLM Router.

Responsibilities:
  • Track per-provider RPM / RPD usage in real-time
  • Token-bucket rate limiter per provider (prevents burst overuse)
  • Sliding-window per-minute counter
  • Daily quota reset scheduling (midnight UTC)
  • Exhaustion prediction
  • Priority queue of providers sorted by score (for round-robin and scoring)

Dependencies:
  limits  — pip install limits     (token bucket / sliding window)
  Uses stdlib threading / asyncio for scheduling.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from datetime import UTC, datetime
from typing import Any

try:
    from limits import parse
    from limits.storage import MemoryStorage
    from limits.strategies import MovingWindowRateLimiter

    _LIMITS_AVAILABLE = True
except ImportError:
    _LIMITS_AVAILABLE = False

from .config import PROVIDER_CATALOGUE, has_api_key, settings
from .models import ProviderState

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Sliding-window counter (stdlib fallback if `limits` not installed)
# ══════════════════════════════════════════════════════════════════════════════


class _SlidingWindowCounter:
    """Thread-safe sliding-window request counter using a deque of timestamps."""

    def __init__(self, window_seconds: float = 60.0) -> None:
        self._window: float = window_seconds
        self._ts: deque[float] = deque()

    def record(self) -> None:
        now: float = time.monotonic()
        self._ts.append(now)
        self._evict(now)

    def count(self) -> int:
        self._evict(time.monotonic())
        return len(self._ts)

    def _evict(self, now: float) -> None:
        cutoff: float = now - self._window
        while self._ts and self._ts[0] < cutoff:
            self._ts.popleft()


# ══════════════════════════════════════════════════════════════════════════════
# QuotaManager
# ══════════════════════════════════════════════════════════════════════════════


class QuotaManager:
    """
    Manages quota state for all configured providers.

    Usage::

        qm = QuotaManager()
        await qm.start()                # launches background reset task

        if qm.can_accept(provider):
            qm.consume(provider)
            ... make LLM call ...
            qm.record_latency(provider, latency_ms)

        qm.mark_rate_limited(provider)  # triggers appropriate cooldown
        qm.mark_daily_exhausted(provider)

        stats = qm.get_stats()
    """

    def __init__(self) -> None:
        self.states: dict[str, ProviderState] = {}
        self._rpm_counters: dict[str, _SlidingWindowCounter] = {}
        self._reset_task: asyncio.Task | None = None

        # `limits` integration (optional — provides more accurate token buckets)
        self._limiter: Any | None = None
        self._limit_items: dict[str, Any] = {}
        if _LIMITS_AVAILABLE:
            try:
                storage = MemoryStorage()
                self._limiter = MovingWindowRateLimiter(storage)
            except Exception as e:
                logger.debug("limits init failed: %s", e)
                self._limiter = None

        self._init_providers()

    # ── Initialisation ──────────────────────────────────────────────────────

    def _init_providers(self) -> None:
        for name, cfg in PROVIDER_CATALOGUE.items():
            rpm = cfg.get("rpm_limit", 60)
            rpd = cfg.get("rpd_limit", 10_000)
            self.states[name] = ProviderState(name=name, rpm_limit=rpm, rpd_limit=rpd)
            self._rpm_counters[name] = _SlidingWindowCounter(window_seconds=60.0)

            if _LIMITS_AVAILABLE and self._limiter:
                try:
                    self._limit_items[name] = parse(f"{rpm} per minute")
                except Exception:
                    pass

        logger.info("QuotaManager initialised for %d providers", len(self.states))

    async def start(self) -> None:
        """Launch background task that resets per-minute counters and daily at midnight UTC."""
        self._reset_task = asyncio.create_task(self._reset_loop(), name="quota_reset")

    async def stop(self) -> None:
        if self._reset_task and not self._reset_task.done():
            self._reset_task.cancel()
            try:
                await self._reset_task
            except asyncio.CancelledError:
                pass

    # ── Per-request API ─────────────────────────────────────────────────────

    def can_accept(self, provider: str) -> bool:
        """True if provider has both RPM and RPD headroom and has API key configured."""
        state = self.states.get(provider)
        if state is None:
            return False
        if not state.is_available():
            return False

        # Check if provider has API key configured (skip for ollama which
        # doesn't need one)
        if provider != "ollama" and not has_api_key(provider):
            logger.debug("Skipping %s — no API key configured", provider)
            return False

        # Sliding-window RPM check
        window_count = self._rpm_counters[provider].count()
        if window_count >= state.rpm_limit:
            logger.debug("RPM gate: %s at %d/%d", provider, window_count, state.rpm_limit)
            return False
        return True

    def consume(self, provider: str) -> None:
        """Record one request against the provider's quotas."""
        state = self.states.get(provider)
        if state is None:
            return
        self._rpm_counters[provider].record()
        # ProviderState.record_success() will increment rpd_used once we get
        # latency

    def record_latency(self, provider: str, latency_ms: float) -> None:
        """Update the provider's average latency using an exponential moving average.

        Uses alpha=0.1 for smoothing. If the existing avg_latency_ms is None or
        non-positive, the incoming latency is used directly.
        """
        state = self.states.get(provider)
        if not state:
            return
        # Ensure incoming value is float
        latency = float(latency_ms)
        # If no prior latency recorded or non-positive, set directly
        if state.avg_latency_ms is None or state.avg_latency_ms <= 0.0:
            state.avg_latency_ms = latency
            return
        alpha = 0.1
        state.avg_latency_ms = (1 - alpha) * state.avg_latency_ms + alpha * latency

    def mark_rate_limited(self, provider: str, retry_after: float | None = None) -> None:
        """Set short cooldown after an RPM/RPD error response."""
        state = self.states.get(provider)
        if not state:
            return
        delay = retry_after or 30.0
        state.set_cooldown(delay)
        state.record_failure(is_rate_limit=True)
        logger.warning("Provider %s rate-limited — cooldown %.0fs", provider, delay)

    def mark_daily_exhausted(self, provider: str) -> None:
        """Set long cooldown when daily quota is fully consumed."""
        state = self.states.get(provider)
        if not state:
            return
        cooldown = float(settings.daily_quota_cooldown_seconds)
        state.set_cooldown(cooldown)
        state.rpd_used = state.rpd_limit  # pin to limit
        logger.warning("Provider %s daily quota exhausted — cooldown %.0fs", provider, cooldown)

    def mark_error(self, provider: str) -> None:
        state = self.states.get(provider)
        if state:
            state.record_failure()

    # ── Scoring helpers ─────────────────────────────────────────────────────

    def score(
        self,
        provider: str,
        *,
        w_quota: float = 0.50,
        w_latency: float = 0.25,
        w_quality: float = 0.15,
        w_errors: float = 0.10,
    ) -> float:
        """
        Composite score in [0, 1].  Higher = more preferred.
        Ollama always returns -1 (never wins cloud selection).
        """
        if provider == "ollama":
            return -1.0
        state = self.states.get(provider)
        if not state:
            return 0.0
        quota_s = (
            state.rpd_remaining / state.rpd_limit if state.rpd_limit > 0 else 0.0
        )
        latency_s = 1.0 / (1.0 + state.avg_latency_ms / 1_000.0)
        error_s = 1.0 / (1.0 + state.consecutive_errors)
        return (
            w_quota * quota_s
            + w_latency * latency_s
            + w_quality * state.quality_score
            + w_errors * error_s
        )

    def scored_providers(
        self, available_only: bool = True, exclude: list[str] | None = None
    ) -> list[tuple[str, float]]:
        """Return [(provider, score)] sorted descending, ollama excluded."""
        excl = set(exclude or []) | {"ollama"}
        pairs: list[tuple[str, float]] = []
        for name in self.states:
            if name in excl:
                continue
            # Skip providers without API keys
            if name != "ollama" and not has_api_key(name):
                continue
            if available_only and not self.states[name].is_available():
                continue
            pairs.append((name, self.score(name)))
        return sorted(pairs, key=lambda x: x[1], reverse=True)

    # ── Stats ───────────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        for name, state in self.states.items():
            # Check if provider has API key configured
            has_key = has_api_key(name) if name != "ollama" else True
            stats[name] = {
                "rpm_used": state.rpm_used,
                "rpm_remaining": state.rpm_remaining,
                "rpm_window": self._rpm_counters[name].count(),
                "rpd_used": state.rpd_used,
                "rpd_remaining": state.rpd_remaining,
                "rpm_utilization": round(state.rpm_utilization, 4),
                "rpd_utilization": round(state.rpd_utilization, 4),
                "avg_latency_ms": round(state.avg_latency_ms, 1),
                "quality_score": round(state.quality_score, 4),
                "error_count": state.error_count,
                "consecutive_errors": state.consecutive_errors,
                "circuit_open": state.circuit_open,
                "available": state.is_available(),
                "has_api_key": has_key,
                "score": round(self.score(name), 4),
                "exhaustion_hours": round(state.predict_exhaustion_hours(), 2),
            }
        return stats

    # ── Reset loop ──────────────────────────────────────────────────────────

    async def _reset_loop(self) -> None:
        """Reset daily quotas at midnight UTC; reset RPM counters every minute."""
        while True:
            try:
                await asyncio.sleep(60)
                self._reset_rpm()
                now_utc = datetime.now(UTC)
                if now_utc.hour == 0 and now_utc.minute == 0:
                    self._reset_daily()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Reset loop error: %s", exc)

    def _reset_rpm(self) -> None:
        """The sliding-window counters are self-expiring; nothing to do here.
        We do, however, clear expired cooldowns."""
        now = datetime.now(UTC)
        for state in self.states.values():
            if state.cooldown_until and now >= state.cooldown_until:
                state.cooldown_until = None
                logger.info("Cooldown expired for %s", state.name)
            if state.circuit_open and state.circuit_open_until and now >= state.circuit_open_until:
                state.circuit_open = False
                state.consecutive_errors = 0
                logger.info("Circuit reset for %s", state.name)

    def _reset_daily(self) -> None:
        for state in self.states.values():
            old_rpd = state.rpd_used
            state.rpd_used = 0
            state.hourly_usage = [0] * 24
            if old_rpd > 0:
                logger.info("Daily quota reset for %s (was %d used)", state.name, old_rpd)
