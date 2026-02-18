"""
cache.py — Two-tier LLM response caching for the Intelligent Router.

Tier 1 — EXACT cache
  Key: SHA-256(messages + model + temperature)
  Backend: diskcache.Cache (persistent across restarts, LRU eviction)
  TTL: configurable (default 1 h)
  Use: identical requests served instantly without any provider call

Tier 2 — SEMANTIC cache  (optional, requires an embedding provider)
  Key: nearest-neighbour lookup in an in-memory vector store
  Threshold: cosine similarity ≥ 0.97 (configurable)
  Backend: simple list-based store; replace with FAISS/hnswlib in production
  Use: near-duplicate questions ("What is AI?" vs "What is artificial intelligence?")

Both tiers are bypassed when CachePolicy.DISABLED or .REFRESH is requested.

Dependencies:
  diskcache      — pip install diskcache
  xxhash         — pip install xxhash     (faster hashing; falls back to hashlib)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import diskcache  # type: ignore[import]
    _DISKCACHE_AVAILABLE = True
except ImportError:
    _DISKCACHE_AVAILABLE = False

try:
    import xxhash  # type: ignore[import]
    def _fast_hash(data: str) -> str:
        return xxhash.xxh64(data.encode()).hexdigest()
except ImportError:
    def _fast_hash(data: str) -> str:  # type: ignore[misc]
        return hashlib.sha256(data.encode()).hexdigest()[:16]

from llm_router.config import settings  # type: ignore[import]
from llm_router.models import CacheKey, CachePolicy  # type: ignore[import]

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# In-memory fallback cache (used when diskcache is not installed)
# ══════════════════════════════════════════════════════════════════════════════

class _SimpleMemCache:
    """TTL-aware dict. Not persistent, but good enough as a fallback."""

    def __init__(self, maxsize: int = 2_000, ttl: float = 3600.0) -> None:
        self._store: Dict[str, Tuple[Any, float]] = {}
        self._maxsize = maxsize
        self._ttl = ttl

    def get(self, key: str, default: Any = None) -> Any:
        entry = self._store.get(key)
        if entry is None:
            return default
        value, ts = entry
        if time.monotonic() - ts > self._ttl:
            del self._store[key]  # type: ignore[misc]
            return default
        return value

    def set(self, key: str, value: Any, expire: Optional[float] = None) -> None:
        if len(self._store) >= self._maxsize:
            # Evict ~10 % LRU by insertion order (approximate)
            to_del = list(self._store.keys())[: max(1, self._maxsize // 10)]  # type: ignore[misc]
            for k in to_del:
                self._store.pop(k, None)
        self._store[key] = (value, time.monotonic())

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    @property
    def volume(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()


# ══════════════════════════════════════════════════════════════════════════════
# Semantic cache vector store
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _VectorEntry:
    key: str          # exact cache key for the stored response
    vector: List[float]
    prompt_text: str  # truncated for logging


class SemanticStore:
    """
    Minimal cosine-similarity vector store for semantic caching.

    In production replace with FAISS, hnswlib, or Chroma for sub-ms lookups
    at scale.  This implementation is linear-scan and suitable for < 10 k entries.
    """

    def __init__(self, threshold: float = 0.97) -> None:
        self._entries: List[_VectorEntry] = []
        self.threshold = threshold

    def add(self, key: str, vector: List[float], prompt_text: str) -> None:
        self._entries.append(_VectorEntry(key=key, vector=vector, prompt_text=prompt_text))

    def find(self, query_vector: List[float]) -> Optional[str]:
        """Return the cache key of the nearest neighbour above threshold, or None."""
        best_sim = -1.0
        best_key: Optional[str] = None
        for entry in self._entries:
            sim = self._cosine(query_vector, entry.vector)
            if sim > best_sim:
                best_sim = sim
                best_key = entry.key
        if best_sim >= self.threshold:
            logger.debug("Semantic cache hit (similarity=%.4f)", best_sim)
            return best_key
        return None

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(y * y for y in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def __len__(self) -> int:
        return len(self._entries)


# ══════════════════════════════════════════════════════════════════════════════
# ResponseCache — unified two-tier API
# ══════════════════════════════════════════════════════════════════════════════

class ResponseCache:
    """
    Two-tier response cache (exact + semantic).

    Usage::

        cache = ResponseCache()
        key = cache.make_key(messages, model, temperature)

        # Read
        hit = cache.get_exact(key)
        if hit is None:
            response = await call_llm(...)
            cache.set(key, response, prompt_text=..., embedding=...)

        # Semantic
        sem_key = cache.find_semantic(query_embedding)
        if sem_key:
            hit = cache.get_exact(sem_key)
    """

    def __init__(self) -> None:
        self._ttl = float(settings.response_cache_ttl)
        self._exact_enabled = settings.exact_cache_enabled
        self._semantic_enabled = settings.semantic_cache_enabled

        # Exact-match backend
        if _DISKCACHE_AVAILABLE and self._exact_enabled:
            import os
            os.makedirs(settings.cache_dir, exist_ok=True)
            size_bytes = settings.cache_max_size_mb * 1024 * 1024
            self._exact: Any = diskcache.Cache(
                settings.cache_dir,
                size_limit=size_bytes,
                eviction_policy="least-recently-used",
            )
            logger.info("Exact cache: diskcache at %s (%d MB)", settings.cache_dir, settings.cache_max_size_mb)
        else:
            self._exact = _SimpleMemCache(maxsize=5_000, ttl=self._ttl)
            if not _DISKCACHE_AVAILABLE:
                logger.warning("diskcache not installed — using in-memory exact cache (not persistent)")

        # Semantic-match backend
        self._semantic = SemanticStore(threshold=0.97) if self._semantic_enabled else None

        # Stats
        self._hits_exact = 0
        self._hits_semantic = 0
        self._misses = 0

    # ── Key construction ───────────────────────────────────────────────────────

    @staticmethod
    def make_key(
        messages: List[Any],
        model: Optional[str],
        temperature: Optional[float],
    ) -> str:
        """Deterministic cache key from request parameters."""
        payload = json.dumps(
            {"m": messages, "model": model, "t": temperature},
            sort_keys=True,
            default=str,
        )
        return "llm:" + _fast_hash(payload)

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_exact(self, key: str) -> Optional[Any]:
        if not self._exact_enabled:
            return None
        val = self._exact.get(key)
        if val is not None:
            self._hits_exact += 1
        else:
            self._misses += 1
        return val

    def find_semantic(self, embedding: List[float]) -> Optional[str]:
        """Return an exact-cache key if a semantically similar query exists."""
        if not self._semantic_enabled or self._semantic is None:
            return None
        store = self._semantic
        assert store is not None
        return store.find(embedding)

    def get_by_semantic(self, embedding: List[float]) -> Optional[Any]:
        """Find semantically similar response and return it (or None)."""
        key = self.find_semantic(embedding)
        if key is None:
            return None
        val = self.get_exact(key)
        if val is not None:
            self._hits_semantic += 1
        return val

    # ── Write ──────────────────────────────────────────────────────────────────

    def set(
        self,
        key: str,
        value: Any,
        prompt_text: str = "",
        embedding: Optional[List[float]] = None,
    ) -> None:
        if not self._exact_enabled:
            return
        if _DISKCACHE_AVAILABLE and isinstance(self._exact, diskcache.Cache):
            self._exact.set(key, value, expire=self._ttl)
        else:
            self._exact.set(key, value)
        if self._semantic_enabled and self._semantic is not None and embedding:
            store: SemanticStore = self._semantic  # type: ignore[assignment]
            store.add(key, embedding, prompt_text[:200])

    def invalidate(self, key: str) -> None:
        if hasattr(self._exact, "delete"):
            try:
                self._exact.delete(key)
            except Exception:
                pass
        elif isinstance(self._exact, _SimpleMemCache):
            self._exact._store.pop(key, None)

    def clear(self) -> None:
        if hasattr(self._exact, "clear"):
            self._exact.clear()
        store = self._semantic
        if store is not None:
            store._entries.clear()
        logger.info("Cache cleared")

    # ── Stats ──────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, Any]:
        total = self._hits_exact + self._hits_semantic + self._misses
        hit_rate = (self._hits_exact + self._hits_semantic) / total if total else 0.0
        semantic_len = 0
        if self._semantic is not None:
            store: SemanticStore = self._semantic  # type: ignore[assignment]
            semantic_len = len(store)
        rounded_rate = float(round(hit_rate * 10000) / 10000)
        return {
            "exact_hits":    self._hits_exact,
            "semantic_hits": self._hits_semantic,
            "misses":        self._misses,
            "hit_rate":      rounded_rate,
            "semantic_entries": semantic_len,
            "backend": "diskcache" if (_DISKCACHE_AVAILABLE and self._exact_enabled) else "memory",
        }
