"""
tests.py — Unit and integration tests for the Intelligent LLM Router.

Run with:  python -m pytest llm_router/tests.py -v
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import AsyncMock, patch

# ══════════════════════════════════════════════════════════════════════════════
# Helpers / fixtures
# ══════════════════════════════════════════════════════════════════════════════


def _make_litellm_response(content: str = "Hello!", model: str = "groq/llama-3.3-70b-versatile"):
    """Minimal litellm-like response namespace."""
    import types

    resp = types.SimpleNamespace(
        id="test-id",
        object="chat.completion",
        created=1_700_000_000,
        model=model,
        choices=[
            types.SimpleNamespace(
                index=0,
                message=types.SimpleNamespace(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    # Allow .model_dump()
    resp.model_dump = lambda: {
        "id": resp.id,
        "object": resp.object,
        "created": resp.created,
        "model": resp.model,
        "choices": [{"message": {"content": content}}],
        "usage": {"total_tokens": 15},
    }
    return resp


# ══════════════════════════════════════════════════════════════════════════════
# Config tests
# ══════════════════════════════════════════════════════════════════════════════


class TestConfig(unittest.TestCase):
    def test_provider_catalogue_has_all_expected_providers(self):
        from llm_router.config import PROVIDER_CATALOGUE

        for prov in [
            "groq",
            "gemini",
            "mistral",
            "openrouter",
            "together",
            "huggingface",
            "cohere",
            "openai",
            "anthropic",
            "openai_compatible",
            "ollama",
        ]:
            self.assertIn(prov, PROVIDER_CATALOGUE, f"Missing provider: {prov}")

    def test_openai_compatible_provider_config(self):
        from llm_router.config import PROVIDER_CATALOGUE

        cfg = PROVIDER_CATALOGUE["openai_compatible"]
        self.assertEqual(cfg["api_key_env"], "OPENAI_COMPATIBLE_API_KEY")
        self.assertEqual(cfg["priority"], 50)
        self.assertIn("chat", cfg["capabilities"])
        self.assertIn("vision", cfg["capabilities"])
        self.assertIn("function_calling", cfg["capabilities"])
        self.assertIn("embedding", cfg["capabilities"])
        self.assertFalse(cfg["free_tier"])

    def test_ollama_has_highest_priority_number(self):
        from llm_router.config import PROVIDER_CATALOGUE

        ollama_prio = PROVIDER_CATALOGUE["ollama"]["priority"]
        for name, cfg in PROVIDER_CATALOGUE.items():
            if name != "ollama":
                self.assertLess(
                    cfg["priority"], ollama_prio, f"{name}.priority should be < ollama.priority"
                )

    def test_cloud_priority_order_excludes_ollama(self):
        from llm_router.config import CLOUD_PRIORITY_ORDER

        self.assertNotIn("ollama", CLOUD_PRIORITY_ORDER)

    def test_bootstrap_models_have_required_fields(self):
        from llm_router.config import BOOTSTRAP_MODELS

        for provider, models in BOOTSTRAP_MODELS.items():
            for m in models:
                self.assertIn("id", m, f"Missing 'id' in {provider} bootstrap")
                self.assertIn("capabilities", m, f"Missing 'capabilities' in {provider}/{m}")

    def test_openai_compatible_models_parsing(self):
        from llm_router.config import settings

        original_value = settings.openai_compatible_models
        try:
            settings.openai_compatible_models = "gpt-4:128001:60,claude-3:200000:50"
            from llm_router.discovery import _parse_openai_compatible_models

            records = _parse_openai_compatible_models("openai_compatible")
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0].model_id, "gpt-4")
            self.assertEqual(records[0].context_window, 128001)
            self.assertEqual(records[1].model_id, "claude-3")
            self.assertEqual(records[1].context_window, 200000)
        finally:
            settings.openai_compatible_models = original_value


# ══════════════════════════════════════════════════════════════════════════════
# Model / dataclass tests
# ══════════════════════════════════════════════════════════════════════════════


class TestModels(unittest.TestCase):
    def test_provider_state_availability(self):
        from llm_router.models import ProviderState

        state = ProviderState(name="test", rpm_limit=10, rpd_limit=100)
        self.assertTrue(state.is_available())
        state.rpd_used = 100
        self.assertFalse(state.is_available())  # daily quota hit

    def test_provider_state_cooldown(self):
        from llm_router.models import ProviderState

        state = ProviderState(name="test", rpm_limit=10, rpd_limit=100)
        state.set_cooldown(3600)
        self.assertFalse(state.is_available())

    def test_provider_state_circuit_breaker(self):
        from llm_router.models import ProviderState

        state = ProviderState(name="test", rpm_limit=10, rpd_limit=100)
        state.trip_circuit(3600)
        self.assertFalse(state.is_available())

    def test_record_success_increments_usage(self):
        from llm_router.models import ProviderState

        state = ProviderState(name="test", rpm_limit=10, rpd_limit=100)
        state.record_success(200.0)
        self.assertEqual(state.rpd_used, 1)
        self.assertEqual(state.consecutive_errors, 0)

    def test_predict_exhaustion_no_usage(self):
        from llm_router.models import ProviderState

        state = ProviderState(name="test", rpm_limit=10, rpd_limit=100)
        hours = state.predict_exhaustion_hours()
        self.assertEqual(hours, 24.0)

    def test_cache_key_deterministic(self):
        from llm_router.models import CacheKey

        k1 = CacheKey.from_request([{"role": "user", "content": "hi"}], "gpt-4", 0.7)
        k2 = CacheKey.from_request([{"role": "user", "content": "hi"}], "gpt-4", 0.7)
        self.assertEqual(k1.messages_hash, k2.messages_hash)

    def test_cache_key_different_messages(self):
        from llm_router.models import CacheKey

        k1 = CacheKey.from_request([{"role": "user", "content": "hello"}], None, None)
        k2 = CacheKey.from_request([{"role": "user", "content": "world"}], None, None)
        self.assertNotEqual(k1.messages_hash, k2.messages_hash)


# ══════════════════════════════════════════════════════════════════════════════
# Discovery tests
# ══════════════════════════════════════════════════════════════════════════════


class TestDiscovery(unittest.TestCase):
    def setUp(self):
        from llm_router.discovery import CapabilityDiscovery

        self.disc = CapabilityDiscovery()

    def test_bootstrap_models_loaded(self):
        models = self.disc.get_models("groq")
        self.assertGreater(len(models), 0)

    def test_bootstrap_model_has_capabilities(self):
        models = self.disc.get_models("groq")
        for m in models:
            self.assertIsInstance(m.capabilities, set)
            self.assertGreater(len(m.capabilities), 0)

    def test_full_id_format(self):
        models = self.disc.get_models("gemini")
        for m in models:
            self.assertTrue(m.full_id.startswith("gemini/"), f"Unexpected full_id: {m.full_id}")

    def test_get_models_with_capability_vision(self):
        vision_models = self.disc.get_models_with_capability("vision")
        providers = {m.provider for m in vision_models}
        # Gemini, Groq, OpenRouter all have vision in bootstrap
        self.assertTrue(providers.intersection({"gemini", "groq", "openrouter"}))

    def test_get_models_with_capability_embedding(self):
        embed_models = self.disc.get_models_with_capability("embedding")
        self.assertGreater(len(embed_models), 0)

    def test_infer_capabilities_vision_hint(self):
        from llm_router.discovery import _infer_capabilities

        caps = _infer_capabilities("llama3.2-vision:latest")
        self.assertIn("vision", caps)

    def test_infer_capabilities_embedding_hint(self):
        from llm_router.discovery import _infer_capabilities

        caps = _infer_capabilities("text-embedding-3-small")
        self.assertIn("embedding", caps)
        self.assertNotIn("chat", caps)  # embedding-only

    def test_get_best_model_returns_free_first(self):
        model = self.disc.get_best_model("groq", "chat", prefer_free=True)
        self.assertIsNotNone(model)
        self.assertTrue(model.is_free)  # type: ignore[union-attr]


# ══════════════════════════════════════════════════════════════════════════════
# Cache tests
# ══════════════════════════════════════════════════════════════════════════════


class TestCache(unittest.TestCase):
    def setUp(self):
        from llm_router.cache import ResponseCache

        self.cache = ResponseCache()

    def test_cache_miss_returns_none(self):
        result = self.cache.get_exact("nonexistent_key")
        self.assertIsNone(result)

    def test_cache_set_and_get(self):
        key = self.cache.make_key([{"role": "user", "content": "test"}], None, None)
        self.cache.set(key, {"answer": 42})
        result = self.cache.get_exact(key)
        self.assertEqual(result, {"answer": 42})

    def test_cache_key_is_deterministic(self):
        msgs = [{"role": "user", "content": "hello world"}]
        k1 = self.cache.make_key(msgs, "gpt-4", 0.5)
        k2 = self.cache.make_key(msgs, "gpt-4", 0.5)
        self.assertEqual(k1, k2)

    def test_cache_key_differs_with_different_temp(self):
        msgs = [{"role": "user", "content": "same"}]
        k1 = self.cache.make_key(msgs, "gpt-4", 0.0)
        k2 = self.cache.make_key(msgs, "gpt-4", 1.0)
        self.assertNotEqual(k1, k2)

    def test_stats_increment_on_hit(self):
        key = self.cache.make_key([{"role": "user", "content": "stat_test"}], None, None)
        self.cache.set(key, {"data": "x"})
        before = self.cache.stats["exact_hits"]
        self.cache.get_exact(key)
        after = self.cache.stats["exact_hits"]
        self.assertEqual(after, before + 1)

    def test_stats_increment_on_miss(self):
        before = self.cache.stats["misses"]
        self.cache.get_exact("totally-fake-key-xyz")
        after = self.cache.stats["misses"]
        self.assertEqual(after, before + 1)

    def test_clear_empties_cache(self):
        key = self.cache.make_key([{"role": "user", "content": "clear_test"}], None, None)
        self.cache.set(key, {"data": "y"})
        self.cache.clear()
        self.assertIsNone(self.cache.get_exact(key))

    def test_semantic_store_cosine(self):
        from llm_router.cache import SemanticStore

        store = SemanticStore(threshold=0.9)
        v1 = [1.0, 0.0, 0.0]
        store.add("key1", v1, "test prompt")
        result = store.find([1.0, 0.0, 0.0])
        self.assertEqual(result, "key1")

    def test_semantic_store_below_threshold(self):
        from llm_router.cache import SemanticStore

        store = SemanticStore(threshold=0.99)
        store.add("key1", [1.0, 0.0, 0.0], "test")
        result = store.find([0.0, 1.0, 0.0])  # orthogonal
        self.assertIsNone(result)


# ══════════════════════════════════════════════════════════════════════════════
# Quota tests
# ══════════════════════════════════════════════════════════════════════════════


class TestQuota(unittest.TestCase):
    def setUp(self):
        from llm_router.quota import QuotaManager

        self.qm = QuotaManager()

    def test_all_providers_initialised(self):
        from llm_router.config import PROVIDER_CATALOGUE

        for prov in PROVIDER_CATALOGUE:
            self.assertIn(prov, self.qm.states)

    def test_can_accept_fresh_provider(self):
        self.assertTrue(self.qm.can_accept("groq"))

    def test_cannot_accept_exhausted_provider(self):
        state = self.qm.states["groq"]
        state.rpd_used = state.rpd_limit
        self.assertFalse(self.qm.can_accept("groq"))

    def test_mark_daily_exhausted_sets_cooldown(self):
        self.qm.mark_daily_exhausted("gemini")
        self.assertFalse(self.qm.states["gemini"].is_available())

    def test_mark_rate_limited_sets_cooldown(self):
        self.qm.mark_rate_limited("groq", retry_after=30.0)
        self.assertFalse(self.qm.states["groq"].is_available())

    def test_score_ollama_is_negative(self):
        score = self.qm.score("ollama")
        self.assertLess(score, 0)

    def test_score_available_provider_positive(self):
        score = self.qm.score("groq")
        self.assertGreaterEqual(score, 0)

    def test_scored_providers_excludes_ollama(self):
        pairs = self.qm.scored_providers()
        providers = [p for p, _ in pairs]
        self.assertNotIn("ollama", providers)

    def test_record_latency_updates_avg(self):
        state = self.qm.states["groq"]
        initial = state.avg_latency_ms
        self.qm.consume("groq")
        self.qm.record_latency("groq", 50.0)
        # avg_latency_ms updated via EWMA
        self.assertNotEqual(state.avg_latency_ms, initial)


# ══════════════════════════════════════════════════════════════════════════════
# Router unit tests  (litellm mocked)
# ══════════════════════════════════════════════════════════════════════════════


class TestRouterTaskDetection(unittest.TestCase):
    def setUp(self):
        from llm_router.router import IntelligentRouter

        self.router = IntelligentRouter()

    def test_detect_text(self):
        from llm_router.models import TaskType
        from llm_router.router import IntelligentRouter

        r = IntelligentRouter()
        tt = r._detect_task_type({"messages": [{"role": "user", "content": "hello"}]})
        self.assertEqual(tt, TaskType.CHAT_COMPLETION)

    def test_detect_embedding(self):
        from llm_router.models import TaskType
        from llm_router.router import IntelligentRouter

        r = IntelligentRouter()
        tt = r._detect_task_type({"input": "some text"})
        self.assertEqual(tt, TaskType.EMBEDDINGS)

    def test_detect_vision_from_image_url_field(self):
        from llm_router.models import TaskType
        from llm_router.router import IntelligentRouter

        r = IntelligentRouter()
        tt = r._detect_task_type({"image_url": "https://example.com/img.jpg"})
        self.assertEqual(tt, TaskType.VISION_UNDERSTANDING)

    def test_detect_vision_from_message_content(self):
        from llm_router.models import TaskType
        from llm_router.router import IntelligentRouter

        r = IntelligentRouter()
        tt = r._detect_task_type(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/x.jpg"},
                            },
                            {"type": "text", "text": "describe this"},
                        ],
                    }
                ]
            }
        )
        self.assertEqual(tt, TaskType.VISION_UNDERSTANDING)

    def test_detect_function_calling(self):
        from llm_router.models import TaskType
        from llm_router.router import IntelligentRouter

        r = IntelligentRouter()
        tt = r._detect_task_type(
            {
                "messages": [{"role": "user", "content": "what time is it"}],
                "tools": [{"type": "function", "function": {"name": "get_time"}}],
            }
        )
        self.assertEqual(tt, TaskType.FUNCTION_CALLING)

    def test_extract_prompt_text_from_messages(self):
        from llm_router.router import IntelligentRouter

        r = IntelligentRouter()
        text = r._extract_prompt_text({"messages": [{"role": "user", "content": "Hello, router!"}]})
        self.assertIn("Hello, router!", text)

    def test_is_daily_limit_detection(self):
        from llm_router.router import IntelligentRouter

        self.assertTrue(IntelligentRouter._is_daily_limit(Exception("tokens per day exceeded")))
        self.assertTrue(IntelligentRouter._is_daily_limit(Exception("credit balance depleted")))
        self.assertFalse(IntelligentRouter._is_daily_limit(Exception("connection refused")))

    def test_parse_retry_delay_from_text(self):
        from llm_router.router import IntelligentRouter

        delay = IntelligentRouter._parse_retry_delay(Exception("retry after 45 seconds"))
        self.assertEqual(delay, 45.0)


class TestWeightedChoice(unittest.TestCase):
    def test_empty_returns_none(self):
        from llm_router.router import _weighted_choice

        self.assertIsNone(_weighted_choice({}))

    def test_single_provider_always_chosen(self):
        from llm_router.router import _weighted_choice

        for _ in range(10):
            result = _weighted_choice({"groq": 0.9})
            self.assertEqual(result, "groq")

    def test_zero_score_falls_to_max(self):
        from llm_router.router import _weighted_choice

        result = _weighted_choice({"a": 0.0, "b": 0.0})
        self.assertIn(result, ["a", "b"])

    def test_higher_score_wins_more_often(self):
        from llm_router.router import _weighted_choice

        counts = {"high": 0, "low": 0}
        for _ in range(500):
            winner = _weighted_choice({"high": 10.0, "low": 1.0})
            counts[winner] += 1
        # high-score provider should win roughly 10x as often
        self.assertGreater(counts["high"], counts["low"] * 3)


# ══════════════════════════════════════════════════════════════════════════════
# Async router tests (mocked litellm)
# ══════════════════════════════════════════════════════════════════════════════


class TestRouterAsync(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import os

        os.environ["GROQ_API_KEY"] = "test-key"
        from llm_router.router import IntelligentRouter

        self.router = IntelligentRouter()
        await self.router.start()

    async def asyncTearDown(self):
        await self.router.stop()

    async def test_cache_hit_bypasses_llm(self):
        """Populate cache manually; route() should return it without calling litellm."""
        messages = [{"role": "user", "content": "cache test unique xyz"}]
        key = self.router.cache.make_key(messages, None, None)
        cached_response = {
            "choices": [{"message": {"content": "cached!"}}],
            "routing_metadata": {"provider": "cache", "model": "cached", "cache_hit": False},
        }
        self.router.cache.set(key, cached_response)

        result = await self.router.route({"messages": messages})
        self.assertTrue(result["routing_metadata"]["cache_hit"])

    @patch("llm_router.router.acompletion", new_callable=AsyncMock)
    async def test_successful_text_route(self, mock_acomp):
        mock_acomp.return_value = _make_litellm_response("Great answer!")
        result = await self.router.route(
            {"messages": [{"role": "user", "content": "What is 2+2?"}]}
        )
        self.assertIn("routing_metadata", result)
        self.assertIn("provider", result["routing_metadata"])

    @patch("llm_router.router.acompletion", new_callable=AsyncMock)
    async def test_fallback_on_rate_limit(self, mock_acomp):
        from litellm import RateLimitError

        # First call raises rate limit; second succeeds
        mock_acomp.side_effect = [
            RateLimitError("rate limited", llm_provider="groq", model="llama-3.3"),
            _make_litellm_response("fallback answer"),
        ]
        result = await self.router.route(
            {"messages": [{"role": "user", "content": "fallback test"}]}
        )
        self.assertIn("routing_metadata", result)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
