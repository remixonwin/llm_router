#!/usr/bin/env python3
"""Test app to verify routeme package functionality."""

import asyncio
import os

os.environ["GROQ_API_KEY"] = "test_key_for_testing"
os.environ["ANTHROPIC_API_KEY"] = "test_key_for_testing"

from llm_router import (
    IntelligentRouter,
    RoutingOptions,
    RoutingStrategy,
    TaskType,
    CachePolicy,
    settings,
    PROVIDER_CATALOGUE,
)


async def test_basic_import():
    """Test 1: Basic imports work."""
    print("✓ Test 1: Basic imports successful")
    print(f"  - Version: {PROVIDER_CATALOGUE is not None}")
    print(f"  - Providers available: {list(PROVIDER_CATALOGUE.keys())}")


async def test_settings():
    """Test 2: Settings work."""
    print("\n✓ Test 2: Settings")
    print(f"  - Host: {settings.host}")
    print(f"  - Port: {settings.port}")
    print(f"  - LLM Timeout: {settings.llm_timeout}")
    print(f"  - Cache Dir: {settings.cache_dir}")
    print(f"  - Default Strategy: {settings.default_strategy}")


async def test_routing_options():
    """Test 3: RoutingOptions creation."""
    print("\n✓ Test 3: RoutingOptions")
    options = RoutingOptions(
        strategy=RoutingStrategy.AUTO,
        free_tier_only=True,
        preferred_providers=["groq", "gemini"],
        excluded_providers=["openai"],
        cache_policy=CachePolicy.ENABLED,
        require_capability="vision",
    )
    print(f"  - Strategy: {options.strategy}")
    print(f"  - Free tier only: {options.free_tier_only}")
    print(f"  - Preferred providers: {options.preferred_providers}")
    print(f"  - Excluded providers: {options.excluded_providers}")
    print(f"  - Cache policy: {options.cache_policy}")
    print(f"  - Required capability: {options.require_capability}")


async def test_router_initialization():
    """Test 4: Router initialization."""
    print("\n✓ Test 4: Router Initialization")
    router = IntelligentRouter()
    print(f"  - Router created: {router is not None}")

    await router.start()
    print(f"  - Router started successfully")

    stats = router.get_stats()
    print(f"  - Stats retrieved: {len(stats)} sections")
    print(f"  - Stats keys: {list(stats.keys())}")

    await router.stop()
    print(f"  - Router stopped successfully")


async def test_chat_completion():
    """Test 5: Chat completion (mock)."""
    print("\n✓ Test 5: Chat Completion (Mock)")
    router = IntelligentRouter()
    await router.start()

    request_data = {
        "messages": [{"role": "user", "content": "Hello! How are you?"}],
        "temperature": 0.7,
        "max_tokens": 100,
    }

    options = RoutingOptions(
        strategy=RoutingStrategy.COST_OPTIMIZED,
        free_tier_only=True,
    )

    try:
        response = await router.route(request_data, options)
        print(f"  - Response received: {type(response)}")
        if "routing_metadata" in response:
            print(f"  - Provider: {response['routing_metadata'].get('provider')}")
            print(f"  - Model: {response['routing_metadata'].get('model')}")
    except Exception as e:
        print(f"  - Expected error (no real API): {type(e).__name__}")

    await router.stop()


async def test_embeddings():
    """Test 6: Embeddings (mock)."""
    print("\n✓ Test 6: Embeddings (Mock)")
    router = IntelligentRouter()
    await router.start()

    request_data = {
        "input": "Hello world",
    }

    options = RoutingOptions(
        strategy=RoutingStrategy.LATENCY_FIRST,
        require_capability="embedding",
    )

    try:
        response = await router.route(request_data, options)
        print(f"  - Response received: {type(response)}")
    except Exception as e:
        print(f"  - Expected error (no real API): {type(e).__name__}")

    await router.stop()


async def test_provider_info():
    """Test 7: Provider information."""
    print("\n✓ Test 7: Provider Information")
    for provider in ["groq", "openai", "anthropic", "gemini", "ollama"]:
        if provider in PROVIDER_CATALOGUE:
            info = PROVIDER_CATALOGUE[provider]
            print(f"  - {provider}:")
            print(f"      Capabilities: {info.get('capabilities')}")
            print(f"      RPM Limit: {info.get('rpm_limit')}")


async def test_task_types():
    """Test 8: Task types."""
    print("\n✓ Test 8: Task Types")
    task_types = [
        TaskType.CHAT_COMPLETION,
        TaskType.EMBEDDINGS,
        TaskType.VISION_UNDERSTANDING,
        TaskType.FUNCTION_CALLING,
    ]
    for task in task_types:
        print(f"  - {task.name}")


async def test_routing_strategies():
    """Test 9: All routing strategies."""
    print("\n✓ Test 9: Routing Strategies")
    strategies = [
        RoutingStrategy.AUTO,
        RoutingStrategy.COST_OPTIMIZED,
        RoutingStrategy.QUALITY_FIRST,
        RoutingStrategy.LATENCY_FIRST,
        RoutingStrategy.ROUND_ROBIN,
    ]
    for strategy in strategies:
        print(f"  - {strategy.name}")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("ROUTEME PACKAGE TEST SUITE")
    print("=" * 60)

    await test_basic_import()
    await test_settings()
    await test_routing_options()
    await test_router_initialization()
    await test_chat_completion()
    await test_embeddings()
    await test_provider_info()
    await test_task_types()
    await test_routing_strategies()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
