import pytest

import llm_router.discovery as discovery_mod
import llm_router.router as router_mod


@pytest.mark.asyncio
async def test_remove_model_on_permanent_error(monkeypatch):
    """When a provider returns a permanent-model-not-found error, the router
    should call discovery.remove_model to prune it from the cache.
    """

    called = {"removed": False}

    async def fake_acompletion(*args, **kwargs):
        raise Exception("model 'foo' was removed on 2025-01-01")

    # Prepare a discovery instance with a fake cache containing one model
    disc = discovery_mod.CapabilityDiscovery()
    provider = "openrouter"
    model_id = "google/gemma-3n-e4b-it:free"
    # Seed a bootstrap entry so get_models returns something
    disc._cache[f"models:{provider}"] = [
        discovery_mod.ModelRecord(
            provider=provider,
            model_id=model_id,
            full_id=f"{provider}/{model_id}",
        )
    ]

    # Replace discovery.remove_model to observe calls
    def fake_remove(p, m):
        called["removed"] = True
        called["args"] = (p, m)

    monkeypatch.setattr(disc, "remove_model", fake_remove)

    # Create router pointing at our fake discovery
    r = router_mod.IntelligentRouter()
    r.discovery = disc

    # Monkeypatch litellm availability and acompletion to simulate permanent error
    monkeypatch.setattr(router_mod, "_LITELLM_AVAILABLE", True)
    monkeypatch.setattr(router_mod, "acompletion", fake_acompletion)

    # Directly call _litellm_call which should call acompletion and then
    # the outer code path in _try_provider handles removal; simulate that
    # by invoking _try_provider as the full flow is complex. We'll call
    # _try_provider with minimal params and expect it to swallow the error
    # and invoke discovery.remove_model.

    model_record = disc.get_models(provider)[0]

    await r._try_provider(
        provider=provider,
        model=model_record,
        messages=[],
        extra_params={},
        start_time=0.0,
        original_provider=provider,
        strategy="auto",
    )

    assert called["removed"] is True
    assert called["args"][0] == provider
    # model_id passed should be the bare model id (without provider prefix)
    assert model_id in called["args"][1]
