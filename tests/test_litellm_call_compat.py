import pytest

import llm_router.router as router_mod


@pytest.mark.asyncio
async def test_litellm_call_with_full_model_id(monkeypatch):
    """Calling _litellm_call with a fully-qualified model id should pass it through."""
    called = {}

    async def fake_acompletion(model, messages=None, timeout=None, **params):
        called["model"] = model
        called["messages"] = messages
        called["params"] = params
        return {"ok": True}

    monkeypatch.setattr(router_mod, "_LITELLM_AVAILABLE", True)
    monkeypatch.setattr(router_mod, "acompletion", fake_acompletion)

    r = router_mod.IntelligentRouter()
    full_id = "openrouter/google/gemma-3n-e4b-it:free"

    await r._litellm_call(full_id, [], {"stream": False})
    assert called.get("model") == full_id


@pytest.mark.asyncio
async def test_litellm_call_with_provider_and_model_kwargs(monkeypatch):
    """Calling _litellm_call with provider+model kwargs creates provider/model id."""
    called = {}

    async def fake_acompletion(model, messages=None, timeout=None, **params):
        called["model"] = model
        called["messages"] = messages
        called["params"] = params
        return {"ok": True}

    monkeypatch.setattr(router_mod, "_LITELLM_AVAILABLE", True)
    monkeypatch.setattr(router_mod, "acompletion", fake_acompletion)

    r = router_mod.IntelligentRouter()

    await r._litellm_call(provider="openai", model="gpt-4o-mini", messages=[], extra=1)
    assert called.get("model") == "openai/gpt-4o-mini"
    assert called.get("params", {}).get("extra") == 1
