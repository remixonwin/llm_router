import pytest

import llm_router.router as router_mod


@pytest.mark.asyncio
async def test_litellm_model_normalization(monkeypatch):
    """Ensure model ids are normalized to include the provider prefix when
    calling litellm.acompletion.
    """

    called = {}

    async def fake_acompletion(model, messages=None, timeout=None, **params):
        called["model"] = model
        return {"ok": True}

    # Ensure router believes litellm is available and patch the acompletion fn
    monkeypatch.setattr(router_mod, "_LITELLM_AVAILABLE", True)
    monkeypatch.setattr(router_mod, "acompletion", fake_acompletion)

    r = router_mod.IntelligentRouter()

    # Case 1: model id already contains provider segment (google/...)
    await r._litellm_call("openrouter", "google/gemma-3n-e4b-it:free", [], {})
    assert called.get("model") == "openrouter/google/gemma-3n-e4b-it:free"

    # Case 2: bare model id (no slash) should be prefixed
    called.clear()
    await r._litellm_call("openai", "gpt-4o-mini", [], {})
    assert called.get("model") == "openai/gpt-4o-mini"
