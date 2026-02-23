"""Lightweight litellm test shim used in unit tests.

This module provides minimal exception classes the tests import and a
small async stub for `acompletion`. It is NOT a real litellm client —
it's purely for test environments where the real dependency is absent.

To avoid accidentally using the test shim in production, importing this
module will raise ImportError unless running under pytest (detected by
`PYTEST_CURRENT_TEST`) or the environment variable
`LLM_ROUTER_ALLOW_TEST_SHIM=1` is set explicitly.

When active (in test runs) the shim provides simple deterministic
responses so integration tests can run without the real litellm lib.
"""

from __future__ import annotations

import os
import asyncio
from typing import Any

# Prevent accidental import outside of tests
if "PYTEST_CURRENT_TEST" not in os.environ and os.getenv("LLM_ROUTER_ALLOW_TEST_SHIM", "") != "1":
    raise ImportError("litellm test shim should not be imported outside of tests")


class RateLimitError(Exception):
    def __init__(self, message: str = "rate limited", **kwargs: Any) -> None:
        super().__init__(message)
        for k, v in kwargs.items():
            setattr(self, k, v)


class APIConnectionError(Exception):
    pass


class AuthenticationError(Exception):
    pass


class Timeout(Exception):
    pass


async def acompletion(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - test shim
    """Return a deterministic chat response or an async generator when
    `stream=True`. Compatible shape with router expectations.
    """
    # Accept either (model, messages, params) or kwargs
    params = {}
    if len(args) >= 3:
        params = dict(args[2] or {})
    params.update({k: v for k, v in kwargs.items() if k not in ("model", "messages")})

    stream = bool(params.get("stream", False) or kwargs.get("stream", False))
    messages = kwargs.get("messages") or (args[1] if len(args) > 1 else [])

    if stream:
        async def _gen():
            # yield a few small chunks then finish
            await asyncio.sleep(0)
            yield {"choices": [{"message": {"content": "chunk1"}}]}
            await asyncio.sleep(0)
            yield {"choices": [{"message": {"content": "chunk2"}}]}
            await asyncio.sleep(0)
            # Some callers expect a final sentinel — provide a simple token
            yield {"choices": [{"message": {"content": "[DONE]"}}]}

        return _gen()

    # Non-streaming: return a single full response
    return {"choices": [{"message": {"content": "Hello from litellm test shim"}}]}


async def aembedding(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - test shim
    """Return a deterministic embedding response shape.

    Expected shape: {"data": [{"embedding": [...]}, ...]}
    """
    input_val = kwargs.get("input") or (args[1] if len(args) > 1 else "")
    texts = input_val if isinstance(input_val, list) else [input_val]
    # Return small fixed-dimension embeddings
    emb = [0.01 * (i + 1) for i in range(8)]
    return {"data": [{"embedding": emb} for _ in texts]}
