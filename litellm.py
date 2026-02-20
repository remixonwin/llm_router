"""Lightweight litellm test shim used in unit tests.

This module provides minimal exception classes the tests import and a
small async stub for `acompletion`. It is NOT a real litellm client —
it's purely for test environments where the real dependency is absent.
"""

from __future__ import annotations

from typing import Any


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
    raise NotImplementedError("mock acompletion should be patched in tests")


async def aembedding(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - test shim
    raise NotImplementedError("mock aembedding should be patched in tests")
