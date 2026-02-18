"""Compatibility shim to support `src/` layout when running tests.

This package exists at repository root and redirects imports to the
actual implementation under `src/llm_router`. Pytest (and some editors)
may not automatically add `src/` to sys.path, so this shim ensures
`import llm_router...` resolves correctly.
"""

import os

# Make this package a namespace that includes the real implementation under
# `src/llm_router`. By extending __path__ we allow imports like
# `from llm_router import config` to resolve to files inside src/llm_router.
_real_pkg = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src", "llm_router")
)
if os.path.isdir(_real_pkg) and _real_pkg not in __path__:
    __path__.insert(0, _real_pkg)

__all__ = []
