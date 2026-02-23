"""Public package surface for llm_router.

Expose the primary entry points used by consumers of the package.
"""

__version__ = "0.1.2"

from llm_router.config import PROVIDER_CATALOGUE, settings
from llm_router.models import (
    CachePolicy,
    RoutingOptions,
    RoutingStrategy,
    TaskType,
)
from llm_router.router import IntelligentRouter

__all__ = [
    "__version__",
    "PROVIDER_CATALOGUE",
    "IntelligentRouter",
    "RoutingOptions",
    "RoutingStrategy",
    "TaskType",
    "CachePolicy",
    "settings",
]
