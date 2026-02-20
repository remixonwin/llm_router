"""Public package surface for llm_router.

Expose the primary entry points used by consumers of the package.
"""

from llm_router.config import PROVIDER_CATALOGUE, settings
from llm_router.models import RoutingOptions, RoutingStrategy, TaskType
from llm_router.router import IntelligentRouter

__all__ = [
    "PROVIDER_CATALOGUE",
    "IntelligentRouter",
    "RoutingOptions",
    "RoutingStrategy",
    "TaskType",
    "settings",
]
