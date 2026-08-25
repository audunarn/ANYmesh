"""Provider-neutral, deterministic mesh command protocol.

Natural-language interpretation belongs to LLM adapters.  This package accepts
only strict JSON-compatible commands and publishes revision-bound plans.
"""

from anygeometry.automation import AutomationError, AutomationResponse, PROTOCOL_VERSION

from .schema import (
    automation_dumps,
    automation_json_schema,
    automation_loads,
    describe_capabilities,
    tool_catalog,
)
from .session import MeshAutomationSession, dispatch_tool
from .types import MeshApplyResult, MeshCommand, MeshCommandBatch, MeshPlan

__all__ = [
    "AutomationError",
    "AutomationResponse",
    "MeshApplyResult",
    "MeshAutomationSession",
    "MeshCommand",
    "MeshCommandBatch",
    "MeshPlan",
    "PROTOCOL_VERSION",
    "automation_dumps",
    "automation_json_schema",
    "automation_loads",
    "describe_capabilities",
    "dispatch_tool",
    "tool_catalog",
]
