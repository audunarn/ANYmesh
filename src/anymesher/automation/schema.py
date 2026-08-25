"""Strict JSON transport and provider-neutral tool discovery."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Mapping

from anygeometry.automation import AutomationError, PROTOCOL_VERSION
from anygeometry.automation.types import canonical_json

from .types import MAX_COMMANDS

TOOLS = (
    "mesh_capabilities",
    "mesh_session_summary",
    "select_geometry",
    "describe_geometry",
    "query_mesh",
    "plan_mesh",
    "apply_mesh",
)


def _header_properties() -> dict[str, object]:
    return {
        "session_id": {"type": "string", "format": "uuid"},
        "model_id": {"type": "string", "format": "uuid"},
        "expected_geometry_revision": {"type": "integer", "minimum": 0},
        "expected_state_revision": {"type": "integer", "minimum": 0},
    }


def automation_json_schema() -> Mapping[str, object]:
    command = {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "operation", "arguments"],
        "properties": {
            "name": {"type": "string", "minLength": 1, "maxLength": 64},
            "operation": {
                "enum": [
                    "configure",
                    "set_scope",
                    "set_edge_divisions",
                    "clear_edge_divisions",
                    "upsert_refinement",
                    "remove_refinement",
                    "clear_refinements",
                    "generate",
                    "undo",
                    "redo",
                ]
            },
            "arguments": {"type": "object"},
        },
    }
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://github.com/audunarn/ANYmesh/schemas/automation-v1.json",
        "title": "ANYmesher Automation Protocol",
        "type": "object",
        "additionalProperties": False,
        "required": ["protocol_version", "request_id", "tool", "arguments"],
        "properties": {
            "protocol_version": {"const": PROTOCOL_VERSION},
            "request_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "tool": {"enum": list(TOOLS)},
            "arguments": {"type": "object"},
        },
        "$defs": {
            "session_header": {
                "type": "object",
                "additionalProperties": True,
                "required": list(_header_properties()),
                "properties": _header_properties(),
            },
            "command": command,
            "batch": {
                "allOf": [{"$ref": "#/$defs/session_header"}],
                "type": "object",
                "required": [*list(_header_properties()), "commands"],
                "properties": {
                    **_header_properties(),
                    "commands": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_COMMANDS,
                        "items": command,
                    },
                },
            },
            "response": {
                "type": "object",
                "additionalProperties": False,
                "required": ["protocol_version", "request_id", "ok", "result", "error"],
                "properties": {
                    "protocol_version": {"const": PROTOCOL_VERSION},
                    "request_id": {"type": "string"},
                    "ok": {"type": "boolean"},
                    "result": {},
                    "error": {"type": ["object", "null"]},
                },
            },
        },
    }
    return deepcopy(schema)


def tool_catalog() -> tuple[Mapping[str, object], ...]:
    descriptions = {
        "mesh_capabilities": "Discover mesh commands, limits, schemas, and the bound session.",
        "mesh_session_summary": "Inspect controls, revisions, mesh identity, and history.",
        "select_geometry": "Run ANYgeometry's bounded deterministic selector.",
        "describe_geometry": "Describe canonical geometry handles used by mesh controls.",
        "query_mesh": "Inspect quality, associations, diagnostics, nodes, elements, or history.",
        "plan_mesh": "Create a revision-bound non-mutating mesh command plan.",
        "apply_mesh": "Publish one verified plan exactly once.",
    }
    mutating = {"apply_mesh"}
    return tuple(
        {
            "protocol_version": PROTOCOL_VERSION,
            "name": name,
            "description": descriptions[name],
            "mutating": name in mutating,
            "strict": True,
            "input_schema": {"type": "object"},
            "output_schema": {"$ref": "#/$defs/response"},
        }
        for name in TOOLS
    )


def describe_capabilities() -> Mapping[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "provider_neutral": True,
        "natural_language_interpretation": False,
        "geometry_automation_version": 1,
        "mesh_command_operations": [
            "configure",
            "set_scope",
            "set_edge_divisions",
            "clear_edge_divisions",
            "upsert_refinement",
            "remove_refinement",
            "clear_refinements",
            "generate",
            "undo",
            "redo",
        ],
        "mesh_query_operations": [
            "summary",
            "quality",
            "diagnostics",
            "associations",
            "nodes",
            "elements",
            "history",
        ],
        "limits": {
            "payload_bytes": 1_048_576,
            "maximum_commands": MAX_COMMANDS,
            "default_page_size": 100,
            "maximum_page_size": 1_000,
        },
        "tools": list(tool_catalog()),
    }


def automation_dumps(value: object) -> str:
    return canonical_json(value)


def automation_loads(payload: str) -> Mapping[str, object]:
    if not isinstance(payload, str) or len(payload.encode("utf-8")) > 1_048_576:
        raise AutomationError(
            "PAYLOAD_TOO_LARGE", "automation request must not exceed 1 MiB"
        )

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise AutomationError(
                    "MALFORMED_REQUEST", f"duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        data = json.loads(
            payload,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                AutomationError("MALFORMED_REQUEST", f"non-finite value {value}")
            ),
        )
    except AutomationError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise AutomationError("MALFORMED_REQUEST", f"invalid JSON: {error}") from error
    if not isinstance(data, Mapping):
        raise AutomationError("MALFORMED_REQUEST", "request must be an object")
    required = {"protocol_version", "request_id", "tool", "arguments"}
    if set(data) != required:
        extra = sorted(set(data) - required)
        missing = sorted(required - set(data))
        code = "UNKNOWN_FIELD" if extra else "MALFORMED_REQUEST"
        raise AutomationError(code, f"request fields mismatch; missing={missing}, extra={extra}")
    if data["protocol_version"] != PROTOCOL_VERSION:
        raise AutomationError("UNSUPPORTED", "unsupported protocol version")
    if not isinstance(data["request_id"], str) or not 1 <= len(data["request_id"]) <= 128:
        raise AutomationError("MALFORMED_REQUEST", "request_id must be bounded")
    if data["tool"] not in TOOLS:
        raise AutomationError("UNSUPPORTED", f"unknown tool {data['tool']!r}")
    if not isinstance(data["arguments"], Mapping):
        raise AutomationError("MALFORMED_REQUEST", "arguments must be an object")
    return data


__all__ = [
    "TOOLS",
    "automation_dumps",
    "automation_json_schema",
    "automation_loads",
    "describe_capabilities",
    "tool_catalog",
]
