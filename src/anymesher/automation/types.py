"""Immutable records for the ANYmesher automation protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
from uuid import UUID

from anygeometry.automation import AutomationError, PROTOCOL_VERSION
from anygeometry.automation.types import canonical_digest, canonical_json

MAX_COMMANDS = 64


def _plain(value: object, *, path: str = "$") -> Any:
    try:
        return __import__("json").loads(canonical_json(value))
    except Exception as error:
        if isinstance(error, AutomationError):
            raise
        raise AutomationError("MALFORMED_REQUEST", str(error), path=path) from error


def _strict(
    value: object,
    required: Sequence[str],
    optional: Sequence[str] = (),
    *,
    path: str = "$",
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AutomationError("MALFORMED_REQUEST", "expected an object", path=path)
    keys = set(value)
    missing = set(required) - keys
    extra = keys - set(required) - set(optional)
    if missing:
        raise AutomationError(
            "MALFORMED_REQUEST", f"missing field(s) {sorted(missing)}", path=path
        )
    if extra:
        raise AutomationError(
            "UNKNOWN_FIELD", f"unknown field(s) {sorted(extra)}", path=path
        )
    return value


def _identifier(value: object, *, path: str, maximum: int = 128) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise AutomationError(
            "MALFORMED_REQUEST", "expected a bounded non-empty string", path=path
        )
    return value


def _revision(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AutomationError(
            "MALFORMED_REQUEST", "expected a non-negative integer", path=path
        )
    return value


def _sha256(value: object, *, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise AutomationError("MALFORMED_PLAN", "expected canonical SHA-256", path=path)
    return value


@dataclass(frozen=True, slots=True)
class MeshCommand:
    name: str
    operation: str
    arguments: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = _identifier(self.name, path="$.name", maximum=64)
        if name[0].isdigit() or not name.replace("_", "a").isalnum():
            raise AutomationError(
                "MALFORMED_COMMAND", "command name must be an identifier", path="$.name"
            )
        _identifier(self.operation, path="$.operation", maximum=64)
        object.__setattr__(self, "arguments", _plain(self.arguments, path="$.arguments"))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "operation": self.operation,
            "arguments": _plain(self.arguments),
        }

    @classmethod
    def from_dict(cls, value: object, *, path: str = "$") -> "MeshCommand":
        made = _strict(value, ("name", "operation", "arguments"), path=path)
        return cls(made["name"], made["operation"], made["arguments"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class MeshCommandBatch:
    protocol_version: int
    request_id: str
    session_id: UUID | str
    model_id: UUID | str
    expected_geometry_revision: int
    expected_state_revision: int
    commands: tuple[MeshCommand, ...]

    def __post_init__(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise AutomationError("UNSUPPORTED", "unsupported protocol version")
        _identifier(self.request_id, path="$.request_id")
        for field_name in ("session_id", "model_id"):
            try:
                object.__setattr__(self, field_name, UUID(str(getattr(self, field_name))))
            except (TypeError, ValueError) as error:
                raise AutomationError(
                    "MALFORMED_REQUEST", "expected UUID", path=f"$.{field_name}"
                ) from error
        _revision(self.expected_geometry_revision, path="$.expected_geometry_revision")
        _revision(self.expected_state_revision, path="$.expected_state_revision")
        commands = tuple(self.commands)
        if not 1 <= len(commands) <= MAX_COMMANDS:
            raise AutomationError(
                "PAYLOAD_TOO_LARGE", f"commands must contain 1..{MAX_COMMANDS} entries"
            )
        if any(not isinstance(item, MeshCommand) for item in commands):
            raise AutomationError("MALFORMED_COMMAND", "invalid command entry")
        if len({item.name for item in commands}) != len(commands):
            raise AutomationError("DUPLICATE_NAME", "command names must be unique")
        object.__setattr__(self, "commands", commands)

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "session_id": str(self.session_id),
            "model_id": str(self.model_id),
            "expected_geometry_revision": self.expected_geometry_revision,
            "expected_state_revision": self.expected_state_revision,
            "commands": [item.to_dict() for item in self.commands],
        }

    @classmethod
    def from_dict(cls, value: object) -> "MeshCommandBatch":
        made = _strict(
            value,
            (
                "protocol_version",
                "request_id",
                "session_id",
                "model_id",
                "expected_geometry_revision",
                "expected_state_revision",
                "commands",
            ),
        )
        raw = made["commands"]
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise AutomationError("MALFORMED_REQUEST", "commands must be an array")
        return cls(
            made["protocol_version"],  # type: ignore[arg-type]
            made["request_id"],  # type: ignore[arg-type]
            made["session_id"],  # type: ignore[arg-type]
            made["model_id"],  # type: ignore[arg-type]
            made["expected_geometry_revision"],  # type: ignore[arg-type]
            made["expected_state_revision"],  # type: ignore[arg-type]
            tuple(
                MeshCommand.from_dict(item, path=f"$.commands[{index}]")
                for index, item in enumerate(raw)
            ),
        )


@dataclass(frozen=True, slots=True)
class MeshPlan:
    protocol_version: int
    request_id: str
    session_id: UUID | str
    model_id: UUID | str
    geometry_revision: int
    state_revision: int
    commands: tuple[MeshCommand, ...]
    resolved_inputs: Mapping[str, tuple[Mapping[str, object], ...]]
    controls_digest: str
    candidate_mesh_digest: str | None
    candidate_summary: Mapping[str, object]
    diagnostics: tuple[str, ...]
    history_action: str
    digest: str

    def digest_payload(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "session_id": str(self.session_id),
            "model_id": str(self.model_id),
            "geometry_revision": self.geometry_revision,
            "state_revision": self.state_revision,
            "commands": [item.to_dict() for item in self.commands],
            "resolved_inputs": {
                name: [_plain(handle) for handle in handles]
                for name, handles in sorted(self.resolved_inputs.items())
            },
            "controls_digest": self.controls_digest,
            "candidate_mesh_digest": self.candidate_mesh_digest,
            "candidate_summary": _plain(self.candidate_summary),
            "diagnostics": list(self.diagnostics),
            "history_action": self.history_action,
        }

    def __post_init__(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise AutomationError("MALFORMED_PLAN", "unsupported protocol version")
        _identifier(self.request_id, path="$.request_id")
        for field_name in ("session_id", "model_id"):
            try:
                object.__setattr__(self, field_name, UUID(str(getattr(self, field_name))))
            except (TypeError, ValueError) as error:
                raise AutomationError(
                    "MALFORMED_PLAN", "expected UUID", path=f"$.{field_name}"
                ) from error
        _revision(self.geometry_revision, path="$.geometry_revision")
        _revision(self.state_revision, path="$.state_revision")
        _sha256(self.controls_digest, path="$.controls_digest")
        if self.candidate_mesh_digest is not None:
            _sha256(self.candidate_mesh_digest, path="$.candidate_mesh_digest")
        if self.history_action not in ("commit", "undo", "redo"):
            raise AutomationError("MALFORMED_PLAN", "invalid history action")
        object.__setattr__(self, "candidate_summary", _plain(self.candidate_summary))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        _sha256(self.digest, path="$.digest")

    def to_dict(self) -> dict[str, object]:
        return {**self.digest_payload(), "digest": self.digest}

    @classmethod
    def from_dict(cls, value: object) -> "MeshPlan":
        made = _strict(
            value,
            (
                "protocol_version",
                "request_id",
                "session_id",
                "model_id",
                "geometry_revision",
                "state_revision",
                "commands",
                "resolved_inputs",
                "controls_digest",
                "candidate_mesh_digest",
                "candidate_summary",
                "diagnostics",
                "history_action",
                "digest",
            ),
        )
        raw_commands = made["commands"]
        raw_resolved = made["resolved_inputs"]
        if not isinstance(raw_commands, Sequence) or isinstance(raw_commands, (str, bytes)):
            raise AutomationError("MALFORMED_PLAN", "commands must be an array")
        if not isinstance(raw_resolved, Mapping):
            raise AutomationError("MALFORMED_PLAN", "resolved_inputs must be an object")
        resolved: dict[str, tuple[Mapping[str, object], ...]] = {}
        for name, handles in raw_resolved.items():
            if not isinstance(name, str) or not isinstance(handles, Sequence) or isinstance(handles, (str, bytes)):
                raise AutomationError("MALFORMED_PLAN", "invalid resolved input")
            if any(not isinstance(item, Mapping) for item in handles):
                raise AutomationError("MALFORMED_PLAN", "resolved handles must be objects")
            resolved[name] = tuple(_plain(item) for item in handles)
        diagnostics = made["diagnostics"]
        if not isinstance(diagnostics, Sequence) or isinstance(diagnostics, (str, bytes)):
            raise AutomationError("MALFORMED_PLAN", "diagnostics must be an array")
        return cls(
            made["protocol_version"],  # type: ignore[arg-type]
            made["request_id"],  # type: ignore[arg-type]
            made["session_id"],  # type: ignore[arg-type]
            made["model_id"],  # type: ignore[arg-type]
            made["geometry_revision"],  # type: ignore[arg-type]
            made["state_revision"],  # type: ignore[arg-type]
            tuple(
                MeshCommand.from_dict(item, path=f"$.commands[{index}]")
                for index, item in enumerate(raw_commands)
            ),
            resolved,
            made["controls_digest"],  # type: ignore[arg-type]
            made["candidate_mesh_digest"],  # type: ignore[arg-type]
            made["candidate_summary"],  # type: ignore[arg-type]
            tuple(str(item) for item in diagnostics),
            made["history_action"],  # type: ignore[arg-type]
            made["digest"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class MeshApplyResult:
    protocol_version: int
    request_id: str
    session_id: str
    model_id: str
    geometry_revision: int
    state_revision_before: int
    state_revision_after: int
    plan_digest: str
    controls_digest: str
    mesh_digest: str | None
    mesh_stale: bool
    history_position: int
    history_size: int
    summary: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "model_id": self.model_id,
            "geometry_revision": self.geometry_revision,
            "state_revision_before": self.state_revision_before,
            "state_revision_after": self.state_revision_after,
            "plan_digest": self.plan_digest,
            "controls_digest": self.controls_digest,
            "mesh_digest": self.mesh_digest,
            "mesh_stale": self.mesh_stale,
            "history_position": self.history_position,
            "history_size": self.history_size,
            "summary": _plain(self.summary),
        }


def make_plan(**values: object) -> MeshPlan:
    payload = dict(values)
    payload["digest"] = "0" * 64
    provisional = MeshPlan(**payload)  # type: ignore[arg-type]
    payload["digest"] = canonical_digest(provisional.digest_payload())
    return MeshPlan(**payload)  # type: ignore[arg-type]


__all__ = [
    "MAX_COMMANDS",
    "MeshApplyResult",
    "MeshCommand",
    "MeshCommandBatch",
    "MeshPlan",
    "make_plan",
]
