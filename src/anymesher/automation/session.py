"""Geometry-bound, revision-guarded mesh automation sessions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import base64
import json
from math import isfinite
from typing import Callable, Mapping, Sequence
from uuid import UUID, uuid4

from anygeometry import EntityRef, GeometryModel
from anygeometry.automation import (
    AutomationError,
    AutomationResponse,
    PROTOCOL_VERSION,
    Quantity,
    SelectionSpec,
    describe_entities,
    describe_model,
    select_entities,
)
from anygeometry.automation.types import (
    canonical_digest,
    canonical_json,
    handle_from_dict,
    handle_to_dict,
)
from anygeometry.identity import EntityHandle

from ..errors import MeshError
from ..hybrid import generate_hybrid_mesh_result
from ..mesh import Mesh
from ..quality import verify_mesh_quality
from ..refinement import refine_around, refine_at
from ..serialize import mesh_from_dict, mesh_to_dict
from .schema import describe_capabilities
from .types import MeshApplyResult, MeshCommand, MeshCommandBatch, MeshPlan, make_plan

Publisher = Callable[[Mesh | None], None]
CancellationCheck = Callable[[str], object]

_LENGTH_SCALE = {
    "m": 1.0,
    "mm": 1.0e-3,
    "cm": 1.0e-2,
    "in": 0.0254,
    "ft": 0.3048,
}


def _error(code: str, message: str, *, path: str = "$", **details: object) -> AutomationError:
    return AutomationError(code, message, path=path, details=details)


def _strict(
    value: object,
    required: Sequence[str] = (),
    optional: Sequence[str] = (),
    *,
    path: str = "$",
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _error("MALFORMED_REQUEST", "expected an object", path=path)
    missing = set(required) - set(value)
    extra = set(value) - set(required) - set(optional)
    if missing:
        raise _error("MALFORMED_REQUEST", f"missing field(s) {sorted(missing)}", path=path)
    if extra:
        raise _error("UNKNOWN_FIELD", f"unknown field(s) {sorted(extra)}", path=path)
    return value


def _plain(value: object) -> object:
    return json.loads(canonical_json(value))


def _length(value: object, *, path: str, vector: bool = False) -> float | tuple[float, ...]:
    quantity = Quantity.from_dict(value, path=path)
    if quantity.unit not in _LENGTH_SCALE:
        raise _error("UNKNOWN_UNIT", "expected a supported length unit", path=path)
    if quantity.frame not in (None, "model_local"):
        raise _error("UNKNOWN_FRAME", "mesh controls require model_local coordinates", path=path)
    factor = _LENGTH_SCALE[quantity.unit]
    if vector:
        if not isinstance(quantity.value, tuple) or len(quantity.value) != 3:
            raise _error("MALFORMED_QUANTITY", "expected a three-component vector", path=path)
        return tuple(float(item) * factor for item in quantity.value)
    if isinstance(quantity.value, tuple):
        raise _error("MALFORMED_QUANTITY", "expected a scalar", path=path)
    result = float(quantity.value) * factor
    if not isfinite(result):
        raise _error("MALFORMED_QUANTITY", "length must be finite", path=path)
    return result


def _digest_mesh(payload: Mapping[str, object] | None) -> str | None:
    return None if payload is None else canonical_digest(payload)


def _default_controls() -> dict[str, object]:
    return {
        "target_size": None,
        "strategy": "auto",
        "native_backend": "auto",
        "order": "linear",
        "recombine": True,
        "qualified_s3": False,
        "structural_preparation": None,
        "structured_options": None,
        "face_ids": None,
        "member_ids": None,
        "beam_edges": [],
        "seed_overrides": {},
        "refinements": {},
    }


@dataclass(frozen=True)
class _State:
    controls: Mapping[str, object]
    mesh_payload: Mapping[str, object] | None
    mesh_digest: str | None
    stale: bool

    @property
    def bytes(self) -> int:
        return len(canonical_json({"controls": self.controls, "mesh": self.mesh_payload}).encode("utf-8"))


@dataclass(frozen=True)
class _Candidate:
    state: _State
    history_action: str
    history_target: int | None


class MeshAutomationSession:
    """Own one LLM-safe mesh state without exposing mutable internal references."""

    def __init__(
        self,
        geometry: GeometryModel,
        mesh: Mesh | None = None,
        *,
        history_limit: int = 8,
        history_bytes: int = 256 * 1024 * 1024,
        cancellation_check: CancellationCheck | None = None,
    ) -> None:
        if not isinstance(geometry, GeometryModel):
            raise TypeError("geometry must be an ANYgeometry GeometryModel")
        if not 1 <= int(history_limit) <= 64:
            raise ValueError("history_limit must be 1..64")
        if int(history_bytes) <= 0:
            raise ValueError("history_bytes must be positive")
        self.geometry = geometry
        self.session_id = uuid4()
        self.state_revision = 0
        self._history_limit = int(history_limit)
        self._history_bytes = int(history_bytes)
        self._cancellation_check = cancellation_check
        payload = None if mesh is None else mesh_to_dict(mesh)
        if mesh is not None:
            if str(mesh.geometry_model_id) != str(geometry.model_id):
                raise _error("WRONG_MODEL", "initial mesh belongs to another geometry model")
            if mesh.geometry_revision != geometry.revision:
                raise _error("STALE_GEOMETRY", "initial mesh geometry revision is stale")
            mesh_from_dict(payload)
        initial = _State(_default_controls(), payload, _digest_mesh(payload), False)
        self._history = [initial]
        self._history_index = 0
        self._plans: dict[str, _Candidate] = {}
        self._applied: set[str] = set()

    @property
    def _state(self) -> _State:
        return self._history[self._history_index]

    def _cancel(self, phase: str) -> None:
        if self._cancellation_check is not None and self._cancellation_check(phase):
            raise _error("CANCELLED", f"cancelled during {phase}")

    def mesh_snapshot(self) -> Mesh | None:
        payload = self._state.mesh_payload
        return None if payload is None else mesh_from_dict(deepcopy(payload))

    def _mesh_stale(self, state: _State | None = None) -> bool:
        made = self._state if state is None else state
        if made.mesh_payload is None:
            return made.stale
        return made.stale or made.mesh_payload.get("geometry_revision") != self.geometry.revision

    def _mesh_summary(self, state: _State | None = None) -> dict[str, object]:
        made = self._state if state is None else state
        if made.mesh_payload is None:
            return {"present": False, "stale": made.stale, "mesh_digest": None}
        payload = made.mesh_payload
        return {
            "present": True,
            "stale": self._mesh_stale(made),
            "mesh_digest": made.mesh_digest,
            "order": payload["order"],
            "nodes": len(payload["nodes"]),  # type: ignore[arg-type]
            "quads": len(payload["quads"]),  # type: ignore[arg-type]
            "tris": len(payload["tris"]),  # type: ignore[arg-type]
            "beams": len(payload["beams"]),  # type: ignore[arg-type]
            "couplings": len(payload["couplings"]),  # type: ignore[arg-type]
        }

    def summary(self) -> dict[str, object]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "session_id": str(self.session_id),
            "model_id": str(self.geometry.model_id),
            "geometry_revision": self.geometry.revision,
            "state_revision": self.state_revision,
            "controls": deepcopy(self._state.controls),
            "controls_digest": canonical_digest(self._state.controls),
            "mesh": self._mesh_summary(),
            "history": {
                "position": self._history_index,
                "size": len(self._history),
                "can_undo": self._history_index > 0,
                "can_redo": self._history_index + 1 < len(self._history),
                "limit": self._history_limit,
                "byte_limit": self._history_bytes,
                "bytes": sum(item.bytes for item in self._history),
            },
        }

    def _check_header(
        self,
        session_id: object,
        model_id: object,
        geometry_revision: object,
        state_revision: object,
    ) -> None:
        if str(session_id) != str(self.session_id):
            raise _error("WRONG_SESSION", "request belongs to another session")
        if str(model_id) != str(self.geometry.model_id):
            raise _error("WRONG_MODEL", "request belongs to another geometry model")
        if geometry_revision != self.geometry.revision:
            raise _error(
                "STALE_GEOMETRY",
                "geometry revision changed",
                expected=geometry_revision,
                actual=self.geometry.revision,
            )
        if state_revision != self.state_revision:
            raise _error(
                "STALE_REVISION",
                "mesh state revision changed",
                expected=state_revision,
                actual=self.state_revision,
            )

    def _resolve_target(
        self,
        value: object,
        *,
        allowed: set[str],
        request_id: str,
        path: str,
    ) -> tuple[EntityHandle, ...]:
        made = _strict(value, optional=("handles", "selection"), path=path)
        if set(made) == {"handles"}:
            raw = made["handles"]
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                raise _error("MALFORMED_REQUEST", "handles must be an array", path=path)
            result = describe_entities(
                self.geometry,
                raw,  # type: ignore[arg-type]
                request_id=request_id,
                model_id=self.geometry.model_id,
                expected_revision=self.geometry.revision,
                page_size=1000,
            )
        elif set(made) == {"selection"}:
            selection = _strict(
                made["selection"],
                ("where",),
                (
                    "order_by",
                    "descending",
                    "page_size",
                    "cursor",
                    "expected_cardinality",
                    "detail",
                ),
                path=f"{path}.selection",
            )
            spec = SelectionSpec.from_dict(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "request_id": request_id,
                    "model_id": str(self.geometry.model_id),
                    "expected_revision": self.geometry.revision,
                    **selection,
                }
            )
            result = select_entities(self.geometry, spec)
            if result.next_cursor is not None:
                raise _error("PAYLOAD_TOO_LARGE", "selection exceeds one bounded page", path=path)
        else:
            raise _error(
                "MALFORMED_REQUEST",
                "target requires exactly handles or selection",
                path=path,
            )
        handles = tuple(item.handle for item in result.entities)
        wrong = sorted({item.kind for item in handles if item.kind not in allowed})
        if wrong:
            raise _error("UNSUPPORTED", f"target contains unsupported kinds {wrong}", path=path)
        return handles

    def _normalize_command(
        self,
        command: MeshCommand,
        controls: dict[str, object],
    ) -> tuple[MeshCommand, tuple[EntityHandle, ...], bool]:
        args = command.arguments
        operation = command.operation
        resolved: tuple[EntityHandle, ...] = ()
        changed = False
        if operation == "configure":
            made = _strict(
                args,
                optional=(
                    "target_size",
                    "strategy",
                    "native_backend",
                    "order",
                    "recombine",
                    "qualified_s3",
                    "structural_preparation",
                    "structured_options",
                ),
            )
            normalized: dict[str, object] = {}
            if "target_size" in made:
                target = _length(made["target_size"], path="$.target_size")
                if target <= 0.0:  # type: ignore[operator]
                    raise _error("MALFORMED_QUANTITY", "target_size must be positive")
                normalized["target_size"] = target
            for key, values in (
                ("strategy", {"auto", "mapped", "native"}),
                ("native_backend", {"auto", "python", "native"}),
                ("order", {"linear", "quadratic"}),
            ):
                if key in made:
                    if made[key] not in values:
                        raise _error("UNSUPPORTED", f"unsupported {key} {made[key]!r}")
                    normalized[key] = made[key]
            if "recombine" in made:
                if not isinstance(made["recombine"], bool):
                    raise _error("MALFORMED_REQUEST", "recombine must be Boolean")
                normalized["recombine"] = made["recombine"]
            if "qualified_s3" in made:
                if not isinstance(made["qualified_s3"], bool):
                    raise _error(
                        "MALFORMED_REQUEST", "qualified_s3 must be Boolean"
                    )
                normalized["qualified_s3"] = made["qualified_s3"]
            for key in ("structural_preparation", "structured_options"):
                if key in made:
                    value = made[key]
                    if value is not None and not isinstance(value, (bool, Mapping)):
                        raise _error("MALFORMED_REQUEST", f"{key} must be object, Boolean, or null")
                    normalized[key] = _plain(value)
            changed = any(controls.get(key) != value for key, value in normalized.items())
            controls.update(normalized)
            return MeshCommand(command.name, operation, normalized), resolved, changed

        if operation == "set_scope":
            made = _strict(args, optional=("faces", "members", "beam_edges"))
            normalized = {}
            kinds = {"faces": {"face"}, "members": {"member"}, "beam_edges": {"edge"}}
            control_names = {"faces": "face_ids", "members": "member_ids", "beam_edges": "beam_edges"}
            all_handles: list[EntityHandle] = []
            for key, allowed in kinds.items():
                if key not in made:
                    continue
                if made[key] is None:
                    value: object = None if key != "beam_edges" else []
                else:
                    handles = self._resolve_target(
                        made[key], allowed=allowed, request_id=f"{command.name}.{key}", path=f"$.{key}"
                    )
                    all_handles.extend(handles)
                    value = [handle.id for handle in handles]
                    normalized[key] = [handle_to_dict(item) for item in handles]
                target_name = control_names[key]
                changed = changed or controls[target_name] != value
                controls[target_name] = value
            resolved = tuple(sorted(set(all_handles)))
            return MeshCommand(command.name, operation, normalized), resolved, changed

        if operation == "set_edge_divisions":
            made = _strict(args, ("targets", "divisions"))
            divisions = made["divisions"]
            if isinstance(divisions, bool) or not isinstance(divisions, int) or divisions <= 0:
                raise _error("MALFORMED_REQUEST", "divisions must be a positive integer")
            handles = self._resolve_target(
                made["targets"], allowed={"edge"}, request_id=command.name, path="$.targets"
            )
            if not handles:
                raise _error("CARDINALITY_MISMATCH", "edge selection is empty")
            overrides = dict(controls["seed_overrides"])  # type: ignore[arg-type]
            for handle in handles:
                overrides[str(handle.id)] = divisions
            changed = overrides != controls["seed_overrides"]
            controls["seed_overrides"] = overrides
            normalized = {"targets": [handle_to_dict(item) for item in handles], "divisions": divisions}
            return MeshCommand(command.name, operation, normalized), handles, changed

        if operation == "clear_edge_divisions":
            made = _strict(args, optional=("targets",))
            overrides = dict(controls["seed_overrides"])  # type: ignore[arg-type]
            if "targets" not in made:
                resolved = ()
                overrides.clear()
                normalized = {}
            else:
                resolved = self._resolve_target(
                    made["targets"], allowed={"edge"}, request_id=command.name, path="$.targets"
                )
                for handle in resolved:
                    overrides.pop(str(handle.id), None)
                normalized = {"targets": [handle_to_dict(item) for item in resolved]}
            changed = overrides != controls["seed_overrides"]
            controls["seed_overrides"] = overrides
            return MeshCommand(command.name, operation, normalized), resolved, changed

        if operation == "upsert_refinement":
            made = _strict(args, ("name", "size"), ("radius", "target", "center"))
            name = made["name"]
            if not isinstance(name, str) or not name or len(name) > 64:
                raise _error("MALFORMED_REQUEST", "refinement name must be bounded")
            if ("target" in made) == ("center" in made):
                raise _error("MALFORMED_REQUEST", "refinement requires exactly target or center")
            size = _length(made["size"], path="$.size")
            radius = _length(made.get("radius", {"value": 0.0, "unit": "m"}), path="$.radius")
            if size <= 0.0 or radius < 0.0:  # type: ignore[operator]
                raise _error("MALFORMED_QUANTITY", "refinement size/radius are invalid")
            record: dict[str, object] = {"size": size, "radius": radius}
            if "target" in made:
                resolved = self._resolve_target(
                    made["target"],
                    allowed={"vertex", "edge", "face"},
                    request_id=command.name,
                    path="$.target",
                )
                if len(resolved) != 1:
                    raise _error("CARDINALITY_MISMATCH", "refinement target must resolve to one entity")
                record["target"] = handle_to_dict(resolved[0])
            else:
                record["center"] = list(_length(made["center"], path="$.center", vector=True))  # type: ignore[arg-type]
            refinements = dict(controls["refinements"])  # type: ignore[arg-type]
            changed = refinements.get(name) != record
            refinements[name] = record
            controls["refinements"] = dict(sorted(refinements.items()))
            return MeshCommand(command.name, operation, {"name": name, **record}), resolved, changed

        if operation == "remove_refinement":
            made = _strict(args, ("names",))
            names = made["names"]
            if not isinstance(names, Sequence) or isinstance(names, (str, bytes)) or any(not isinstance(item, str) for item in names):
                raise _error("MALFORMED_REQUEST", "names must be a string array")
            refinements = dict(controls["refinements"])  # type: ignore[arg-type]
            for name in names:
                refinements.pop(name, None)
            changed = refinements != controls["refinements"]
            controls["refinements"] = dict(sorted(refinements.items()))
            return MeshCommand(command.name, operation, {"names": sorted(set(names))}), (), changed

        if operation == "clear_refinements":
            _strict(args)
            changed = bool(controls["refinements"])
            controls["refinements"] = {}
            return MeshCommand(command.name, operation, {}), (), changed

        if operation in ("generate", "undo", "redo"):
            _strict(args)
            return MeshCommand(command.name, operation, {}), (), operation == "generate"
        raise _error("UNSUPPORTED", f"unsupported mesh command {operation!r}")

    def _make_refinements(self, controls: Mapping[str, object]) -> tuple[object, ...]:
        result = []
        for record in controls["refinements"].values():  # type: ignore[union-attr]
            if "target" in record:
                handle = handle_from_dict(record["target"])
                result.append(
                    refine_around(
                        EntityRef(handle.kind, handle.id),
                        float(record["size"]),
                        float(record["radius"]),
                    )
                )
            else:
                result.append(
                    refine_at(record["center"], float(record["size"]), float(record["radius"]))
                )
        return tuple(result)

    def _generate(self, controls: Mapping[str, object]) -> tuple[Mapping[str, object], dict[str, object]]:
        if controls["target_size"] is None:
            raise _error("MISSING_CONTROL", "target_size must be configured before generate")
        self._cancel("mesh generation start")
        revision = self.geometry.revision
        options: dict[str, object] = {
            "target_size": controls["target_size"],
            "strategy": controls["strategy"],
            "native_backend": controls["native_backend"],
            "order": controls["order"],
            "recombine": controls["recombine"],
            "qualified_s3": controls["qualified_s3"],
            "overrides": {int(key): value for key, value in controls["seed_overrides"].items()},  # type: ignore[union-attr]
            "refinements": self._make_refinements(controls),
        }
        for key in ("face_ids", "member_ids"):
            if controls[key] is not None:
                options[key] = tuple(controls[key])  # type: ignore[arg-type]
        options["beam_edges"] = tuple(controls["beam_edges"])  # type: ignore[arg-type]
        for key in ("structural_preparation", "structured_options"):
            if controls[key] is not None:
                options[key] = controls[key]
        result = generate_hybrid_mesh_result(self.geometry, **options)
        if self.geometry.revision != revision:
            raise _error("STALE_GEOMETRY", "geometry changed during candidate generation")
        payload = mesh_to_dict(result.mesh)
        mesh_from_dict(payload)
        quality = verify_mesh_quality(result.mesh).as_dict()
        self._cancel("mesh generation complete")
        return payload, {
            **self._mesh_summary(_State(controls, payload, _digest_mesh(payload), False)),
            "quality": quality,
            "hybrid_diagnostics": deepcopy(result.mesh.hybrid_diagnostics),
        }

    def plan(self, batch: MeshCommandBatch) -> MeshPlan:
        if not isinstance(batch, MeshCommandBatch):
            raise _error("MALFORMED_REQUEST", "plan requires MeshCommandBatch")
        self._check_header(
            batch.session_id,
            batch.model_id,
            batch.expected_geometry_revision,
            batch.expected_state_revision,
        )
        operations = [item.operation for item in batch.commands]
        history = [item for item in operations if item in ("undo", "redo")]
        if history and (len(batch.commands) != 1 or len(history) != 1):
            raise _error("MALFORMED_COMMAND", "undo/redo must be a standalone command")
        if operations.count("generate") > 1 or ("generate" in operations and operations[-1] != "generate"):
            raise _error("MALFORMED_COMMAND", "generate may appear once and must be last")

        if history:
            action = history[0]
            target = self._history_index + (-1 if action == "undo" else 1)
            if not 0 <= target < len(self._history):
                raise _error("HISTORY_EMPTY", f"cannot {action}")
            state = self._history[target]
            normalized = batch.commands
            resolved_inputs: dict[str, tuple[Mapping[str, object], ...]] = {}
            summary = self._mesh_summary(state)
            history_action = action
        else:
            controls = deepcopy(self._state.controls)
            normalized_list: list[MeshCommand] = []
            resolved_inputs = {}
            controls_changed = False
            for command in batch.commands:
                normalized_command, handles, changed = self._normalize_command(command, controls)
                normalized_list.append(normalized_command)
                if handles:
                    resolved_inputs[command.name] = tuple(handle_to_dict(item) for item in handles)
                controls_changed = controls_changed or changed
            if operations[-1] == "generate":
                payload, summary = self._generate(controls)
                stale = False
            else:
                payload = self._state.mesh_payload
                summary = self._mesh_summary(
                    _State(controls, payload, _digest_mesh(payload), self._state.stale or controls_changed)
                )
                stale = self._state.stale or controls_changed
            state = _State(controls, payload, _digest_mesh(payload), stale)
            normalized = tuple(normalized_list)
            target = None
            history_action = "commit"

        plan = make_plan(
            protocol_version=PROTOCOL_VERSION,
            request_id=batch.request_id,
            session_id=self.session_id,
            model_id=self.geometry.model_id,
            geometry_revision=self.geometry.revision,
            state_revision=self.state_revision,
            commands=tuple(normalized),
            resolved_inputs=resolved_inputs,
            controls_digest=canonical_digest(state.controls),
            candidate_mesh_digest=state.mesh_digest,
            candidate_summary=summary,
            diagnostics=(),
            history_action=history_action,
        )
        self._plans[plan.digest] = _Candidate(state, history_action, target)
        return plan

    def _trim_history(self) -> None:
        while len(self._history) > 1 and (
            len(self._history) > self._history_limit
            or sum(item.bytes for item in self._history) > self._history_bytes
        ):
            if self._history_index == 0:
                self._history.pop()
            else:
                self._history.pop(0)
                self._history_index -= 1

    def apply(self, plan: MeshPlan, *, publisher: Publisher | None = None) -> MeshApplyResult:
        if not isinstance(plan, MeshPlan):
            raise _error("MALFORMED_PLAN", "apply requires MeshPlan")
        if canonical_digest(plan.digest_payload()) != plan.digest:
            raise _error("TAMPERED_PLAN", "plan digest does not match its payload")
        if plan.digest in self._applied:
            raise _error("ALREADY_APPLIED", "plan has already been applied")
        self._check_header(plan.session_id, plan.model_id, plan.geometry_revision, plan.state_revision)
        candidate = self._plans.get(plan.digest)
        if candidate is None:
            raise _error("STALE_PLAN", "candidate is unavailable in this session")
        if canonical_digest(candidate.state.controls) != plan.controls_digest:
            raise _error("TAMPERED_PLAN", "candidate controls disagree with plan")
        if candidate.state.mesh_digest != plan.candidate_mesh_digest:
            raise _error("TAMPERED_PLAN", "candidate mesh disagrees with plan")
        self._cancel("mesh publication start")
        mesh = None if candidate.state.mesh_payload is None else mesh_from_dict(deepcopy(candidate.state.mesh_payload))
        if publisher is not None:
            try:
                publisher(mesh)
            except Exception as error:
                raise _error("OUTPUT_FAILED", f"mesh output failed: {error}") from error
        before = self.state_revision
        if candidate.history_action == "undo" or candidate.history_action == "redo":
            assert candidate.history_target is not None
            self._history_index = candidate.history_target
        else:
            del self._history[self._history_index + 1 :]
            self._history.append(candidate.state)
            self._history_index = len(self._history) - 1
            self._trim_history()
        self.state_revision += 1
        self._applied.add(plan.digest)
        self._plans.pop(plan.digest, None)
        self._cancel("mesh publication complete")
        return MeshApplyResult(
            PROTOCOL_VERSION,
            plan.request_id,
            str(self.session_id),
            str(self.geometry.model_id),
            self.geometry.revision,
            before,
            self.state_revision,
            plan.digest,
            canonical_digest(self._state.controls),
            self._state.mesh_digest,
            self._mesh_stale(),
            self._history_index,
            len(self._history),
            self._mesh_summary(),
        )

    def _cursor(self, offset: int) -> str:
        raw = f"{self._state.mesh_digest}:{offset}".encode("ascii")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    def _offset(self, cursor: object) -> int:
        if cursor is None:
            return 0
        if not isinstance(cursor, str) or len(cursor) > 512:
            raise _error("STALE_CURSOR", "cursor must be bounded")
        try:
            raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode("ascii")
            digest, offset = raw.rsplit(":", 1)
            if digest != self._state.mesh_digest:
                raise ValueError
            return int(offset)
        except (ValueError, UnicodeError) as error:
            raise _error("STALE_CURSOR", "cursor does not match the current mesh") from error

    def query(self, operation: str, arguments: Mapping[str, object]) -> Mapping[str, object]:
        if operation == "summary":
            _strict(arguments)
            return self.summary()
        if operation == "history":
            _strict(arguments)
            return {
                "position": self._history_index,
                "states": [
                    {
                        "index": index,
                        "controls_digest": canonical_digest(state.controls),
                        "mesh_digest": state.mesh_digest,
                        "stale": self._mesh_stale(state),
                    }
                    for index, state in enumerate(self._history)
                ],
            }
        mesh = self.mesh_snapshot()
        if mesh is None:
            raise _error("NOT_FOUND", "session has no published mesh")
        if operation == "quality":
            _strict(arguments)
            return verify_mesh_quality(mesh).as_dict()
        if operation == "diagnostics":
            _strict(arguments)
            return {
                "structural_preparation": deepcopy(mesh.structural_preparation),
                "hybrid_diagnostics": deepcopy(mesh.hybrid_diagnostics),
            }
        if operation == "associations":
            made = _strict(arguments, ("handle",))
            handle = handle_from_dict(made["handle"])
            describe_entities(
                self.geometry,
                (handle,),
                model_id=self.geometry.model_id,
                expected_revision=self.geometry.revision,
            )
            return {
                "handle": handle_to_dict(handle),
                "nodes": mesh.nodes_on(handle),
                "elements": mesh.elements_on(handle),
            }
        if operation not in ("nodes", "elements"):
            raise _error("UNSUPPORTED", f"unsupported mesh query {operation!r}")
        made = _strict(arguments, optional=("page_size", "cursor"))
        page_size = made.get("page_size", 100)
        if isinstance(page_size, bool) or not isinstance(page_size, int) or not 1 <= page_size <= 1000:
            raise _error("PAYLOAD_TOO_LARGE", "page_size must be 1..1000")
        offset = self._offset(made.get("cursor"))
        if operation == "nodes":
            records = [
                {"id": node_id, "position": [float(item) for item in mesh.nodes[node_id]]}
                for node_id in sorted(mesh.nodes)
            ]
        else:
            records = []
            for kind, values in (("quad", mesh.quads), ("triangle", mesh.tris), ("beam", mesh.beams)):
                records.extend(
                    {"id": element_id, "kind": kind, "connectivity": list(connectivity)}
                    for element_id, connectivity in sorted(values.items())
                )
            records.sort(key=lambda item: (item["id"], item["kind"]))
        page = records[offset : offset + page_size]
        next_offset = offset + len(page)
        return {
            "items": page,
            "total": len(records),
            "next_cursor": None if next_offset >= len(records) else self._cursor(next_offset),
            "mesh_digest": self._state.mesh_digest,
        }


def _session_header(session: MeshAutomationSession, args: Mapping[str, object], optional: Sequence[str]) -> Mapping[str, object]:
    required = (
        "session_id",
        "model_id",
        "expected_geometry_revision",
        "expected_state_revision",
    )
    made = _strict(args, required, optional)
    session._check_header(*(made[key] for key in required))  # noqa: SLF001
    return made


def dispatch_tool(
    session: MeshAutomationSession,
    request: Mapping[str, object],
    *,
    publisher: Publisher | None = None,
) -> Mapping[str, object]:
    request_id = str(request.get("request_id", "unknown"))
    try:
        tool = request["tool"]
        args = request["arguments"]
        if not isinstance(args, Mapping):
            raise _error("MALFORMED_REQUEST", "arguments must be an object")
        if tool == "mesh_capabilities":
            _strict(args)
            result = {**describe_capabilities(), "session": session.summary()}
        elif tool == "mesh_session_summary":
            _session_header(session, args, ())
            result = session.summary()
        elif tool == "select_geometry":
            made = _session_header(
                session,
                args,
                ("where", "order_by", "descending", "page_size", "cursor", "expected_cardinality", "detail"),
            )
            selection = {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": request_id,
                "model_id": str(session.geometry.model_id),
                "expected_revision": session.geometry.revision,
                **{key: value for key, value in made.items() if key not in {
                    "session_id", "model_id", "expected_geometry_revision", "expected_state_revision"
                }},
            }
            result = select_entities(session.geometry, SelectionSpec.from_dict(selection)).to_dict()
        elif tool == "describe_geometry":
            made = _session_header(session, args, ("handles", "detail", "page_size", "cursor"))
            if "handles" not in made:
                result = describe_model(
                    session.geometry,
                    request_id=request_id,
                    model_id=session.geometry.model_id,
                    expected_revision=session.geometry.revision,
                )
            else:
                result = describe_entities(
                    session.geometry,
                    made["handles"],  # type: ignore[arg-type]
                    request_id=request_id,
                    model_id=session.geometry.model_id,
                    expected_revision=session.geometry.revision,
                    detail=bool(made.get("detail", False)),
                    page_size=int(made.get("page_size", 100)),
                    cursor=made.get("cursor"),  # type: ignore[arg-type]
                ).to_dict()
        elif tool == "query_mesh":
            made = _session_header(session, args, ("operation", "query"))
            if "operation" not in made or "query" not in made:
                raise _error("MALFORMED_REQUEST", "query_mesh requires operation and query")
            if not isinstance(made["operation"], str) or not isinstance(made["query"], Mapping):
                raise _error("MALFORMED_REQUEST", "invalid query_mesh arguments")
            result = session.query(made["operation"], made["query"])
        elif tool == "plan_mesh":
            made = _session_header(session, args, ("commands",))
            batch = MeshCommandBatch.from_dict(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "request_id": request_id,
                    **made,
                }
            )
            result = session.plan(batch).to_dict()
        elif tool == "apply_mesh":
            made = _session_header(session, args, ("plan",))
            if "plan" not in made:
                raise _error("MALFORMED_REQUEST", "apply_mesh requires plan")
            result = session.apply(MeshPlan.from_dict(made["plan"]), publisher=publisher).to_dict()
        else:
            raise _error("UNSUPPORTED", f"unknown tool {tool!r}")
        return AutomationResponse(PROTOCOL_VERSION, request_id, True, result).to_dict()
    except AutomationError as error:
        return AutomationResponse(PROTOCOL_VERSION, request_id, False, error=error).to_dict()
    except (MeshError, ValueError, TypeError) as error:
        made = _error("OPERATION_FAILED", str(error))
        return AutomationResponse(PROTOCOL_VERSION, request_id, False, error=made).to_dict()
    except Exception as error:  # provider and native failures remain visible and typed
        made = _error("INTERNAL_ERROR", f"{type(error).__name__}: {error}")
        return AutomationResponse(PROTOCOL_VERSION, request_id, False, error=made).to_dict()


__all__ = ["MeshAutomationSession", "dispatch_tool"]
