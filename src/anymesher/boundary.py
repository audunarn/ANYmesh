"""Exact topological boundary and structural-member registries."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np

from .errors import MeshError
from .meshing_view import GeometryMeshingView

__all__ = [
    "BoundaryEntry",
    "BoundaryKey",
    "GlobalEdgeBoundaryRegistry",
    "MemberLocation",
    "MemberRegistry",
    "MemberSpan",
]


@dataclass(frozen=True, slots=True, order=True)
class BoundaryKey:
    """Identity of a station: geometry edge ID plus exact float parameter."""

    edge_id: int
    parameter: float

    def __post_init__(self) -> None:
        edge_id = int(self.edge_id)
        parameter = float(self.parameter)
        if edge_id <= 0:
            raise MeshError("edge ID must be positive")
        if not np.isfinite(parameter) or not 0.0 <= parameter <= 1.0:
            raise MeshError("edge parameter must be finite and in [0, 1]")
        object.__setattr__(self, "edge_id", edge_id)
        object.__setattr__(self, "parameter", 0.0 if parameter == 0.0 else parameter)


@dataclass(frozen=True, slots=True)
class BoundaryEntry:
    key: BoundaryKey
    point: np.ndarray
    node_id: int | None = None
    owners: tuple[Hashable, ...] = ()

    def __post_init__(self) -> None:
        point = np.array(self.point, dtype=float, copy=True)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise MeshError("boundary point must be a finite 3-vector")
        point.flags.writeable = False
        object.__setattr__(self, "point", point)
        if self.node_id is not None and int(self.node_id) <= 0:
            raise MeshError("boundary node ID must be positive")
        if self.node_id is not None:
            object.__setattr__(self, "node_id", int(self.node_id))


class GlobalEdgeBoundaryRegistry:
    """One document-wide station registry shared by all mesh components.

    Parameters are dictionary keys and are never rounded or tolerance-merged.
    Geometry tolerance is used only to verify that repeated registration of the
    *same* topological station describes the same point.
    """

    def __init__(self, view: GeometryMeshingView) -> None:
        if not isinstance(view, GeometryMeshingView):
            raise TypeError("boundary registry requires a GeometryMeshingView")
        self.view = view
        self._entries: dict[BoundaryKey, BoundaryEntry] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def register(
        self,
        edge_id: int,
        parameter: float,
        point: object | None = None,
        *,
        node_id: int | None = None,
        owner: Hashable | None = None,
    ) -> BoundaryEntry:
        key = BoundaryKey(edge_id, parameter)
        made_point = (
            self.view.edge_point(key.edge_id, key.parameter)
            if point is None
            else np.asarray(point, dtype=float)
        )
        candidate = BoundaryEntry(
            key,
            made_point,
            node_id=node_id,
            owners=() if owner is None else (owner,),
        )
        current = self._entries.get(key)
        if current is None:
            self._entries[key] = candidate
            return candidate

        extent = max(self.view.edge_length(key.edge_id), 0.0)
        tolerance = self.view.effective_length(extent)
        residual = float(np.linalg.norm(current.point - candidate.point))
        if residual > tolerance:
            raise MeshError(
                f"edge {key.edge_id} parameter {key.parameter!r} was registered "
                f"at inconsistent points (residual {residual:.7g}, tolerance "
                f"{tolerance:.7g})"
            )
        if (
            current.node_id is not None
            and candidate.node_id is not None
            and current.node_id != candidate.node_id
        ):
            raise MeshError(
                f"edge {key.edge_id} parameter {key.parameter!r} already owns "
                f"mesh node {current.node_id}, not {candidate.node_id}"
            )
        owners = current.owners
        if owner is not None and owner not in owners:
            owners = owners + (owner,)
        updated = replace(
            current,
            node_id=current.node_id if current.node_id is not None else candidate.node_id,
            owners=owners,
        )
        self._entries[key] = updated
        return updated

    def register_many(
        self,
        edge_id: int,
        parameters: Iterable[float],
        *,
        points: Sequence[object] | None = None,
        node_ids: Sequence[int] | None = None,
        owner: Hashable | None = None,
    ) -> tuple[BoundaryEntry, ...]:
        made_parameters = tuple(float(value) for value in parameters)
        made_points = (None,) * len(made_parameters) if points is None else tuple(points)
        made_nodes = (None,) * len(made_parameters) if node_ids is None else tuple(node_ids)
        if len(made_points) != len(made_parameters) or len(made_nodes) != len(
            made_parameters
        ):
            raise MeshError("parameters, points, and node IDs must have equal lengths")
        return tuple(
            self.register(
                edge_id,
                parameter,
                point,
                node_id=node,
                owner=owner,
            )
            for parameter, point, node in zip(
                made_parameters, made_points, made_nodes
            )
        )

    def lookup(self, edge_id: int, parameter: float) -> BoundaryEntry | None:
        return self._entries.get(BoundaryKey(edge_id, parameter))

    def require(self, edge_id: int, parameter: float) -> BoundaryEntry:
        made = self.lookup(edge_id, parameter)
        if made is None:
            raise MeshError(
                f"edge {edge_id} has no registered station at exact parameter "
                f"{float(parameter)!r}"
            )
        return made

    def parameters(self, edge_id: int) -> tuple[float, ...]:
        made_edge = int(edge_id)
        return tuple(
            key.parameter
            for key in sorted(self._entries)
            if key.edge_id == made_edge
        )

    def entries(self, edge_id: int | None = None) -> tuple[BoundaryEntry, ...]:
        keys = sorted(self._entries)
        if edge_id is not None:
            keys = [key for key in keys if key.edge_id == int(edge_id)]
        return tuple(self._entries[key] for key in keys)

    @property
    def mapping(self) -> MappingProxyType:
        return MappingProxyType(dict(self._entries))


@dataclass(frozen=True, slots=True)
class MemberSpan:
    use_id: int
    member_id: int
    edge_id: int
    parent_start: float
    parent_end: float
    forward: bool

    @property
    def length(self) -> float:
        return self.parent_end - self.parent_start

    def contains(self, parameter: float, tolerance: float = 0.0) -> bool:
        return (
            self.parent_start - tolerance
            <= float(parameter)
            <= self.parent_end + tolerance
        )

    def edge_parameter(self, member_parameter: float) -> float:
        local = (float(member_parameter) - self.parent_start) / self.length
        local = float(np.clip(local, 0.0, 1.0))
        return local if self.forward else 1.0 - local

    def member_parameter(self, edge_parameter: float) -> float:
        local = float(edge_parameter) if self.forward else 1.0 - float(edge_parameter)
        return self.parent_start + local * self.length


@dataclass(frozen=True, slots=True)
class MemberLocation:
    span: MemberSpan
    member_parameter: float
    edge_parameter: float

    @property
    def boundary_key(self) -> BoundaryKey:
        return BoundaryKey(self.span.edge_id, self.edge_parameter)


class MemberRegistry:
    """Document-wide member mapping assembled across every MemberEdgeUse."""

    def __init__(self, view: GeometryMeshingView) -> None:
        self.view = view
        by_member: dict[int, tuple[MemberSpan, ...]] = {}
        by_edge: dict[int, list[MemberSpan]] = {}
        seen: set[int] = set()
        for member_id in sorted(view.members):
            spans: list[MemberSpan] = []
            for use in view.edge_uses_for_member(member_id):
                if use.id in seen:
                    raise MeshError(f"member edge use {use.id} has multiple owners")
                seen.add(use.id)
                span = MemberSpan(
                    use_id=int(use.id),
                    member_id=int(member_id),
                    edge_id=int(use.edge_id),
                    parent_start=float(use.parent_range.start),
                    parent_end=float(use.parent_range.end),
                    forward=int(use.orientation) > 0,
                )
                spans.append(span)
                by_edge.setdefault(span.edge_id, []).append(span)
            by_member[member_id] = tuple(spans)
        if seen != set(view.member_edge_uses):
            unowned = sorted(set(view.member_edge_uses) - seen)
            raise MeshError(f"unowned member edge uses: {unowned}")
        self.by_member = MappingProxyType(by_member)
        self.by_edge = MappingProxyType(
            {
                edge: tuple(sorted(spans, key=lambda value: (value.member_id, value.parent_start)))
                for edge, spans in by_edge.items()
            }
        )

    def spans(self, member_id: int) -> tuple[MemberSpan, ...]:
        try:
            return self.by_member[int(member_id)]
        except KeyError as error:
            raise MeshError(f"no member {member_id}") from error

    def uses_on_edge(self, edge_id: int) -> tuple[MemberSpan, ...]:
        return self.by_edge.get(int(edge_id), ())

    def locate(
        self,
        member_id: int,
        parameter: float,
        *,
        prefer: str = "following",
    ) -> MemberLocation:
        made = float(parameter)
        if not np.isfinite(made) or not 0.0 <= made <= 1.0:
            raise MeshError("member parameter must be finite and in [0, 1]")
        if prefer not in {"following", "preceding"}:
            raise ValueError("prefer must be 'following' or 'preceding'")
        spans = self.spans(member_id)
        tolerance = float(self.view.tolerance.parameter)
        candidates = [span for span in spans if span.contains(made, tolerance)]
        if not candidates:
            raise MeshError(
                f"member {member_id} parameter {made!r} is not covered by an edge use"
            )
        if len(candidates) > 1:
            span = candidates[-1] if prefer == "following" else candidates[0]
        else:
            span = candidates[0]
        clamped = float(np.clip(made, span.parent_start, span.parent_end))
        return MemberLocation(span, made, span.edge_parameter(clamped))

    def boundary_key(self, member_id: int, parameter: float) -> BoundaryKey:
        return self.locate(member_id, parameter).boundary_key

    def register(
        self,
        boundaries: GlobalEdgeBoundaryRegistry,
        member_id: int,
        parameter: float,
        *,
        node_id: int | None = None,
    ) -> BoundaryEntry:
        location = self.locate(member_id, parameter)
        return boundaries.register(
            location.span.edge_id,
            location.edge_parameter,
            node_id=node_id,
            owner=("member", int(member_id)),
        )

