"""The geometry model: points, lines, faces, and the operations on them.

Modelling is bottom-up and point-driven, which is the paradigm the mapped
mesher wants anyway:

    points  ->  lines between points  ->  faces bounded by line loops
                                      ->  beams carried on lines

Faces are Coons patches blended from their four boundary sides.  Where the
side curves are straight, a Coons patch reduces exactly to the ruled surface,
so a cylindrical face bounded by two arcs and two straight generators is
represented exactly rather than faceted.  One surface type therefore covers
flat plates, ruled surfaces, cylinders and cones.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np

from .curves import (
    Arc,
    ArcFrame,
    CurveShape,
    Straight,
    arc_frame,
    arc_tangent,
    sample_arc,
    sample_straight,
    straight_tangent,
)
from ..errors import GeometryError
from .entities import Edge, EntityRef, Face, OrientedEdge, Vertex

# GeometryError is re-exported so ``from anymesher.geometry.model import
# GeometryError`` keeps working; it is defined in anymesher.errors so the
# chain-sampling helpers can raise without importing the mesher.
__all__ = ["GeometryError", "GeometryModel"]


def _rotate_about_axis(
    point: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    angle: float,
) -> np.ndarray:
    """Rotate a point about an arbitrary axis (Rodrigues' formula)."""

    offset = np.asarray(point, dtype=float) - origin
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return (
        origin
        + offset * cosine
        + np.cross(direction, offset) * sine
        + direction * float(direction @ offset) * (1.0 - cosine)
    )


class GeometryModel:
    """A container of vertices, edges and faces with persistent IDs."""

    def __init__(self) -> None:
        self.vertices: Dict[int, Vertex] = {}
        self.edges: Dict[int, Edge] = {}
        self.faces: Dict[int, Face] = {}
        self._next_id: Dict[str, int] = {"vertex": 1, "edge": 1, "face": 1}
        self._arc_cache: Dict[int, Tuple[int, ArcFrame]] = {}
        # What each removed entity was replaced by, so attributes attached to
        # it can follow.  Splitting a line that carries a load must not throw
        # the load away.
        self._replacements: List[Tuple[EntityRef, Tuple[EntityRef, ...]]] = []

    # ------------------------------------------------------------------
    # replacement log
    # ------------------------------------------------------------------
    def begin_replacement_log(self) -> None:
        """Start recording what replaces what, for the duration of an edit."""

        self._replacements = []

    def replacement_log(self) -> List[Tuple[EntityRef, Tuple[EntityRef, ...]]]:
        """Entities removed during the current edit, and what took their place."""

        return list(self._replacements)

    def record_replacement(
        self, old: EntityRef, new: Sequence[EntityRef]
    ) -> None:
        """Note that one entity has been superseded by others."""

        self._replacements.append((old, tuple(new)))

    # ------------------------------------------------------------------
    # identity
    # ------------------------------------------------------------------
    def _allocate(self, kind: str) -> int:
        entity_id = self._next_id[kind]
        store = {"vertex": self.vertices, "edge": self.edges, "face": self.faces}[kind]
        if entity_id in store:
            raise GeometryError(
                f"{kind} ID {entity_id} is already in use; the ID counter and "
                "the model have gone out of step"
            )
        self._next_id[kind] = entity_id + 1
        return entity_id

    def id_state(self) -> Dict[str, int]:
        """Snapshot the ID counters.

        Undo restores this so a redone operation re-allocates exactly the same
        IDs, which keeps every attribute reference valid across undo and redo.
        """

        return dict(self._next_id)

    def restore_id_state(self, state: Mapping[str, int]) -> None:
        self._next_id = dict(state)

    def topology_snapshot(self) -> Dict[str, object]:
        """Cheap snapshot of the whole topology, for undo.

        Entity objects are referenced rather than copied; face loops and
        corners are captured as the tuples they are, because operations like
        splitting an edge rewrite them in place.
        """

        return {
            "vertices": dict(self.vertices),
            "edges": dict(self.edges),
            "faces": dict(self.faces),
            "face_state": {
                face_id: (face.loop, face.corners)
                for face_id, face in self.faces.items()
            },
            "ids": dict(self._next_id),
        }

    def restore_topology(self, snapshot: Mapping[str, object]) -> None:
        """Put the model back exactly as ``topology_snapshot`` found it."""

        self.vertices.clear()
        self.vertices.update(snapshot["vertices"])  # type: ignore[arg-type]
        self.edges.clear()
        self.edges.update(snapshot["edges"])  # type: ignore[arg-type]
        self.faces.clear()
        self.faces.update(snapshot["faces"])  # type: ignore[arg-type]
        for face_id, (loop, corners) in snapshot["face_state"].items():  # type: ignore[union-attr]
            self.faces[face_id].loop = loop
            self.faces[face_id].corners = corners
        self._next_id = dict(snapshot["ids"])  # type: ignore[arg-type]
        self._arc_cache.clear()

    def entity_keys(self) -> Set[Tuple[str, int]]:
        """Every entity in the model, as ``(kind, id)`` pairs."""

        return (
            {("vertex", key) for key in self.vertices}
            | {("edge", key) for key in self.edges}
            | {("face", key) for key in self.faces}
        )

    # ------------------------------------------------------------------
    # dependencies and removal
    # ------------------------------------------------------------------
    def edges_using_vertex(self, vertex_id: int) -> List[int]:
        """Edges that reference a vertex, as an end point or as an arc's via."""

        return [
            edge.id
            for edge in self.edges.values()
            if vertex_id in (edge.start, edge.end)
            or (isinstance(edge.curve, Arc) and edge.curve.via_vertex == vertex_id)
        ]

    def faces_using_edge(self, edge_id: int) -> List[int]:
        return [
            face.id
            for face in self.faces.values()
            if any(item.edge == edge_id for item in face.loop)
        ]

    def remove_face(self, face_id: int) -> None:
        self._require_face(face_id)
        del self.faces[face_id]

    def remove_edge(self, edge_id: int) -> None:
        self._require_edge(edge_id)
        users = self.faces_using_edge(edge_id)
        if users:
            raise GeometryError(
                f"cannot remove edge {edge_id}: it bounds face(s) {sorted(users)}"
            )
        del self.edges[edge_id]
        self._arc_cache.pop(edge_id, None)

    def remove_vertex(self, vertex_id: int) -> None:
        self._require_vertex(vertex_id)
        users = self.edges_using_vertex(vertex_id)
        if users:
            raise GeometryError(
                f"cannot remove point {vertex_id}: it is used by edge(s) "
                f"{sorted(users)}"
            )
        del self.vertices[vertex_id]

    def remove_entities(self, keys: Iterable[Tuple[str, int]]) -> None:
        """Remove a set of entities, innermost dependency last."""

        remaining = list(keys)
        order = {"face": 0, "edge": 1, "vertex": 2}
        for kind, entity_id in sorted(remaining, key=lambda k: (order[k[0]], -k[1])):
            if kind == "face":
                self.remove_face(entity_id)
            elif kind == "edge":
                self.remove_edge(entity_id)
            else:
                self.remove_vertex(entity_id)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def add_point(self, x: float, y: float, z: float = 0.0) -> int:
        """Place a point and return its vertex ID."""

        vertex_id = self._allocate("vertex")
        self.vertices[vertex_id] = Vertex(
            id=vertex_id, position=np.array([x, y, z], dtype=float)
        )
        return vertex_id

    def add_points(self, positions: Iterable[Sequence[float]]) -> List[int]:
        """Place several points at once."""

        return [self.add_point(*np.asarray(p, dtype=float)) for p in positions]

    def add_line(self, start: int, end: int) -> int:
        """Connect two points with a straight line."""

        self._require_vertex(start)
        self._require_vertex(end)
        if start == end:
            raise GeometryError("a line needs two distinct points")
        return self._add_edge(start, end, Straight())

    def add_arc(self, start: int, via: int, end: int) -> int:
        """Connect two points with a circular arc through a third point."""

        self._require_vertex(start)
        self._require_vertex(via)
        self._require_vertex(end)
        if len({start, via, end}) != 3:
            raise GeometryError("an arc needs three distinct points")
        # Resolve now so a bad arc is rejected at modelling time rather than
        # at mesh time, where the diagnostic would be far from the cause.
        arc_frame(
            self.vertices[start].position,
            self.vertices[via].position,
            self.vertices[end].position,
        )
        return self._add_edge(start, end, Arc(via_vertex=via))

    def add_polyline(self, vertex_ids: Sequence[int], close: bool = False) -> List[int]:
        """Connect a run of points with straight lines."""

        ids = list(vertex_ids)
        if len(ids) < 2:
            raise GeometryError("a polyline needs at least two points")
        pairs = list(zip(ids, ids[1:]))
        if close:
            pairs.append((ids[-1], ids[0]))
        return [self.add_line(a, b) for a, b in pairs]

    def _add_edge(self, start: int, end: int, curve: CurveShape) -> int:
        edge_id = self._allocate("edge")
        self.edges[edge_id] = Edge(id=edge_id, start=start, end=end, curve=curve)
        return edge_id

    def add_face(
        self,
        edge_ids: Sequence[int],
        corners: Sequence[int] | None = None,
    ) -> int:
        """Create a plate bounded by a closed loop of edges.

        The edges may be given in any order and any direction; the loop is
        ordered here.  ``corners`` optionally overrides the four loop indices
        where the sides begin, for faces whose corners are not obvious from
        boundary turn angle.
        """

        loop = self._order_loop(edge_ids)
        resolved = (
            self._detect_corners(loop)
            if corners is None
            else self._validate_corners(tuple(int(c) for c in corners), len(loop))
        )
        return self._add_face_from_loop(loop, resolved)

    def order_loop(self, edge_ids: Sequence[int]) -> Tuple[OrientedEdge, ...]:
        """Order an unordered edge set into a closed, oriented boundary loop."""

        return self._order_loop(edge_ids)

    def add_face_from_loop(
        self, loop: Sequence[OrientedEdge], corners: Sequence[int]
    ) -> int:
        """Create a face from an explicit oriented loop and corner positions.

        Used by the decomposition tools, which know exactly where the corners
        belong and must not leave it to turn-angle detection.
        """

        ordered = tuple(loop)
        if len(ordered) < 4:
            raise GeometryError("a mapped face needs at least four edges")
        for item in ordered:
            self._require_edge(item.edge)
        for current, following in zip(ordered, ordered[1:] + ordered[:1]):
            if self.oriented_end_vertex(current) != self.oriented_start_vertex(
                following
            ):
                raise GeometryError(
                    f"loop is not continuous at edge {following.edge}"
                )
        return self._add_face_from_loop(
            ordered, self._validate_corners(tuple(int(c) for c in corners), len(ordered))
        )

    def _add_face_from_loop(
        self, loop: Tuple[OrientedEdge, ...], corners: Tuple[int, int, int, int]
    ) -> int:
        face_id = self._allocate("face")
        self.faces[face_id] = Face(id=face_id, loop=loop, corners=corners)
        return face_id

    def add_plate(self, vertex_ids: Sequence[int]) -> int:
        """Create a plate directly from an ordered ring of points.

        Convenience for the common case: the lines are created too.
        """

        edge_ids = self.add_polyline(vertex_ids, close=True)
        return self.add_face(edge_ids)

    # ------------------------------------------------------------------
    # operations
    # ------------------------------------------------------------------
    def extrude(
        self, edge_ids: Sequence[int], vector: Sequence[float]
    ) -> List[int]:
        """Sweep edges along a vector, producing one face per edge.

        Shared points between consecutive edges produce shared swept lines, so
        extruding a chain gives a strip of faces that is conformal by
        construction rather than by coincident-node merging.
        """

        offset = np.asarray(vector, dtype=float)
        if offset.shape != (3,):
            raise GeometryError("extrusion vector must be a 3 component vector")
        if float(np.linalg.norm(offset)) <= 0.0:
            raise GeometryError("extrusion vector must be non-zero")

        swept_vertex: Dict[int, int] = {}
        swept_line: Dict[int, int] = {}

        def translated(vertex_id: int) -> int:
            if vertex_id not in swept_vertex:
                position = self.vertices[vertex_id].position + offset
                swept_vertex[vertex_id] = self.add_point(*position)
            return swept_vertex[vertex_id]

        def connector(vertex_id: int) -> int:
            if vertex_id not in swept_line:
                swept_line[vertex_id] = self.add_line(
                    vertex_id, translated(vertex_id)
                )
            return swept_line[vertex_id]

        face_ids: List[int] = []
        for edge_id in edge_ids:
            edge = self._require_edge(edge_id)
            start_top = translated(edge.start)
            end_top = translated(edge.end)
            if isinstance(edge.curve, Arc):
                via_top = translated(edge.curve.via_vertex)
                top_edge = self.add_arc(start_top, via_top, end_top)
            else:
                top_edge = self.add_line(start_top, end_top)

            loop = (
                OrientedEdge(edge_id, True),
                OrientedEdge(connector(edge.end), True),
                OrientedEdge(top_edge, False),
                OrientedEdge(connector(edge.start), False),
            )
            face_ids.append(self._add_face_from_loop(loop, (0, 1, 2, 3)))
        return face_ids

    def revolve(
        self,
        edge_ids: Sequence[int],
        axis_point: Sequence[float],
        axis_direction: Sequence[float],
        angle: float,
        segments: int | None = None,
    ) -> List[int]:
        """Sweep edges about an axis, producing one face per edge per segment.

        The swept boundaries are true arcs, so a revolved profile is exact
        rather than faceted.  The sweep is cut into segments of at most a
        quarter turn, which keeps every arc well conditioned.
        """

        origin = np.asarray(axis_point, dtype=float)
        direction = np.asarray(axis_direction, dtype=float)
        norm = float(np.linalg.norm(direction))
        if origin.shape != (3,) or direction.shape != (3,):
            raise GeometryError("the revolve axis needs a point and a direction")
        if norm <= 0.0:
            raise GeometryError("the revolve axis direction must be non-zero")
        direction = direction / norm
        if not np.isfinite(angle) or angle == 0.0:
            raise GeometryError("the revolve angle must be non-zero")

        profile = list(dict.fromkeys(int(e) for e in edge_ids))
        for edge_id in profile:
            self._require_edge(edge_id)
        self._reject_on_axis(profile, origin, direction)

        if segments is None:
            segments = max(1, int(np.ceil(abs(angle) / (0.5 * np.pi) - 1.0e-9)))
        segments = int(segments)
        if segments < 1:
            raise GeometryError("a revolve needs at least one segment")
        step = float(angle) / segments

        # A full turn must land back on the profile it started from, otherwise
        # the result is a slit cylinder with a seam of coincident-but-separate
        # points rather than a closed one.
        closes = abs(abs(float(angle)) - 2.0 * np.pi) <= 1.0e-9
        start_edges = list(profile)
        start_vertices = {edge_id: edge_id for edge_id in profile}
        edge_origin = {edge_id: edge_id for edge_id in profile}
        vertex_origin: Dict[int, int] = {}
        for edge_id in profile:
            edge = self.edges[edge_id]
            for vertex_id in (edge.start, edge.end):
                vertex_origin[vertex_id] = vertex_id
        del start_vertices

        face_ids: List[int] = []
        for index in range(segments):
            closing = closes and index == segments - 1
            profile, made, edge_origin, vertex_origin = self._revolve_once(
                profile,
                origin,
                direction,
                step,
                edge_origin=edge_origin,
                vertex_origin=vertex_origin,
                closing=closing,
            )
            face_ids.extend(made)
        del start_edges
        return face_ids

    def _revolve_once(
        self,
        profile: Sequence[int],
        origin: np.ndarray,
        direction: np.ndarray,
        step: float,
        *,
        edge_origin: Dict[int, int],
        vertex_origin: Dict[int, int],
        closing: bool = False,
    ) -> Tuple[List[int], List[int], Dict[int, int], Dict[int, int]]:
        swept_vertex: Dict[int, int] = {}
        swept_arc: Dict[int, int] = {}

        def rotated(vertex_id: int) -> int:
            if closing:
                # Land back on the point this one was swept from.
                return vertex_origin[vertex_id]
            if vertex_id not in swept_vertex:
                position = _rotate_about_axis(
                    self.vertices[vertex_id].position, origin, direction, step
                )
                swept_vertex[vertex_id] = self.add_point(*position)
            return swept_vertex[vertex_id]

        def connector(vertex_id: int) -> int:
            if vertex_id not in swept_arc:
                midpoint = _rotate_about_axis(
                    self.vertices[vertex_id].position,
                    origin,
                    direction,
                    0.5 * step,
                )
                via = self.add_point(*midpoint)
                swept_arc[vertex_id] = self.add_arc(
                    vertex_id, via, rotated(vertex_id)
                )
            return swept_arc[vertex_id]

        next_profile: List[int] = []
        face_ids: List[int] = []
        next_edge_origin: Dict[int, int] = {}
        next_vertex_origin: Dict[int, int] = {}

        for edge_id in profile:
            edge = self.edges[edge_id]
            start_top = rotated(edge.start)
            end_top = rotated(edge.end)
            if closing:
                top_edge = edge_origin[edge_id]
            elif isinstance(edge.curve, Arc):
                top_edge = self.add_arc(
                    start_top, rotated(edge.curve.via_vertex), end_top
                )
            else:
                top_edge = self.add_line(start_top, end_top)

            loop = (
                OrientedEdge(edge_id, True),
                OrientedEdge(connector(edge.end), True),
                OrientedEdge(top_edge, False),
                OrientedEdge(connector(edge.start), False),
            )
            face_ids.append(self._add_face_from_loop(loop, (0, 1, 2, 3)))
            next_profile.append(top_edge)

            next_edge_origin[top_edge] = edge_origin[edge_id]
            next_vertex_origin[start_top] = vertex_origin[edge.start]
            next_vertex_origin[end_top] = vertex_origin[edge.end]
            if isinstance(edge.curve, Arc) and not closing:
                via_top = rotated(edge.curve.via_vertex)
                next_vertex_origin[via_top] = vertex_origin[edge.curve.via_vertex]

        return next_profile, face_ids, next_edge_origin, next_vertex_origin

    def _reject_on_axis(
        self,
        edge_ids: Sequence[int],
        origin: np.ndarray,
        direction: np.ndarray,
        tolerance: float = 1.0e-9,
    ) -> None:
        """A point on the axis would sweep into itself, not into an arc."""

        checked: set[int] = set()
        for edge_id in edge_ids:
            edge = self.edges[edge_id]
            vertices = [edge.start, edge.end]
            if isinstance(edge.curve, Arc):
                vertices.append(edge.curve.via_vertex)
            for vertex_id in vertices:
                if vertex_id in checked:
                    continue
                checked.add(vertex_id)
                offset = self.vertices[vertex_id].position - origin
                radial = offset - float(offset @ direction) * direction
                if float(np.linalg.norm(radial)) <= tolerance:
                    raise GeometryError(
                        f"point {vertex_id} lies on the revolve axis, so it "
                        "would sweep into itself rather than into an arc. "
                        "Move it off the axis, or model the apex region "
                        "separately."
                    )

    # ------------------------------------------------------------------
    # splitting
    # ------------------------------------------------------------------
    def split_edge(
        self, edge_id: int, t: float = 0.5
    ) -> Tuple[int, Tuple[int, int]]:
        """Split a line or arc at parameter ``t``, keeping every face valid.

        Returns the new point and the two replacement edges.  Faces that used
        the original edge have it swapped for the pair in traversal order, and
        their corner indices shift to match, so a side that was one edge simply
        becomes a chain of two.  This is the primitive behind imprinting.
        """

        edge = self._require_edge(edge_id)
        if not 0.0 < float(t) < 1.0:
            raise GeometryError(
                f"split parameter must be strictly between 0 and 1, got {t}"
            )

        new_vertex = self.add_point(*self.sample_edge(edge_id, np.array([t]))[0])
        if isinstance(edge.curve, Arc):
            first_via = self.add_point(
                *self.sample_edge(edge_id, np.array([0.5 * t]))[0]
            )
            second_via = self.add_point(
                *self.sample_edge(edge_id, np.array([0.5 * (1.0 + t)]))[0]
            )
            first = self.add_arc(edge.start, first_via, new_vertex)
            second = self.add_arc(new_vertex, second_via, edge.end)
        else:
            first = self.add_line(edge.start, new_vertex)
            second = self.add_line(new_vertex, edge.end)

        for face in self.faces.values():
            self._replace_edge_in_loop(face, edge_id, first, second)

        del self.edges[edge_id]
        self._arc_cache.pop(edge_id, None)
        self.record_replacement(
            EntityRef("edge", edge_id),
            (EntityRef("edge", first), EntityRef("edge", second)),
        )
        return new_vertex, (first, second)

    @staticmethod
    def _replace_edge_in_loop(
        face: Face, edge_id: int, first: int, second: int
    ) -> None:
        positions = [
            index for index, item in enumerate(face.loop) if item.edge == edge_id
        ]
        for position in reversed(positions):
            item = face.loop[position]
            if item.forward:
                replacement = (OrientedEdge(first, True), OrientedEdge(second, True))
            else:
                # Traversed backwards, the far half comes first.
                replacement = (
                    OrientedEdge(second, False),
                    OrientedEdge(first, False),
                )
            face.loop = face.loop[:position] + replacement + face.loop[position + 1 :]
            # A corner sitting on the split edge still starts where it did;
            # everything after it moves along by one.
            face.corners = tuple(  # type: ignore[assignment]
                corner + 1 if corner > position else corner
                for corner in face.corners
            )

    def set_face_corners(self, face_id: int, corners: Sequence[int]) -> None:
        """Override which loop positions begin each of the four sides."""

        face = self._require_face(face_id)
        face.corners = self._validate_corners(
            tuple(int(c) for c in corners), len(face.loop)
        )

    def face_side_lengths(self, face_id: int) -> Tuple[float, float, float, float]:
        face = self._require_face(face_id)
        return tuple(  # type: ignore[return-value]
            self.side_length(side) for side in face.sides()
        )

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    def vertex_position(self, vertex_id: int) -> np.ndarray:
        return self._require_vertex(vertex_id).position

    def sample_edge(self, edge_id: int, t: np.ndarray) -> np.ndarray:
        """Sample points along an edge in its own direction.

        ``t`` runs from 0 at the start vertex to 1 at the end vertex.  Uniform
        ``t`` gives uniform arc length for both straight lines and arcs.
        """

        edge = self._require_edge(edge_id)
        start = self.vertices[edge.start].position
        end = self.vertices[edge.end].position
        if isinstance(edge.curve, Arc):
            return sample_arc(self._arc_frame(edge), t)
        return sample_straight(start, end, t)

    def edge_length(self, edge_id: int) -> float:
        edge = self._require_edge(edge_id)
        if isinstance(edge.curve, Arc):
            return self._arc_frame(edge).length
        start = self.vertices[edge.start].position
        end = self.vertices[edge.end].position
        return float(np.linalg.norm(end - start))

    def edge_tangent(self, edge_id: int, t: float) -> np.ndarray:
        """Unit tangent along the edge's own direction at parameter ``t``."""

        edge = self._require_edge(edge_id)
        if isinstance(edge.curve, Arc):
            return arc_tangent(self._arc_frame(edge), t)
        return straight_tangent(
            self.vertices[edge.start].position, self.vertices[edge.end].position
        )

    def arc_frame(self, edge_id: int) -> ArcFrame:
        """The resolved circle of an arc edge: centre, radius, axes and sweep.

        Public because a mesh backend that rebuilds the model in another kernel
        needs the circle, not just samples along it.  Raises for a straight edge
        rather than returning a degenerate frame.
        """

        edge = self._require_edge(edge_id)
        if not isinstance(edge.curve, Arc):
            raise GeometryError(f"edge {edge_id} is not an arc")
        return self._arc_frame(edge)

    def _arc_frame(self, edge: Edge) -> ArcFrame:
        """Resolve and cache an arc's circle, invalidated when points move."""

        assert isinstance(edge.curve, Arc)
        stamp = self._geometry_stamp(edge)
        cached = self._arc_cache.get(edge.id)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        frame = arc_frame(
            self.vertices[edge.start].position,
            self.vertices[edge.curve.via_vertex].position,
            self.vertices[edge.end].position,
        )
        self._arc_cache[edge.id] = (stamp, frame)
        return frame

    def _geometry_stamp(self, edge: Edge) -> int:
        assert isinstance(edge.curve, Arc)
        return hash(
            (
                self.vertices[edge.start].position.tobytes(),
                self.vertices[edge.curve.via_vertex].position.tobytes(),
                self.vertices[edge.end].position.tobytes(),
            )
        )

    def move_point(self, vertex_id: int, x: float, y: float, z: float = 0.0) -> None:
        """Move a point; every curve referencing it follows."""

        vertex = self._require_vertex(vertex_id)
        vertex.position = np.array([x, y, z], dtype=float)

    # ------------------------------------------------------------------
    # oriented traversal helpers
    # ------------------------------------------------------------------
    def oriented_start_vertex(self, oriented: OrientedEdge) -> int:
        edge = self._require_edge(oriented.edge)
        return edge.start if oriented.forward else edge.end

    def oriented_end_vertex(self, oriented: OrientedEdge) -> int:
        edge = self._require_edge(oriented.edge)
        return edge.end if oriented.forward else edge.start

    def oriented_start_tangent(self, oriented: OrientedEdge) -> np.ndarray:
        if oriented.forward:
            return self.edge_tangent(oriented.edge, 0.0)
        return -self.edge_tangent(oriented.edge, 1.0)

    def oriented_end_tangent(self, oriented: OrientedEdge) -> np.ndarray:
        if oriented.forward:
            return self.edge_tangent(oriented.edge, 1.0)
        return -self.edge_tangent(oriented.edge, 0.0)

    def face_corner_vertices(self, face_id: int) -> Tuple[int, int, int, int]:
        """The four corner points of a face, in loop order."""

        face = self._require_face(face_id)
        return tuple(  # type: ignore[return-value]
            self.oriented_start_vertex(face.loop[index]) for index in face.corners
        )

    def side_length(self, side: Sequence[OrientedEdge]) -> float:
        return float(sum(self.edge_length(item.edge) for item in side))

    # ------------------------------------------------------------------
    # loop ordering and corner detection
    # ------------------------------------------------------------------
    def _order_loop(self, edge_ids: Sequence[int]) -> Tuple[OrientedEdge, ...]:
        remaining = list(dict.fromkeys(int(e) for e in edge_ids))
        if len(remaining) < 3:
            raise GeometryError("a face needs at least three edges")
        for edge_id in remaining:
            self._require_edge(edge_id)

        first = remaining.pop(0)
        loop = [OrientedEdge(first, True)]
        start_vertex = self.edges[first].start
        current = self.edges[first].end

        while remaining:
            for index, edge_id in enumerate(remaining):
                edge = self.edges[edge_id]
                if edge.start == current:
                    loop.append(OrientedEdge(edge_id, True))
                    current = edge.end
                elif edge.end == current:
                    loop.append(OrientedEdge(edge_id, False))
                    current = edge.start
                else:
                    continue
                remaining.pop(index)
                break
            else:
                raise GeometryError(
                    "edges do not form a single closed loop: "
                    f"no edge continues from vertex {current}"
                )

        if current != start_vertex:
            raise GeometryError(
                "edges do not form a closed loop: the chain ends at vertex "
                f"{current} but starts at vertex {start_vertex}"
            )
        return tuple(loop)

    def _detect_corners(
        self, loop: Tuple[OrientedEdge, ...]
    ) -> Tuple[int, int, int, int]:
        """Pick the four sharpest boundary turns as the mapped-face corners."""

        count = len(loop)
        if count < 4:
            raise GeometryError(
                f"a mapped face needs at least four edges, got {count}; "
                "split the boundary so it forms four sides"
            )
        if count == 4:
            return (0, 1, 2, 3)

        deviations = []
        for index in range(count):
            incoming = self.oriented_end_tangent(loop[index - 1])
            outgoing = self.oriented_start_tangent(loop[index])
            cosine = float(np.clip(incoming @ outgoing, -1.0, 1.0))
            deviations.append(float(np.arccos(cosine)))

        sharpest = sorted(
            sorted(range(count), key=lambda i: (-deviations[i], i))[:4]
        )
        return self._validate_corners(tuple(sharpest), count)

    @staticmethod
    def _validate_corners(
        corners: Tuple[int, ...], loop_length: int
    ) -> Tuple[int, int, int, int]:
        if len(corners) != 4:
            raise GeometryError("a mapped face needs exactly four corners")
        if len(set(corners)) != 4:
            raise GeometryError("face corners must be four distinct loop positions")
        if any(not 0 <= c < loop_length for c in corners):
            raise GeometryError("face corner index outside the boundary loop")
        if list(corners) != sorted(corners):
            raise GeometryError("face corners must be given in loop order")
        return tuple(corners)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # lookup
    # ------------------------------------------------------------------
    def _require_vertex(self, vertex_id: int) -> Vertex:
        try:
            return self.vertices[vertex_id]
        except KeyError:
            raise GeometryError(f"no vertex {vertex_id}") from None

    def _require_edge(self, edge_id: int) -> Edge:
        try:
            return self.edges[edge_id]
        except KeyError:
            raise GeometryError(f"no edge {edge_id}") from None

    def _require_face(self, face_id: int) -> Face:
        try:
            return self.faces[face_id]
        except KeyError:
            raise GeometryError(f"no face {face_id}") from None

    def entity_ref(self, kind: str, entity_id: int) -> EntityRef:
        """Build a reference after checking the entity exists."""

        if kind == "vertex":
            self._require_vertex(entity_id)
        elif kind == "edge":
            self._require_edge(entity_id)
        elif kind == "face":
            self._require_face(entity_id)
        else:
            raise GeometryError(f"unknown entity kind {kind!r}")
        return EntityRef(kind, entity_id)  # type: ignore[arg-type]

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"GeometryModel(vertices={len(self.vertices)}, "
            f"edges={len(self.edges)}, faces={len(self.faces)})"
        )
