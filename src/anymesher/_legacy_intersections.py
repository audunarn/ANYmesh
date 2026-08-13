"""Deprecated clone/imprint compatibility for pre-0.2 ANYmesher callers.

This module is deliberately private and is never selected by the production
hybrid backend. It preserves historical behavior for an explicit migration
window only; new code must consume ANYgeometry's qualified intersection,
imprint, structural-record, tolerance and spatial-index contracts instead.

The design model deliberately does not weld coincident entities.  Meshing is
different: shell faces that physically cross must share a topology edge, beam
spans need nodes at joints, and an embedded beam needs a shell interpolation
attachment. This module prepares those relationships on a working clone and
folds the mesh associations back onto the original design owners. The user's
geometry and persistent entity IDs are therefore unchanged by meshing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

import numpy as np
from anygeometry import IntersectionKind, MutationPolicy, query_intersection
from anygeometry.entities import EntityRef
from anygeometry.errors import GeometryError
from anygeometry.intersections import intersect_faces
from anygeometry.model import GeometryModel
from anygeometry.overlaps import find_coplanar_overlaps
from anygeometry.curves import Straight

from .beam_connections import connect_beam_mesh, connect_shell_boundaries
from .mapped import Mesh, MeshError, generate_mesh
from .refinement import Refinement
from .seeding import Seeding

__all__ = ["generate_mesh_with_intersections"]


@dataclass(frozen=True)
class _PreparedGeometry:
    geometry: GeometryModel
    original_faces: tuple[int, ...]
    original_edges: tuple[int, ...]
    intersections: int

    def descendants(self, kind: str, identifier: int) -> tuple[int, ...]:
        return tuple(
            item.id
            for item in self.geometry.resolve_ref(EntityRef(kind, int(identifier)))
            if item.kind == kind
        )


def _face_bounds(geometry: GeometryModel, face_id: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.vstack(
        [
            geometry.sample_edge(item.edge, np.linspace(0.0, 1.0, 5))
            for item in geometry.faces[face_id].loop
        ]
    )
    return points.min(axis=0), points.max(axis=0)


def _face_length_scale(geometry: GeometryModel, face_id: int) -> float:
    lower, upper = _face_bounds(geometry, face_id)
    return max(float(np.max(upper - lower)), 1.0)

def _candidate_pairs(geometry: GeometryModel) -> list[tuple[int, int]]:
    """Broad-phase candidates from the model spatial index."""

    candidates: list[tuple[int, int]] = []
    for first in sorted(geometry.faces):
        lower, upper = _face_bounds(geometry, first)
        for kind, second in geometry.spatial_candidates(lower, upper, kinds=("face",)):
            if kind == "face" and second > first:
                candidates.append((first, int(second)))
    return candidates

def _shared_loop_edge(geometry: GeometryModel, first: int, second: int) -> bool:
    first_edges = {item.edge for item in geometry.faces[first].loop}
    return any(item.edge in first_edges for item in geometry.faces[second].loop)


def _is_one_sided_shell_junction(
    geometry: GeometryModel, first: int, second: int
) -> bool:
    """Whether the full intersection lies on one existing boundary edge."""

    tolerance = geometry.tolerance.effective_length(
        max(_face_length_scale(geometry, first), _face_length_scale(geometry, second))
    )
    try:
        endpoints = intersect_faces(geometry, first, second, fragment=False)
    except GeometryError:
        return False
    if len(endpoints) != 2:
        return False
    for face_id in (first, second):
        for item in geometry.faces[face_id].loop:
            if all(
                geometry.closest_edge_point(item.edge, point)[2]
                <= tolerance
                for point in endpoints
            ):
                return True
    return False


def _planar_face(
    geometry: GeometryModel, face_id: int
) -> tuple[np.ndarray, np.ndarray] | None:
    points = np.vstack(
        [
            geometry.sample_edge(item.edge, np.linspace(0.0, 1.0, 5))
            for item in geometry.faces[face_id].loop
        ]
    )
    origin = points.mean(axis=0)
    _left, singular, vectors = np.linalg.svd(points - origin)
    if singular[-1] > geometry.tolerance.effective_surface_residual(
        max(float(singular[0]), 1.0)
    ):
        return None
    return origin, vectors[-1] / np.linalg.norm(vectors[-1])


def _split_beams_at_shell_crossings(
    geometry: GeometryModel, original_beams: Sequence[int]
) -> int:
    """Put a beam node wherever a straight beam crosses a planar shell."""

    split_count = 0
    while True:
        changed = False
        for original in original_beams:
            for reference in geometry.resolve_ref(EntityRef("edge", original)):
                edge = geometry.edges[reference.id]
                if not isinstance(edge.curve, Straight):
                    continue
                start = geometry.vertex_position(edge.start)
                end = geometry.vertex_position(edge.end)
                direction = end - start
                scale = max(float(np.linalg.norm(direction)), 1.0)
                tolerance = geometry.tolerance.effective_length(scale)
                boundary_tolerance = geometry.tolerance.effective_parameter(scale, scale)
                boundary_faces = set(geometry.faces_using_edge(edge.id))
                for face_id in tuple(geometry.faces):
                    if face_id in boundary_faces:
                        continue
                    plane = _planar_face(geometry, face_id)
                    if plane is None:
                        continue
                    origin, normal = plane
                    first = float((start - origin) @ normal)
                    second = float((end - origin) @ normal)
                    if abs(first) <= tolerance and abs(second) <= tolerance:
                        # A coplanar embedded beam is connected after meshing at
                        # all of its stations by interpolation couplings.
                        continue
                    denominator = first - second
                    if abs(denominator) <= tolerance:
                        continue
                    parameter = first / denominator
                    if parameter <= boundary_tolerance or parameter >= 1.0 - boundary_tolerance:
                        continue
                    point = start + parameter * direction
                    projected, uv, distance = geometry.project_to_face(face_id, point)
                    if (
                        distance > geometry.tolerance.effective_length(scale)
                        or not geometry.face_contains_uv(face_id, uv)
                    ):
                        continue
                    geometry.split_edge(edge.id, float(parameter))
                    split_count += 1
                    changed = True
                    break
                if changed:
                    break
            if changed:
                break
        if not changed:
            return split_count


def _split_beam_end_on_beam(
    geometry: GeometryModel, original_beams: Sequence[int]
) -> int:
    """Split a target beam when another beam endpoint terminates on it."""

    split_count = 0
    while True:
        current = [
            item.id
            for original in original_beams
            for item in geometry.resolve_ref(EntityRef("edge", original))
        ]
        changed = False
        for source_id in current:
            source = geometry.edges[source_id]
            for vertex_id in (source.start, source.end):
                point = geometry.vertex_position(vertex_id)
                for target_id in current:
                    if target_id == source_id:
                        continue
                    target = geometry.edges[target_id]
                    if vertex_id in (target.start, target.end):
                        continue
                    _closest, parameter, distance = geometry.closest_edge_point(
                        target_id, point
                    )
                    scale = max(geometry.edge_length(target_id), 1.0)
                    tolerance = geometry.tolerance.effective_length(scale)
                    boundary_tolerance = geometry.tolerance.effective_parameter(scale, scale)
                    if (
                        distance <= tolerance
                        and boundary_tolerance < parameter < 1.0 - boundary_tolerance
                    ):
                        geometry.split_edge(target_id, float(parameter))
                        split_count += 1
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
        if not changed:
            return split_count


def _split_crossing_beams(
    geometry: GeometryModel, original_beams: Sequence[int]
) -> int:
    """Split straight beam spans where their interiors physically cross."""

    split_count = 0
    while True:
        current = sorted(
            {
                item.id
                for original in original_beams
                for item in geometry.resolve_ref(EntityRef("edge", original))
            }
        )
        changed = False
        for index, first_id in enumerate(current):
            first = geometry.edges[first_id]
            if not isinstance(first.curve, Straight):
                continue
            p0 = geometry.vertex_position(first.start)
            p1 = geometry.vertex_position(first.end)
            u = p1 - p0
            for second_id in current[index + 1 :]:
                second = geometry.edges[second_id]
                if not isinstance(second.curve, Straight):
                    continue
                q0 = geometry.vertex_position(second.start)
                q1 = geometry.vertex_position(second.end)
                v = q1 - q0
                w = p0 - q0
                a = float(u @ u)
                b = float(u @ v)
                c = float(v @ v)
                d = float(u @ w)
                e = float(v @ w)
                denominator = a * c - b * b
                scale = max(np.sqrt(a), np.sqrt(c), 1.0)
                boundary_tolerance = geometry.tolerance.effective_parameter(scale, scale)
                if abs(denominator) <= geometry.tolerance.relative_length * max(a * c, 1.0):
                    # Parallel/collinear spans have no unique joint. Exact end
                    # coincidence is handled by the node-merging phase.
                    continue
                first_parameter = (b * e - c * d) / denominator
                second_parameter = (a * e - b * d) / denominator
                if not (
                    -boundary_tolerance <= first_parameter <= 1.0 + boundary_tolerance
                    and -boundary_tolerance <= second_parameter <= 1.0 + boundary_tolerance
                ):
                    continue
                first_point = p0 + first_parameter * u
                second_point = q0 + second_parameter * v
                if float(np.linalg.norm(first_point - second_point)) > geometry.tolerance.effective_length(
                    scale
                ):
                    continue
                if boundary_tolerance < first_parameter < 1.0 - boundary_tolerance:
                    geometry.split_edge(first_id, float(first_parameter))
                    split_count += 1
                    changed = True
                    break
                if boundary_tolerance < second_parameter < 1.0 - boundary_tolerance:
                    geometry.split_edge(second_id, float(second_parameter))
                    split_count += 1
                    changed = True
                    break
            if changed:
                break
        if not changed:
            return split_count


def _prepare(
    geometry: GeometryModel,
    face_ids: Iterable[int] | None,
    beam_edges: Sequence[int],
) -> _PreparedGeometry:
    original_faces = tuple(
        geometry.faces if face_ids is None else dict.fromkeys(int(item) for item in face_ids)
    )
    overlaps = find_coplanar_overlaps(geometry, original_faces)
    if overlaps:
        detail = ", ".join(
            f"plates {item.first}/{item.second}: {item.area:.7g} m^2"
            for item in overlaps[:5]
        )
        more = "" if len(overlaps) <= 5 else f"; plus {len(overlaps) - 5} more"
        raise MeshError(
            "positive-area coplanar plate overlap would create duplicate shell "
            f"stiffness ({detail}{more}). In Geometry > Operations, select the "
            "plates in section-ownership order and use 'Fragment plate overlaps'."
        )
    original_edges = tuple(geometry.edges)
    working = geometry.clone(include_features=False)
    working.begin_replacement_log()
    _split_beam_end_on_beam(working, beam_edges)
    _split_crossing_beams(working, beam_edges)
    _split_beams_at_shell_crossings(working, beam_edges)
    count = 0
    # One imprint replaces both faces, so restart the broad phase after each
    # success.  This also discovers a third plate crossing the new fragments.
    maximum = max(1_000, 8 * max(len(original_faces), 1) ** 2)
    while True:
        changed = False
        for first, second in _candidate_pairs(working):
            if _shared_loop_edge(working, first, second):
                continue
            if _is_one_sided_shell_junction(working, first, second):
                # The complete intersection is already a boundary of at
                # least one plate.  Imprinting that same segment would create
                # a zero-width face (notably when adjacent edges were
                # extruded in separate modelling commands).  The retained
                # shell-boundary coupling stage makes the meshes conformal.
                continue
            result = query_intersection(
                working,
                working.handle("face", first),
                working.handle("face", second),
            )
            if result.classified and result.kind is IntersectionKind.DISJOINT:
                continue
            try:
                intersect_faces(
                    working,
                    first,
                    second,
                    fragment=True,
                    policy=MutationPolicy.IMPRINT,
                )
            except GeometryError as error:
                message = str(error)
                if (
                    "intersection endpoint is not on the face boundary" in message
                    and _is_one_sided_shell_junction(working, first, second)
                ):
                    # A boundary edge terminating inside another shell is tied
                    # after meshing by interpolation MPCs. The other face stays
                    # a valid mapped patch rather than being arbitrarily cut.
                    continue
                raise MeshError(
                    "automatic plate-intersection imprint failed for model "
                    f"plates {first} and {second}: {message}. Partition this "
                    "junction into conformal four-sided regions before meshing."
                ) from error
            count += 1
            if count > maximum:
                raise MeshError(
                    "automatic plate-intersection imprint did not settle; "
                    "check for duplicate or overlapping plates"
                )
            changed = True
            break
        if not changed:
            break
    return _PreparedGeometry(working, original_faces, original_edges, count)


def _remap_refinements(
    prepared: _PreparedGeometry, refinements: Iterable[Refinement]
) -> list[Refinement]:
    result: list[Refinement] = []
    for refinement in refinements:
        if refinement.ref is None:
            result.append(refinement)
            continue
        descendants = prepared.geometry.resolve_ref(refinement.ref)
        result.extend(replace(refinement, ref=item) for item in descendants)
    return result


def _remap_overrides(
    prepared: _PreparedGeometry, overrides: Mapping[int, int] | None
) -> dict[int, int] | None:
    if overrides is None:
        return None
    remapped: dict[int, int] = {}
    for edge_id, requested in overrides.items():
        descendants = prepared.descendants("edge", int(edge_id))
        if not descendants:
            continue
        if len(descendants) == 1:
            remapped[descendants[0]] = int(requested)
            continue
        lengths = np.asarray(
            [prepared.geometry.edge_length(item) for item in descendants], dtype=float
        )
        total = float(lengths.sum())
        raw = lengths / total * max(int(requested), len(descendants))
        counts = np.maximum(1, np.floor(raw).astype(int))
        remainder = max(int(requested), len(descendants)) - int(counts.sum())
        for index in np.argsort(-(raw - counts))[: max(remainder, 0)]:
            counts[int(index)] += 1
        for identifier, count in zip(descendants, counts):
            remapped[identifier] = int(count)
    return remapped


def _ordered_nodes_on_original_edge(
    geometry: GeometryModel, edge_id: int, mesh: Mesh, node_ids: Iterable[int]
) -> list[int]:
    unique = set(int(item) for item in node_ids)
    return sorted(
        unique,
        key=lambda node_id: geometry.closest_edge_point(
            edge_id, mesh.nodes[node_id]
        )[1],
    )


def _fold_associations(
    mesh: Mesh,
    original: GeometryModel,
    prepared: _PreparedGeometry,
    beam_edges: Sequence[int],
) -> Mesh:
    child_faces = dict(mesh.elements_of_face)
    child_grids = dict(mesh.grid_of_face)
    mesh.elements_of_face = {}
    mesh.grid_of_face = {}
    for face_id in prepared.original_faces:
        descendants = prepared.descendants("face", face_id)
        elements = sorted(
            {
                element
                for child in descendants
                for element in child_faces.get(child, ())
            }
        )
        if elements:
            mesh.elements_of_face[face_id] = elements
        if descendants == (face_id,) and face_id in child_grids:
            mesh.grid_of_face[face_id] = child_grids[face_id]

    child_nodes = dict(mesh.nodes_of_edge)
    child_offset_nodes = dict(mesh.offset_nodes_of_edge)
    child_beams = dict(mesh.elements_of_edge)
    mesh.nodes_of_edge = {}
    mesh.offset_nodes_of_edge = {}
    mesh.elements_of_edge = {}
    active_original_edges = set(prepared.original_edges)
    for edge_id in active_original_edges:
        descendants = prepared.descendants("edge", edge_id)
        nodes = [node for child in descendants for node in child_nodes.get(child, ())]
        if nodes:
            mesh.nodes_of_edge[edge_id] = _ordered_nodes_on_original_edge(
                original, edge_id, mesh, nodes
            )

    for edge_id in beam_edges:
        descendants = prepared.descendants("edge", edge_id)
        elements = [
            element for child in descendants for element in child_beams.get(child, ())
        ]
        if elements:
            mesh.elements_of_edge[edge_id] = sorted(
                elements,
                key=lambda element: original.closest_edge_point(
                    edge_id,
                    np.mean([mesh.nodes[node] for node in mesh.beams[element]], axis=0),
                )[1],
            )
        pairs = [
            (plate, offset)
            for child in descendants
            for plate, offset in zip(
                child_nodes.get(child, ()), child_offset_nodes.get(child, ())
            )
        ]
        if pairs:
            pairs.sort(
                key=lambda pair: original.closest_edge_point(
                    edge_id, mesh.nodes[pair[0]]
                )[1]
            )
            mesh.offset_nodes_of_edge[edge_id] = [pair[1] for pair in pairs]

    child_vertices = dict(mesh.node_of_vertex)
    mesh.node_of_vertex = {}
    for vertex_id in original.vertices:
        descendants = prepared.descendants("vertex", vertex_id)
        for child in descendants:
            if child in child_vertices:
                mesh.node_of_vertex[vertex_id] = child_vertices[child]
                break
    return mesh


def generate_mesh_with_intersections(
    geometry: GeometryModel,
    *,
    target_size: float,
    overrides: Mapping[int, int] | None = None,
    beam_edges: Iterable[int] = (),
    beam_offsets: Mapping[int, float] | None = None,
    face_ids: Iterable[int] | None = None,
    seeding: Seeding | None = None,
    refinements: Iterable[Refinement] = (),
    order: str = "linear",
) -> Mesh:
    """Generate a mapped mesh with automatic shell and beam connectivity."""

    original_beams = tuple(dict.fromkeys(int(item) for item in beam_edges))
    prepared = _prepare(geometry, face_ids, original_beams)
    if prepared.intersections and seeding is not None:
        raise MeshError(
            "a precomputed seeding cannot be reused after automatic plate "
            "intersection imprint; provide target size and pins instead"
        )
    resolved_faces = tuple(
        child
        for face_id in prepared.original_faces
        for child in prepared.descendants("face", face_id)
    )
    resolved_beams = tuple(
        child
        for edge_id in original_beams
        for child in prepared.descendants("edge", edge_id)
    )
    child_offsets = {
        child: float((beam_offsets or {}).get(edge_id, 0.0))
        for edge_id in original_beams
        for child in prepared.descendants("edge", edge_id)
    }
    mesh = generate_mesh(
        prepared.geometry,
        target_size=target_size,
        overrides=_remap_overrides(prepared, overrides),
        beam_edges=resolved_beams,
        beam_offsets=child_offsets,
        face_ids=resolved_faces,
        seeding=seeding,
        refinements=_remap_refinements(prepared, refinements),
        order=order,
    )
    mesh.automatic_intersections = prepared.intersections
    mesh = _fold_associations(mesh, geometry, prepared, original_beams)
    mesh.automatic_beam_connections = connect_beam_mesh(mesh)
    mesh.automatic_shell_connections = connect_shell_boundaries(mesh, geometry)
    return mesh
