"""Mapped-face decomposition kept deliberately on the meshing side.

The neutral topology, curve and surface operations live in :mod:`anygeometry`.
The operations in this module have a narrower purpose: turn regions into the
four-sided patches required by the built-in transfinite quadrilateral mesher.
They therefore remain ANYmesher responsibilities even though their inputs and
outputs are ANYgeometry entities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from anygeometry.chains import chain_breaks
from anygeometry.entities import EntityRef, Face, OrientedEdge
from anygeometry.errors import GeometryError
from anygeometry.model import GeometryModel
from anygeometry.operations import surface_point

__all__ = [
    "MappabilityReport",
    "check_mappable",
    "punch_circular_hole",
    "split_face_at",
    "split_face_between",
    "strip_face",
    "triangle_to_quads",
]

SURFACE_TOLERANCE = 1.0e-6
_FIDELITY_SAMPLES = 24


@dataclass
class MappabilityReport:
    """Whether a face can be mapped, and if not, what to do about it."""

    face_id: int
    ok: bool
    edge_count: int
    side_edge_counts: Tuple[int, int, int, int]
    messages: Tuple[str, ...] = ()

    def __str__(self) -> str:  # pragma: no cover - display only
        state = "mappable" if self.ok else "not mappable"
        return f"face {self.face_id}: {state}. " + " ".join(self.messages)


def check_mappable(geometry: GeometryModel, face_id: int) -> MappabilityReport:
    """Report whether a face is suitable for the mapped quad backend."""

    face = geometry.faces.get(face_id)
    if face is None:
        raise GeometryError(f"no face {face_id}")

    messages: List[str] = []

    if len(face.corners) != 4:
        counts = (0, 0, 0, 0)
        messages.append(
            f"the boundary has only {len(face.loop)} edges; a mapped face "
            "needs four sides. Split the boundary, or use triangle_to_quads "
            "for a three-sided region."
        )
        return MappabilityReport(
            face_id=face_id,
            ok=False,
            edge_count=len(face.loop),
            side_edge_counts=counts,
            messages=tuple(messages),
        )

    sides = face.sides()
    counts = tuple(len(side) for side in sides)
    if any(count == 0 for count in counts):
        messages.append(
            "one side is empty; the corner assignment puts two corners in the "
            "same place. Set the corners explicitly."
        )

    lengths = geometry.face_side_lengths(face_id)
    for first, second, label in ((0, 2, "0 and 2"), (1, 3, "1 and 3")):
        longer = max(lengths[first], lengths[second])
        shorter = min(lengths[first], lengths[second])
        if longer > 0.0 and shorter / longer < 0.05:
            messages.append(
                f"opposite sides {label} differ in length by more than 20 to 1, "
                "so the mapped mesh will be badly distorted. Consider splitting "
                "this face."
            )

    return MappabilityReport(
        face_id=face_id,
        ok=not messages,
        edge_count=len(face.loop),
        side_edge_counts=counts,  # type: ignore[arg-type]
        messages=tuple(messages),
    )


def split_face_between(
    geometry: GeometryModel,
    face_id: int,
    start_vertex: int,
    end_vertex: int,
    *,
    tolerance: float = SURFACE_TOLERANCE,
) -> Tuple[int, Tuple[int, int]]:
    """Partition a mapped face between vertices on opposite mapped sides.

    This compatibility operation deliberately preserves the original mapped
    semantics. General face fragmentation is :func:`anygeometry.split_face`.
    """

    face = geometry.faces.get(face_id)
    if face is None:
        raise GeometryError(f"no face {face_id}")

    start_side, start_fraction = _locate_on_boundary(geometry, face, start_vertex)
    end_side, end_fraction = _locate_on_boundary(geometry, face, end_vertex)
    if {start_side, end_side} not in ({0, 2}, {1, 3}):
        raise GeometryError(
            f"points {start_vertex} and {end_vertex} lie on sides "
            f"{start_side} and {end_side} of face {face_id}. A dividing edge "
            "must run between opposite sides; split the boundary first if you "
            "need a cut between adjacent sides."
        )

    samples = _surface_curve(
        geometry, face, start_side, start_fraction, end_side, end_fraction
    )
    dividing_edge = _fit_dividing_edge(
        geometry, start_vertex, end_vertex, samples, tolerance
    )
    first_loop, second_loop = _split_mapped_loop(
        face, start_vertex, end_vertex, geometry
    )
    metadata = dict(face.metadata)
    # The explicit replacement below is the public edit.  Suppress the
    # intermediate deletion record so consumers see one unambiguous lineage
    # entry (old face -> the two new faces), rather than old -> () followed by
    # old -> descendants.
    geometry.remove_face(face_id, record=False)

    first_face = geometry.add_face(
        [item.edge for item in first_loop] + [dividing_edge]
    )
    second_face = geometry.add_face(
        [item.edge for item in second_loop] + [dividing_edge]
    )
    for made in (first_face, second_face):
        geometry.set_face_metadata(made, metadata)
    geometry.record_replacement(
        EntityRef("face", face_id),
        (EntityRef("face", first_face), EntityRef("face", second_face)),
    )
    return dividing_edge, (first_face, second_face)


def split_face_at(
    geometry: GeometryModel,
    face_id: int,
    axis: int,
    fraction: float,
    *,
    tolerance: float = SURFACE_TOLERANCE,
) -> Tuple[int, Tuple[int, int]]:
    """Partition a mapped face across one parameter direction."""

    if axis not in (0, 1):
        raise GeometryError("axis must be 0 or 1")
    if not 0.0 < float(fraction) < 1.0:
        raise GeometryError(
            f"fraction must be strictly between 0 and 1, got {fraction}"
        )
    face = geometry.faces.get(face_id)
    if face is None:
        raise GeometryError(f"no face {face_id}")
    if len(face.corners) != 4:
        raise GeometryError(
            f"face {face_id} has no four-side mapped parameterization"
        )

    start_vertex = _split_side_at(geometry, face_id, axis, float(fraction))
    end_vertex = _split_side_at(
        geometry, face_id, axis + 2, 1.0 - float(fraction)
    )
    return split_face_between(
        geometry, face_id, start_vertex, end_vertex, tolerance=tolerance
    )


def _split_side_at(
    geometry: GeometryModel, face_id: int, side_index: int, fraction: float
) -> int:
    side = geometry.faces[face_id].sides()[side_index]
    breaks = chain_breaks(geometry, side)
    segment = int(np.searchsorted(breaks, fraction, side="right") - 1)
    segment = min(max(segment, 0), len(side) - 1)
    span = breaks[segment + 1] - breaks[segment]
    local = 0.0 if span <= 0.0 else (fraction - breaks[segment]) / span
    item = side[segment]
    parameter_tolerance = geometry.tolerance.effective_parameter(
        geometry.edge_length(item.edge)
    )
    if local <= parameter_tolerance:
        return geometry.oriented_start_vertex(item)
    if local >= 1.0 - parameter_tolerance:
        return geometry.oriented_end_vertex(item)
    parameter = local if item.forward else 1.0 - local
    new_vertex, _halves = geometry.split_edge(item.edge, parameter)
    return new_vertex


def _locate_on_boundary(
    geometry: GeometryModel, face: Face, vertex_id: int
) -> Tuple[int, float]:
    for side_index, side in enumerate(face.sides()):
        breaks = chain_breaks(geometry, side)
        for position, item in enumerate(side):
            if geometry.oriented_start_vertex(item) == vertex_id:
                return side_index, float(breaks[position])
        if side and geometry.oriented_end_vertex(side[-1]) == vertex_id:
            return side_index, 1.0
    raise GeometryError(
        f"point {vertex_id} is not a junction on the boundary of face "
        f"{face.id}; split the boundary edge there first"
    )


def _surface_curve(
    geometry: GeometryModel,
    face: Face,
    start_side: int,
    start_fraction: float,
    end_side: int,
    end_fraction: float,
) -> np.ndarray:
    start_uv = _boundary_uv(start_side, start_fraction)
    end_uv = _boundary_uv(end_side, end_fraction)
    steps = np.linspace(0.0, 1.0, _FIDELITY_SAMPLES)
    return np.array(
        [
            surface_point(
                geometry,
                face,
                float(start_uv[0] + (end_uv[0] - start_uv[0]) * step),
                float(start_uv[1] + (end_uv[1] - start_uv[1]) * step),
            )
            for step in steps
        ]
    )


def _boundary_uv(side_index: int, fraction: float) -> Tuple[float, float]:
    if side_index == 0:
        return fraction, 0.0
    if side_index == 1:
        return 1.0, fraction
    if side_index == 2:
        return 1.0 - fraction, 1.0
    return 0.0, 1.0 - fraction


def _fit_dividing_edge(
    geometry: GeometryModel,
    start_vertex: int,
    end_vertex: int,
    samples: np.ndarray,
    tolerance: float,
) -> int:
    start = samples[0]
    end = samples[-1]
    scale = float(np.linalg.norm(end - start))
    if scale <= geometry.tolerance.effective_length(scale):
        raise GeometryError("a dividing edge needs two distinct points")
    limit = max(
        tolerance * scale,
        geometry.tolerance.effective_surface_residual(scale),
    )
    straight = start + np.outer(
        np.linspace(0.0, 1.0, len(samples)), end - start
    )
    if float(np.abs(straight - samples).max()) <= limit:
        return geometry.add_line(start_vertex, end_vertex)

    midpoint = samples[len(samples) // 2]
    via_vertex = geometry.add_point(*midpoint)
    try:
        candidate = geometry.add_arc(start_vertex, via_vertex, end_vertex)
    except GeometryError as error:
        # ``via_vertex`` was only a fitting probe and was never exposed as a
        # result of this operation, so it must not enter persistent history.
        geometry.remove_vertex(via_vertex, record=False)
        raise GeometryError(
            f"the cut across this face is neither straight nor circular ({error})."
            " Split the boundary into smaller pieces and cut those."
        ) from None

    fitted = geometry.sample_edge(
        candidate, np.linspace(0.0, 1.0, len(samples))
    )
    if float(np.abs(fitted - samples).max()) <= limit:
        return candidate
    # A rejected fit is temporary implementation detail, not a user-visible
    # deletion.  Keeping it out of replacement history also leaves a failed
    # decomposition transaction observationally clean.
    geometry.remove_edge(candidate, record=False)
    geometry.remove_vertex(via_vertex, record=False)
    raise GeometryError(
        "the cut across this face follows neither a straight line nor a "
        "circular arc, so it cannot be represented exactly. Split the "
        "boundary into smaller pieces and cut those."
    )


def _split_mapped_loop(
    face: Face, start_vertex: int, end_vertex: int, geometry: GeometryModel
) -> Tuple[List[OrientedEdge], List[OrientedEdge]]:
    positions: Dict[int, int] = {}
    for index, item in enumerate(face.loop):
        positions[geometry.oriented_start_vertex(item)] = index
    if start_vertex not in positions or end_vertex not in positions:
        raise GeometryError(
            "both ends of a cut must be junctions on the face boundary"
        )
    start = positions[start_vertex]
    end = positions[end_vertex]
    count = len(face.loop)
    first = [face.loop[(start + step) % count] for step in range((end - start) % count)]
    second = [face.loop[(end + step) % count] for step in range((start - end) % count)]
    if not first or not second:
        raise GeometryError(
            "a cut must separate the boundary into two non-empty chains"
        )
    return first, second


def strip_face(
    geometry: GeometryModel,
    face_id: int,
    axis: int,
    count: int,
    *,
    tolerance: float = SURFACE_TOLERANCE,
) -> Tuple[List[int], List[int]]:
    """Partition a mapped face into equal strips and shared divider edges."""

    if axis not in (0, 1):
        raise GeometryError("axis must be 0 or 1")
    count = int(count)
    if count < 2:
        raise GeometryError(f"a strip count needs to be at least 2, got {count}")
    if face_id not in geometry.faces:
        raise GeometryError(f"no face {face_id}")

    fractions = [index / count for index in range(1, count)]
    cuts: List[Tuple[int, int]] = []
    for fraction in fractions:
        start = _split_side_at(geometry, face_id, axis, fraction)
        end = _split_side_at(geometry, face_id, axis + 2, 1.0 - fraction)
        cuts.append((start, end))

    strips: List[int] = []
    dividers: List[int] = []
    remainder = face_id
    for index, (start, end) in enumerate(cuts):
        divider, (first, second) = split_face_between(
            geometry, remainder, start, end, tolerance=tolerance
        )
        dividers.append(divider)
        remaining_cuts = cuts[index + 1 :]
        if remaining_cuts:
            following = _face_holding(
                geometry, (first, second), *remaining_cuts[0]
            )
        else:
            following = second
        finished = first if following == second else second
        strips.append(finished)
        remainder = following
    strips.append(remainder)
    return strips, dividers


def _face_holding(
    geometry: GeometryModel,
    candidates: Sequence[int],
    start_vertex: int,
    end_vertex: int,
) -> int:
    for face_id in candidates:
        vertices = set()
        for item in geometry.faces[face_id].loop:
            edge = geometry.edges[item.edge]
            vertices.add(edge.start)
            vertices.add(edge.end)
        if {start_vertex, end_vertex} <= vertices:
            return face_id
    raise GeometryError(
        "after splitting, neither half carries the next cut; the strip "
        "fractions do not lie on one face"
    )


def triangle_to_quads(
    geometry: GeometryModel, edge_ids: Sequence[int]
) -> List[int]:
    """Turn a three-sided region into three non-degenerate mapped quads."""

    loop = geometry.order_loop(edge_ids)
    if len(loop) != 3:
        raise GeometryError(
            f"triangle_to_quads needs exactly three edges, got {len(loop)}"
        )
    for item in loop:
        if geometry.faces_using_edge(item.edge):
            raise GeometryError(
                f"edge {item.edge} already bounds a face; delete it first"
            )

    corners = [geometry.oriented_start_vertex(item) for item in loop]
    centre = np.mean(
        [geometry.vertex_position(vertex) for vertex in corners], axis=0
    )
    centre_vertex = geometry.add_point(*centre)

    halves: List[Tuple[OrientedEdge, OrientedEdge]] = []
    midpoints: List[int] = []
    for item in loop:
        midpoint, (first, second) = geometry.split_edge(item.edge, 0.5)
        midpoints.append(midpoint)
        if item.forward:
            halves.append((OrientedEdge(first, True), OrientedEdge(second, True)))
        else:
            halves.append((OrientedEdge(second, False), OrientedEdge(first, False)))

    spokes = [geometry.add_line(midpoint, centre_vertex) for midpoint in midpoints]

    faces: List[int] = []
    for index in range(3):
        previous = (index - 1) % 3
        face_loop = (
            halves[previous][1],
            halves[index][0],
            OrientedEdge(spokes[index], True),
            OrientedEdge(spokes[previous], False),
        )
        faces.append(geometry.add_face_from_loop(face_loop, (0, 1, 2, 3)))
    return faces


def punch_circular_hole(
    geometry: GeometryModel,
    face_id: int,
    centre: Sequence[float],
    radius: float,
    *,
    planarity_tolerance: float = 1.0e-6,
) -> Tuple[List[int], List[int]]:
    """Replace a planar face with four mapped butterfly patches around a hole.

    This is the legacy ANYmesher operation.  It is intentionally distinct from
    a neutral trimmed-face/hole operation: the four-patch O-grid exists solely
    to make the region suitable for mapped quadrilateral meshing.
    """

    face = geometry.faces.get(face_id)
    if face is None:
        raise GeometryError(f"no face {face_id}")

    radius = float(radius)
    if radius <= 0.0:
        raise GeometryError("the hole radius must be positive")

    sides = face.sides()
    corner_vertices = list(geometry.face_corner_vertices(face_id))
    corner_points = np.array(
        [geometry.vertex_position(vertex) for vertex in corner_vertices]
    )

    normal, origin = _plane_of(geometry, face, planarity_tolerance)
    hub = np.asarray(centre, dtype=float)
    if hub.shape != (3,):
        raise GeometryError("the hole centre must be a 3 component point")
    offset = hub - origin
    if abs(float(offset @ normal)) > planarity_tolerance * max(
        1.0, float(np.linalg.norm(offset))
    ):
        raise GeometryError("the hole centre does not lie in the plate plane")

    basis_u, basis_v = _plane_basis(normal)
    corner_angles = [
        _angle_in_plane(point - hub, basis_u, basis_v) for point in corner_points
    ]
    reach = min(float(np.linalg.norm(point - hub)) for point in corner_points)
    if radius >= reach:
        raise GeometryError(
            f"a hole of radius {radius:g} does not fit: the nearest plate "
            f"corner is only {reach:g} away"
        )

    counter_clockwise = _winding(corner_points - hub, basis_u, basis_v) > 0.0

    ring_vertices: List[int] = []
    for angle in corner_angles:
        position = hub + radius * (
            np.cos(angle) * basis_u + np.sin(angle) * basis_v
        )
        ring_vertices.append(geometry.add_point(*position))

    arcs: List[int] = []
    for index in range(4):
        following = (index + 1) % 4
        start_angle = corner_angles[index]
        sweep = _wrapped_sweep(
            start_angle,
            corner_angles[following],
            counter_clockwise,
            geometry.tolerance.angular,
        )
        via_angle = start_angle + 0.5 * sweep
        via_position = hub + radius * (
            np.cos(via_angle) * basis_u + np.sin(via_angle) * basis_v
        )
        via_vertex = geometry.add_point(*via_position)
        arcs.append(
            geometry.add_arc(
                ring_vertices[index], via_vertex, ring_vertices[following]
            )
        )

    spokes = [
        geometry.add_line(corner_vertices[index], ring_vertices[index])
        for index in range(4)
    ]

    # Record only the semantic one-to-many replacement below.  Recording this
    # physical removal as a deletion first would make selection and attribute
    # propagation observe contradictory lineage for the same face.
    geometry.remove_face(face_id, record=False)

    faces: List[int] = []
    for index in range(4):
        following = (index + 1) % 4
        outer = tuple(sides[index])
        face_loop = outer + (
            OrientedEdge(spokes[following], True),
            OrientedEdge(arcs[index], False),
            OrientedEdge(spokes[index], False),
        )
        corners = (0, len(outer), len(outer) + 1, len(outer) + 2)
        faces.append(geometry.add_face_from_loop(face_loop, corners))
    geometry.record_replacement(
        EntityRef("face", face_id),
        tuple(EntityRef("face", made) for made in faces),
    )
    return faces, arcs


def _plane_of(
    geometry: GeometryModel, face: Face, tolerance: float
) -> Tuple[np.ndarray, np.ndarray]:
    samples = np.array(
        [
            surface_point(geometry, face, u, v)
            for u in (0.0, 0.5, 1.0)
            for v in (0.0, 0.5, 1.0)
        ]
    )
    origin = samples.mean(axis=0)
    _left, _values, vectors = np.linalg.svd(samples - origin)
    normal = vectors[2]
    scale = float(np.abs(samples - origin).max())
    deviation = float(np.abs((samples - origin) @ normal).max())
    residual_tolerance = max(
        tolerance * scale,
        geometry.tolerance.effective_surface_residual(scale),
    )
    if (
        scale <= geometry.tolerance.effective_length(scale)
        or deviation > residual_tolerance
    ):
        raise GeometryError(
            f"face {face.id} is not planar, so a circular hole in it is not a "
            "circle in any one plane. Punch the hole in a planar plate."
        )
    return normal, origin


def _plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    reference = (
        np.array([1.0, 0.0, 0.0])
        if abs(float(normal @ np.array([1.0, 0.0, 0.0]))) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    basis_u = reference - float(reference @ normal) * normal
    basis_u /= float(np.linalg.norm(basis_u))
    return basis_u, np.cross(normal, basis_u)


def _angle_in_plane(
    offset: np.ndarray, basis_u: np.ndarray, basis_v: np.ndarray
) -> float:
    return float(np.arctan2(offset @ basis_v, offset @ basis_u))


def _winding(
    offsets: np.ndarray, basis_u: np.ndarray, basis_v: np.ndarray
) -> float:
    x = offsets @ basis_u
    y = offsets @ basis_v
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _wrapped_sweep(
    start: float,
    end: float,
    counter_clockwise: bool,
    angular_tolerance: float,
) -> float:
    delta = (end - start) % (2.0 * np.pi)
    if not counter_clockwise:
        delta -= 2.0 * np.pi
    if abs(delta) <= angular_tolerance:
        raise GeometryError(
            "two plate corners lie on the same ray from the hole centre, so "
            "the butterfly patches would be degenerate. Move the hole centre."
        )
    return float(delta)
