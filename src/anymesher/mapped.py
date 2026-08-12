"""Mapped (transfinite) meshing of Coons faces.

Node generation order is what guarantees conformity, so it is fixed:

1. one node per used vertex,
2. ``n - 1`` interior nodes per edge, stored in the edge's own direction,
3. face interior nodes from the Coons blend.

Faces look their boundary nodes up from the vertex and edge registries and
reverse the list when they traverse an edge backwards.  Neighbouring faces
therefore share the very same node objects, and coincident-node merging with a
tolerance is never needed.

The face interior is the transfinite (Coons) blend of the four boundary sides::

    P(u,v) = (1-v)A(u) + v C(u) + (1-u)D(v) + u B(v)
             - [(1-u)(1-v)P00 + u(1-v)P10 + u v P11 + (1-u)v P01]

Evaluated on the boundary this reproduces the boundary node positions exactly,
so the blend never moves a node off its curve.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from anygeometry.entities import OrientedEdge
from anygeometry.model import GeometryModel

from .errors import MeshError
from .mesh import Coupling, Mesh
from .refinement import Refinement, SizeField
from .seeding import Seeding, edge_distribution, solve_seeding

__all__ = [
    "ELEMENT_ORDERS",
    "coons_grid",
    "generate_mesh",
    "nodal_normals",
]

# Linear is Q4 shells and 2-node beams; quadratic is 8-node serendipity shells
# and 3-node beams.  The solver also has an experimental reduced-integration
# Q8R, which is not offered here: it is qualified for full integration only,
# and an element that is experimental in the solver would be no less
# experimental for being reachable from a menu.
ELEMENT_ORDERS: Tuple[str, ...] = ("linear", "quadratic")


def coons_grid(
    side_a: np.ndarray,
    side_b: np.ndarray,
    side_c: np.ndarray,
    side_d: np.ndarray,
) -> np.ndarray:
    """Blend four boundary polylines into a grid of positions.

    ``side_a`` and ``side_c`` run along ``u`` at ``v = 0`` and ``v = 1``;
    ``side_d`` and ``side_b`` run along ``v`` at ``u = 0`` and ``u = 1``.  The
    result has shape ``(len(side_a), len(side_b), 3)`` and reproduces the four
    boundaries exactly.

    This is shared by the mesher and by display tessellation so a face is never
    drawn as a different surface from the one that gets meshed.
    """

    side_a = np.asarray(side_a, dtype=float)
    side_b = np.asarray(side_b, dtype=float)
    side_c = np.asarray(side_c, dtype=float)
    side_d = np.asarray(side_d, dtype=float)
    if side_a.shape != side_c.shape or side_b.shape != side_d.shape:
        raise MeshError("opposite sides must have the same number of points")

    n_u = side_a.shape[0] - 1
    n_v = side_b.shape[0] - 1
    u = np.linspace(0.0, 1.0, n_u + 1).reshape(-1, 1, 1)
    v = np.linspace(0.0, 1.0, n_v + 1).reshape(1, -1, 1)

    corner_00 = side_a[0]
    corner_10 = side_a[-1]
    corner_11 = side_c[-1]
    corner_01 = side_c[0]

    return (
        (1.0 - v) * side_a[:, None, :]
        + v * side_c[:, None, :]
        + (1.0 - u) * side_d[None, :, :]
        + u * side_b[None, :, :]
        - (
            (1.0 - u) * (1.0 - v) * corner_00
            + u * (1.0 - v) * corner_10
            + u * v * corner_11
            + (1.0 - u) * v * corner_01
        )
    )


def generate_mesh(
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
    """Mesh the faces of a geometry model, plus any edges carrying beams.

    ``beam_offsets`` gives a stiffener an eccentricity: its nodes stand off the
    plating along the plate normal and are coupled back to it, rather than
    sharing the plate nodes.  That distinction matters -- a stiffener whose
    neutral axis sits in the plate midsurface is a materially different
    structure from one that stands proud of it.

    ``refinements`` are local size zones: the seeding integrates the resulting
    size field along each edge, and node placement follows the same field, so
    the counts and the positions cannot disagree.

    ``order`` picks Q4 and 2-node beams or Q8 and 3-node beams.  A quadratic
    mesh has the same *elements* as a linear one at the same division counts,
    with mid-side nodes added -- so element counts stay comparable and only the
    interpolation improves.
    """

    if order not in ELEMENT_ORDERS:
        raise MeshError(
            f"unknown element order {order!r}; expected one of "
            f"{', '.join(ELEMENT_ORDERS)}"
        )

    faces = (
        list(geometry.faces)
        if face_ids is None
        else list(dict.fromkeys(int(f) for f in face_ids))
    )
    for face_id in faces:
        if face_id not in geometry.faces:
            raise MeshError(f"no face {face_id}")
        face = geometry.faces[face_id]
        if len(face.corners) != 4:
            raise MeshError(
                f"face {face_id} has no four-side mapped parameterization; "
                "partition it first (triangle_to_quads handles triangles)"
            )
        if getattr(face, "holes", ()):
            raise MeshError(
                f"face {face_id} has trimmed holes; decompose it into mapped "
                "patches before using the built-in quad backend"
            )

    beam_edge_ids = list(dict.fromkeys(int(e) for e in beam_edges))
    for edge_id in beam_edge_ids:
        if edge_id not in geometry.edges:
            raise MeshError(f"no edge {edge_id}")
    if order == "quadratic":
        _refuse_curved_beams(geometry, beam_edge_ids)

    active_edges: List[int] = []
    for face_id in faces:
        for item in geometry.faces[face_id].loop:
            active_edges.append(item.edge)
    active_edges.extend(beam_edge_ids)
    active_edges = list(dict.fromkeys(active_edges))
    if not active_edges:
        raise MeshError("nothing to mesh: no faces and no beam edges")

    if seeding is None:
        seeding = solve_seeding(
            geometry,
            size_field=SizeField(geometry, target_size, refinements),
            overrides=overrides,
            edge_ids=active_edges,
        )
    size_field = seeding.size_field or SizeField(geometry, target_size)

    mesh = Mesh(
        geometry_model_id=geometry.model_id,
        geometry_revision=geometry.revision,
        seeding=seeding,
        order=order,
    )
    next_node = _Counter()
    next_element = _Counter()

    _build_vertex_nodes(geometry, mesh, active_edges, next_node)
    _build_edge_nodes(
        geometry, mesh, active_edges, seeding, size_field, next_node
    )
    for face_id in faces:
        _build_face(geometry, mesh, face_id, next_node, next_element)

    offsets = dict(beam_offsets or {})
    for edge_id in beam_edge_ids:
        offset = float(offsets.get(edge_id, 0.0))
        if offset:
            _build_offset_nodes(geometry, mesh, edge_id, offset, next_node)
        _build_beam(mesh, edge_id, next_element)

    return mesh


def _refuse_curved_beams(
    geometry: GeometryModel, beam_edge_ids: Sequence[int]
) -> None:
    """A quadratic beam on a curve is refused, because the solver refuses it.

    ANYsolver's 3-node beam is straight-sided: it requires the middle node at
    the chord midpoint and rejects anything else, since a displaced mid node
    is not a curved-beam definition.  Mid-side nodes here are placed *on the
    curve*, which is right for a Q8 shell edge and wrong for a B3.  Catching it
    at mesh time names the line and costs nothing; leaving it to the solver
    would spend the assembly first and name only an element number.
    """

    from anygeometry.curves import Straight

    curved = [
        edge_id
        for edge_id in beam_edge_ids
        if not isinstance(geometry.edges[edge_id].curve, Straight)
    ]
    if curved:
        raise MeshError(
            f"line(s) {curved} carry a beam and are curved, but a quadratic "
            "mesh would put 3-node beams on them and the solver's B3 is "
            "straight-sided. Split the curve into straight segments, or mesh "
            "this model with order='linear'."
        )


def nodal_normals(mesh: Mesh) -> Dict[int, np.ndarray]:
    """A unit normal at every node, averaged over the plates meeting it."""

    accumulated: Dict[int, np.ndarray] = {}
    for element_id, nodes in mesh.quads.items():
        # The diagonals come from the corners; a mid-side node would not
        # describe the element's plane. But every node of the element,
        # mid-sides included, gets the normal -- an eccentric stiffener on a
        # quadratic mesh needs a normal at its mid-side nodes too.
        corners = np.array(
            [mesh.nodes[node] for node in mesh.corners_of(element_id)]
        )
        normal = np.cross(corners[2] - corners[0], corners[3] - corners[1])
        length = float(np.linalg.norm(normal))
        if length <= 0.0:
            continue
        normal = normal / length
        for node in nodes:
            accumulated[node] = accumulated.get(node, np.zeros(3)) + normal

    normals: Dict[int, np.ndarray] = {}
    for node, total in accumulated.items():
        length = float(np.linalg.norm(total))
        if length > 0.0:
            normals[node] = total / length
    return normals


def _build_offset_nodes(
    geometry: GeometryModel,
    mesh: Mesh,
    edge_id: int,
    offset: float,
    next_node: "_Counter",
) -> None:
    """Stand a stiffener off the plating, along the plate normal."""

    if not geometry.faces_using_edge(edge_id):
        raise MeshError(
            f"line {edge_id} has a beam eccentricity but bounds no plate, so "
            "there is nothing for it to be eccentric to. Remove the "
            "eccentricity, or attach the line to a plate."
        )

    normals = nodal_normals(mesh)
    sequence = mesh.nodes_of_edge[edge_id]
    missing = [node for node in sequence if node not in normals]
    if missing:
        raise MeshError(
            f"line {edge_id} has a beam eccentricity but no plate normal could "
            f"be found at node(s) {missing[:4]}"
        )

    offset_nodes: List[int] = []
    for node in sequence:
        node_id = next_node.next()
        mesh.nodes[node_id] = mesh.nodes[node] + offset * normals[node]
        offset_nodes.append(node_id)
    mesh.offset_nodes_of_edge[edge_id] = offset_nodes


class _Counter:
    """Monotonic 1-based ID source."""

    def __init__(self) -> None:
        self._value = 0

    def next(self) -> int:
        self._value += 1
        return self._value


def _build_vertex_nodes(
    geometry: GeometryModel,
    mesh: Mesh,
    active_edges: Sequence[int],
    next_node: _Counter,
) -> None:
    used: List[int] = []
    for edge_id in active_edges:
        edge = geometry.edges[edge_id]
        used.append(edge.start)
        used.append(edge.end)
    for vertex_id in dict.fromkeys(used):
        node_id = next_node.next()
        mesh.node_of_vertex[vertex_id] = node_id
        mesh.nodes[node_id] = geometry.vertex_position(vertex_id).copy()


def _build_edge_nodes(
    geometry: GeometryModel,
    mesh: Mesh,
    active_edges: Sequence[int],
    seeding: Seeding,
    size_field: SizeField,
    next_node: _Counter,
) -> None:
    """Interior nodes along every active edge, in the edge's own direction.

    A quadratic mesh needs a node at each element mid-side, so an edge with
    ``n`` divisions gets ``2n - 1`` interior nodes rather than ``n - 1``.  The
    mid-side node is placed *on the curve*, not at the midpoint of the chord,
    so a Q8 on an arc follows the arc.
    """

    steps_per_division = 2 if mesh.is_quadratic else 1
    for edge_id in active_edges:
        edge = geometry.edges[edge_id]
        stations = seeding[edge_id] * steps_per_division
        sequence = [mesh.node_of_vertex[edge.start]]
        if stations > 1:
            parameters = edge_distribution(
                geometry, edge_id, stations, size_field
            )
            points = geometry.sample_edge(edge_id, parameters)
            for point in points:
                node_id = next_node.next()
                mesh.nodes[node_id] = np.asarray(point, dtype=float)
                sequence.append(node_id)
        sequence.append(mesh.node_of_vertex[edge.end])
        mesh.nodes_of_edge[edge_id] = sequence


def _chain_nodes(
    mesh: Mesh, side: Sequence[OrientedEdge], face_id: int
) -> List[int]:
    """Concatenate a side's edge node lists in traversal order."""

    result: List[int] = []
    for item in side:
        sequence = mesh.nodes_of_edge[item.edge]
        if not item.forward:
            sequence = sequence[::-1]
        if not result:
            result.extend(sequence)
            continue
        if result[-1] != sequence[0]:
            raise MeshError(
                f"face {face_id} boundary is not continuous at edge "
                f"{item.edge}; the loop ordering is inconsistent"
            )
        result.extend(sequence[1:])
    return result


def _build_face(
    geometry: GeometryModel,
    mesh: Mesh,
    face_id: int,
    next_node: _Counter,
    next_element: _Counter,
) -> None:
    face = geometry.faces[face_id]
    sides = face.sides()

    # Loop order runs corner0 -A-> corner1 -B-> corner2 -C-> corner3 -D-> corner0.
    # Sides C and D are traversed against increasing u and v, so reverse them
    # to get node lists running with the parameter directions.
    side_a = _chain_nodes(mesh, sides[0], face_id)
    side_b = _chain_nodes(mesh, sides[1], face_id)
    side_c = _chain_nodes(mesh, sides[2], face_id)[::-1]
    side_d = _chain_nodes(mesh, sides[3], face_id)[::-1]

    # Grid *stations*, which is one per division when linear and two when
    # quadratic -- the extra one being the element mid-side.
    step = 2 if mesh.is_quadratic else 1
    u_stations = len(side_a) - 1
    v_stations = len(side_b) - 1
    if len(side_c) - 1 != u_stations or len(side_d) - 1 != v_stations:
        raise MeshError(
            f"face {face_id} opposite sides have different division counts "
            f"({u_stations} vs {len(side_c) - 1}, "
            f"{v_stations} vs {len(side_d) - 1}); "
            "the seeding solver should have prevented this"
        )
    n_u, n_v = u_stations // step, v_stations // step
    if n_u < 1 or n_v < 1:
        raise MeshError(f"face {face_id} needs at least one division per side")

    _check_corners(mesh, face_id, side_a, side_b, side_c, side_d)

    # ``0`` marks a grid position with no node: the centre of a quadratic
    # element, which serendipity interpolation does not use.  Node IDs are
    # 1-based, so it cannot collide with a real one.
    grid = np.zeros((u_stations + 1, v_stations + 1), dtype=int)
    grid[:, 0] = side_a
    grid[:, v_stations] = side_c
    grid[0, :] = side_d
    grid[u_stations, :] = side_b

    blended = coons_grid(
        np.array([mesh.nodes[node] for node in side_a]),
        np.array([mesh.nodes[node] for node in side_b]),
        np.array([mesh.nodes[node] for node in side_c]),
        np.array([mesh.nodes[node] for node in side_d]),
    )
    for i in range(1, u_stations):
        for j in range(1, v_stations):
            if step == 2 and i % 2 == 1 and j % 2 == 1:
                continue  # element centre; no serendipity node lives here
            node_id = next_node.next()
            mesh.nodes[node_id] = blended[i, j]
            grid[i, j] = node_id

    mesh.grid_of_face[face_id] = grid

    element_ids: List[int] = []
    for i in range(n_u):
        for j in range(n_v):
            element_id = next_element.next()
            base_u, base_v = i * step, j * step
            corners = (
                int(grid[base_u, base_v]),
                int(grid[base_u + step, base_v]),
                int(grid[base_u + step, base_v + step]),
                int(grid[base_u, base_v + step]),
            )
            if step == 1:
                mesh.quads[element_id] = corners
            else:
                # Serendipity ordering: corners, then the mid-side node of
                # each edge in the same rotation, starting with 0-1.
                mesh.quads[element_id] = corners + (
                    int(grid[base_u + 1, base_v]),
                    int(grid[base_u + 2, base_v + 1]),
                    int(grid[base_u + 1, base_v + 2]),
                    int(grid[base_u, base_v + 1]),
                )
            element_ids.append(element_id)
    mesh.elements_of_face[face_id] = element_ids


def _check_corners(
    mesh: Mesh,
    face_id: int,
    side_a: Sequence[int],
    side_b: Sequence[int],
    side_c: Sequence[int],
    side_d: Sequence[int],
) -> None:
    """The four sides must meet at shared corner nodes."""

    checks = (
        ("u=0,v=0", side_a[0], side_d[0]),
        ("u=1,v=0", side_a[-1], side_b[0]),
        ("u=1,v=1", side_b[-1], side_c[-1]),
        ("u=0,v=1", side_c[0], side_d[-1]),
    )
    for label, left, right in checks:
        if left != right:
            raise MeshError(
                f"face {face_id} corner {label} is not shared between its "
                f"sides (nodes {left} and {right}); check the corner assignment"
            )


def _build_beam(mesh: Mesh, edge_id: int, next_element: _Counter) -> None:
    """Beam elements along a line, on the offset nodes when there are any."""

    plating = mesh.nodes_of_edge[edge_id]
    sequence = mesh.offset_nodes_of_edge.get(edge_id) or plating

    # A quadratic beam spans two stations and carries the one between them.
    step = 2 if mesh.is_quadratic else 1
    element_ids: List[int] = []
    for start in range(0, len(sequence) - step, step):
        element_id = next_element.next()
        span = sequence[start : start + step + 1]
        mesh.beams[element_id] = tuple(int(node) for node in span)
        element_ids.append(element_id)
    mesh.elements_of_edge[edge_id] = element_ids

    if sequence is not plating:
        # One coupling per station, tying the stiffener back to the plating.
        # A beam node here sits directly above its plate node, so the record has
        # a single master.  The eccentricity is read back from the node positions
        # rather than recomputed from the offset and the normal, so the record
        # describes where the nodes actually are.
        for beam_node, plate_node in zip(sequence, plating):
            element_id = next_element.next()
            mesh.couplings[element_id] = Coupling.node_to_node(
                beam_node,
                plate_node,
                mesh.nodes[beam_node] - mesh.nodes[plate_node],
            )
