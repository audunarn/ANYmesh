"""Beams that are meshed at their own size on a shared shell edge.

A beam line that is also a shell boundary edge normally shares the canonical
``nodes_of_edge`` chain with the plating.  When the size field carries entity
sizes and the beam's own target asks for a different number of elements than
that chain, the beam is rebuilt on its **own** stations:

* the shell chain, ``nodes_of_edge`` and every shell element stay exactly as
  generated (canonical geometry chain, never regraded for the beam);
* stations that coincide with a canonical chain node reuse that node (the two
  vertex ends always do), and an eccentric station reuses an existing offset
  node of a neighbouring beam when it stands off the same chain node by the
  same displacement, so joints keep their shared nodes *and* their couplings;
* every other station is a new node on the line, tied to the shell with an
  interpolation :class:`~anymesher.mesh.Coupling` over the actual chain nodes
  (linear for Q4/T3, quadratic Lagrange for Q8/T6), with the eccentricity read
  back from the node positions exactly as the ``Coupling`` record documents;
* declared attachment and junction point stations on the edge are kept as
  beam stations, so connectivity still finds a node at every declared station;
* an eccentric beam gets offset nodes only at its real stations. Old nodes and
  couplings that belonged to the replaced beam alone are deleted; anything a
  neighbouring beam still uses is kept.  ``offset_nodes_of_edge`` is not
  written for the edge because its documented zip with ``nodes_of_edge``
  cannot hold when the chains differ.

Ownership is carried by the mesh itself (beam elements, couplings, canonical
chains); ``mesh.hybrid_diagnostics["independent_beam_edges"]`` is only an
informational record of what was rebuilt.  All indexes are built once per
call (request scoped, nothing global).
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .errors import MeshError
from .mesh import Coupling, Mesh

__all__ = [
    "INDEPENDENT_BEAM_EDGES",
    "STATION_POLL",
    "apply_independent_beams",
    "independent_beam_edges",
]

INDEPENDENT_BEAM_EDGES = "independent_beam_edges"
# Stations created between two cancellation polls inside one edge.
STATION_POLL = 64
# Same per-edge ceiling as ``solve_seeding(max_divisions=...)``.
MAX_BEAM_DIVISIONS = 100_000


def _poll(callback: Callable[[str], None] | None, phase: str) -> None:
    if callback is not None:
        callback(phase)


def independent_beam_edges(
    geometry,
    mesh: Mesh,
    beam_edge_ids: Iterable[int],
    size_field,
    shell_edges: Iterable[int],
) -> dict[int, int]:
    """``{edge: beam divisions}`` for beams whose own size differs from the chain.

    Only beams on shell edges, and only when the field carries a size for that
    edge, can qualify; an ordinary model never does.
    """

    if not getattr(size_field, "has_entity_sizes", False):
        return {}
    shell = {int(edge) for edge in shell_edges}
    step = 2 if mesh.is_quadratic else 1
    found: dict[int, int] = {}
    for edge_id in sorted({int(edge) for edge in beam_edge_ids}):
        if edge_id not in shell:
            continue
        if edge_id not in size_field.edge_sizes and edge_id not in size_field.beam_sizes:
            continue
        chain = mesh.nodes_of_edge.get(edge_id)
        if not chain:
            continue
        stations = (len(chain) - 1) // step
        length = float(geometry.edge_length(edge_id))
        # Same rounding as the seeding demand, so a beam that already has the
        # stations it asks for (a free line) is never rebuilt.
        wanted = max(1, int(round(length / size_field.beam_target(edge_id))))
        if wanted > MAX_BEAM_DIVISIONS:
            raise MeshError(
                f"beam line {edge_id} asks for {wanted} divisions at its own size, "
                f"above the {MAX_BEAM_DIVISIONS} per-edge bound (the seeding bound)"
            )
        if wanted != stations:
            found[edge_id] = wanted
    return found


def apply_independent_beams(
    geometry,
    mesh: Mesh,
    beam_edge_ids: Iterable[int],
    size_field,
    offsets: Mapping[int, float | Sequence[float]],
    shell_edges: Iterable[int],
    cancellation_check: Callable[[str], None] | None = None,
) -> tuple[int, ...]:
    """Rebuild every qualifying beam on its own stations; returns the edge IDs.

    ``cancellation_check`` (the hybrid callback) is polled before the
    request-wide indexes are built, before every edge, and every
    ``STATION_POLL`` stations inside an edge.  The mesh is the caller's
    candidate working copy: cancelling raises out of this function and the
    caller publishes nothing, so no partial candidate can be returned.
    """

    # ``beam_offsets`` is optional on the public API (None); no offsets means
    # centerline beams.
    offsets = {} if offsets is None else offsets
    flagged = independent_beam_edges(geometry, mesh, beam_edge_ids, size_field, shell_edges)
    if not flagged:
        return ()
    _poll(cancellation_check, "independent beams: start")
    index = _Index(geometry, mesh, cancellation_check)
    for edge_id in sorted(flagged):
        _poll(cancellation_check, f"independent beams: edge {edge_id}")
        _rebuild_edge(
            geometry, mesh, edge_id, flagged[edge_id], size_field, offsets, index,
            cancellation_check,
        )
    known = {int(edge) for edge in mesh.hybrid_diagnostics.get(INDEPENDENT_BEAM_EDGES, ())}
    mesh.hybrid_diagnostics[INDEPENDENT_BEAM_EDGES] = sorted(known | set(flagged))
    return tuple(sorted(flagged))


class _LocalShells:
    """The shell elements touching one chain, shaped like a mesh for normals."""

    def __init__(self, mesh: Mesh, element_ids: Iterable[int], owner_of: Mapping[int, int]):
        self.nodes = mesh.nodes
        self._corners = mesh.corners_of
        shells: dict[int, tuple[int, ...]] = {}
        faces: dict[int, list[int]] = {}
        for element in sorted(element_ids):
            body = mesh.quads.get(element)
            if body is None:
                body = mesh.tris[element]
            shells[element] = body
            owner = owner_of.get(element)
            if owner is not None:
                faces.setdefault(owner, []).append(element)
        self.shells = shells
        self.elements_of_face = faces

    def corners_of(self, element_id: int):
        return self._corners(element_id)


class _Index:
    """Request-scoped lookups, built once and kept current while rebuilding."""

    def __init__(
        self, geometry, mesh: Mesh,
        cancellation_check: Callable[[str], None] | None = None,
    ) -> None:
        self.geometry = geometry
        self.mesh = mesh
        self.cancellation_check = cancellation_check
        self.node = max((0, *mesh.nodes), default=0)
        self.element = max(
            (0, *mesh.quads, *mesh.tris, *mesh.beams, *mesh.couplings), default=0
        )
        self.beam_refs: dict[int, int] = {}
        for body in mesh.beams.values():
            for node in body:
                self.beam_refs[int(node)] = self.beam_refs.get(int(node), 0) + 1
        self.couplings_of: dict[int, list[int]] = {}
        self.plate_couplings: dict[int, list[tuple[int, int, np.ndarray]]] = {}
        for record, coupling in mesh.couplings.items():
            self._note_coupling(int(record), coupling)
        self._shells_of_node: dict[int, list[int]] | None = None
        self._owner_of: dict[int, int] | None = None
        self._required: dict[int, list] | None = None
        self._view = None
        self._registry = None

    # -- ids -------------------------------------------------------------
    def next_node(self) -> int:
        self.node += 1
        return self.node

    def next_element(self) -> int:
        self.element += 1
        return self.element

    # -- couplings -------------------------------------------------------
    def _note_coupling(self, record: int, coupling: Coupling) -> None:
        beam = int(coupling.beam_node)
        self.couplings_of.setdefault(beam, []).append(record)
        if coupling.is_node_to_node:
            self.plate_couplings.setdefault(int(coupling.plate_node), []).append(
                (record, beam, np.asarray(coupling.eccentricity, dtype=float))
            )

    def add_coupling(self, coupling: Coupling) -> None:
        record = self.next_element()
        self.mesh.couplings[record] = coupling
        self._note_coupling(record, coupling)

    def single_owner(self, node: int, tolerance: float) -> None:
        """Leave one coupling record for a reused (shared) beam node.

        The ordinary mapped emitter can give a shared offset node one identical
        node-to-node record per beam that touches it.  For a node this rebuild
        keeps or reuses, exactly equivalent records collapse to the lowest
        record id; records that disagree (other masters, weights or offset)
        are never dropped - they are a typed failure.  Nodes this call does not
        touch are left exactly as generated.
        """

        records = sorted(
            record for record in self.couplings_of.get(node, ())
            if record in self.mesh.couplings
        )
        if len(records) < 2:
            return
        keep = self.mesh.couplings[records[0]]
        reference = sorted(
            (int(n), float(w)) for n, w in zip(keep.plate_nodes, keep.weights)
        )
        for record in records[1:]:
            other = self.mesh.couplings[record]
            masters = sorted(
                (int(n), float(w)) for n, w in zip(other.plate_nodes, other.weights)
            )
            if (
                len(masters) != len(reference)
                or any(
                    a[0] != b[0] or abs(a[1] - b[1]) > 1.0e-10
                    for a, b in zip(reference, masters)
                )
                or float(np.linalg.norm(
                    np.asarray(other.eccentricity, dtype=float)
                    - np.asarray(keep.eccentricity, dtype=float)
                )) > tolerance
            ):
                raise MeshError(
                    f"beam node {node} has competing couplings (records "
                    f"{records[0]} and {record}) that are not equivalent"
                )
        for record in records[1:]:
            del self.mesh.couplings[record]
        self.couplings_of[node] = [records[0]]

    def drop_node(self, node: int) -> None:
        for record in self.couplings_of.pop(node, ()):
            self.mesh.couplings.pop(record, None)
        self.mesh.nodes.pop(node, None)

    def shared_offset_node(self, plate_node: int, displacement: np.ndarray, tolerance: float):
        for record, beam, ecc in self.plate_couplings.get(plate_node, ()):
            if (
                record in self.mesh.couplings
                and beam in self.mesh.nodes
                and self.beam_refs.get(beam, 0) > 0
                and float(np.linalg.norm(ecc - displacement)) <= tolerance
            ):
                return beam
        return None

    # -- shells ----------------------------------------------------------
    def local_shells(self, chain: Sequence[int]) -> _LocalShells:
        mesh = self.mesh
        if self._shells_of_node is None:
            table: dict[int, list[int]] = {}
            for count, (element, body) in enumerate(
                [*mesh.quads.items(), *mesh.tris.items()]
            ):
                if count % 4096 == 0:
                    _poll(self.cancellation_check, "independent beams: shell index")
                for node in body:
                    table.setdefault(int(node), []).append(int(element))
            self._shells_of_node = table
            self._owner_of = {
                int(element): int(face)
                for face, elements in mesh.elements_of_face.items()
                for element in elements
            }
        touching = {
            element for node in chain for element in self._shells_of_node.get(int(node), ())
        }
        return _LocalShells(mesh, touching, self._owner_of)

    # -- declared stations -------------------------------------------------
    def required_parameters(self, edge_id: int) -> list[float]:
        """Edge parameters of declared attachment/junction point stations."""

        # Owner records (ANYgeometry structural): ``Attachment.member_id`` /
        # ``member_range`` and ``Junction.member_uses[*].member_id`` /
        # ``member_range``, read through the same meshing view the structural
        # pipeline uses.  A point station is only a station when it lies on a
        # member that uses this edge; one that cannot be resolved there is a
        # typed failure, never skipped.
        if self._required is None:
            from .boundary import MemberRegistry
            from .meshing_view import GeometryMeshingView

            view = GeometryMeshingView(self.geometry)
            self._view = view
            self._registry = MemberRegistry(view)
            by_member: dict[int, list] = {}
            for attachment in view.attachments.values():
                if attachment.member_id is not None and attachment.member_range.is_point:
                    by_member.setdefault(int(attachment.member_id), []).append(
                        ("attachment", int(attachment.id), attachment.member_range)
                    )
            for junction in view.junctions.values():
                for use in junction.member_uses:
                    if use.member_id is not None and use.member_range.is_point:
                        by_member.setdefault(int(use.member_id), []).append(
                            ("junction", int(junction.id), use.member_range)
                        )
            self._required = by_member  # type: ignore[assignment]
        edge = int(edge_id)
        found: set[float] = set()
        for member_id in sorted(int(item) for item in self._view.members_using_edge(edge)):
            for kind, record_id, member_range in self._required.get(member_id, ()):  # type: ignore[union-attr]
                try:
                    location = self._registry.locate(member_id, member_range.start)
                except (KeyError, ValueError) as error:
                    raise MeshError(
                        f"declared {kind} {record_id} station on member {member_id} "
                        f"cannot be resolved for beam line {edge}: {error}"
                    ) from error
                if int(location.span.edge_id) == edge:
                    found.add(float(location.edge_parameter))
        return sorted(found)


def _chain_weights(
    chain: Sequence[int], chain_parameters: np.ndarray, parameter: float, step: int
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Interpolation of one edge parameter over the actual canonical chain."""

    corners = chain_parameters[::step]
    last = len(corners) - 2
    element = int(np.clip(np.searchsorted(corners, parameter, side="right") - 1, 0, last))
    if step == 1:
        first, second = chain_parameters[element], chain_parameters[element + 1]
        span = float(second - first)
        if span <= 0.0:
            raise MeshError("shell edge stations are not increasing along the edge")
        tau = float((parameter - first) / span)
        return (
            (int(chain[element]), int(chain[element + 1])),
            (1.0 - tau, tau),
        )
    base = 2 * element
    a, b, c = (float(chain_parameters[base + offset]) for offset in range(3))
    if not (a < b < c):
        raise MeshError("shell edge stations are not increasing along the edge")
    weights = (
        (parameter - b) * (parameter - c) / ((a - b) * (a - c)),
        (parameter - a) * (parameter - c) / ((b - a) * (b - c)),
        (parameter - a) * (parameter - b) / ((c - a) * (c - b)),
    )
    return tuple(int(chain[base + offset]) for offset in range(3)), weights


def _rebuild_edge(
    geometry,
    mesh: Mesh,
    edge_id: int,
    divisions: int,
    size_field,
    offsets: Mapping[int, float | Sequence[float]],
    index: _Index,
    cancellation_check: Callable[[str], None] | None = None,
) -> None:
    from .mapped import _attached_face_normal, _offset_target_faces, _owner_normals_around, nodal_normals
    from .refinement import SizeField
    from .seeding import edge_distribution

    step = 2 if mesh.is_quadratic else 1
    chain = [int(node) for node in mesh.nodes_of_edge[edge_id]]
    chain_set = set(chain)
    length = float(geometry.edge_length(edge_id))
    tolerance = max(
        float(geometry.tolerance.effective_length(length)), 1.0e-12 * max(length, 1.0)
    )
    raw_offset = np.asarray(offsets.get(edge_id, 0.0), dtype=float).reshape(-1)
    eccentric = bool(np.any(raw_offset != 0.0))

    # Drop this edge's earlier beam: its elements, and any node that only it
    # used (a previous own-station node, an offset node) with that node's
    # couplings.  A node a neighbouring beam still references (a shared joint)
    # keeps its node *and* its couplings.
    old_nodes: set[int] = {
        int(node) for node in mesh.offset_nodes_of_edge.pop(edge_id, ())
    }
    for element in mesh.elements_of_edge.pop(edge_id, []):
        for node in mesh.beams.pop(int(element), ()):
            node = int(node)
            old_nodes.add(node)
            index.beam_refs[node] = index.beam_refs.get(node, 1) - 1
    for node in sorted(old_nodes - chain_set):
        if index.beam_refs.get(node, 0) <= 0:
            index.drop_node(node)

    # Stations: uniform in arc length at the beam's own size, keeping declared
    # attachment/junction point stations.
    corners = np.concatenate((
        [0.0],
        edge_distribution(
            geometry, edge_id, divisions, SizeField(geometry, size_field.beam_target(edge_id))
        ),
        [1.0],
    ))
    snapped = [float(value) for value in corners]
    for anchor in index.required_parameters(edge_id):
        if not (1.0e-9 < anchor < 1.0 - 1.0e-9):
            continue
        nearest = min(range(len(snapped)), key=lambda item: abs(snapped[item] - anchor))
        if abs(snapped[nearest] - anchor) <= 1.0e-10:
            continue
        snapped.append(float(anchor))
        snapped.sort()
    corners = np.asarray(snapped, dtype=float)
    if np.any(np.diff(corners) <= 1.0e-12):
        raise MeshError(f"edge {edge_id}: a declared station coincides with a beam corner")
    if step == 2:
        parameters = np.empty(2 * len(corners) - 1, dtype=float)
        parameters[::2] = corners
        parameters[1::2] = 0.5 * (corners[:-1] + corners[1:])
    else:
        parameters = corners
    axis_points = np.asarray(geometry.sample_edge(edge_id, parameters), dtype=float)

    chain_points = np.asarray([mesh.nodes[node] for node in chain], dtype=float)
    chain_parameters = np.asarray(
        [float(geometry.closest_edge_point(edge_id, point)[1]) for point in chain_points],
        dtype=float,
    )
    if np.any(np.diff(chain_parameters) <= 0.0):
        raise MeshError(f"edge {edge_id} shell stations are not increasing along the edge")

    normals: dict[int, np.ndarray] = {}
    target_faces: tuple[int, ...] = ()
    if eccentric and len(raw_offset) == 1:
        target_faces = _offset_target_faces(geometry, edge_id)
        if not target_faces:
            raise MeshError(
                f"line {edge_id} has a beam eccentricity but neither bounds nor has "
                "an exact attachment to a plate. Remove the eccentricity, or declare "
                "the member/sheet attachment."
            )
        local = index.local_shells(chain)
        normals = nodal_normals(
            local,
            element_owner_normals=_owner_normals_around(geometry, local, chain, edge_id),
            include_triangles=True,
        )
        for node in chain:
            if node not in normals:
                normals[node] = _attached_face_normal(
                    geometry, target_faces, mesh.nodes[node], edge_id=edge_id, node_id=node
                )

    beam_nodes: list[int] = []
    pending: list[Coupling] = []
    for station, (parameter, axis) in enumerate(zip(parameters, axis_points)):
        if station and station % STATION_POLL == 0:
            _poll(cancellation_check, f"independent beams: edge {edge_id} station {station}")
        distances = np.linalg.norm(chain_points - axis, axis=1)
        nearest = int(np.argmin(distances))
        coincident = chain[nearest] if float(distances[nearest]) <= tolerance else None
        masters = weights = None
        if coincident is None or eccentric:
            masters, weights = _chain_weights(chain, chain_parameters, float(parameter), step)

        if not eccentric:
            if coincident is not None:
                beam_nodes.append(coincident)
                continue
            node = index.next_node()
            mesh.nodes[node] = np.array(axis, dtype=float)
            interpolated = sum(w * mesh.nodes[m] for w, m in zip(weights, masters))
            pending.append(Coupling(
                node, masters, tuple(float(w) for w in weights),
                tuple(float(v) for v in mesh.nodes[node] - interpolated),
            ))
            beam_nodes.append(node)
            continue

        if len(raw_offset) == 3:
            displacement = np.asarray(raw_offset, dtype=float)
        else:
            if coincident is not None:
                direction = normals[coincident]
            else:
                direction = sum(w * normals[m] for w, m in zip(weights, masters))
                norm = float(np.linalg.norm(direction))
                if norm <= 0.0:
                    direction = _attached_face_normal(
                        geometry, target_faces, axis, edge_id=edge_id, node_id=-1
                    )
                else:
                    direction = direction / norm
            displacement = float(raw_offset[0]) * np.asarray(direction, dtype=float)
        if coincident is not None:
            shared = index.shared_offset_node(int(coincident), displacement, tolerance)
            if shared is not None:
                beam_nodes.append(int(shared))
                continue
        node = index.next_node()
        mesh.nodes[node] = np.asarray(axis + displacement, dtype=float)
        if coincident is not None:
            pending.append(Coupling.node_to_node(
                node, coincident, mesh.nodes[node] - mesh.nodes[coincident]
            ))
        else:
            interpolated = sum(w * mesh.nodes[m] for w, m in zip(weights, masters))
            pending.append(Coupling(
                node, masters, tuple(float(w) for w in weights),
                tuple(float(v) for v in mesh.nodes[node] - interpolated),
            ))
        beam_nodes.append(node)

    new_elements: list[int] = []
    for start in range(0, len(beam_nodes) - step, step):
        element = index.next_element()
        body = tuple(int(node) for node in beam_nodes[start : start + step + 1])
        mesh.beams[element] = body
        for node in body:
            index.beam_refs[node] = index.beam_refs.get(node, 0) + 1
        new_elements.append(element)
    mesh.elements_of_edge[edge_id] = new_elements
    # Coupling records follow the beam elements, as the ordinary builder does.
    for coupling in pending:
        index.add_coupling(coupling)
    # Match the existing structural coupling-conflict comparison: eccentricity
    # equivalence uses the owner absolute length tolerance, not edge-length scale.
    coupling_tolerance = float(geometry.tolerance.effective_length(0.0))
    for node in dict.fromkeys(beam_nodes):
        index.single_owner(int(node), coupling_tolerance)
