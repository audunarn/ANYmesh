"""Mapped plate regeneration on a declared, fixed interface of a prior mesh."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace

from anygeometry.curves import Straight
from anygeometry.surfaces import CoonsSurface, Plane

import numpy as np

from . import mapped
from .errors import MeshError
from .mesh import Mesh
from .regional_update import (
    RegionalAssembly, RegionalQualityLimits, _as_int, assemble_regional_replacement,
)
from .seeding import Seeding, solve_seeding

__all__ = ["remesh_mapped_region"]


def _face_users(geometry):
    edges = defaultdict(set)
    vertices = defaultdict(set)
    for face in geometry.faces.values():
        surface = face.surface
        if (face.holes or len(face.loop) != 4 or len(face.corners) != 4
                or face.parameterization is not None
                or not (type(surface) is Plane or type(surface) is CoonsSurface and surface.bottom is None)
                or any(type(geometry.edges[use.edge].curve) is not Straight for use in face.loop)):
            raise MeshError(f"face {face.id} is outside the supported four-sided mapped plates")
        for use in face.loop:
            edges[use.edge].add(face.id)
            edge = geometry.edges[use.edge]
            vertices[edge.start].add(face.id)
            vertices[edge.end].add(face.id)
    if any(len(faces) > 2 for faces in edges.values()):
        raise MeshError("a mapped region cannot cross non-manifold edges")
    return edges, vertices


def _shell_sides(mesh):
    # Anchor edge associations to actual shell sides and their midside nodes.
    sides = defaultdict(list)
    width = 8 if mesh.order == "quadratic" else 4
    seen = set()
    for face, elements in mesh.elements_of_face.items():
        for eid in elements:
            if eid in seen:
                raise MeshError(f"element {eid} has ambiguous face ownership")
            seen.add(eid)
            nodes = mesh.quads.get(eid)
            if nodes is None or len(nodes) != width:
                raise MeshError(f"face {face} contains an element outside the mapped quad family")
            for i in range(4):
                key = tuple(sorted((nodes[i], nodes[(i+1) % 4])))
                sides[key].append((face, nodes[4+i] if width == 8 else None))
    if seen != set(mesh.quads):
        raise MeshError("mapped face ownership does not cover every element")
    return sides


def _edge_stations(mesh, geometry, edge_id, faces, sides):
    edge = geometry.edges[edge_id]
    stations = tuple(mesh.nodes_of_edge.get(edge_id, ()))
    step = 2 if mesh.order == "quadratic" else 1
    if (len(stations) <= step or (len(stations)-1) % step or len(set(stations)) != len(stations)
            or stations[0] != mesh.node_of_vertex.get(edge.start)
            or stations[-1] != mesh.node_of_vertex.get(edge.end)):
        raise MeshError(f"edge {edge_id} has invalid ordered stations")
    for i in range(0, len(stations)-1, step):
        current = sides.get(tuple(sorted((stations[i], stations[i+step]))), ())
        mid = stations[i+1] if step == 2 else None
        if sorted(current) != sorted((f, mid) for f in faces):
            raise MeshError(f"edge {edge_id} station {i} is not the common side of its owning faces")
    return stations


def _segments(stations, order):
    corners = stations[::2 if order == "quadratic" else 1]
    return {tuple(sorted(pair)) for pair in zip(corners, corners[1:])}


def _fresh_numbers(ids, prior_max, high_water, kind):
    high_water = _as_int(high_water, f"{kind} high water")
    if high_water < 0 or high_water > 2**63-1:
        raise MeshError(f"invalid {kind} high water")
    start = max(prior_max, high_water)
    if len(ids) > 2**63-1-start:
        raise MeshError(f"{kind} numbering exceeds signed 64-bit range")
    return {temp: start+i for i, temp in enumerate(sorted(ids), start=1)}


def remesh_mapped_region(
    previous: Mesh, geometry, *, affected_faces: Iterable[int],
    interface_stations: Mapping[int, Sequence[int]], interface_vertices: Mapping[int, int],
    seeding: Seeding, target_size: float, node_high_water: int, element_high_water: int,
    quality: RegionalQualityLimits, cancellation_check: Callable[[str], None] | None = None,
) -> RegionalAssembly:
    """Generate only the selected mapped faces and splice them onto frozen stations.

    The caller proves source continuity. Supply every edge between a selected and retained
    face, with its prior node IDs ordered in source edge direction (including Q8 midsides).
    Supply every shared source vertex and its prior node ID, including vertex-only contacts.
    Geometry must match the prior mesh's source identity/revision. Uniform mapped seeding,
    shell faces and complete declared junction edges are supported. Returned sets and quality
    describe the newly assembled candidate. No project state or input mesh is mutated.
    """
    if (previous.geometry_model_id is None or str(previous.geometry_model_id) != str(geometry.model_id)
            or previous.geometry_revision != geometry.revision):
        raise MeshError("prior mesh and source geometry identity/revision differ")
    if (previous.order not in {"linear", "quadratic"} or previous.tris or previous.beams or previous.couplings
            or previous.automatic_beam_connections or previous.automatic_shell_connections):
        raise MeshError("mapped region requires a linear or quadratic quad-only plate mesh")
    if (previous.seeding is not None and previous.seeding.size_field is not None
            and not previous.seeding.size_field.is_uniform):
        raise MeshError("prior mapped seeding uses a nonuniform size field")
    if seeding.size_field is not None and not seeding.size_field.is_uniform:
        raise MeshError("mapped regional regeneration requires a uniform size field")
    faces = {_as_int(f, "affected face") for f in affected_faces}
    if not faces or not faces <= set(geometry.faces) or set(previous.elements_of_face) != set(geometry.faces):
        raise MeshError("affected mapped faces are missing or the prior face ownership differs")
    if faces & set(previous.thickness_of_face):
        raise MeshError("affected plate has mesh-owned thickness")

    edge_users, vertex_users = _face_users(geometry)
    boundary_edges = {e for e, users in edge_users.items() if users & faces and users - faces}
    boundary_vertices = {v for v, users in vertex_users.items() if users & faces and users - faces}
    supplied_edges = {_as_int(e, "frozen edge") for e in interface_stations}
    supplied_vertices = {_as_int(v, "frozen vertex") for v in interface_vertices}
    if supplied_edges != boundary_edges or supplied_vertices != boundary_vertices:
        raise MeshError("frozen interface does not cover exactly the shared source edges and vertices")

    prior_sides = _shell_sides(previous)
    prior_stations = {e: _edge_stations(previous, geometry, e, users, prior_sides)
                      for e, users in edge_users.items()}
    for e, ids in interface_stations.items():
        if tuple(_as_int(n, "frozen station") for n in ids) != prior_stations[e]:
            raise MeshError(f"edge {e} does not match its frozen ordered stations")
    for v, n in interface_vertices.items():
        if previous.node_of_vertex.get(v) != _as_int(n, "frozen vertex node"):
            raise MeshError(f"vertex {v} does not match its frozen node")

    declared = {tuple(sorted(pair)) for pair in previous.declared_plate_junction_edges}
    junctions = {e: _segments(prior_stations[e], previous.order)
                 for e, users in edge_users.items() if len(users) == 2}
    proven = {e for e, segments in junctions.items() if segments <= declared}
    if set().union(*(junctions[e] for e in proven)) != declared or previous.automatic_intersections > len(proven):
        raise MeshError("prior intersections lack complete declared source-edge junctions")

    active_edges = {e for e, users in edge_users.items() if users & faces}
    if not active_edges <= set(seeding.divisions):
        raise MeshError("mapped seeding lacks an affected edge division")
    if cancellation_check:
        cancellation_check("connected generate")
    work_seeding = replace(seeding, divisions=dict(seeding.divisions), classes=dict(seeding.classes))
    sub = mapped.generate_mesh(geometry, target_size=target_size, seeding=work_seeding,
                               face_ids=tuple(sorted(faces)), order=previous.order)
    if set(sub.elements_of_face) != faces or sub.order != previous.order:
        raise MeshError("mapped generator returned a different face selection or order")
    new_sides = _shell_sides(sub)
    stations = {e: _edge_stations(sub, geometry, e, edge_users[e] & faces, new_sides)
                for e in active_edges}

    interface = {}
    def bind(temp, prior, ref):
        if interface.setdefault(temp, prior) != prior or not np.array_equal(sub.nodes[temp], previous.nodes[prior]):
            raise MeshError(f"{ref} differs from the frozen interface station")

    for e in boundary_edges:
        if len(stations[e]) != len(prior_stations[e]):
            raise MeshError(f"edge {e} station count differs from the frozen interface")
        for temp, prior in zip(stations[e], prior_stations[e]):
            bind(temp, prior, f"edge {e}")
    for v, prior in interface_vertices.items():
        bind(sub.node_of_vertex[v], prior, f"vertex {v}")

    for sheet in geometry.sheets.values():
        elements = sorted({e for use in sheet.face_use_ids
                           for e in sub.elements_of_face.get(geometry.face_uses[use].face_id, ())})
        if elements:
            sub.elements_of_sheet[sheet.id] = elements
    sub.declared_plate_junction_edges = tuple(sorted(set().union(
        *(_segments(stations[e], previous.order) for e in proven & active_edges))))
    # The assembly accepts a partial seed table only for an identical prior field.
    # Resolve the whole, uniform, explicitly pinned field from actual station counts
    # below, so stale or unrelated seeding metadata never survives the splice.
    sub.seeding = None
    fresh_nodes = _fresh_numbers(set(sub.nodes)-set(interface), max(previous.nodes),
                                 node_high_water, "node")
    fresh_elements = _fresh_numbers(set(sub.quads), max(previous.quads),
                                    element_high_water, "element")
    if cancellation_check:
        cancellation_check("connected assemble")
    result = assemble_regional_replacement(
        previous, sub, quality=quality, node_ids=fresh_nodes, element_ids=fresh_elements,
        affected_element_ids={e for f in faces for e in previous.elements_of_face[f]},
        interface_node_ids=interface, stations={n: previous.nodes[n] for n in interface.values()},
        discard_stale_diagnostics=True,
    )
    final_junctions = set()
    for e in proven:
        ordered = tuple(result.mesh.nodes_of_edge[e])
        final_junctions.update(_segments(ordered, previous.order))
    if {tuple(sorted(p)) for p in result.mesh.declared_plate_junction_edges} != final_junctions:
        raise MeshError("assembled junction segments differ from the declared complete edges")
    if previous.seeding is not None:
        step = 2 if previous.order == "quadratic" else 1
        counts = {e: (len(ids)-1)//step for e, ids in result.mesh.nodes_of_edge.items() if e in edge_users}
        result.mesh.seeding = solve_seeding(geometry, target_size=target_size, overrides=counts,
                                            edge_ids=sorted(counts))
        if result.mesh.seeding.divisions != counts:
            raise MeshError("assembled stations differ from the resolved mapped seeding")
        changed = tuple(sorted(e for e, count in counts.items() if previous.seeding.divisions.get(e) != count))
        result = replace(result, seeding_changed_edges=changed)
    return result
