"""Mapped plate regeneration on a declared, fixed interface of a prior mesh.

Why the O-grid sector Jacobian proof applies to what ``mapped.generate_mesh`` builds
------------------------------------------------------------------------------------
Sources: ``mapped._build_face`` / ``mapped.coons_grid`` (discrete transfinite blend on
``u = linspace(0, 1, n_u)``, ``v = linspace(0, 1, n_v)``, boundary nodes taken from
``mapped._build_edge_nodes``), ``seeding.edge_distribution`` (station parameters) and
``seeding.solve_seeding`` / ``mapped._build_face`` (opposite sides share a division count).

1. *Blend reduction.*  With the loop A (arc, p_a -> p_b), B (spoke p_b -> q_b), C (chord
   q_b -> q_a), D (spoke q_a -> p_a), ``coons_grid`` gets A(u), L(u) = C reversed, and
   spokes B(v) = p_b + v (q_b - p_b), D(v) = p_a + v (q_a - p_a).  Writing
   Pl(u) = (1-u) p_a + u p_b, the spoke terms sum to Pl + v (L - Pl) and the bilinear
   corner term is (1-v) Pl + v L, so ``P = (1-v) A(u) + v L(u)`` identically.  This needs
   (a) spoke stations at exactly j / n_v on straight edges, and (b) arc and chord stations
   at exactly i / n_u (the same ``u`` on both).  Any n_u, n_v >= 1 and either element
   order work: a Q8 grid is the same map sampled at 2 n stations.
2. *Station parameters.*  ``edge_distribution`` inverts the cumulative size-weighted chord
   length of 64 equal parameter samples with ``np.interp``.  Under a uniform size field
   every chord of a straight edge or constant-angular-speed arc has the same length, so
   the stations are k / N to rounding.  A graded field would not be.  This module
   therefore (i) refuses non-uniform size fields before generating, and (ii) after
   generating, re-checks every edge station of an arc face against ``sample_edge`` at
   exactly k / N (``_check_uniform_stations``); a mismatch raises, never falls back.
3. *Edge parametrizations.*  The predicate samples the arc at t = 1/4, 1/2, 3/4 against
   the constant-angular-speed circle of ``arc_frame`` and each straight edge at the same
   t against the line through its ends, so the "exact arc / linear straight" premises are
   checked on the actual geometry rather than assumed.
4. *Jacobian.*  See ``_ogrid_sector_refusal``: det(dF) is affine in t, strictly negative at
   t = 0 and t = 1 whenever Ro cos^2(h) > r, so F is regular, and with a simple closed
   boundary injective.  This certifies the *continuous* map; the discrete cells are still
   gated by ``mapped.grid_folds`` and the caller's ``RegionalQualityLimits``.

Sectors that violate the premises are refused (fail closed), not approximated.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace

from anygeometry.curves import Arc, Straight, arc_frame
from anygeometry.errors import GeometryError
from anygeometry.surfaces import CoonsSurface, Plane

import numpy as np

from . import mapped
from .errors import MeshError
from .mesh import Mesh
from .regional_update import (
    RegionalAssembly, RegionalQualityLimits, _as_int, assemble_regional_replacement,
)
from .seeding import Seeding, solve_seeding

__all__ = ["mapped_regional_face_supported", "remesh_mapped_region"]

_LENGTH_RTOL = 1.0e-9  # coplanarity / station tolerance, relative to the patch extent
_INNER = np.array([0.25, 0.5, 0.75])  # interior parameters probing an edge's parametrization


def _ogrid_sector_refusal(geometry, face):
    """Why a one-arc four-edge face is not a certified O-grid sector, else ``None``.

    Admitted shape (and nothing more general): an inner circular arc about centre O, radius
    r and angle 0 < theta < pi; two straight spokes that continue the arc's end radii
    outward from O; and a straight outer chord joining the spoke ends, which lie at one
    common radius Ro.  With half angle h = theta/2 the Coons map is, because the straight
    spokes cancel the bilinear cross terms, exactly ``F(s, t) = (1-t) A(s) + t L(s)`` with
    A the arc at polar angle phi = h (2s - 1) and L the chord.  Its Jacobian is affine
    in t, so it keeps one strict sign on the patch when both t = 0 and t = 1 do:

    * t = 1: det = -2 Ro sin(h) (Ro cos(h) - r cos(phi)) < 0 whenever Ro cos(h) > r;
    * t = 0: det = -2 h r (L . radial - r) and L . radial >= Ro cos(h) cos(phi)
      >= Ro cos(h)^2, because the transverse parts of L and of the unit radial vector
      have the same sign; so det < 0 whenever Ro cos(h)^2 > r.

    Ro cos(h) > r also keeps the chord (x = Ro cos h) clear of the arc (x <= r), so the
    boundary is a simple closed curve and a regular map of it is injective.  The test is
    sufficient, not necessary: sectors it cannot prove are refused (fail closed).  Every
    authored deviation is bounded by ``eps`` and charged against the margin.  The only
    sampling is the fixed-parameter probe of each edge's parametrization (module docstring,
    point 3), which backs the premise that the mapped blend equals F(s, t).
    """
    sides = face.sides()
    if any(len(side) != 1 for side in sides):
        return "each mapped side must be exactly one edge"
    uses = face.loop
    if len({use.edge for use in uses}) != 4:
        return "the loop repeats an edge"
    edges = [geometry.edges[use.edge] for use in uses]
    starts = [e.start if u.forward else e.end for u, e in zip(uses, edges)]
    ends = [e.end if u.forward else e.start for u, e in zip(uses, edges)]
    if any(ends[i] != starts[(i+1) % 4] for i in range(4)) or len(set(starts)) != 4:
        return "the loop is not a closed chain of four distinct vertices"
    kinds = [type(e.curve) for e in edges]
    if kinds.count(Arc) != 1 or kinds.count(Straight) != 3:
        return "the loop is not one arc and three straight edges"
    a = kinds.index(Arc)  # loop order from here: arc, spoke, outer chord, spoke
    corners = [np.asarray(geometry.vertex_position(v), dtype=float) for v in starts]
    p_a, p_b = corners[a], corners[(a+1) % 4]            # arc ends (loop direction)
    q_b, q_a = corners[(a+2) % 4], corners[(a+3) % 4]    # outer ends of the spokes at p_b, p_a
    arc_edge = edges[a]
    via = np.asarray(geometry.vertex_position(arc_edge.curve.via_vertex), dtype=float)
    frame = arc_frame(geometry.vertex_position(arc_edge.start), via, geometry.vertex_position(arc_edge.end))
    centre, normal_arc = np.asarray(frame.center, dtype=float), np.asarray(frame.normal, dtype=float)
    r, sweep = float(frame.radius), float(frame.sweep)
    authored = np.array([*corners, via, centre, normal_arc], dtype=float)
    if not np.all(np.isfinite(authored)) or not np.isfinite([r, sweep]).all() or r <= 0.0:
        return "the authored geometry is not finite"
    points = np.array([*corners, via, centre])
    extent = max(float(np.linalg.norm(np.ptp(points, axis=0))), r)
    tol = _LENGTH_RTOL * extent
    eps = 2.0 * tol  # in-plane plus out-of-plane authored deviation bound
    surface = face.surface
    if type(surface) is Plane:
        origin, normal = np.asarray(surface.origin, dtype=float), np.asarray(surface.normal, dtype=float)
    else:
        origin, normal = centre, normal_arc
    if float(np.linalg.norm(np.cross(normal, normal_arc))) > _LENGTH_RTOL:
        return "the arc axis is not normal to the face plane"
    if float(np.max(np.abs((points - origin) @ normal))) > tol:
        return "the corners, arc via point and arc centre are not coplanar with the face"

    for edge in edges:
        ends_xyz = np.asarray(geometry.sample_edge(edge.id, np.array([0.0, 1.0])), dtype=float)
        if ends_xyz.shape != (2, 3) or not np.all(np.isfinite(ends_xyz)):
            return "an edge cannot be evaluated"
        if (np.linalg.norm(ends_xyz[0] - np.asarray(geometry.vertex_position(edge.start), dtype=float)) > tol
                or np.linalg.norm(ends_xyz[1] - np.asarray(geometry.vertex_position(edge.end), dtype=float)) > tol):
            return "an edge leaves its authored vertices"
        if type(edge.curve) is Straight:
            inner = np.asarray(geometry.sample_edge(edge.id, _INNER), dtype=float)
            line = ends_xyz[0] + _INNER[:, None]*(ends_xyz[1] - ends_xyz[0])
            if (inner.shape != (3, 3) or not np.all(np.isfinite(inner))
                    or float(np.max(np.linalg.norm(inner - line, axis=1))) > tol):
                return "a straight edge is not parametrized linearly"

    radial = []
    for p, q in ((p_a, q_a), (p_b, q_b)):
        u = p - centre
        rho = float(np.linalg.norm(u))
        if abs(rho - r) > tol:
            return "an arc end is not on the authored circle"
        u = u / rho
        v = q - centre
        along = float(v @ u)
        if float(np.linalg.norm(v - along*u)) > tol:
            return "a spoke is not collinear with the circle centre"
        if along <= r:
            return "a spoke does not run outward from the circle"
        radial.append((u, float(np.linalg.norm(v))))
    (u_a, ro_a), (u_b, ro_b) = radial
    if abs(ro_a - ro_b) > tol:
        return "the outer chord ends are not at one common radius"

    theta_arc = abs(sweep)
    theta_ends = float(np.arctan2(np.linalg.norm(np.cross(u_a, u_b)), u_a @ u_b))  # in [0, pi]
    delta = 4.0 * eps / r  # angular allowance for positions known to eps on radius r
    if abs(theta_arc - theta_ends) > delta:
        return "the authored arc is not the minor arc between its end radii"
    bisector = u_a + u_b
    if float(np.linalg.norm(bisector)) <= 0.0:
        return "the arc sweeps a half circle"
    mid = np.asarray(geometry.sample_edge(arc_edge.id, np.array([0.5])), dtype=float)
    if mid.shape != (1, 3) or float(np.linalg.norm(mid[0] - (centre + r*bisector/np.linalg.norm(bisector)))) > tol:
        return "the arc does not evaluate to its authored circle"
    angles = frame.sweep*_INNER
    circle = centre + r*(np.outer(np.cos(angles), frame.e1) + np.outer(np.sin(angles), frame.e2))
    inner = np.asarray(geometry.sample_edge(arc_edge.id, _INNER), dtype=float)
    if (inner.shape != (3, 3) or not np.all(np.isfinite(inner))
            or float(np.max(np.linalg.norm(inner - circle, axis=1))) > tol):
        return "the arc is not parametrized at constant angular speed"
    h_lo = 0.5*min(theta_arc, theta_ends) - delta
    h_hi = 0.5*max(theta_arc, theta_ends) + delta
    if not h_lo > 0.0:
        return "the arc angle is not positive"
    if not h_hi < 0.5*np.pi:
        return "the arc sweeps half a circle or more"
    ro_lo, r_hi, cos_h = min(ro_a, ro_b) - eps, r + eps, float(np.cos(h_hi))
    if not (ro_lo*cos_h > r_hi + eps and ro_lo*cos_h*cos_h > r_hi + eps):
        return "the sector radial Jacobian margin is not proven positive"
    return None


def _face_refusal(geometry, face_id):
    """``None`` when the face is a certified mapped plate, else a (possibly empty) reason."""
    try:
        face = geometry.faces.get(face_id)
        if face is None:
            return "it is not in the source geometry"
        surface = face.surface
        if (face.holes or len(face.loop) != 4 or len(face.corners) != 4
                or face.parameterization is not None
                or not (type(surface) is Plane or type(surface) is CoonsSurface and surface.bottom is None)):
            return ""
        curves = [type(geometry.edges[use.edge].curve) for use in face.loop]
        if all(curve is Straight for curve in curves):
            return None
        if curves.count(Arc) == 1 and curves.count(Straight) == 3:
            return _ogrid_sector_refusal(geometry, face)
        return ""
    except (GeometryError, ValueError, TypeError, KeyError, AttributeError, ArithmeticError,
            np.linalg.LinAlgError, StopIteration):
        return "its analytic geometry cannot be certified"


def mapped_regional_face_supported(geometry, face_id) -> bool:
    """Whether ``remesh_mapped_region`` can regenerate this source face.

    Admitted: a four-edge face with explicit Plane or topology-backed Coons support and
    straight edges (unchanged), or a planar symmetric radial O-grid sector: one true
    circular arc (angle strictly between 0 and pi) with two straight spokes collinear
    outward from its centre and a straight outer chord at the spokes' common radius, whose
    analytic Jacobian margin (see ``_ogrid_sector_refusal``) is proven positive.
    This is the same predicate the remesher applies; it never raises.
    """
    try:
        face_id = _as_int(face_id, "face")
    except (MeshError, TypeError, ValueError, OverflowError):
        return False
    return _face_refusal(geometry, face_id) is None


def _face_users(geometry):
    edges = defaultdict(set)
    vertices = defaultdict(set)
    for face in geometry.faces.values():
        reason = _face_refusal(geometry, face.id)
        if reason is not None:
            raise MeshError(f"face {face.id} is outside the supported four-sided mapped plates"
                            + (f" ({reason})" if reason else ""))
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
    if type(edge.curve) is Arc:
        _check_arc_stations(mesh, geometry, edge, stations)
    return stations


def _check_arc_stations(mesh, geometry, edge, stations):
    """Every station is monotone along, and exactly on, the edge's actual arc parameter."""
    try:
        frame = arc_frame(geometry.vertex_position(edge.start),
                          geometry.vertex_position(edge.curve.via_vertex), geometry.vertex_position(edge.end))
        points = np.array([mesh.nodes[n] for n in stations], dtype=float)
        rel = points - frame.center
        angle = np.arctan2(rel @ frame.e2, rel @ frame.e1)
        angle = np.mod(angle, 2.0*np.pi) if frame.sweep > 0.0 else -np.mod(-angle, 2.0*np.pi)
        t = angle / frame.sweep
        t[0], t[-1] = 0.0, 1.0  # ends are the vertices; their wrap is resolved by the check below
        tol = _LENGTH_RTOL * max(frame.radius, frame.length)
        ok = (bool(np.all(np.isfinite(t))) and bool(np.all(np.diff(t) > 0.0)) and float(t[1]) > 0.0
              and float(t[-2]) < 1.0)
        if ok:
            expected = np.asarray(geometry.sample_edge(edge.id, t), dtype=float)
            ok = (expected.shape == points.shape
                  and float(np.max(np.linalg.norm(expected - points, axis=1))) <= tol)
    except (GeometryError, ValueError, TypeError, KeyError, ArithmeticError, IndexError):
        ok = False
    if not ok:
        raise MeshError(f"edge {edge.id} stations are not ordered on its curved edge geometry")


def _check_uniform_stations(mesh, geometry, edge_id, stations):
    """Stations sit at exactly k / N of the edge parameter (premise of the sector proof)."""
    try:
        points = np.array([mesh.nodes[n] for n in stations], dtype=float)
        expected = np.asarray(geometry.sample_edge(edge_id, np.linspace(0.0, 1.0, len(stations))),
                              dtype=float)
        tol = _LENGTH_RTOL * max(float(geometry.edge_length(edge_id)), 1.0e-300)
        ok = (expected.shape == points.shape
              and float(np.max(np.linalg.norm(expected - points, axis=1))) <= tol)
    except (GeometryError, ValueError, TypeError, KeyError, ArithmeticError, IndexError):
        ok = False
    if not ok:
        raise MeshError(f"edge {edge_id} stations are not uniform in the edge parameter, "
                        "which the arc sector Jacobian proof requires")


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
    shell faces and complete declared junction edges are supported. Every face must satisfy
    ``mapped_regional_face_supported`` (straight four-edge plates, or planar O-grid sectors
    with one true circular arc); curved edge stations are checked against the edge's actual arc,
    and every edge of an arc face must carry stations uniform in its parameter. Returned sets and quality
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
    for f in sorted(faces):
        loop = geometry.faces[f].loop
        if any(type(geometry.edges[use.edge].curve) is Arc for use in loop):
            for use in loop:
                _check_uniform_stations(sub, geometry, use.edge, stations[use.edge])

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
