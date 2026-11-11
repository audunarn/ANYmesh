"""Mapped regional remeshing of planar one-arc four-edge patches (A8 O-grid shape)."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from anygeometry import GeometryModel
from anygeometry.curves import Arc, Straight
from anygeometry.errors import GeometryError
from anygeometry.surfaces import CoonsSurface, Plane
from anymesher import mapped
from anymesher.errors import MeshError
from anymesher.regional_mapped import (
    _face_refusal, mapped_regional_face_supported, remesh_mapped_region,
)
from anymesher.regional_update import RegionalQualityLimits
from anymesher.seeding import solve_seeding
from anymesher.serialize import mesh_to_dict

R, B = 0.05, 0.12
PLANE = Plane(origin=(0.0, 0.0, 0.0), u_vector=(1.0, 0.0, 0.0), v_vector=(0.0, 1.0, 0.0))
DEFAULT = object()


def ogrid(surface=PLANE, lift=0.0):
    """Four patches around a hole, as ANYfem's e4_hole.py builds them."""
    g = GeometryModel()
    corners = [(-B, -B), (B, -B), (B, B), (-B, B)]
    c = [g.add_point(x, y, lift if k == 1 else 0.0) for k, (x, y) in enumerate(corners)]
    pts = []
    for x, y in corners:
        rho = float(np.hypot(x, y))
        pts.append(g.add_point(R*x/rho, R*y/rho, 0.0))
    vias = [(0.0, -R), (R, 0.0), (0.0, R), (-R, 0.0)]
    arcs, spokes, sides = [], [], []
    for k in range(4):
        via = g.add_point(vias[k][0], vias[k][1], 0.0)
        arcs.append(g.add_arc(pts[k], via, pts[(k+1) % 4]))
        spokes.append(g.add_line(c[k], pts[k]))
        sides.append(g.add_line(c[k], c[(k+1) % 4]))
    kwargs = {} if surface is DEFAULT else {"surface": surface}
    faces = [g.add_face([sides[k], spokes[(k+1) % 4], arcs[k], spokes[k]], **kwargs) for k in range(4)]
    return g, dict(c=c, pts=pts, arcs=arcs, spokes=spokes, sides=sides, faces=faces)


def request(order, surface=PLANE):
    g, t = ogrid(surface)
    pinned = {e: 4 for e in g.edges}
    old = mapped.generate_mesh(g, target_size=0.05, overrides=pinned, order=order)
    arc, side = t["arcs"][0], t["sides"][0]
    seed = solve_seeding(g, target_size=0.05, overrides={**pinned, arc: 6, side: 6})
    shared = (t["spokes"][0], t["spokes"][1])
    verts = (t["c"][0], t["pts"][0], t["c"][1], t["pts"][1])
    args = dict(affected_faces=(t["faces"][0],),
                interface_stations={e: tuple(old.nodes_of_edge[e]) for e in shared},
                interface_vertices={v: old.node_of_vertex[v] for v in verts},
                seeding=seed, target_size=0.05, node_high_water=1000, element_high_water=2000,
                quality=RegionalQualityLimits(max_aspect_ratio=6., min_scaled_jacobian=0.2))
    return g, t, old, args


class Altered:
    """The real (valid) geometry with some face records or vertex positions replaced.

    The owner API is otherwise delegated, so malformed or conflicting records can be
    shown to the predicate without asking GeometryModel to author an invalid surface.
    """

    def __init__(self, base, moved=None, **faces):
        self._base = base
        self._moved = {int(k): tuple(v) for k, v in (moved or {}).items()}
        self.faces = {**base.faces, **{int(k[1:]): v for k, v in faces.items()}}

    def vertex_position(self, vertex):
        return self._moved.get(int(vertex), self._base.vertex_position(vertex))

    def __getattr__(self, name):
        return getattr(self._base, name)


def sector(r=1.0, ro=3.0, h_deg=45.0, rot_deg=0.0, mirror=False, shift=0, centre=(0.0, 0.0),
           surface=PLANE, ro_b=None, turn_a=0.0, outer_arc=False):
    """One planar O-grid sector with valid owner geometry.

    Inner arc of radius ``r`` and half angle ``h_deg`` about direction ``rot_deg``; spokes
    continue its end radii out to ``ro`` (``ro_b`` for the second end, ``turn_a`` radians to
    turn the first spoke off its ray); the outer edge is a chord (or an arc if ``outer_arc``).
    ``mirror`` flips the loop orientation and ``shift`` rotates which edge the loop starts on.
    """
    sign = -1.0 if mirror else 1.0
    cx, cy = centre
    h, alpha = np.radians(h_deg), np.radians(rot_deg)

    def at(radius, angle):
        return (cx + radius*np.cos(angle), cy + sign*radius*np.sin(angle), 0.0)

    g = GeometryModel()
    p_a, p_b = g.add_point(*at(r, alpha-h)), g.add_point(*at(r, alpha+h))
    q_a = g.add_point(*at(ro, alpha-h+turn_a))
    q_b = g.add_point(*at(ro if ro_b is None else ro_b, alpha+h))
    via = g.add_point(*at(r, alpha))
    arc = g.add_arc(p_a, via, p_b)
    if outer_arc:
        bulge = g.add_point(*at(ro*1.2, alpha))
        outer = g.add_arc(q_b, bulge, q_a)
    else:
        outer = g.add_line(q_b, q_a)
    loop = [arc, g.add_line(p_b, q_b), outer, g.add_line(q_a, p_a)]
    loop = loop[shift:] + loop[:shift]
    kwargs = {} if surface is DEFAULT else {"surface": surface}
    return g, g.add_face(loop, **kwargs), dict(p_a=p_a, p_b=p_b, q_a=q_a, q_b=q_b, via=via)


@pytest.mark.parametrize("surface", [PLANE, DEFAULT])
def test_predicate_admits_source_backed_planar_arc_patches(surface):
    g, t = ogrid(surface)
    assert all(mapped_regional_face_supported(g, f) for f in t["faces"])


@pytest.mark.parametrize("surface", [PLANE, DEFAULT])
@pytest.mark.parametrize("kwargs", [
    dict(ro=3.0, h_deg=45.0),                                  # plain
    dict(ro=3.0, h_deg=45.0, rot_deg=37.0, mirror=True, shift=1, centre=(0.7, -0.4)),
    dict(ro=3.0, h_deg=10.0, rot_deg=200.0, shift=2),          # narrow sector
    dict(ro=40.0, h_deg=80.0, rot_deg=-130.0, mirror=True, shift=3),  # near half circle, proven
    dict(ro=4.001, h_deg=60.0, rot_deg=15.0),                  # just inside the margin
])
def test_predicate_admits_rotated_permuted_cw_ccw_sectors(kwargs, surface):
    g, face, _ = sector(surface=surface, **kwargs)
    assert mapped_regional_face_supported(g, face)


def test_predicate_tolerates_only_perturbations_charged_against_the_margin():
    g, face, ids = sector(ro=3.0)
    base = g.vertex_position(ids["q_a"])
    assert mapped_regional_face_supported(Altered(g, moved={ids["q_a"]: np.array(base) * (1.0 + 1.0e-12)}), face)
    assert not mapped_regional_face_supported(Altered(g, moved={ids["q_a"]: np.array(base) * (1.0 + 1.0e-6)}), face)


@pytest.mark.parametrize("surface", [PLANE, DEFAULT])
@pytest.mark.parametrize("kwargs", [
    dict(ro=2.0000001, h_deg=60.0),            # chord just clear of the circle: margin unproven
    dict(ro=4.0, h_deg=60.0),                  # Ro cos^2 h == r: no strict margin
    dict(ro=30.0, h_deg=80.0),                 # near-tangent; sound but not provable by the margin
    dict(ro=0.9, h_deg=45.0),                  # spokes run inward
    dict(ro=10.0, h_deg=100.0),                # more than a half circle
    dict(ro=3.0, h_deg=45.0, ro_b=3.01),       # outer ends at different radii
    dict(ro=3.0, h_deg=45.0, turn_a=1.0e-3),   # spoke not collinear with the centre
    dict(ro=3.0, h_deg=45.0, outer_arc=True),  # two arcs
])
def test_predicate_refuses_unprovable_folded_and_off_contract_sectors(kwargs, surface):
    g, face, _ = sector(surface=surface, **kwargs)
    assert not mapped_regional_face_supported(g, face)


@pytest.mark.parametrize("surface", [PLANE, DEFAULT])
@pytest.mark.parametrize("kwargs", [
    dict(ro=2.0, h_deg=60.0),     # chord exactly tangent to the circle
    dict(ro=5.0, h_deg=80.0),     # chord cuts the arc
    dict(ro=10.0, h_deg=90.0),    # exactly a half circle: chord overlays the spokes
])
def test_owner_rejects_self_intersecting_sectors_at_construction(kwargs, surface):
    # Recorded mesh-arc-02-check evidence: add_face itself raises "loop 0 self-intersects"
    # for these six cases (three parameter sets x explicit Plane / Coons).  They are
    # intrinsically invalid, so there is no source geometry for the predicate to see; the
    # predicate side of the same shapes is covered via Moved views below.
    with pytest.raises((GeometryError, ValueError), match="loop 0 self-intersects"):
        sector(surface=surface, **kwargs)


class Moved(Altered):
    """Altered whose straight edges follow the moved vertices (arcs are left untouched)."""

    def sample_edge(self, edge_id, t):
        edge = self._base.edges[int(edge_id)]
        if type(edge.curve) is not Straight:
            return self._base.sample_edge(edge_id, t)
        a = np.asarray(self.vertex_position(edge.start), dtype=float)
        b = np.asarray(self.vertex_position(edge.end), dtype=float)
        return a + np.atleast_1d(np.asarray(t, dtype=float))[:, None]*(b - a)


class Warped(Altered):
    """Altered whose one edge is reparametrized t -> t + 0.05 sin(2 pi t) (ends and middle fixed)."""

    def __init__(self, base, edge_id):
        super().__init__(base)
        self._edge = int(edge_id)

    def sample_edge(self, edge_id, t):
        t = np.asarray(t, dtype=float)
        if int(edge_id) == self._edge:
            t = t + 0.05*np.sin(2.0*np.pi*t)
        return self._base.sample_edge(edge_id, t)


def _move_outer_radius(g, ids, ro_new, ro_old):
    return Moved(g, moved={ids[k]: np.array(g.vertex_position(ids[k])) * (ro_new/ro_old)
                           for k in ("q_a", "q_b")})


@pytest.mark.parametrize("ro_new, h_deg", [
    (2.0, 60.0),    # tangent chord (owner refuses to author this directly)
    (5.0, 80.0),    # chord cuts the arc (likewise)
    (4.0, 60.0),    # Ro cos^2 h == r exactly
    (4.0*(1 + 1e-12), 60.0),  # margin smaller than the charged authored deviation
])
def test_predicate_refuses_margin_violations_on_valid_source_geometry(ro_new, h_deg):
    g, face, ids = sector(ro=40.0, h_deg=h_deg)
    assert mapped_regional_face_supported(g, face)
    moved = _move_outer_radius(g, ids, ro_new, 40.0)
    assert not mapped_regional_face_supported(moved, face)
    assert "Jacobian margin" in _face_refusal(moved, face)


@pytest.mark.parametrize("ro_new, h_deg, admitted", [
    (4.0*(1 + 1e-5), 60.0, True),    # just inside the margin
    (4.0*(1 - 1e-5), 60.0, False),   # just outside it
    (33.2, 80.0, True),              # Ro cos^2 h = 1.0011 at a near-tangent 80 degree half angle
    (33.1, 80.0, False),             # Ro cos^2 h = 0.998
])
def test_narrow_margins_are_admitted_only_when_proven(ro_new, h_deg, admitted):
    g, face, ids = sector(ro=40.0, h_deg=h_deg)
    assert mapped_regional_face_supported(_move_outer_radius(g, ids, ro_new, 40.0), face) is admitted


def test_unprovable_but_buildable_sector_names_the_margin():
    g, face, _ = sector(ro=30.0, h_deg=80.0)
    assert "Jacobian margin" in _face_refusal(g, face)


def test_predicate_refuses_edges_that_are_not_exactly_parametrized():
    g, face, _ = sector(ro=3.0)
    assert mapped_regional_face_supported(g, face)
    loop = g.faces[face].loop
    arc_edge = next(u.edge for u in loop if type(g.edges[u.edge].curve) is Arc)
    line_edge = next(u.edge for u in loop if type(g.edges[u.edge].curve) is Straight)
    assert "constant angular speed" in _face_refusal(Warped(g, arc_edge), face)
    assert "parametrized linearly" in _face_refusal(Warped(g, line_edge), face)


@pytest.mark.parametrize("order", ["linear", "quadratic"])
def test_nonuniform_stations_on_an_arc_face_are_refused(order, monkeypatch):
    g, t, old, args = request(order)
    original = mesh_to_dict(old)
    graded = lambda geometry, edge_id, divisions, size_field: (   # noqa: E731
        np.linspace(0.0, 1.0, divisions + 1)[1:-1] ** 1.5)
    monkeypatch.setattr(mapped, "edge_distribution", graded)
    with pytest.raises(MeshError, match="not uniform"):
        remesh_mapped_region(old, g, **args)
    assert mesh_to_dict(old) == original


def test_predicate_refuses_malformed_records_over_valid_owner_geometry():
    g, face, ids = sector(ro=3.0)
    assert mapped_regional_face_supported(g, face)
    tilted_axis = np.array(g.vertex_position(ids["via"])) + [0.0, 0.0, 0.1]
    lifted_corner = np.array(g.vertex_position(ids["q_b"])) + [0.0, 0.0, 1.0e-3]
    moved_end = np.array(g.vertex_position(ids["p_b"])) * 1.001
    for vertex, position in ((ids["via"], tilted_axis), (ids["q_b"], lifted_corner), (ids["p_b"], moved_end)):
        assert not mapped_regional_face_supported(Altered(g, moved={vertex: position}), face)


def test_predicate_keeps_straight_domain_and_rejects_unknown_ids():
    g = GeometryModel()
    p = g.add_points([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    f = g.add_face(g.add_polyline(p, close=True))
    assert mapped_regional_face_supported(g, f)
    assert not mapped_regional_face_supported(g, 999)
    assert not mapped_regional_face_supported(g, "x")


def test_predicate_refuses_nonplanar_and_conflicting_face_records():
    g, t = ogrid()
    lifted = Altered(g, moved={t["c"][1]: (B, -B, 0.02)})
    assert [mapped_regional_face_supported(lifted, f) for f in t["faces"]] == [False, False, True, True]
    g, t = ogrid()
    face = g.faces[t["faces"][0]]
    fid = face.id
    off = Plane(origin=(0.0, 0.0, 0.01), u_vector=(1.0, 0.0, 0.0), v_vector=(0.0, 1.0, 0.0))
    tilt = Plane(origin=(0.0, 0.0, 0.0), u_vector=(1.0, 0.0, 0.0), v_vector=(0.0, 1.0, 1.0))
    explicit = CoonsSurface(bottom=[[-B, -B, 0], [B, -B, 0]], right=[[B, -B, 0], [B, B, 0]],
                            top=[[-B, B, 0], [B, B, 0]], left=[[-B, -B, 0], [-B, B, 0]])
    for changed in (replace(face, surface=off), replace(face, surface=tilt), replace(face, surface=explicit),
                    replace(face, holes=(face.loop,)), replace(face, parameterization=PLANE),
                    replace(face, surface=None)):
        assert not mapped_regional_face_supported(Altered(g, **{f"f{fid}": changed}), fid)
    assert mapped_regional_face_supported(Altered(g, **{f"f{fid}": face}), fid)


@pytest.mark.parametrize("surface", [PLANE, DEFAULT])
@pytest.mark.parametrize("order", ["linear", "quadratic"])
def test_real_arc_patch_regeneration_keeps_interface_ids_and_unchanged_cells(order, surface):
    g, t, old, args = request(order, surface)
    original = mesh_to_dict(old)
    result = remesh_mapped_region(old, g, **args)
    assert mesh_to_dict(old) == original
    mesh, step = result.mesh, 2 if order == "quadratic" else 1
    assert result.stations_validated
    for e in (t["spokes"][0], t["spokes"][1], *t["sides"][1:], *t["arcs"][1:]):
        assert tuple(mesh.nodes_of_edge[e]) == tuple(old.nodes_of_edge[e])
    kept = sorted(n for f in t["faces"][1:] for n in old.elements_of_face[f])
    assert tuple(result.retained_element_ids) == tuple(kept)
    assert all(mesh.quads[e] == old.quads[e] for e in kept)
    assert all(np.array_equal(mesh.nodes[n], old.nodes[n]) for n in result.retained_node_ids)
    assert min(result.new_element_ids) > 2000 and min(result.new_node_ids) > 1000
    assert len(mesh.elements_of_face[t["faces"][0]]) == 24 and not mesh.tris
    # The regenerated hole edge keeps its distinct curved stations, on the true circle.
    stations = mesh.nodes_of_edge[t["arcs"][0]]
    assert len(stations) == 6*step + 1
    for n in stations:
        assert abs(float(np.linalg.norm(mesh.nodes[n][:2])) - R) < 1e-12 and mesh.nodes[n][2] == 0.0
    assert set(result.seeding_changed_edges) == {t["arcs"][0], t["sides"][0]}


def test_curved_edge_prior_stations_and_source_identity_are_validated():
    g, t, old, args = request("quadratic")
    original = mesh_to_dict(old)
    off = copy.deepcopy(old)
    n = off.nodes_of_edge[t["arcs"][1]][3]
    off.nodes[n] = np.asarray(off.nodes[n], dtype=float) + np.array([0.0, 0.0, 1.0e-3])
    with pytest.raises(MeshError, match="curved"):
        remesh_mapped_region(off, g, **args)
    swapped = copy.deepcopy(old)
    ids = list(swapped.nodes_of_edge[t["arcs"][2]])
    ids[2], ids[4] = ids[4], ids[2]
    swapped.nodes_of_edge[t["arcs"][2]] = ids
    with pytest.raises(MeshError):
        remesh_mapped_region(swapped, g, **args)
    other, _ = ogrid()
    with pytest.raises(MeshError, match="identity"):
        remesh_mapped_region(old, other, **args)
    assert mesh_to_dict(old) == original


def test_remesh_refuses_unqualified_faces_without_fallback():
    g, t = ogrid()
    pinned = {e: 4 for e in g.edges}
    old = mapped.generate_mesh(g, target_size=0.05, overrides=pinned)
    lifted = Altered(g, moved={t["c"][1]: (B, -B, 0.02)})
    with pytest.raises(MeshError, match="outside the supported"):
        remesh_mapped_region(
            old, lifted, affected_faces=(t["faces"][3],), interface_stations={}, interface_vertices={},
            seeding=solve_seeding(g, target_size=0.05, overrides=pinned), target_size=0.05,
            node_high_water=1000, element_high_water=2000,
            quality=RegionalQualityLimits(max_aspect_ratio=6., min_scaled_jacobian=0.2))
