"""Regional patch producer: planar end-to-end fixtures and typed refusals.

``lattice_kernel`` is an explicitly INJECTED stand-in kernel (reproduces the fixed
stations exactly).  Tests using it are named ``*_injected_kernel``; the
``*_owner_kernel`` tests exercise the real native planar mesher.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from anymesher.mesh import Mesh
from anymesher.regional_patch import (
    KernelOutput,
    NewDomain,
    PatchLoop,
    PatchStation,
    PatchTolerances,
    RegionalPatchRefusal,
    RegionalPatchRequest,
    RegionalPatchResult,
    _orient,
    _overlaps,
    remesh_region,
    try_remesh_region,
    verify_regional_patch,
)
from anymesher.regional_update import RegionalQualityLimits

N = 6
LIMITS = RegionalQualityLimits(max_aspect_ratio=10.0, max_warp=1.0, min_scaled_jacobian=0.1)
TOL = PatchTolerances(planarity=1e-9)


def nid(i, j, n=N):
    return j * (n + 1) + i + 1


def eid(i, j, n=N):
    return j * n + i + 1


def grid_mesh(quadratic=False, n=N):
    mesh = Mesh(geometry_model_id="model-1", geometry_revision=1)
    for j in range(n + 1):
        for i in range(n + 1):
            mesh.nodes[nid(i, j, n)] = np.array([float(i), float(j), 0.0])
    mids = {}
    next_id = (n + 1) ** 2 + 1
    for j in range(n):
        for i in range(n):
            conn = (nid(i, j, n), nid(i + 1, j, n), nid(i + 1, j + 1, n), nid(i, j + 1, n))
            if quadratic:
                extra = []
                for k in range(4):
                    a, b = conn[k], conn[(k + 1) % 4]
                    key = (min(a, b), max(a, b))
                    if key not in mids:
                        mids[key] = next_id
                        mesh.nodes[next_id] = 0.5 * (mesh.nodes[a] + mesh.nodes[b])
                        next_id += 1
                    extra.append(mids[key])
                conn = conn + tuple(extra)
            mesh.quads[eid(i, j, n)] = conn
    mesh.elements_of_face[1] = [eid(i, j, n) for j in range(n) for i in range(n)]
    mesh.node_of_vertex[1] = nid(0, 0, n)
    mesh.node_of_vertex[2] = nid(n, 0, n)
    if quadratic:
        mesh.order = "quadratic"
    return mesh


def edge_mesh():
    """Grid whose bottom row is geometry edge 1 (nodes + elements)."""

    mesh = grid_mesh()
    mesh.nodes_of_edge[1] = [nid(i, 0) for i in range(N + 1)]
    mesh.elements_of_edge[1] = [eid(i, 0) for i in range(N)]
    return mesh


def _inside(p, poly):
    inside = False
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        if (a[1] > p[1]) != (b[1] > p[1]):
            if a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]) > p[0]:
                inside = not inside
    return inside


def lattice_kernel(kin):
    loops = [kin.outer, *kin.holes]
    xs = sorted({float(p[0]) for lp in loops for p in lp})
    ys = sorted({float(p[1]) for lp in loops for p in lp})
    cells = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            c = ((xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2)
            if _inside(c, kin.outer) and not any(_inside(c, h) for h in kin.holes):
                cells.append(((xs[i], ys[j]), (xs[i + 1], ys[j]), (xs[i + 1], ys[j + 1]), (xs[i], ys[j + 1])))
    pts = sorted({p for c in cells for p in c})
    row = {p: k for k, p in enumerate(pts)}
    return KernelOutput(np.array(pts), tuple(tuple(row[p] for p in c) for c in cells))


def split_to_tris(kin):
    out = lattice_kernel(kin)
    tris = []
    for a, b, c, d in out.quads:
        tris += [(a, b, c), (a, c, d)]
    return KernelOutput(out.points, (), tuple(tris))


def block(i0, i1, j0, j1):
    return [eid(i, j) for j in range(j0, j1) for i in range(i0, i1)]


def req(mesh, **kw):
    """Request with every mandatory input present (kernel injected unless overridden)."""

    base = dict(old_mesh=mesh, face_id=1, tolerances=TOL, quality=LIMITS,
                source_check=lambda o, n, r: True)
    if "kernel" not in kw:
        base["kernel"] = lattice_kernel
    base.update(kw)
    return RegionalPatchRequest(**base)


def full_snapshot(mesh):
    return copy.deepcopy(mesh), {k: v.copy() for k, v in mesh.nodes.items()}


def assert_unchanged(mesh, snap):
    ref, nodes = snap
    assert mesh.quads == ref.quads and mesh.tris == ref.tris
    assert mesh.elements_of_face == ref.elements_of_face and mesh.nodes_of_edge == ref.nodes_of_edge
    assert mesh.node_of_vertex == ref.node_of_vertex and mesh.activity == ref.activity
    assert set(mesh.nodes) == set(nodes) and all(np.array_equal(mesh.nodes[k], nodes[k]) for k in nodes)


def assert_retained_exact(old, new, replaced):
    for e, conn in old.quads.items():
        if e not in replaced:
            assert new.quads[e] == conn
            for n in conn:
                assert np.array_equal(old.nodes[n], new.nodes[n])


# ------------------------------------------------------------- regression


def test_orient_accepts_numpy_floats_and_overlap_predicates_work():
    a, b, c = (np.float64(0), np.float64(0)), (np.float64(1), np.float64(0)), (np.float64(0), np.float64(1))
    assert _orient(a, b, c) == 1 and _orient(a, c, b) == -1 and _orient(a, b, (2.0, 0.0)) == 0
    sq = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    far = np.array([[3.0, 0.0], [4.0, 0.0], [4.0, 1.0], [3.0, 1.0]])
    touch = np.array([[2.0, 0.0], [3.0, 0.0], [3.0, 2.0], [2.0, 2.0]])
    inner = np.array([[0.5, 0.5], [1.0, 0.5], [1.0, 1.0], [0.5, 1.0]])
    assert not _overlaps(sq, far) and not _overlaps(sq, touch) and _overlaps(sq, inner)


# ---------------------------------------------------- injected-kernel tests


def test_same_domain_resize_injected_kernel():
    old = grid_mesh()
    snap = full_snapshot(old)
    region = block(2, 4, 2, 4)
    result = remesh_region(req(old, replace_element_ids=region))
    assert isinstance(result, RegionalPatchResult)
    rep = result.report
    assert rep.status == "accepted" and rep.all_quad and rep.growth_rings == 0
    assert rep.route == "planar_injected_kernel" and rep.kernel_injected
    assert set(rep.replaced_element_ids) == set(region) and len(rep.new_element_ids) == 4
    assert "interior_overlap_hole_and_area_coverage" in rep.checks
    assert rep.quality is not None
    assert_retained_exact(old, result.mesh, set(region))
    assert len(rep.new_node_ids) == 1 and len(rep.removed_node_ids) == 1
    assert set(rep.retained_element_ids) == set(old.quads) - set(region)
    assert len(rep.interface_node_ids) == 8
    assert_unchanged(old, snap)  # old mesh not mutated
    verify_regional_patch(old, result.mesh, 1, rep.replaced_element_ids, rep.new_element_ids)


def test_monotonic_ids_high_water_floor_and_int64_exhaustion():
    old = grid_mesh()
    region = block(2, 4, 2, 4)
    r = remesh_region(req(old, replace_element_ids=region, node_high_water=500, element_high_water=900))
    assert min(r.report.new_node_ids) == 501 and min(r.report.new_element_ids) == 901
    assert r.report.node_high_water >= 501 and r.report.element_high_water == 904
    f = remesh_region(req(old, replace_element_ids=region, node_id_floor=700, element_id_floor=800))
    assert min(f.report.new_node_ids) == 700 and min(f.report.new_element_ids) == 800
    for kw in ({"node_high_water": 2**63 - 1}, {"element_high_water": 2**63 - 1}):
        out = try_remesh_region(req(old, replace_element_ids=region, **kw))
        assert isinstance(out, RegionalPatchRefusal) and out.code == "ID_EXHAUSTED"
    bad = try_remesh_region(req(old, replace_element_ids=region, node_high_water=-1))
    assert bad.code == "BAD_ID_FLOOR"
    bad = try_remesh_region(req(old, replace_element_ids=region, element_high_water=True))
    assert bad.code == "BAD_ID_FLOOR"


def test_footprint_selects_only_overlapped_complete_cells():
    footprint = [(2.25, 2.25, 0.0), (3.75, 2.25, 0.0), (3.75, 3.75, 0.0), (2.25, 3.75, 0.0)]
    result = remesh_region(req(grid_mesh(), footprints=[footprint]))
    assert set(result.report.replaced_element_ids) == set(block(2, 4, 2, 4))


def test_associations_and_activity_are_rebuilt_on_candidate():
    old = edge_mesh()
    region = block(1, 3, 0, 2)
    old.elements_of_sheet[7] = list(region) + [eid(5, 5)]
    old.elements_of_member[8] = [eid(5, 5)]
    for e in region:
        old.activity[e] = 0.25
    old.activity[eid(5, 5)] = 1.0
    result = remesh_region(req(old, replace_element_ids=region))
    new, rep = result.mesh, result.report
    news = set(rep.new_element_ids)
    assert len(news) == 4
    assert set(new.elements_of_sheet[7]) == (set(old.elements_of_sheet[7]) - set(region)) | news
    assert new.elements_of_member[8] == [eid(5, 5)]
    assert all(new.activity[e] == 0.25 for e in news) and not set(region) & set(new.activity)
    assert new.activity[eid(5, 5)] == 1.0
    # bottom-row geometry edge: replaced 2 elements, rebuilt 2, node list substituted
    assert len([e for e in new.elements_of_edge[1] if e in news]) == 2
    assert set(new.elements_of_edge[1]) == (set(old.elements_of_edge[1]) - set(region)) | {
        e for e in new.elements_of_edge[1] if e in news}
    renumbered = dict(rep.renumbered_free_nodes)
    assert renumbered and nid(2, 0) in renumbered
    assert new.nodes_of_edge[1] == [renumbered.get(n, n) for n in old.nodes_of_edge[1]]
    assert np.array_equal(new.nodes[renumbered[nid(2, 0)]], old.nodes[nid(2, 0)])
    assert new.node_of_vertex == old.node_of_vertex


def test_mixed_activity_and_mixed_sheet_are_refused():
    old = grid_mesh()
    region = block(2, 4, 2, 4)
    old.activity[region[0]] = 0.5
    assert try_remesh_region(req(old, replace_element_ids=region)).code == "MIXED_ACTIVITY"
    old = grid_mesh()
    old.elements_of_sheet[7] = [region[0]]
    assert try_remesh_region(req(old, replace_element_ids=region)).code == "MIXED_ASSOCIATION"


def test_hole_punch_requires_new_edge_and_removal_requires_owner_retire():
    old = grid_mesh()
    region = block(1, 5, 1, 5)
    hole = PatchLoop(tuple(PatchStation((float(x), float(y), 0.0)) for x, y in
                           [(2, 2), (3, 2), (4, 2), (4, 3), (4, 4), (3, 4), (2, 4), (2, 3)]))
    out = try_remesh_region(req(old, replace_element_ids=region, added_holes=[hole]))
    assert out.code == "NEW_HOLE_ASSOCIATIONS_REQUIRED"
    punched = remesh_region(req(old, replace_element_ids=region, added_holes=[hole], added_hole_edge_ids=[100]))
    mesh = punched.mesh
    assert len(punched.report.new_element_ids) == 12
    assert len(mesh.elements_of_face[1]) == len(old.elements_of_face[1]) - 4
    assert len(mesh.nodes_of_edge[100]) == 8 and len(mesh.elements_of_edge[100]) == 8
    assert tuple(mesh.nodes_of_edge[100]) == punched.report.new_hole_edges[100]
    assert not any(np.array_equal(p, np.array([3.0, 3.0, 0.0])) for p in mesh.nodes.values())
    assert_retained_exact(old, mesh, set(region))
    hole_node = mesh.nodes_of_edge[100][0]
    # removal without the owner retire of edge 100 is refused (no stale map)
    out = try_remesh_region(req(mesh, replace_element_ids=punched.report.new_element_ids,
                                drop_hole_node_ids=[hole_node]))
    assert out.code == "ASSOCIATION_UPDATE_REQUIRED"
    filled = remesh_region(req(mesh, replace_element_ids=punched.report.new_element_ids,
                               drop_hole_node_ids=[hole_node], retired_geometry_keys={"edge": [100]},
                               node_high_water=punched.report.node_high_water,
                               element_high_water=punched.report.element_high_water))
    assert len(filled.report.new_element_ids) == 16 and 100 not in filled.mesh.nodes_of_edge
    assert len(filled.mesh.elements_of_face[1]) == len(old.elements_of_face[1])
    assert hole_node not in filled.mesh.nodes
    # ids never reused after deletion
    assert min(filled.report.new_element_ids) > max(punched.report.new_element_ids)
    assert min(filled.report.new_node_ids) > punched.report.node_high_water


def test_trim_dropping_associated_node_is_refused_not_published_with_missing_maps():
    old = grid_mesh()

    def st(i, j, node=None):
        return PatchStation((float(i), float(j), 0.0), node)

    outer = PatchLoop((st(1, 0, nid(1, 0)), st(2, 0, nid(2, 0)), st(2, 1, nid(2, 1)), st(2, 2, nid(2, 2)),
                       st(1, 2, nid(1, 2)), st(0, 2, nid(0, 2)), st(0, 1, nid(0, 1)), st(1, 1)))
    out = try_remesh_region(req(old, replace_element_ids=block(0, 2, 0, 2), new_domain=NewDomain(outer)))
    assert isinstance(out, RegionalPatchRefusal) and out.code == "ASSOCIATION_UPDATE_REQUIRED"


def test_trim_cutting_through_retained_interface_is_refused():
    outer = PatchLoop(tuple(PatchStation((float(x), float(y), 0.0), nid(x, y)) for x, y in
                            [(2, 2), (3, 2), (3, 3), (2, 3)]))
    out = try_remesh_region(req(grid_mesh(), replace_element_ids=block(2, 4, 2, 4), new_domain=NewDomain(outer)))
    assert out.code in {"INTERFACE_NOT_PRESERVED", "INTERFACE_MISMATCH"}


def test_domain_overlapping_retained_cells_is_refused():
    outer = PatchLoop(tuple(PatchStation((float(x), float(y), 0.0)) for x, y in
                            [(2, 2), (3, 2), (4, 2), (5, 2), (5, 3), (5, 4), (4, 4), (3, 4), (2, 4), (2, 3)]))
    out = try_remesh_region(req(grid_mesh(), replace_element_ids=block(2, 4, 2, 4), new_domain=NewDomain(outer)))
    assert isinstance(out, RegionalPatchRefusal)
    assert out.code in {"PATCH_OVERLAPS_RETAINED", "INTERFACE_MISMATCH", "INTERFACE_NOT_PRESERVED",
                        "STATION_NOT_ON_REGION_BOUNDARY"}


def test_quadratic_patch_keeps_interface_midsides_and_order_injected_kernel():
    old = grid_mesh(quadratic=True)
    region = block(2, 4, 2, 4)
    result = remesh_region(req(old, replace_element_ids=region))
    new = result.mesh
    assert result.report.order == "quadratic"
    assert all(len(new.quads[e]) == 8 for e in result.report.new_element_ids)
    neighbour = new.quads[eid(1, 2)]
    assert neighbour == old.quads[eid(1, 2)]
    assert neighbour[5] in {m for e in result.report.new_element_ids for m in new.quads[e][4:]}
    verify_regional_patch(old, new, 1, result.report.replaced_element_ids, result.report.new_element_ids)


def test_odd_boundary_parity_is_refused_without_growth():
    tri = PatchLoop(tuple(PatchStation(p) for p in [(2.25, 2.25, 0.0), (2.75, 2.25, 0.0), (2.5, 2.75, 0.0)]))
    out = try_remesh_region(req(grid_mesh(), replace_element_ids=block(2, 4, 2, 4), added_holes=[tri],
                                added_hole_edge_ids=[100], max_growth_rings=0))
    assert out.code == "ODD_BOUNDARY_PARITY"


def test_kernel_failures_quality_and_owner_failures_are_not_retried_as_growth():
    def moving(kin):
        out = lattice_kernel(kin)
        pts = out.points.copy()
        pts[0, 0] += 0.125
        return KernelOutput(pts, out.quads)

    old, region = grid_mesh(), block(2, 4, 2, 4)
    out = try_remesh_region(req(old, replace_element_ids=region, kernel=moving))
    assert out.code == "KERNEL_MOVED_FIXED_STATION" and len(out.details["attempts"]) == 1

    def boom(kin):
        raise RuntimeError("x")

    out = try_remesh_region(req(old, replace_element_ids=region, kernel=boom))
    assert out.code == "KERNEL_FAILED" and len(out.details["attempts"]) == 1
    strict = RegionalQualityLimits(max_aspect_ratio=10.0, max_warp=1.0, min_scaled_jacobian=1.5)
    out = try_remesh_region(req(old, replace_element_ids=region, quality=strict))
    assert out.code == "QUALITY_REJECTED" and len(out.details["attempts"]) == 1


def test_triangles_refused_for_all_quad_and_allowed_when_not_required():
    kw = dict(replace_element_ids=block(2, 4, 2, 4), kernel=split_to_tris)
    out = try_remesh_region(req(grid_mesh(), max_growth_rings=0, **kw))
    assert out.code == "NON_QUAD_PATCH"
    result = remesh_region(req(grid_mesh(), require_all_quad=False, **kw))
    assert result.report.all_quad is False and len(result.mesh.tris) == 8


def test_hooks_can_refuse_and_mutation_is_absent_on_refusal():
    old = grid_mesh()
    snap = full_snapshot(old)
    kw = dict(replace_element_ids=block(2, 4, 2, 4))
    assert try_remesh_region(req(old, quality_gate=lambda m, ids: False, max_growth_rings=0, **kw)).code == "QUALITY_REJECTED"
    assert try_remesh_region(req(old, source_check=lambda o, n, r: False, **kw)).code == "SOURCE_CORRESPONDENCE_REJECTED"
    assert_unchanged(old, snap)


# -------------------------------------------------------- mandatory inputs


def test_incomplete_inputs_never_accepted():
    old, kw = grid_mesh(), dict(replace_element_ids=block(2, 4, 2, 4))
    assert try_remesh_region(req(old, quality=None, **kw)).code == "QUALITY_LIMITS_REQUIRED"
    assert try_remesh_region(req(old, source_check=None, **kw)).code == "SOURCE_PROOF_REQUIRED"
    assert try_remesh_region(req(old, tolerances=PatchTolerances(), **kw)).code == "PLANARITY_TOLERANCE_REQUIRED"
    nogeo = grid_mesh()
    nogeo.geometry_revision = None
    assert try_remesh_region(req(nogeo, **kw)).code == "GEOMETRY_PROVENANCE_MISSING"
    assert try_remesh_region(req(old, replace_element_ids=[])).code == "EMPTY_REGION"


def test_nonplanar_face_claimed_plane_is_refused():
    old = grid_mesh()
    old.nodes[nid(5, 5)] = np.array([5.0, 5.0, 0.5])
    out = try_remesh_region(req(old, replace_element_ids=block(2, 4, 2, 4)))
    assert out.code == "NOT_PLANAR"


def test_unknown_diagnostics_and_mapped_grid_are_handled_explicitly():
    old = grid_mesh()
    old.hybrid_diagnostics = {"mystery_ids": [1, 2, 3]}
    kw = dict(replace_element_ids=block(2, 4, 2, 4))
    out = try_remesh_region(req(old, discard_stale_diagnostics=True, **kw))
    assert out.code == "INVALID_MESH" or out.code == "METADATA_REFUSED"
    old = grid_mesh()
    old.hybrid_diagnostics = {"certifiable": True}
    assert try_remesh_region(req(old, **kw)).code == "METADATA_REFUSED"
    ok = remesh_region(req(old, discard_stale_diagnostics=True, **kw))
    assert ok.mesh.hybrid_diagnostics == {} and ok.report.discarded_diagnostic_keys
    old = grid_mesh()
    old.grid_of_face[1] = np.arange(1, 50).reshape(7, 7)
    assert try_remesh_region(req(old, **kw)).code == "MAPPED_GRID_REMOVAL_REQUIRED"
    gone = remesh_region(req(old, remove_mapped_grid=True, **kw))
    assert 1 not in gone.mesh.grid_of_face and gone.report.removed_grids == (1,)
    assert 1 in old.grid_of_face


def test_curved_unknown_and_held_cases_fail_closed_with_typed_reasons():
    kw = dict(replace_element_ids=block(2, 4, 2, 4))
    out = try_remesh_region(req(grid_mesh(), surface_kind="cylinder", **kw))
    assert out.code == "CURVED_FAMILY_UNIMPLEMENTED" and out.details["family"] == "cylinder"
    assert try_remesh_region(req(grid_mesh(), surface_kind="blob", **kw)).code == "UNKNOWN_SURFACE_KIND"
    for held in ("two_split_quarter_cylinders", "principal_dome_cut_nurbs", "four_face_spline"):
        assert try_remesh_region(req(grid_mesh(), held_case=held, **kw)).code == "OWNER_HOLD"


# ----------------------------------------------------------- real kernel


def test_owner_planar_kernel_end_to_end_positive():
    """Real native planar mesher (no injected kernel); acceptance is required."""

    old = grid_mesh()
    region = block(1, 5, 1, 5)
    result = remesh_region(req(old, replace_element_ids=region, target_size=1.0, kernel=None))
    rep = result.report
    assert rep.route == "planar_owner_kernel" and not rep.kernel_injected
    assert rep.status == "accepted" and rep.all_quad
    assert_retained_exact(old, result.mesh, set(region))
    verify_regional_patch(old, result.mesh, 1, rep.replaced_element_ids, rep.new_element_ids)
    assert len(rep.interface_node_ids) == 16


@pytest.mark.parametrize("quadratic", [False, True])
def test_owner_refines_core_at_requested_size_and_keeps_fixed_stations(quadratic):
    old = grid_mesh(quadratic=quadratic, n=8)
    zone = [(3.75, 3.75, 0.), (4.25, 3.75, 0.), (4.25, 4.25, 0.), (3.75, 4.25, 0.)]
    result = remesh_region(req(old, footprints=[zone], target_size=.5,
                               sizing_zones=[(zone, .5)], kernel=None))
    new, report = result.mesh, result.report
    assert set(report.replaced_element_ids) == {eid(i, j, n=8) for i in (3, 4) for j in (3, 4)}
    assert len(report.new_element_ids) == 12
    assert "new_quad_edge_target_in_each_sizing_zone" in report.checks
    assert_retained_exact(old, new, set(report.replaced_element_ids))
    for node in report.interface_node_ids:
        np.testing.assert_array_equal(new.nodes[node], old.nodes[node])
    fine = []
    for e in report.new_element_ids:
        pts = np.array([new.nodes[n] for n in new.quads[e][:4]])
        assert len(new.quads[e]) == (8 if quadratic else 4)
        if max(np.linalg.norm(p - q) for p, q in zip(pts, np.roll(pts, -1, axis=0))) <= .5:
            fine.append(pts)
    # Four core quads meet at the center of the requested physical zone.
    assert len(fine) == 4 and all(np.any(np.all(pts == [4., 4., 0.], axis=1)) for pts in fine)


@pytest.mark.parametrize("bad", [
    ([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.)], 0.),
    ([(0., 0., 0.), (1., 0., .1), (0., 1., 0.)], .5),
    ([(0., 0., 0.), (1., 1., 0.), (1., 0., 0.), (0., 1., 0.)], .5),
])
def test_bad_sizing_zones_refuse_before_kernel(bad):
    old = grid_mesh()
    with pytest.raises(RegionalPatchRefusal) as caught:
        remesh_region(req(old, replace_element_ids=block(2, 4, 2, 4), sizing_zones=[bad],
                          kernel=lambda _: pytest.fail("kernel ran on invalid sizing data")))
    assert caught.value.code == "BAD_SIZING_ZONE"


def test_unsized_injected_patch_and_zone_outside_selected_cells_are_refused():
    old = grid_mesh()
    before = full_snapshot(old)
    inside = [(2.75, 2.75, 0.), (3.25, 2.75, 0.), (3.25, 3.25, 0.), (2.75, 3.25, 0.)]
    outside = [(x-2., y-2., z) for x, y, z in inside]
    for zone, kernel in ((inside, lattice_kernel), (outside, None)):
        with pytest.raises(RegionalPatchRefusal) as caught:
            remesh_region(req(old, replace_element_ids=block(2, 4, 2, 4), target_size=.5,
                              sizing_zones=[(zone, .5)], kernel=kernel))
        assert caught.value.code == "LOCAL_SIZE_UNMET"
        assert_unchanged(old, before)


def _hole_loop():
    return PatchLoop(tuple(PatchStation((float(x), float(y), 0.0)) for x, y in
                           [(2, 2), (3, 2), (4, 2), (4, 3), (4, 4), (3, 4), (2, 4), (2, 3)]))


@pytest.mark.parametrize("size", [None, 1.0, 0.5, 0.25])
def test_owner_kernel_rectangular_hole_all_quad_positive_no_injection(size):
    old = grid_mesh()
    region = block(1, 5, 1, 5)
    result = remesh_region(req(old, replace_element_ids=region, added_holes=[_hole_loop()],
                               added_hole_edge_ids=[100], target_size=size, kernel=None))
    rep, mesh = result.report, result.mesh
    assert rep.route == "planar_owner_kernel" and not rep.kernel_injected
    assert rep.status == "accepted" and rep.all_quad and not mesh.tris
    assert len(rep.new_element_ids) == 12 and len(rep.interface_node_ids) == 16
    assert len(mesh.nodes_of_edge[100]) == 8 and len(mesh.elements_of_edge[100]) == 8
    assert not any(np.array_equal(p, np.array([3.0, 3.0, 0.0])) for p in mesh.nodes.values())
    assert_retained_exact(old, mesh, set(region))
    verify_regional_patch(old, mesh, 1, rep.replaced_element_ids, rep.new_element_ids)


def _ring(x0, x1, y0, y1):
    pts = [(x, y0) for x in range(x0, x1)] + [(x1, y) for y in range(y0, y1)]
    pts += [(x, y1) for x in range(x1, x0, -1)] + [(x0, y) for y in range(y1, y0, -1)]
    return np.array(pts, dtype=float)


def test_aligned_station_grid_two_holes_and_rejections():
    from anymesher.regional_patch import KernelInput, _aligned_station_grid, owner_planar_kernel

    outer = _ring(0, 8, 0, 8)
    a, b = _ring(2, 3, 2, 3)[::-1], _ring(5, 7, 5, 7)[::-1]
    out = owner_planar_kernel(KernelInput(outer, (a, b), 0.25))
    assert not out.tris and len(out.quads) == 64 - 1 - 4
    assert len(out.points) == 81 - 1  # only the 2x2 hole's centre station is dropped
    # nonaligned / unsupported configurations are not claimed by the grid path
    shifted = _ring(2, 3, 2, 3)[::-1] + np.array([0.5, 0.0])
    assert _aligned_station_grid(KernelInput(outer, (shifted,), 1.0)) is None
    assert _aligned_station_grid(KernelInput(outer, (_ring(2, 3, 2, 3),), 1.0)) is None  # wrong winding
    touching = _ring(0, 2, 2, 4)[::-1]
    assert _aligned_station_grid(KernelInput(outer, (touching,), 1.0)) is None
    adjacent = _ring(3, 4, 2, 3)[::-1]
    assert _aligned_station_grid(KernelInput(outer, (a, adjacent), 1.0)) is None
    missing = np.delete(_ring(2, 4, 2, 4), 1, axis=0)[::-1]  # perimeter station absent
    assert _aligned_station_grid(KernelInput(outer, (missing,), 1.0)) is None
    skew = outer.copy()
    skew[3] += np.array([0.0, 0.25])
    assert _aligned_station_grid(KernelInput(skew, (), 1.0)) is None
