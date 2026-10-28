import numpy as np
import pytest

from anymesher.floating_source_mesh import SourceMeshError, build_wet_source_mesh, choose_panel_size

BOTTOM = np.array([[0, 0, -1], [0, 4, -1], [8, 4, -1], [8, 0, -1]], float)      # outward -z (clockwise seen from above)


def plates():
    return [dict(label='bottom', polygon=BOTTOM)]


def test_given_size_is_recorded_and_area_is_preserved():
    out = build_wet_source_mesh(plates(), size=2.0, max_panels=200)
    area = sum(0.5 * np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0])) for t in out['triangles'])
    assert area == pytest.approx(32.0, rel=1e-12)
    assert out['provenance']['panel_size_m'] == 2.0 and not out['provenance']['size_chosen_automatically']


def test_automatic_size_is_deterministic_and_inside_budget():
    a = build_wet_source_mesh(plates(), max_panels=60)
    b = build_wet_source_mesh(plates(), max_panels=60)
    assert a['provenance'] == b['provenance'] and a['provenance']['panels'] <= 60
    assert choose_panel_size(plates(), 60) == a['provenance']['panel_size_m']


def _inclined_box_plates(heel=0.03, trim=-0.04, z0=-1.2, L=5.0, B=4.0, H=2.0):
    """Posed (heel+trim), waterplane-clipped plates of a box with OUTWARD loops, like the production plate source."""
    from anygeometry import floating_overlap as ov
    c = [(0, -B / 2, 0), (L, -B / 2, 0), (L, B / 2, 0), (0, B / 2, 0), (0, -B / 2, H), (L, -B / 2, H), (L, B / 2, H), (0, B / 2, H)]
    loops = {'bottom': (0, 3, 2, 1), 'port': (0, 1, 5, 4), 'starboard': (3, 7, 6, 2), 'end0': (0, 4, 7, 3), 'end1': (1, 2, 6, 5)}
    ch, sh, ct, st = np.cos(heel), np.sin(heel), np.cos(trim), np.sin(trim)
    Rx = np.array([[1, 0, 0], [0, ch, -sh], [0, sh, ch]])
    Ry = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    Q = Rx @ Ry
    plates = []
    for name, loop in loops.items():
        world = np.array([c[i] for i in loop], float) @ Q.T + np.array([0.0, 0.0, z0])
        wet = ov.wet_polygon(world, 0.0)
        assert wet is not None
        plates.append(dict(label=name, polygon=wet))
    return plates


def test_combined_heel_and_trim_source_is_closed_below_the_waterplane_with_conserved_area_and_normals():
    from anygeometry import floating_overlap as ov
    plates = _inclined_box_plates()
    out = build_wet_source_mesh(plates, size=0.55, max_panels=600, require_closed=True)   # raises if any edge below WL is open/non-manifold
    tris, labels, prov = out['triangles'], out['labels'], out['provenance']
    assert prov['panels'] == len(tris) <= 600 and prov['closed_hull_verified'] is True
    assert prov['conforming_removed_area_m2'] <= 1e-6 * prov['conforming_area_in_m2']     # only declared-roundoff slivers, with accounting
    assert prov['conforming_area_out_m2'] == pytest.approx(prov['conforming_area_in_m2'] - prov['conforming_removed_area_m2'], rel=1e-8)
    # independent adjacency check (declared tolerance): every edge used twice with opposite directions except free waterplane edges
    tol = prov['conforming_tolerance_m']
    key = lambda p: tuple(np.round(np.asarray(p) / tol).astype(np.int64))
    uses = {}
    for t in tris:
        k = [key(p) for p in t]
        for a, b in ((k[0], k[1]), (k[1], k[2]), (k[2], k[0])):
            uses.setdefault(frozenset((a, b)), []).append((a, b))
    free = [pair for pair, u in uses.items() if len(u) == 1]
    assert all(all(v[2] == 0 for v in pair) for pair in free)                    # only the waterline boundary is free
    assert all(len(u) == 1 or (len(u) == 2 and u[0] == u[1][::-1]) for u in uses.values())
    # area and outward normals conserved per physical plate
    for plate in plates:
        mine = [t for t, lab in zip(tris, labels) if lab == plate['label']]
        assert sum(ov.polygon_area(t) for t in mine) == pytest.approx(ov.polygon_area(plate['polygon']), rel=1e-9)
        n_plate = ov.polygon_plane(plate['polygon'])[0]
        assert all(ov.polygon_plane(t)[0] @ n_plate > 1.0 - 1e-9 for t in mine)
    assert all(abs(p[2]) == 0.0 or p[2] < 0.0 for t in tris for p in t)           # exact waterline, nothing above it


def test_translated_hull_gives_the_same_tolerance_and_panels_origin_covariance():
    """The declared merging tolerance is scaled by the hull's own extent, never by its absolute survey position."""
    plates = _inclined_box_plates()
    shift = np.array([1.0e6, -3.0e5, 0.0])                                                  # horizontal translation only (waterplane z=0 fixed)
    from anymesher.floating_source_mesh import conform_triangles
    # Same panels, translated: isolates the conformer from the (origin-independent but discontinuous) grid count of the panelisation.
    base_tris = build_wet_source_mesh(plates, size=0.7, max_panels=600)['triangles']
    labels = ['x'] * len(base_tris)
    a_tris, _, a = conform_triangles(base_tris, labels)
    b_tris, _, b = conform_triangles([t + shift for t in base_tris], labels)
    assert b['tolerance'] == pytest.approx(a['tolerance'], rel=1e-8)                    # scaled by the hull's own extent, not x = 1e6
    assert len(a_tris) == len(b_tris) and a['edge_splits'] == b['edge_splits'] and a['removed_panels'] == b['removed_panels']
    assert b['area_in'] == pytest.approx(a['area_in'], rel=1e-9)


def test_conforming_refinement_resolves_a_t_junction_with_nondegenerate_panels_and_counts_toward_the_budget():
    from anymesher.floating_source_mesh import conform_triangles
    # two coplanar-edge neighbours on different plates: one edge split at its midpoint by the other plate's vertex
    big = np.array([[0, 0, -2], [2, 0, -2], [0, 0, 0]], float)                   # plate A edge (0,0,-2)-(2,0,-2)... along x
    left = np.array([[0, 0, -2], [1, -1, -2], [0, -1, -2]], float)
    right = np.array([[1, 0, -2], [2, 0, -2], [1, -1, -2]], float)
    tris, labels, stats = conform_triangles([big, left, right], ['a', 'b', 'c'])
    assert stats['edge_splits'] >= 1 and len(tris) > 3
    from anygeometry import floating_overlap as ov
    assert all(ov.polygon_area(t) > 0 for t in tris)                             # no degenerate panel from collinear nodes
    assert sum(ov.polygon_area(t) for t in tris) == pytest.approx(sum(ov.polygon_area(t) for t in (big, left, right)))
    with pytest.raises(SourceMeshError, match='outside the budget'):
        build_wet_source_mesh(_inclined_box_plates(), size=0.55, max_panels=10)  # the refined count is what the budget judges


def test_size_outside_budget_is_refused_not_coarsened():
    with pytest.raises(SourceMeshError):
        build_wet_source_mesh(plates(), size=0.1, max_panels=50)
    with pytest.raises(SourceMeshError):
        build_wet_source_mesh([], size=1.0)
