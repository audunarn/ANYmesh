"""Submodel point correspondence (docs/SUBMODEL_CORRESPONDENCE_CONTRACT.md), planar S3a."""

from __future__ import annotations

import numpy as np
import pytest

from anymesher.mesh import Mesh
from anymesher.submodel_correspondence import CorrespondenceError, locate_for_transfer


def grid_mesh(n=2, *, split_faces=True, triangles=False):
    """Plate [0,1]^2, n x n cells; with split_faces, columns i < n/2 are face 1, others face 2."""

    nodes, quads, tris, of_face = {}, {}, {}, {1: [], 2: []}
    ids = {}
    k = 1
    for j in range(n + 1):
        for i in range(n + 1):
            nodes[k] = np.array([i / n, j / n, 0.0])
            ids[i, j] = k
            k += 1
    e = 1
    for j in range(n):
        for i in range(n):
            a, b, c, d = ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1]
            face = 1 if (not split_faces or i < n // 2) else 2
            if triangles:
                for conn in ((a, b, c), (a, c, d)):
                    tris[e] = conn
                    of_face[face].append(e)
                    e += 1
            else:
                quads[e] = (a, b, c, d)
                of_face[face].append(e)
                e += 1
    return Mesh(nodes=nodes, quads=quads, tris=tris, elements_of_face={f: v for f, v in of_face.items() if v})


def everything(mesh, keys):
    return {key: tuple(mesh.shells) for key in keys}


def linear(x):
    return 2.0 * x[0] - 3.0 * x[1] + 0.5


@pytest.mark.parametrize("triangles", [False, True])
def test_interior_points_and_linear_reproduction(triangles):
    mesh = grid_mesh(4, triangles=triangles)
    rng = np.random.default_rng(3)
    points = {100 + k: (*rng.uniform(0.01, 0.99, size=2), 0.0) for k in range(25)}
    hits = locate_for_transfer(mesh, points, candidate_elements=everything(mesh, points))
    for key, hit in hits.items():
        assert hit.source_element_type == ("T3" if triangles else "Q4")
        value = sum(w * linear(mesh.nodes[n]) for w, n in zip(hit.weights, hit.source_node_ids))
        assert value == pytest.approx(linear(np.asarray(points[key])), abs=1e-14)
        assert hit.distance <= 1e-15 and hit.conforming


def test_shared_edge_and_vertex_are_conforming_and_report_both_faces():
    mesh = grid_mesh(2)  # face 1 = column 0, face 2 = column 1; shared line x = 0.5
    points = {1: (0.5, 0.25, 0.0), 2: (0.5, 0.5, 0.0), 3: (0.25, 0.25, 0.0)}
    hits = locate_for_transfer(mesh, points, candidate_elements=everything(mesh, points))
    assert hits[1].incident_source_faces == (1, 2) and hits[1].conforming
    assert hits[2].incident_source_faces == (1, 2) and hits[2].conforming
    assert hits[3].incident_source_faces == (1,)
    # On the shared vertex only that node carries weight.
    weighted = [n for n, w in zip(hits[2].source_node_ids, hits[2].weights) if abs(w) > 1e-12]
    assert len(weighted) == 1 and np.allclose(mesh.nodes[weighted[0]], (0.5, 0.5, 0.0))


def coincident_sheets():
    """Two unconnected unit squares at the same place (duplicated nodes)."""

    nodes = {1: (0, 0, 0), 2: (1, 0, 0), 3: (1, 1, 0), 4: (0, 1, 0),
             5: (0, 0, 0), 6: (1, 0, 0), 7: (1, 1, 0), 8: (0, 1, 0)}
    return Mesh(nodes={k: np.asarray(v, float) for k, v in nodes.items()},
                quads={1: (1, 2, 3, 4), 2: (5, 6, 7, 8)}, elements_of_face={1: [1], 2: [2]})


def test_disconnected_coincident_sheets_refuse_unless_lineage_selects_one():
    mesh = coincident_sheets()
    points = {9: (0.4, 0.6, 0.0)}
    with pytest.raises(CorrespondenceError, match="CORRESPONDENCE_NONCONFORMING"):
        locate_for_transfer(mesh, points, candidate_elements={9: (1, 2)})
    hit = locate_for_transfer(mesh, points, candidate_elements={9: (2,)})[9]
    assert hit.source_element_id == 2 and hit.incident_source_faces == (2,)


def test_lineage_excludes_elements_and_unlocated_points_refuse():
    mesh = grid_mesh(2)
    face1 = tuple(mesh.elements_of_face[1])
    with pytest.raises(CorrespondenceError, match="CORRESPONDENCE_NOT_LOCATED"):
        locate_for_transfer(mesh, {5: (0.75, 0.25, 0.0)}, candidate_elements={5: face1})
    with pytest.raises(CorrespondenceError, match="CORRESPONDENCE_NO_LINEAGE"):
        locate_for_transfer(mesh, {5: (0.25, 0.25, 0.0)}, candidate_elements={})


def test_physical_tolerance_is_separate_from_natural_slack():
    mesh = grid_mesh(2)
    near = {7: (0.3, 0.3, 1.0e-12)}
    hit = locate_for_transfer(mesh, near, candidate_elements=everything(mesh, near))[7]
    assert hit.distance == pytest.approx(1.0e-12, rel=1e-6)
    far = {7: (0.3, 0.3, 1.0e-6)}
    with pytest.raises(CorrespondenceError, match="CORRESPONDENCE_NOT_LOCATED"):
        locate_for_transfer(mesh, far, candidate_elements=everything(mesh, far))
    hit = locate_for_transfer(mesh, far, candidate_elements=everything(mesh, far),
                              physical_tolerance=1.0e-5, bounds_padding=1.0e-5)[7]
    assert hit.distance == pytest.approx(1.0e-6, rel=1e-6)
    # Just outside the plate edge in-plane: refused by the natural slack, not by distance.
    outside = {8: (1.0 + 1.0e-6, 0.5, 0.0)}
    with pytest.raises(CorrespondenceError, match="CORRESPONDENCE_NOT_LOCATED"):
        locate_for_transfer(mesh, outside, candidate_elements=everything(mesh, outside),
                            physical_tolerance=1.0e-5, bounds_padding=1.0e-5)


def test_quadratic_elements_are_not_admitted():
    nodes = {k: np.asarray(v, float) for k, v in enumerate(
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.5, 0, 0), (1, 0.5, 0), (0.5, 1, 0), (0, 0.5, 0)], start=1)}
    mesh = Mesh(nodes=nodes, quads={1: tuple(range(1, 9))}, elements_of_face={1: [1]}, order="quadratic")
    with pytest.raises(CorrespondenceError, match="CORRESPONDENCE_ELEMENT_NOT_ADMITTED"):
        locate_for_transfer(mesh, {3: (0.3, 0.3, 0.0)}, candidate_elements={3: (1,)})


def test_results_are_ordered_and_deterministic():
    mesh = grid_mesh(4)
    points = {30: (0.5, 0.5, 0.0), 10: (0.1, 0.9, 0.0), 20: (0.75, 0.25, 0.0)}
    first = locate_for_transfer(mesh, points, candidate_elements=everything(mesh, points))
    second = locate_for_transfer(mesh, dict(reversed(list(points.items()))), candidate_elements=everything(mesh, points))
    assert list(first) == [10, 20, 30] and first == second


# ----------------------------------------------------------------------
# review round 1 additions
# ----------------------------------------------------------------------
def single_quad(points, faces=((1, (1,)),)):
    nodes = {k: np.asarray(v, float) for k, v in enumerate(points, start=1)}
    return Mesh(nodes=nodes, quads={1: (1, 2, 3, 4)}, elements_of_face={f: list(e) for f, e in faces})


def test_warped_and_degenerate_elements_refuse():
    warped = single_quad([(0, 0, 0), (1, 0, 0), (1, 1, 0.01), (0, 1, 0)])
    with pytest.raises(CorrespondenceError, match="CORRESPONDENCE_ELEMENT_NOT_PLANAR"):
        locate_for_transfer(warped, {1: (0.5, 0.5, 0.0)}, candidate_elements={1: (1,)},
                            physical_tolerance=0.1, bounds_padding=0.1)
    folded = single_quad([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)])
    with pytest.raises(CorrespondenceError, match="CORRESPONDENCE_ELEMENT_DEGENERATE"):
        locate_for_transfer(folded, {1: (0.5, 0.5, 0.0)}, candidate_elements={1: (1,)})


def test_near_edge_weight_disagreement_refuses_beyond_tolerance():
    mesh = grid_mesh(2)
    beside = {4: (0.5 + 1.0e-4, 0.25, 0.0)}  # inside both columns only with a large natural slack
    wide = dict(natural_tolerance=1.0e-2, bounds_padding=2.0e-4)
    with pytest.raises(CorrespondenceError, match="CORRESPONDENCE_NONCONFORMING"):
        locate_for_transfer(mesh, beside, candidate_elements=everything(mesh, beside), **wide)
    # With a permissive weight tolerance both elements are examined and reported.
    hit = locate_for_transfer(mesh, beside, candidate_elements=everything(mesh, beside), weight_tolerance=1.0, **wide)[4]
    assert hit.candidates >= 2 and len(hit.alternatives) == 1 and hit.weight_disagreement > 1.0e-5
    on_edge = {4: (0.5, 0.25, 0.0)}
    hit = locate_for_transfer(mesh, on_edge, candidate_elements=everything(mesh, on_edge))[4]
    assert hit.weight_disagreement <= 1.0e-12


def test_face_membership_is_complete_or_refused():
    shared = single_quad([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], faces=((1, (1,)), (2, (1,))))
    hit = locate_for_transfer(shared, {1: (0.5, 0.5, 0.0)}, candidate_elements={1: (1,)})[1]
    assert hit.incident_source_faces == (1, 2)
    orphan = single_quad([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], faces=())
    with pytest.raises(CorrespondenceError, match="CORRESPONDENCE_FACE_MISSING"):
        locate_for_transfer(orphan, {1: (0.5, 0.5, 0.0)}, candidate_elements={1: (1,)})
    assert locate_for_transfer(orphan, {1: (0.5, 0.5, 0.0)}, candidate_elements={1: (1,)},
                               require_faces=False)[1].incident_source_faces == ()


def test_mapped_point_is_the_element_map_and_distance_is_measured_to_it():
    mesh = grid_mesh(2)
    point = {3: (0.3, 0.3, 4.0e-10)}
    hit = locate_for_transfer(mesh, point, candidate_elements=everything(mesh, point))[3]
    mapped = sum(w * mesh.nodes[n] for w, n in zip(hit.weights, hit.source_node_ids))
    assert np.allclose(hit.x_p, mapped, atol=1e-15)
    assert hit.distance == pytest.approx(4.0e-10, rel=1e-6)


def test_disconnected_sheets_refuse_even_with_a_permissive_weight_tolerance():
    mesh = coincident_sheets()
    with pytest.raises(CorrespondenceError, match="CORRESPONDENCE_NONCONFORMING"):
        locate_for_transfer(mesh, {9: (0.4, 0.6, 0.0)}, candidate_elements={9: (1, 2)}, weight_tolerance=1.0)


def test_subthreshold_warp_uses_the_element_map_not_the_fitted_plane():
    warp = 4.0e-10  # below the 1e-9 planarity limit for a unit quad
    mesh = single_quad([(0, 0, 0), (1, 0, warp), (1, 1, 0), (0, 1, warp)])
    point = {1: (0.25, 0.25, 0.0)}
    hit = locate_for_transfer(mesh, point, candidate_elements={1: (1,)}, physical_tolerance=1e-8, bounds_padding=1e-8)[1]
    mapped = sum(w * mesh.nodes[n] for w, n in zip(hit.weights, hit.source_node_ids))
    assert np.allclose(hit.x_p, mapped, atol=1e-16)
    assert hit.distance == pytest.approx(float(np.linalg.norm(np.asarray(point[1]) - mapped)), rel=1e-9)
    assert hit.distance > 0.0
