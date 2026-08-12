from __future__ import annotations

import numpy as np
import pytest

from anymesher.core import MeshCore
from anymesher.quality_v2 import MeshValidityError, assert_valid_mesh, mesh_quality
from anymesher.recombine import recombine_triangles
from anymesher.surface_mesh import insert_midside_nodes, mesh_planar_surface
from anymesher.triangulation import _validate_ring, triangulate_polygon
from anymesher.errors import MeshError


def _edge_set(triangles: np.ndarray) -> set[tuple[int, int]]:
    return {
        tuple(sorted((int(triangle[index]), int(triangle[(index + 1) % 3]))))
        for triangle in triangles
        for index in range(3)
    }


def test_mesh_core_is_dense_stable_lazy_and_topological() -> None:
    points = np.array(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    core = MeshCore(
        points,
        ((0, 1, 2), (0, 2, 3)),
        node_ids=(10, 20, 30, 40),
        triangle_ids=(101, 103),
        triangle_owners=("face-7", "face-7"),
    )

    assert core.node_coordinates.dtype == np.float64
    assert core.triangle_connectivity.dtype == np.int64
    assert core.node_coordinates.flags.c_contiguous
    assert core.owner_table == ("face-7",)
    assert core.owner_of_element(103) == "face-7"
    assert core.nodes is core.nodes
    assert core.tris[101] == (10, 20, 30)
    assert core.element_to_element[0].tolist() == [1]
    assert core.node_to_element[0].tolist() == [0, 1]
    assert np.all(core.node_characteristic_lengths > 0.0)

    changed = core.deactivate_elements((101,))
    assert changed.triangle_ids.tolist() == [101, 103]
    assert list(changed.tris) == [103]
    assert changed.element_to_element[1].size == 0


def test_python_constrained_triangulation_is_deterministic_and_keeps_a_hole() -> None:
    outer = np.array(((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)))
    hole = np.array(((1.0, 1.0), (1.0, 3.0), (3.0, 3.0), (3.0, 1.0)))
    constraint = np.array(((0.0, 2.0), (1.0, 2.0)))

    first = triangulate_polygon(outer, (hole,), (constraint,))
    second = triangulate_polygon(outer, (hole,), (constraint,))

    assert np.array_equal(first.points, second.points)
    assert np.array_equal(first.triangles, second.triangles)
    assert set(map(tuple, first.segments)).issubset(_edge_set(first.triangles))
    triangle_points = first.points[first.triangles]
    first_side = triangle_points[:, 1] - triangle_points[:, 0]
    second_side = triangle_points[:, 2] - triangle_points[:, 0]
    area = 0.5 * np.sum(
        np.abs(first_side[:, 0] * second_side[:, 1] - first_side[:, 1] * second_side[:, 0])
    )
    assert area == pytest.approx(12.0)
    centroids = np.mean(triangle_points, axis=1)
    assert not np.any(
        (centroids[:, 0] > 1.0) & (centroids[:, 0] < 3.0)
        & (centroids[:, 1] > 1.0) & (centroids[:, 1] < 3.0)
    )


def test_dense_straight_edge_subdivisions_are_not_self_intersections() -> None:
    vertices = np.array(
        (
            (0.5, 1.0),
            (0.0, 0.6180339887498949),
            (0.1909830056250525, 0.0),
            (0.8090169943749473, 0.0),
            (1.0, 0.6180339887498947),
        )
    )
    subdivisions = (76, 152, 76, 76, 76)
    ring = np.vstack(
        tuple(
            np.linspace(
                vertices[index],
                vertices[(index + 1) % len(vertices)],
                subdivisions[index],
                endpoint=False,
            )
            for index in range(len(vertices))
        )
    )

    assert ring.shape == (456, 2)
    _validate_ring(ring, tuple(range(len(ring))), "outer loop")


def test_ring_validator_still_rejects_a_true_self_intersection() -> None:
    bow_tie = np.array(((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)))

    with pytest.raises(MeshError, match="outer loop is self-intersecting"):
        _validate_ring(bow_tie, (0, 1, 2, 3), "outer loop")


def test_recombination_makes_a_good_quad_and_preserves_a_constraint() -> None:
    points = np.array(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    triangles = np.array(((0, 1, 2), (0, 2, 3)))
    mesh = MeshCore(points, triangles, triangle_ids=(11, 12))

    hybrid = recombine_triangles(mesh)
    assert isinstance(hybrid, MeshCore)
    assert hybrid.active_quad_count == 1
    assert hybrid.active_triangle_count == 0
    assert hybrid.triangle_ids.tolist() == [11, 12]
    assert not np.any(hybrid.triangle_active)

    protected = recombine_triangles(mesh, protected_edges=((0, 2),))
    assert isinstance(protected, MeshCore)
    assert protected.active_quad_count == 0
    assert protected.active_triangle_count == 2


def test_recombination_does_not_force_a_concave_quad() -> None:
    points = np.array(((0.0, 0.0), (2.0, 0.0), (0.2, 0.01), (0.0, 1.0)))
    mesh = MeshCore(points, ((0, 1, 2), (0, 2, 3)))
    result = recombine_triangles(mesh)
    assert isinstance(result, MeshCore)
    assert result.active_quad_count == 0
    assert result.active_triangle_count == 2


def test_quadratic_enrichment_uses_one_shared_midside_node() -> None:
    mesh = MeshCore(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        ((0, 1, 2), (0, 2, 3)),
        node_ids=(10, 20, 30, 40),
    )
    quadratic = insert_midside_nodes(mesh)

    assert quadratic.triangle_connectivity.shape == (2, 6)
    assert quadratic.num_nodes == 9
    first_shared = quadratic.triangle_connectivity[0, 5]
    second_shared = quadratic.triangle_connectivity[1, 3]
    assert first_shared == second_shared
    assert np.allclose(quadratic.node_coordinates[first_shared], (0.5, 0.5, 0.0))
    assert_valid_mesh(quadratic)


def test_quality_metrics_and_hard_quad_validity() -> None:
    valid = MeshCore(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        quad_connectivity=((0, 1, 2, 3),),
    )
    report = mesh_quality(valid)
    assert report.minimum_scaled_jacobian == pytest.approx(1.0)
    assert report.maximum_aspect_ratio == pytest.approx(1.0)

    bow_tie = MeshCore(valid.node_coordinates, quad_connectivity=((0, 1, 3, 2),))
    with pytest.raises(MeshValidityError):
        assert_valid_mesh(bow_tie)


def test_surface_slice_builds_a_valid_quadratic_hybrid_mesh() -> None:
    mesh = mesh_planar_surface(
        ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)),
        recombine=True,
        order="quadratic",
        owner="panel",
    )

    assert mesh.is_quadratic
    assert mesh.active_element_count > 0
    assert mesh.active_quad_count > 0
    assert mesh.owner_table == ("panel",)
    assert_valid_mesh(mesh)
