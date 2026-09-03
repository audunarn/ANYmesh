from __future__ import annotations

from math import sqrt

import numpy as np
import pytest

from anymesher import (
    Mesh,
    MeshError,
    S3QualityError,
    S3QualityPolicy,
    assert_s3_admissible,
    evaluate_s3_admission,
    nodal_normals,
)


def _equilateral(*, reversed_winding: bool = False) -> Mesh:
    connectivity = (1, 3, 2) if reversed_winding else (1, 2, 3)
    return Mesh(
        nodes={
            1: np.array([0.0, 0.0, 0.0]),
            2: np.array([1.0, 0.0, 0.0]),
            3: np.array([0.5, sqrt(3.0) / 2.0, 0.0]),
        },
        tris={11: connectivity},
    )


def test_qualified_s3_admits_an_oriented_equilateral_triangle() -> None:
    report = assert_s3_admissible(
        _equilateral(), element_owner_normals={11: (0.0, 0.0, 1.0)}
    )

    assert report.admitted
    assert report.elements[0].normalized_area == pytest.approx(1.0)
    assert report.elements[0].minimum_angle_deg == pytest.approx(60.0)
    assert report.elements[0].minimum_scaled_jacobian == pytest.approx(sqrt(3.0) / 2.0)


def test_qualified_s3_rejects_missing_authority_and_complete_reversal() -> None:
    with pytest.raises(S3QualityError, match="authoritative owner normal is missing"):
        assert_s3_admissible(_equilateral())

    report = evaluate_s3_admission(
        _equilateral(reversed_winding=True),
        element_owner_normals={11: (0.0, 0.0, 1.0)},
    )
    assert not report.admitted
    assert any("signed area ratio" in item for item in report.violations)


def test_quality_envelope_rejects_a_slender_triangle_without_fallback() -> None:
    mesh = Mesh(
        nodes={
            1: np.array([0.0, 0.0, 0.0]),
            2: np.array([4.0, 0.0, 0.0]),
            3: np.array([0.1, 0.05, 0.0]),
        },
        tris={7: (1, 2, 3)},
    )

    with pytest.raises(S3QualityError) as caught:
        assert_s3_admissible(mesh, element_owner_normals={7: (0.0, 0.0, 1.0)})

    message = str(caught.value)
    assert "minimum angle" in message
    assert "normalized area" in message


def test_t6_is_never_reported_as_qualified_s3() -> None:
    mesh = Mesh(
        nodes={
            1: np.asarray((0.0, 0.0, 0.0)),
            2: np.asarray((1.0, 0.0, 0.0)),
            3: np.asarray((0.5, sqrt(3.0) / 2.0, 0.0)),
            4: np.asarray((0.5, 0.0, 0.0)),
            5: np.asarray((0.75, sqrt(3.0) / 4.0, 0.0)),
            6: np.asarray((0.25, sqrt(3.0) / 4.0, 0.0)),
        },
        tris={7: (1, 2, 3, 4, 5, 6)},
    )

    report = evaluate_s3_admission(
        mesh, element_owner_normals={7: (0.0, 0.0, 1.0)}
    )

    assert not report.admitted
    assert report.elements == ()
    assert "exactly 3 required" in report.topology_violations[0]


def test_explicit_s3_scope_ignores_a_disconnected_legacy_t6() -> None:
    mesh = _equilateral()
    mesh.nodes.update(
        {
            20: np.asarray((4.0, 0.0, 0.0)),
            21: np.asarray((5.0, 0.0, 0.0)),
            22: np.asarray((4.5, sqrt(3.0) / 2.0, 0.0)),
            23: np.asarray((4.5, 0.0, 0.0)),
            24: np.asarray((4.75, sqrt(3.0) / 4.0, 0.0)),
            25: np.asarray((4.25, sqrt(3.0) / 4.0, 0.0)),
        }
    )
    mesh.tris[99] = (20, 21, 22, 23, 24, 25)

    report = evaluate_s3_admission(
        mesh,
        element_ids=(11,),
        element_owner_normals={11: (0.0, 0.0, 1.0)},
    )

    assert report.admitted
    assert tuple(item.element_id for item in report.elements) == (11,)


def test_directed_shared_edge_must_have_opposite_traversal() -> None:
    mesh = Mesh(
        nodes={
            1: np.array([0.0, 0.0, 0.0]),
            2: np.array([1.0, 0.0, 0.0]),
            3: np.array([0.5, 1.0, 0.0]),
            4: np.array([1.5, 1.0, 0.0]),
        },
        tris={10: (1, 2, 3), 20: (2, 3, 4)},
    )
    permissive = S3QualityPolicy(
        minimum_angle_deg=1.0,
        maximum_angle_deg=179.0,
        maximum_edge_ratio=100.0,
        minimum_scaled_jacobian=1.0e-6,
        minimum_normalized_area=1.0e-6,
    )

    report = evaluate_s3_admission(
        mesh,
        element_owner_normals={10: (0.0, 0.0, 1.0), 20: (0.0, 0.0, -1.0)},
        policy=permissive,
    )

    assert not report.admitted
    assert report.topology_violations == (
        "edge (2, 3) has equal traversal in elements 10 and 20",
    )


def test_declared_balanced_four_way_two_sheet_junction_is_qualified() -> None:
    mesh = Mesh(
        nodes={
            1: np.asarray((0.0, 0.0, 0.0)),
            2: np.asarray((1.0, 0.0, 0.0)),
            3: np.asarray((0.0, 1.0, 0.0)),
            4: np.asarray((1.0, -1.0, 0.0)),
            5: np.asarray((0.0, 0.0, 1.0)),
            6: np.asarray((1.0, 0.0, -1.0)),
            7: np.asarray((1.0, 0.0, 1.0)),
            8: np.asarray((0.0, 0.0, -1.0)),
        },
        tris={10: (1, 2, 3), 20: (2, 1, 4)},
        quads={30: (1, 2, 7, 5), 40: (2, 1, 8, 6)},
        elements_of_sheet={101: [10, 20], 202: [30, 40]},
        declared_plate_junction_edges=((1, 2),),
    )
    normals = {10: (0.0, 0.0, 1.0), 20: (0.0, 0.0, 1.0)}

    report = evaluate_s3_admission(mesh, element_owner_normals=normals)

    assert report.admitted
    assert report.topology_violations == ()
    assert report.qualified_junction_edges == ((1, 2),)

    mesh.declared_plate_junction_edges = ()
    rejected = evaluate_s3_admission(mesh, element_owner_normals=normals)
    assert not rejected.admitted
    assert rejected.qualified_junction_edges == ()
    assert "junction is not explicitly declared" in rejected.topology_violations[0]


def test_mixed_nodal_normals_include_triangles_and_follow_owner_normal() -> None:
    mesh = Mesh(
        nodes={
            1: np.array([0.0, 0.0, 0.0]),
            2: np.array([1.0, 0.0, 0.0]),
            3: np.array([1.0, 1.0, 0.0]),
            4: np.array([0.0, 1.0, 0.0]),
            5: np.array([2.0, 0.0, 0.0]),
            6: np.array([2.5, 1.0, 0.0]),
            7: np.array([3.0, 0.0, 0.0]),
        },
        quads={10: (1, 2, 3, 4)},
        tris={20: (5, 6, 7)},  # Wound toward -z on purpose.
    )

    normals = nodal_normals(mesh, element_owner_normals={20: (0.0, 0.0, 1.0)})

    assert set(normals) == set(mesh.nodes)
    for value in normals.values():
        np.testing.assert_allclose(value, np.array([0.0, 0.0, 1.0]))

    with pytest.raises(MeshError, match="ambiguous or tangential"):
        nodal_normals(mesh, element_owner_normals={20: (1.0, 0.0, 0.0)})


def test_legacy_nodal_normal_call_remains_quadrilateral_only() -> None:
    mesh = Mesh(
        nodes={
            1: np.asarray((0.0, 0.0, 0.0)),
            2: np.asarray((1.0, 0.0, 0.0)),
            3: np.asarray((0.0, 1.0, 0.0)),
        },
        tris={20: (1, 2, 3)},
    )

    assert nodal_normals(mesh) == {}
    assert set(nodal_normals(mesh, include_triangles=True)) == {1, 2, 3}


def test_scaled_jacobian_is_shape_only_with_orientation_checked_separately() -> None:
    mesh = _equilateral()
    tilted = np.asarray((np.sqrt(0.99), 0.0, 0.1))

    report = evaluate_s3_admission(
        mesh, element_owner_normals={11: tilted}
    )

    assert report.admitted
    assert report.elements[0].minimum_scaled_jacobian == pytest.approx(
        np.sqrt(3.0) / 2.0
    )


def test_nearly_tangential_owner_normal_fails_the_shared_contract() -> None:
    mesh = _equilateral()
    near_tangent = np.asarray((1.0, 0.0, 1.0e-9))

    report = evaluate_s3_admission(
        mesh, element_owner_normals={11: near_tangent}
    )

    assert not report.admitted
    assert any(
        "owner-normal alignment" in item
        for item in report.elements[0].violations
    )
