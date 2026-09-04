from __future__ import annotations

from math import sqrt

import numpy as np
import pytest

from anymesher import (
    Mesh,
    S3RepairError,
    S3RepairPolicy,
    assert_s3_admissible,
    nodal_normals,
    repair_s3_admission,
)


OWNER = (0.0, 0.0, 1.0)


def _equilateral(*, reversed_winding: bool = False) -> Mesh:
    connectivity = (1, 3, 2) if reversed_winding else (1, 2, 3)
    return Mesh(
        nodes={
            1: np.asarray((0.0, 0.0, 0.0)),
            2: np.asarray((1.0, 0.0, 0.0)),
            3: np.asarray((0.5, sqrt(3.0) / 2.0, 0.0)),
        },
        tris={11: connectivity},
    )


def _flippable_patch() -> Mesh:
    return Mesh(
        nodes={
            1: np.asarray((0.0, 0.0, 0.0)),
            2: np.asarray((1.0, 0.0, 0.0)),
            3: np.asarray((1.0, 2.0, 0.0)),
            4: np.asarray((0.0, 1.0, 0.0)),
        },
        tris={10: (1, 2, 3), 20: (1, 3, 4)},
        elements_of_face={7: [10, 20]},
        elements_of_sheet={8: [10, 20]},
        activity={10: 0.75, 20: 0.75},
    )


def _signature(mesh: Mesh) -> tuple[object, ...]:
    return (
        tuple(
            (node_id, tuple(float(value) for value in mesh.nodes[node_id]))
            for node_id in sorted(mesh.nodes)
        ),
        tuple((element_id, tuple(mesh.tris[element_id])) for element_id in sorted(mesh.tris)),
        tuple((key, tuple(value)) for key, value in sorted(mesh.elements_of_face.items())),
        tuple((key, tuple(value)) for key, value in sorted(mesh.elements_of_sheet.items())),
        tuple(sorted(mesh.activity.items())),
    )


def test_authoritative_winding_repair_is_explicit_and_nonmutating() -> None:
    original = _equilateral(reversed_winding=True)

    result = repair_s3_admission(
        original,
        element_owner_normals={11: OWNER},
    )

    assert original.tris[11] == (1, 3, 2)
    assert result.mesh.tris[11] == (1, 2, 3)
    assert result.winding_repairs == 1
    assert result.edge_flips == 0
    assert result.refinement_splits == 0
    assert result.admission.admitted
    assert result.owner_normal_map() == {11: OWNER}
    assert [item.status for item in result.attempts] == ["accepted", "accepted"]
    np.testing.assert_allclose(nodal_normals(result.mesh, element_owner_normals={11: OWNER})[1], OWNER)


def test_winding_limit_is_a_typed_auditable_rejection() -> None:
    original = _equilateral(reversed_winding=True)

    with pytest.raises(S3RepairError) as caught:
        repair_s3_admission(
            original,
            element_owner_normals={11: OWNER},
            repair_policy=S3RepairPolicy(maximum_winding_repairs=0),
        )

    assert original.tris[11] == (1, 3, 2)
    assert caught.value.attempts[-1].action == "winding"
    assert caught.value.attempts[-1].status == "limit"
    assert "maximum_winding_repairs" in caught.value.attempts[-1].detail


def test_winding_repair_restores_opposite_shared_edge_traversal() -> None:
    mesh = Mesh(
        nodes={
            1: np.asarray((0.0, 0.0, 0.0)),
            2: np.asarray((1.0, 0.0, 0.0)),
            3: np.asarray((1.0, 1.0, 0.0)),
            4: np.asarray((0.0, 1.0, 0.0)),
        },
        tris={10: (1, 2, 3), 20: (1, 4, 3)},
    )

    result = repair_s3_admission(
        mesh,
        element_owner_normals={10: OWNER, 20: OWNER},
    )

    assert result.mesh.tris[20] == (1, 3, 4)
    assert result.admission.topology_violations == ()
    assert result.winding_repairs == 1


def test_diagonal_flip_is_deterministic_and_preserves_stable_scopes() -> None:
    original = _flippable_patch()
    owners = {10: OWNER, 20: OWNER}

    first = repair_s3_admission(original, element_owner_normals=owners)
    second = repair_s3_admission(original, element_owner_normals=owners)

    assert _signature(first.mesh) == _signature(second.mesh)
    assert first.attempts == second.attempts
    assert first.owner_normals == second.owner_normals
    assert first.element_ids == (10, 20)
    assert set(first.mesh.tris) == {10, 20}
    assert {frozenset(value) for value in first.mesh.tris.values()} == {
        frozenset((1, 2, 4)),
        frozenset((2, 3, 4)),
    }
    assert first.mesh.elements_of_face == {7: [10, 20]}
    assert first.mesh.elements_of_sheet == {8: [10, 20]}
    assert first.mesh.activity == {10: 0.75, 20: 0.75}
    assert first.edge_flips == 1
    assert first.edge_flip_attempts == 1
    assert first.admission.admitted
    assert first.admission.topology_violations == ()
    assert_s3_admissible(first.mesh, element_owner_normals=first.owner_normal_map())
    assert _signature(original) == _signature(_flippable_patch())


def test_flip_limit_never_returns_a_legacy_or_partial_mesh() -> None:
    original = _flippable_patch()

    with pytest.raises(S3RepairError) as caught:
        repair_s3_admission(
            original,
            element_owner_normals={10: OWNER, 20: OWNER},
            repair_policy=S3RepairPolicy(
                maximum_edge_flips=0,
                maximum_refinement_splits=0,
            ),
        )

    assert _signature(original) == _signature(_flippable_patch())
    assert caught.value.admission is not None
    assert not caught.value.admission.admitted
    assert any(
        item.action == "edge_flip" and item.status == "limit"
        for item in caught.value.attempts
    )
    assert caught.value.attempts[-1].action == "adjudication"
    assert caught.value.attempts[-1].status == "rejected"
    diagnostic = caught.value.to_diagnostic()
    assert diagnostic["schema"] == "anymesher.s3-repair-error-diagnostic-v1"
    assert diagnostic["admission"]["failing_element_count"] == 2
    assert diagnostic["repair"]["attempt_count"] == len(caught.value.attempts)
    assert diagnostic["quality_policy"]["minimum_angle_deg"] == 30.0


def test_candidate_attempt_limits_bound_rejected_work() -> None:
    mesh = _flippable_patch()

    with pytest.raises(S3RepairError) as caught:
        repair_s3_admission(
            mesh,
            element_owner_normals={10: OWNER, 20: OWNER},
            repair_policy=S3RepairPolicy(
                maximum_edge_flip_attempts=0,
                maximum_refinement_attempts=0,
            ),
        )

    limits = [item.detail for item in caught.value.attempts if item.status == "limit"]
    assert "maximum_edge_flip_attempts exhausted" in limits
    assert "maximum_refinement_attempts exhausted" in limits


def test_local_refinement_refuses_a_nonconforming_scope_with_an_audit() -> None:
    original = _flippable_patch()

    with pytest.raises(S3RepairError) as caught:
        repair_s3_admission(
            original,
            element_ids=(10,),
            element_owner_normals={10: OWNER},
            repair_policy=S3RepairPolicy(maximum_edge_flips=0),
        )

    assert _signature(original) == _signature(_flippable_patch())
    assert any(
        item.action == "refinement"
        and item.status == "rejected"
        and "unselected shell" in item.detail
        for item in caught.value.attempts
    )


def test_repair_rejects_mismatched_element_scopes_before_a_flip() -> None:
    mesh = _flippable_patch()
    mesh.elements_of_face = {7: [10], 8: [20]}

    with pytest.raises(S3RepairError) as caught:
        repair_s3_admission(
            mesh,
            element_owner_normals={10: OWNER, 20: OWNER},
            repair_policy=S3RepairPolicy(maximum_refinement_splits=0),
        )

    assert any(
        item.action == "edge_flip"
        and item.status == "rejected"
        and "association scopes differ" in item.detail
        for item in caught.value.attempts
    )


def test_repair_requires_exact_t3_ids_and_authoritative_normals() -> None:
    mesh = _equilateral()

    with pytest.raises(S3RepairError, match="authoritative owner normal is missing") as missing:
        repair_s3_admission(mesh, element_owner_normals={})
    assert missing.value.attempts[0].action == "validate"

    with pytest.raises(S3RepairError, match="invalid IDs") as wrong_id:
        repair_s3_admission(
            mesh,
            element_ids=(99,),
            element_owner_normals={99: OWNER},
        )
    assert wrong_id.value.attempts[0].element_ids == (99,)


def test_repair_policy_limits_reject_booleans_and_negative_values() -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        S3RepairPolicy(maximum_edge_flips=True)
    with pytest.raises(ValueError, match="non-negative integer"):
        S3RepairPolicy(maximum_added_nodes=-1)


def test_geometry_bound_edges_are_never_refined_by_chord_midpoint() -> None:
    mesh = Mesh(
        nodes={
            1: np.asarray((0.0, 0.0, 0.0)),
            2: np.asarray((4.0, 0.0, 0.0)),
            3: np.asarray((0.1, 0.05, 0.0)),
        },
        tris={11: (1, 2, 3)},
        nodes_of_edge={
            101: [1, 2],
            102: [2, 3],
            103: [3, 1],
        },
    )

    with pytest.raises(S3RepairError) as caught:
        repair_s3_admission(
            mesh,
            element_owner_normals={11: OWNER},
            repair_policy=S3RepairPolicy(maximum_edge_flips=0),
        )

    assert any(
        item.action == "refinement"
        and item.status == "rejected"
        and "geometry-bound" in item.detail
        for item in caught.value.attempts
    )
    assert set(mesh.nodes) == {1, 2, 3}


def test_legacy_triangle_behavior_changes_only_on_explicit_repair() -> None:
    legacy = _equilateral(reversed_winding=True)
    connectivity = tuple(legacy.tris[11])

    assert nodal_normals(legacy) == {}
    assert tuple(legacy.tris[11]) == connectivity

    repaired = repair_s3_admission(
        legacy,
        element_owner_normals={11: OWNER},
    )
    assert tuple(legacy.tris[11]) == connectivity
    assert repaired.mesh.tris[11] != connectivity
