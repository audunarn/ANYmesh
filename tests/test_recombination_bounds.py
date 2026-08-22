from __future__ import annotations

import numpy as np
import pytest

from anymesher.core import MeshCore
from anymesher.recombine import recombine_triangles_with_report
from anymesher.surface_mesh import insert_midside_nodes


def _square_mesh(*, owners: tuple[int, int] = (0, 0)) -> MeshCore:
    return MeshCore(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        ((0, 1, 2), (0, 2, 3)),
        triangle_ids=(11, 12),
        owner_table=("plate-a", "plate-b"),
        triangle_owner_handles=owners,
    )


def test_recombination_preserves_equal_owners_and_quadratic_midsides() -> None:
    quadratic = insert_midside_nodes(_square_mesh())

    report = recombine_triangles_with_report(quadratic)

    assert report.pair_count == 1
    quad_id = int(report.created_quad_ids[0])
    assert report.mesh.owner_of_element(quad_id) == "plate-a"
    assert report.mesh.quad_connectivity.shape == (1, 8)
    assert len(set(map(int, report.mesh.quad_connectivity[0, 4:]))) == 4


def test_recombination_does_not_invent_an_owner_across_a_boundary() -> None:
    report = recombine_triangles_with_report(_square_mesh(owners=(0, 1)))

    assert report.pair_count == 1
    assert report.mesh.owner_of_element(int(report.created_quad_ids[0])) is None


def test_local_exchange_avoids_greedy_stranding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (2.0, 0.7), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0))
    )
    triangles = ((0, 1, 4), (0, 4, 3), (1, 2, 5), (1, 5, 4))

    def quality(
        _points: np.ndarray, corners: tuple[int, int, int, int]
    ) -> dict[str, float]:
        central = set(corners) == {0, 1, 4, 5}
        return {
            "scaled_jacobian": 1.0 if central else 0.8,
            "aspect_ratio": 1.0,
            "minimum_angle": 90.0,
            "maximum_angle": 90.0,
            "warpage": 0.0,
        }

    monkeypatch.setattr("anymesher.recombine.quad_candidate_quality", quality)
    report = recombine_triangles_with_report(MeshCore(points, triangles))

    assert report.exchange_count == 1
    assert report.pair_count == 2
    assert report.mesh.active_triangle_count == 0


def test_exchange_work_is_hard_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    points = np.asarray([(float(index), 0.0) for index in range(18)])
    triangles = np.asarray(
        [(3 * index, 3 * index + 1, 3 * index + 2) for index in range(6)]
    )
    mesh = MeshCore(points, triangles)
    incidence = {
        (0, 1): [0, 1],
        (1, 2): [0, 2],
        (2, 3): [0, 3],
        (3, 4): [1, 4],
        (4, 5): [1, 5],
    }

    monkeypatch.setattr("anymesher.recombine._edge_incidence", lambda _mesh: incidence)
    monkeypatch.setattr(
        "anymesher.recombine._quad_cycle", lambda _mesh, _first, _second: (0, 1, 2, 3)
    )
    monkeypatch.setattr(
        "anymesher.recombine.quad_candidate_quality",
        lambda _points, _corners: {
            "scaled_jacobian": 1.0,
            "aspect_ratio": 1.0,
            "minimum_angle": 90.0,
            "maximum_angle": 90.0,
            "warpage": 0.0,
        },
    )

    report = recombine_triangles_with_report(mesh, max_exchange_work=1)

    assert report.exchange_work == 1
    assert report.exchange_truncated
    assert report.pair_count == 1


def test_recombination_cancellation_is_checked_before_candidate_work() -> None:
    phases: list[str] = []

    def cancel(phase: str) -> None:
        phases.append(phase)
        if phase == "triangle recombination candidates":
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        recombine_triangles_with_report(_square_mesh(), cancellation_check=cancel)

    assert phases == [
        "triangle recombination start",
        "triangle recombination candidates",
    ]


@pytest.mark.parametrize("bad", (0, -1, 1.5, True))
def test_exchange_budget_must_be_a_positive_integer(bad: object) -> None:
    with pytest.raises(Exception, match="max_exchange_work"):
        recombine_triangles_with_report(_square_mesh(), max_exchange_work=bad)  # type: ignore[arg-type]
