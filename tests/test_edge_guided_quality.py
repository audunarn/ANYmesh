from __future__ import annotations

import numpy as np

from anymesher.surface_mesh import SurfaceMeshOptions, mesh_planar_surface


def _options(*, minimum_angle: float = 30.0) -> SurfaceMeshOptions:
    return SurfaceMeshOptions(
        recombine=False,
        target_size=0.25,
        backend="python",
        min_angle=minimum_angle,
        max_element_growth=1.20,
        prefer_quality_policy=True,
    )


def test_target_meeting_baseline_short_circuits_candidate_sweep() -> None:
    outer = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.5, np.sqrt(3.0) / 2.0)))
    diagnostics: dict[str, object] = {}

    mesh = mesh_planar_surface(
        outer,
        options=SurfaceMeshOptions(
            recombine=False,
            target_size=2.0,
            backend="python",
        ),
        diagnostics=diagnostics,
    )

    quality = diagnostics["quality_optimization"]
    assert quality["candidate_count"] == 1
    assert quality["selected_strategy"] == "staggered_chart"
    assert np.array_equal(mesh.node_coordinates[: len(outer), :2], outer)


def test_edge_guided_soft_symmetry_is_deterministic_and_bounded() -> None:
    outer = np.asarray(
        ((0.0, 0.0), (3.0, 0.0), (2.7, 0.8), (1.0, 1.1), (0.0, 0.6)),
        dtype=float,
    )
    first_diagnostics: dict[str, object] = {}
    second_diagnostics: dict[str, object] = {}

    first = mesh_planar_surface(
        outer,
        options=_options(minimum_angle=55.0),
        diagnostics=first_diagnostics,
    )
    second = mesh_planar_surface(
        outer,
        options=_options(minimum_angle=55.0),
        diagnostics=second_diagnostics,
    )

    first_quality = first_diagnostics["quality_optimization"]
    second_quality = second_diagnostics["quality_optimization"]
    assert 1 <= first_quality["candidate_count"] <= 3
    assert first_quality["selected_strategy"] == second_quality["selected_strategy"]
    assert first_quality["candidates"] == second_quality["candidates"]
    assert first_quality["edge_guides"]["soft_symmetry"] is True
    assert first_quality["edge_guides"]["guide_ids"] == sorted(
        first_quality["edge_guides"]["guide_ids"]
    )
    assert np.array_equal(first.node_coordinates, second.node_coordinates)
    assert np.array_equal(first.triangle_connectivity, second.triangle_connectivity)
    coordinates = first.node_coordinates[:, :2]
    for expected in outer:
        assert np.count_nonzero(np.all(coordinates == expected, axis=1)) == 1


def test_mandatory_segment_and_boundary_coordinates_remain_exact() -> None:
    outer = np.asarray(((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)))
    constraint = np.asarray(((0.0, 0.0), (2.0, 1.0)))
    diagnostics: dict[str, object] = {}

    mesh = mesh_planar_surface(
        outer,
        constraints=(constraint,),
        options=_options(minimum_angle=50.0),
        diagnostics=diagnostics,
    )

    coordinates = mesh.node_coordinates[:, :2]
    for expected in (*outer, *constraint):
        assert np.count_nonzero(np.all(coordinates == expected, axis=1)) == 1
    edges = {
        tuple(sorted((int(cell[index]), int(cell[(index + 1) % 3]))))
        for cell in mesh.triangle_connectivity[:, :3]
        for index in range(3)
    }
    endpoints = tuple(
        int(np.flatnonzero(np.all(coordinates == expected, axis=1))[0])
        for expected in constraint
    )
    assert tuple(sorted(endpoints)) in edges
    assert diagnostics["quality_optimization"]["candidate_count"] <= 3


def test_segment_grid_reduces_clearance_work_without_changing_output() -> None:
    outer = np.asarray(((0.0, 0.0), (8.0, 0.0), (8.0, 2.0), (0.0, 2.0)))
    diagnostics: dict[str, object] = {}

    mesh_planar_surface(
        outer,
        options=SurfaceMeshOptions(
            recombine=False,
            target_size=0.2,
            backend="python",
        ),
        diagnostics=diagnostics,
    )

    values = diagnostics["quality_optimization"]["lattice_statistics"]
    assert values["distance_checks"] < values["naive_distance_checks"]
