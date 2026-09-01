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


def test_recombined_plate_with_hole_prefers_boundary_aligned_collars() -> None:
    outer = np.asarray(
        ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)),
        dtype=float,
    )
    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    hole = np.column_stack((
        2.0 + 0.50 * np.cos(angles),
        2.0 + 0.50 * np.sin(angles),
    ))
    first_diagnostics: dict[str, object] = {}
    second_diagnostics: dict[str, object] = {}

    first = mesh_planar_surface(
        outer,
        holes=(hole,),
        options=SurfaceMeshOptions(
            recombine=True,
            target_size=0.25,
            backend="python",
            prefer_quality_policy=True,
        ),
        diagnostics=first_diagnostics,
    )
    second = mesh_planar_surface(
        outer,
        holes=(hole,),
        options=SurfaceMeshOptions(
            recombine=True,
            target_size=0.25,
            backend="python",
            prefer_quality_policy=True,
        ),
        diagnostics=second_diagnostics,
    )

    assert np.array_equal(first.node_coordinates, second.node_coordinates)
    assert np.array_equal(first.triangle_connectivity, second.triangle_connectivity)
    assert np.array_equal(first.quad_connectivity, second.quad_connectivity)
    quality = first_diagnostics["quality_optimization"]
    assert quality == second_diagnostics["quality_optimization"]
    assert quality["candidate_count"] == 3
    assert quality["boundary_collar_skip_reason"] is None
    assert {item["strategy"] for item in quality["candidates"]} == {
        "staggered_chart",
        "outer_boundary_collar",
        "outer_hole_collar",
    }
    baseline, outer_collar, complete_collar = quality["candidates"]
    first_collar, second_collar = quality["boundary_collars"]
    assert first_collar["preparation_cache_hit"] is False
    assert second_collar["preparation_cache_hit"] is True
    assert outer_collar["outer_alignment"]["maximum_normal_error_degrees"] < (
        baseline["outer_alignment"]["maximum_normal_error_degrees"]
    )
    assert complete_collar["outer_alignment"]["maximum_normal_error_degrees"] <= (
        baseline["outer_alignment"]["maximum_normal_error_degrees"]
    )
    assert complete_collar["hole_alignment"]["maximum_normal_error_degrees"] < (
        baseline["hole_alignment"]["maximum_normal_error_degrees"]
    )
    assert outer_collar["alignment_qualified"] is True
    assert complete_collar["alignment_qualified"] is True
    selected = quality["published_alignment"]
    assert selected["outer"]["mean_normal_error_degrees"] <= 20.0
    assert selected["outer"]["maximum_normal_error_degrees"] <= 50.0
    assert selected["holes"]["mean_normal_error_degrees"] <= 25.0
    assert selected["holes"]["maximum_normal_error_degrees"] <= 50.0
    assert quality["final_quality"]["invalid_element_count"] == 0
    coordinates = first.node_coordinates[:, :2]
    for expected in (*outer, *hole):
        assert np.count_nonzero(np.all(coordinates == expected, axis=1)) == 1
