from __future__ import annotations

import numpy as np
import pytest

import anymesher.metric as metric_module

from anygeometry.entities import OrientedEdge
from anygeometry.surfaces import Plane

from anymesher import (
    ExperimentalMetricProvider,
    FeatureDistanceMetricControl,
    GeometryModel,
    EntityRef,
    ImportedMetricSamples,
    IsotropicMetricControl,
    MetricFieldSpec,
    NativeMeshingOptions,
    MutableT3Topology,
    generate_hybrid_mesh,
    limit_metric_gradation,
)
from anymesher.errors import MeshError
from anymesher.refinement import refine_at
from anymesher.surface_mesh import SurfaceMeshOptions, mesh_planar_surface


SQUARE = np.asarray(((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)))


def _frontal_options(**changes) -> SurfaceMeshOptions:
    native = NativeMeshingOptions(
        point_placement="frontal_delaunay",
        metric_mode="isotropic_spatial",
        metric_field=MetricFieldSpec.uniform(0.20),
        max_insertions=24,
        max_topology_operations=2_000,
        cancellation_interval=1,
    )
    return SurfaceMeshOptions(
        target_size=0.45,
        backend="python",
        recombine=False,
        native_options=native,
        **changes,
    )


def test_legacy_defaults_are_byte_equivalent() -> None:
    implicit = mesh_planar_surface(SQUARE, target_size=0.4, backend="python", recombine=False)
    explicit = mesh_planar_surface(
        SQUARE,
        options=SurfaceMeshOptions(
            target_size=0.4,
            backend="python",
            recombine=False,
            native_options=NativeMeshingOptions(),
        ),
    )
    assert implicit.node_coordinates.tobytes() == explicit.node_coordinates.tobytes()
    assert implicit.triangle_connectivity.tobytes() == explicit.triangle_connectivity.tobytes()


def test_opt_in_frontal_route_is_deterministic_and_reports_budgets() -> None:
    first_diagnostics: dict[str, object] = {}
    second_diagnostics: dict[str, object] = {}
    first = mesh_planar_surface(SQUARE, options=_frontal_options(), diagnostics=first_diagnostics)
    second = mesh_planar_surface(SQUARE, options=_frontal_options(), diagnostics=second_diagnostics)
    assert first.node_coordinates.tobytes() == second.node_coordinates.tobytes()
    assert first.triangle_connectivity.tobytes() == second.triangle_connectivity.tobytes()
    report = first_diagnostics["native_v2"]
    assert report["selected_route"] in {
        "frontal_delaunay",
        "frontal_delaunay_baseline_satisfied",
        "frontal_delaunay_geometry_limited",
        "frontal_delaunay_budget_limited",
        "legacy_seed_quality_guard",
    }
    assert report["insertion_budget"] == 24
    assert report["topology_budget"] == 2_000
    assert report["insertions"] > 0
    assert first.num_nodes > 4


def test_uniform_metric_stops_at_an_already_qualified_legacy_seed() -> None:
    baseline = mesh_planar_surface(
        SQUARE,
        options=SurfaceMeshOptions(target_size=0.2, backend="python", recombine=False),
    )
    diagnostics: dict[str, object] = {}
    frontal = mesh_planar_surface(
        SQUARE,
        options=SurfaceMeshOptions(
            target_size=0.2,
            backend="python",
            recombine=False,
            native_options=NativeMeshingOptions(
                point_placement="frontal_delaunay",
                metric_mode="isotropic_spatial",
                metric_field=MetricFieldSpec.uniform(0.2),
            ),
        ),
        diagnostics=diagnostics,
    )
    assert diagnostics["native_v2"]["selected_route"] == "frontal_delaunay_baseline_satisfied"
    assert diagnostics["native_v2"]["insertions"] == 0
    assert frontal.node_coordinates.tobytes() == baseline.node_coordinates.tobytes()
    assert frontal.triangle_connectivity.tobytes() == baseline.triangle_connectivity.tobytes()


def test_frontal_route_preserves_hole_boundary_and_transform_covariance() -> None:
    hole = np.asarray(((0.8, 0.8), (1.2, 0.8), (1.2, 1.2), (0.8, 1.2)))
    first = mesh_planar_surface(SQUARE, (hole,), options=_frontal_options())
    rotation = np.asarray(((0.0, -1.0), (1.0, 0.0)))
    rotated = mesh_planar_surface(SQUARE @ rotation.T, (hole @ rotation.T,), options=_frontal_options())
    assert first.num_nodes == rotated.num_nodes
    assert first.num_triangles == rotated.num_triangles
    first_rows = np.round(first.node_coordinates[:, :2] @ rotation.T, 12)
    second_rows = np.round(rotated.node_coordinates[:, :2], 12)
    np.testing.assert_array_equal(
        first_rows[np.lexsort((first_rows[:, 1], first_rows[:, 0]))],
        second_rows[np.lexsort((second_rows[:, 1], second_rows[:, 0]))],
    )


def test_frontal_cancellation_propagates_without_publication() -> None:
    phases: list[str] = []

    def cancel(phase: str) -> None:
        phases.append(phase)
        if phase == "native-v2 frontal queue":
            raise RuntimeError("cancel front")

    acute = np.asarray(((0.0, 0.0), (2.0, 0.0), (0.02, 0.1)))
    with pytest.raises(RuntimeError, match="cancel front"):
        mesh_planar_surface(acute, options=_frontal_options(), cancellation_check=cancel)
    assert "native-v2 frontal queue" in phases


def test_fixed_acute_protected_corner_is_counted_once_as_geometry_limited() -> None:
    from anymesher.native_v2 import frontal_delaunay_refine
    from anymesher.triangulation import PlanarTriangulation

    points = np.asarray(((0.0, 0.0), (2.0, 0.0), (0.02, 0.1)))
    boundary = np.asarray(((0, 1), (1, 2), (0, 2)), dtype=np.int64)
    triangulation = PlanarTriangulation(
        points=points,
        triangles=np.asarray(((0, 1, 2),), dtype=np.int64),
        segments=boundary,
        boundary_segments=boundary,
        mandatory_segments=np.empty((0, 2), dtype=np.int64),
        outer_loop=np.asarray((0, 1, 2), dtype=np.int64),
        hole_loops=(),
    )
    _refined, report = frontal_delaunay_refine(
        triangulation,
        NativeMeshingOptions(
            point_placement="frontal_delaunay",
            metric_mode="isotropic_spatial",
            max_insertions=8,
            max_topology_operations=32,
        ),
        target_size=2.0,
    )
    assert report["selected_route"] == "frontal_delaunay_geometry_limited"
    assert report["geometry_limited_regions"] == 1
    assert report["insertions"] == 0


def test_curved_activation_is_not_silently_exposed_by_planar_api_options() -> None:
    encoded = NativeMeshingOptions(
        point_placement="frontal_delaunay",
        metric_mode="isotropic_spatial",
        metric_field=MetricFieldSpec.uniform(0.2),
    ).to_dict()
    assert encoded["point_placement"] == "frontal_delaunay"
    assert set(encoded).isdisjoint({"q_morph", "global_matching", "adaptive_remesh"})


def test_public_hybrid_native_path_accepts_spatial_metric_options() -> None:
    geometry = GeometryModel()
    points = geometry.add_points(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    face = geometry.add_plate(points)
    geometry.add_sheet((face,))
    mesh = generate_hybrid_mesh(
        geometry,
        target_size=0.25,
        strategy="native",
        recombine=False,
        native_backend="python",
        native_options=NativeMeshingOptions(
            point_placement="frontal_delaunay",
            metric_mode="isotropic_spatial",
            max_insertions=16,
            max_topology_operations=1_000,
        ),
    )
    assert mesh.nodes
    assert mesh.elements_on(EntityRef("face", face))


def test_public_hybrid_path_binds_and_pullbacks_revisioned_3d_metric() -> None:
    geometry = GeometryModel()
    points = geometry.add_points(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    face = geometry.add_plate(points)
    geometry.add_sheet((face,))
    imported = ImportedMetricSamples(
        str(geometry.model_id),
        geometry.revision,
        ((0.5, 0.5, 0.0),),
        (((100.0, 0.0, 0.0), (0.0, 100.0, 0.0), (0.0, 0.0, 100.0)),),
    )
    mesh = generate_hybrid_mesh(
        geometry,
        target_size=0.25,
        strategy="native",
        recombine=False,
        native_backend="python",
        native_options=NativeMeshingOptions(
            point_placement="frontal_delaunay",
            metric_mode="isotropic_spatial",
            metric_field=MetricFieldSpec(
                IsotropicMetricControl(0.5), imported_samples=(imported,)
            ),
            max_insertions=4,
            max_topology_operations=100,
        ),
    )
    assert mesh.nodes
    assert mesh.elements_on(EntityRef("face", face))


def test_spatial_options_reject_two_metric_authorities() -> None:
    provider = ExperimentalMetricProvider(
        lambda points: np.repeat(np.eye(2)[None, :, :], len(points), axis=0)
    )
    with pytest.raises(MeshError, match="mutually exclusive"):
        NativeMeshingOptions(
            point_placement="frontal_delaunay",
            metric_mode="isotropic_spatial",
            metric_field=MetricFieldSpec.uniform(0.2),
            experimental_metric_provider=provider,
        )
    with pytest.raises(MeshError, match="legacy_lattice"):
        NativeMeshingOptions(
            point_placement="legacy_lattice",
            metric_mode="isotropic_spatial",
            metric_field=MetricFieldSpec.uniform(0.2),
        )


def test_gradation_fails_closed_and_checks_cancellation_inside_sweeps() -> None:
    physical_points = np.asarray(((0.0, 0.0), (0.1, 0.0), (10.0, 0.0)))
    physical_edges = np.asarray(((0, 1), (1, 2)), dtype=np.int64)
    physical, _ = limit_metric_gradation(
        physical_points,
        physical_edges,
        (1.0, 100.0, 100.0),
        1.2,
        max_iterations=8,
    )
    np.testing.assert_allclose(physical, (1.0, 1.02, 3.0), rtol=0.0, atol=1.0e-15)

    points = np.column_stack((np.arange(130, dtype=float), np.zeros(130)))
    edges = np.column_stack((np.arange(129), np.arange(1, 130)))
    targets = np.full(130, 10.0)
    targets[-1] = 0.1
    with pytest.raises(MeshError, match="did not converge"):
        limit_metric_gradation(points, edges, targets, 1.01, max_iterations=2)

    phases: list[str] = []

    def cancel(phase: str) -> None:
        phases.append(phase)
        if phase == "native-v2 metric gradation scan":
            raise RuntimeError("cancel gradation")

    with pytest.raises(RuntimeError, match="cancel gradation"):
        limit_metric_gradation(
            points,
            edges,
            targets,
            1.5,
            cancellation_check=cancel,
            cancellation_interval=1,
        )
    assert "native-v2 metric gradation scan" in phases


@pytest.mark.parametrize(
    "points",
    (
        np.asarray((0.0, 1.0)),
        np.zeros((2, 4)),
        np.asarray(((0.0, 0.0), (np.nan, 1.0))),
        np.asarray(((0.0, 0.0, 0.0), (1.0, np.inf, 0.0))),
    ),
)
def test_gradation_rejects_nonfinite_or_noncoordinate_rows(points: np.ndarray) -> None:
    with pytest.raises(MeshError, match="finite 2D or 3D"):
        limit_metric_gradation(points, ((0, 1),), (1.0, 1.0), 1.5)


def test_large_gradation_and_metric_lengths_use_complete_native_abi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(metric_module, "_NATIVE_BATCH_THRESHOLD", 1)
    import anymesher.native_cpp as native_cpp

    monkeypatch.setattr(native_cpp, "NATIVE_CPP_AVAILABLE", True)
    monkeypatch.setattr(native_cpp, "COMPILED_NATIVE_V2_AVAILABLE", True)
    monkeypatch.setattr(
        native_cpp,
        "native_gradation_limit",
        lambda *_args: (calls.append("gradation") or (np.asarray((1.0, 1.0)), 1)),
    )
    monkeypatch.setattr(
        native_cpp,
        "native_metric_lengths",
        lambda *_args: (calls.append("lengths") or np.asarray((1.0,))),
    )
    points = np.asarray(((0.0, 0.0), (1.0, 0.0)))
    edges = np.asarray(((0, 1),), dtype=np.int64)
    limited, _ = limit_metric_gradation(points, edges, (1.0, 1.0), 1.5)
    lengths = metric_module._metric_lengths(
        points, edges, np.repeat(np.eye(2)[None, :, :], 2, axis=0)
    )
    np.testing.assert_array_equal(limited, (1.0, 1.0))
    np.testing.assert_array_equal(lengths, (1.0,))
    assert calls == ["gradation", "lengths"]

    monkeypatch.setattr(
        native_cpp,
        "native_gradation_limit",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("native failure")),
    )
    with pytest.raises(RuntimeError, match="native failure"):
        limit_metric_gradation(points, edges, (1.0, 1.0), 1.5)


def test_topology_fallback_and_shared_reconstruction_are_cancellable() -> None:
    topology = MutableT3Topology(
        SQUARE,
        np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
    )
    with pytest.raises(RuntimeError, match="cancel locate"):
        topology.locate(
            (0.5, 0.5),
            cancellation_check=lambda phase: (
                (_ for _ in ()).throw(RuntimeError("cancel locate"))
                if phase == "native-v2 fallback location scan"
                else None
            ),
            cancellation_interval=1,
        )

    registry = metric_module  # retain a stable local before the cancellation closure
    del registry
    from anymesher.native_v2 import ComponentSeedRegistry

    shared = MutableT3Topology(
        SQUARE,
        np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        splittable_edges={(0, 2): (7, 0, 1)},
        seed_registry=ComponentSeedRegistry(100),
    )
    before = shared.canonical_export()

    def cancel_reconstruction(phase: str) -> None:
        if phase == "native-v2 shared segment reconstruction":
            raise RuntimeError("cancel reconstruction")

    with pytest.raises(RuntimeError, match="cancel reconstruction"):
        shared.split_segment((0, 2), cancellation_check=cancel_reconstruction)
    after = shared.canonical_export()
    np.testing.assert_array_equal(after[0], before[0])
    np.testing.assert_array_equal(after[1], before[1])


def test_experimental_metric_composes_public_size_field_refinements() -> None:
    geometry = GeometryModel()
    points = geometry.add_points(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    face = geometry.add_plate(points)
    provider = ExperimentalMetricProvider(
        lambda values: np.repeat(np.eye(2)[None, :, :] * 0.01, len(values), axis=0)
    )
    mesh = generate_hybrid_mesh(
        geometry,
        target_size=0.5,
        strategy="native",
        recombine=False,
        native_backend="python",
        refinements=(refine_at((0.5, 0.5, 0.0), 0.08, 0.2),),
        native_options=NativeMeshingOptions(
            point_placement="frontal_delaunay",
            metric_mode="isotropic_spatial",
            experimental_metric_provider=provider,
            max_insertions=24,
            max_topology_operations=2_000,
            cancellation_interval=1,
        ),
    )
    report = mesh.hybrid_diagnostics["triangulation_backend_by_face"][face]["native_v2"]
    assert report["metric_target_minimum"] <= 0.08 * (1.0 + 1.0e-12)


def test_mutable_topology_preserves_cavity_and_flip_owners() -> None:
    topology = MutableT3Topology(
        SQUARE,
        np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        triangle_owners=np.asarray((7, 7), dtype=np.int64),
    )
    topology.insert_point((1.0, 1.0))
    assert set(topology.triangle_owners) == {7}

    flipped = MutableT3Topology(
        SQUARE,
        np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        triangle_owners=np.asarray((9, 9), dtype=np.int64),
    )
    assert flipped.flip_edge((0, 2))
    assert set(flipped.triangle_owners) == {9}


def test_hybrid_frontal_shared_splits_use_one_exported_node_identity() -> None:
    geometry = GeometryModel()
    v0, v1, v2, v3, v4 = geometry.add_points(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.8, 0.5, 0.0),
        )
    )
    e0 = geometry.add_line(v0, v1)
    shared_edge = geometry.add_line(v1, v2)
    e2 = geometry.add_line(v2, v3)
    e3 = geometry.add_line(v3, v0)
    e4 = geometry.add_line(v1, v4)
    e5 = geometry.add_line(v4, v2)
    first = geometry.add_face_from_loop(
        tuple(OrientedEdge(edge, True) for edge in (e0, shared_edge, e2, e3)),
        corners=(0, 1, 2, 3),
        surface=Plane(
            np.asarray((0.0, 0.0, 0.0)),
            np.asarray((1.0, 0.0, 0.0)),
            np.asarray((0.0, 1.0, 0.0)),
        ),
    )
    second = geometry.add_face_from_loop(
        (OrientedEdge(shared_edge, False), OrientedEdge(e4, True), OrientedEdge(e5, True)),
        surface=Plane(
            np.asarray((1.0, 0.0, 0.0)),
            np.asarray((0.8, 0.0, 0.0)),
            np.asarray((0.0, 1.0, 0.0)),
        ),
    )
    mesh = generate_hybrid_mesh(
        geometry,
        target_size=0.5,
        strategy="native",
        recombine=False,
        native_backend="python",
        native_options=NativeMeshingOptions(
            point_placement="frontal_delaunay",
            metric_mode="isotropic_spatial",
            metric_field=MetricFieldSpec(
                IsotropicMetricControl(0.15),
                feature_controls=(
                    FeatureDistanceMetricControl(
                        ((1.0, 0.5, 0.0),), 0.08, 0.6, 1.5, "shared"
                    ),
                ),
            ),
            max_insertions=24,
            max_topology_operations=2_000,
            cancellation_interval=1,
        ),
    )
    edge_nodes = mesh.nodes_on(EntityRef("edge", shared_edge))
    assert len(edge_nodes) > 3
    assert set(edge_nodes).issubset(mesh.nodes_on(EntityRef("face", first)))
    assert set(edge_nodes).issubset(mesh.nodes_on(EntityRef("face", second)))
