from fractions import Fraction

import numpy as np
import pytest

import anymesher.native_v2 as native
from anymesher.errors import MeshError
from anymesher.triangulation import PlanarTriangulation


def seed():
    edges = np.array(((0, 1), (1, 2), (2, 3), (0, 3)), dtype=np.int64)
    return PlanarTriangulation(
        points=np.array(((0., 0.), (1., 0.), (1., 1.), (0., 1.))),
        triangles=np.array(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        segments=edges, boundary_segments=edges,
        mandatory_segments=np.empty((0, 2), dtype=np.int64),
        outer_loop=np.array((0, 1, 2, 3), dtype=np.int64), hole_loops=(),
    )


def run(proposal, size=0.2, **changes):
    registry = native.ComponentSeedRegistry(100)
    original = seed()
    options = native.NativeMeshingOptions(
        point_placement="frontal_delaunay", metric_mode="isotropic_spatial",
        max_insertions=1, max_topology_operations=8,
    )
    result, report = native.frontal_delaunay_refine(
        original, options, target_size=size,
        automatically_seeded_shared_segments={(0, 1): (7, 0, 1)},
        component_seed_registry=registry, _split_proposal=proposal, **changes,
    )
    return original, result, report, registry


def test_overlong_segment_uses_reserved_station_without_claiming_new_mesh_node():
    calls = []
    station = Fraction.from_float(0.3)

    def proposal(*args):
        calls.append(args)
        return station, (0.3, -0.0), 23

    original, result, report, registry = run(proposal)
    assert calls == [(7, Fraction(0), Fraction(1))]
    assert result.points[4].tobytes() == np.array((0.3, -0.0)).tobytes()
    assert result.points[:4].tobytes() == original.points.tobytes()
    assert report["insertions"] == 0
    assert report["reserved_node_reuses"] == report["staged_point_insertions"] == 1
    assert report["shared_segment_splits"] == 1
    assert report["selected_route"] == "frontal_delaunay_budget_limited"
    assert registry.assigned_node_ids == (23,)
    assert report["shared_nodes"][0]["station"] == (station.numerator, station.denominator)


def test_encroached_segment_also_uses_reserved_proposal(monkeypatch):
    # Isolate queue routing with a deterministic interior proposal. The actual
    # encroachment predicate, split, incidence, validation and export all run.
    monkeypatch.setattr(native, "_offcentre", lambda *args: np.array((0.5, 0.05)))
    monkeypatch.setattr(native, "_circumcentre", lambda *args: None)
    phases = []
    _, result, report, registry = run(
        lambda *args: (Fraction(1, 2), (0.5, 0.), 23), size=0.8,
        cancellation_check=phases.append,
    )
    assert "native-v2 proposed-point encroachment scan" in phases
    assert report["shared_segment_splits"] == report["reserved_node_reuses"] == 1
    assert report["insertions"] == 0
    assert len(result.points) == 5 and registry.assigned_node_ids == (23,)


def test_refused_shared_edge_is_not_retried_or_allocated():
    calls = []

    def refuse(*args):
        calls.append(args)
        return None

    _, _, report, registry = run(refuse)
    assert len(calls) == 1
    assert registry.assigned_node_ids == ()
    assert report["shared_segment_splits"] == report["reserved_node_reuses"] == 0
    assert report["refused_shared_edges"] == [(0, 1)]
    assert report["topology_operations"] <= 8


def test_invalid_proposal_does_not_become_geometry_limited():
    with pytest.raises(MeshError, match="exact interior fraction"):
        run(lambda *args: (0.3, (0.3, 0.), 23))


def test_owner_failure_remains_the_original_operational_failure():
    error = RuntimeError("owner unavailable")

    def fail(*args):
        raise error

    with pytest.raises(RuntimeError) as caught:
        run(fail)
    assert caught.value is error


def test_no_proposal_keeps_legacy_allocation_and_diagnostic_shape():
    _, _, report, registry = run(None)
    assert report["insertions"] == report["shared_segment_splits"] == 1
    assert registry.assigned_node_ids == (100,)
    assert "reserved_node_reuses" not in report
    assert "staged_point_insertions" not in report


def test_reversed_native_interval_survives_two_refinement_levels():
    results = []
    for edge, interval in (((0, 1), (7, 0, 1)), ((1, 0), (7, 1, 0))):
        calls = []
        def proposal(owner, lower, upper):
            station = (lower + upper) / 2
            calls.append((owner, lower, upper, station))
            return station, (float(station), 0.), None
        result, report = native.frontal_delaunay_refine(
            seed(), native.NativeMeshingOptions(
                point_placement="frontal_delaunay", metric_mode="isotropic_spatial",
                max_insertions=2, max_topology_operations=8), target_size=.2,
            automatically_seeded_shared_segments={edge: interval},
            component_seed_registry=native.ComponentSeedRegistry(100),
            _split_proposal=proposal)
        assert calls == [(7, Fraction(0), Fraction(1), Fraction(1, 2)),
                         (7, Fraction(0), Fraction(1, 2), Fraction(1, 4))]
        np.testing.assert_array_equal(result.points[4:], ((.5, 0.), (.25, 0.)))
        assert report['insertions'] == report['shared_segment_splits'] == 2
        assert report['reserved_node_reuses'] == 0
        results.append((result, report))
    assert results[0][0].points.tobytes() == results[1][0].points.tobytes()
    assert results[0][0].triangles.tobytes() == results[1][0].triangles.tobytes()
    assert results[0][1] == results[1][1]
