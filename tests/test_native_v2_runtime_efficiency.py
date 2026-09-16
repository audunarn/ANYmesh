"""Runtime-efficiency contracts for the native-v2 quality pipeline."""

import numpy as np
import pytest

from anymesher import native_cpp, native_v2
from anymesher.surface_mesh import (
    SurfaceMeshOptions,
    _make_candidate,
    _optimize_candidate,
    mesh_planar_surface,
)


POINTS = np.asarray(
    ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
    dtype=np.float64,
)
TRIANGLES = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
BOUNDARY = np.asarray(((0, 1), (1, 2), (2, 3), (0, 3)), dtype=np.int64)


@pytest.mark.skipif(
    not native_cpp.COMPILED_NATIVE_V2_AVAILABLE,
    reason="optional native-v2 extension is absent",
)
def test_small_compiled_insertion_does_not_recompute_python_oracle(monkeypatch):
    topology = native_v2.MutableT3Topology(POINTS, TRIANGLES, BOUNDARY)

    def forbidden(*args, **kwargs):
        raise AssertionError("compiled insertion re-entered the Python oracle")

    monkeypatch.setattr(
        native_v2.MutableT3Topology, "_python_insert_with_owners", forbidden
    )
    report = topology.insert_point((0.35, 0.42))

    assert report["native"] is True
    assert topology.epoch == 1


def test_python_insertion_remains_the_absence_fallback(monkeypatch):
    topology = native_v2.MutableT3Topology(POINTS, TRIANGLES, BOUNDARY)
    calls = 0
    original = topology._python_insert_with_owners

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(native_v2, "native_mutable_t3_insert", lambda *a, **k: None)
    monkeypatch.setattr(topology, "_python_insert_with_owners", counted)
    report = topology.insert_point((0.35, 0.42))

    assert report["native"] is False
    assert calls == 1


def test_converged_fixed_optimization_reuses_existing_quality_candidate():
    statistics = {"quality_evaluations": 0}
    candidate = _make_candidate(
        POINTS,
        TRIANGLES,
        settings=SurfaceMeshOptions(recombine=False),
        statistics=statistics,
    )
    before = statistics["quality_evaluations"]

    result = _optimize_candidate(
        candidate,
        BOUNDARY,
        np.empty((0, 2), dtype=np.float64),
        SurfaceMeshOptions(recombine=False),
        statistics=statistics,
    )

    assert result is candidate
    assert statistics["quality_evaluations"] == before
    assert statistics["optimization_noop_short_circuits"] == 1


def test_surface_diagnostics_publish_candidate_work_counts():
    diagnostics = {}
    mesh_planar_surface(
        POINTS,
        target_size=0.25,
        recombine=False,
        backend="python",
        diagnostics=diagnostics,
    )

    work = diagnostics["quality_optimization"]["work_totals"]
    assert work["full_triangulations"] >= 1
    assert work["quality_evaluations"] >= 1
    assert work["optimization_noop_short_circuits"] >= 0
