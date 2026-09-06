from types import SimpleNamespace
import signal

import numpy as np
import pytest

from anymesher._complementary_trims import complementary_trim_domains


def _ring(*edges):
    return tuple(SimpleNamespace(edge=abs(edge), forward=edge > 0) for edge in edges)


def _trim_model(ring, *, endpoints=None):
    vertices = {
        1: (1, 2), 2: (2, 3), 3: (3, 1), 4: (3, 4),
        10: (10, 11), 11: (11, 12), 12: (12, 10),
    }
    if endpoints is not None:
        vertices.update(endpoints)
    return SimpleNamespace(
        faces={
            1: SimpleNamespace(loop=_ring(10, 11, 12), holes=(ring,)),
            2: SimpleNamespace(loop=_ring(1, 2, 3), holes=()),
        },
        oriented_start_vertex=lambda item: vertices[item.edge][0 if item.forward else 1],
        oriented_end_vertex=lambda item: vertices[item.edge][1 if item.forward else 0],
    )


@pytest.mark.parametrize("ring", [_ring(1, 2, 3), _ring(2, 3, 1), _ring(-3, -2, -1)])
def test_complete_owned_hole_ring_is_complementary(ring):
    model = _trim_model(ring)
    assert complementary_trim_domains(model, 1, 2)
    assert complementary_trim_domains(model, 2, 1)


@pytest.mark.parametrize("ring", [_ring(1, 2, 4), _ring(1, -2, 3), _ring(1, 1, 3), ()])
def test_partial_or_inconsistent_ring_is_not_complementary(ring):
    model = _trim_model(ring)
    assert not complementary_trim_domains(model, 1, 2)


@pytest.mark.parametrize("endpoints", [
    {2: (8, 3)},
    {3: (3, 4)},
    {12: (12, 99)},
])
def test_identical_rings_do_not_exempt_disconnected_or_open_trims(endpoints):
    model = _trim_model(_ring(1, 2, 3), endpoints=endpoints)
    assert not complementary_trim_domains(model, 1, 2)
    assert not complementary_trim_domains(model, 2, 1)


def test_trim_endpoint_failure_propagates():
    model = _trim_model(_ring(1, 2, 3))
    failure = RuntimeError("owner endpoint lookup failed")

    def fail(item):
        raise failure

    model.oriented_end_vertex = fail
    with pytest.raises(RuntimeError) as caught:
        complementary_trim_domains(model, 1, 2)
    assert caught.value is failure


@pytest.mark.parametrize("failure", [None, "initialize", "body", "finalize"])
def test_gmsh_session_restores_callers_sigint(monkeypatch, failure):
    pytest.importorskip("gmsh")
    from anymesher.backends.gmsh import _gmsh_session, gmsh

    previous = signal.getsignal(signal.SIGINT)
    handler = lambda signum, frame: None
    calls = []

    def initialize():
        calls.append("initialize")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        if failure == "initialize":
            raise RuntimeError(failure)

    def finalize():
        calls.append("finalize")
        if failure == "finalize":
            raise RuntimeError(failure)

    monkeypatch.setattr(gmsh, "initialize", initialize)
    monkeypatch.setattr(gmsh, "finalize", finalize)
    monkeypatch.setattr(gmsh.option, "setNumber", lambda *args: None)
    signal.signal(signal.SIGINT, handler)
    try:
        if failure is None:
            with _gmsh_session():
                pass
        else:
            with pytest.raises(RuntimeError, match=failure):
                with _gmsh_session():
                    if failure == "body":
                        raise RuntimeError(failure)
        assert signal.getsignal(signal.SIGINT) is handler
        assert calls == (["initialize"] if failure == "initialize" else ["initialize", "finalize"])
    finally:
        signal.signal(signal.SIGINT, previous)


def test_frontal_rejected_queue_cannot_exceed_topology_budget():
    from anymesher import NativeMeshingOptions
    from anymesher.native_v2 import frontal_delaunay_refine
    from anymesher.triangulation import PlanarTriangulation

    # All corners are protected; several skinny cells must not drain the queue
    # beyond the single authorized topology operation.
    points = np.asarray(((0., 0.), (1., 0.), (2., 0.), (2., .01), (1., .01), (0., .01)))
    segments = np.asarray(((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)), dtype=np.int64)
    seed = PlanarTriangulation(
        points=points,
        triangles=np.asarray(((0, 1, 5), (1, 4, 5), (1, 2, 4), (2, 3, 4)), dtype=np.int64),
        segments=segments,
        boundary_segments=segments,
        mandatory_segments=np.empty((0, 2), dtype=np.int64),
        outer_loop=np.arange(6, dtype=np.int64),
        hole_loops=(),
    )
    _, report = frontal_delaunay_refine(
        seed,
        NativeMeshingOptions(point_placement="frontal_delaunay", metric_mode="isotropic_spatial",
                             max_insertions=8, max_topology_operations=1),
        target_size=1.,
    )
    assert report["topology_operations"] <= 1
