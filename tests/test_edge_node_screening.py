"""Focused tests for vectorized nearest-node screening and its cache (M2)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from anygeometry import GeometryModel
from anygeometry.entities import OrientedEdge
from anygeometry.structural import (
    AttachmentKind,
    AttachmentTargetKind,
    JunctionKind,
    JunctionMemberUse,
    ParameterRange,
)

from anymesher.mesh import Mesh
from anymesher.meshing_view import GeometryMeshingView
from anymesher.structural_pipeline import StructuralMeshingPipeline


class _ScreeningStub:
    """Minimal pipeline stand-in exercising only the screening helpers."""

    _EDGE_SCREEN_MARGIN_ULPS = 64.0
    _edge_node_coordinates = StructuralMeshingPipeline._edge_node_coordinates
    _scalar_edge_distances = staticmethod(
        StructuralMeshingPipeline._scalar_edge_distances
    )
    _nearest_edge_node = StructuralMeshingPipeline._nearest_edge_node

    def __init__(self, cache: dict | None = None) -> None:
        self._edge_coordinate_cache = cache


def _scalar_reference(mesh, candidates, expected):
    distances = [
        float(np.linalg.norm(mesh.nodes[node] - expected)) for node in candidates
    ]
    index = int(np.argmin(distances))
    return index, distances[index]


def test_screening_matches_the_full_scalar_scan_on_random_cases() -> None:
    rng = np.random.default_rng(20261007)
    stub = _ScreeningStub()
    for _ in range(200):
        count = int(rng.integers(1, 12))
        scale = float(rng.choice((1.0e-8, 1.0, 1.0e6)))
        coordinates = rng.normal(size=(count, 3)) * scale
        mesh = SimpleNamespace(
            nodes={index: coordinates[index] for index in range(count)}
        )
        candidates = tuple(range(count))
        expected = rng.normal(size=3) * scale
        index, distance = StructuralMeshingPipeline._nearest_edge_node(
            stub, mesh, 3, expected, candidates
        )
        reference_index, reference_distance = _scalar_reference(
            mesh, candidates, expected
        )
        assert index == reference_index
        assert distance == reference_distance


def test_screening_preserves_original_node_order_ties() -> None:
    mesh = SimpleNamespace(
        nodes={
            1: np.asarray((1.0, 0.0, 0.0)),
            2: np.asarray((1.0, 0.0, 0.0)),
            3: np.asarray((0.0, 0.0, 0.0)),
        }
    )
    expected = np.asarray((1.0, 0.0, 0.0))
    stub = _ScreeningStub()
    index, distance = StructuralMeshingPipeline._nearest_edge_node(
        stub, mesh, 9, expected, (1, 2, 3)
    )
    assert (index, distance) == (0, 0.0)
    index, distance = StructuralMeshingPipeline._nearest_edge_node(
        stub, mesh, 9, expected, (3, 1, 2)
    )
    assert (index, distance) == (1, 0.0)


def test_nonfinite_and_extreme_input_falls_back_to_the_scalar_scan() -> None:
    stub = _ScreeningStub()
    cases = [
        # nonfinite expected point
        (
            {1: np.asarray((0.0, 0.0, 0.0)), 2: np.asarray((1.0, 0.0, 0.0))},
            np.asarray((np.inf, 0.0, 0.0)),
        ),
        # nonfinite candidate coordinates
        (
            {1: np.asarray((0.0, 0.0, 0.0)), 2: np.asarray((np.nan, 1.0, 0.0))},
            np.asarray((0.5, 0.0, 0.0)),
        ),
        # overflow-scale coordinates: the batch norm saturates, the scalar
        # scan reproduces the original behavior exactly
        (
            {1: np.asarray((1.0e200, 0.0, 0.0)), 2: np.asarray((0.0, 0.0, 0.0))},
            np.asarray((0.0, 0.0, 0.0)),
        ),
    ]
    for nodes, expected in cases:
        mesh = SimpleNamespace(nodes=nodes)
        candidates = tuple(nodes)
        index, distance = StructuralMeshingPipeline._nearest_edge_node(
            stub, mesh, 2, expected, candidates
        )
        reference_index, reference_distance = _scalar_reference(
            mesh, candidates, expected
        )
        assert index == reference_index
        assert distance == reference_distance or (
            np.isnan(distance) and np.isnan(reference_distance)
        )


def test_edge_coordinate_cache_reuses_arrays_per_invocation() -> None:
    mesh = SimpleNamespace(
        nodes={
            1: np.asarray((0.0, 0.0, 0.0)),
            2: np.asarray((1.0, 0.0, 0.0)),
        }
    )
    cache: dict = {}
    stub = _ScreeningStub(cache)
    first = stub._edge_node_coordinates(mesh, 7, (1, 2))
    second = stub._edge_node_coordinates(mesh, 7, (1, 2))
    assert first is second
    assert cache[(7, (1, 2))] is first
    other = stub._edge_node_coordinates(mesh, 7, (1,))
    assert other is not first
    # without an invocation cache nothing is retained
    bare = _ScreeningStub(None)
    assert bare._edge_node_coordinates(mesh, 7, (1, 2)) is not None


class _StubView:
    def __init__(self, point, length) -> None:
        self._point = np.asarray(point, dtype=float)
        self._length = float(length)

    def edge_point(self, edge_id, parameter):
        return self._point

    def edge_length(self, edge_id):
        return self._length

    def effective_length(self, value):
        return float(value)


class _StubBoundaries:
    def __init__(self) -> None:
        self.registrations: list[tuple] = []

    def register(self, edge_id, parameter, expected, node_id=None, owner=None):
        self.registrations.append((edge_id, parameter, node_id, owner))


class _EdgeNodeStub(_ScreeningStub):
    _edge_node = StructuralMeshingPipeline._edge_node

    def __init__(self, view, boundaries) -> None:
        super().__init__({})
        self.view = view
        self.boundaries = boundaries


def test_edge_node_tolerance_gate_uses_the_scalar_winner_distance() -> None:
    mesh = SimpleNamespace(
        nodes={
            1: np.asarray((0.0, 0.0, 0.0)),
            2: np.asarray((0.3, 0.0, 0.0)),
        },
        nodes_of_edge={4: [1, 2]},
    )
    # winner node 2 at scalar distance 0.3
    stub = _EdgeNodeStub(_StubView((0.6, 0.0, 0.0), 0.5), _StubBoundaries())
    assert stub._edge_node(mesh, 4, 0.5, "owner") == 2
    assert stub.boundaries.registrations == [(4, 0.5, 2, "owner")]
    # the same winner distance exceeds a tighter tolerance
    stub = _EdgeNodeStub(_StubView((0.6, 0.0, 0.0), 0.1), _StubBoundaries())
    assert stub._edge_node(mesh, 4, 0.5, "owner") is None
    assert stub.boundaries.registrations == []


def test_edge_node_accepts_a_winner_exactly_at_tolerance() -> None:
    mesh = SimpleNamespace(
        nodes={
            1: np.asarray((0.0, 0.0, 0.0)),
            2: np.asarray((0.6, 0.0, 0.0)),
        },
        nodes_of_edge={4: [1, 2]},
    )
    # both candidates sit exactly 0.3 from the expected point; the first
    # in node order wins and 0.3 > 0.3 is False, so it is accepted
    stub = _EdgeNodeStub(_StubView((0.3, 0.0, 0.0), 0.3), _StubBoundaries())
    assert stub._edge_node(mesh, 4, 0.5, "owner") == 1


def _junction_setup():
    geometry = GeometryModel()
    a0, a1, b1 = geometry.add_points(((0, 0, 0), (1, 0, 0), (2, 0, 0)))
    first = geometry.add_member((geometry.add_line(a0, a1),))
    second = geometry.add_member((geometry.add_line(a1, b1),))
    geometry.add_junction(
        JunctionKind.ENDPOINT,
        (
            JunctionMemberUse(first, ParameterRange.point(1.0)),
            JunctionMemberUse(second, ParameterRange.point(0.0)),
        ),
    )
    view = GeometryMeshingView(geometry)
    edges = [
        use.edge_id
        for member in (first, second)
        for use in view.edge_uses_for_member(member)
    ]
    return view, edges


def _junction_mesh(edges) -> Mesh:
    mesh = Mesh()
    mesh.nodes.update(
        {
            1: np.asarray((0.0, 1.0, 0.0)),
            2: np.asarray((1.0, 1.0, 0.0)),
            3: np.asarray((0.0, 0.0, 0.0)),
            4: np.asarray((2.0, 1.0, 0.0)),
            5: np.asarray((0.0, 0.0, 0.0)),
            6: np.asarray((1.0, 0.0, 0.0)),
            7: np.asarray((1.0, 0.0, 0.0)),
            8: np.asarray((2.0, 0.0, 0.0)),
        }
    )
    mesh.quads[10] = (3, 6, 2, 1)
    mesh.beams[11] = (5, 6)
    mesh.beams[12] = (7, 8)
    mesh.nodes_of_edge[edges[0]] = [5, 6]
    mesh.nodes_of_edge[edges[1]] = [7, 8]
    mesh.elements_of_edge[edges[0]] = [11]
    mesh.elements_of_edge[edges[1]] = [12]
    return mesh


def _junction_pipeline(view) -> StructuralMeshingPipeline:
    return StructuralMeshingPipeline(
        view, overlap_policy="connect_declared", mutation_policy="working_copy"
    )


def test_junction_merge_invocation_restores_the_previous_cache() -> None:
    view, edges = _junction_setup()
    pipeline = _junction_pipeline(view)
    previous = {"stale": object()}
    pipeline._edge_coordinate_cache = previous
    report = pipeline.apply_connectivity(_junction_mesh(edges))
    assert not report.issues
    # The junction merge rewrote beam node identifiers mid-invocation; the
    # invocation-owned cache never survives the call: the previous value
    # (normally None) is restored in finally.
    assert pipeline._edge_coordinate_cache is previous


def test_apply_connectivity_does_not_reuse_a_previous_invocation_cache() -> None:
    view, edges = _junction_setup()
    pipeline = _junction_pipeline(view)
    previous = {"stale": object()}
    pipeline._edge_coordinate_cache = previous
    first = pipeline.apply_connectivity(_junction_mesh(edges))
    assert not first.issues
    # Each invocation owns a fresh cache and restores the previous value
    # (normally None) afterwards; nothing survives between invocations.
    assert pipeline._edge_coordinate_cache is previous
    second = pipeline.apply_connectivity(_junction_mesh(edges))
    assert not second.issues
    assert pipeline._edge_coordinate_cache is previous


def _edge_attachment_case() -> tuple[Mesh, StructuralMeshingPipeline]:
    geometry = GeometryModel()
    vertices = geometry.add_points(((0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)))
    face = geometry.add_plate(vertices)
    geometry.add_sheet((face,))
    edge = geometry.faces[face].loop[0].edge
    member = geometry.add_member((OrientedEdge(edge, False),))
    geometry.add_attachment(
        member,
        AttachmentKind.MEMBER_ON_FACE_BOUNDARY,
        AttachmentTargetKind.EDGE,
        edge,
        ParameterRange(0.0, 1.0),
        (ParameterRange(0.0, 1.0),),
    )
    mesh = Mesh()
    for node, vertex in enumerate(vertices, start=1):
        mesh.nodes[node] = geometry.vertex_position(vertex)
    mesh.quads[10] = (1, 2, 3, 4)
    mesh.beams[11] = (1, 2)
    mesh.elements_of_face[face] = [10]
    mesh.elements_of_edge[edge] = [11]
    mesh.nodes_of_edge[edge] = [1, 2]
    pipeline = StructuralMeshingPipeline(
        GeometryMeshingView(geometry),
        overlap_policy="connect_declared",
        mutation_policy="working_copy",
    )
    return mesh, pipeline


def test_apply_connectivity_restores_the_edge_cache_on_success() -> None:
    mesh, pipeline = _edge_attachment_case()
    report = pipeline.apply_connectivity(mesh)
    assert not report.issues
    assert pipeline._edge_coordinate_cache is None
    # A direct private station call after a completed invocation must use
    # current coordinates, never anything retained by the invocation.
    edge = next(iter(mesh.nodes_of_edge))
    mesh.nodes[1] = np.asarray((100.0, 0.0, 0.0))
    mesh.nodes[2] = np.asarray((1.0, 0.0, 0.0))
    assert pipeline._edge_node(mesh, edge, 0.5, ("test", "after")) == 2


def test_failed_apply_connectivity_cleans_up_the_edge_cache(monkeypatch) -> None:
    mesh, pipeline = _edge_attachment_case()
    original = StructuralMeshingPipeline._add_attachment_coupling

    def failing(self, mesh, bvh, attachment, member_parameter, beam_node,
                next_record, index):
        original(self, mesh, bvh, attachment, member_parameter, beam_node,
                next_record, index)
        raise RuntimeError("injected connectivity failure")

    monkeypatch.setattr(
        StructuralMeshingPipeline, "_add_attachment_coupling", failing
    )
    with pytest.raises(RuntimeError, match="injected connectivity failure"):
        pipeline.apply_connectivity(mesh)
    assert pipeline._edge_coordinate_cache is None
    # Direct subsequent private station calls must not reuse coordinates
    # cached by the failed invocation.
    edge = next(iter(mesh.nodes_of_edge))
    mesh.nodes[1] = np.asarray((100.0, 0.0, 0.0))
    mesh.nodes[2] = np.asarray((1.0, 0.0, 0.0))
    assert pipeline._edge_node(mesh, edge, 0.5, ("test", "after-failure")) == 2
