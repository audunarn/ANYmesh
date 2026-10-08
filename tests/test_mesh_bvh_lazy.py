"""Focused tests for eager shell-input validation and lazy BVH builds (M1)."""

from __future__ import annotations

import numpy as np
import pytest
from anygeometry import GeometryModel
from anygeometry.entities import OrientedEdge
from anygeometry.structural import (
    AttachmentKind,
    AttachmentTargetKind,
    ParameterRange,
)

from anymesher._runtime_counters import (
    MESHER_BVH_BUILDS,
    MESHER_BVH_LOOKUPS,
    operation_counts_scope,
)
from anymesher.errors import MeshError
from anymesher.mesh import Mesh
from anymesher.mesh_bvh import MeshElementBVH
from anymesher.meshing_view import GeometryMeshingView
from anymesher.structural_pipeline import StructuralMeshingPipeline


def _quad_mesh() -> Mesh:
    mesh = Mesh()
    mesh.nodes.update(
        {
            1: np.asarray((0.0, 0.0, 0.0)),
            2: np.asarray((1.0, 0.0, 0.0)),
            3: np.asarray((1.0, 1.0, 0.0)),
            4: np.asarray((0.0, 1.0, 0.0)),
            5: np.asarray((2.0, 0.0, 0.0)),
            6: np.asarray((2.0, 1.0, 0.0)),
        }
    )
    mesh.quads[10] = (1, 2, 3, 4)
    mesh.quads[11] = (2, 5, 6, 3)
    return mesh


def test_constructor_rejects_duplicate_shell_ids_without_building() -> None:
    mesh = _quad_mesh()
    mesh.tris[10] = (1, 2, 3)
    with pytest.raises(MeshError, match="both quad and triangle"):
        MeshElementBVH(mesh)


def test_constructor_rejects_missing_node_eagerly() -> None:
    mesh = _quad_mesh()
    mesh.quads[12] = (1, 2, 3, 99)
    with pytest.raises(MeshError, match="references missing node 99"):
        MeshElementBVH(mesh)


def test_constructor_rejects_unsupported_connectivity_length() -> None:
    mesh = _quad_mesh()
    mesh.quads[12] = (1, 2, 3)
    with pytest.raises(MeshError, match="unsupported connectivity length 3"):
        MeshElementBVH(mesh)


def test_constructor_rejects_invalid_tolerance_and_leaf_size() -> None:
    mesh = _quad_mesh()
    with pytest.raises(MeshError, match="tolerance"):
        MeshElementBVH(mesh, tolerance=-1.0)
    with pytest.raises(MeshError, match="leaf size"):
        MeshElementBVH(mesh, leaf_size=0)


def test_public_constructor_builds_eagerly_and_listing_adds_no_lookups() -> None:
    mesh = _quad_mesh()
    counts: dict[str, int] = {}
    with operation_counts_scope(counts):
        bvh = MeshElementBVH(mesh, tolerance=1.0e-9)
        assert counts == {MESHER_BVH_BUILDS: 1}
        assert bvh.element_ids == (10, 11)
        assert bvh.active_elements == frozenset({10, 11})
        bvh.set_active((10,), False)
        assert bvh.active_elements == frozenset({11})
        bvh.replace_active((10, 11))
        assert bvh.active_elements == frozenset({10, 11})
    assert counts == {MESHER_BVH_BUILDS: 1}


def test_connectivity_only_listing_never_builds_the_tree() -> None:
    mesh = _quad_mesh()
    counts: dict[str, int] = {}
    with operation_counts_scope(counts):
        bvh = MeshElementBVH._connectivity_only(mesh, tolerance=1.0e-9)
        assert bvh.element_ids == (10, 11)
        assert bvh.active_elements == frozenset({10, 11})
        bvh.set_active((10,), False)
        assert bvh.active_elements == frozenset({11})
        bvh.replace_active((10, 11))
        assert bvh.active_elements == frozenset({10, 11})
    assert counts == {}


def test_selected_element_ids_stay_available_before_any_build() -> None:
    mesh = _quad_mesh()
    bvh = MeshElementBVH._connectivity_only(mesh, element_ids=(10,), tolerance=1.0e-9)
    assert bvh.element_ids == (10,)
    counts: dict[str, int] = {}
    with operation_counts_scope(counts):
        assert bvh.locate((1.5, 0.5, 0.0)) is None
    assert counts == {MESHER_BVH_BUILDS: 1, MESHER_BVH_LOOKUPS: 1}


def test_first_locate_builds_once_and_counts_each_public_lookup() -> None:
    mesh = _quad_mesh()
    bvh = MeshElementBVH._connectivity_only(mesh, tolerance=1.0e-9)
    counts: dict[str, int] = {}
    with operation_counts_scope(counts):
        hit = bvh.locate((0.5, 0.5, 0.0))
        assert hit is not None and hit.element_id == 10
        assert counts == {MESHER_BVH_BUILDS: 1, MESHER_BVH_LOOKUPS: 1}
        assert bvh.locate((1.5, 0.5, 0.0)).element_id == 11  # type: ignore[union-attr]
        assert counts[MESHER_BVH_BUILDS] == 1
        assert counts[MESHER_BVH_LOOKUPS] == 2
        # Bounds queries stay coarse: one leaf holds both elements, so any
        # intersecting query returns both; a disjoint query returns none.
        assert bvh.candidates((0.5, 0.5, 0.0)) == (10, 11)
        assert bvh.query_bounds((10.0, 10.0, 10.0), (11.0, 11.0, 11.0)) == ()
        assert bvh.locate_all((0.5, 0.5, 0.0))
        assert bvh.locate((5.0, 5.0, 5.0)) is None
    assert counts[MESHER_BVH_BUILDS] == 1
    assert counts[MESHER_BVH_LOOKUPS] == 6


def test_locate_results_match_the_eager_contract() -> None:
    mesh = _quad_mesh()
    mesh.tris[12] = (1, 2, 4)
    bvh = MeshElementBVH(mesh, tolerance=1.0e-9)
    hit = bvh.locate((0.25, 0.25, 0.0))
    assert hit is not None
    assert hit.element_id in (10, 12)
    assert hit.node_ids == (1, 2, 3, 4) or hit.node_ids == (1, 2, 4)
    assert abs(hit.residual) <= 1.0e-9
    inside = bvh.locate((0.5, 0.5, 0.5), tolerance=1.0e-12)
    assert inside is None


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


def test_apply_connectivity_without_point_location_never_builds() -> None:
    mesh, pipeline = _edge_attachment_case()
    counts: dict[str, int] = {}
    with operation_counts_scope(counts):
        report = pipeline.apply_connectivity(mesh)
    assert not report.issues
    assert counts == {}


def _face_attachment_case() -> tuple[Mesh, StructuralMeshingPipeline]:
    geometry = GeometryModel()
    vertices = geometry.add_points(((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)))
    face = geometry.add_plate(vertices)
    geometry.add_sheet((face,))
    p0, p1 = geometry.add_points(((0.5, 0.5, 1.0), (1.5, 0.5, 1.0)))
    member = geometry.add_member((geometry.add_line(p0, p1),))
    geometry.add_attachment(
        member,
        AttachmentKind.MEMBER_ON_FACE,
        AttachmentTargetKind.FACE,
        face,
        ParameterRange.point(0.5),
        (ParameterRange(0.25, 0.75), ParameterRange(0.25, 0.25)),
    )
    mesh = Mesh()
    mesh.nodes.update(
        {
            1: np.asarray((0.0, 0.0, 0.0)),
            2: np.asarray((2.0, 0.0, 0.0)),
            3: np.asarray((2.0, 2.0, 0.0)),
            4: np.asarray((0.0, 2.0, 0.0)),
            5: np.asarray((0.5, 0.5, 1.0)),
            6: np.asarray((1.5, 0.5, 1.0)),
            7: np.asarray((1.0, 0.5, 1.0)),
        }
    )
    mesh.quads[10] = (1, 2, 3, 4)
    view = GeometryMeshingView(geometry)
    edge = view.edge_uses_for_member(member)[0].edge_id
    mesh.beams[11] = (5, 7, 6)
    mesh.nodes_of_edge[edge] = [5, 6, 7]
    mesh.elements_of_edge[edge] = [11]
    mesh.elements_of_face[face] = [10]
    pipeline = StructuralMeshingPipeline(
        view,
        overlap_policy="connect_declared",
        mutation_policy="working_copy",
    )
    return mesh, pipeline


def test_apply_connectivity_builds_once_for_a_needed_face_location() -> None:
    mesh, pipeline = _face_attachment_case()
    counts: dict[str, int] = {}
    with operation_counts_scope(counts):
        report = pipeline.apply_connectivity(mesh)
    assert not report.issues
    assert len(mesh.couplings) == 1
    coupling = next(iter(mesh.couplings.values()))
    assert coupling.beam_node == 7
    assert tuple(coupling.plate_nodes) == (1, 2, 3, 4)
    assert counts == {MESHER_BVH_BUILDS: 1, MESHER_BVH_LOOKUPS: 1}


def test_public_constructor_snapshots_coordinates_at_construction() -> None:
    mesh = _quad_mesh()
    bvh = MeshElementBVH(mesh, tolerance=1.0e-9)
    # Mutating mesh coordinates after public construction must not change
    # the coordinate snapshot the index serves (the historical contract).
    mesh.nodes[1] = np.asarray((50.0, 50.0, 50.0))
    hit = bvh.locate((0.5, 0.5, 0.0))
    assert hit is not None and hit.element_id == 10
    assert hit.node_ids == (1, 2, 3, 4)
    assert bvh.locate((50.0, 50.0, 50.0)) is None


def test_public_constructor_rejects_malformed_coordinates_eagerly() -> None:
    mesh = _quad_mesh()
    mesh.nodes[1] = "not-a-coordinate"  # type: ignore[assignment]
    with pytest.raises(ValueError):
        MeshElementBVH(mesh)


def test_connectivity_only_rejects_malformed_coordinates_without_lookup() -> None:
    mesh = _quad_mesh()
    mesh.nodes[1] = "not-a-coordinate"  # type: ignore[assignment]
    # The private lazy path keeps the historical eager rejection of
    # non-numeric node coordinates, before any locating operation.
    with pytest.raises(ValueError):
        MeshElementBVH._connectivity_only(mesh)


def test_connectivity_only_rejects_ragged_coordinates_without_lookup() -> None:
    mesh = _quad_mesh()
    mesh.nodes[1] = np.asarray((0.0, 0.0))  # ragged against the other rows
    with pytest.raises(ValueError):
        MeshElementBVH._connectivity_only(mesh)
    with pytest.raises(ValueError):
        MeshElementBVH(mesh)


def test_nan_coordinates_stay_accepted_without_new_rejection() -> None:
    mesh = _quad_mesh()
    mesh.nodes[1] = np.asarray((np.nan, 0.0, 0.0))
    public = MeshElementBVH(mesh)
    private = MeshElementBVH._connectivity_only(mesh)
    assert public.element_ids == private.element_ids == (10, 11)
