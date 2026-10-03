"""Whole connected Sheet-joint scope is a prerequisite, not publication."""

import pytest
import numpy as np
from fractions import Fraction

import anygeometry as owner
from anymesher._authored_component_binding import bind_authored_sheet_joint_component
from anymesher._authored_planar_stations import plan_authored_planar_stations
from anymesher.boundary import GlobalEdgeBoundaryRegistry
from anymesher.errors import MeshError
from anymesher.mesh import Mesh
from anymesher.meshing_view import GeometryMeshingView


def prepared(*, explicit_sheets=True):
    model = owner.GeometryModel()
    root = model.add_plate(model.add_points(
        ((0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0))))
    model.set_face_surface(root, owner.Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    cutter = model.add_plate(model.add_points(
        ((3, -1, -1), (3, 5, -1), (3, 5, 1), (3, -1, 1))))
    if explicit_sheets:
        model.add_sheet((root,), name="source root")
        model.add_sheet((cutter,), name="source cutter")
    owner.apply_intersections(
        model, owner.plan_intersections(model, tuple(model.faces), policy="connect"),
        policy="connect",
    )
    correspondences = tuple(
        owner.query_prepared_authored_boundary_correspondence(model, face)
        for face in (root, cutter)
    )
    edge = correspondences[0].interior_incidence[0][0]
    return model, (root, cutter), correspondences, edge


def test_whole_component_and_occurrence_provenance_are_bound_without_publication():
    model, roots, correspondences, edge = prepared()
    before = owner.to_dict(model)
    bound = bind_authored_sheet_joint_component(model, edge, correspondences, roots)
    assert bound.authored_face_ids == roots
    assert len(bound.current_face_ids) == 8
    assert len(bound.sheet_ids) == 2
    assert {row[1] for row in bound.occurrence_correspondence} == set(roots)
    assert [len(row[3]) for row in bound.occurrence_correspondence] == [2, 6]
    assert bound.owner_receipt.occurrence_mapping_qualified is True
    assert bound.owner_receipt.semantic_mapping_qualified is False
    assert bound.publication_qualified is False
    assert owner.to_dict(model) == before


def test_partial_or_extra_root_boundary_refuses():
    model, roots, correspondences, edge = prepared()
    with pytest.raises(owner.GeometryError, match="omits connected authored roots"):
        bind_authored_sheet_joint_component(model, edge, correspondences[:1], roots[:1])
    with pytest.raises(MeshError, match="lacks root boundaries"):
        bind_authored_sheet_joint_component(model, edge, correspondences[:1], roots)
    with pytest.raises(MeshError, match="repeats a root"):
        bind_authored_sheet_joint_component(model, edge, correspondences, roots + roots[:1])


def test_stale_and_source_less_sheet_refuse():
    model, roots, correspondences, edge = prepared()
    model.add_point(20, 20, 20)
    with pytest.raises(owner.GeometryError):
        bind_authored_sheet_joint_component(model, edge, correspondences, roots)
    model, roots, correspondences, edge = prepared(explicit_sheets=False)
    with pytest.raises(owner.GeometryError, match="source-less Sheet"):
        bind_authored_sheet_joint_component(model, edge, correspondences, roots)


def test_shared_registry_stations_cover_both_authored_roots_without_rekeying():
    model, roots, correspondences, edge = prepared()
    bound = bind_authored_sheet_joint_component(model, edge, correspondences, roots)
    mesh = Mesh(nodes={vid: np.asarray(model.vertex_position(vid), dtype=float)
                       for vid in model.vertices})
    registry = GlobalEdgeBoundaryRegistry(GeometryMeshingView(model))
    all_edges = {
        edge_id for correspondence in correspondences
        for loop in correspondence.exterior_loops
        for _source, _forward, current in loop for edge_id in current
    } | {
        edge_id for correspondence in correspondences
        for edge_id, _uses in correspondence.interior_incidence
    }
    for edge_id in sorted(all_edges):
        current = model.edges[edge_id]
        registry.register(edge_id, 0., node_id=current.start)
        registry.register(edge_id, 1., node_id=current.end)
    plans = tuple(plan_authored_planar_stations(model, correspondence, mesh, registry)
                  for correspondence in bound.boundary_correspondences)
    assert tuple(plan.authored_face for plan in plans) == roots
    assert all(plan.publication_qualified is False for plan in plans)
    assert all(any(receipt.edge_id == edge for receipt in plan.interior_receipts)
               for plan in plans)
    assert dict(plans[1].node_material_uv)[11][0] == Fraction(1, 6)
    assert all(len(plan.material_receipts) > 0 for plan in plans)
    assert tuple(entry.node_id for entry in registry.entries(edge)) == (
        model.edges[edge].start, model.edges[edge].end)
