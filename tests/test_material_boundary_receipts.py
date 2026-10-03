"""Exact owner station identities do not grant curved refinement permission."""
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from anygeometry import GeometryError, GeometryModel, query_material_surface_regions, to_dict
from anymesher._material_region_binding import MaterialRegionBinding
from anymesher.boundary import GlobalEdgeBoundaryRegistry
from anymesher.errors import MeshError
from anymesher.meshing_view import GeometryMeshingView


def owner_boundary():
    model = GeometryModel()
    points = model.add_points(((0, 0, 0), (.5, -.4, 0), (1, 0, 0)))
    curve = model.add_spline(points[0], points[1:-1], points[-1])
    face = model.extrude((curve,), (0, 0, 1))[0]
    collection = query_material_surface_regions(model, (face,))
    assert len(collection.regions) == 1
    binding = MaterialRegionBinding(model, collection, collection.regions[0], face)
    registry = GlobalEdgeBoundaryRegistry(GeometryMeshingView(model))
    mesh = SimpleNamespace(nodes={}, nodes_of_edge={})
    for path in binding.region.boundaries[0]:
        edge = model.edges[path.source_edge]
        nodes = (edge.start, edge.end)
        mesh.nodes_of_edge[edge.id] = list(nodes)
        for parameter, node in zip((0., 1.), nodes):
            mesh.nodes[node] = tuple(model.vertex_position(node))
            registry.register(edge.id, parameter, node_id=node)
    return model, binding, mesh, registry, curve


def test_protected_boundary_identity_is_retained_without_split_permission():
    model, binding, mesh, registry, curve = owner_boundary()
    original = to_dict(model)
    old_loops = binding.loops(mesh, registry, ())
    assert all(spec is None for _, _, specs in old_loops for spec in specs)
    receipts = binding.boundary_intervals(mesh, registry)
    assert len(receipts) == 4
    receipt = next(r for r in receipts if r.source_edge == curve)
    uv, xyz = binding.boundary_station(receipt, mesh, registry, Fraction(1, 2))
    np.testing.assert_array_equal(xyz, (.5, -.2, 0.))
    np.testing.assert_allclose(binding.region.support.evaluate(*uv), xyz, atol=1e-14)
    assert not hasattr(receipt, 'splittable')
    assert not hasattr(receipt, 'permission')
    assert registry.lookup(curve, .5) is None
    assert to_dict(model) == original


def test_reversed_occurrence_uses_original_source_parameter():
    _, binding, mesh, registry, _ = owner_boundary()
    receipts = binding.boundary_intervals(mesh, registry)
    receipt = next(r for r in receipts if r.parameters[0] > r.parameters[1])
    parameter = Fraction(1, 4)
    uv, xyz = binding.boundary_station(receipt, mesh, registry, parameter)
    np.testing.assert_array_equal(xyz, binding.geometry.sample_edge(receipt.source_edge, np.asarray((.25,)))[0])
    np.testing.assert_allclose(binding.region.support.evaluate(*uv), xyz, atol=1e-14)


@pytest.mark.parametrize('parameter', (0., .5, Fraction(0), Fraction(1), Fraction(1, 3)))
def test_station_refuses_rounded_or_noninterior_parameters(parameter):
    _, binding, mesh, registry, curve = owner_boundary()
    receipt = next(r for r in binding.boundary_intervals(mesh, registry) if r.source_edge == curve)
    with pytest.raises(MeshError):
        binding.boundary_station(receipt, mesh, registry, parameter)


def test_interval_receipt_invalidates_when_registry_chain_changes():
    model, binding, mesh, registry, curve = owner_boundary()
    receipt = next(r for r in binding.boundary_intervals(mesh, registry) if r.source_edge == curve)
    point = tuple(model.sample_edge(curve, np.asarray((.25,)))[0])
    node = max(mesh.nodes)+1
    registry.register(curve, .25, node_id=node)
    mesh.nodes[node] = point
    mesh.nodes_of_edge[curve].insert(1, node)
    with pytest.raises(MeshError, match='no longer current'):
        binding.boundary_station(receipt, mesh, registry, Fraction(1, 2))


def test_receipts_refuse_node_bound_to_multiple_source_stations():
    _, binding, mesh, registry, curve = owner_boundary()
    node = mesh.nodes_of_edge[curve][0]
    registry.register(curve, 2.**-40, node_id=node)
    with pytest.raises(MeshError, match='multiple source stations'):
        binding.boundary_intervals(mesh, registry)


def test_unrepresented_registered_station_invalidates_receipt():
    _, binding, mesh, registry, curve = owner_boundary()
    receipt = next(r for r in binding.boundary_intervals(mesh, registry) if r.source_edge == curve)
    registry.register(curve, .25, node_id=max(mesh.nodes)+1)
    with pytest.raises(MeshError, match='omitted a registered'):
        binding.boundary_station(receipt, mesh, registry, Fraction(1, 2))


def test_unvalidated_region_cannot_issue_boundary_receipts():
    _, binding, mesh, registry, _ = owner_boundary()
    unvalidated = replace(binding, region=replace(binding.region))
    with pytest.raises(MeshError, match='validated owner collection'):
        unvalidated.boundary_intervals(mesh, registry)


@pytest.mark.parametrize('field,value', (('loop_index', -1), ('path_index', True),
                                      ('interval_index', .5), ('source_edge', 999)))
def test_receipt_does_not_discover_another_occurrence(field, value):
    _, binding, mesh, registry, _ = owner_boundary()
    receipt = binding.boundary_intervals(mesh, registry)[0]
    with pytest.raises(MeshError):
        binding.boundary_station(replace(receipt, **{field: value}), mesh, registry, Fraction(1, 2))


def test_receipt_cannot_transfer_between_binding_objects():
    _, binding, mesh, registry, _ = owner_boundary()
    receipt = binding.boundary_intervals(mesh, registry)[0]
    other = replace(binding)
    with pytest.raises(MeshError, match='source receipt'):
        other.boundary_station(receipt, mesh, registry, Fraction(1, 2))


def test_owner_revision_change_refuses_stale_receipt():
    model, binding, mesh, registry, _ = owner_boundary()
    receipt = binding.boundary_intervals(mesh, registry)[0]
    model.add_point(9, 9, 9)
    with pytest.raises(GeometryError, match='revision|stale|changed'):
        binding.boundary_station(receipt, mesh, registry, Fraction(1, 2))


def test_cancellation_after_reading_station_leaves_all_inputs_unchanged():
    model, binding, mesh, registry, _ = owner_boundary()
    receipt = binding.boundary_intervals(mesh, registry)[0]
    before = to_dict(model), dict(mesh.nodes), dict(mesh.nodes_of_edge), registry.entries()
    class Cancelled(Exception):
        pass
    def cancel(stage):
        if stage == 'material region membership':
            raise Cancelled
    with pytest.raises(Cancelled):
        binding.boundary_station(receipt, mesh, registry, Fraction(1, 2), cancel)
    assert to_dict(model) == before[0]
    assert mesh.nodes == before[1]
    assert mesh.nodes_of_edge == before[2]
    assert registry.entries() == before[3]
