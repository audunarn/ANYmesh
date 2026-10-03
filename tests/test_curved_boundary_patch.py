"""Fixture-first off-chord boundary staging; no automatic route/publication."""
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace
import json

import numpy as np
import pytest

from anygeometry import GeometryModel, Plane, OrientedEdge, query_material_surface_regions, to_dict, GeometryError
from anymesher.boundary import GlobalEdgeBoundaryRegistry
from anymesher.meshing_view import GeometryMeshingView
from anymesher._material_region_binding import MaterialRegionBinding
from anymesher._curved_boundary_patch import stage_curved_boundary_patch
from anymesher.native_v2 import MutableT3Topology, ComponentSeedRegistry, NativeMeshingOptions, _GeometryLimited
from anymesher.errors import MeshError
from anymesher.surface_mesh import SurfaceMeshOptions


def fixture(*, shape='proof', operations=2, insertions=2, hole=False):
    if shape == 'proof':
        xy = [(0., 0.), (1., 0.), (.5, -.8), (48/13, 20/13)]
        control = (64/65, 122/65)
        cells = [(0, 1, 3), (0, 2, 1)]
        loop = [0, 2, 1, 3]
    else:
        # Material lies above the inward-bowed lower boundary, below BI.
        xy = [(0., 0.), (0., 2.), (2., 0.)] if shape == 'outside' else [(0., 0.), (.5, 1.), (2., 0.)]
        control = (1., 1.) if shape == 'outside' else (-10., -1.)
        cells = [(0, 2, 1)]
        loop = [0, 2, 1]
    model = GeometryModel()
    vertices = model.add_points([(*p, 0.) for p in (*xy, control)])
    curve_ends = (0, 3) if shape == 'proof' else (0, 2)
    edge_ids = []
    for a, b in zip(loop, loop[1:]+loop[:1]):
        if {a, b} == set(curve_ends):
            curved = model.add_spline(vertices[a], [vertices[-1]], vertices[b])
            edge_ids.append(curved)
        else:
            edge_ids.append(model.add_line(vertices[a], vertices[b]))
    face = model.add_face(edge_ids, surface=Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
    if hole:
        # A void wholly inside the expanded child, beyond the old chord. Define
        # it before querying the immutable owner domain; no mesher classifier.
        corners = model.add_points(((.96, .76, 0.), (1., .76, 0.), (1., .8, 0.), (.96, .8, 0.)))
        hole_edges = model.add_polyline(corners, close=True)
        with model.transaction():
            # Fixture definition only: query the resulting domain through the
            # public owner qualification before supplying any mesher receipt.
            model._put_entity('face', replace(model.faces[face], holes=(tuple(OrientedEdge(edge, True) for edge in hole_edges),)))
        edge_ids.extend(hole_edges)
    collection = query_material_surface_regions(model, (face,))
    binding = MaterialRegionBinding(model, collection, collection.regions[0], face)
    mesh = SimpleNamespace(nodes={node: np.asarray(model.vertex_position(node)).copy() for node in model.vertices},
        nodes_of_edge={edge: [model.edges[edge].start, model.edges[edge].end] for edge in edge_ids})
    registry = GlobalEdgeBoundaryRegistry(GeometryMeshingView(model))
    for edge, nodes in mesh.nodes_of_edge.items():
        for parameter, node in zip((0., 1.), nodes):
            registry.register(edge, parameter, mesh.nodes[node], node_id=node)
    receipt = next(item for item in binding.boundary_intervals(mesh, registry) if item.source_edge == curved)
    row_map = {node: row for row, node in enumerate(vertices[:-1])}
    boundary = [tuple(sorted((a, b))) for a, b in zip(loop, loop[1:]+loop[:1])]
    if shape == 'proof':
        boundary.append((0, 1))  # Fixed internal OI; never subdivide it.
    topology = MutableT3Topology(np.array(xy), np.array(cells), boundary,
        node_owners=np.arange(len(xy))+10, triangle_owners=[17, 23] if shape == 'proof' else [17],
        seed_registry=ComponentSeedRegistry(100))
    topology.quality_cache[tuple(cells[0])] = (1.,)
    settings = SurfaceMeshOptions(min_angle=15., native_options=NativeMeshingOptions(
        max_topology_operations=operations, max_insertions=insertions))
    return topology, receipt, mesh, registry, row_map, settings, model


def state(topology, mesh, registry, model):
    return (topology.points.tobytes(), topology.triangles.tobytes(), topology.node_owners.tobytes(),
        topology.triangle_owners.tobytes(), topology.constraint_edges.tobytes(), topology.epoch,
        dict(topology.quality_cache), id(topology._topology_index), dict(topology.shared_node_ids),
        dict(topology._seed_registry._values), topology._seed_registry._next,
        {key: np.asarray(value).tobytes() for key, value in mesh.nodes.items()},
        {key: tuple(value) for key, value in mesh.nodes_of_edge.items()},
        tuple((entry.key, entry.node_id, entry.point.tobytes(), entry.owners) for entry in registry.entries()),
        to_dict(model))


def stage(data, work=None, **kwargs):
    topology, receipt, mesh, registry, mapping, settings, _ = data
    work = dict(topology_operations=0, insertions=0, reserved_node_reuses=0) if work is None else work
    return stage_curved_boundary_patch(topology, receipt, mesh, registry, Fraction(1, 2),
        node_to_row=mapping, settings=settings, work=work, **kwargs)


def area(points, cells):
    result = Fraction()
    for cell in cells:
        a, b, c = [[Fraction(float(x)) for x in points[row]] for row in cell]
        determinant = (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        assert determinant > 0
        result += determinant/2
    return result


def test_actual_rational_patch_progress_owner_domain_and_exact_source_nonmutation():
    data = fixture()
    topology, receipt, mesh, registry, _, _, model = data
    before = state(topology, mesh, registry, model)
    material_area = receipt.binding.region.material_area
    work = dict(topology_operations=0, insertions=0, reserved_node_reuses=0)
    patch = stage(data, work)
    assert work == dict(topology_operations=1, insertions=1, reserved_node_reuses=0)
    assert state(topology, mesh, registry, model) == before
    assert patch.topology.points[:4].tobytes() == topology.points.tobytes()
    assert patch.topology.node_owners[:4].tobytes() == topology.node_owners.tobytes()
    assert patch.topology.triangle_owners.tolist().count(17) == 2
    assert patch.topology.triangle_owners.tolist().count(23) == 1
    assert (0, 1) in patch.topology.protected_edges and (0, 1) in patch.topology._topology_index
    assert not patch.topology._seed_registry.assigned_node_ids
    assert patch.parameter == Fraction(1, 2) and patch.local_node == 4
    endpoints = tuple(data[4][node] for node in receipt.nodes)
    assert patch.child_intervals == (((endpoints[0], 4), (receipt.source_edge, receipt.parameters[0], Fraction(1, 2))),
        ((4, endpoints[1]), (receipt.source_edge, Fraction(1, 2), receipt.parameters[1])))
    # The authoritative evaluator retains its own binary64 grouping; a rational
    # formula rounded independently is not a station-identity oracle.
    np.testing.assert_array_equal(patch.xyz[-1], model.sample_edge(receipt.source_edge, [.5])[0])
    assert patch.physical_progress and not patch.publication_qualified
    assert patch.candidate.report['violation_counts'] == dict.fromkeys(patch.baseline.report['violation_counts'], 0)
    assert patch.baseline.report['elements_above_maximum_growth'] == patch.candidate.report['elements_above_maximum_growth'] == 2
    assert patch.baseline.report['quality_violation_count'] == patch.candidate.report['quality_violation_count'] == 2
    assert patch.candidate.report['max_element_growth'] < patch.baseline.report['max_element_growth']
    assert patch.candidate.report['poor_element_ids']  # Progress is not final acceptance.
    assert area(topology.points, topology.triangles) != area(patch.topology.points, patch.topology.triangles)
    assert receipt.binding.region.material_area == material_area
    print('CURVED_PATCH_PROOF', json.dumps({name: {key: getattr(patch, name).report[key]
        for key in ('violation_counts', 'quality_violation_count', 'elements_above_maximum_growth', 'max_element_growth')}
        for name in ('baseline', 'candidate')}, sort_keys=True))


def test_exact_rational_curve_polygon_error_changes_without_changing_domain():
    old = Fraction(76, 65)
    new = Fraction(164, 65)
    curved_domain = Fraction(116, 39)
    assert curved_domain-old == Fraction(352, 195)
    assert curved_domain-new == Fraction(88, 195)
    data = fixture()
    patch = stage(data)
    assert area(data[0].points, data[0].triangles) == pytest.approx(float(old), abs=1e-14)
    assert area(patch.topology.points, patch.topology.triangles) == pytest.approx(float(new), abs=1e-14)
    assert data[-1].faces and data[1].binding.region.material_area == pytest.approx(float(curved_domain), abs=1e-12)


def test_existing_collinear_api_still_refuses_authoritative_off_chord_point():
    data = fixture()
    topology, receipt, mesh, registry, mapping, _, model = data
    edge = tuple(sorted(mapping[node] for node in receipt.nodes))
    # The old API requires a declared straight interval and a collinear point.
    old = MutableT3Topology(topology.points, topology.triangles,
        tuple(topology.protected_edges-{edge}), splittable_edges={edge: (receipt.source_edge, 0, 1)},
        seed_registry=topology._seed_registry)
    before = state(old, mesh, registry, model)
    uv, _ = receipt.binding.boundary_station(receipt, mesh, registry, Fraction(1, 2))
    with pytest.raises(MeshError, match='strictly on the segment'):
        old.split_segment(edge, _split_proposal=lambda *_: (Fraction(1, 2), uv, None))
    assert state(old, mesh, registry, model) == before


def test_off_curve_coordinate_midpoint_is_not_owner_station():
    data = fixture()
    before = state(data[0], data[2], data[3], data[-1])
    chord = np.mean(data[1].xyz, axis=0)
    work = dict(topology_operations=0, insertions=0, reserved_node_reuses=0)
    with pytest.raises(MeshError, match='exact owner station'):
        stage(data, work, station_xyz=chord)
    assert work == dict(topology_operations=1, insertions=1, reserved_node_reuses=0)
    assert state(data[0], data[2], data[3], data[-1]) == before


def test_owner_exact_side_partition_rejects_outside_material_children():
    data = fixture(shape='outside')
    before = state(data[0], data[2], data[3], data[-1])
    work = dict(topology_operations=0, insertions=0, reserved_node_reuses=0)
    with pytest.raises(_GeometryLimited, match='outside owner material'):
        stage(data, work)
    assert work == dict(topology_operations=1, insertions=1, reserved_node_reuses=0)
    assert state(data[0], data[2], data[3], data[-1]) == before


def test_valid_owner_curve_station_that_inverts_a_child_is_refused():
    data = fixture(shape='inverted')
    before = state(data[0], data[2], data[3], data[-1])
    work = dict(topology_operations=0, insertions=0, reserved_node_reuses=0)
    with pytest.raises(_GeometryLimited, match='invalidate parent orientation'):
        stage(data, work)
    assert work == dict(topology_operations=1, insertions=1, reserved_node_reuses=0)
    assert state(data[0], data[2], data[3], data[-1]) == before


def test_expanded_children_cannot_overlap_an_unchanged_owner_material_cell():
    data = fixture()
    original, receipt, mesh, registry, mapping, settings, model = data
    extra = np.array([(1., .8), (1.1, .8), (1., .9)])
    points = np.vstack((original.points, extra))
    cells = np.vstack((original.triangles, (4, 5, 6)))
    topology = MutableT3Topology(points, cells,
        tuple(original.protected_edges) + ((4, 5), (5, 6), (4, 6)),
        seed_registry=original._seed_registry, triangle_owners=[17, 23, 41])
    for node, row in zip((101, 102, 103), range(4, 7)):
        mesh.nodes[node] = np.append(points[row], 0.)
        mapping[node] = row
    changed = topology, receipt, mesh, registry, mapping, settings, model
    before = state(topology, mesh, registry, model)
    work = dict(topology_operations=0, insertions=0, reserved_node_reuses=0)
    with pytest.raises(_GeometryLimited, match='overlaps unchanged material cells'):
        stage(changed, work)
    assert work == dict(topology_operations=1, insertions=1, reserved_node_reuses=0)
    assert state(topology, mesh, registry, model) == before


def test_owner_void_wholly_inside_expanded_child_cannot_hide_from_side_checks():
    data = fixture(hole=True)
    before = state(data[0], data[2], data[3], data[-1])
    work = dict(topology_operations=0, insertions=0, reserved_node_reuses=0)
    with pytest.raises(_GeometryLimited, match='encloses owner void'):
        stage(data, work)
    assert work == dict(topology_operations=1, insertions=1, reserved_node_reuses=0)
    assert state(data[0], data[2], data[3], data[-1]) == before


@pytest.mark.parametrize('invalid', ['nonfinite', 'inverted'])
def test_invalid_source_topology_is_not_repaired_or_published(invalid):
    data = fixture()
    if invalid == 'nonfinite':
        data[0]._points[1, 0] = np.nan
    else:
        data[0]._triangles[0] = data[0]._triangles[0][::-1]
    before = state(data[0], data[2], data[3], data[-1])
    work = dict(topology_operations=0, insertions=0, reserved_node_reuses=0)
    with pytest.raises(MeshError):
        stage(data, work)
    assert work == dict(topology_operations=1, insertions=1, reserved_node_reuses=0)
    assert state(data[0], data[2], data[3], data[-1]) == before


def test_nonplanar_whole_child_coverage_is_explicitly_unqualified():
    from test_material_boundary_receipts import owner_boundary
    model, binding, mesh, registry, edge = owner_boundary()
    receipt = next(item for item in binding.boundary_intervals(mesh, registry) if item.source_edge == edge)
    topology = MutableT3Topology(((0., 0.), (1., 0.), (0., 1.)), ((0, 1, 2),),
        ((0, 1), (1, 2), (0, 2)), seed_registry=ComponentSeedRegistry(100))
    before = state(topology, mesh, registry, model)
    work = dict(topology_operations=0, insertions=0, reserved_node_reuses=0)
    with pytest.raises(_GeometryLimited, match='qualified Plane support'):
        stage_curved_boundary_patch(topology, receipt, mesh, registry, Fraction(1, 2),
            node_to_row={}, settings=SurfaceMeshOptions(min_angle=15.), work=work)
    assert work == dict(topology_operations=1, insertions=1, reserved_node_reuses=0)
    assert state(topology, mesh, registry, model) == before


@pytest.mark.parametrize('change', ['missing', 'ambiguous', 'noninteger'])
def test_known_source_endpoint_identity_is_required_without_coordinate_search(change):
    data = fixture()
    mapping = dict(data[4])
    if change == 'missing':
        del mapping[data[1].nodes[0]]
    elif change == 'ambiguous':
        mapping[data[1].nodes[0]] = mapping[data[1].nodes[1]]
    else:
        mapping[data[1].nodes[0]] = .5
    changed = (*data[:4], mapping, *data[5:])
    before = state(data[0], data[2], data[3], data[-1])
    work = dict(topology_operations=0, insertions=0, reserved_node_reuses=0)
    with pytest.raises(MeshError):
        stage(changed, work)
    assert work == dict(topology_operations=1, insertions=1, reserved_node_reuses=0)
    assert state(data[0], data[2], data[3], data[-1]) == before


@pytest.mark.parametrize('missing', ['topology_operations', 'insertions', 'reserved_node_reuses'])
def test_missing_original_work_field_cannot_start_a_new_pool(missing):
    data = fixture()
    before = state(data[0], data[2], data[3], data[-1])
    work = dict(topology_operations=0, insertions=0, reserved_node_reuses=0)
    del work[missing]
    saved = dict(work)
    with pytest.raises(MeshError, match='every original work receipt field'):
        stage(data, work)
    assert work == saved and state(data[0], data[2], data[3], data[-1]) == before


@pytest.mark.parametrize('value', [True, -1, 0.5, float('nan'), None])
def test_invalid_original_work_refuses_before_any_debit(value):
    data = fixture()
    before = state(data[0], data[2], data[3], data[-1])
    work = dict(topology_operations=value, insertions=0, reserved_node_reuses=0)
    with pytest.raises(MeshError, match='invalid curved boundary work receipt'):
        stage(data, work)
    assert work['topology_operations'] is value
    assert work['insertions'] == work['reserved_node_reuses'] == 0
    assert state(data[0], data[2], data[3], data[-1]) == before


@pytest.mark.parametrize('phase', ['curved boundary staging start', 'curved boundary owner domain roots',
    'curved boundary owner side interval', 'curved boundary staging final binding'])
@pytest.mark.parametrize('error_type', [RuntimeError, _GeometryLimited])
def test_cancellation_identity_no_partial_source_change_and_no_work_refund(phase, error_type):
    data = fixture()
    before = state(data[0], data[2], data[3], data[-1])
    error = error_type('cancel detached curved patch')
    work = dict(topology_operations=0, insertions=0, reserved_node_reuses=0)
    def cancel(where):
        if where == phase:
            raise error
    with pytest.raises(error_type) as caught:
        stage(data, work, cancellation_check=cancel)
    assert caught.value is error and work == dict(topology_operations=1, insertions=1, reserved_node_reuses=0)
    assert state(data[0], data[2], data[3], data[-1]) == before


@pytest.mark.parametrize('field', ['source_edge', 'nodes', 'uv', 'parameters'])
def test_tampered_source_receipt_refuses_without_mutation(field):
    data = fixture()
    receipt = data[1]
    replacements = dict(source_edge=999, nodes=receipt.nodes[::-1],
        uv=((9., 9.), receipt.uv[1]), parameters=(Fraction(1, 8), receipt.parameters[1]))
    changed = (data[0], replace(receipt, **{field: replacements[field]}), *data[2:])
    before = state(data[0], data[2], data[3], data[-1])
    work = dict(topology_operations=0, insertions=0, reserved_node_reuses=0)
    with pytest.raises(MeshError):
        stage(changed, work)
    assert work == dict(topology_operations=1, insertions=1, reserved_node_reuses=0)
    assert state(data[0], data[2], data[3], data[-1]) == before


@pytest.mark.parametrize('kind', ['topology', 'insertion', 'reuse'])
def test_insufficient_original_allowance_does_not_attempt_or_mutate(kind):
    data = fixture(operations=1, insertions=1)
    before = state(data[0], data[2], data[3], data[-1])
    work = dict(topology_operations=int(kind == 'topology'), insertions=int(kind == 'insertion'),
                reserved_node_reuses=int(kind == 'reuse'))
    original = dict(work)
    with pytest.raises(_GeometryLimited, match='allowance exhausted'):
        stage(data, work)
    assert work == original and state(data[0], data[2], data[3], data[-1]) == before


def test_missing_receipt_is_not_coordinate_permission_and_attempt_is_charged():
    data = fixture()
    before = state(data[0], data[2], data[3], data[-1])
    work = dict(topology_operations=0, insertions=0, reserved_node_reuses=0)
    with pytest.raises(MeshError, match='authoritative source receipt'):
        stage((data[0], None, *data[2:]), work)
    assert work == dict(topology_operations=1, insertions=1, reserved_node_reuses=0)
    assert state(data[0], data[2], data[3], data[-1]) == before


def test_stale_owner_after_detached_scoring_never_returns_patch():
    data = fixture()
    topology, _, mesh, registry, _, _, model = data
    before = state(topology, mesh, registry, model)[:-1]
    work = dict(topology_operations=0, insertions=0, reserved_node_reuses=0)
    def mutate(phase):
        if phase == 'curved boundary staging final binding':
            model.add_point(8., 8., 8.)
    with pytest.raises(GeometryError, match='stale'):
        stage(data, work, cancellation_check=mutate)
    assert state(topology, mesh, registry, model)[:-1] == before
    assert work == dict(topology_operations=1, insertions=1, reserved_node_reuses=0)


@pytest.mark.parametrize('kind', ['registry', 'mesh_endpoint'])
def test_final_source_receipt_recheck_detects_changes_without_geometry_revision(kind):
    data = fixture()
    topology, receipt, mesh, registry, _, _, model = data
    original_revision = model.revision
    original_geometry = to_dict(model)
    after_mutation = []
    work = dict(topology_operations=0, insertions=0, reserved_node_reuses=0)
    def mutate(phase):
        if phase != 'curved boundary staging final binding':
            return
        if kind == 'registry':
            entry = registry.entries(receipt.source_edge)[0]
            # Fault injection changes only the registry station identity; no
            # geometry revision changes and no model operation is involved.
            changed = replace(entry, node_id=1001)
            registry._entries[entry.key] = changed
            registry._by_edge[receipt.source_edge][entry.key.parameter] = changed
        else:
            mesh.nodes[receipt.nodes[0]][0] += .01
        after_mutation.append(state(topology, mesh, registry, model))
    with pytest.raises(MeshError):
        stage(data, work, cancellation_check=mutate)
    assert len(after_mutation) == 1
    assert state(topology, mesh, registry, model) == after_mutation[0]
    assert model.revision == original_revision and to_dict(model) == original_geometry
    assert work == dict(topology_operations=1, insertions=1, reserved_node_reuses=0)
