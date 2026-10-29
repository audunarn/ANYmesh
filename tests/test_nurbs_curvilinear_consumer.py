"""Prepared public NURBS consumer contracts; run only with owner proof available."""
from dataclasses import replace
import numpy as np
import pytest
import anygeometry as owner
from anygeometry import GeometryModel, GeometryError, to_dict
from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface
from anymesher.errors import MeshError
from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.native_v2 import NativeMeshingOptions
from anymesher._material_region_binding import MaterialRegionBinding, prepare_material_regions


def patch_model(*, warp=.1):
    model = GeometryModel()
    support = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
        (((0., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (1., 1., warp))))
    corners = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
    curves = tuple(CurveOnSurface(support, NurbsCurve(1, (0., 0., 1., 1.),
        (corners[i], corners[(i + 1) % 4]))) for i in range(4))
    vertices = tuple(model.add_point(*curve.exact_point(0)) for curve in curves)
    edges = tuple(model.add_curve(vertices[i], vertices[(i + 1) % 4], curve)
                  for i, curve in enumerate(curves))
    face = model.add_face(edges, surface=support)
    model.add_sheet((face,))
    return model, face, edges


def options():
    return dict(target_size=.5, strategy='native', native_backend='python',
        order='linear', certification_mode='none',
        geometry_approximation_error=.01,
        native_options=NativeMeshingOptions(point_placement='frontal_delaunay',
            metric_mode='isotropic_spatial'), structural_preparation=False)


@pytest.mark.parametrize('value', [True, np.bool_(False), 0, -1, float('nan'), float('inf'), '0.01',
    pytest.param(10**1000, id='overflowing-real')])
def test_public_approximation_limit_rejects_invalid_values_before_mutation(value):
    model, _, _ = patch_model()
    before = to_dict(model)
    request = options()
    request['geometry_approximation_error'] = value
    with pytest.raises(MeshError, match='finite positive real'):
        generate_hybrid_mesh_result(model, **request)
    assert to_dict(model) == before


def test_public_nurbs_route_resolves_automatic_approximation(monkeypatch):
    from anymesher import resolve_geometry_approximation
    model, face, _ = patch_model(warp=0.)
    before = to_dict(model)
    request = options()
    request.pop('geometry_approximation_error')
    expected = resolve_geometry_approximation(model, (face,))
    actual = owner.validate_prepared_material_curvilinear_cells
    receipts = []
    def observe(*args, **kwargs):
        receipt = actual(*args, **kwargs)
        receipts.append(receipt)
        return receipt
    monkeypatch.setattr(owner, 'validate_prepared_material_curvilinear_cells', observe)
    result = generate_hybrid_mesh_result(model, **request)
    diagnostic = result.mesh.hybrid_diagnostics['geometry_approximation']
    assert diagnostic == expected.to_dict()
    assert result.mesh.hybrid_diagnostics['geometry_approximation_error'] == expected.effective_error
    assert result.mesh.elements_of_face[face]
    assert to_dict(model) == before

    assert receipts and all(r.surface_error == expected.effective_error for r in receipts)


def test_public_success_calls_actual_owner_cell_proof_before_and_after_shared_splits(monkeypatch):
    model, face, _ = patch_model()
    before = to_dict(model)
    actual = owner.validate_prepared_material_curvilinear_cells
    calls = []
    def observe(*args, **kwargs):
        result = actual(*args, **kwargs)
        calls.append(result)
        return result
    monkeypatch.setattr(owner, 'validate_prepared_material_curvilinear_cells', observe)
    result = generate_hybrid_mesh_result(model, **options())
    assert result.mesh.elements_of_face[face]
    assert len(calls) >= 2
    assert all(isinstance(receipt, owner.PreparedMaterialCurvilinearCells) for receipt in calls)
    assert to_dict(model) == before


def test_public_owner_boundary_refusal_never_falls_back_to_unproved_mesh(monkeypatch):
    model, _, _ = patch_model()
    before = to_dict(model)
    calls = []
    def refuse(*args, **kwargs):
        calls.append(True)
        raise GeometryError('unresolved source interval')
    monkeypatch.setattr(owner, 'query_prepared_material_boundary_sampling', refuse)
    with pytest.raises((GeometryError, MeshError), match='unresolved source interval'):
        generate_hybrid_mesh_result(model, **options())
    assert calls
    assert to_dict(model) == before


@pytest.mark.parametrize('transported', [False, True])
def test_public_owner_proved_resolution_request_rebuilds_then_requires_real_proof(monkeypatch, transported):
    from fractions import Fraction
    model, face, _ = patch_model()
    before = to_dict(model)
    actual = owner.validate_prepared_material_curvilinear_cells
    calls = []
    def require_resolution(*args, **kwargs):
        cells = args[3]
        calls.append(cells)
        if len(calls) == 1:
            limit = Fraction(kwargs['surface_error'])
            allocated = limit-Fraction(1, 10**12) if transported else limit
            raise owner.SourceCellRefinementRequired(
                cells[0].face.id, cells[0].node_ids, allocated, 2*limit*limit,
                requested_error=limit)
        return actual(*args, **kwargs)
    monkeypatch.setattr(owner, 'validate_prepared_material_curvilinear_cells', require_resolution)
    result = generate_hybrid_mesh_result(model, **options())
    assert result.mesh.elements_of_face[face]
    assert len(calls) >= 3  # refused attempt, actual rebuilt proof, final proof
    assert to_dict(model) == before


def test_public_generic_cell_refusal_is_not_a_resolution_request(monkeypatch):
    model, _, _ = patch_model()
    before = to_dict(model)
    calls = []
    def refuse(*args, **kwargs):
        calls.append(True)
        raise GeometryError('source-cell approximation requires finer mesh resolution')
    monkeypatch.setattr(owner, 'validate_prepared_material_curvilinear_cells', refuse)
    with pytest.raises(GeometryError):
        generate_hybrid_mesh_result(model, **options())
    assert len(calls) == 1
    assert to_dict(model) == before


@pytest.mark.parametrize('typed_signal', [False, True])
def test_public_cell_callback_exception_identity_never_grants_retry(monkeypatch, typed_signal):
    from fractions import Fraction
    model, _, _ = patch_model()
    before = to_dict(model)
    calls, failures = [], []
    def callback(phase):
        if phase == 'test whole-cell callback':
            raise failures[0]
    def owner_query(*args, **kwargs):
        calls.append(True)
        cell = args[3][0]
        limit = Fraction(kwargs['surface_error'])
        failures.append(owner.SourceCellRefinementRequired(
            cell.face.id, cell.node_ids, limit, 2*limit**2)
            if typed_signal else RuntimeError('user callback exception'))
        kwargs['cancellation_check']('test whole-cell callback')
        pytest.fail('callback exception swallowed')
    monkeypatch.setattr(owner, 'validate_prepared_material_curvilinear_cells', owner_query)
    with pytest.raises(Exception) as caught:
        generate_hybrid_mesh_result(model, cancellation_check=callback, **options())
    assert caught.value is failures[0]
    assert len(calls) == 1
    assert to_dict(model) == before


def test_public_owner_boundary_cancellation_preserves_source(monkeypatch):
    model, _, _ = patch_model()
    before = to_dict(model)
    failure = RuntimeError('cancel NURBS chord proof')
    def cancel(phase):
        if phase == 'test owner chord checkpoint':
            raise failure
    def query(*args, **kwargs):
        kwargs['cancellation_check']('test owner chord checkpoint')
        pytest.fail('cancelled owner operation returned')
    monkeypatch.setattr(owner, 'query_prepared_material_boundary_sampling', query)
    with pytest.raises(RuntimeError) as caught:
        generate_hybrid_mesh_result(model, cancellation_check=cancel, **options())
    assert caught.value is failure
    assert to_dict(model) == before


@pytest.mark.parametrize('stop_phase', ['NURBS curvilinear published cells',
    'material boundary source receipts', 'material region published cells'])
def test_public_owner_cell_cancellation_return_value_preserves_source(stop_phase):
    model, _face, _edges = patch_model()
    before = to_dict(model)
    phases = []
    def cancel(phase):
        phases.append(phase)
        return phase == stop_phase
    with pytest.raises(MeshError, match='cancelled'):
        generate_hybrid_mesh_result(model, cancellation_check=cancel, **options())
    assert stop_phase in phases
    assert to_dict(model) == before


def test_metric_anchor_rejects_changed_limit_and_stale_actual_receipt():
    model, face, edges = patch_model()
    regions = owner.query_material_surface_regions(model, (face,))
    receipt = owner.query_prepared_material_boundary_chords(model, regions,
        tuple((model.handle('edge', edge), 0., 1.) for edge in edges), chord_error=.01)
    binding = MaterialRegionBinding(model, regions, regions.regions[0], face,
                                    geometry_approximation_error=.01)
    binding._curvilinear_state['chords'] = receipt
    assert len(binding.reference_uv()) == 2
    with pytest.raises(MeshError, match='another approximation limit'):
        replace(binding, geometry_approximation_error=.001).reference_uv()
    model.add_point(2., 2., 2.)
    with pytest.raises(GeometryError):
        binding.reference_uv()


def test_standalone_legacy_coons_does_not_require_new_approximation_authority():
    model = GeometryModel()
    points = model.add_points(((0,0,0),(.5,.1,0),(1,0,0)))
    edge = model.add_spline(points[0], points[1:-1], points[-1])
    faces = model.extrude((edge,), (0,0,1))
    before = to_dict(model)
    assert prepare_material_regions(model, faces, {face:(face,) for face in faces},
        order='linear', native_options=NativeMeshingOptions()) == {}
    assert to_dict(model) == before


@pytest.mark.parametrize('membership', [(7, 8), (), (8,)])
def test_final_face_membership_change_refuses_before_owner_dispatch(monkeypatch, membership):
    # Injected invalid state tests only the refusal frontier, not authority.
    from types import SimpleNamespace
    model, face, _ = patch_model()
    regions = owner.query_material_surface_regions(model, (face,))
    binding = MaterialRegionBinding(model, regions, regions.regions[0], face,
                                    geometry_approximation_error=.01)
    binding._curvilinear_state.update(chords=object(), cells=SimpleNamespace(cells=()),
                                     published_elements=((7, (1, 2, 3)),))
    mesh = SimpleNamespace(elements_of_face={face: membership}, tris={7:(1,2,3)}, quads={})
    def no_authority(*args, **kwargs):
        pytest.fail('changed actual face membership must refuse before owner dispatch')
    monkeypatch.setattr(owner, 'validate_prepared_material_curvilinear_cells', no_authority)
    with pytest.raises(MeshError, match='face cell membership changed'):
        binding.validate_curvilinear_publication(mesh)


def test_positive_fraction_error_normalizes_to_literal_float():
    from fractions import Fraction
    from anymesher._material_region_binding import checked_geometry_approximation_error
    result = checked_geometry_approximation_error(Fraction(1, 100))
    assert type(result) is float and result == .01

@pytest.mark.parametrize('strict', [False, True])
def test_new_domain_recovery_selector_uses_owner_chart_and_preserves_strict_intent(strict):
    from anymesher.recovery import select_curvilinear_recovery_policy, MeshAutomationOptions
    model, _, _ = patch_model()
    original = MeshAutomationOptions(strict_method=True) if strict else None
    policy, record = select_curvilinear_recovery_policy(model,
        requested_strategy='quad_first', order='linear', certification_mode='interactive',
        geometry_approximation_error=.01, explicit_controls=False, automation=original)
    if strict:
        assert policy is original and record is None
    else:
        assert policy is not None and not policy.strict_method and not policy.allow_inspection
        assert record['requested_strategy'] == 'quad_first'


@pytest.mark.parametrize('explicit', [True, False])
def test_shared_selector_preserves_explicit_controls_and_legacy_coons(explicit):
    from anymesher.recovery import select_curvilinear_recovery_policy
    model, _, _ = patch_model()
    if not explicit:
        model = GeometryModel()
        p = model.add_points(((0,0,0),(.5,.1,0),(1,0,0)))
        model.extrude((model.add_spline(p[0], p[1:-1], p[-1]),), (0,0,1))
    policy, record = select_curvilinear_recovery_policy(model,
        requested_strategy='quad_first', order='linear', certification_mode='interactive',
        geometry_approximation_error=.01, explicit_controls=explicit)
    assert policy is None and record is None


def test_existing_native_recipe_carries_exact_quality_policy_and_budgets():
    from anymesher.recovery import _attempt_options, _prefer_analytic_trim_recipe
    from anymesher.structured import StructuredMeshingOptions
    model, _, _ = patch_model()
    quality = StructuredMeshingOptions(max_element_growth=1.2)
    controls = NativeMeshingOptions(max_insertions=17, max_topology_operations=91)
    request = dict(strategy='auto', order='linear', native_options=controls,
                   _native_surface_options=quality, geometry_approximation_error=.01)
    recipes = _prefer_analytic_trim_recipe(model, request, _attempt_options(request, None))
    method, recipe = recipes[0]
    assert method == 'native'
    assert recipe['_native_surface_options'] is quality
    assert recipe['native_options'].max_insertions == 17
    assert recipe['native_options'].max_topology_operations == 91


def test_runtime_reference_candidates_preserve_schema_and_do_not_transfer_on_copy_or_serde():
    from dataclasses import fields
    from copy import deepcopy
    from anymesher.mesh import Mesh
    from anymesher.serialize import mesh_to_dict, mesh_from_dict
    from anymesher._material_region_binding import (
        retain_curvilinear_reference_candidates, curvilinear_reference_candidates)
    mesh = Mesh()
    candidate = object()  # Opaque lookup candidate; never claimed as authority.
    before = mesh_to_dict(mesh)
    retain_curvilinear_reference_candidates(mesh, ())
    assert set(vars(mesh)) == {field.name for field in fields(Mesh)}
    retain_curvilinear_reference_candidates(mesh, (candidate,))
    assert curvilinear_reference_candidates(mesh) == (candidate,)
    assert set(vars(mesh)) == {field.name for field in fields(Mesh)}
    assert mesh_to_dict(mesh) == before
    assert curvilinear_reference_candidates(deepcopy(mesh)) == ()
    assert curvilinear_reference_candidates(mesh_from_dict(before)) == ()


def test_runtime_reference_candidate_cleanup_and_reused_identity_refusal():
    import gc
    from weakref import ref
    from anymesher.mesh import Mesh
    from anymesher._material_region_binding import (
        retain_curvilinear_reference_candidates, curvilinear_reference_candidates,
        _CURVILINEAR_MESH_REFERENCES)
    mesh = Mesh()
    key = id(mesh)
    retain_curvilinear_reference_candidates(mesh, (object(),))
    reference = ref(mesh)
    del mesh
    gc.collect()
    assert reference() is None and key not in _CURVILINEAR_MESH_REFERENCES
    old, other = Mesh(), Mesh()
    _CURVILINEAR_MESH_REFERENCES[id(other)] = (ref(old), (object(),))
    try:
        assert curvilinear_reference_candidates(other) == ()
    finally:
        _CURVILINEAR_MESH_REFERENCES.pop(id(other), None)


def seeded_actual_nurbs_binding(*, flat=False):
    from anymesher.mesh import Mesh
    from anymesher.seeding import Seeding
    from anymesher.refinement import SizeField
    from anymesher.meshing_view import GeometryMeshingView
    from anymesher.boundary import GlobalEdgeBoundaryRegistry
    from anymesher.hybrid import _ensure_edge_registry
    model, face, edges = patch_model(warp=0. if flat else .1)
    controls = NativeMeshingOptions(point_placement='frontal_delaunay', metric_mode='isotropic_spatial')
    binding = prepare_material_regions(model, (face,), {face:(face,)}, order='linear',
        native_options=controls, geometry_approximation_error=.01)[face]
    parameters = {edge:np.asarray((0.,.5,1.)) for edge in edges}
    targets = binding.prepare_curvilinear_seed(parameters)
    seeding = Seeding(divisions={edge:2 for edge in edges})
    mesh = Mesh(geometry_model_id=model.model_id, geometry_revision=model.revision,
                seeding=seeding)
    view = GeometryMeshingView(model)
    registry = GlobalEdgeBoundaryRegistry(view)
    _ensure_edge_registry(model,view,mesh,registry,edges,seeding,SizeField(model,.5),
                          'linear',owner_stations=targets)
    binding.prepare_curvilinear_chords(mesh,registry)
    return model, face, edges, binding, mesh, registry


def test_actual_owner_seed_allocates_literal_targets_and_canonical_uv():
    model, face, edges, binding, mesh, registry = seeded_actual_nurbs_binding()
    chords = binding._curvilinear_state['chords']
    owner.validate_prepared_material_boundary_chords_binding(model,binding.collection,chords)
    for interval in chords.intervals:
        entries = {entry.key.parameter:entry.node_id for entry in registry.entries(interval.edge.id)}
        occurrence = next(row for row in interval.occurrences if row.face.id == face)
        for parameter, xyz, uv in zip(interval.parameter_range,interval.xyz,occurrence.uv):
            node = entries[float(parameter)]
            assert tuple(mesh.nodes[node]) == xyz
            for receipt in binding.boundary_intervals(mesh,registry):
                if receipt.source_edge == interval.edge.id and parameter in receipt.parameters:
                    assert receipt.uv[receipt.parameters.index(parameter)] == uv


def test_nurbs_metric_roundtrip_preserves_normalized_boundary_without_clipping():
    from anymesher._analytic_metric_chart import AnalyticMetricChart
    model, face, _edges, binding, _mesh, _registry = seeded_actual_nurbs_binding()
    chart = AnalyticMetricChart(model, face, region_binding=binding)
    uv = np.asarray(((0., 0.), (1., 0.), (1., 1.), (0., 1.), (.375, .625)))
    metric_rows = uv @ chart.transform
    assert np.array_equal(metric_rows @ chart.inverse, uv)
    np.testing.assert_array_equal(chart.evaluate(metric_rows),
        owner.evaluate_material_surface_region(model, binding.collection, face, uv))
    du, dv = owner.evaluate_material_surface_region(model, binding.collection, face, uv,
                                                   derivatives=True)
    np.testing.assert_array_equal(chart.jacobians(metric_rows) @ chart.transform.T,
                                  np.stack((du, dv), axis=2))
    with pytest.raises(GeometryError, match='in \\[0, 1\\]'):
        chart.evaluate(np.asarray(((-2.**-100, .5),)) @ chart.transform)


def test_registered_ulp_station_edit_refuses_actual_owner_literal_gate():
    _model, _face, edges, binding, mesh, registry = seeded_actual_nurbs_binding()
    node = mesh.nodes_of_edge[edges[0]][0]
    mesh.nodes[node][0] = np.nextafter(mesh.nodes[node][0], np.inf)
    with pytest.raises(MeshError, match='registered stations|literal target'):
        binding.prepare_curvilinear_chords(mesh,registry)


def test_actual_shared_station_refresh_preserves_known_node_identity():
    from fractions import Fraction
    model, _face, edges, binding, mesh, registry = seeded_actual_nurbs_binding()
    edge = edges[0]
    row, point = binding.curvilinear_station(edge,Fraction(1,4))
    original_nodes = tuple(mesh.nodes_of_edge[edge])
    node = max(mesh.nodes)+1
    mesh.nodes[node] = point.copy()
    mesh.nodes_of_edge[edge].insert(1,node)
    registry.register(edge,.25,point,node_id=node,owner=model.handle('edge',edge))
    binding.prepare_curvilinear_chords(mesh,registry)
    assert tuple(mesh.nodes_of_edge[edge][::1]) == (original_nodes[0],node,*original_nodes[1:])
    receipts = binding.boundary_intervals(mesh,registry)
    canonical = [receipt.uv[receipt.nodes.index(node)] for receipt in receipts if node in receipt.nodes]
    assert canonical and all(tuple(value) == tuple(row) for value in canonical)
    assert tuple(mesh.nodes[node]) == tuple(point)


def actual_triangle_fan_binding():
    # Publication attacks do not depend on curvature. Keep a real tensor
    # support and owner receipt while avoiding repeated curved-cell campaigns.
    model, face, edges, binding, mesh, registry = seeded_actual_nurbs_binding(flat=True)
    centre = max(mesh.nodes)+1
    mesh.nodes[centre] = np.asarray(model.faces[face].surface.evaluate(.5, .5))
    binding._curvilinear_state['uv_by_node'] = {centre:(.5, .5)}
    boundary = tuple(node for edge in edges for node in mesh.nodes_of_edge[edge][:-1])
    mesh.tris = {i+1:(a, boundary[(i+1) % len(boundary)], centre)
                 for i, a in enumerate(boundary)}
    mesh.elements_of_face[face] = tuple(mesh.tris)
    mesh.elements_of_sheet[next(iter(model.sheets))] = tuple(mesh.tris)
    return model, face, binding, mesh, registry


@pytest.mark.parametrize('attack', ['interior-node', 'quality-limit', 'quality-array-alias'])
def test_physical_quality_callback_cannot_change_entry_inputs(monkeypatch, attack):
    from anymesher.surface_mesh import SurfaceMeshOptions
    model, _face, binding, mesh, registry = actual_triangle_fan_binding()
    before = to_dict(model)
    settings = SurfaceMeshOptions(min_angle=np.array(46.)) if attack == 'quality-array-alias' else SurfaceMeshOptions()
    if attack != 'interior-node':
        # Exercise the outer quality callback after the real owner proof.
        # Owner-cell callback integrity has separate regressions; repeating
        # its nested callbacks is unnecessary for this settings attack.
        actual = MaterialRegionBinding.certify_curvilinear_published
        def owner_proof(self, live_mesh, live_registry, cancellation_check=None):
            return actual(self, live_mesh, live_registry, None)
        monkeypatch.setattr(MaterialRegionBinding, 'certify_curvilinear_published', owner_proof)
    attacked = False
    def poison(phase):
        nonlocal attacked
        target = ('material region constraint incidence' if attack == 'interior-node'
                  else 'analytic physical quality start')
        if attacked or phase != target:
            return False
        attacked = True
        if attack == 'interior-node':
            mesh.nodes[max(mesh.nodes)][0] += .005
        elif attack == 'quality-limit':
            object.__setattr__(settings, 'min_angle', 0.)
        else:
            settings.min_angle[...] = 0.
        return False
    with pytest.raises(MeshError, match='physical quality inputs changed during publication proof'):
        binding.certify_published(mesh, registry, settings, poison)
    assert attacked and to_dict(model) == before


@pytest.mark.parametrize('attack', ['extra-cell', 'connectivity', 'xyz', 'registry-owner'])
def test_owner_cell_callback_cannot_change_entry_mesh(attack):
    model, face, binding, mesh, registry = actual_triangle_fan_binding()
    before = to_dict(model)
    changed = False
    def poison(phase):
        nonlocal changed
        if changed or phase != 'NURBS curvilinear published cells':
            return False
        changed = True
        first = mesh.elements_of_face[face][0]
        if attack == 'extra-cell':
            extra = max(mesh.tris)+1
            mesh.tris[extra] = mesh.tris[first]
            mesh.elements_of_face[face] += (extra,)
        elif attack == 'connectivity':
            mesh.tris[first] = mesh.tris[first][::-1]
        elif attack == 'xyz':
            mesh.nodes[mesh.tris[first][0]][0] += .125
        else:
            from anymesher.meshing_view import GeometryMeshingView
            registry.view = GeometryMeshingView(model.clone(preserve_identity=True))
        return False
    with pytest.raises(MeshError, match='mesh or stations changed during owner proof'):
        binding.certify_curvilinear_published(mesh, registry, poison)
    assert changed and to_dict(model) == before


@pytest.mark.parametrize('attack', ['extra-cell', 'connectivity', 'element-kind'])
def test_publication_candidate_map_cannot_launder_cells_outside_receipt(attack):
    _model, face, binding, mesh, registry = actual_triangle_fan_binding()
    receipt = binding.certify_curvilinear_published(mesh, registry)
    assert len(receipt.cells) == len(mesh.tris)
    published = binding._curvilinear_state['published_elements']
    first, nodes = published[0]
    if attack == 'extra-cell':
        extra = max(mesh.tris)+1
        mesh.tris[extra] = nodes
        mesh.elements_of_face[face] += (extra,)
        binding._curvilinear_state['published_elements'] += ((extra, nodes),)
    elif attack == 'connectivity':
        mesh.tris[first] = nodes[::-1]
        binding._curvilinear_state['published_elements'] = ((first, nodes[::-1]), *published[1:])
    else:
        mesh.quads[first] = (*nodes, nodes[0])
    with pytest.raises(MeshError, match='differs from owner-proved cells|ambiguous element kinds'):
        binding.validate_curvilinear_publication(mesh)


@pytest.mark.parametrize('attack', ['bound', 'owner', 'parent'])
def test_construction_authority_cannot_be_replaced_before_proof(attack):
    model, _face, _edges, binding, _mesh, _registry = seeded_actual_nurbs_binding()
    if attack == 'bound':
        object.__setattr__(binding, 'geometry_approximation_error', 1.)
    elif attack == 'owner':
        object.__setattr__(binding, 'geometry', model.clone(preserve_identity=True))
    else:
        object.__setattr__(binding, 'authored_face', 900)
    with pytest.raises(MeshError, match='construction authority'):
        binding.reference_uv()
    with pytest.raises(MeshError, match='construction authority'):
        binding.__post_init__()


@pytest.mark.parametrize('attack', ['element-kind', 'extra-global-cell', 'sheet-association',
                                  'registry-owner', 'registry-replacement', 'registry-sheet-index',
                                  'registry-edge-index'])
def test_final_publication_callback_cannot_change_unproved_mesh_state(monkeypatch, attack):
    from anymesher.meshing_view import GeometryMeshingView
    from anymesher.boundary import GlobalEdgeBoundaryRegistry
    model, _face, _edges = patch_model(warp=0.)
    before = to_dict(model)
    captured = {}
    actual = MaterialRegionBinding.certify_curvilinear_published
    def observe(self, mesh, registry, cancellation_check=None):
        captured.update(mesh=mesh, registry=registry)
        return actual(self, mesh, registry, cancellation_check)
    monkeypatch.setattr(MaterialRegionBinding, 'certify_curvilinear_published', observe)
    def poison(phase):
        if phase != 'hybrid generation complete':
            return False
        captured['attacked'] = True
        mesh, registry = captured['mesh'], captured['registry']
        cells = mesh.tris or mesh.quads
        element, nodes = next(iter(cells.items()))
        if attack == 'element-kind':
            if element in mesh.tris:
                mesh.quads[element] = (*nodes, nodes[0])
            else:
                mesh.tris[element] = nodes[:3]
        elif attack == 'extra-global-cell':
            extra = max((*mesh.tris, *mesh.quads))+1
            mesh.tris[extra] = nodes[:3]
        elif attack == 'sheet-association':
            mesh.elements_of_sheet[next(iter(mesh.elements_of_sheet))] = ()
        elif attack == 'registry-owner':
            registry.view = GeometryMeshingView(model.clone(preserve_identity=True))
        elif attack == 'registry-sheet-index':
            registry.view.sheet_faces = {}
        elif attack == 'registry-edge-index':
            registry._by_edge[next(iter(registry._by_edge))].clear()
        else:
            mesh.boundary_registry = GlobalEdgeBoundaryRegistry(registry.view)
        return False
    with pytest.raises(MeshError, match='final mesh or registry changed during publication callback|boundary registry indexes disagree'):
        request = options()
        request['target_size'] = 2.
        request['mutation_policy'] = 'working_copy'
        # Generation proves a real mesh without progress callbacks. Exercise
        # the actual finalization callback separately; repeating callback
        # integrity at every source-cell proof box belongs to its own tests.
        candidate = generate_hybrid_mesh_result(model, _defer_finalization=True, **request)
        from anymesher.hybrid import _finalize_hybrid_candidate
        _finalize_hybrid_candidate(candidate, cancellation_check=poison)
    assert captured.get('attacked') and to_dict(model) == before


@pytest.mark.parametrize('erase_selection', [False, True])
def test_curvilinear_source_remap_refuses_before_proximity_registry(monkeypatch, erase_selection):
    import importlib
    hybrid = importlib.import_module('anymesher.hybrid')
    model, _face, _edges = patch_model(warp=0.)
    before = to_dict(model)
    request = options()
    request['target_size'] = 2.
    candidate = generate_hybrid_mesh_result(model, _defer_finalization=True, **request)
    candidate.preparation_report = object()
    if erase_selection:
        candidate.faces = ()
        candidate.preparation_report = None
        candidate.structured_report = None
    def refuse_projection(*args, **kwargs):
        pytest.fail('unqualified source-parameter reconstruction must not run')
    monkeypatch.setattr(hybrid, '_published_boundary_registry', refuse_projection)
    with pytest.raises(MeshError, match='requires qualified source-parameter transport'):
        hybrid._finalize_hybrid_candidate(candidate)
    assert to_dict(model) == before


def test_actual_owner_sampling_refines_fe_stations_before_allocation():
    from fractions import Fraction
    from anymesher.mesh import Mesh
    from anymesher.seeding import Seeding
    from anymesher.refinement import SizeField
    from anymesher.meshing_view import GeometryMeshingView
    from anymesher.boundary import GlobalEdgeBoundaryRegistry
    from anymesher.hybrid import _ensure_edge_registry
    model = GeometryModel()
    # An injective planar tensor chart with an S-shaped boundary: the midpoint
    # of the whole chord lies on the chord, so midpoint-only sampling is unsafe.
    support = NurbsSurface(3, 1, (0.,)*4 + (1.,)*4, (0., 0., 1., 1.),
        tuple(((u, y, 0.), (u, y+1., 0.)) for u, y in
              ((0., 0.), (1./3., .4), (2./3., -.4), (1., 0.))))
    corners = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
    curves = tuple(CurveOnSurface(support, NurbsCurve(1, (0., 0., 1., 1.),
        (corners[i], corners[(i+1) % 4]))) for i in range(4))
    vertices = tuple(model.add_point(*curve.exact_point(0)) for curve in curves)
    edges = tuple(model.add_curve(vertices[i], vertices[(i+1) % 4], curve)
                  for i, curve in enumerate(curves))
    face = model.add_face(edges, surface=support)
    model.add_sheet((face,))
    controls = NativeMeshingOptions(point_placement='frontal_delaunay',
                                   metric_mode='isotropic_spatial')
    binding = prepare_material_regions(model, (face,), {face:(face,)}, order='linear',
        native_options=controls, geometry_approximation_error=.01)[face]
    requested = {edge:(0., .5, 1.) for edge in edges}
    targets = binding.prepare_curvilinear_seed(requested)
    assert len(targets[edges[0]]) > 3
    assert all(set(requested[edge]) <= set(targets[edge]) for edge in edges)
    sampling = binding._curvilinear_state['sampling']
    owner.validate_prepared_material_boundary_sampling_binding(model, binding.collection, sampling)
    # Simulate a station requested by another incident face. The union is
    # authenticated as actual intervals, without silently regrading it.
    shared = {edge:tuple(sorted(set(values) | {0.125})) for edge, values in targets.items()}
    targets = binding.prepare_curvilinear_seed(shared, refine=False)
    assert all(set(shared[edge]) == set(targets[edge]) for edge in edges)
    seeding = Seeding(divisions={edge:2 for edge in edges})
    mesh = Mesh(geometry_model_id=model.model_id, geometry_revision=model.revision,
                seeding=seeding)
    view = GeometryMeshingView(model)
    registry = GlobalEdgeBoundaryRegistry(view)
    _ensure_edge_registry(model, view, mesh, registry, edges, seeding, SizeField(model,.5),
                          'linear', owner_stations=targets)
    binding.prepare_curvilinear_chords(mesh, registry)
    for edge in edges:
        assert tuple(entry.key.parameter for entry in registry.entries(edge)) == tuple(sorted(targets[edge]))
        assert len(mesh.nodes_of_edge[edge]) == len(targets[edge])
        for entry in registry.entries(edge):
            assert tuple(mesh.nodes[entry.node_id]) == targets[edge][entry.key.parameter]
            assert Fraction(entry.key.parameter) in dict(sampling.edge_parameters)[model.handle('edge', edge)] or entry.key.parameter == .125
