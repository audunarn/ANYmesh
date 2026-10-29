"""Curvilinear S3 normals and source admission use actual owner cell maps."""
import numpy as np
import pytest
from copy import deepcopy
from dataclasses import replace
from uuid import uuid4
import anygeometry as owner
from anymesher.errors import MeshError
from anymesher.material_source_transfer import (
    _publish_identity_source_transfer, _qualify_identity_source_s3,
)
from test_material_source_transfer_consumer import prepared_mesh


@pytest.mark.parametrize('requested', (True, False))
def test_new_curvilinear_recovery_preserves_effective_recombination(monkeypatch, requested):
    import importlib
    recovery = importlib.import_module('anymesher.recovery')
    from test_nurbs_curvilinear_consumer import patch_model
    source, _face, _edges = patch_model()
    class RecipeReached(Exception):
        pass
    def inspect(_geometry, **options):
        assert options['recombine'] is requested
        raise RecipeReached()
    monkeypatch.setattr(recovery, 'generate_hybrid_mesh_result', inspect)
    with pytest.raises(RecipeReached):
        recovery.generate_automatic_mesh_result(source, strategy='quad_first',
            order='linear', recombine=requested, geometry_approximation_error=.01,
            curvilinear_recovery_selection={'requested_strategy':'quad_first'})


def test_nurbs_interior_size_proposal_uses_current_owner_chords_without_granting_proof():
    from test_nurbs_curvilinear_consumer import patch_model
    from anymesher._material_region_binding import MaterialRegionBinding
    source, face, edges = patch_model()
    regions = owner.query_material_surface_regions(source, (face,))
    intervals = tuple((source.handle('edge', edge), i/8, (i+1)/8)
                      for edge in edges for i in range(8))
    chords = owner.query_prepared_material_boundary_chords(
        source, regions, intervals, chord_error=.01)
    binding = MaterialRegionBinding(source, regions, regions.regions[0], face,
                                    geometry_approximation_error=.01)
    binding._curvilinear_state['chords'] = chords
    before = owner.to_dict(source)
    proposed = binding.interior_size_proposal(.25)
    assert 0 < proposed < .2
    assert binding.interior_size_proposal(.05) == .05
    assert binding._curvilinear_state['chords'] is chords
    assert 'cells' not in binding._curvilinear_state
    assert owner.to_dict(source) == before
    with pytest.raises(owner.GeometryError, match='cancelled'):
        binding.interior_size_proposal(.25, lambda _phase: True)
    units = source._units
    try:
        source._units = 'mm'
        with pytest.raises(owner.GeometryError):
            binding.interior_size_proposal(.25)
    finally:
        source._units = units


@pytest.mark.parametrize('failure', ('nan', 'cancel'))
def test_public_nurbs_seed_failure_preserves_source(monkeypatch, failure):
    import anymesher.hybrid as hybrid
    from anymesher._analytic_metric_chart import AnalyticMetricChart
    from test_nurbs_curvilinear_consumer import patch_model, options
    source, _face, _edges = patch_model(warp=.25)
    before = owner.to_dict(source)

    class Cancelled(BaseException):
        pass

    def fail_seed_jacobians(chart, points):
        if failure == 'cancel':
            raise Cancelled()
        return np.full((len(points), 3, 2), np.nan)

    monkeypatch.setattr(AnalyticMetricChart, 'jacobians', fail_seed_jacobians)
    with pytest.raises(Cancelled if failure == 'cancel' else MeshError):
        hybrid.generate_hybrid_mesh_result(source, mutation_policy='read_only', **options())
    assert owner.to_dict(source) == before


@pytest.mark.parametrize('explicit, expected, warp',
    ((None, 2., 0.), (.02, .02, 0.), (5., 2., 0.), (.02, .02, .25)))
def test_public_curved_proposal_reaches_native_metric_and_preserves_finer_controls(monkeypatch, explicit, expected, warp):
    import anymesher.hybrid as hybrid
    from anymesher._analytic_metric_chart import AnalyticMetricChart
    from anymesher.metric import MetricFieldSpec
    from test_nurbs_curvilinear_consumer import patch_model, options
    source, _face, _edges = patch_model(warp=warp)
    before = owner.to_dict(source)
    request = options()
    request['target_size'] = 2.
    if explicit is not None:
        request['native_options'] = replace(request['native_options'],
                                           metric_field=MetricFieldSpec.uniform(explicit))
    captured = []
    seed_stretches = []
    actual_jacobians = AnalyticMetricChart.jacobians

    def observe_jacobians(chart, points):
        result = actual_jacobians(chart, points)
        seed_stretches.append(float(np.max(np.linalg.svd(result, compute_uv=False))))
        return result

    class ConfigurationObserved(BaseException):
        pass

    def inspect_native_configuration(*args, **kwargs):
        native = kwargs.get('native_options')
        if native is None:
            native = kwargs['options'].native_options
        settings = kwargs.get('options')
        seed_size = kwargs['target_size'] if settings is None else settings.target_size
        captured.append((native.metric_field, seed_size))
        raise ConfigurationObserved()

    monkeypatch.setattr(hybrid, 'mesh_planar_surface', inspect_native_configuration)
    monkeypatch.setattr(AnalyticMetricChart, 'jacobians', observe_jacobians)
    with pytest.raises(ConfigurationObserved):
        hybrid.generate_hybrid_mesh_result(source, mutation_policy='working_copy', **request)
    assert len(captured) == 1
    field, seed_size = captured[0]
    assert field.global_control.target_size == expected
    assert 0 < seed_size <= expected * (1. + 1e-12)
    if warp:
        assert len(seed_stretches) == 1 and seed_stretches[0] > 1.
        assert seed_size == pytest.approx(expected / seed_stretches[0], rel=1e-12)
    if explicit is not None:
        original = request['native_options'].metric_field
        assert field.maximum_gradation <= original.maximum_gradation
        assert field.maximum_anisotropy <= original.maximum_anisotropy
    assert owner.to_dict(source) == before


def small_actual_mesh(*, ordinary_face=False, recombine=False):
    from anymesher.hybrid import generate_hybrid_mesh_result
    from test_nurbs_curvilinear_consumer import patch_model, options
    source, face, _ = patch_model(warp=0.)
    if ordinary_face:
        points = [source.add_point(*point) for point in
                  ((2., 0., 0.), (3., 0., 0.), (3., 1., 0.), (2., 1., 0.))]
        edges = [source.add_line(points[i], points[(i+1)%4]) for i in range(4)]
        ordinary = source.add_face(edges, surface=owner.Plane((2., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
        source.add_sheet((ordinary,))
    prepared = owner.clone_prepared_geometry(source)
    request = options()
    request.update(target_size=2., recombine=recombine)
    mesh = generate_hybrid_mesh_result(prepared, mutation_policy='working_copy', **request).mesh
    return source, prepared, mesh, face


def test_prepared_s3_normals_use_proven_reference_uv_without_centroid_projection(monkeypatch):
    from anymesher.s3_production import _shell_owner_authority
    source, prepared, mesh, _ = prepared_mesh(recombine=False)
    before = owner.to_dict(source), owner.to_dict(prepared)
    def refuse(*args, **kwargs):
        raise AssertionError('curvilinear S3 cannot discover reference UV by projection')
    monkeypatch.setattr(prepared, 'project_to_face', refuse)
    monkeypatch.setattr(prepared, 'project_to_face_many', refuse)
    normals, sources = _shell_owner_authority(prepared, mesh)
    assert set(normals) == set(mesh.shells) == set(sources)
    assert all(np.isclose(np.linalg.norm(normal), 1.) for normal in normals.values())
    assert (owner.to_dict(source), owner.to_dict(prepared)) == before


def test_source_s3_rederives_live_source_faceuse_authority_without_cell_changes():
    source, prepared, mesh, face = prepared_mesh(recombine=False)
    before = owner.to_dict(source), owner.to_dict(prepared)
    _publish_identity_source_transfer(source, prepared, mesh, mesh.boundary_registry)
    physical = (dict(mesh.tris), dict(mesh.quads),
                {node: tuple(point) for node, point in mesh.nodes.items()})
    record = _qualify_identity_source_s3(source, prepared, mesh)
    assert record['authority_model']['source_model_id'] == str(source.model_id)
    assert record['authority_model']['source_revision'] == source.revision
    assert record['authority_model']['scope'] == 'SOURCE_GEOMETRY_ORIENTED_SHEET_FACE_USE'
    assert record['status'] == 'ADMITTED'
    assert all(row['face_id'] == face for row in record['element_owner_sources'].values())
    assert (dict(mesh.tris), dict(mesh.quads),
            {node: tuple(point) for node, point in mesh.nodes.items()}) == physical
    assert (owner.to_dict(source), owner.to_dict(prepared)) == before


@pytest.mark.parametrize('attack', ['namespace', 'copy', 'xyz', 'connectivity', 'planar_reassignment', 'copied_planar_reassignment'])
def test_standalone_s3_requires_actual_current_proven_cells(attack):
    from anymesher.s3_production import prepare_qualified_s3_mesh
    source, prepared, mesh, _ = small_actual_mesh(ordinary_face=attack.endswith('planar_reassignment'))
    before = owner.to_dict(source), owner.to_dict(prepared)
    if attack == 'namespace':
        mesh.geometry_model_id = uuid4()
    elif attack == 'copy':
        from dataclasses import fields
        from anymesher.mesh import Mesh
        mesh = Mesh(**deepcopy({field.name: getattr(mesh, field.name) for field in fields(Mesh)}))
    elif attack == 'xyz':
        node = next(iter(mesh.nodes))
        mesh.nodes[node] = np.asarray(mesh.nodes[node])+np.array((.01, 0., 0.))
    elif attack.endswith('planar_reassignment'):
        # A known ordinary face must not hide the retained curved cell proof.
        from anygeometry.nurbs import NurbsSurface
        plane = next(face for face, definition in prepared.faces.items()
                     if not isinstance(definition.surface, NurbsSurface))
        if attack == 'copied_planar_reassignment':
            from dataclasses import fields
            from anymesher.mesh import Mesh
            mesh = Mesh(**deepcopy({field.name: getattr(mesh, field.name) for field in fields(Mesh)}))
        mesh.elements_of_face = {plane: list(mesh.shells)}
    else:
        element = next(iter(mesh.tris))
        mesh.tris[element] = tuple(reversed(mesh.tris[element]))
    with pytest.raises(MeshError, match='namespace|authority|owner-proved|reference point'):
        prepare_qualified_s3_mesh(mesh, prepared, target_policy=None)
    assert (owner.to_dict(source), owner.to_dict(prepared)) == before


def test_s3_physical_repair_cannot_reuse_old_owner_cells(monkeypatch):
    from anymesher import s3_production as s3
    _, prepared, mesh, _ = small_actual_mesh()
    actual = s3.repair_s3_admission
    def changed(*args, **kwargs):
        repair = actual(*args, **kwargs)
        node = next(iter(repair.mesh.nodes))
        repair.mesh.nodes[node] = np.asarray(repair.mesh.nodes[node])+np.array((.01, 0., 0.))
        return repair
    monkeypatch.setattr(s3, 'repair_s3_admission', changed)
    with pytest.raises(s3.S3OwnerAuthorityError, match='fresh owner cell qualification'):
        s3.prepare_qualified_s3_mesh(mesh, prepared, target_policy=None)


def test_source_faceuse_edit_invalidates_source_s3_transfer():
    source, prepared, mesh, _ = small_actual_mesh()
    _publish_identity_source_transfer(source, prepared, mesh, mesh.boundary_registry)
    use = next(iter(source.face_uses))
    original = source.face_uses[use]
    source._face_uses[use] = replace(original, orientation=-original.orientation)
    with pytest.raises(owner.GeometryError, match='stale|changed'):
        _qualify_identity_source_s3(source, prepared, mesh)


@pytest.mark.parametrize('label', ['triangle', 'face', 'association'])
@pytest.mark.parametrize('value', [1.0, 1.5, True, np.bool_(True), 'bad'])
def test_public_s3_refuses_noninteger_entity_labels_before_mutation(label, value):
    from anymesher.mesh import Mesh
    from anymesher.s3_production import prepare_qualified_s3_mesh, S3OwnerAuthorityError
    from anymesher.material_source_transfer import _publication_pin
    from test_s3_production import _square
    geometry, face = _square()
    mesh = Mesh(geometry_model_id=geometry.model_id, geometry_revision=geometry.revision,
                nodes={1: np.array((0., 0., 0.)), 2: np.array((1., 0., 0.)),
                       3: np.array((0., 1., 0.))}, tris={1: (1, 2, 3)},
                elements_of_face={face: [1]})
    if label == 'triangle':
        mesh.tris = {value: (1, 2, 3)}
    elif label == 'face':
        mesh.elements_of_face = {value: [1]}
    else:
        mesh.elements_of_face = {face: [value]}
    before = _publication_pin(mesh), owner.to_dict(geometry)
    with pytest.raises(S3OwnerAuthorityError, match='integer identifier'):
        prepare_qualified_s3_mesh(mesh, geometry, target_policy=None)
    assert (_publication_pin(mesh), owner.to_dict(geometry)) == before


def test_public_s3_mixed_type_face_keys_are_typed_refused():
    from anymesher.mesh import Mesh
    from anymesher.s3_production import prepare_qualified_s3_mesh, S3OwnerAuthorityError
    from test_s3_production import _square
    geometry, face = _square()
    mesh = Mesh(nodes={1: np.array((0., 0., 0.)), 2: np.array((1., 0., 0.)),
                       3: np.array((0., 1., 0.))}, tris={1: (1, 2, 3)},
                elements_of_face={face: [1], 'bad': []})
    with pytest.raises(S3OwnerAuthorityError, match='integer identifier'):
        prepare_qualified_s3_mesh(mesh, geometry, target_policy=None)


@pytest.fixture(scope='module')
def review_actual_candidate():
    from anymesher.hybrid import generate_hybrid_mesh_result
    from test_nurbs_curvilinear_consumer import patch_model, options
    source, _, _ = patch_model(warp=0.)
    points = [source.add_point(*point) for point in
              ((2., 0., 0.), (3., 0., 0.), (3., 1., 0.), (2., 1., 0.))]
    edges = [source.add_line(points[i], points[(i+1)%4]) for i in range(4)]
    ordinary = source.add_face(edges, surface=owner.Plane((2., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
    source.add_sheet((ordinary,))
    request = options()
    request.update(target_size=2., recombine=False, mutation_policy='working_copy')
    candidate = generate_hybrid_mesh_result(source, _defer_finalization=True, **request)
    return source, candidate, ordinary


@pytest.mark.parametrize('attack', ['missing', 'planar', 'deferred_s3'])
def test_actual_finalizer_requires_live_curved_cells_before_generic_paths(review_actual_candidate, monkeypatch, attack):
    from anymesher import hybrid
    source, candidate, ordinary = review_actual_candidate
    original = candidate.mesh.elements_of_face
    deferred = candidate.defer_qualified_s3
    before = owner.to_dict(source)
    def refuse(*args, **kwargs):
        pytest.fail('unproved curved cells cannot enter projection or deferred repair')
    monkeypatch.setattr(hybrid, 'remap_prepared_mesh_associations', refuse)
    monkeypatch.setattr(hybrid, '_published_boundary_registry', refuse)
    monkeypatch.setattr(hybrid, 'prepare_qualified_s3_mesh', refuse)
    try:
        if attack == 'missing':
            candidate.mesh.elements_of_face = {}
        elif attack == 'planar':
            candidate.mesh.elements_of_face = {ordinary: list(candidate.mesh.shells)}
        else:
            candidate.defer_qualified_s3 = True
        with pytest.raises(MeshError, match='exactly one|owner-proved|fresh owner cell'):
            hybrid._finalize_hybrid_candidate(candidate)
    finally:
        candidate.mesh.elements_of_face = original
        candidate.defer_qualified_s3 = deferred
    assert owner.to_dict(source) == before


def test_quad_only_source_record_does_not_claim_s3_faceuse_normals():
    # Construct a literal quad and obtain genuine owner coverage authority;
    # native recombination is free to retain triangles and is not this test.
    from anymesher.mesh import Mesh
    from anymesher.boundary import GlobalEdgeBoundaryRegistry
    from anymesher.meshing_view import GeometryMeshingView
    from anymesher._material_region_binding import retain_curvilinear_reference_candidates
    from test_nurbs_curvilinear_consumer import patch_model
    source, face, _ = patch_model(warp=0.)
    prepared = owner.clone_prepared_geometry(source)
    uv = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
    xyz = tuple((*point, 0.) for point in uv)
    loop = prepared.faces[face].loop
    regions = owner.query_material_surface_regions(prepared, (face,))
    chords = owner.query_prepared_material_boundary_chords(prepared, regions,
        tuple((prepared.handle('edge', use.edge), 0., 1.) for use in loop), chord_error=.01)
    cell = owner.CurvilinearMaterialCell(prepared.handle('face', face), (1, 2, 3, 4),
        uv, xyz, tuple((i, i, True) for i in range(4)))
    receipt = owner.validate_prepared_material_curvilinear_cells(prepared, regions,
        chords, (cell,), surface_error=.001)
    mesh = Mesh(geometry_model_id=prepared.model_id, geometry_revision=prepared.revision,
        nodes={i+1: np.asarray(point) for i, point in enumerate(xyz)},
        quads={1: (1, 2, 3, 4)}, elements_of_face={face: [1]})
    registry = GlobalEdgeBoundaryRegistry(GeometryMeshingView(prepared))
    mesh.node_of_vertex = {prepared.edges[use.edge].start: i+1 for i, use in enumerate(loop)}
    for use in loop:
        edge = prepared.edges[use.edge]
        nodes = [mesh.node_of_vertex[edge.start], mesh.node_of_vertex[edge.end]]
        mesh.nodes_of_edge[edge.id] = nodes
        for parameter, node in zip((0., 1.), nodes):
            registry.register(edge.id, parameter, mesh.nodes[node], node_id=node,
                              owner=prepared.handle('edge', edge.id))
    mesh.boundary_registry = registry
    retain_curvilinear_reference_candidates(mesh, ((prepared, regions, chords, receipt),),
                                           boundary_registry=registry)
    _publish_identity_source_transfer(source, prepared, mesh, mesh.boundary_registry)
    record = _qualify_identity_source_s3(source, prepared, mesh)
    assert record['status'] == 'NOT_APPLICABLE_NO_TRIANGLES'
    assert record['authority_model']['scope'] != 'SOURCE_GEOMETRY_ORIENTED_SHEET_FACE_USE'


def test_public_readonly_nurbs_mesh_and_source_s3_remain_admitted_after_review_guards():
    from anymesher.hybrid import generate_hybrid_mesh_result
    from test_nurbs_curvilinear_consumer import patch_model, options
    import anygeometry.source_cell_complex as complex_owner
    source, face, _ = patch_model(warp=0.)
    before = owner.to_dict(source)
    request = options()
    request.update(target_size=2., recombine=False, geometry_approximation_error=.0025,
                   qualified_s3=True)
    actual_proof, full_proofs = complex_owner.certify_source_cells, []
    def observed_proof(*args, **kwargs):
        full_proofs.append(1)
        return actual_proof(*args, **kwargs)
    with pytest.MonkeyPatch.context() as observe:
        observe.setattr(complex_owner, 'certify_source_cells', observed_proof)
        result = generate_hybrid_mesh_result(source, **request)
    assert len(full_proofs) == 1
    mesh = result.mesh
    record = mesh.structural_preparation['qualified_s3']
    assert mesh.tris and mesh.elements_of_face[face]
    assert mesh.geometry_model_id == source.model_id
    assert mesh.geometry_revision == source.revision
    assert record['status'] == 'ADMITTED'
    assert record['authority_model']['source_model_id'] == str(source.model_id)
    assert record['authority_model']['scope'] == 'SOURCE_GEOMETRY_ORIENTED_SHEET_FACE_USE'
    assert owner.to_dict(source) == before


def test_public_renumbered_closure_source_publication_and_atomic_cancellation():
    from copy import copy
    from types import MappingProxyType
    from anymesher import publish_model_closure_mesh_to_source
    from anymesher.hybrid import generate_hybrid_mesh_result
    from anymesher.material_source_transfer import _publication_pin, _IDENTITY_PUBLICATIONS
    from test_nurbs_curvilinear_consumer import patch_model, options
    source = owner.GeometryModel()
    dummy = source.add_points(((9., 9., 0.), (10., 9., 0.), (10., 10., 0.), (9., 10., 0.)))
    source.add_sheet((source.add_plate(dummy),))
    patch, patch_face, patch_edges = patch_model(warp=0.)
    vertices = [source.add_point(*patch.vertex_position(patch.edges[edge].start)) for edge in patch_edges]
    edges = [source.add_curve(vertices[i], vertices[(i+1)%4], patch.edges[edge].curve)
             for i, edge in enumerate(patch_edges)]
    face = source.add_face(edges, surface=patch.faces[patch_face].surface)
    sheet = source.add_sheet((face,))
    before = owner.to_dict(source)
    closure = owner.extract_model_closure(source, (source.handle('sheet', sheet),))
    work = closure.working_model
    assert next(iter(work.faces)) != face and set(work.edges) != set(edges)
    owner.qualify_model_closure_source(source, closure)
    request = options()
    request.update(target_size=2., recombine=False, geometry_approximation_error=.0025)
    mesh = generate_hybrid_mesh_result(work, mutation_policy='working_copy', **request).mesh
    old_pin, old_registry = _publication_pin(mesh), mesh.boundary_registry
    old_publication = _IDENTITY_PUBLICATIONS.get(id(mesh))
    old_preparation = deepcopy(mesh.structural_preparation)
    old_face_table = mesh.elements_of_face
    for phase in ('closure source qualification complete', 'closure source publication start',
                  'closure source publication complete'):
        with pytest.raises((owner.GeometryError, MeshError), match='cancelled'):
            publish_model_closure_mesh_to_source(source, closure, mesh,
                cancellation_check=lambda current, phase=phase: current == phase)
        assert _publication_pin(mesh) == old_pin and mesh.boundary_registry is old_registry
        assert _IDENTITY_PUBLICATIONS.get(id(mesh)) is old_publication
        assert mesh.structural_preparation == old_preparation and mesh.elements_of_face is old_face_table
        assert owner.to_dict(source) == before
    with pytest.raises(MeshError, match='original live'):
        publish_model_closure_mesh_to_source(source, closure, copy(mesh))
    mapped = dict(closure.work_to_source)
    vertex_keys = [key for key in mapped if key.kind == 'vertex']
    mapped[vertex_keys[0]], mapped[vertex_keys[1]] = mapped[vertex_keys[1]], mapped[vertex_keys[0]]
    forged = replace(closure, work_to_source=MappingProxyType(mapped),
                     source_to_work=MappingProxyType({value: key for key, value in mapped.items()}))
    with pytest.raises(owner.GeometryError):
        publish_model_closure_mesh_to_source(source, forged, mesh)
    assert _publication_pin(mesh) == old_pin
    nodes = {key: tuple(value) for key, value in mesh.nodes.items()}
    assert publish_model_closure_mesh_to_source(source, closure, mesh) is mesh
    assert mesh.geometry_model_id == source.model_id and mesh.geometry_revision == source.revision
    assert set(mesh.elements_of_face) == {face} and set(mesh.nodes_of_edge) == set(edges)
    assert {key: tuple(value) for key, value in mesh.nodes.items()} == nodes
    assert mesh.boundary_registry.view.source is source
    assert mesh.structural_preparation['qualified_s3']['status'] == 'ADMITTED'
    assert mesh.structural_preparation['qualified_s3']['authority_model']['source_model_id'] == str(source.model_id)
    assert set(mesh.structural_preparation['source_to_working_faces']) == {str(face)}
    assert set(mesh.structural_preparation['source_to_working_edges']) == {str(edge) for edge in edges}
    assert owner.to_dict(source) == before
