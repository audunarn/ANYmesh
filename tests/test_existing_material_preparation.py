"""Default same-chart material preparation; genuine owner plans and epochs.

Different-support principal batches and reuse=True source adoption are outside
this slice. Geometry classification/coverage is never fabricated by these tests.
"""
import pytest
import anygeometry as owner
from anygeometry.serialization import to_dict, from_dict
from anymesher import preparation
from test_same_support_material_admission import transformed_pair


def material_source(kind, transport):
    model,pair = transformed_pair(kind,'direct')
    for index,face in enumerate(pair):
        model.update_face_metadata(face,material_assignment=f'section-{index}',
            authored_cell=face)
    if transport == 'load':
        model = from_dict(to_dict(model))
    return model,pair


@pytest.mark.parametrize('kind', ['flat','closed'])
@pytest.mark.parametrize('transport', ['direct','load'])
@pytest.mark.parametrize('options', [None,True])
def test_actual_default_preparation_issues_new_epoch_for_current_material_source(
        monkeypatch, kind, transport, options):
    source,pair = material_source(kind,transport)
    before = to_dict(source)
    shared = set(use.edge for use in source.faces[pair[0]].loop)
    shared.update(use.edge for loop in source.faces[pair[0]].holes for use in loop)
    shared &= {use.edge for loop in (source.faces[pair[1]].loop,*source.faces[pair[1]].holes)
               for use in loop}
    assert shared
    incidence = {edge:tuple(sorted(source.faces_using_edge(edge))) for edge in shared}
    actual_clone = owner.clone_prepared_geometry
    actual_plan,actual_apply = preparation.plan_intersections,preparation.apply_intersections
    clones,plans,applications = [],[],[]
    def clone(model,**kwargs):
        result = actual_clone(model,**kwargs)
        clones.append((model,result,kwargs))
        return result
    def plan(model,operands,**kwargs):
        assert kwargs['policy'].cancellation_check is None
        result = actual_plan(model,operands,**kwargs)
        plans.append((model,tuple(operands),result))
        return result
    def apply(model,plan,**kwargs):
        assert kwargs['policy'].cancellation_check is None
        result = actual_apply(model,plan,**kwargs)
        applications.append((model,plan,result))
        return result
    monkeypatch.setattr(owner,'clone_prepared_geometry',clone)
    monkeypatch.setattr(preparation,'plan_intersections',plan)
    monkeypatch.setattr(preparation,'apply_intersections',apply)
    working,report = preparation.prepare_structural_closure(source,options=options)
    assert len(clones) == 1 and clones[0][0] is source and clones[0][1] is working
    assert clones[0][2] == {'new_preparation_epoch':True}
    assert working is not source and working.model_id != source.model_id
    assert plans and applications and all(row[0] is working for row in plans+applications)
    assert set((handle.kind,handle.id) for handle in plans[0][1]) == {
        ('face',face) for face in source.faces}
    assert owner.has_current_intersection_preparation(working,face_ids=tuple(source.faces))
    preimages = owner.query_prepared_face_preimages(working,face_ids=tuple(source.faces))
    owner.validate_prepared_face_preimages_binding(working,preimages)
    assert preimages.authored_face_ids == tuple(sorted(source.faces))
    assert dict(preimages.face_descendants) == {face:(face,) for face in source.faces}
    scope = owner.query_prepared_model_scope(working)
    owner.validate_prepared_model_scope_binding(working,scope)
    # Public readout authenticates complete immediate authored contents. It
    # does not itself assert cell/source-transfer or qualified-S3 admission.
    authored_document = scope.authored_document
    working_document = to_dict(working)
    for key in ('faces','edges','vertices','sheets','face_uses','coedges'):
        assert authored_document.get(key) == before.get(key)
        assert working_document.get(key) == before.get(key)
    assert dict(report.source_to_working_faces) == {face:(face,) for face in source.faces}
    assert dict(report.source_to_working_edges) == {edge:(edge,) for edge in source.edges}
    assert {edge:tuple(sorted(working.faces_using_edge(edge))) for edge in shared} == incidence
    assert to_dict(source) == before


@pytest.mark.parametrize('stop', ['plan','apply'])
def test_owner_batch_callback_failure_preserves_loaded_source(monkeypatch,stop):
    source,_pair = material_source('flat','load')
    before = to_dict(source)
    actual_plan,actual_apply = preparation.plan_intersections,preparation.apply_intersections
    stage = []
    batch_callbacks = []
    failure = RuntimeError(f'cancel actual owner {stop}')
    def plan(*args,**kwargs):
        assert kwargs['policy'].cancellation_check is not None
        batch_callbacks.append(kwargs['policy'].cancellation_check)
        stage[:] = ['plan']
        return actual_plan(*args,**kwargs)
    def apply(*args,**kwargs):
        assert kwargs['policy'].cancellation_check is not None
        assert kwargs['policy'].cancellation_check is batch_callbacks[0]
        stage[:] = ['apply']
        return actual_apply(*args,**kwargs)
    def callback(phase):
        if stage == [stop] and phase == 'structural intersection batch':
            raise failure
    monkeypatch.setattr(preparation,'plan_intersections',plan)
    monkeypatch.setattr(preparation,'apply_intersections',apply)
    with pytest.raises(RuntimeError) as caught:
        preparation.prepare_structural_closure(source,cancellation_check=callback)
    assert caught.value is failure and stage == [stop]
    assert to_dict(source) == before


def test_actual_overlapping_source_refusal_does_not_publish_or_mutate_source():
    from test_prepared_partition_admission import partition_pair
    _original,source,_pair = partition_pair(clone=False,extra_parent=True)
    source = from_dict(to_dict(source))
    before = to_dict(source)
    with pytest.raises((owner.GeometryError,preparation.MeshError)):
        preparation.prepare_structural_closure(source)
    assert to_dict(source) == before


@pytest.mark.parametrize('legacy_boundary', ['reuse','unselected_plane'])
def test_new_epoch_does_not_broaden_reserved_or_unselected_routes(monkeypatch,legacy_boundary):
    source,pair = material_source('flat','load')
    if legacy_boundary == 'unselected_plane':
        source.add_plate(source.add_points(((2.,0.,0.),(3.,0.,0.),(3.,1.,0.),(2.,1.,0.))))
    before = to_dict(source)
    class LegacyPlanningReached(Exception):
        pass
    def forbidden(*args,**kwargs):
        pytest.fail('new epoch crossed reserved/unsupported clone boundary')
    def observe(working,operands,**kwargs):
        assert working is not source
        assert kwargs['policy'].cancellation_check is not None
        if legacy_boundary == 'reuse':
            assert working.model_id == source.model_id
        raise LegacyPlanningReached()
    monkeypatch.setattr(owner,'clone_prepared_geometry',forbidden)
    monkeypatch.setattr(preparation,'plan_intersections',observe)
    with pytest.raises(LegacyPlanningReached):
        preparation.prepare_structural_closure(source,face_ids=pair,
            reuse_working_copy=legacy_boundary == 'reuse')
    assert to_dict(source) == before


def test_public_default_loaded_flat_mesh_retains_shared_trace_and_source_s3(monkeypatch):
    from anymesher.hybrid import generate_hybrid_mesh_result
    from test_nurbs_curvilinear_consumer import options
    source,pair = material_source('flat','load')
    before = to_dict(source)
    shared = {edge for edge in source.edges if set(source.faces_using_edge(edge)) == set(pair)}
    assert shared
    actual_cells = owner.validate_prepared_material_curvilinear_cells
    actual_transfer = owner.query_prepared_material_source_transfer
    cell_proofs,transfers = [],[]
    def cells(*args,**kwargs):
        receipt = actual_cells(*args,**kwargs)
        cell_proofs.append(receipt)
        return receipt
    def transfer(*args,**kwargs):
        receipt = actual_transfer(*args,**kwargs)
        transfers.append((args[0],args[1],receipt))
        return receipt
    monkeypatch.setattr(owner,'validate_prepared_material_curvilinear_cells',cells)
    monkeypatch.setattr(owner,'query_prepared_material_source_transfer',transfer)
    request = options()
    request.update(target_size=.25,order='linear',certification_mode='interactive',
        geometry_approximation_error=.0025,qualified_s3=True)
    request.pop('structural_preparation')  # actual public default automatic path
    result = generate_hybrid_mesh_result(source,**request)
    mesh = result.mesh
    assert cell_proofs and transfers
    assert any(original is source and prepared is not source
               for original,prepared,_receipt in transfers)
    assert mesh.structural_preparation['qualified_s3']['status'] == 'ADMITTED'
    assert mesh.structural_preparation['qualified_s3']['authority_model']['scope'] == 'SOURCE_GEOMETRY_ORIENTED_SHEET_FACE_USE'
    assert mesh.geometry_model_id == source.model_id and mesh.geometry_revision == source.revision
    assert set(mesh.elements_of_face) == set(source.faces)
    face_nodes = {face:{node for element in mesh.elements_of_face[face]
        for node in mesh.shells[element]} for face in pair}
    for edge in shared:
        stations = mesh.nodes_of_edge[edge]
        assert len(stations) >= 2 and len(set(stations)) == len(stations)
        assert set(stations) <= face_nodes[pair[0]] & face_nodes[pair[1]]
    assert to_dict(source) == before
