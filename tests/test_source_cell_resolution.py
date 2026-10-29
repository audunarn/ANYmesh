"""Bounded consumer recovery; mocks test control flow, never source admission."""
from fractions import Fraction as Q
from types import SimpleNamespace as NS

import pytest
import anygeometry as owner
import anymesher.hybrid as hybrid
from anymesher.errors import MeshError
from anymesher._material_region_binding import (
    _SourceCellResolutionRequired, _is_source_cell_resolution_request,
)


def test_only_consistent_current_owner_signal_can_request_resolution(monkeypatch):
    cell = NS(face=NS(id=7), node_ids=(11, 12, 13))
    limit = Q(.0025)
    signal = owner.SourceCellRefinementRequired(7, cell.node_ids, limit, 2*limit**2)
    assert _is_source_cell_resolution_request(signal, (cell,), .0025)
    transported = owner.SourceCellRefinementRequired(7, cell.node_ids,
        limit-Q(1, 10**12), 2*limit**2, requested_error=limit)
    assert _is_source_cell_resolution_request(transported, (cell,), .0025)
    assert not _is_source_cell_resolution_request(
        owner.GeometryError(str(signal)), (cell,), .0025)
    for field, value in [('face_id', 8), ('node_ids', (12, 11, 13)),
                         ('node_ids', (True, 12, 13)), ('error_limit', Q(.005)),
                         ('requested_error', Q(.005)), ('error_limit', Q(0)),
                         ('lower_error_squared', limit**2),
                         ('lower_error_squared', float(2*limit**2))]:
        altered = owner.SourceCellRefinementRequired(7, cell.node_ids, limit, 2*limit**2)
        setattr(altered, field, value)
        assert not _is_source_cell_resolution_request(altered, (cell,), .0025)
    monkeypatch.delattr(owner, 'SourceCellRefinementRequired')
    assert not _is_source_cell_resolution_request(signal, (cell,), .0025)


def state():
    # Mutable roots deliberately retain aliases, as do real mesh/seed objects.
    nodes, accepted_face, stations, reservations = {1: 'boundary'}, [8], [1], [1]
    mesh = {'nodes': nodes, 'accepted_face': accepted_face}
    registry = {'stations': stations}
    seeds = {'reservations': reservations}
    receipt = object()
    binding = NS(requires_curvilinear=True, _curvilinear_state={'cells': receipt},
        validate=lambda _check: None)
    options = dict(_material_region_binding=binding, boundary_registry=registry,
        component_seed_registry=seeds, size_field=NS(target_size=.25))
    return mesh, options, (nodes, accepted_face, stations, reservations, receipt)


def assert_restored(mesh, options, aliases):
    nodes, accepted, stations, reservations, receipt = aliases
    assert mesh['nodes'] is nodes and nodes == {1: 'boundary'}
    assert mesh['accepted_face'] is accepted and accepted == [8]
    assert options['boundary_registry']['stations'] is stations and stations == [1]
    assert options['component_seed_registry']['reservations'] is reservations
    assert reservations == [1]
    assert options['_material_region_binding']._curvilinear_state == {'cells': receipt}
    assert options['size_field'].target_size == .25


def provisional_mutation(mesh, options):
    mesh['nodes'][2] = 'provisional'
    mesh['accepted_face'].append(9)  # a tentative split of an accepted neighbour
    options['boundary_registry']['stations'].append(2)
    options['component_seed_registry']['reservations'].append(2)
    options['_material_region_binding']._curvilinear_state['cells'] = object()


def test_retry_restores_aliases_receipts_and_shared_ids_without_nominal_change(monkeypatch):
    mesh, options, aliases = state()
    scales = []
    def attempt(geometry, actual_mesh, face_id, **kw):
        assert actual_mesh is mesh and face_id == 7
        assert_restored(mesh, options, aliases)
        scales.append(kw['_interior_resolution_scale'])
        provisional_mutation(mesh, options)
        if len(scales) < 3:
            raise _SourceCellResolutionRequired('proved too coarse')
        return {'proof': 'final owner proof passed'}
    monkeypatch.setattr(hybrid, '_mesh_native_face_once', attempt)
    diagnostics = hybrid._mesh_native_face(None, mesh, 7, **options)
    assert scales == [1., .5, .25]
    assert diagnostics['source_cell_resolution'] == {
        'refinement_retries': 2, 'interior_scale': .25, 'localized_failure_zones': 0}
    assert mesh['nodes'][2] == 'provisional'
    assert options['size_field'].target_size == .25


@pytest.mark.parametrize('failure', [
    _SourceCellResolutionRequired('too coarse'),
    owner.GeometryError('generic owner refusal'), MeshError('quality refused'),
    RuntimeError('cancel callback'), KeyboardInterrupt(),
])
def test_failure_cap_and_hard_refusals_restore_all_attempt_state(monkeypatch, failure):
    mesh, options, aliases = state()
    calls = []
    def attempt(*args, **kw):
        assert_restored(mesh, options, aliases)
        calls.append(kw['_interior_resolution_scale'])
        provisional_mutation(mesh, options)
        raise failure
    monkeypatch.setattr(hybrid, '_mesh_native_face_once', attempt)
    with pytest.raises(type(failure)) as caught:
        hybrid._mesh_native_face(None, mesh, 7, **options)
    assert caught.value is failure
    assert calls == ([1., .5, .25, .125]
        if isinstance(failure, _SourceCellResolutionRequired) else [1.])
    assert_restored(mesh, options, aliases)


@pytest.mark.parametrize('phase', ['entry', 'native'])
def test_private_retry_exception_from_callback_propagates_once(monkeypatch, phase):
    mesh, options, aliases = state()
    failure = _SourceCellResolutionRequired('caller constructed control class')
    callbacks, attempts = [], []
    def callback(label):
        callbacks.append(label)
        if phase == 'entry' or label == 'native checkpoint':
            raise failure
    def attempt(*args, **kw):
        attempts.append(True)
        provisional_mutation(mesh, options)
        kw['cancellation_check']('native checkpoint')
    options['cancellation_check'] = callback
    monkeypatch.setattr(hybrid, '_mesh_native_face_once', attempt)
    with pytest.raises(_SourceCellResolutionRequired) as caught:
        hybrid._mesh_native_face(None, mesh, 7, **options)
    assert caught.value is failure
    assert len(attempts) == (0 if phase == 'entry' else 1)
    assert len(callbacks) == (1 if phase == 'entry' else 2)
    assert_restored(mesh, options, aliases)


@pytest.mark.parametrize('mutation', ['nodes', 'stations', 'seeds', 'receipt'])
def test_attempt_entry_callback_mutation_refuses_and_restores(monkeypatch, mutation):
    mesh, options, aliases = state()
    def callback(_phase):
        if mutation == 'nodes':
            mesh['nodes'][2] = 'foreign'
        elif mutation == 'stations':
            options['boundary_registry']['stations'].append(2)
        elif mutation == 'seeds':
            options['component_seed_registry']['reservations'].append(2)
        else:
            options['_material_region_binding']._curvilinear_state['cells'] = object()
    options['cancellation_check'] = callback
    monkeypatch.setattr(hybrid, '_mesh_native_face_once',
        lambda *args, **kwargs: pytest.fail('mutated attempt started'))
    with pytest.raises(MeshError, match='changed during callback'):
        hybrid._mesh_native_face(None, mesh, 7, **options)
    assert_restored(mesh, options, aliases)


@pytest.mark.parametrize('result', [None, False, object()])
def test_entry_void_false_or_unrelated_return_object_is_not_cancellation(monkeypatch, result):
    mesh, options, _ = state()
    options['cancellation_check'] = lambda phase: result
    def attempt(*args, **kw):
        assert kw['cancellation_check']('native checkpoint') is None
        return {}
    monkeypatch.setattr(hybrid, '_mesh_native_face_once', attempt)
    assert hybrid._mesh_native_face(None, mesh, 7, **options) == {}


@pytest.mark.parametrize('cancelled', [False, True])
def test_entry_passive_signal_cancellation_and_identity(monkeypatch, cancelled):
    mesh, options, aliases = state()
    signal = owner.CancellationSignal()
    if cancelled:
        signal.cancel()
    options['cancellation_check'] = signal
    def attempt(*args, **kw):
        assert kw['cancellation_check'] is signal
        return {}
    monkeypatch.setattr(hybrid, '_mesh_native_face_once', attempt)
    if cancelled:
        with pytest.raises(MeshError, match='cancelled'):
            hybrid._mesh_native_face(None, mesh, 7, **options)
    else:
        assert hybrid._mesh_native_face(None, mesh, 7, **options) == {}
    assert_restored(mesh, options, aliases)


def test_cancel_between_attempts_does_not_start_another_mesh(monkeypatch):
    mesh, options, aliases = state()
    calls = []
    def attempt(*args, **kw):
        calls.append(True)
        provisional_mutation(mesh, options)
        raise _SourceCellResolutionRequired('too coarse')
    options['cancellation_check'] = lambda phase: 'attempt 1' in phase
    monkeypatch.setattr(hybrid, '_mesh_native_face_once', attempt)
    with pytest.raises(MeshError, match='cancelled'):
        hybrid._mesh_native_face(None, mesh, 7, **options)
    assert len(calls) == 1
    assert_restored(mesh, options, aliases)


def inconclusive_request(requested=Q(.0025)):
    from uuid import uuid4
    from anygeometry.prepared_material_curvilinear import PreparedMaterialBoundaryChords
    from anygeometry.definition_binding import definition_checksum
    geometry=NS(model_id=uuid4(),revision=7)
    chords=PreparedMaterialBoundaryChords(geometry.model_id,geometry.revision,'a'*64,(),(),float(requested))
    cell=NS(face=NS(id=7,model_id=geometry.model_id),node_ids=(11,12,13))
    error=owner.SourceCellApproximationInconclusive(7,cell.node_ids,requested/2,2*requested,
        requested_error=requested,jacobian_lower=Q(1,4),proof_box=((Q(0),Q(1)),)*2,
        proof_depth=12,source_model_id=geometry.model_id,source_revision=geometry.revision,
        source_checksum=chords.source_checksum,chord_checksum=definition_checksum(chords))
    return error,cell,geometry,chords


@pytest.mark.parametrize('requested',(Q(.0025),Q(.007),Q(1,10000)))
def test_inconclusive_request_is_current_bound_and_not_a_lower_witness(requested):
    error,cell,geometry,chords=inconclusive_request(requested)
    assert _is_source_cell_resolution_request(error,(cell,),requested,geometry=geometry,chords=chords)
    assert not _is_source_cell_resolution_request(error,(cell,),requested)
    assert not isinstance(error,owner.SourceCellRefinementRequired)
    assert not hasattr(error,'lower_error_squared')


@pytest.mark.parametrize('field,value',(
    ('face_id',8),('node_ids',(12,11,13)),('requested_error',Q(.005)),
    ('upper_error',Q(0)),('upper_error',1.),('jacobian_lower',Q(0)),
    ('proof_depth',True),('proof_box',((Q(0),Q(0)),)*2),
    ('source_revision',8),('source_model_id',None),('source_checksum','b'*64),('chord_checksum','b'*64)))
def test_inconclusive_malformed_or_stale_fields_never_request_retry(field,value):
    error,cell,geometry,chords=inconclusive_request()
    setattr(error,field,value)
    assert not _is_source_cell_resolution_request(error,(cell,),Q(.0025),geometry=geometry,chords=chords)


def test_inconclusive_subclass_or_generic_message_has_no_authority():
    error,cell,geometry,chords=inconclusive_request()
    class Derived(owner.SourceCellApproximationInconclusive):pass
    forged=Derived.__new__(Derived)
    forged.__dict__.update(error.__dict__)
    for failure in (forged,owner.GeometryError(str(error))):
        assert not _is_source_cell_resolution_request(failure,(cell,),Q(.0025),geometry=geometry,chords=chords)


def test_inconclusive_request_uses_existing_four_attempt_restore_cap(monkeypatch):
    mesh,options,aliases=state();calls=[]
    error,cell,geometry,chords=inconclusive_request()
    def attempt(*args,**kwargs):
        assert_restored(mesh,options,aliases)
        calls.append(kwargs['_interior_resolution_scale']);provisional_mutation(mesh,options)
        assert _is_source_cell_resolution_request(error,(cell,),Q(.0025),geometry=geometry,chords=chords)
        raise _SourceCellResolutionRequired('inconclusive approximation') from error
    monkeypatch.setattr(hybrid,'_mesh_native_face_once',attempt)
    with pytest.raises(_SourceCellResolutionRequired) as caught:
        hybrid._mesh_native_face(None,mesh,7,**options)
    assert caught.value.__cause__ is error
    assert calls==[1.,.5,.25,.125]
    assert_restored(mesh,options,aliases)


def test_actual_failure_cell_adds_refine_only_zone_without_changing_nominal_limit(monkeypatch):
    from anymesher.refinement import SizeField
    import numpy as np
    model=owner.GeometryModel()
    mesh,options,aliases=state()
    original=SizeField(model,.25)
    options['size_field']=original
    seen=[]
    def attempt(geometry,actual_mesh,face_id,**kw):
        assert_restored(mesh,options,aliases)
        seen.append(kw['size_field'])
        if len(seen)==1:
            request=_SourceCellResolutionRequired('actual owner cell')
            request.face_id=7
            request.cell_xyz=((0.,0.,0.),(1.,0.,0.),(0.,1.,0.))
            raise request
        assert kw['size_field'].target_size==original.target_size
        assert kw['size_field'].size_at(np.array([[1/3,1/3,0.]]))[0]<original.target_size
        return {}
    monkeypatch.setattr(hybrid,'_mesh_native_face_once',attempt)
    result=hybrid._mesh_native_face(model,mesh,7,**options)
    assert len(seen)==2 and not original.zones
    assert result['source_cell_resolution']['localized_failure_zones']==1
