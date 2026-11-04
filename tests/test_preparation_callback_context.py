"""Actual preparation callers keep closure cancellation after chart warming."""
import pytest
from anygeometry import GeometryModel, query_trimmed_surface_charts, to_dict
from anymesher.preparation import prepare_structural_closure


def crossing_plates():
    model = GeometryModel()
    for points in (
        ((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)),
        ((-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1)),
    ):
        face = model.add_plate(model.add_points(points))
        model.add_sheet((face,))
    query_trimmed_surface_charts(model, tuple(model.faces))
    return model


def test_warmed_chart_entrypoint_calls_ordinary_closure():
    model = crossing_plates()
    before = to_dict(model)
    phases = []
    working, report = prepare_structural_closure(model, face_ids=tuple(model.faces),
        cancellation_check=lambda phase: phases.append(phase))
    assert 'structural intersection batch' in phases
    assert working.validate_topology() == ()
    assert report.applications == 1
    assert to_dict(model) == before


def test_warmed_chart_entrypoint_preserves_cancellation_identity_and_source():
    model = crossing_plates()
    before = to_dict(model)
    failure = RuntimeError('ordinary closure cancellation')
    def cancel(phase):
        if phase == 'structural intersection batch':
            raise failure
    with pytest.raises(RuntimeError) as caught:
        prepare_structural_closure(model, face_ids=tuple(model.faces), cancellation_check=cancel)
    assert caught.value is failure
    assert to_dict(model) == before


def test_none_callback_does_not_create_batch_closure(monkeypatch):
    from anymesher import preparation
    failure=RuntimeError('stop before numerical call')
    def observe(model,operands,*,policy):
        assert policy.cancellation_check is None
        raise failure
    monkeypatch.setattr(preparation,'plan_intersections',observe)
    with pytest.raises(RuntimeError) as caught:prepare_structural_closure(crossing_plates())
    assert caught.value is failure


def test_private_plan_observer_is_inactive_and_nested():
    from types import SimpleNamespace
    from anymesher import preparation
    class Unreadable:
        @property
        def content_checksum(self):pytest.fail('inactive observation')
    preparation._record_preparation_plan(Unreadable())
    plan=SimpleNamespace(content_checksum='plan')
    with preparation._collect_preparation_evidence() as outer:
        with pytest.raises(RuntimeError):
            with preparation._collect_preparation_evidence() as inner:
                preparation._record_preparation_plan(plan)
                raise RuntimeError('restore')
        preparation._record_preparation_plan(plan)
    assert outer==[{'plan':plan,'plan_checksum':'plan'}]*2 and inner==outer[:1]


_CALLER_PROFILE_CONTRACT='mesh-warm-closure-qualified-v1'
_INITIAL_ADMISSION_REASONS=frozenset((
    'Windows private-commit accounting unavailable',
    'planning resource helper requires Windows CPython with GIL',
    "planning resource helper unavailable: No module named 'anygeometry._planning_native_core'",
    'NumPy array dispatcher interface unavailable',
    'NumPy context owner modules unavailable',
    'NumPy context source/native image outside profile',
    'runtime RLock state interface unavailable',
))


def _qualified_caller_profile():
    import sys
    from importlib.metadata import PackageNotFoundError,version
    import numpy as np
    if sys.platform!='win32' or sys.version_info[:3]!=(3,14,2) or np.__version__!='2.4.6':return False
    try:return version('scipy')=='1.17.1'
    except PackageNotFoundError:return False


def _caller_execution_scope(qualified,required):
    assert not required or qualified,'qualified caller lane requires the immutable Windows/Python/NumPy/SciPy profile'
    return 'original-profile-assertions' if qualified else 'initial-admission-serial-fallback'


def _assert_initial_serial_receipt(receipt):
    # Exact known INITIAL boundaries only. Unknown source/mutation/IO failures
    # remain failures and cannot stand in for an unsupported runtime image.
    assert receipt['schema']=='R5-production-v1' and receipt['ci_verified'] is False
    import sys
    known=receipt['reason'] in _INITIAL_ADMISSION_REASONS or (
        sys.version_info[:2]<(3,14) and receipt['reason']=='infrastructure function global owner')
    assert receipt['disposition']=='serial' and known,receipt
    assert receipt['stages']==receipt['attempted_stages']==[] and receipt['retired'] is True
    assert receipt['costs']['abandoned']=={}
    assert not {'manifest_sha256','model_sha256','infrastructure'}.intersection(receipt)


def test_unavailable_private_accounting_is_initial_serial_with_off_identity(monkeypatch,tmp_path,request):
    from types import SimpleNamespace
    from anygeometry import _planning_memory as memory
    # Exercise the real observer's unsupported branch, not an invented decline.
    # This is a simulated backend boundary, never a native/platform qualification.
    monkeypatch.setattr(memory,'sys',SimpleNamespace(platform='unsupported-platform'))
    reason=_initial_serial_warm_closure_matches_identical_serial_owner(monkeypatch,tmp_path)
    assert reason=='Windows private-commit accounting unavailable'
    request.node.user_properties.extend((('execution_scope','simulated-initial-accounting-refusal'),
        ('postcapture_qualification',False),('numerical_worker_execution',False)))


def test_actual_parallel_warm_closure_matches_identical_serial_owner(monkeypatch,tmp_path,request):
    import os
    qualified=_qualified_caller_profile()
    scope=_caller_execution_scope(qualified,os.environ.get('ANYGEOMETRY_REQUIRE_QUALIFIED_PLANNING_TESTS')=='1')
    if qualified:
        _qualified_parallel_warm_closure_matches_identical_serial_owner(monkeypatch,tmp_path)
    else:
        reason=_initial_serial_warm_closure_matches_identical_serial_owner(monkeypatch,tmp_path)
        request.node.user_properties.append(('initial_admission_reason',reason))
    request.node.user_properties.extend((('caller_planning_contract',_CALLER_PROFILE_CONTRACT),
        ('execution_scope',scope),('original_test_body_executed',qualified),
        ('numerical_worker_execution',qualified),('postcapture_qualification',qualified)))


def _persist_caller_parallel_receipt(tmp_path, receipt):
    """Exact fixture-owned receipt, before any positive assertion can elide it."""
    import json
    text = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False)
    (tmp_path / 'caller-parallel-receipt.json').write_text(text + '\n', encoding='utf-8')
    print('[caller parallel receipt]\n' + text, flush=True)


def _qualified_parallel_warm_closure_matches_identical_serial_owner(monkeypatch,tmp_path):
    import json
    from anygeometry import from_dict
    from anygeometry._planning_code import exact_numerical_record
    from anygeometry.runtime_diagnostics import collect_runtime_records
    from anymesher import preparation
    source=crossing_plates();initial=to_dict(source)
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING','1')
    monkeypatch.setenv('ANYGEOMETRY_PLANNING_SCRATCH_ROOT',str(tmp_path/'production'))
    with preparation._collect_preparation_evidence() as rows,collect_runtime_records() as records:
        working,report=prepare_structural_closure(source,face_ids=tuple(source.faces),reuse_working_copy=True)
    receipt=records['parallel_planning'][-1]
    _persist_caller_parallel_receipt(tmp_path, receipt)
    assert receipt['disposition']=='parallel',receipt
    assert receipt['workers']==receipt['startup_concurrency']==8 and receipt['retired'] is True
    assert receipt['costs']['callback_calls']==0 and len(rows)==1
    raw=exact_numerical_record(rows[0]['plan'])
    application=json.dumps(to_dict(working),sort_keys=True,separators=(',',':'),allow_nan=False)
    reference=from_dict(initial);query_trimmed_surface_charts(reference,tuple(reference.faces))
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING','0')
    with preparation._collect_preparation_evidence() as control:
        serial,serial_report=prepare_structural_closure(reference,face_ids=tuple(reference.faces),reuse_working_copy=True)
    assert len(control)==1 and raw==exact_numerical_record(control[0]['plan'])
    assert rows[0]['plan_checksum']==control[0]['plan_checksum']
    assert application==json.dumps(to_dict(serial),sort_keys=True,separators=(',',':'),allow_nan=False)
    assert report.to_dict()==serial_report.to_dict()
    assert _surface_arrays(working)==_surface_arrays(serial)


def _initial_serial_warm_closure_matches_identical_serial_owner(monkeypatch,tmp_path):
    """Actual caller fallback plus OFF identity; no native/postcapture claim."""
    import json
    from anygeometry import from_dict
    from anygeometry._planning_code import exact_numerical_record
    from anygeometry.runtime_diagnostics import collect_runtime_records
    from anymesher import preparation
    source=crossing_plates();initial=to_dict(source)
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING','1')
    monkeypatch.setenv('ANYGEOMETRY_PLANNING_SCRATCH_ROOT',str(tmp_path/'production'))
    with preparation._collect_preparation_evidence() as rows,collect_runtime_records() as records:
        working,report=prepare_structural_closure(source,face_ids=tuple(source.faces),reuse_working_copy=True)
    receipt=records['parallel_planning'][-1]
    _persist_caller_parallel_receipt(tmp_path, receipt)
    _assert_initial_serial_receipt(receipt)
    assert receipt['costs']['callback_calls']==0 and len(rows)==1
    # This explicit working-copy API finalizes its owned input. The captured
    # initial serialization supplies the independent OFF control below.
    assert report.applications==1 and working is source
    raw=exact_numerical_record(rows[0]['plan'])
    application=json.dumps(to_dict(working),sort_keys=True,separators=(',',':'),allow_nan=False)
    reference=from_dict(initial);query_trimmed_surface_charts(reference,tuple(reference.faces))
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING','0')
    with preparation._collect_preparation_evidence() as control,collect_runtime_records() as off_records:
        serial,serial_report=prepare_structural_closure(reference,face_ids=tuple(reference.faces),reuse_working_copy=True)
    assert off_records.get('parallel_planning',[])==[]
    assert len(control)==1 and raw==exact_numerical_record(control[0]['plan'])
    assert rows[0]['plan_checksum']==control[0]['plan_checksum']
    assert application==json.dumps(to_dict(serial),sort_keys=True,separators=(',',':'),allow_nan=False)
    assert report.to_dict()==serial_report.to_dict()
    assert _surface_arrays(working)==_surface_arrays(serial)
    assert json.dumps(to_dict(source),sort_keys=True,separators=(',',':'),allow_nan=False)==application
    return receipt['reason']


def _surface_arrays(model):
    import numpy as np
    values=[(face_id,name,value) for face_id,face in model.faces.items()
        for name,value in sorted(vars(face.surface).items()) if type(value) is np.ndarray]
    rows=[(face_id,name,value.dtype.str,value.shape,value.tobytes(),bool(value.flags.writeable),
           bool(value.flags.owndata),bool(value.flags.c_contiguous),value.base is None)
        for face_id,name,value in values]
    aliases=[(i,j,first is second,bool(np.shares_memory(first,second,max_work=1000)))
        for i,(_,_,first) in enumerate(values) for j,(_,_,second) in enumerate(values[:i])]
    return rows,aliases


@pytest.mark.parametrize('authority',['extra','hook','features'])
def test_original_clone_decline_forces_serial_and_restores_scope(monkeypatch,tmp_path,authority):
    from anygeometry import _planning_controller as controller
    from anygeometry.runtime_diagnostics import collect_runtime_records
    model=crossing_plates()
    if authority=='extra':model.foreign_authority=True
    elif authority=='hook':model._change_hooks.append(lambda *args:None)
    else:model.features._next_id=2
    before=_surface_arrays(model)
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING','1')
    monkeypatch.setenv('ANYGEOMETRY_PLANNING_SCRATCH_ROOT',str(tmp_path/'production'))
    with collect_runtime_records() as records:
        working,report=prepare_structural_closure(model,face_ids=tuple(model.faces))
    receipt=records['parallel_planning'][-1]
    assert receipt['disposition']=='serial' and 'caller clone preflight:' in receipt['reason']
    assert receipt['attempted_stages']==[] and report.applications==1
    assert controller._INITIAL_DECLINE.get() is None and _surface_arrays(model)==before
