"""Callback transport checks only: interrupted engines grant no mesh authority."""
from copy import copy
import pytest
import anygeometry as owner
import anymesher.recovery as recovery
from anymesher.errors import MeshError
from test_recovery import _plate


@pytest.mark.parametrize('kind', ('none','signal','arbitrary'))
@pytest.mark.parametrize('deadline', (False,True))
def test_recovery_callback_transport_identity_and_deadline_guard(monkeypatch, kind, deadline):
    geometry = _plate()
    before = owner.to_dict(geometry)
    callback = None if kind == 'none' else (owner.CancellationSignal() if kind == 'signal' else lambda _phase: None)
    class Reached(BaseException):
        pass
    def inspect(_geometry, **options):
        forwarded = options['cancellation_check']
        if not deadline and kind in ('none','signal'):
            assert forwarded is callback
        else:
            assert callable(forwarded) and forwarded is not callback
        if forwarded is not None:
            assert not forwarded('engine frontier')
        raise Reached()
    monkeypatch.setattr(recovery, 'generate_hybrid_mesh_result', inspect)
    with pytest.raises(Reached):
        recovery.generate_automatic_mesh_result(geometry, target_size=.25, strategy='native',
            automation=recovery.MeshAutomationOptions(strict_method=True,
                max_seconds=10. if deadline else None), cancellation_check=callback)
    assert owner.to_dict(geometry) == before


def test_recovery_passive_cancel_after_candidate_refuses_before_publication(monkeypatch):
    geometry = _plate()
    before = owner.to_dict(geometry)
    signal = owner.CancellationSignal()
    calls = []
    def interrupted(_geometry, **options):
        assert options['cancellation_check'] is signal
        calls.append('detached work')
        signal.cancel()
        return object()  # Must be refused BEFORE result/mesh processing.
    monkeypatch.setattr(recovery, 'generate_hybrid_mesh_result', interrupted)
    with pytest.raises(MeshError, match='cancelled'):
        recovery.generate_automatic_mesh_result(geometry, target_size=.25, strategy='native',
            cancellation_check=signal)
    assert calls == ['detached work'] and owner.to_dict(geometry) == before


def test_recovery_passive_cancel_after_refusal_stops_next_recipe(monkeypatch):
    from anymesher.quad.public_integration import QuadCapabilityMissing
    geometry = _plate()
    signal = owner.CancellationSignal()
    calls = []
    def refuse(_geometry, **_options):
        calls.append('attempt')
        signal.cancel()
        raise QuadCapabilityMissing('detached refusal')
    monkeypatch.setattr(recovery, 'generate_hybrid_mesh_result', refuse)
    with pytest.raises(MeshError, match='cancelled'):
        recovery.generate_automatic_mesh_result(geometry, target_size=.25, strategy='auto',
            cancellation_check=signal)
    assert calls == ['attempt']


def test_recovery_deadline_after_candidate_keeps_resource_refusal(monkeypatch):
    clock = [0.]
    monkeypatch.setattr(recovery, 'monotonic', lambda: clock[0])
    def work(*_args, **_kwargs):
        clock[0] = 2.
        return object()
    monkeypatch.setattr(recovery, 'generate_hybrid_mesh_result', work)
    with pytest.raises(recovery.MeshRecoveryIncomplete, match='time budget expired'):
        recovery.generate_automatic_mesh_result(_plate(), target_size=.25, strategy='native',
            automation=recovery.MeshAutomationOptions(max_seconds=1.))


def test_recovery_fake_signal_and_copy_never_gain_passive_identity():
    signal = owner.CancellationSignal()
    with pytest.raises(owner.GeometryError, match='cannot be copied'):
        copy(signal)
    forged = object.__new__(owner.CancellationSignal)
    assert not owner.is_cancellation_signal(forged)
    with pytest.raises(owner.GeometryError, match='not issued'):
        recovery.generate_automatic_mesh_result(_plate(), target_size=.25, strategy='native',
            cancellation_check=forged)


def test_recovery_arbitrary_callback_baseexception_identity_is_preserved():
    class Cancelled(BaseException):
        pass
    failure = Cancelled()
    def callback(_phase):
        raise failure
    with pytest.raises(Cancelled) as caught:
        recovery.generate_automatic_mesh_result(_plate(), target_size=.25, strategy='native',
            cancellation_check=callback)
    assert caught.value is failure

@pytest.mark.parametrize('limit', ('signal','deadline'))
@pytest.mark.parametrize('stage', ('final_refusal','quality'))
def test_cached_inspection_cannot_publish_after_final_refusal_or_quality_limit(monkeypatch, limit, stage):
    """Real small completed mesh; injected refusals exercise cache transport only."""
    from anymesher.s3_quality import S3QualityError
    from anymesher.quad.public_integration import QuadCapabilityMissing
    geometry = _plate()
    before = owner.to_dict(geometry)
    completed = recovery.generate_hybrid_mesh_result(geometry, target_size=1.,
        strategy='native', order='linear', recombine=False, native_backend='python')
    signal = owner.CancellationSignal()
    clock = [0.]
    monkeypatch.setattr(recovery, 'monotonic', lambda: clock[0])
    def stop():
        if limit == 'signal':
            signal.cancel()
        else:
            clock[0] = 2.
    calls = []
    def refuse(_geometry, **_options):
        calls.append('recipe')
        if len(calls) == 1:
            failure = S3QualityError('injected admission refusal with real inspection candidate')
            failure.inspectable_result = completed
            raise failure
        if stage == 'final_refusal':
            stop()
        raise QuadCapabilityMissing('final detached recipe refusal')
    actual_quality = recovery.verify_mesh_quality
    quality_calls = []
    def inspect_quality(mesh):
        quality_calls.append(mesh)
        assert stage == 'quality'  # Pre-publication check must stop earlier case.
        result = actual_quality(mesh)
        stop()
        return result
    monkeypatch.setattr(recovery, 'generate_hybrid_mesh_result', refuse)
    monkeypatch.setattr(recovery, 'verify_mesh_quality', inspect_quality)
    expected = MeshError if limit == 'signal' else recovery.MeshRecoveryIncomplete
    with pytest.raises(expected, match='cancelled' if limit == 'signal' else 'time budget expired'):
        recovery.generate_automatic_mesh_result(geometry, target_size=1., strategy='auto',
            cancellation_check=signal if limit == 'signal' else None,
            automation=recovery.MeshAutomationOptions(allow_inspection=True,
                max_seconds=1. if limit == 'deadline' else None))
    assert calls == ['recipe','recipe']
    assert len(quality_calls) == (1 if stage == 'quality' else 0)
    assert 'automation' not in completed.mesh.hybrid_diagnostics
    assert owner.to_dict(geometry) == before
