"""Recovery callback forwarding, intercepted before any numerical backend."""
import pytest

from anygeometry import GeometryModel
import anymesher.recovery as recovery
from anymesher.errors import StructuredQualityRejected


class BackendReached(BaseException):
    """Stop before a mesh or admission result can be manufactured."""


def plate():
    model = GeometryModel()
    model.add_plate(model.add_points(((0.,0.,0.),(1.,0.,0.),(1.,1.,0.),(0.,1.,0.))))
    return model


@pytest.mark.parametrize('explicit_none', (False, True))
def test_recovery_without_deadline_or_callback_preserves_none(monkeypatch, explicit_none):
    seen = []
    def backend(model, **options):
        seen.append(options['cancellation_check'])
        assert options['qualified_s3'] is True
        raise BackendReached()
    monkeypatch.setattr(recovery, 'generate_hybrid_mesh_result', backend)
    options = {'cancellation_check': None} if explicit_none else {}
    with pytest.raises(BackendReached):
        recovery.generate_automatic_mesh_result(plate(), strategy='auto', qualified_s3=True,
            automation=recovery.MeshAutomationOptions(allow_inspection=False), **options)
    assert seen == [None]


@pytest.mark.parametrize('with_callback', (False, True))
def test_recovery_deadline_remains_live_and_precedes_caller_callback(monkeypatch, with_callback):
    clock, phases, seen = [0.], [], []
    monkeypatch.setattr(recovery, 'monotonic', lambda: clock[0])
    def backend(model, **options):
        check = options['cancellation_check']
        assert callable(check)
        seen.append(check)
        clock[0] = 2.
        check('expired backend phase')
        raise AssertionError('expired deadline did not interrupt the backend')
    monkeypatch.setattr(recovery, 'generate_hybrid_mesh_result', backend)
    with pytest.raises(recovery.MeshRecoveryIncomplete, match='expired backend phase'):
        recovery.generate_automatic_mesh_result(plate(), strategy='auto',
            automation=recovery.MeshAutomationOptions(max_seconds=1., allow_inspection=False),
            cancellation_check=phases.append if with_callback else None)
    assert len(seen) == 1
    assert phases == (['source geometry validation', 'automatic meshing: auto'] if with_callback else [])


def test_recovery_callback_without_deadline_keeps_phases_and_exception(monkeypatch):
    phases = []
    class Cancelled(Exception):
        pass
    def callback(phase):
        phases.append(phase)
        if phase == 'backend checkpoint':
            raise Cancelled('caller cancellation')
        return True  # Existing phase callbacks cancel by raising, not return value.
    def backend(model, **options):
        assert callable(options['cancellation_check'])
        options['cancellation_check']('backend checkpoint')
        raise AssertionError('caller exception did not propagate')
    monkeypatch.setattr(recovery, 'generate_hybrid_mesh_result', backend)
    with pytest.raises(Cancelled, match='caller cancellation'):
        recovery.generate_automatic_mesh_result(plate(), strategy='auto',
            cancellation_check=callback,
            automation=recovery.MeshAutomationOptions(allow_inspection=False))
    assert phases == ['source geometry validation', 'automatic meshing: auto', 'backend checkpoint']


@pytest.mark.parametrize('deadline', (None, 1.))
@pytest.mark.parametrize('with_callback', (False, True))
def test_recovery_retry_preserves_callback_selection_and_quality_options(monkeypatch, deadline, with_callback):
    monkeypatch.setattr(recovery, 'monotonic', lambda: 0.)
    seen, phases = [], []
    def backend(model, **options):
        seen.append((options['strategy'], options['cancellation_check']))
        assert options['qualified_s3'] is True
        assert options['order'] == 'linear'
        if deadline is None and not with_callback:
            assert options['cancellation_check'] is None
        else:
            assert callable(options['cancellation_check'])
            options['cancellation_check']('backend ' + options['strategy'])
        if len(seen) == 1:
            raise StructuredQualityRejected('first recipe retains its quality refusal')
        raise BackendReached()
    monkeypatch.setattr(recovery, 'generate_hybrid_mesh_result', backend)
    with pytest.raises(BackendReached):
        recovery.generate_automatic_mesh_result(plate(), strategy='auto', qualified_s3=True, order='linear',
            cancellation_check=phases.append if with_callback else None,
            automation=recovery.MeshAutomationOptions(max_seconds=deadline, allow_inspection=False))
    assert [method for method, check in seen] == ['auto', 'native']
    assert seen[0][1] is seen[1][1]
    assert phases == (['source geometry validation', 'automatic meshing: auto', 'backend auto',
                       'automatic meshing: native', 'backend native'] if with_callback else [])
