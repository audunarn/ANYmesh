"""A held capability never masks unrelated failures or a new positive result."""
import pytest

from anymesher.errors import MeshError
from _held_capabilities import HeldExtrusionContact, held_extrusion_contact


def _message(pair):
    return ('positive-area overlap requires Fragment Overlaps to select ownership before meshing: '
            f'intersection batch face:{pair[0]}/face:{pair[1]}: '
            'positive-area overlap requires an explicit ownership policy')


def test_exact_held_refusal_is_classified_and_original_is_retained():
    error = MeshError(_message((2, 3)))
    def original():
        raise error
    wrapped = held_extrusion_contact((2, 3))(original)
    assert wrapped.__wrapped__ is original
    with pytest.raises(HeldExtrusionContact) as caught:
        wrapped()
    assert caught.value.__cause__ is error
    mark = wrapped.pytestmark[0]
    assert mark.name == 'xfail'
    assert mark.kwargs == dict(strict=True, raises=HeldExtrusionContact,
                               reason='HELD-GEOMETRY-EXTRUSION-CONTACT: see docs/HELD_CAPABILITIES.md')


@pytest.mark.parametrize('error', [
    AssertionError('positive geometry assertion'), RuntimeError('cleanup failed'),
    MeshError('new geometry error'), MeshError(_message((3, 5))),
    MeshError(_message((2, 3)) + ' changed'),
])
def test_unrelated_failure_is_never_classified(error):
    def original():
        raise error
    with pytest.raises(type(error)) as caught:
        held_extrusion_contact((2, 3))(original)()
    assert caught.value is error


def test_positive_success_returns_for_pytest_strict_xpass():
    token = object()
    def original():
        return token
    wrapped = held_extrusion_contact((2, 3))(original)
    assert wrapped() is token
    assert wrapped.pytestmark[0].kwargs['strict'] is True
