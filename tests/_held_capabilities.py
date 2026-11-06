"""Narrow release exclusions explicitly approved by the owner; no production use."""
from functools import wraps
import pytest
from anymesher.errors import MeshError


class HeldExtrusionContact(Exception):
    """Only the reproduced boundary-contact refusal, never arbitrary mesh errors."""


def held_extrusion_contact(pair):
    expected = ('positive-area overlap requires Fragment Overlaps to select ownership before meshing: '
                f'intersection batch face:{pair[0]}/face:{pair[1]}: '
                'positive-area overlap requires an explicit ownership policy')
    def decorate(original):
        @wraps(original)
        def run(*args, **kwargs):
            try:
                return original(*args, **kwargs)
            except MeshError as error:
                if type(error) is not MeshError or str(error) != expected:
                    raise
                raise HeldExtrusionContact(expected) from error
        return pytest.mark.xfail(
            strict=True, raises=HeldExtrusionContact,
            reason='HELD-GEOMETRY-EXTRUSION-CONTACT: see docs/HELD_CAPABILITIES.md')(run)
    return decorate
