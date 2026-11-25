"""The exact global-seed element-limit refusal is a typed, picklable MeshError (same text, same number)."""

from __future__ import annotations

import pickle

import pytest

from anygeometry.generators.structural import cylinder
from anymesher import MeshError
from anymesher.hybrid import generate_hybrid_mesh_result


def _mesh(limit):
    options = {} if limit is None else {"maximum_estimated_elements": limit}
    return generate_hybrid_mesh_result(
        cylinder(1.0, 2.0, circumferential_segments=8, longitudinal_spacing=1.0),
        target_size=0.25, strategy="auto", structured_options=options, native_backend="python")


def test_limit_refusal_is_typed_and_exact():
    from anymesher.errors import StructuredElementLimitExceeded

    accepted = _mesh(None).mesh
    actual = len(accepted.quads) + len(accepted.tris)
    assert len(_mesh(actual).mesh.quads) == len(accepted.quads)  # the limit itself is admitted
    with pytest.raises(StructuredElementLimitExceeded) as caught:
        _mesh(actual - 1)
    error = caught.value
    assert isinstance(error, MeshError)
    assert (error.estimated, error.maximum) == (actual, actual - 1)
    assert str(error) == (f"global seed solution estimates {actual} structured elements, exceeding "
                          f"maximum_estimated_elements={actual - 1}")


def test_limit_refusal_survives_pickling_for_worker_transport():
    from anymesher.errors import StructuredElementLimitExceeded

    error = StructuredElementLimitExceeded(6031842, 2000000)
    copy = pickle.loads(pickle.dumps(error))
    assert type(copy) is StructuredElementLimitExceeded
    assert (copy.estimated, copy.maximum, str(copy), copy.args) == (
        6031842, 2000000, str(error), error.args)
