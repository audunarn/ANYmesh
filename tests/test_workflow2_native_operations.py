"""Tiny compiled-operation witnesses; no meshing or optional provider calls."""
import importlib.machinery
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    any(os.environ.get(name) != '1' for name in ('ANYGEOMETRY_REQUIRE_NATIVE', 'ANYMESHER_REQUIRE_NATIVE')),
    reason='compiled-operation witness requires the explicit native compatibility profile',
)


def test_required_geometry_native_plane_kernel():
    import anygeometry
    from anygeometry import backend
    import anygeometry._core as core

    assert anygeometry.backend_info()['backend'] == 'native'
    assert isinstance(core.__loader__, importlib.machinery.ExtensionFileLoader)
    assert 'site-packages' in Path(core.__file__).parts
    assert backend.plane_evaluate is core.plane_evaluate
    # frame=(origin, first basis, second basis): exact integer oracle.
    frame = np.array([1., 2., 3., 1., 0., 0., 0., 1., 0.])
    uv = np.array([[0., 0.], [2., 3.]], dtype=np.float64)
    actual = np.empty((2, 3), dtype=np.float64)
    backend.plane_evaluate(frame, uv, actual)
    np.testing.assert_array_equal(actual, [[1., 2., 3.], [3., 5., 3.]])


def test_required_mesher_native_orientation_kernel():
    import anymesher._native as native
    from anymesher import native_cpp

    assert isinstance(native.__loader__, importlib.machinery.ExtensionFileLoader)
    assert 'site-packages' in Path(native.__file__).parts
    assert native_cpp.NATIVE_CPP_AVAILABLE
    assert native_cpp._compiled is native
    assert native.orient2d(0., 0., 2., 0., 0., 3.) == 6.
    assert native.orient2d(0., 0., 0., 3., 2., 0.) == -6.
    assert native_cpp.orient2d((0., 0.), (2., 0.), (0., 3.)) == 6.
