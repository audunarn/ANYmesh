"""Single-point Jacobian fast path must equal the vectorized original exactly."""
from __future__ import annotations

import random

import numpy as np
import pytest

from anymesher.quad.high_order import (
    ElementFamily,
    _jacobian_function,
    _point_gradients,
    certify_mapping_validity,
    shape_gradients,
)


def _reference_jacobian_function(xyz, fam, nref):
    """The 0.5.1 implementation, verbatim."""
    def evaluate(a, b):
        g = np.asarray(shape_gradients(fam, np.array([a, b], dtype=np.float64)))
        dxi = g[:, 0] @ xyz
        deta = g[:, 1] @ xyz
        return float(np.dot(np.cross(dxi, deta), nref))
    return evaluate


_COUNTS = {ElementFamily.Q4: 4, ElementFamily.Q8: 8, ElementFamily.T3: 3, ElementFamily.T6: 6}


def _params(rng):
    values = [-1.0, -0.5, -0.0, 0.0, 1.0 / 3.0, 0.5, 1.0, 1e-300]
    return [(a, b) for a in values for b in values] + [
        (rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(200)
    ]


@pytest.mark.parametrize("fam", list(_COUNTS))
def test_point_gradients_match_vectorized(fam) -> None:
    for a, b in _params(random.Random(1)):
        expected = np.asarray(shape_gradients(fam, np.array([a, b], dtype=np.float64)))
        made = _point_gradients(fam, a, b)
        assert made.shape == expected.shape and made.flags.c_contiguous
        assert np.array_equal(made, expected)
        assert np.array_equal(np.signbit(made), np.signbit(expected))


@pytest.mark.parametrize("fam", list(_COUNTS))
def test_jacobian_function_matches_reference_bitwise(fam) -> None:
    rng = random.Random(2)
    for _ in range(20):
        xyz = np.asarray([[rng.uniform(-3, 3) for _ in range(3)] for _ in range(_COUNTS[fam])])
        nref = np.asarray([rng.uniform(-1, 1) for _ in range(3)])
        nref /= np.linalg.norm(nref)
        fast = _jacobian_function(xyz, fam, nref)
        slow = _reference_jacobian_function(xyz, fam, nref)
        for a, b in _params(rng)[:80] * 2:  # second pass hits the memo
            expected, made = slow(a, b), fast(a, b)
            assert made == expected or (np.isnan(made) and np.isnan(expected))
            assert np.signbit(made) == np.signbit(expected)


def test_certification_unchanged_by_the_fast_path(monkeypatch) -> None:
    import anymesher.quad.high_order as high_order

    rng = random.Random(3)
    cases = []
    for _ in range(12):
        corners = np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        corners += np.asarray([[rng.uniform(-.3, .3), rng.uniform(-.3, .3), rng.uniform(-.2, .2)]
                               for _ in range(4)])
        mids = [(corners[i] + corners[(i + 1) % 4]) / 2 + rng.uniform(-.25, .25) for i in range(4)]
        cases.append(np.vstack((corners, mids)))
    fast = [certify_mapping_validity(nodes, "Q8") for nodes in cases]
    monkeypatch.setattr(high_order, "_jacobian_function", _reference_jacobian_function)
    slow = [certify_mapping_validity(nodes, "Q8") for nodes in cases]
    assert fast == slow
