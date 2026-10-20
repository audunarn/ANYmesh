"""Single-point Jacobian fast path must equal the vectorized original exactly."""
from __future__ import annotations

import random
from math import copysign

import numpy as np
import pytest

from anymesher.quad.high_order import (
    ElementFamily,
    HighOrderCertificationCancelled,
    _certification_gradients,
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


@pytest.mark.parametrize("fam", list(_COUNTS))
def test_shared_reference_gradients_preserve_signed_zero_and_are_readonly(fam) -> None:
    for a in (-0.0, 0.0, -1.0, 1.0):
        for b in (-0.0, 0.0, -1.0, 1.0):
            cached = _certification_gradients(fam, a, b, copysign(1., a), copysign(1., b))
            expected = shape_gradients(fam, [a, b])
            np.testing.assert_array_equal(cached, expected)
            np.testing.assert_array_equal(np.signbit(cached), np.signbit(expected))
            assert not cached.flags.writeable
    assert _certification_gradients.cache_info().maxsize == 4096


@pytest.mark.parametrize("fam", list(_COUNTS))
def test_repeated_certification_observes_geometry_normal_and_budget(monkeypatch, fam) -> None:
    """Reference caches cannot retain element-dependent results across calls."""
    import anymesher.quad.high_order as high_order

    quad = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
    tri = quad[[0, 1, 3]]
    corners = quad if fam in (ElementFamily.Q4, ElementFamily.Q8) else tri
    nodes = corners.copy()
    if fam in (ElementFamily.Q8, ElementFamily.T6):
        nodes = np.vstack((nodes, (corners + np.roll(corners, -1, axis=0)) / 2))
    rng = np.random.default_rng(14)
    cases = [(nodes.copy(), [0., 0., 1.], 6, 4096),
             (nodes.copy(), [0., 0., -1.], 6, 4096)]
    for _ in range(8):
        nodes[:] += rng.uniform(-.3, .3, nodes.shape)
        cases.extend((nodes.copy(), [0., 0., 1.], depth, budget)
                     for depth, budget in ((0, 0), (2, 16)))
    fast = [certify_mapping_validity(xyz, fam, reference_normal=normal,
                                    max_depth=depth, max_subdivisions=budget)
            for xyz, normal, depth, budget in cases]
    monkeypatch.setattr(high_order, "_jacobian_function", _reference_jacobian_function)
    slow = [certify_mapping_validity(xyz, fam, reference_normal=normal,
                                    max_depth=depth, max_subdivisions=budget)
            for xyz, normal, depth, budget in cases]
    assert fast == slow


@pytest.mark.parametrize("fam", list(_COUNTS))
def test_warm_reference_cache_does_not_bypass_patch_cancellation(fam) -> None:
    quad = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
    nodes = quad if fam in (ElementFamily.Q4, ElementFamily.Q8) else quad[[0, 1, 3]]
    if fam in (ElementFamily.Q8, ElementFamily.T6):
        nodes = np.vstack((nodes, (nodes + np.roll(nodes, -1, axis=0)) / 2))
    before = nodes.copy()
    certify_mapping_validity(nodes, fam)
    calls = []

    def cancel_at_patch(stage):
        calls.append(stage)
        return len(calls) == 2

    with pytest.raises(HighOrderCertificationCancelled):
        certify_mapping_validity(nodes, fam, cancellation_check=cancel_at_patch)
    assert calls[0] == "high-order validity start"
    assert calls[1] == (
        "high-order validity square patch" if fam in (ElementFamily.Q4, ElementFamily.Q8)
        else "high-order validity triangle patch"
    )
    np.testing.assert_array_equal(nodes, before)


def _mapping_nodes(fam):
    quad = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., .1], [0., 1., 0.]])
    nodes = quad if fam in (ElementFamily.Q4, ElementFamily.Q8) else quad[[0, 1, 3]]
    if fam in (ElementFamily.Q8, ElementFamily.T6):
        nodes = np.vstack((nodes, (nodes + np.roll(nodes, -1, axis=0)) / 2))
    return nodes


def _assert_mapping_bits(first, second):
    for field in first.__dataclass_fields__:
        a, b = getattr(first, field), getattr(second, field)
        assert a.shape == b.shape and a.dtype == b.dtype
        assert a.tobytes() == b.tobytes(), field


@pytest.mark.parametrize("fam", list(_COUNTS))
def test_mapping_basis_cache_matches_uncached_layouts_and_mutation(monkeypatch, fam):
    import anymesher.quad.high_order as high_order

    nodes = _mapping_nodes(fam)
    p = np.array([[-0., 0.], [.2, .4], [1., -1.], [-.5, .5]])
    readonly = p.copy()
    readonly.flags.writeable = False
    for params in (p, p[::-1], p[0], p.tolist(), readonly, np.empty((0, 2))):
        for xyz in (nodes, nodes[:, :2]):
            expected = high_order.evaluate_mapping(xyz, fam, params)
            with monkeypatch.context() as patch:
                patch.setattr(high_order, "_mapping_reference_basis",
                              high_order._mapping_reference_basis.__wrapped__)
                actual = high_order.evaluate_mapping(xyz, fam, params)
            _assert_mapping_bits(expected, actual)

    first = high_order.evaluate_mapping(nodes, fam, p)
    for field in first.__dataclass_fields__:
        getattr(first, field)[:] = 7.
    unmodified = high_order.evaluate_mapping(nodes, fam, p)
    nodes[1, 0] += .25
    p[1, 0] = -.25
    changed = high_order.evaluate_mapping(nodes, fam, p)
    with monkeypatch.context() as patch:
        patch.setattr(high_order, "_mapping_reference_basis",
                      high_order._mapping_reference_basis.__wrapped__)
        oracle = high_order.evaluate_mapping(nodes, fam, p)
    _assert_mapping_bits(changed, oracle)
    assert changed.points.tobytes() != unmodified.points.tobytes()


def test_mapping_basis_cache_is_bounded_readonly_and_signed_zero_aware():
    import anymesher.quad.high_order as high_order

    cache = high_order._mapping_reference_basis
    cache.cache_clear()
    fam = ElementFamily.Q8
    nodes = _mapping_nodes(fam)
    for zero in (-0., 0.):
        params = np.array([[zero, 0.]])
        high_order.evaluate_mapping(nodes, fam, params)
        values, gradients = cache(fam, params.tobytes())
        assert not values.flags.writeable and not gradients.flags.writeable
        assert values.tobytes() == high_order.shape_values(fam, params).tobytes()
        assert gradients.tobytes() == high_order.shape_gradients(fam, params).tobytes()
    assert cache.cache_info().currsize == 2
    for number in range(40):
        high_order.evaluate_mapping(nodes, fam, [[number / 40., .25]])
    assert cache.cache_info().currsize == cache.cache_info().maxsize == 32
    previous = cache.cache_info()
    for params in (np.zeros((257, 2)), [[1.01, 0.]]):
        high_order.evaluate_mapping(nodes, fam, params)
    assert cache.cache_info() == previous


def test_mapping_basis_warm_cache_keeps_underflow_errors_and_warnings():
    import anymesher.quad.high_order as high_order

    nodes = _mapping_nodes(ElementFamily.Q8)
    params = np.array([[1e-200, 1e-200]])
    with np.errstate(under="ignore"):
        high_order.evaluate_mapping(nodes, "Q8", params)
    cache_before = high_order._mapping_reference_basis.cache_info()
    for _ in range(2):
        with np.errstate(under="raise"), pytest.raises(FloatingPointError, match="underflow"):
            high_order.evaluate_mapping(nodes, "Q8", params)
        with np.errstate(under="warn"), pytest.warns(RuntimeWarning, match="underflow"):
            high_order.evaluate_mapping(nodes, "Q8", params)
    assert high_order._mapping_reference_basis.cache_info() == cache_before


@pytest.mark.parametrize("params", ([0.], [[0., 0., 0.]], [[np.nan, 0.]], [[np.inf, 0.]]))
def test_mapping_basis_does_not_bypass_parameter_refusals(params):
    import anymesher.quad.high_order as high_order

    nodes = _mapping_nodes(ElementFamily.Q8)
    high_order.evaluate_mapping(nodes, "Q8", [[0., 0.]])
    with pytest.raises(ValueError, match="reference parameter"):
        high_order.evaluate_mapping(nodes, "Q8", params)


@pytest.mark.parametrize("normal", ([0., 0., 0.], [0., 1.], [0., 0., np.nan]))
def test_mapping_basis_does_not_bypass_reference_normal_refusals(normal):
    import anymesher.quad.high_order as high_order

    nodes = _mapping_nodes(ElementFamily.Q8)
    high_order.evaluate_mapping(nodes, "Q8", [[0., 0.]])
    with pytest.raises(ValueError, match="reference_normal"):
        high_order.evaluate_mapping(nodes, "Q8", [[0., 0.]], reference_normal=normal)
