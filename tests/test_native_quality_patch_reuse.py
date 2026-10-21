"""Exact local-polish topology reuse; frozen scalar arithmetic remains the oracle."""
import struct
import warnings
import importlib.util
import sys

import numpy as np
import pytest
import anymesher.surface_mesh as surface
from anymesher.errors import MeshError


def bits(value):
    if isinstance(value, (float, np.floating)):
        return struct.pack('!d', float(value))
    if isinstance(value, np.ndarray):
        return value.dtype.str, value.shape, value.tobytes()
    if isinstance(value, dict):
        return {key: bits(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(map(bits, value))
    return value


def patch():
    points = np.array([(x, y) for y in range(4) for x in range(5)], dtype=float)
    points[7] += (.39, -.28)
    triangles = []
    for y in range(3):
        for x in range(4):
            a = y*5+x
            triangles.extend(((a, a+1, a+6), (a, a+6, a+5)))
    return points, np.asarray(triangles, dtype=np.int64)


def compiled():
    if not surface._quality_patch_reuse_available():
        pytest.skip('installed compiled quality services required')


@pytest.mark.parametrize('displacement', [0., 1.e-12, -.19, .26])
@pytest.mark.parametrize('policy', ['default', 'raise', 'ignore'])
def test_all_report_bits_and_aspects_match_scalar(displacement, policy):
    compiled()
    points, triangles = patch()
    settings = surface.SurfaceMeshOptions(recombine=False)
    evidence = {}
    with np.errstate(**({} if policy == 'default' else {'all': policy})):
        for offset in (0., displacement, -displacement):
            points[8, 0] += offset
            expected = surface._triangle_quality(points, triangles, settings)
            actual = surface._triangle_quality(points, triangles, settings, _quality_patch=evidence)
            assert bits(actual) == bits(expected)
    assert 'adjacent_rows' in evidence


def test_evidence_is_patch_local_and_connectivity_bound():
    compiled()
    points, triangles = patch()
    evidence = {}
    surface._triangle_quality(points, triangles, _quality_patch=evidence)
    first = evidence['adjacent_rows']
    surface._triangle_quality(points, triangles, _quality_patch=evidence)
    assert evidence['adjacent_rows'] is first
    other = triangles.copy()
    surface._triangle_quality(points, other, _quality_patch=evidence)
    assert evidence['triangles'] is other
    assert evidence['adjacent_rows'] is not first
    before_mutation = evidence['adjacent_rows']
    other[0] = np.roll(other[0], 1)
    actual = surface._triangle_quality(points, other, _quality_patch=evidence)
    assert bits(actual) == bits(surface._triangle_quality(points, other))
    assert evidence['adjacent_rows'] is not before_mutation


@pytest.mark.parametrize('layout', ['empty', 'disconnected', 'far_unused', 'reversed'])
def test_unusual_connectivity_and_global_scale_keep_exact_report(layout):
    compiled()
    points, triangles = patch()
    if layout == 'empty':
        triangles = triangles[:0]
    elif layout == 'disconnected':
        triangles = triangles[[0, -1]]
    elif layout == 'far_unused':
        points = np.vstack((points, [[1.e7, -1.e7]]))
    else:
        triangles = triangles[:, ::-1].copy()
    evidence = {}
    for _ in range(2):
        assert bits(surface._triangle_quality(points, triangles, _quality_patch=evidence)) == bits(
            surface._triangle_quality(points, triangles))


@pytest.mark.parametrize('scale', [2.**-210, 1., 2.**210])
@pytest.mark.parametrize('policy', ['default', 'raise'])
def test_threshold_adjacent_growth_and_extreme_fallback(scale, policy):
    compiled()
    points, triangles = patch()
    points *= scale
    evidence = {}
    with np.errstate(**({} if policy == 'default' else {'all': 'raise'})):
        ordinary = surface._triangle_quality(points, triangles)
        growth = ordinary[0]['max_element_growth']
        # Inputs are constructed under default NumPy policy, not all-raise.
    thresholds = (np.nextafter(growth, 0.), growth, np.nextafter(growth, np.inf))
    for threshold in thresholds:
        settings = surface.SurfaceMeshOptions(recombine=False, max_element_growth=max(1., threshold))
        with np.errstate(**({} if policy == 'default' else {'all': 'raise'})):
            assert bits(surface._triangle_quality(points, triangles, settings, _quality_patch=evidence)) == bits(
                surface._triangle_quality(points, triangles, settings))


@pytest.mark.parametrize('value', [0., np.nan, np.inf])
@pytest.mark.parametrize('policy', ['default', 'raise'])
def test_zero_nonfinite_warning_and_refusal_parity(value, policy):
    compiled()
    points, triangles = patch()
    evidence = {}
    surface._triangle_quality(points, triangles, _quality_patch=evidence)
    points[:] = value
    def outcome(cached):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with np.errstate(**({} if policy == 'default' else {'all': 'raise'})):
                try:
                    result = surface._triangle_quality(points, triangles, _quality_patch=evidence if cached else None)
                    made = ('result', bits(result))
                except Exception as error:
                    made = ('error', type(error), str(error))
            return made, [(item.category, str(item.message)) for item in caught]
    assert outcome(True) == outcome(False)


@pytest.mark.parametrize('warm', [False, True])
def test_patched_incidence_is_not_hidden_by_evidence(monkeypatch, warm):
    compiled()
    points, triangles = patch()
    evidence = {}
    if warm:
        surface._triangle_quality(points, triangles, _quality_patch=evidence)
    calls = []
    def malformed(rows):
        calls.append(rows)
        return np.array([[0, 1, 0, 999999]], dtype=np.int64)
    monkeypatch.setattr(surface._native_cpp, 'triangle_edge_incidence', malformed)
    with pytest.raises(MeshError, match='malformed output'):
        surface._triangle_quality(points, triangles, _quality_patch=evidence)
    assert len(calls) == 1


def test_patched_quality_exception_identity_after_warm_evidence(monkeypatch):
    compiled()
    points, triangles = patch()
    evidence = {}
    surface._triangle_quality(points, triangles, _quality_patch=evidence)
    marker = MeshError('quality cancellation marker')
    def cancel(*args):
        raise marker
    monkeypatch.setattr(surface._native_cpp, 'native_element_quality', cancel)
    with pytest.raises(MeshError) as caught:
        surface._triangle_quality(points, triangles, _quality_patch=evidence)
    assert caught.value is marker


@pytest.mark.parametrize('growth', [1.1, 1.5, 3.])
def test_polish_exact_points_reports_and_work_limit(monkeypatch, growth):
    compiled()
    points, triangles = patch()
    settings = surface.SurfaceMeshOptions(recombine=False, max_element_growth=growth)
    candidate = surface._make_candidate(points, triangles, settings=settings)
    saved = points.copy(), triangles.copy()
    edges = np.array([(0, 4), (4, 19), (19, 15), (15, 0)], dtype=np.int64)
    stats_a, stats_b = {}, {}
    actual = surface._polish_quality_candidate(candidate, edges, np.empty((0, 2)), settings, stats_a)
    monkeypatch.setattr(surface, '_quality_patch_reuse_available', lambda: False)
    expected = surface._polish_quality_candidate(candidate, edges, np.empty((0, 2)), settings, stats_b)
    for name in ('points', 'triangles', 'report', 'score', 'aspect_ratios',
                 'moved_nodes', 'added_points', 'rounds', 'flips'):
        assert bits(getattr(actual, name)) == bits(getattr(expected, name))
    assert stats_a == stats_b
    assert np.array_equal(points, saved[0]) and np.array_equal(triangles, saved[1])


def test_scalar_private_quality_override_keeps_original_signature(monkeypatch):
    points, triangles = patch()
    settings = surface.SurfaceMeshOptions(recombine=False)
    candidate = surface._make_candidate(points, triangles, settings=settings)
    original = surface._triangle_quality
    calls = []
    def overridden(points, triangles, settings=None):
        calls.append(1)
        return original(points, triangles, settings)
    monkeypatch.setattr(surface, '_triangle_quality', overridden)
    surface._polish_quality_candidate(candidate, np.empty((0, 2), dtype=np.int64),
                                      np.empty((0, 2)), settings, {})
    assert calls


@pytest.mark.parametrize('binding', ['raising_geterr', 'returning_geterr', 'maximum', 'minimum'])
def test_replaced_growth_bindings_keep_scalar_calls_and_results(monkeypatch, binding):
    compiled()
    points, triangles = patch()
    evidence = {}
    surface._triangle_quality(points, triangles, _quality_patch=evidence)
    calls = []
    marker = RuntimeError('replaced geterr must not be called by growth')
    if binding.endswith('geterr'):
        original = np.geterr
        def replacement():
            calls.append('geterr')
            if binding == 'raising_geterr':
                raise marker
            return original()
        monkeypatch.setattr(np, 'geterr', replacement)
    else:
        original = getattr(np, binding)
        def replacement(*args, **kwargs):
            calls.append(binding)
            return original(*args, **kwargs)
        # np.max/min use the public ufunc's reduce protocol in this runtime.
        # Preserve that original interface, while deliberately omitting .at:
        # the scalar reference must succeed before testing added requirements.
        replacement.reduce = original.reduce
        monkeypatch.setattr(np, binding, replacement)
    expected = surface._triangle_quality(points, triangles)
    scalar_calls = list(calls)
    calls.clear()
    actual = surface._triangle_quality(points, triangles, _quality_patch=evidence)
    assert bits(actual) == bits(expected)
    assert calls == scalar_calls


def test_preimport_replaced_geterr_keeps_original_growth_path(monkeypatch):
    compiled()
    original = np.geterr
    calls = []
    def replacement():
        calls.append('geterr')
        return original()
    monkeypatch.setattr(np, 'geterr', replacement)
    name = 'anymesher._test_preimport_quality_patch'
    spec = importlib.util.spec_from_file_location(name, surface.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    points, triangles = patch()
    evidence = {}
    expected = module._triangle_quality(points, triangles)
    scalar_calls = list(calls)
    calls.clear()
    actual = module._triangle_quality(points, triangles, _quality_patch=evidence)
    assert bits(actual) == bits(expected)
    assert calls == scalar_calls
