"""Bitwise scalar oracles for bounded transition characteristic batching."""
import importlib.util
import warnings

import numpy as np
import pytest

from anymesher.quad import transition_smoothing as smoothing


def scalar_characteristics(xyz, index, cells):
    return {cid: float(np.mean([np.linalg.norm(xyz[index[b]]-xyz[index[a]])
            for a, b in zip(body, body[1:]+body[:1])])) for cid, body in cells.items()}


def assert_bits(actual, expected):
    np.testing.assert_array_equal(np.asarray(actual, dtype=np.float64).view(np.uint64),
                                  np.asarray(expected, dtype=np.float64).view(np.uint64))


def signalling_nan_outcome(function, policy):
    errors = ({'divide':'warn','over':'warn','under':'ignore','invalid':'warn'}
              if policy == 'default' else {'all':policy})
    with np.errstate(**errors), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            result = function()
            if isinstance(result, dict):
                made = ('result',tuple(result),np.asarray(list(result.values()),dtype=np.float64).tobytes())
            else:
                made = ('result',np.asarray(result,dtype=np.float64).tobytes())
        except Exception as error:
            made = ('error',type(error),str(error))
    return made,[(item.category,str(item.message)) for item in caught]


@pytest.mark.parametrize('policy',['default','warn','raise'])
def test_direct_edge_norm_signalling_nan_matches_scalar(policy):
    values = np.array([[1.,0.,0.],[0.,0.,0.],[1.,2.,0.]])
    values.view(np.uint64)[1,0] = 0x7ff0000000000001
    before = values.tobytes()
    expected = signalling_nan_outcome(lambda:[np.linalg.norm(row) for row in values],policy)
    actual = signalling_nan_outcome(lambda:smoothing._scalar_compatible_edge_norms(values),policy)
    assert actual == expected
    assert values.tobytes() == before


@pytest.mark.parametrize('used',[False,True])
@pytest.mark.parametrize('policy',['default','warn','raise'])
def test_characteristic_signalling_nan_preserves_warning_and_refusal_order(policy,used):
    xyz = np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.],[2.,2.,0.]])
    xyz.view(np.uint64)[0 if used else 4,0] = 0x7ff0000000000001
    index = dict(enumerate(range(len(xyz))))
    cells = {9:(0,1,2,3)}
    before = xyz.tobytes()
    expected = signalling_nan_outcome(lambda:scalar_characteristics(xyz,index,cells),policy)
    actual = signalling_nan_outcome(lambda:smoothing._characteristic_lengths(xyz,index,cells),policy)
    assert actual == expected
    assert xyz.tobytes() == before


@pytest.mark.parametrize('policy', ['ignore', 'warn', 'raise'])
def test_edge_norm_bits_random_axis_extreme_signed_zero_and_policy(policy):
    rng = np.random.default_rng(1831)
    values = rng.normal(size=(4097, 3))
    values[::2, 1:] = 0.
    limits = [0., -0., np.nextafter(0., 1.), np.finfo(float).tiny,
              2.**-500, 2.**500, np.finfo(float).max, np.inf, np.nan]
    values = np.vstack((values, [[x, 0., -0.] for x in limits],
                        [[x, x, 0.] for x in limits]))
    original = values.copy()
    outcomes = []
    for function in (lambda rows: [np.linalg.norm(row) for row in rows],
                     smoothing._scalar_compatible_edge_norms):
        with np.errstate(all=policy), warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            try:
                outcomes.append(('value', function(values), [str(w.message) for w in captured]))
            except FloatingPointError as exc:
                outcomes.append(('error', str(exc), [str(w.message) for w in captured]))
    assert outcomes[0][0] == outcomes[1][0]
    assert outcomes[0][2] == outcomes[1][2]
    if outcomes[0][0] == 'value':
        assert_bits(outcomes[1][1], outcomes[0][1])
    else:
        assert outcomes[0][1] == outcomes[1][1]
    assert_bits(values, original)


@pytest.mark.parametrize('layout', ['contiguous', 'strided', 'readonly', 'subclass'])
def test_edge_norm_layouts_preserve_bits(layout):
    values = np.array([[1., 0., 0.], [1., 2., 3.], [-0., 0., -0.]])
    if layout == 'strided':
        values = np.repeat(values, 2, axis=1)[:, ::2]
    elif layout == 'readonly':
        values.flags.writeable = False
    elif layout == 'subclass':
        class Array(np.ndarray):
            pass
        values = values.view(Array)
    assert_bits(smoothing._scalar_compatible_edge_norms(values),
                [np.linalg.norm(row) for row in values])


def test_native_axis_kernel_skips_only_admissible_rows(monkeypatch):
    # Exercise the kernel separately from its real binding admission, then
    # verify the installed runtime admission before replacing any binding.
    if np.__version__.startswith('2.4.'):
        assert smoothing._native_norm_available()
    norm = np.linalg.norm
    seen = []
    monkeypatch.setattr(smoothing, '_native_norm_available', lambda: True)
    def observe(row):
        seen.append(tuple(row))
        return norm(row)
    monkeypatch.setattr(np.linalg, 'norm', observe)
    rows = np.array([[1.,0.,0.], [0.,-2.,0.], [0.,0.,0.],
                     [1.,2.,0.], [2.**-501,0.,0.]])
    assert_bits(smoothing._scalar_compatible_edge_norms(rows), [norm(row) for row in rows])
    assert seen == [tuple(rows[3]), tuple(rows[4])]


def test_patched_norm_preserves_calls_inputs_and_exception_identity(monkeypatch):
    xyz = np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]])
    cells = {0:(0,1,2,3)}
    marker = RuntimeError('custom norm')
    seen = []
    def custom(row):
        seen.append((row.copy(), row.base))
        raise marker
    monkeypatch.setattr(np.linalg, 'norm', custom)
    with pytest.raises(RuntimeError) as caught:
        smoothing._characteristic_lengths(xyz,dict(enumerate(range(4))),cells)
    assert caught.value is marker
    assert len(seen) == 1 and seen[0][1] is None
    assert_bits(seen[0][0], [1.,0.,0.])


@pytest.mark.parametrize('binding', ['sqrt', 'asarray'])
def test_replaced_norm_core_binding_remains_visible(monkeypatch, binding):
    if not smoothing._NATIVE_NORM:
        pytest.skip('runtime deliberately uses scalar fallback')
    marker = RuntimeError(binding)
    def fail(*args, **kwargs):
        raise marker
    monkeypatch.setitem(smoothing._NORM_GLOBALS, binding, fail)
    with pytest.raises(RuntimeError) as caught:
        smoothing._scalar_compatible_edge_norms(np.array([[1.,0.,0.]]))
    assert caught.value is marker


def test_preimport_custom_norm_cannot_become_native_anchor(monkeypatch):
    norm = np.linalg.norm
    seen = []
    def observe(row):
        seen.append(tuple(row))
        return norm(row)
    monkeypatch.setattr(np.linalg, 'norm', observe)
    spec = importlib.util.spec_from_file_location('anymesher.quad._vector_test_copy', smoothing.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not module._native_norm_available()
    assert_bits(module._scalar_compatible_edge_norms(np.array([[1.,0.,0.]])), [1.])
    assert seen == [(1.,0.,0.)]


@pytest.mark.parametrize('scale', [2.**-510, 1., 2.**510])
def test_characteristic_chunking_mixed_cells_and_scale_bits(scale):
    xyz, cells = [], {}
    for cid in range(519):
        x = float(2*cid)
        start = len(xyz)
        xyz.extend(((x,0.,0.),(x+1.,0.,0.),(x+1.,1.,0.),(x,1.,0.)))
        cells[cid] = tuple(range(start, start+(3 if cid%3 == 0 else 4)))
    xyz = np.asarray(xyz)*scale
    index = dict(enumerate(range(len(xyz))))
    with np.errstate(all='ignore'):
        expected = scalar_characteristics(xyz,index,cells)
        actual = smoothing._characteristic_lengths(xyz,index,cells)
    assert list(actual) == list(expected)
    assert_bits(list(actual.values()),list(expected.values()))


@pytest.mark.parametrize('shear', [0., 0.2, np.sqrt(3.), np.nextafter(np.sqrt(3.), np.inf)])
def test_complete_common_evidence_matches_scalar_oracle(monkeypatch, shear):
    positions = {0:(0.,0.,0.),1:(1.,0.,0.),2:(2.,0.,0.),
                 3:(shear,1.,0.),4:(1.+shear,1.,0.),5:(2.+shear,1.,0.)}
    cells = {17:(0,1,4,3),8:(1,2,5,4)}
    actual = smoothing._common_evidence(positions,cells,((8,17),),None)
    monkeypatch.setattr(smoothing,'_characteristic_lengths',scalar_characteristics)
    expected = smoothing._common_evidence(positions,cells,((8,17),),None)
    assert list(actual) == list(expected)
    for key in expected:
        assert_bits(actual[key],expected[key])
