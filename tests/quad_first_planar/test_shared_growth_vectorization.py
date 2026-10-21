"""Public-facade growth keeps the original sorted scalar audit contract."""
import numpy as np
import pytest
import struct
import warnings

from anymesher.hybrid import _element_growth
from anymesher.mesh import Mesh
from anymesher.quad import transition_smoothing as transition


def scalar_growth(mesh, *, limit):
    characteristic = {}
    incidence = {}
    for element_id in sorted(mesh.shells):
        corners = mesh.corners_of(element_id)
        lengths = []
        for first, second in zip(corners, corners[1:] + corners[:1]):
            edge = tuple(sorted((int(first), int(second))))
            lengths.append(float(np.linalg.norm(mesh.nodes[second] - mesh.nodes[first])))
            incidence.setdefault(edge, []).append(int(element_id))
        characteristic[int(element_id)] = float(np.mean(lengths))
    maximum = 1.0
    violations = []
    for attached in incidence.values():
        if len(attached) != 2:
            continue
        first, second = attached
        small = min(characteristic[first], characteristic[second])
        ratio = (float('inf') if small <= 0.0
                 else max(characteristic[first], characteristic[second]) / small)
        maximum = max(maximum, ratio)
        if ratio > limit + 1.0e-14:
            violations.append((min(first, second), max(first, second), ratio))
    return maximum, tuple(sorted(violations))


def strip(count=270, dtype=np.float64, quadratic=False):
    nodes = {2*i+j:np.array((float(i),float(j),0.),dtype=dtype)
             for i in range(count+1) for j in (0,1)}
    quads = {1000-i:(2*i,2*i+2,2*i+3,2*i+1) for i in range(count)}
    tris = {2000:(0,1,2)}  # Includes a three-owner edge and mixed corner count.
    if quadratic:
        quads = {cid:body+body for cid,body in quads.items()}
        tris = {cid:body+body for cid,body in tris.items()}
    return Mesh(nodes=nodes,quads=quads,tris=tris,order='quadratic' if quadratic else 'linear')


def assert_exact(actual, expected):
    assert actual == expected
    values = lambda result: np.asarray([result[0],*(row[2] for row in result[1])]).view(np.uint64)
    np.testing.assert_array_equal(values(actual),values(expected))


@pytest.mark.parametrize('used',[False,True])
@pytest.mark.parametrize('policy',['default','warn','raise'])
def test_shared_growth_signalling_nan_warning_error_and_bits(policy,used):
    mesh = strip(count=3)
    mesh.nodes[9999] = np.array([2.,2.,0.])
    mesh.nodes[4 if used else 9999].view(np.uint64)[0] = 0x7ff0000000000001
    before = {node:point.tobytes() for node,point in mesh.nodes.items()}
    errors = ({'divide':'warn','over':'warn','under':'ignore','invalid':'warn'}
              if policy == 'default' else {'all':policy})
    def outcome(function):
        with np.errstate(**errors),warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                maximum, pairs = function(mesh,limit=1.)
                made = ('result',struct.pack('>d',maximum),
                        tuple((a,b,struct.pack('>d',value)) for a,b,value in pairs))
            except Exception as error:
                made = ('error',type(error),str(error))
        return made,[(item.category,str(item.message)) for item in caught]
    assert outcome(_element_growth) == outcome(scalar_growth)
    assert {node:point.tobytes() for node,point in mesh.nodes.items()} == before


@pytest.mark.parametrize('quadratic',[False,True])
@pytest.mark.parametrize('limit',[1.,np.nextafter(1.,0.),1.515])
def test_shared_growth_exact_native_and_bounded_chunks(monkeypatch,quadratic,limit):
    mesh = strip(quadratic=quadratic)
    before = {node:point.copy() for node,point in mesh.nodes.items()}
    expected = scalar_growth(mesh,limit=limit)
    calls = []
    original = transition._characteristic_lengths
    def observe(xyz,index,cells):
        calls.append((len(xyz),len(cells)))
        return original(xyz,index,cells)
    monkeypatch.setattr(transition,'_characteristic_lengths',observe)
    assert_exact(_element_growth(mesh,limit=limit),expected)
    if transition._native_norm_available() and transition._NATIVE_MEAN:
        from anymesher.mesh import _NATIVE_MESH_NAMESPACE
        anchored = dict(_NATIVE_MESH_NAMESPACE)
        delta = {'added':sorted(vars(Mesh).keys()-anchored.keys()),
                 'changed':[name for name,value in anchored.items() if vars(Mesh).get(name) is not value]}
        assert len(calls) == 2, delta
        assert all(points <= 1024 and cells <= 256 for points,cells in calls)
    assert all(np.array_equal(point,before[node]) for node,point in mesh.nodes.items())


def test_standard_deepcopy_adds_only_empty_slot_cache_and_keeps_activation(monkeypatch):
    from copy import deepcopy
    from anymesher.mesh import _NATIVE_MESH_NAMESPACE
    monkeypatch.delattr(Mesh,'__slotnames__',raising=False)
    before = dict(vars(Mesh))
    mesh = deepcopy(strip(count=3))
    after = dict(vars(Mesh))
    assert after.keys()-before.keys() == {'__slotnames__'}
    assert type(after['__slotnames__']) is list and after['__slotnames__'] == []
    assert all(after[name] is value for name,value in before.items())
    assert all(after[name] is value for name,value in _NATIVE_MESH_NAMESPACE)
    calls = []
    original = transition._characteristic_lengths
    def observe(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(transition,'_characteristic_lengths',observe)
    assert_exact(_element_growth(mesh,limit=1.),scalar_growth(mesh,limit=1.))
    if transition._native_norm_available() and transition._NATIVE_MEAN:
        assert calls == [True]


@pytest.mark.parametrize('name,value',[('__slotnames__',['custom']),('__slotnames__',()),
                                      ('custom_growth_hook',object())])
def test_other_namespace_additions_still_decline_batching(monkeypatch,name,value):
    monkeypatch.setattr(Mesh,name,value,raising=False)
    monkeypatch.setattr(transition,'_characteristic_lengths',
                        lambda *args:pytest.fail('nonstandard namespace was batched'))
    mesh = strip(count=3)
    assert_exact(_element_growth(mesh,limit=1.),scalar_growth(mesh,limit=1.))


@pytest.mark.parametrize('kind',['float32','subclass','strided','instance_method','mesh_subclass'])
def test_unsupported_facades_preserve_scalar_semantics(monkeypatch,kind):
    mesh = strip(count=3,dtype=np.float32 if kind=='float32' else np.float64)
    if kind == 'subclass':
        class Coordinate(np.ndarray):
            pass
        mesh.nodes = {node:point.view(Coordinate) for node,point in mesh.nodes.items()}
    elif kind == 'strided':
        mesh.nodes = {node:np.repeat(point,2)[::2] for node,point in mesh.nodes.items()}
    elif kind == 'instance_method':
        original = mesh.corners_of
        mesh.corners_of = lambda cid: original(cid)
    elif kind == 'mesh_subclass':
        class CustomMesh(Mesh):
            pass
        mesh = CustomMesh(nodes=mesh.nodes,quads=mesh.quads,tris=mesh.tris)
    monkeypatch.setattr(transition,'_characteristic_lengths',
                        lambda *args:pytest.fail('unsupported facade was batched'))
    assert_exact(_element_growth(mesh,limit=1.),scalar_growth(mesh,limit=1.))


def test_changed_shells_descriptor_observes_original_call_order(monkeypatch):
    original = Mesh.shells.fget
    calls = []
    def shells(mesh):
        calls.append(len(mesh.quads))
        return original(mesh)
    monkeypatch.setattr(Mesh,'shells',property(shells))
    monkeypatch.setattr(transition,'_characteristic_lengths',
                        lambda *args:pytest.fail('custom descriptor was batched'))
    mesh = strip(count=3)
    expected = scalar_growth(mesh,limit=1.)
    baseline_calls = list(calls)
    calls.clear()
    assert_exact(_element_growth(mesh,limit=1.),expected)
    assert calls == baseline_calls


def test_missing_later_node_keeps_earlier_callback_failure(monkeypatch):
    mesh = strip(count=3)
    del mesh.nodes[0]
    marker = RuntimeError('earlier norm')
    def fail(vector):
        raise marker
    monkeypatch.setattr(np.linalg,'norm',fail)
    for function in (scalar_growth,_element_growth):
        with pytest.raises(RuntimeError) as caught:
            function(mesh,limit=1.)
        assert caught.value is marker


@pytest.mark.parametrize('kind',['zero','overlap','empty','extreme'])
def test_zero_overlap_empty_and_extreme_semantics(kind):
    mesh = strip(count=3)
    if kind == 'zero':
        mesh.nodes = {node:np.zeros(3) for node in mesh.nodes}
    elif kind == 'overlap':
        mesh.tris[999] = (0,1,2)  # shells merges tris, corners_of prefers quads.
    elif kind == 'empty':
        mesh = Mesh()
    elif kind == 'extreme':
        mesh.nodes = {node:point*2.**510 for node,point in mesh.nodes.items()}
    with np.errstate(all='ignore'):
        assert_exact(_element_growth(mesh,limit=1.),scalar_growth(mesh,limit=1.))
