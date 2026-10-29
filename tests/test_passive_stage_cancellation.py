import pytest
import anygeometry as owner
from anymesher.errors import MeshError
from anymesher import hybrid, surface_mesh
from anymesher._stage_cancellation import check_stage, stage_scope, preserve_stage_cancellation

@pytest.mark.parametrize('value',[False,None])
def test_non_cancelled_stage_returns_normally(value):
    calls=[]
    assert hybrid._check_cancellation(lambda phase: calls.append(phase) or value,'stage') is None
    assert calls == ['stage']

@pytest.mark.parametrize('checker',[hybrid._check_cancellation,check_stage])
def test_passive_signal_is_honored(checker):
    signal=owner.CancellationSignal()
    checker(signal,'active')
    signal.cancel()
    with pytest.raises(MeshError,match='cancelled'): checker(signal,'cancelled')

@pytest.mark.parametrize('failure',[MeshError('caller stop'),RuntimeError('caller stop')])
def test_callback_exception_identity_survives_optional_fallback(failure):
    def callback(phase): raise failure
    @stage_scope
    def operation():
        try: check_stage(callback,'candidate')
        except Exception as error:
            preserve_stage_cancellation(error)
            pytest.fail('cancellation was treated as optional candidate refusal')
    with pytest.raises(type(failure)) as caught: operation()
    assert caught.value is failure
    # The operation-local marker is reset; normal later fallback is allowed.
    preserve_stage_cancellation(failure)

def test_surface_passive_cancellation_after_real_triangulation_prevents_recombination(monkeypatch):
    signal=owner.CancellationSignal()
    actual=surface_mesh.triangulate_polygon
    seen=[]
    def triangulate(*args,**kwargs):
        assert kwargs['cancellation_check'] is signal
        result=actual(*args,**kwargs)
        seen.append('triangulated')
        signal.cancel()
        return result
    def recombine(*args,**kwargs):
        pytest.fail('cancelled triangulation reached recombination')
    monkeypatch.setattr(surface_mesh,'triangulate_polygon',triangulate)
    monkeypatch.setattr(surface_mesh,'recombine_triangles_with_report',recombine)
    with pytest.raises(MeshError,match='cancelled'):
        surface_mesh.mesh_planar_surface(((0.,0.),(1.,0.),(.5,.8)),
            target_size=2.,backend='python',cancellation_check=signal)
    assert seen == ['triangulated']

def test_direct_hybrid_passive_cancellation_preserves_source():
    geometry=owner.GeometryModel()
    geometry.add_plate(geometry.add_points(((0,0,0),(1,0,0),(1,1,0),(0,1,0))))
    before=owner.to_dict(geometry)
    signal=owner.CancellationSignal();signal.cancel()
    with pytest.raises(MeshError,match='cancelled'):
        hybrid.generate_hybrid_mesh_result(geometry,target_size=.5,strategy='native',
            cancellation_check=signal)
    assert owner.to_dict(geometry) == before
