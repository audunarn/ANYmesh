"""Bounded cross-cell owner queries preserve repair decisions and callbacks."""
from copy import deepcopy
import numpy as np
import pytest
from anygeometry import GeometryModel
from anygeometry.nurbs import NurbsCurve
from anymesher.mesh import Mesh
from anymesher.serialize import mesh_to_dict
from anymesher.quad.domain import ParametricQuadDomain
from anymesher.hybrid import _repair_quad_first_quality


def fixture():
    geometry = GeometryModel()
    curve = NurbsCurve(2, (0.,0.,0.,1.,1.,1.),
                       ((0.,0.,0.),(0.,1.,.8),(0.,2.,0.)))
    a, b = geometry.add_points(curve.evaluate([0.,1.]))
    face, = geometry.extrude((geometry.add_curve(a,b,curve),), (2.,0.,0.))
    domain = ParametricQuadDomain.from_geometry(geometry,face)
    mesh = Mesh()
    for j in range(3):
        for i in range(5):
            mesh.nodes[1+i+5*j] = geometry.face_point(face,i/4,j/2)
    for j in range(2):
        for i in range(4):
            n=1+i+5*j
            mesh.quads[1+i+4*j]=(n,n+1,n+6,n+5)
    mesh.elements_of_face[face]=list(mesh.quads)
    return geometry,mesh,{face:domain}


def run(batched, callback=None):
    g,m,d=fixture()
    _repair_quad_first_quality(m,g,d,cancellation_check=(lambda stage:callback(m,g,d)) if callback else None,
                              _batch_initial_normals=batched)
    return mesh_to_dict(m)


def test_lookahead_matches_old_per_cell_and_reduces_owner_calls(monkeypatch):
    g,m,d=fixture();counts=[];original=g.face_local_uv_many
    # Overridden instance protocols MUST bypass lookahead.
    def custom(face, points):
        counts.append(len(points))
        return original(face,points)
    monkeypatch.setattr(g,'face_local_uv_many',custom)
    _repair_quad_first_quality(m,g,d)
    assert counts and max(counts)<=81
    assert run(True)==run(False)


def test_callback_node_geometry_and_domain_mutation_invalidates():
    def attempt(batched):
        count=[0]
        def change(mesh,g,domains):
            count[0]+=1
            if count[0]==2:
                face=next(iter(domains))
                mesh.nodes[2]=g.face_point(face,.2501,0.)
                g.add_point(40.,40.,40.)
                face=next(iter(domains))
                domains[face]=ParametricQuadDomain.from_geometry(g,face)
        return run(batched,change),count[0]
    before,c1=attempt(False);after,c2=attempt(True)
    assert after==before
    assert c1==c2


def test_cancel_precedes_prefetch_and_does_not_publish_mesh_changes(monkeypatch):
    g,m,d=fixture();original=mesh_to_dict(m);calls=[]
    def stop(stage):
        calls.append(stage)
        raise RuntimeError('cancelled')
    monkeypatch.setattr(g,'face_local_uv_many',lambda *a:pytest.fail('prefetch before cancellation'))
    with pytest.raises(RuntimeError,match='cancelled'):
        _repair_quad_first_quality(m,g,d,cancellation_check=stop)
    assert len(calls)==1
    assert mesh_to_dict(m)==original


def test_eager_owner_failure_replays_after_original_callback(monkeypatch):
    import anygeometry.model as model
    from anygeometry import GeometryError
    g,m,d=fixture();before=mesh_to_dict(m);calls=[]
    original=model._closest_uv_many
    def fail(surface, points, **kwargs):
        if len(points)>=81:
            raise GeometryError('owner projection refused')
        return original(surface,points,**kwargs)
    monkeypatch.setattr(model,'_closest_uv_many',fail)
    with pytest.raises(GeometryError,match='owner projection refused'):
        _repair_quad_first_quality(m,g,d,cancellation_check=lambda stage:calls.append(stage))
    assert len(calls)==1
    assert mesh_to_dict(m)==before


def test_callback_protocol_and_floatmode_change_invalidates():
    def attempt(batched):
        g,m,d=fixture();calls=[0];normal=g.face_normal_many
        def change(stage):
            calls[0]+=1
            if calls[0]==2:
                g.face_normal_many=lambda face,uv:normal(face,uv)
                np.seterr(under='warn')
        with np.errstate():
            _repair_quad_first_quality(m,g,d,cancellation_check=change,
                                      _batch_initial_normals=batched)
        return mesh_to_dict(m),calls[0]
    assert attempt(True)==attempt(False)


def test_future_invalid_shape_replays_original_error_and_callback_order():
    def attempt(batched):
        g,m,d=fixture();m.nodes[15]=np.array([0.,1.]);calls=[]
        try:
            _repair_quad_first_quality(m,g,d,cancellation_check=lambda stage:calls.append(stage),
                                      _batch_initial_normals=batched)
        except (ValueError,TypeError) as error:
            return type(error),str(error),calls
        pytest.fail('invalid future node admitted')
    assert attempt(True)==attempt(False)


def test_strict_float_mode_uses_unchanged_per_cell_path():
    def attempt(batched):
        g,m,d=fixture()  # Construct the same accepted input under ordinary modes.
        with np.errstate(all='raise'):
            try:
                _repair_quad_first_quality(m,g,d,_batch_initial_normals=batched)
            except FloatingPointError as error:
                return type(error),str(error)
        return mesh_to_dict(m)
    assert attempt(True)==attempt(False)
