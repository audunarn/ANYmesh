"""Routing hints defer raw sweep scale/material authority to ANYgeometry."""
from dataclasses import replace
from fractions import Fraction as Q
import anygeometry as owner
from anygeometry.nurbs import NurbsCurve
from anymesher import GeometryApproximationPolicy,resolve_geometry_approximation,validate_geometry_approximation_binding
from anymesher._material_region_binding import face_requires_curvilinear


def raw_sweep():
    model=owner.GeometryModel()
    curve=NurbsCurve(2,(0.,0.,0.,1.,1.,1.),((0.,0.,0.),(0.,.5,1.),(0.,1.,0.)))
    points=model.add_points(tuple(tuple(float(x) for x in curve.exact_point(t)) for t in (Q(0),Q(1,2),Q(1))))
    edges=tuple(model.add_curve(a,b,replace(curve,domain=domain))
        for a,b,domain in zip(points,points[1:],((0.,.5),(.5,1.))))
    faces=tuple(model.extrude(edges,(5.,0.,0.)));model.add_sheet(faces)
    return model,faces


def test_raw_spline_sweep_routes_to_fresh_owner_scale_without_installation():
    model,faces=raw_sweep();before=owner.to_dict(model)
    assert all(isinstance(model.faces[f].surface,owner.CoonsSurface) for f in faces)
    assert all(face_requires_curvilinear(model,f) for f in faces)
    result=resolve_geometry_approximation(model,faces,policy=GeometryApproximationPolicy(adaptation='scale'))
    assert result.mode=='automatic' and result.effective_error>0
    assert result.face_ids==faces
    validate_geometry_approximation_binding(model,result)
    assert owner.to_dict(model)==before


def test_unrelated_plain_face_keeps_not_applicable_route():
    model,_=raw_sweep()
    plain=model.add_plate(model.add_points(((10,0,0),(20,0,0),(20,10,0),(10,10,0))))
    before=owner.to_dict(model)
    assert not face_requires_curvilinear(model,plain)
    result=resolve_geometry_approximation(model,(plain,))
    assert result.mode=='not_applicable' and result.effective_error is None
    assert owner.to_dict(model)==before
