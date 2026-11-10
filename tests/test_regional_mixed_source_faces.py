"""An authored Plane and a separately owned curved face in one source publication."""
import json

import numpy as np
import pytest
import anygeometry as owner
from anymesher import (generate_hybrid_mesh_result, publish_model_closure_mesh_to_source,
                       qualify_model_closure_source_publication)
from anymesher.material_source_transfer import _validated_curvilinear_reference_uv
from anymesher.regional_source_faces import capture_reusable_source_faces, reuse_source_faces
from test_regional_source_generation import patch


@pytest.mark.parametrize('family,retained', [('nurbs','plane'),('extruded','curved')])
def test_independent_plane_and_curve_reuse(monkeypatch, family, retained):
    source, curved, sheet = patch(family)
    plane = source.add_plate(source.add_points(((4.,0.,0.),(5.,0.,0.),(5.,1.,0.),(4.,1.,0.))))
    plane_sheet = source.add_sheet((plane,))
    assert type(source.faces[plane].surface) is owner.Plane
    def generate(model, **options):
        closure = owner.extract_model_closure(model, tuple(model.handle('sheet', s) for s in (sheet,plane_sheet)))
        token = qualify_model_closure_source_publication(model, closure)
        mesh = generate_hybrid_mesh_result(closure.working_model, target_size=.45, strategy='auto',
            geometry_approximation_error=.0125, structural_preparation=False,
            mutation_policy='working_copy', native_backend='python', **options).mesh
        publish_model_closure_mesh_to_source(model, closure, mesh, qualification=token)
        assert set(_validated_curvilinear_reference_uv(model, mesh)) == set(mesh.shells)
        return mesh
    original = generate(source)
    candidates = json.loads(json.dumps(capture_reusable_source_faces(source, original)))
    assert set(candidates) == {str(curved), str(plane)}
    kept, edited = (plane,curved) if retained=='plane' else (curved,plane)
    reopened = owner.from_dict(owner.to_dict(source))
    from anymesher import hybrid
    generator = hybrid._try_parametric_tensor
    invoked = set()
    def check(geometry, face, *args, **kwargs):
        assert face != kept
        invoked.add(face)
        return generator(geometry, face, *args, **kwargs)
    monkeypatch.setattr(hybrid, '_try_parametric_tensor', check)
    with reuse_source_faces(reopened, {str(kept):candidates[str(kept)]}) as stats:
        mesh = generate(reopened, overrides={u.edge:4 for u in reopened.faces[edited].loop})
    assert stats['reused_faces'] == {kept} and invoked == {edited}
    assert len(mesh.elements_of_face[edited]) != len(original.elements_of_face[edited])
    for a,b in zip(original.elements_of_face[kept], mesh.elements_of_face[kept], strict=True):
        assert np.array_equal([original.nodes[n] for n in original.shells[a]], [mesh.nodes[n] for n in mesh.shells[b]])


def test_plain_selection_keeps_ordinary_pipeline_and_shared_vertex_disables_mixed_admission():
    from anymesher._material_region_binding import generation_curvilinear_faces
    from anymesher.regional_source_faces import independent_source_faces
    geometry = owner.GeometryModel()
    plain = geometry.add_plate(geometry.add_points(((0,0,0),(1,0,0),(1,1,0),(0,1,0))))
    assert generation_curvilinear_faces(geometry, (plain,)) == ()
    assert not independent_source_faces(geometry)
    model, curve, _ = patch('nurbs')
    common = model.edges[model.faces[curve].loop[0].edge].start
    neighboring = model.add_plate((common, *model.add_points(((-1.,0.,0.),(-1.,-1.,0.),(0.,-1.,0.)))))
    assert generation_curvilinear_faces(model, (curve,neighboring)) == (curve,)
    assert not independent_source_faces(model)


def test_mixed_default_uses_curved_scale_and_plane_source_proof():
    from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface
    from anygeometry.entities import OrientedEdge
    model = owner.GeometryModel()
    knots = (0.,0.,1.,1.)
    surface = NurbsSurface(1,1,knots,knots,(((0.,0.,0.),(0.,1.,0.)), ((1.,0.,0.),(1.,1.,1/1024))))
    uvs = ((0.,0.),(1.,0.),(1.,1.),(0.,1.))
    vertices = model.add_points(tuple(surface.evaluate(*p) for p in uvs))
    edges = [model.add_curve(vertices[i], vertices[(i+1)%4],
              CurveOnSurface(surface, NurbsCurve(1,knots,(uvs[i],uvs[(i+1)%4])))) for i in range(4)]
    curve = model.add_face_from_loop(tuple(OrientedEdge(e,True) for e in edges), corners=(0,1,2,3), surface=surface)
    plain = model.add_plate(model.add_points(((3.,0.,0.),(4.,0.,0.),(4.,1.,0.),(3.,1.,0.))))
    sheets = [model.add_sheet((f,)) for f in (curve, plain)]
    closure = owner.extract_model_closure(model, tuple(model.handle('sheet',s) for s in sheets))
    token = qualify_model_closure_source_publication(model, closure)
    mesh = generate_hybrid_mesh_result(closure.working_model, target_size=.5, strategy='auto',
        structural_preparation=False, mutation_policy='working_copy', native_backend='python').mesh
    publish_model_closure_mesh_to_source(model, closure, mesh, qualification=token)
    report = mesh.hybrid_diagnostics['geometry_approximation']
    assert report['mode'] == 'automatic' and list(report['face_ids']) == [curve]
    assert set(_validated_curvilinear_reference_uv(model, mesh)) == set(mesh.shells)
