"""Source cells produced by ordinary signed model-closure mesh generation."""
import pytest

import anygeometry as owner
from anygeometry.entities import OrientedEdge
from anygeometry.extrusions import BezierDirectrix
from anygeometry.surfaces import CoonsSurface, ExtrudedSurface, RuledSurface
from anymesher import (generate_hybrid_mesh_result, publish_model_closure_mesh_to_source,
                       qualify_model_closure_source_publication)
from anymesher.material_source_transfer import _validated_curvilinear_reference_uv


def patch(family):
    model = owner.GeometryModel()
    if family == 'nurbs':
        from anygeometry.nurbs import NurbsCurve, NurbsSurface, CurveOnSurface
        knots = (0.,0.,1.,1.)
        support = NurbsSurface(1,1,knots,knots,
            (((0.,0.,0.),(0.,1.,0.)),((1.,0.,0.),(1.,1.,.125))))
        uv = ((0.,0.),(1.,0.),(1.,1.),(0.,1.))
        curves = [CurveOnSurface(support, NurbsCurve(1,knots,(u,uv[(i+1)%4]))) for i,u in enumerate(uv)]
        vertices = [model.add_point(*curve.exact_point(0)) for curve in curves]
        edges = [model.add_curve(vertices[i], vertices[(i+1)%4], curve) for i,curve in enumerate(curves)]
    elif family in ('ruled', 'coons'):
        a, b, c, d = ((0.,0.,0.), (1.,0.,0.), (1.,1.,.125), (0.,1.,0.))
        vertices = model.add_points((a,b,c,d))
        edges = [model.add_line(vertices[i], vertices[(i+1)%4]) for i in range(4)]
        support = (RuledSurface((a,b),(d,c)) if family == 'ruled' else
                   CoonsSurface((a,b),(b,c),(d,c),(a,d)))
    else:
        profile = ((0.,0.,0.), (.5,.0625,0.), (1.,0.,0.))
        shift = (0.,0.,1.)
        lo = model.add_points(profile)
        hi = model.add_points([tuple(a+b for a,b in zip(p,shift)) for p in profile])
        edges = [model.add_spline(lo[0],lo[1:-1],lo[-1]), model.add_line(lo[-1],hi[-1]),
                 model.add_spline(hi[-1],hi[-2:0:-1],hi[0]), model.add_line(hi[0],lo[0])]
        support = ExtrudedSurface(BezierDirectrix(profile),shift) if family == 'extruded' else None
    face = model.add_face_from_loop(tuple(OrientedEdge(e, True) for e in edges), corners=(0,1,2,3),
                                    surface=support)
    sheet = model.add_sheet((face,))
    return model, face, sheet


@pytest.mark.parametrize('family', ('extruded','implicit','ruled','coons','nurbs'))
def test_original_source_cells_for_mapped_closure(family):
    source, face, sheet = patch(family)
    original = owner.to_dict(source)
    closure = owner.extract_model_closure(source, (source.handle('sheet',sheet),))
    token = qualify_model_closure_source_publication(source, closure)
    work = closure.working_model
    result = generate_hybrid_mesh_result(work, target_size=.45, strategy='auto' if family=='nurbs' else 'mapped',
        geometry_approximation_error=.0125, structural_preparation=False,
        mutation_policy='working_copy', native_backend='python')
    mesh = result.mesh
    assert mesh.quads and not mesh.tris
    publish_model_closure_mesh_to_source(source, closure, mesh, qualification=token)
    assert mesh.geometry_model_id == source.model_id
    rows = _validated_curvilinear_reference_uv(source, mesh)
    assert rows
    assert owner.to_dict(source) == original


def test_source_tensor_default_local_approximation():
    source, face, _ = patch('extruded')
    result = generate_hybrid_mesh_result(source, target_size=.25, strategy='auto',
        structural_preparation=False, mutation_policy='working_copy', native_backend='python')
    assert _validated_curvilinear_reference_uv(source, result.mesh)
    assert result.mesh.hybrid_diagnostics['geometry_approximation']['mode'] == 'automatic'

@pytest.mark.parametrize('family,strategy', [('nurbs','auto'), ('extruded','mapped'), ('ruled','mapped')])
def test_reissue_source_face_after_reopen(monkeypatch, family, strategy):
    import json
    from anymesher.regional_source_faces import capture_reusable_source_faces, reuse_source_faces
    from anymesher import hybrid
    from anymesher.errors import MeshError

    source, face, sheet = patch(family)
    other = source.add_points(((6.,0.,0.), (7.,0.,0.), (7.,1.,0.), (6.,1.,0.)))
    source.add_plate(other)
    def generate(model, **overrides):
        closure = owner.extract_model_closure(model, (model.handle('sheet',sheet),))
        token = qualify_model_closure_source_publication(model, closure)
        opts = dict(target_size=.45, strategy=strategy, geometry_approximation_error=.0125,
                    structural_preparation=False, mutation_policy='working_copy', native_backend='python')
        mesh = generate_hybrid_mesh_result(closure.working_model, **(opts | overrides)).mesh
        publish_model_closure_mesh_to_source(model, closure, mesh, qualification=token)
        return mesh

    first = generate(source)
    from anymesher.regional_source_numbering import remap_source_mesh_ids
    numbered = remap_source_mesh_ids(first, {n: 1000 + n for n in first.nodes},
        {e: 1000 + e for e in first.shells}, source_geometry=source)
    proposals = json.loads(json.dumps(capture_reusable_source_faces(source, first)))
    assert capture_reusable_source_faces(source, numbered) == proposals
    assert list(proposals) == [str(face)]
    data = proposals[str(face)]['data']
    assert {row for cell in data['cells'] for row in cell} == set(range(len(data['node_rows'])))
    reopened = owner.from_dict(owner.to_dict(source))
    reopened.move_point(other[0], 5.875, 0., 0.)
    bad = json.loads(json.dumps(proposals))
    bad[str(face)]['data']['cells'][0][0] += 100
    with pytest.raises(MeshError, match='differs from its source'):
        with reuse_source_faces(reopened, bad):
            pass
    def should_skip(*args, **kwargs):
        pytest.fail('retained face invoked surface cell generation')
    monkeypatch.setattr(hybrid, '_try_parametric_tensor', should_skip)
    if family == 'nurbs':
        # Even a self-consistent persisted packet is only input to the fresh
        # owner operation. The registered rim and the current support win.
        from anymesher.regional_source_faces import _digest
        wrong = json.loads(json.dumps(proposals))
        record = wrong[str(face)]
        edge_nodes = {n for _, rows in record['data']['rim'] for _, n in rows}
        interior = next(row for i,row in enumerate(record['data']['node_rows']) if i not in edge_nodes)
        interior[1][2] += .125
        record['digest'] = _digest(record['data'])
        with reuse_source_faces(reopened, wrong) as refused:
            with pytest.raises(MeshError, match='current surface evaluation'):
                generate(reopened)
        assert not refused['reused_faces']
        def cancel(phase):
            if phase == 'native face %d lifting start' % face:
                raise RuntimeError('cancel retained lifting')
        with reuse_source_faces(reopened, proposals) as cancelled:
            with pytest.raises(RuntimeError, match='cancel retained lifting'):
                generate(reopened, cancellation_check=cancel)
        assert not cancelled['reused_faces']
    with reuse_source_faces(reopened, proposals) as stats:
        second = generate(reopened)
    assert stats['reused_faces'] == {face}
    assert all(tuple(first.nodes[n]) == tuple(second.nodes[n]) for n in first.nodes)
    assert first.shells == second.shells
    if strategy == 'mapped':
        assert first.grid_of_face[face].tolist() == second.grid_of_face[face].tolist()
    proposals = capture_reusable_source_faces(reopened, second)
    with reuse_source_faces(reopened, proposals) as repeated:
        third = generate(reopened)
    assert repeated['reused_faces'] == {face}
    assert third.shells == first.shells


def test_automatic_bezier_selects_the_same_material_chart_as_generation():
    from anymesher.approximation_policy import resolve_geometry_approximation
    from anymesher.recovery import select_curvilinear_recovery_policy

    model, face, _ = patch('extruded')
    policy = resolve_geometry_approximation(model, (face,))
    automation, choice = select_curvilinear_recovery_policy(model, requested_strategy='auto', order='linear',
        certification_mode='interactive', geometry_approximation_error=policy.effective_error,
        explicit_controls=False, face_ids=(face,))
    assert automation is not None and choice['source_faces'] == choice['nurbs_tensor_faces'] == [face]
