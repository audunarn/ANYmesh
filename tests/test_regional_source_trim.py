"""Fresh material publication of a curved chart with an aligned hole."""
from copy import deepcopy
import json

import numpy as np
import pytest
import anygeometry as owner
from anygeometry.entities import OrientedEdge
from anygeometry.nurbs import CurveOnSurface, NurbsCurve
from anygeometry.operations import trim_face
from anymesher import (generate_hybrid_mesh_result, publish_model_closure_mesh_to_source,
                       qualify_model_closure_source_publication)
from anymesher.errors import MeshError
from anymesher.material_source_transfer import _validated_curvilinear_reference_uv
from anymesher.regional_source_faces import active_source_trim, capture_reusable_source_faces, reuse_source_faces
from anymesher.regional_source_trim import plan_source_face_trim
from test_regional_source_generation import patch


@pytest.mark.parametrize('divisions', (1, 2))
def test_aligned_hole_admits_the_exact_exterior(monkeypatch, divisions):
    source, face, sheet = patch('nurbs')
    def generate(model, **options):
        closure = owner.extract_model_closure(model, (model.handle('sheet',sheet),))
        token = qualify_model_closure_source_publication(model, closure)
        mesh = generate_hybrid_mesh_result(closure.working_model, target_size=.25, strategy='auto',
            geometry_approximation_error=.0125, structural_preparation=False, native_backend='python',
            mutation_policy='working_copy', **options).mesh
        publish_model_closure_mesh_to_source(model, closure, mesh, qualification=token)
        return mesh
    original = generate(source)
    model = owner.from_dict(owner.to_dict(source))
    records = json.loads(json.dumps(capture_reusable_source_faces(source, original)))
    record = records[str(face)]
    data = record['data']
    uv = np.asarray([r[0] for r in data['node_rows']])
    if divisions == 1:
        cut = min(range(len(data['cells'])), key=lambda i: np.linalg.norm(uv[data['cells'][i]].mean(axis=0) - .375))
        corners = list(reversed(data['cells'][cut]))
        removed = [cut]
    else:
        index = {tuple(v):i for i,v in enumerate(uv)}
        corners = [index[u] for u in ((.25,.25),(.5,.25),(.75,.25),(.75,.5),(.75,.75),
                                       (.5,.75),(.25,.75),(.25,.5))]
        removed = [i for i,c in enumerate(data['cells']) if np.all((uv[c] >= .25) & (uv[c] <= .75))]
    surface = model.faces[face].surface
    points = model.add_points([data['node_rows'][i][1] for i in corners])
    edges = []
    for i, n in enumerate(corners):
        j = (i+1) % len(corners)
        pcurve = NurbsCurve(1,(0.,0.,1.,1.),(tuple(uv[n]),tuple(uv[corners[j]])))
        edges.append(model.add_curve(points[i],points[j],CurveOnSurface(surface,pcurve)))
    trim_face(model,face,[tuple(OrientedEdge(e,True) for e in edges)])
    counts = {e:1 for e in edges}
    plan = plan_source_face_trim(model,record,target_size=.25,overrides=counts)
    assert plan is not None and plan['replaced_cell_rows']==removed
    assert plan_source_face_trim(model,record,target_size=.25,overrides={edges[0]:2}) is None
    assert plan_source_face_trim(model,record,target_size=.25)['replaced_cell_rows'] == removed
    from anymesher import hybrid
    def forbidden(*args,**kwargs):
        pytest.fail('aligned trim invoked whole-face tessellation')
    monkeypatch.setattr(hybrid, '_try_parametric_tensor', forbidden)
    monkeypatch.setattr(hybrid, 'mesh_planar_surface', forbidden)
    assert not active_source_trim(model)
    with reuse_source_faces(model,records,trim_patches={str(face):plan}) as stats:
        assert active_source_trim(model) and not active_source_trim(source)
        child = generate(model,overrides=counts)
    assert not active_source_trim(model)
    assert not stats['reused_faces'] and not stats['patched_faces']
    evidence = stats['trimmed_faces'][face]
    assert len(child.shells)==len(original.shells)-divisions**2
    assert set(_validated_curvilinear_reference_uv(model, child)) == set(child.shells)
    assert set(evidence['kept_cell_rows']) == set(range(len(data['cells']))) - set(removed)
    for old,e in evidence['kept_cell_rows'].items():
        assert child.shells[e] == tuple(evidence['old_node_rows'][n] for n in data['cells'][old])
    for edge,stations in plan['hole_rim']:
        assert child.nodes_of_edge[edge] == [evidence['old_node_rows'][n] for t,n in stations]
    if divisions == 2:
        assert len(evidence['old_node_rows']) == len(data['node_rows'])-1
        lost, = set(range(len(data['node_rows']))) - set(evidence['old_node_rows'])
        corrupted = deepcopy(record)
        corrupted['data']['node_rows'][lost][1][2] += .125
        from anymesher.regional_source_faces import _digest
        corrupted['digest'] = _digest(corrupted['data'])
        with pytest.raises(MeshError,match='current surface'):
            plan_source_face_trim(model,corrupted,target_size=.25)
    broken = deepcopy(plan)
    broken['replaced_cell_rows'] = []
    with pytest.raises(MeshError,match='planned cut'):
        with reuse_source_faces(model,records,trim_patches={str(face):broken}):
            generate(model,overrides=counts)
    assert not active_source_trim(model)
    def cancel(phase):
        if phase == 'source trim owned hole interval':
            raise RuntimeError('cancel aligned trim')
    with reuse_source_faces(model,records,trim_patches={str(face):plan}) as cancelled:
        with pytest.raises(RuntimeError,match='cancel aligned trim'):
            generate(model,overrides=counts,cancellation_check=cancel)
    assert not cancelled['trimmed_faces']


@pytest.mark.parametrize('single', [False, True], ids=['default-graded','explicit-growth-2'])
def test_non_aligned_convex_trim_builds_a_source_annulus(monkeypatch, single):
    model, face, sheet = patch('nurbs')
    from anymesher.structured import StructuredMeshingOptions
    quality_options = StructuredMeshingOptions(max_element_growth=2.) if single else None
    quality = {'max_element_growth':2.} if single else None
    target = .2
    def generate(owner_model, **options):
        native_quality = options.pop('native_quality',quality_options)
        closure = owner.extract_model_closure(owner_model,(owner_model.handle('sheet',sheet),))
        token = qualify_model_closure_source_publication(owner_model, closure)
        mesh = generate_hybrid_mesh_result(closure.working_model, target_size=target, strategy='auto',
            geometry_approximation_error=.0125, structural_preparation=False, native_backend='python',
            mutation_policy='working_copy', quality_options=quality_options,
            _native_surface_options=native_quality, **options).mesh
        publish_model_closure_mesh_to_source(owner_model, closure, mesh, qualification=token)
        return mesh
    parent = generate(model)
    records = json.loads(json.dumps(capture_reusable_source_faces(model,parent)))
    record = records[str(face)]
    before = owner.to_dict(model)
    model = owner.from_dict(before)
    def cut_hole(source, coordinates):
        surface = source.faces[face].surface
        vertices = source.add_points([tuple(map(float,surface.exact_point(*p))) for p in coordinates])
        edges = []
        for i,p in enumerate(coordinates):
            j = (i+1)%4
            curve = CurveOnSurface(surface,NurbsCurve(1,(0.,0.,1.,1.),(p,coordinates[j])))
            edges.append(source.add_curve(vertices[i],vertices[j],curve))
        trim_face(source,face,[tuple(OrientedEdge(e,True) for e in edges)])
        return {e:1 for e in edges}
    interior = [(.453125,.45703125),(.45703125,.55078125),(.55078125,.54296875),(.54296875,.44921875)]
    controls = cut_hole(model,interior)
    if not single:
        crossed = owner.from_dict(before)
        wider = [(u+.125,v) for u,v in interior]
        over = cut_hole(crossed,wider)
        assert plan_source_face_trim(crossed,record,target_size=target,overrides=over) is None
    plan = plan_source_face_trim(model,record,target_size=target,overrides=controls,quality=quality)
    assert plan is not None and plan['annulus']
    removed, added = (1,4) if single else (17,52)
    assert len(plan['replaced_cell_rows'])==removed and len(plan['new_cells'])==added
    assert plan_source_face_trim(model,record,target_size=target,overrides={min(controls):2}) is None
    assert plan_source_face_trim(model,record,target_size=target,overrides=controls,quality={'min_angle':75.}) is None
    altered = deepcopy(plan)
    altered['interface_rows'].reverse()
    with pytest.raises(MeshError,match='planned cut'):
        with reuse_source_faces(model,records,trim_patches={str(face):altered}):
            pytest.fail('wrong annulus interface was accepted')
    from anymesher import hybrid
    def refuse(*args,**kwargs):
        pytest.fail('trim annulus invoked whole-face tessellation')
    monkeypatch.setattr(hybrid, 'mesh_planar_surface', refuse)
    monkeypatch.setattr(hybrid, '_try_parametric_tensor', refuse)
    if single:
        with reuse_source_faces(model,records,trim_patches={str(face):plan}) as denied:
            with pytest.raises(MeshError,match='actual generation request'):
                generate(model,overrides=controls,native_quality=None)
        assert not denied['trimmed_faces']
    with reuse_source_faces(model,records,trim_patches={str(face):plan}) as stats:
        child = generate(model,overrides=controls)
    issued = stats['trimmed_faces'][face]
    assert issued['annulus'] and set(issued['new_cell_ids']).isdisjoint(issued['kept_cell_rows'].values())
    assert len(child.shells)==len(parent.shells)-removed+added
    assert set(_validated_curvilinear_reference_uv(model, child)) == set(child.shells)
    for old,e in issued['kept_cell_rows'].items():
        assert child.quads[e] == tuple(issued['old_node_rows'][n] for n in record['data']['cells'][old])
    for e,stations in plan['hole_rim']:
        assert child.nodes_of_edge[e] == [issued['new_node_rows'][n] for t,n in stations]
