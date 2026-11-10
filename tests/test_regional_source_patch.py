from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest
import anygeometry as owner

from anymesher import (generate_hybrid_mesh_result, publish_model_closure_mesh_to_source,
                       qualify_model_closure_source_publication)
from anymesher.errors import MeshError
from anymesher.refinement import Refinement
from anymesher.regional_source_faces import (
    capture_reusable_source_faces, capture_source_face_identity_rows, reuse_source_faces)
from anymesher.regional_source_patch import plan_source_face_point_patch
from test_regional_source_generation import patch


def test_fixed_interface_point_patch_reopen(monkeypatch):
    from anymesher import hybrid
    source, face, sheet = patch('nurbs')
    target = .20
    def generate(model, **options):
        closure = owner.extract_model_closure(model, (model.handle('sheet', sheet),))
        token = qualify_model_closure_source_publication(model, closure)
        mesh = generate_hybrid_mesh_result(closure.working_model, target_size=target, strategy='auto',
            geometry_approximation_error=.0125, structural_preparation=False,
            mutation_policy='working_copy', native_backend='python', **options).mesh
        publish_model_closure_mesh_to_source(model, closure, mesh, qualification=token)
        return mesh

    old = generate(source)
    assert not old.tris
    table = capture_source_face_identity_rows(source, old, face)
    records = json.loads(json.dumps(capture_reusable_source_faces(source, old)))
    record = records[str(face)]
    data = record['data']
    uv = np.array([r[0] for r in data['node_rows']])
    xyz = np.array([r[1] for r in data['node_rows']])
    centre = min(data['cells'], key=lambda c: np.linalg.norm(uv[c].mean(axis=0) - .5))
    child_uv = (uv[centre].mean(axis=0) + uv[centre[0]]) / 2
    h = max(np.linalg.norm(xyz[centre] - np.roll(xyz[centre], 1, axis=0), axis=1))
    reopened = owner.from_dict(owner.to_dict(source))
    zone = Refinement(size=.54*h, radius=.04*h, growth=6., center=reopened.face_point(face, *child_uv))
    plan = plan_source_face_point_patch(reopened, record, zone, target_size=target)
    assert plan is not None, (len(data['cells']), centre, child_uv)
    assert len(plan['replaced_cell_rows']) == 7 and len(plan['interface_rows']) == 12
    assert plan_source_face_point_patch(reopened, record, zone, target_size=target, strategy='mapped') is None
    assert plan_source_face_point_patch(reopened, record, replace(zone, radius=2.), target_size=target) is None
    assert plan_source_face_point_patch(reopened, record, replace(zone, center=(.5,.5,2.)), target_size=target) is None
    assert plan_source_face_point_patch(reopened, record, replace(zone, size=zone.size/4), target_size=target) is None
    def skip(*args, **kwargs):
        pytest.fail('whole-face tessellation invoked on the patch')
    monkeypatch.setattr(hybrid, '_try_parametric_tensor', skip)
    monkeypatch.setattr(hybrid, 'mesh_planar_surface', skip)
    with reuse_source_faces(reopened, records, point_patches={str(face):plan}) as stats:
        fresh = generate(reopened, refinements=[zone])
    report = stats['patched_faces'][face]
    assert not stats['reused_faces'] and not fresh.tris
    assert len(old.quads) + 15 == len(fresh.quads)
    assert set(report['kept_cell_rows']) == set(range(len(data['cells']))) - set(plan['replaced_cell_rows'])
    for k, new in report['old_node_rows'].items():
        assert np.array_equal(fresh.nodes[new], old.nodes[table['nodes'][k]])
    for k, e in report['kept_cell_rows'].items():
        assert fresh.quads[e] == tuple(report['old_node_rows'][n] for n in data['cells'][k])
    new_cells = set(fresh.quads) - set(report['kept_cell_rows'].values())
    fine = [np.array([fresh.nodes[n] for n in fresh.quads[e]]) for e in new_cells]
    assert any(np.linalg.norm(p.mean(axis=0) - zone.center) <= zone.radius and
               max(np.linalg.norm(p - np.roll(p, 1, axis=0), axis=1)) <= zone.size for p in fine)
    from anymesher.serialize import mesh_from_dict, mesh_to_dict
    from anymesher.regional_source_numbering import remap_source_mesh_ids
    numbered = remap_source_mesh_ids(fresh, {n: n+100 for n in fresh.nodes},
                                     {e: e+200 for e in fresh.shells}, source_geometry=reopened)
    assert numbered.seeding.size_field.zones == (zone,)
    assert np.array_equal(numbered.seeding.size_field.size_at([zone.center]), [zone.size])
    recovered = mesh_from_dict(json.loads(json.dumps(mesh_to_dict(numbered))))
    metadata = recovered.structural_preparation['prepared_boundary_provenance']['source_seeding_size_field']
    assert metadata == {'kind':'point_refinements', 'target_size':target, 'source_model_id':str(reopened.model_id),
                        'zones':[dict(plan['zone'], mode=zone.mode, name=zone.name)]}
    with pytest.raises(MeshError, match='live material source authority'):
        capture_source_face_identity_rows(reopened, recovered, face)
    damaged = deepcopy(plan)
    damaged['interface_rows'].reverse()
    with reuse_source_faces(reopened, records, point_patches={str(face):damaged}) as refused:
        with pytest.raises(MeshError, match='operator'):
            generate(reopened, refinements=[zone])
    assert not refused['patched_faces']
    from anymesher.regional_source_faces import _digest
    corrupt = deepcopy(records)
    packet = corrupt[str(face)]
    rim_row = packet['data']['rim'][0][1][0][1]
    packet['data']['node_rows'][rim_row][1][2] += .125
    packet['digest'] = _digest(packet['data'])
    proposed = plan_source_face_point_patch(reopened, packet, zone, target_size=target)
    with reuse_source_faces(reopened, corrupt, point_patches={str(face):proposed}) as refused:
        with pytest.raises(MeshError, match='boundary changed'):
            generate(reopened, refinements=[zone])
    assert not refused['patched_faces']
    def cancel(phase):
        if phase == f'native face {face} lifting start':
            raise RuntimeError('cancel source patch')
    with reuse_source_faces(reopened, records, point_patches={str(face):plan}) as cancelled:
        with pytest.raises(RuntimeError, match='cancel source patch'):
            generate(reopened, refinements=[zone], cancellation_check=cancel)
    assert not cancelled['patched_faces']


def test_point_seeding_refuses_inconsistent_prepared_coordinates():
    from anymesher._source_seeding_publication import _source_size_field
    from anymesher.refinement import SizeField
    source, _, _ = patch('nurbs')
    zone = Refinement(center=(.5,.5,.03125), size=.1, growth=4.)
    original = SizeField(source, .2, (zone,))
    original._sources[0][0, 0] = .125
    with pytest.raises(MeshError, match='prepared coordinates'):
        _source_size_field(source, original)


def test_one_fine_cell_cannot_satisfy_a_coarse_neighbor_inside_the_same_zone():
    from anymesher.regional_source_patch import _achieves_zone
    from anymesher.refinement import SizeField
    ref = np.array([[-.5,-.5,0.],[.5,-.5,0.],[.5,.5,0.],[-.5,.5,0.]])
    cells = np.stack([.1*ref, .3*ref + [.2,0.,0.]])
    # Second centroid lies in the constant core, then in the transition.
    for radius in (.3, .15):
        zone = Refinement(center=(0.,0.,0.), radius=radius, size=.12, growth=1.5)
        field = SizeField(None, .4, (zone,))
        assert _achieves_zone(field, zone, cells[:1], 0.)
        assert not _achieves_zone(field, zone, cells, 0.)
