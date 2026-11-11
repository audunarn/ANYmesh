"""Prospective five-case geometry qualification runner checks; no native calls."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'tools/qualify_volume_geometry.py'
spec=importlib.util.spec_from_file_location('geometry_qualification',path)
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
manifest=json.loads((path.parents[1]/'docs/volume_geometry_manifest_20261028.json').read_text())


@pytest.mark.parametrize('case',manifest['cases'],ids=lambda c:c['id'])
def test_required_case_actual_geometry_identity_snapshot_roundtrip(case):
    from anygeometry import from_dict,to_dict
    from anymesher._volume_validation import validate_source
    model,snapshot,analytics=q.build_source(case)
    assert snapshot.volume==pytest.approx(analytics['volume'],rel=1e-12)
    assert model.strict_audit().certifiable
    validate_source(snapshot,case['boundary_tolerance'])
    restored=from_dict(to_dict(model));other=restored.solid_region_snapshot(snapshot.region_id)
    assert q.canonical_hash(q.snapshot_record(other))==q.canonical_hash(q.snapshot_record(snapshot))
    assert snapshot.face_groups['load_face'][0] in snapshot.face_groups['loaded_overlap']
    assert set(snapshot.face_groups['shell_transition'])==set(snapshot.face_refs[:2])
    assert not snapshot.coordinates.flags.writeable
    triangles=q.triangles(snapshot)
    assert triangles.shape[1]==3 and len(triangles)==sum(len(loop)-2 for loop in snapshot.face_loops)


def test_rotated_negative_mm_conversion_is_explicit_and_analytic():
    case=next(c for c in manifest['cases'] if c['id']=='rotated_negative_mm')
    profile,vector,offset,area,_,dims=q.source_shape(case)
    np.testing.assert_allclose(profile[0],[.7,-.3,1.2],rtol=0,atol=1e-15)
    assert np.linalg.norm(profile[1]-profile[0])==pytest.approx(.3)
    assert np.linalg.norm(profile[-1]-profile[0])==pytest.approx(.2)
    normal=np.cross(profile[1]-profile[0],profile[-1]-profile[0]);normal/=np.linalg.norm(normal)
    assert vector@normal==pytest.approx(-.06)
    assert offset==.0125 and area==.06 and dims==[.3,.2,.06]


def test_nonfinite_json_never_publishes_a_result(tmp_path):
    for value in (float('nan'),np.float64('inf'),np.array([1.,float('-inf')])):
        with pytest.raises(ValueError,match='nonfinite'):
            q.write_json(tmp_path/'result.json',{'value':value})
        assert not (tmp_path/'result.json').exists()
    q.write_json(tmp_path/'result.json',{'count':np.int32(4),'array':np.array([1,2])})
    assert json.loads((tmp_path/'result.json').read_text())=={'count':4,'array':[1,2]}


def test_regular_tetrahedron_mean_ratio_formula_and_resolution_distribution():
    points=np.array([[0,0,0],[1,0,0],[.5,np.sqrt(3)/2,0],[.5,np.sqrt(3)/6,np.sqrt(2/3)]])
    volume=np.linalg.det(points[1:]-points[0])/6
    edges=np.array([np.linalg.norm(points[i]-points[j]) for i in range(4) for j in range(i+1,4)])
    assert 12*(3*volume)**(2/3)/(edges@edges)==pytest.approx(1.)
    assert q.distribution(edges)['median']==pytest.approx(1.)
    with pytest.raises(ValueError):q.distribution([np.nan])


def test_manifest_preserves_finite_authority_and_physical_criteria():
    assert len(manifest['cases'])==manifest['execution']['maximum_native_calls']==5
    assert manifest['execution']['native_calls_per_case']==1 and manifest['execution']['retries']==0
    assert manifest['execution']['maximum_native_process_seconds']==120
    assert manifest['execution']['complete_process_budget_seconds']==900
    assert manifest['limits']=={'maximum_cells':100000,'maximum_relative_volume_error':.001,'maximum_relative_face_area_error':.001,'maximum_face_normal_degrees':1}
    assert manifest['options']=={'simplify':False,'coarsen':False,'optimize':True,'num_threads':1}
