"""Controlled metric/fixture tests; never execute a native mesher."""
import importlib.util
import itertools
import json
from pathlib import Path
import numpy as np
import pytest

PATH=Path(__file__).resolve().parents[1]/'tools/qualify_thin_volume.py'
spec=importlib.util.spec_from_file_location('thin_qualification',PATH)
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
REGULAR=np.array([[0.,0.,0.],[1.,0.,0.],[.5,np.sqrt(3)/2,0.],
    [.5,np.sqrt(3)/6,np.sqrt(2/3)]])


def test_regular_reference_and_all_vertex_permutations():
    permutations=np.array(list(itertools.permutations(range(4))))
    np.testing.assert_allclose(study.regular_tet_condition(REGULAR[permutations]),1.,rtol=2e-14,atol=2e-14)


@pytest.mark.parametrize('scale',[1e-4,1.,1e3])
def test_condition_is_invariant_under_relabeling_translation_rotation_and_scale(scale):
    angle=.71
    rotation=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
    points=(REGULAR@np.diag([2.,3.,7.])@rotation.T)*scale+[.3,-.2,.7]
    permutations=np.array(list(itertools.permutations(range(4))))
    np.testing.assert_allclose(study.regular_tet_condition(points[permutations]),3.5,rtol=2e-12,atol=2e-12)


@pytest.mark.parametrize('bad',['flat','nan','infinite'])
def test_invalid_shape_never_has_accepted_condition(bad):
    points=REGULAR.copy()
    if bad=='flat':points[:,2]=0
    else:points[1,1]=np.nan if bad=='nan' else np.inf
    with pytest.raises(ValueError):study.regular_tet_condition(points[None])


def test_sampling_uses_physical_director_and_absolute_thickness():
    points=np.array([[[0,0,0],[.03,0,0],[0,.02,0],[0,0,.004]]],dtype=float)
    np.testing.assert_allclose(study.sampling_fraction(points,[0,0,1],.01),.4)
    np.testing.assert_allclose(study.sampling_fraction(points+[.3,-.2,.7],[0,0,-1],.01),.4)
    coarse=points.copy();coarse[0,3,2]=.006
    assert study.sampling_fraction(coarse,[0,0,1],.01)[0]>.5
    for director,thickness in (([0,0,2],.01),([0,0,1],-.01),([0,0,np.nan],.01)):
        with pytest.raises(ValueError):study.sampling_fraction(points,director,thickness)


@pytest.mark.parametrize('case_id',['thin_sampling_aligned','thin_sampling_rotated_negative_offset'])
def test_fresh_actual_owner_fixture_roundtrip_and_analytical_geometry(case_id):
    from anygeometry import from_dict,to_dict
    manifest=json.loads(study.MANIFEST.read_text())
    case=next(case for case in manifest['cases'] if case['id']==case_id)
    q=study.helpers();model,snapshot,analytics=q.build_source(case)
    assert analytics['volume']==pytest.approx(.08*.03*.01,rel=1e-14)
    assert analytics['thickness']==pytest.approx(.01,rel=1e-14)
    assert np.linalg.norm(analytics['extrusion_direction'])==pytest.approx(1.)
    assert model.strict_audit().certifiable
    restored=from_dict(to_dict(model)).solid_region_snapshot(snapshot.region_id)
    assert q.canonical_hash(q.snapshot_record(restored))==q.canonical_hash(q.snapshot_record(snapshot))
    assert not snapshot.coordinates.flags.writeable


def test_prospective_limits_preserve_original_numerics_and_bound_new_calls():
    manifest=json.loads(study.MANIFEST.read_text())
    limits=manifest['limits']
    assert [limits[key] for key in ('residual','work_symmetry','resultant','patch_tensors','solved_displacement')]==[1e-9,1e-10,1e-8,1e-10,1e-7]
    assert limits['maximum_regular_tet_condition']==32 and limits['maximum_axial_span_fraction']==.5
    assert manifest['execution']['maximum_native_calls']==len(manifest['cases'])==2
    assert manifest['execution']['retries']==0
