"""Certified contact acceleration, original fallback and retained-array timing."""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pytest

from anymesher import _volume_validation as v
from anymesher.volume import InvalidVolumeMesh, VolumeMeshingCancelled


def positive(points):
    p = np.array(points, dtype=float)
    if np.linalg.det(p[1:]-p[0]) < 0:
        p[[1,2]] = p[[2,1]]
    assert np.linalg.det(p[1:]-p[0]) > 0
    return p


def exact_sign(a,b,c,d):
    u=[F(float(x))-F(float(y)) for x,y in zip(b,a)]
    w=[F(float(x))-F(float(y)) for x,y in zip(c,a)]
    z=[F(float(x))-F(float(y)) for x,y in zip(d,a)]
    determinant = u[0]*(w[1]*z[2]-w[2]*z[1])+u[1]*(w[2]*z[0]-w[0]*z[2])+u[2]*(w[0]*z[1]-w[1]*z[0])
    return (determinant > 0)-(determinant < 0)


def invoke(check,p,q,common):
    return check(p,q,np.asarray(common,dtype=float).reshape(-1,3),v._tet_planes(p),v._tet_planes(q))


def outcome(check,p,q,common):
    try:
        invoke(check,p,q,common)
        return 'accept'
    except InvalidVolumeMesh:
        return 'refuse'


def test_orientation_filter_agrees_with_independent_rational_oracle():
    rng=np.random.default_rng(846901)
    certified=0
    for scale in (1.,1e-100,1e-200):
        for _ in range(120):
            a,b,c,d=rng.uniform(-1,1,(4,3))*scale
            sign=v._orientation_sign(a,b,c,d)
            if sign:
                certified+=1
                assert sign==exact_sign(a,b,c,d)
    assert certified >= 200
    assert v._orientation_sign([0,0,0],[1,0,0],[0,1,0],[.5,.5,0])==0
    # Subnormal arithmetic must never produce a false certified sign.
    assert v._orientation_sign([0,0,0],[1e-200,0,0],[0,1e-200,0],[0,0,1e-200])==0


def test_near_coplanar_filtered_signs_never_disagree_with_exact_binary_sign():
    rng=np.random.default_rng(39114)
    uncertain=0
    for _ in range(100):
        a,b,c=rng.uniform(-.4,.4,(3,3))
        point=a+.35*(b-a)+.25*(c-a)
        for direction in (-np.inf,np.inf):
            perturbed=np.nextafter(point,direction)
            sign=v._orientation_sign(a,b,c,perturbed)
            if sign:
                assert sign==exact_sign(a,b,c,perturbed)
            else:
                uncertain+=1
    assert uncertain>0


@pytest.mark.parametrize('shared',[1,2])
def test_proved_vertex_edge_contact_avoids_original_pair_loop(monkeypatch,shared):
    p=positive([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
    q=positive([[0,0,0],[-1,.2,.1],[-.2,-.5,.1],[-.3,0,-.4]] if shared==1 else
               [[0,0,0],[0,1,0],[-1,0,0],[-.4,.3,.7]])
    common=np.array([[0,0,0]] if shared==1 else [[0,0,0],[0,1,0]])
    assert outcome(v._check_pair_fallback,p,q,common)=='accept'
    def forbidden(*args):
        raise AssertionError('proved contact must not execute original pair loop')
    monkeypatch.setattr(v,'_check_pair_fallback',forbidden)
    invoke(v._check_pair,p,q,common)


def test_face_axis_separation_avoids_edge_cross_axes(monkeypatch):
    p=positive([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
    q=p+[2,0,0]
    monkeypatch.setattr(v,'_check_pair_fallback',lambda *a:pytest.fail('strictly separated tets used expensive SAT'))
    invoke(v._check_pair,p,q,[])


@pytest.mark.parametrize('shared',[1,2])
def test_shared_vertices_do_not_exempt_overlapping_tetrahedra(shared):
    p=positive([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
    q=positive([[0,0,0],[.7,.1,.1],[.1,.7,.1],[.1,.1,.7]] if shared==1 else
               [[0,0,0],[1,0,0],[.2,.3,.1],[.2,.1,.3]])
    common=[[0,0,0]] if shared==1 else [[0,0,0],[1,0,0]]
    with pytest.raises(InvalidVolumeMesh):
        invoke(v._check_pair,p,q,common)
    assert outcome(v._check_pair,p,q,common)==outcome(v._check_pair_fallback,p,q,common)


def test_extra_coplanar_contact_is_not_exempted_by_a_shared_vertex():
    p=positive([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
    q=positive([[0,0,0],[.25,.25,0],[-.3,.2,-1],[.2,-.3,-1]])
    with pytest.raises(InvalidVolumeMesh):
        invoke(v._check_pair,p,q,[[0,0,0]])


def test_uncertain_filter_uses_unchanged_fallback(monkeypatch):
    p=positive([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
    q=positive([[0,0,0],[-1,.2,.1],[-.2,-.5,.1],[-.3,0,-.4]])
    calls=[]
    monkeypatch.setattr(v,'_orientation_sign',lambda *a:0)
    monkeypatch.setattr(v,'_check_pair_fallback',lambda *a:calls.append(a))
    invoke(v._check_pair,p,q,[[0,0,0]])
    assert len(calls)==1


@pytest.mark.parametrize('shared',[0,1,2])
def test_pair_decisions_match_original_across_random_positive_cells(shared):
    rng=np.random.default_rng(4840+shared)
    for _ in range(80):
        p=positive(rng.uniform(-.4,.4,(4,3)))
        q=positive(np.vstack((p[:shared],rng.uniform(-.4,.4,(4-shared,3)))))
        decision=outcome(v._check_pair,p,q,p[:shared])
        assert decision==outcome(v._check_pair_fallback,p,q,p[:shared])
        if decision=='accept':
            v._exact_contact(p,q,p[:shared])


@pytest.mark.parametrize('height',[1.,1e-5,1e-9])
def test_complete_cell_validation_matches_original_for_thin_meshes(monkeypatch,height):
    from test_volume_mesh import box
    snapshot,cells=box(height)
    actual=v.validate_cells(snapshot.coordinates,cells)
    with monkeypatch.context() as context:
        context.setattr(v,'_check_pair',v._check_pair_fallback)
        expected=v.validate_cells(snapshot.coordinates,cells)
    np.testing.assert_array_equal(actual,expected)
    with pytest.raises(VolumeMeshingCancelled):
        v.validate_cells(snapshot.coordinates,cells,cancel=lambda:True)


@pytest.mark.parametrize('gap',[0.,float(np.spacing(np.float32(1)))])
def test_float32_projection_boundary_preserves_original_pair_routine(monkeypatch,gap):
    p=positive([[0,0,0],[1,0,0],[0,1,0],[0,0,1]]).astype(np.float32)
    q=(p+np.array([1+gap,0,0],dtype=np.float32)).astype(np.float32)
    common=[] if gap else [[1,0,0]]
    expected=outcome(v._check_pair_fallback,p,q,common)
    original=v._check_pair_fallback
    calls=[]
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(v,'_check_pair_fallback',counted)
    assert outcome(v._check_pair,p,q,common)==expected
    assert calls==[1]


def retained_admission(workspace,case,output,baseline=False):
    """One common production admission; no backend/solver invocation or repair."""
    from anygeometry import GeometryModel,from_dict,to_dict
    from anymesher.volume import _admit_provider_output
    workspace=Path(workspace);output=Path(output);output.mkdir(exist_ok=False,parents=True)
    manifest=json.loads((workspace/'ANYmesh/docs/retained_volume_qualification_manifest_20261028.json').read_text())
    parameters=next(c for c in manifest['cases'] if c['id']==case)
    raw=workspace/parameters['retained_mesh']
    digest=hashlib.sha256(raw.read_bytes()).hexdigest()
    assert digest==parameters['retained_sha256']
    if case=='box':
        geometry=workspace/'evidence/retained-box-qualification-20261028-results/geometry.json'
        model=from_dict(json.loads(geometry.read_text()))
        region_id=next(iter(model.solid_regions))
    else:
        model=GeometryModel()
        region_id=model.add_solid_box((0,0,0),parameters['dimensions'])
        original=model.solid_region_snapshot(region_id)
        model.set_solid_face_group(region_id,'load_face',(original.face_refs[1],))
        model.set_solid_face_group(region_id,'shell_transition',(original.face_refs[0],original.face_refs[1]))
    source=model.solid_region_snapshot(region_id)
    (output/'geometry.json').write_text(json.dumps(to_dict(model)))
    with np.load(raw,allow_pickle=False) as data:
        x,t=data['coordinates'],data['cells']
    source_validation=v.validate_source(source,parameters['boundary_tolerance'])
    if baseline:
        v._check_pair=v._check_pair_fallback
    started=time.perf_counter()
    mesh=_admit_provider_output(source_validation,x,t,{'backend':manifest['backend'],'retained_mesh_sha256':digest},parameters['target_edge_length'])
    elapsed=time.perf_counter()-started
    temporary=output/'mesh.tmp.npz'
    np.savez_compressed(temporary,coordinates=mesh.coordinates,cells=mesh.cells,boundary_facets=mesh.boundary_facets,
                        boundary_ids=mesh.boundary_ids,cell_ids=mesh.cell_ids,node_ids=mesh.node_ids)
    temporary.replace(output/'mesh.npz')
    result={'case':case,'baseline_original_pair_routine':baseline,'input_sha256':digest,
            'source_signature':source.source_signature,'source_model_id':source.model_id,'source_region_id':source.region_id,
            'admission_seconds':elapsed,'nodes':len(mesh.coordinates),'cells':len(mesh.cells),'boundary_facets':len(mesh.boundary_facets),
            'validation':dict(mesh.provenance['validation']),'mesh_sha256':hashlib.sha256((output/'mesh.npz').read_bytes()).hexdigest(),
            'native_calls':0,'fe_calls':0}
    temporary=output/'result.tmp.json';temporary.write_text(json.dumps(result,indent=2));temporary.replace(output/'result.json')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',required=True)
    parser.add_argument('--case',choices=('box','thin'),required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--baseline',action='store_true')
    options=parser.parse_args()
    retained_admission(options.workspace,options.case,options.output,options.baseline)
