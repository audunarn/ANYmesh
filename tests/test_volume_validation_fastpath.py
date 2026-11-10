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


from itertools import permutations


def kuhn_grid(xs,ys,zs,*,shuffle_seed=None,flip_voxel=None,drop_voxel=None):
    """Positive Kuhn tet4 mesh of the product grid; optional shuffling/tampering.

    flip_voxel uses the opposite main diagonal there (a conflicting
    triangulation); drop_voxel omits a voxel (hole).
    """
    xs,ys,zs=(np.asarray(a,dtype=float) for a in (xs,ys,zs))
    nx,ny,nz=len(xs),len(ys),len(zs)
    node=lambda i,j,k:(i*ny+j)*nz+k
    x=np.array([[xs[i],ys[j],zs[k]] for i in range(nx) for j in range(ny) for k in range(nz)])
    cells=[]
    for i in range(nx-1):
        for j in range(ny-1):
            for k in range(nz-1):
                if (i,j,k)==drop_voxel:
                    continue
                flip=(i,j,k)==flip_voxel
                for perm in permutations(range(3)):
                    corner=np.array([i+1,j,k] if flip else [i,j,k]);chain=[corner.copy()]
                    for axis in perm:
                        corner=corner.copy()
                        corner[axis]+= -1 if (flip and axis==0) else 1
                        chain.append(corner)
                    cell=[node(*c) for c in chain]
                    if np.linalg.det(x[cell[1:]]-x[cell[0]])<0:
                        cell[1],cell[2]=cell[2],cell[1]
                    cells.append(cell)
    cells=np.array(cells)
    if shuffle_seed is not None:
        rng=np.random.default_rng(shuffle_seed)
        new=rng.permutation(len(x))                    # old -> new row
        inverse=np.argsort(new)
        x=x[inverse];cells=new[cells]
        cells=cells[rng.permutation(len(cells))]
        # even permutation of local vertices preserves orientation
        even=[(0,1,2,3),(1,2,0,3),(2,0,1,3),(0,2,3,1),(0,3,1,2),(1,0,3,2)]
        cells=np.array([c[list(even[rng.integers(len(even))])] for c in cells])
    return x,cells


def normalized(x,t):
    return v._normalized_mesh(x,t)


GRID=([0,.3,1.1,1.2],[0,.5,.6],[0,.25,.9,1.3])


@pytest.mark.parametrize('seed',[None,1,2])
def test_certificate_accepts_kuhn_grid_in_any_ordering(seed):
    y,t=normalized(*kuhn_grid(*GRID,shuffle_seed=seed))
    assert v._certify_cartesian_kuhn(y,t)


def test_fast_certificate_skips_pair_search_and_matches_generic(monkeypatch):
    x,t=kuhn_grid(*GRID,shuffle_seed=7)
    with monkeypatch.context() as context:
        context.setattr(v,'_check_pair',lambda *a:pytest.fail('certified mesh used pair search'))
        fast=v.validate_cells(x,t)
    with monkeypatch.context() as context:
        context.setattr(v,'_certify_cartesian_kuhn',lambda *a:False)
        generic=v.validate_cells(x,t)
    np.testing.assert_array_equal(fast,generic)


def test_certified_mesh_still_honours_cancellation():
    x,t=kuhn_grid(*GRID)
    with pytest.raises(VolumeMeshingCancelled):
        v.validate_cells(x,t,cancel=lambda:True)


def test_rotated_mesh_is_not_certified_and_uses_generic_search(monkeypatch):
    x,t=kuhn_grid(*GRID)
    c,s=np.cos(.4),np.sin(.4)
    rotated=x@np.array([[c,-s,0],[s,c,0],[0,0,1]]).T
    y,tt=normalized(rotated,t)
    assert not v._certify_cartesian_kuhn(y,tt)
    calls=[]
    original=v._check_pair
    monkeypatch.setattr(v,'_check_pair',lambda *a:(calls.append(1),original(*a))[1])
    v.validate_cells(rotated,t)
    assert calls


@pytest.mark.parametrize('vertex,shift',[(5,[1e-3,0,0]),(11,[0,0,2e-4]),(0,[0,-1e-6,1e-6])])
def test_perturbed_vertex_is_not_certified(vertex,shift):
    x,t=kuhn_grid(*GRID)
    x[vertex]+=shift
    y,tt=normalized(x,t)
    assert not v._certify_cartesian_kuhn(y,tt)


def test_conflicting_triangulation_is_not_certified_and_is_refused():
    x,t=kuhn_grid(*GRID,flip_voxel=(1,0,1))
    y,tt=normalized(x,t)
    assert not v._certify_cartesian_kuhn(y,tt)
    with pytest.raises(InvalidVolumeMesh):
        v.validate_cells(x,t)


def test_missing_voxel_is_not_certified():
    x,t=kuhn_grid(*GRID,drop_voxel=(0,1,1))
    y,tt=normalized(x,t)
    assert not v._certify_cartesian_kuhn(y,tt)


def test_duplicate_pattern_overlap_counterexample_is_not_certified():
    x,t=kuhn_grid(*GRID)
    y,_=normalized(x,t)
    t=t.copy()
    t[1]=t[0][[1,0,3,2]]            # same tet, other vertex order, replaces a cell
    assert not v._certify_cartesian_kuhn(y,t)
    t=t.copy()
    t[1]=t[2]                       # exact duplicate key
    assert not v._certify_cartesian_kuhn(y,t)


def test_non_kuhn_cell_patterns_are_not_certified():
    x,t=kuhn_grid(*GRID)
    y,_=normalized(x,t)
    # a cell whose vertex-sum ordering is not a unit-step chain
    bad=t.copy()
    bad[0]=[t[0][0],t[0][1],t[6][1],t[6][2]]
    assert not v._certify_cartesian_kuhn(y,bad)


def test_certificate_predicates_refuse_dtype_gap_and_shape_doubt():
    x,t=kuhn_grid(*GRID)
    y,tt=normalized(x,t)
    assert not v._certify_cartesian_kuhn(y.astype(np.float32),tt)
    assert not v._certify_cartesian_kuhn(y,tt.astype(float))
    assert not v._certify_cartesian_kuhn(y,tt[:-1])
    thin=([0,1e-14,1],[0,.5,1],[0,.5,1])
    x2,t2=kuhn_grid(*thin)
    y2=(x2-x2[0])/np.linalg.norm(x2.max(axis=0)-x2.min(axis=0))
    assert not v._certify_cartesian_kuhn(y2,t2)
    assert not v._certify_cartesian_kuhn(y[:,:2],tt)


def count_generic_fallbacks(monkeypatch):
    calls=[]
    original=v._generic_overlap_check
    monkeypatch.setattr(v,'_generic_overlap_check',lambda *a,**k:(calls.append(1),original(*a,**k))[1])
    return calls


def test_fast_path_matches_generic_and_never_enters_fallback(monkeypatch):
    g=np.linspace(0,1,5),np.linspace(0,1,5),np.linspace(0,.5,3)
    x,t=kuhn_grid(*g,shuffle_seed=3)
    calls=count_generic_fallbacks(monkeypatch)
    start=time.perf_counter();fast=v.validate_cells(x,t);fast_time=time.perf_counter()-start
    assert calls==[]
    with monkeypatch.context() as context:
        context.setattr(v,'_certify_cartesian_kuhn',lambda *a:False)
        start=time.perf_counter();generic=v.validate_cells(x,t);generic_time=time.perf_counter()-start
    assert calls==[1]
    np.testing.assert_array_equal(fast,generic)
    print(f'diagnostic only: fast={fast_time:.4f}s generic={generic_time:.4f}s')   # never asserted


# ---- near-Cartesian (ULP-noise) certificate ---------------------------------

def trilinear_grid(shape,origin,lengths):
    """Kuhn mesh whose coordinates come from weighted trilinear generation.

    Each node is sum_c w_c(u,v,w)*corner_c over the 8 box corners with
    non-dyadic parameters, so adjacent lattice rows may differ by a few ulps.
    (Whether they do on a given platform is not asserted here.)
    """
    params=[np.linspace(0,1,n) for n in shape]
    ideal,cells=kuhn_grid(*params)              # ideal holds the (u,v,w) parameters
    origin,lengths=np.asarray(origin,float),np.asarray(lengths,float)
    corners=[origin+lengths*np.array(c) for c in np.ndindex(2,2,2)]
    out=np.zeros_like(ideal)
    for point,(a,b,c) in enumerate(ideal):
        weights=[(a if i else 1-a)*(b if j else 1-b)*(c if k else 1-c) for i,j,k in np.ndindex(2,2,2)]
        out[point]=sum(w*corner for w,corner in zip(weights,corners))
    return out,cells


def ulp_noise(x,seed,ulps=4):
    rng=np.random.default_rng(seed)
    k=rng.integers(-ulps,ulps+1,x.shape)
    return x+k*np.spacing(np.abs(x))


NOISE_GRID=([0,.3,1.1,1.2,1.7],[0,.4,.5,.9],[0,.1,.65])


@pytest.mark.parametrize('seed',[None,5])
def test_ulp_noisy_grid_is_certified_with_more_values_than_stations(seed):
    x,t=kuhn_grid(*NOISE_GRID,shuffle_seed=seed)
    noisy=ulp_noise(x,11)
    y,tt=normalized(noisy,t)
    assert max(len(np.unique(y[:,a])) for a in range(3))>max(len(g) for g in NOISE_GRID)   # noise really present
    assert not np.array_equal(noisy,x)
    assert v._certify_cartesian_kuhn(y,tt)


def test_actual_trilinear_9x9x5_grid_is_certified_without_fallback(monkeypatch):
    x,t=trilinear_grid((9,9,5),(.1,-.2,.3),(2.3,1.7,.9))
    y,tt=normalized(x,t)
    assert v._certify_cartesian_kuhn(y,tt)
    calls=count_generic_fallbacks(monkeypatch)
    v.validate_cells(x,t)
    assert calls==[]


def test_noisy_grid_output_matches_generic_validator(monkeypatch):
    x,t=kuhn_grid(*NOISE_GRID,shuffle_seed=2)
    x=ulp_noise(x,3)
    fast=v.validate_cells(x,t)
    with monkeypatch.context() as context:
        context.setattr(v,'_certify_cartesian_kuhn',lambda *a:False)
        generic=v.validate_cells(x,t)
    np.testing.assert_array_equal(fast,generic)


def test_noisy_grid_still_honours_cancellation():
    x,t=kuhn_grid(*NOISE_GRID)
    with pytest.raises(VolumeMeshingCancelled):
        v.validate_cells(ulp_noise(x,1),t,cancel=lambda:True)


def test_bound_is_evaluated_and_can_refuse(monkeypatch):
    x,t=kuhn_grid(*NOISE_GRID)
    y,tt=normalized(ulp_noise(x,4),t)
    assert v._certify_cartesian_kuhn(y,tt)
    monkeypatch.setattr(v,'_KUHN_MAX_LIPSCHITZ',1e-12)
    assert not v._certify_cartesian_kuhn(y,tt)


def test_documented_caps_imply_a_bound_far_below_one():
    delta=v._STATION_SPAN/2+v._BOUND_CUSHION
    assert 6*delta/(v._KUHN_MIN_GAP-v._BOUND_CUSHION)<1e-3


@pytest.mark.parametrize('scale',[1e-9,1e-6,1e-3])
def test_perturbation_beyond_cap_is_not_certified(scale):
    x,t=kuhn_grid(*NOISE_GRID)
    x[7]+=scale
    y,tt=normalized(x,t)
    assert not v._certify_cartesian_kuhn(y,tt)


def test_large_overlapping_perturbation_is_refused_by_validation(monkeypatch):
    x,t=kuhn_grid(*NOISE_GRID)
    interior=(1*len(NOISE_GRID[1])+1)*len(NOISE_GRID[2])+1
    x[interior]+=[1.0,0,0]                     # carries the vertex past its neighbours
    y,tt=normalized(x,t)
    assert not v._certify_cartesian_kuhn(y,tt)
    with pytest.raises(InvalidVolumeMesh):
        v.validate_cells(x,t)


def test_hidden_near_duplicate_vertex_is_not_certified():
    x,t=kuhn_grid(*NOISE_GRID)
    x[6]=x[5]+np.spacing(np.abs(x[5]))          # same station triple as row 5
    y,tt=normalized(x,t)
    assert len(np.unique(y,axis=0))==len(y)     # exact-duplicate screen cannot see it
    assert not v._certify_cartesian_kuhn(y,tt)


def chained_layer(spacing_eps,count_axis=2,level=2):
    x,t=kuhn_grid(*NOISE_GRID)
    diagonal=np.linalg.norm(x.max(axis=0)-x.min(axis=0))
    layer=np.flatnonzero(x[:,count_axis]==NOISE_GRID[count_axis][level])
    for rank,row in enumerate(layer):
        x[row,count_axis]+=rank*spacing_eps*np.finfo(float).eps*diagonal
    return x,t,len(layer)


def test_chain_clustering_within_span_is_certified():
    eps=np.finfo(float).eps
    x,t,count=chained_layer(8)
    assert 8*eps<v._STATION_JOIN and (count-1)*8*eps<v._STATION_SPAN
    assert v._certify_cartesian_kuhn(*normalized(x,t))


def test_unbounded_chain_clustering_is_refused():
    # every link (40 eps) is below the join threshold, the total span is not
    eps=np.finfo(float).eps
    x,t,count=chained_layer(40)
    assert 40*eps<v._STATION_JOIN
    assert (count-1)*40*eps>v._STATION_SPAN
    assert not v._certify_cartesian_kuhn(*normalized(x,t))


def test_skinny_margin_around_minimum_station_gap():
    def thin(gap_over_min,noise):
        probe=np.array([[0,0,0],[1,1,1.]])
        diagonal=np.linalg.norm(probe[1]-probe[0])
        x,t=kuhn_grid([0,.5,1],[0,.5,1],[0,gap_over_min*v._KUHN_MIN_GAP*diagonal,1])
        x=ulp_noise(x,8) if noise else x
        return v._certify_cartesian_kuhn(*normalized(x,t))
    assert thin(4,False) and thin(4,True)
    assert not thin(.5,False) and not thin(.5,True)


def test_oblique_and_float32_noisy_meshes_use_generic_search(monkeypatch):
    x,t=kuhn_grid(*NOISE_GRID)
    x=ulp_noise(x,9)
    c,s=np.cos(.3),np.sin(.3)
    rotated=x@np.array([[c,-s,0],[s,c,0],[0,0,1]]).T
    assert not v._certify_cartesian_kuhn(*normalized(rotated,t))
    assert not v._certify_cartesian_kuhn(*normalized(x.astype(np.float32),t))


def test_noise_cannot_hide_a_missing_voxel_or_conflicting_diagonal():
    for kwargs in ({'drop_voxel':(0,1,1)},{'flip_voxel':(1,0,1)}):
        x,t=kuhn_grid(*NOISE_GRID,**kwargs)
        assert not v._certify_cartesian_kuhn(*normalized(ulp_noise(x,5),t))


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
