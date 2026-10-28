"""Finite Workflow 2 admission/physics checks on registered retained meshes.

Never executes a mesher or changes provider admission. The production postcheck
path is shared, while volumes, loads, work and tensor references are analytical.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import traceback
from uuid import NAMESPACE_URL, uuid5
import numpy as np


def face_measure(points):
    cross = np.cross(points[1:-1]-points[0], points[2:]-points[0])
    area = np.linalg.norm(cross,axis=1)/2
    centroid = np.sum(area[:,None]*(points[0]+points[1:-1]+points[2:])/3,axis=0)/area.sum()
    normal = cross.sum(axis=0)
    return float(area.sum()), centroid, normal/np.linalg.norm(normal)


def vector_resultants(prepared, load):
    if not np.isfinite(load).all() or not np.isfinite(prepared.coefficient_coordinates).all():
        raise ValueError('nonfinite load or coefficient coordinates')
    components = prepared.coefficient_components
    force = np.bincount(components,weights=load,minlength=3)
    vectors = np.zeros((prepared.ndof,3))
    vectors[np.arange(prepared.ndof),components] = load
    moment = np.cross(prepared.coefficient_coordinates,vectors).sum(axis=0)
    if not np.isfinite(force).all() or not np.isfinite(moment).all():
        raise ValueError('nonfinite force or moment resultant')
    return force,moment


def rigid_checks(prepared, stiffness, limits):
    """Check physical rank/span independently of the backend's rigid vectors."""
    modes = np.asarray(prepared.rigid_modes())
    if modes.shape != (prepared.ndof,6) or not np.isfinite(modes).all():
        raise ValueError('invalid rigid mode shape or nonfinite values')
    lengths = np.linalg.norm(modes,axis=0)
    if np.any(lengths == 0) or not np.isfinite(lengths).all():
        raise ValueError('zero or nonfinite rigid mode column')
    modes = modes/lengths
    coords = np.asarray(prepared.coefficient_coordinates)
    components = np.asarray(prepared.coefficient_components)
    center = coords.mean(axis=0)
    physical = np.column_stack([np.eye(3)[components,i] for i in range(3)] + [
        np.cross(np.eye(3)[i],coords-center)[np.arange(prepared.ndof),components]
        for i in range(3)])
    physical /= np.linalg.norm(physical,axis=0)
    if not np.isfinite(physical).all():
        raise ValueError('nonfinite physical rigid modes')
    basis,singular,_ = np.linalg.svd(physical,full_matrices=False)
    candidate_singular = np.linalg.svd(modes,compute_uv=False)
    rank_floor = limits['work_symmetry']
    if min(singular[-1],candidate_singular[-1]) <= rank_floor:
        raise ValueError('rigid mode rank is less than six')
    span_error = float(np.linalg.norm(modes-basis@(basis.T@modes)))/np.linalg.norm(modes)
    denominator = max(float(np.linalg.norm(stiffness.data)),1.)
    physical_residual = float(np.linalg.norm(stiffness@physical))/denominator
    candidate_residual = float(np.linalg.norm(stiffness@modes))/denominator
    metrics = dict(rigid_span_error=span_error,rigid_min_singular=float(candidate_singular[-1]),
        physical_rigid_min_singular=float(singular[-1]),rigid_residual=candidate_residual,
        physical_rigid_residual=physical_residual)
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError('nonfinite rigid mode metrics')
    return dict(metrics=metrics,checks={'six_rigid_modes':
        span_error <= limits['work_symmetry'] and
        max(physical_residual,candidate_residual) <= limits['residual']})


def physical_checks(mesh, source, order, limits, output):
    import anysolid
    from scipy.sparse.linalg import splu
    checks, metrics = {}, {}
    E, nu = 1000., .3
    prepared = anysolid.prepare(anysolid.SolidMesh.snapshot(mesh),
        {'elastic_modulus':E,'poisson_ratio':nu},order=order)
    if prepared.ndof > limits['maximum_coefficients']:
        raise ValueError('registered coefficient resource cap exceeded before assembly')
    k = prepared.assemble_linear()
    if not np.isfinite(k.data).all():
        raise ValueError('nonfinite assembled stiffness')
    normk = max(float(np.linalg.norm(k.data)),1.)
    metrics['coefficients'] = int(prepared.ndof)
    metrics['symmetry'] = float(np.linalg.norm((k-k.T).data))/normk
    checks['symmetry'] = metrics['symmetry'] <= limits['work_symmetry']
    rigid = rigid_checks(prepared,k,limits)
    metrics.update(rigid['metrics'])
    checks.update(rigid['checks'])
    gradient = np.array([[.03,.08,-.02],[.04,-.01,.05],[.02,-.03,.06]])
    strain = (gradient+gradient.T)/2
    mu,lam = E/(2*(1+nu)),E*nu/((1+nu)*(1-2*nu))
    stress = 2*mu*strain+lam*np.trace(strain)*np.eye(3)
    values = prepared.coefficient_coordinates@gradient.T+[.1,-.2,.3]
    q = values[np.arange(prepared.ndof),prepared.coefficient_components]
    loads = np.zeros(prepared.ndof)
    resultant_error = 0.
    measures = []
    for i,loop in enumerate(source.face_loops):
        area,center,normal = face_measure(source.coordinates[list(loop)])
        measures.append((area,center,normal))
        group = 'start' if i==0 else 'end' if i==1 else f'side_{i-2}'
        traction = stress@normal
        load = prepared.traction(group,traction)
        f,m = vector_resultants(prepared,load)
        expected = area*traction
        scale = max(float(np.linalg.norm(expected)),1.)
        resultant_error = max(resultant_error,float(np.linalg.norm(f-expected))/scale,
            float(np.linalg.norm(m-np.cross(center,expected)))/max(scale*np.linalg.norm(center),1.))
        loads += load
    metrics['face_resultants'] = resultant_error
    checks['face_force_moment'] = resultant_error <= limits['resultant']
    metrics['patch_residual'] = float(np.linalg.norm(k@q-loads))/max(float(np.linalg.norm(loads)),1.)
    checks['affine_equilibrium'] = metrics['patch_residual'] <= limits['residual']
    recovery = prepared.recover(q)
    if not np.isfinite(recovery.strain).all() or not np.isfinite(recovery.stress).all():
        raise ValueError('nonfinite recovered strain or stress tensor')
    metrics['strain_error'] = float(np.max(np.abs(recovery.strain-strain)))
    metrics['stress_error'] = float(np.max(np.abs(recovery.stress-stress)))/max(float(np.max(np.abs(stress))),1.)
    checks['engineering_shear_full_tensors'] = max(metrics['strain_error'],metrics['stress_error']) <= limits['patch_tensors']
    reference = source.volume*float(np.sum(strain*stress))
    # Prospectively remove rigid motion for quadratic/work evaluation; the full
    # shifted/rotated field above and all six rigid modes are checked separately.
    # This is the same physical deformation and avoids large rigid cancellation.
    center = (measures[0][1]+measures[1][1])/2
    centered = (prepared.coefficient_coordinates-center)@strain.T
    work_q = centered[np.arange(prepared.ndof),prepared.coefficient_components]
    metrics['shifted_energy_diagnostic'] = abs(float(q@k@q)/reference-1)
    metrics['energy_error'] = abs(float(work_q@k@work_q)/reference-1)
    metrics['work_error'] = abs(float(work_q@loads)/reference-1)
    checks['positive_energy_and_work'] = float(work_q@k@work_q)>0 and max(metrics['energy_error'],metrics['work_error']) <= limits['work_symmetry']
    body = np.array([2.,-3.,5.])
    body_load = prepared.body_load(body)
    f,m = vector_resultants(prepared,body_load)
    centroid = (measures[0][1]+measures[1][1])/2
    expected = source.volume*body
    metrics['body_resultant_error'] = max(float(np.linalg.norm(f-expected)),float(np.linalg.norm(m-np.cross(centroid,expected))))/max(float(np.linalg.norm(expected)),1.)
    checks['body_force_moment'] = metrics['body_resultant_error'] <= limits['resultant']
    pressure = prepared.pressure('load_face',3.)
    f,m = vector_resultants(prepared,pressure)
    area,center,normal = measures[1]
    expected = -3*area*normal
    metrics['pressure_resultant_error'] = max(float(np.linalg.norm(f-expected)),float(np.linalg.norm(m-np.cross(center,expected))))/max(float(np.linalg.norm(expected)),1.)
    checks['pressure_force_moment'] = metrics['pressure_resultant_error'] <= limits['resultant']
    del k,prepared,recovery,body_load,loads,pressure
    # A second independently specified BVP: uniform normal tension, nu=0,
    # all translations fixed at the start cap. Its exact affine field is known.
    prepared = anysolid.prepare(mesh,{'elastic_modulus':E,'poisson_ratio':0.},order=order)
    k = prepared.assemble_linear()
    area,_,axis = measures[1]
    f = prepared.traction('load_face',axis)
    fixed = prepared.boundary_dofs('start')
    free = np.setdiff1d(np.arange(prepared.ndof),fixed)
    solved = np.zeros(prepared.ndof)
    solved[free] = splu(k[free][:,free].tocsc()).solve(f[free])
    if not np.isfinite(solved).all():
        raise ValueError('nonfinite solved coefficients')
    origin = measures[0][1]
    axial = (prepared.coefficient_coordinates-origin)@axis
    exact = (axial[:,None]*axis/E)[np.arange(prepared.ndof),prepared.coefficient_components]
    metrics['solved_displacement_error'] = float(np.linalg.norm(solved-exact))/max(float(np.linalg.norm(exact)),1e-30)
    residual = k@solved-f
    metrics['solved_free_residual'] = float(np.linalg.norm(residual[free]))/max(float(np.linalg.norm(f)),1.)
    force,moment = vector_resultants(prepared,f+np.where(np.isin(np.arange(prepared.ndof),fixed),residual,0.))
    metrics['reaction_balance'] = max(float(np.linalg.norm(force)),float(np.linalg.norm(moment)))/max(area,1.)
    checks['solved_affine_displacement'] = metrics['solved_displacement_error'] <= limits['solved_displacement']
    checks['constrained_equilibrium'] = metrics['solved_free_residual'] <= limits['residual']
    checks['reactions'] = metrics['reaction_balance'] <= limits['resultant']
    np.savez_compressed(output/f'order{order}-solution.npz',coefficients=solved,exact=exact,
        fixed=fixed,load=f,reactions=residual,coefficient_coordinates=prepared.coefficient_coordinates)
    nonfinite = [key for key,value in metrics.items() if not np.isfinite(value)]
    if nonfinite:
        raise ValueError('nonfinite qualification metrics: '+', '.join(nonfinite))
    return dict(order=order,metrics=metrics,checks=checks,status='PASS' if all(checks.values()) else 'FAIL')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--case',required=True)
    p.add_argument('--workspace',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--rigid-only',action='store_true',help='Supplement previous checks without repeating FE solves')
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    assert manifest['status']=='READY_RETAINED_MESH_QUALIFICATION'
    case = next(c for c in manifest['cases'] if c['id']==a.case)
    raw = a.workspace/case['retained_mesh']
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==case['retained_sha256']
    a.output.mkdir(parents=True,exist_ok=False)
    result = dict(case=case,status='FAILED',native_calls=0,phases={})
    started = time.perf_counter()
    try:
        import anygeometry
        from anymesher._volume_validation import validate_source
        from anymesher.volume import _admit_provider_output,validate_volume_mesh
        model = anygeometry.GeometryModel(model_id=str(uuid5(NAMESPACE_URL,'Workflow2/'+case['id'])))
        rid = model.add_solid_box((0,0,0),case['dimensions'])
        source = model.solid_region_snapshot(rid)
        model.set_solid_face_group(rid,'load_face',(source.face_refs[1],))
        model.set_solid_face_group(rid,'shell_transition',(source.face_refs[0],source.face_refs[1]))
        source = model.solid_region_snapshot(rid)
        assert model.strict_audit().certifiable
        (a.output/'geometry.json').write_text(json.dumps(anygeometry.to_dict(model)),encoding='utf-8')
        with np.load(raw,allow_pickle=False) as data:
            coordinates,cells = data['coordinates'],data['cells']
        validation = validate_source(source,case['boundary_tolerance'])
        t = time.perf_counter()
        mesh = _admit_provider_output(validation,coordinates,cells,
            {'backend':manifest['backend'],'retained_mesh_sha256':case['retained_sha256']},case['target_edge_length'])
        result['phases']['production_admission_seconds'] = time.perf_counter()-t
        print(json.dumps({'case':case['id'],'phase':'admitted','nodes':len(mesh.coordinates),'cells':len(mesh.cells),'wall':time.perf_counter()-started}),flush=True)
        # Independent consumer numerical integration also exercises all boundary
        # groups and the source-owned region selection, with no global rotations.
        assert set(map(int,mesh.cell_groups[source.region_id]))==set(map(int,mesh.cell_ids))
        result['orders'] = []
        for order in (1,2):
            if a.rigid_only:
                import anysolid
                prepared = anysolid.prepare(anysolid.SolidMesh.snapshot(mesh),
                    {'elastic_modulus':1000.,'poisson_ratio':.3},order=order)
                if prepared.ndof > manifest['limits']['maximum_coefficients']:
                    raise ValueError('registered coefficient resource cap exceeded')
                checked = rigid_checks(prepared,prepared.assemble_linear(),manifest['limits'])
                checked.update(order=order,status='PASS' if all(checked['checks'].values()) else 'FAIL')
                del prepared
            else:
                checked = physical_checks(mesh,source,order,manifest['limits'],a.output)
            result['orders'].append(checked)
            (a.output/f'order{order}-checks.json').write_text(json.dumps(checked,indent=2,allow_nan=False),encoding='utf-8')
            print(json.dumps({'case':case['id'],'phase':f'order{order}',
                'status':checked['status'],'wall':time.perf_counter()-started}),flush=True)
        t = time.perf_counter()
        validate_volume_mesh(mesh,source,boundary_tolerance=case['boundary_tolerance'])
        result['phases']['readmission_seconds'] = time.perf_counter()-t
        result.update(nodes=len(mesh.coordinates),cells=len(mesh.cells),
            versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','scikit-fem')},
            status='PASS' if all(x['status']=='PASS' for x in result['orders']) else 'FAIL')
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        result['traceback'] = traceback.format_exc()
        traceback.print_exc()
    result['total_seconds'] = time.perf_counter()-started
    (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('orders','traceback')}),flush=True)
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
