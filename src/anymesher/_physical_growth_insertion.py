"""Bounded interior growth proposals; full physical/native gates own admission."""
from dataclasses import replace
import numpy as np
from .errors import MeshError
from .native_v2 import MutableT3Topology,_GeometryLimited


def refine_interior_growth(candidate,triangulation,settings,report,evaluate,cancellation_check=None):
    from .surface_mesh import _make_candidate,_physical_quality_candidate_from_xyz,_check_cancellation
    from ._physical_t3_refinement import _alternative_progress
    used=report.get('topology_operations',0);insertions=report.get('insertions',0)
    reused=report.get('reserved_node_reuses',0)
    if any(type(v) is not int or v<0 for v in (used,insertions,reused)):
        raise MeshError('growth insertion has invalid work receipt')
    original=(candidate.points.tobytes(),candidate.triangles.tobytes(),triangulation.segments.tobytes())
    def controls():
        return (settings.min_angle,settings.max_angle,settings.max_aspect_ratio,
            settings.min_scaled_jacobian,settings.max_warpage,settings.max_element_growth,
            settings.native_options.max_insertions,settings.native_options.max_topology_operations)
    policy=controls();callback_failure=owner_failure=None
    def check(phase):
        nonlocal callback_failure
        if controls()!=policy or original!=(candidate.points.tobytes(),candidate.triangles.tobytes(),triangulation.segments.tobytes()):
            raise MeshError('growth insertion input or policy changed')
        try:_check_cancellation(cancellation_check,phase)
        except BaseException as error:callback_failure=error;raise
        if controls()!=policy or original!=(candidate.points.tobytes(),candidate.triangles.tobytes(),triangulation.segments.tobytes()):
            raise MeshError('growth insertion input or policy changed')
    topology=MutableT3Topology(candidate.points,candidate.triangles,triangulation.segments)
    fixed=set(map(int,np.unique(triangulation.segments)))
    def state(points,cells):
        nonlocal owner_failure
        check('native-v2 growth insertion physical evaluation')
        stamp=(points.shape,points.dtype.str,points.tobytes(),cells.shape,cells.dtype.str,cells.tobytes())
        try:xyz=np.asarray(evaluate(points),dtype=float)
        except BaseException as error:owner_failure=error;raise
        check('native-v2 growth insertion physical evaluated')
        if stamp!=(points.shape,points.dtype.str,points.tobytes(),cells.shape,cells.dtype.str,cells.tobytes()):
            raise MeshError('growth insertion evaluated candidate changed')
        return _physical_quality_candidate_from_xyz(_make_candidate(points,cells,settings=settings),settings,xyz),xyz
    current,xyz=state(topology.points,topology.triangles)
    attempts=commits=refused=0;visited=set();staged=None
    def validate(points,cells):
        nonlocal staged
        trial,coordinates=state(points,cells)
        if not _alternative_progress(current.report,trial.report):
            raise _GeometryLimited('growth insertion makes no admissible physical progress')
        staged=trial,coordinates
    stop='no_untried_interior_growth_cell'
    while used<policy[-1] and insertions+reused<policy[-2]:
        check('native-v2 growth insertion priority')
        cells=current.triangles
        if current.report['invalid_element_count'] or not current.report['elements_above_maximum_growth']:
            stop='invalid_candidate' if current.report['invalid_element_count'] else 'growth_satisfied';break
        means=np.linalg.norm(np.roll(xyz[cells],-1,axis=1)-xyz[cells],axis=2).mean(axis=1)
        incidence={}
        for row,cell in enumerate(cells):
            for a,b in zip(cell,np.roll(cell,-1)):
                incidence.setdefault(tuple(sorted((int(a),int(b)))),[]).append(row)
        priority=[]
        for adjacent in incidence.values():
            if len(adjacent)!=2:continue
            large,small=sorted(adjacent,key=lambda row:(means[row],row),reverse=True)
            ratio=means[large]/means[small]
            key=tuple(sorted(map(int,cells[large])))
            if ratio>settings.max_element_growth and key not in visited and not fixed.intersection(key):
                priority.append((-float(ratio),key,large))
        if not priority:break
        _,key,row=min(priority);visited.add(key)
        point=current.points[cells[row]].mean(axis=0)
        check('native-v2 growth insertion trial')
        used+=1;attempts+=1;staged=None;callback_failure=owner_failure=None
        try:
            topology.insert_point(point,cancellation_check=check,preserve_splittable=True,_validate_candidate=validate)
        except _GeometryLimited as error:
            if error is callback_failure or error is owner_failure:raise
            refused+=1;continue
        if staged is None:raise MeshError('growth insertion lost physical admission')
        current,xyz=staged;insertions+=1;commits+=1
    else:stop='topology_budget' if used>=policy[-1] else 'insertion_budget'
    # Complete fresh evaluation is mandatory even after every local admission.
    current,xyz=state(topology.points,topology.triangles)
    check('native-v2 growth insertion complete')
    result=replace(candidate,points=current.points,triangles=current.triangles,
        report=current.report,score=current.score,aspect_ratios=current.aspect_ratios)
    updated=dict(report,topology_operations=used,insertions=insertions,
        growth_targeted_insertion={'attempts':attempts,'commits':commits,'refused':refused,
            'stop_reason':stop,'final_quality':dict(current.report)})
    return result,replace(triangulation,points=result.points,triangles=result.triangles),updated
