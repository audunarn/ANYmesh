"""Canonical shape-star scheduling around the existing mixed numerical core."""
from dataclasses import replace
from types import FunctionType
import numpy as np
from .errors import MeshError
from .core import MeshCore
from ._physical_mixed_sizing import _eligible_chart,_score,_positive,_cover
from ._physical_corner_cavity import _closed_mixed_link
from ._physical_t3_refinement import _alternative_progress


class _OriginalStars:
    __slots__=('_data',)
    def __init__(self,points,mesh,movable):
        frozen=tuple(sorted(set(range(len(points)))-set(movable)))
        cells=tuple(tuple(map(int,cell)) for rows,active in ((mesh.triangle_connectivity,mesh.triangle_active),(mesh.quad_connectivity,mesh.quad_active)) for cell in rows[active] if set(map(int,cell))&set(movable))
        object.__setattr__(self,'_data',(points.shape,points.tobytes(),frozen,cells))
    def __setattr__(self,name,value):raise AttributeError('original-star data is immutable')
    def __call__(self,proposed):
        data=object.__getattribute__(self,'_data')
        if type(data) is not tuple or len(data)!=4:raise MeshError('original-star data must be an immutable four-tuple')
        dimensions=data[0]
        if (type(dimensions) is not tuple or len(dimensions)!=2 or type(dimensions[0]) is not int or type(dimensions[1]) is not int
                or dimensions[0]<0 or dimensions[1]!=2 or type(data[1]) is not bytes or len(data[1])!=16*dimensions[0]):raise MeshError('invalid original-star shape or bytes')
        if type(data[2]) is not tuple or any(type(node) is not int or node<0 or node>=dimensions[0] for node in data[2]):raise MeshError('invalid original-star frozen rows')
        if type(data[3]) is not tuple:raise MeshError('invalid original-star cells')
        for cell in data[3]:
            if type(cell) is not tuple or len(cell) not in (3,4) or any(type(node) is not int or node<0 or node>=dimensions[0] for node in cell):raise MeshError('invalid original-star cell data')
        shape,raw,frozen,cells=data
        if type(proposed) is not np.ndarray or proposed.shape!=shape:return False
        original=np.frombuffer(raw,dtype=np.float64).reshape(shape)
        return (np.array_equal(proposed[list(frozen)],original[list(frozen)])
                and all(_positive(proposed,cell) for cell in cells))


def _check_context(context):
    from . import surface_mesh as surface
    full,fullcode,fullservices,pair,paircode,pairservices=context[:6]
    if not (surface._physical_full_quality_services is full and type(full) is FunctionType
            and full.__code__ is fullcode and full.__defaults__ is None and full.__kwdefaults__ is None
            and surface._physical_pairing_services is pair and type(pair) is FunctionType
            and pair.__code__ is paircode and pair.__defaults__ is None and pair.__kwdefaults__ is None
            and fullservices is surface._REFERENCE_PHYSICAL_FULL_SERVICES
            and pairservices is surface._REFERENCE_PHYSICAL_PAIRING_SERVICES
            and fullservices is not None and pairservices is not None
            and full()==fullservices and pair()==pairservices):
        raise MeshError('shape patch calculation changed before dispatch')


def _capture_context():
    from . import surface_mesh as surface
    full,fullcode=surface._PHYSICAL_FULL_BUILDER_REF,surface._PHYSICAL_FULL_BUILDER_CODE
    pair,paircode=surface._PHYSICAL_PAIRING_BUILDER_REF,surface._PHYSICAL_PAIRING_BUILDER_CODE
    if not (surface._physical_full_quality_services is full and type(full) is FunctionType and full.__code__ is fullcode and full.__defaults__ is None and full.__kwdefaults__ is None
            and surface._physical_pairing_services is pair and type(pair) is FunctionType and pair.__code__ is paircode and pair.__defaults__ is None and pair.__kwdefaults__ is None):return None
    fs,ps=surface._REFERENCE_PHYSICAL_FULL_SERVICES,surface._REFERENCE_PHYSICAL_PAIRING_SERVICES
    if fs is None or ps is None or full()!=fs or pair()!=ps:return None
    return (full,fullcode,fs,pair,paircode,ps,_CHECK_REF,_CHECK_CODE,_OriginalStars,_STAR_CALL,_STAR_CODE)


def _patch(mesh,row,fixed):
    free=set(map(int,mesh.triangle_connectivity[row]))-fixed
    if not free:return ()
    touched=[cell for cells,active in ((mesh.triangle_connectivity,mesh.triangle_active),(mesh.quad_connectivity,mesh.quad_active)) for cell in cells[active] if set(map(int,cell))&free]
    movable=set(int(n) for cell in touched for n in cell)-fixed
    if not movable or len(movable)>24 or not all(_closed_mixed_link(mesh,n) for n in movable):return ()
    owners={int(owners[i]) for cells,active,owners in ((mesh.triangle_connectivity,mesh.triangle_active,mesh.triangle_owner_handles),(mesh.quad_connectivity,mesh.quad_active,mesh.quad_owner_handles)) for i in np.flatnonzero(active) if set(map(int,cells[i]))&movable}
    return tuple(sorted(movable)) if len(owners)==1 else ()


def _world_core(mesh,xyz):
    return MeshCore(xyz,mesh.triangle_connectivity,mesh.quad_connectivity,node_ids=mesh.node_ids,triangle_ids=mesh.triangle_ids,quad_ids=mesh.quad_ids,node_active=mesh.node_active,triangle_active=mesh.triangle_active,quad_active=mesh.quad_active,owner_table=mesh.owner_table,node_owner_handles=mesh.node_owner_handles,triangle_owner_handles=mesh.triangle_owner_handles,quad_owner_handles=mesh.quad_owner_handles)


def _admissible(before,after):
    return (not after['invalid_element_count']
            and all(after['violation_counts'][key]<=value for key,value in before['violation_counts'].items())
            and _alternative_progress(before,after))


def repair_shape_patches(points,report,settings,evaluate,fixed_rows,*,cancellation_check=None):
    facts=dict(enabled=False,attempts=0,commits=0,rejections=0,work=0,patches=[],stop_reason='ineligible')
    if cancellation_check is not None:return points,report,facts
    # Both introduced checkers are captured and checked BEFORE dispatch.
    if not ( _capture_context is _CAPTURE_REF and type(_capture_context) is FunctionType and _capture_context.__code__ is _CAPTURE_CODE and _capture_context.__defaults__ is None and _capture_context.__kwdefaults__ is None):return points,report,facts
    context=_CAPTURE_REF()
    if context is None:return points,report,facts
    checker,code=context[6:8]
    def check():
        if not (_check_context is checker and type(checker) is FunctionType and checker.__code__ is code and checker.__defaults__ is None and checker.__kwdefaults__ is None):raise MeshError('shape patch checker changed before dispatch')
        checker(context)
    check();chart=_eligible_chart(evaluate)
    if chart is None:return points,report,facts
    binding=chart.region_binding;binding.validate(None);chart.validate_reference()
    current=np.asarray(points,dtype=float).copy();mesh=report.mesh
    if evaluate(current).tobytes()!=mesh.node_coordinates.tobytes():raise MeshError('shape patch initial owner XYZ differs')
    check();before,q=_score(mesh,settings);facts.update(enabled=True,initial_quality=before)
    from .surface_mesh import _physical_chart_invalid_count
    if before['invalid_element_count'] or _physical_chart_invalid_count(current,mesh.triangle_connectivity):raise MeshError('shape patch requires a valid complete UV cover')
    if not before['quality_violation_count'] or not any(before['violation_counts'].values()):
        binding.validate(None);chart.validate_reference();facts.update(stop_reason='no_shape_failure',final_quality=before);return points,report,facts
    fixed=set(map(int,fixed_rows));original=current.copy();protected_uv=current[sorted(fixed)].tobytes();protected_xyz=mesh.node_coordinates[sorted(fixed)].tobytes()
    initial=q.triangles
    seeds=sorted((float(initial.minimum_angle[i]),-float(initial.maximum_angle[i]),int(identifier)) for i,identifier in enumerate(initial.element_ids) if initial.minimum_angle[i]<settings.min_angle or initial.maximum_angle[i]>settings.max_angle or initial.scaled_jacobian[i]<settings.min_scaled_jacobian or initial.aspect_ratio[i]>settings.max_aspect_ratio or initial.warpage[i]>settings.max_warpage)
    from ._physical_mixed_repair import repair_mixed_core_coordinates
    visited=set();accepted_movable=set()
    for _,_,identifier in seeds:
        if identifier in visited:continue
        visited.add(identifier)
        if report.exchange_work>=settings.max_recombination_work:facts['stop_reason']='matching_budget';break
        check();before,q=_score(mesh,settings)
        indices=np.flatnonzero(q.triangles.element_ids==identifier)
        if not len(indices):continue
        i=int(indices[0]);g=q.triangles
        if not (g.minimum_angle[i]<settings.min_angle or g.maximum_angle[i]>settings.max_angle or g.scaled_jacobian[i]<settings.min_scaled_jacobian or g.aspect_ratio[i]>settings.max_aspect_ratio or g.warpage[i]>settings.max_warpage):continue
        row=int(np.flatnonzero(mesh.triangle_ids==identifier)[0]);movable=_patch(mesh,row,fixed);check()
        if not movable:continue
        guard=_OriginalStars(current,mesh,movable);guard_data=object.__getattribute__(guard,'_data');check()
        if not _STAR_CALL(guard,current):continue
        freeze=sorted(set(range(len(current)))-set(movable));diagnostics={};facts['attempts']+=1
        local_settings=replace(settings,max_recombination_work=min(settings.max_recombination_work,report.exchange_work+128))
        made,used,truncated=repair_mixed_core_coordinates(current,mesh.triangle_connectivity,mesh,local_settings,evaluate,freeze,exchange_work=report.exchange_work,evaluate_jacobians=chart.jacobians,_trial_guard=guard,_diagnostics=diagnostics,_owned_context=context+(guard_data,))
        check()
        if object.__getattribute__(guard,'_data') is not guard_data:raise MeshError('shape patch original-star data changed')
        globally_exhausted=report.exchange_work+used>=settings.max_recombination_work
        report=replace(report,exchange_work=report.exchange_work+used,exchange_truncated=report.exchange_truncated or bool(truncated and globally_exhausted));facts['work']+=used
        xyz=evaluate(made);check();candidate=_world_core(mesh,xyz);after,quality=_score(candidate,settings);check()
        safe=(_admissible(before,after) and not _physical_chart_invalid_count(made,mesh.triangle_connectivity) and _STAR_CALL(guard,made))
        facts['patches'].append(dict(id=identifier,movable=movable,work=used,accepted=bool(safe),local_truncated=bool(truncated and not globally_exhausted),global_exhausted=globally_exhausted,core_commits=diagnostics['commits'],patch_nodes=diagnostics.get('patch_nodes'),final_quality=after))
        if safe:
            qm=quality.quadrilaterals;scores=qm.scaled_jacobian-.08*np.log(np.maximum(qm.aspect_ratio,1.))-.002*np.maximum(0.,90.-qm.minimum_angle)-.002*np.maximum(0.,qm.maximum_angle-90.)-.25*qm.warpage
            current,mesh=made,candidate;report=replace(report,mesh=mesh,scores=scores);facts['commits']+=1;accepted_movable.update(movable)
        if not safe:
            facts['rejections']+=1
            continue
        if not any(after['violation_counts'].values()) and safe:facts['stop_reason']='shape_satisfied';break
    else:facts['stop_reason']='initial_shape_stars_exhausted'
    outside=sorted(set(range(len(current)))-accepted_movable)
    if current[outside].tobytes()!=original[outside].tobytes():raise MeshError('shape patch changed rows outside accepted movable sets')
    check();fresh=evaluate(current);check()
    if fresh.tobytes()!=mesh.node_coordinates.tobytes() or current[sorted(fixed)].tobytes()!=protected_uv or fresh[sorted(fixed)].tobytes()!=protected_xyz:raise MeshError('shape patch final source rows or owner XYZ changed')
    binding.validate(None);chart.validate_reference();final,_=_score(mesh,settings)
    facts.update(final_quality=final,stop_reason=facts['stop_reason'] if facts['stop_reason']!='ineligible' else 'initial_shape_stars_exhausted')
    return current,report,facts


_CHECK_REF=_check_context
_CHECK_CODE=_check_context.__code__
_CAPTURE_REF=_capture_context
_CAPTURE_CODE=_capture_context.__code__
_STAR_CALL=vars(_OriginalStars)['__call__']
_STAR_CODE=_STAR_CALL.__code__
