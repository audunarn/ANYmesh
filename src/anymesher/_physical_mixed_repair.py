"""Detached owner-evaluated UV trials for a fixed mixed-cell partition."""
from fractions import Fraction
import numpy as np


def _batch_chart(evaluate, cancellation_check):
    """Admit only the callback-free, owner-bound two-NURBS repair route."""
    from types import MethodType
    if cancellation_check is not None or type(evaluate) is not MethodType:
        return None
    from ._analytic_metric_chart import AnalyticMetricChart, NurbsSurface
    from ._material_region_binding import MaterialRegionBinding
    if (evaluate.__func__ is not AnalyticMetricChart.evaluate
            or type(evaluate.__self__) is not AnalyticMetricChart):
        return None
    chart = evaluate.__self__
    if chart.check is not None or type(chart.region_binding) is not MaterialRegionBinding:
        return None
    region = chart.region_binding.region
    if type(region.support) is not NurbsSurface:
        return None
    try:
        from anygeometry.nurbs_intersection_curve import NurbsSurfaceIntersectionCurve
    except ModuleNotFoundError as error:
        if error.name != 'anygeometry.nurbs_intersection_curve':
            raise
        return None
    if not any(
            type(path.curve) is NurbsSurfaceIntersectionCurve
            for loop in region.boundaries for path in loop):
        return None
    if getattr(chart, '_authority', None) is None:
        from .errors import MeshError
        raise MeshError('mixed UV repair chart is missing owner reference authority')
    chart.validate_reference()
    if chart.check is not None:
        return None
    return chart


def _growth_descent(mesh, xyz, limit):
    """Descent proposal for squared positive mean-edge excess; no admission."""
    cells=[];lengths=[];incidence={}
    for rows,active in ((mesh.triangle_connectivity,mesh.triangle_active),
                        (mesh.quad_connectivity,mesh.quad_active)):
        for cell in rows[active]:
            index=len(cells);cells.append(cell)
            lengths.append(float(np.linalg.norm(np.roll(xyz[cell],-1,axis=0)-xyz[cell],axis=1).mean()))
            for a,b in zip(cell,np.roll(cell,-1)):
                incidence.setdefault(tuple(sorted((int(a),int(b)))),[]).append(index)
    violations=[]
    for owners in incidence.values():
        if len(owners)!=2:continue
        a,b=owners
        if min(lengths[a],lengths[b])<=0:continue
        big,small=(a,b) if lengths[a]>=lengths[b] else (b,a)
        ratio=lengths[big]/lengths[small]
        if ratio>limit:violations.append((ratio,big,small))
    gradient={}
    for ratio,big,small in violations:
        excess=ratio-limit
        for index,factor in ((big,excess/lengths[small]),(small,-excess*ratio/lengths[small])):
            cell=cells[index]
            for k,node in enumerate(cell):
                left=xyz[node]-xyz[cell[k-1]];right=xyz[node]-xyz[cell[(k+1)%len(cell)]]
                a,b=float(np.linalg.norm(left)),float(np.linalg.norm(right))
                if a<=0 or b<=0:return {}
                value=factor*(left/a+right/b)/len(cell)
                gradient[int(node)]=gradient.get(int(node),np.zeros(3))+value
    return gradient


def _project_growth_halfspaces(values, jacobian, anchor, *, iterations=2048):
    """Bounded numerical proposal only: intersect linearized halfspaces/disks."""
    norms=np.linalg.norm(jacobian,axis=1)
    movable=norms>0
    if np.any(values<0) and not np.any(movable&(values<0)):return None,0
    # Rows farther away than the entire normalized box cannot become active.
    active=movable&(values<np.sum(np.abs(jacobian),axis=1)*2)
    if not np.any(active):return anchor.copy(),0
    a=jacobian[active]/norms[active,None]
    b=(jacobian[active]@anchor-values[active])/norms[active]
    # Strict interior proposal buffer at floating precision; the actual policy
    # is unchanged and every resulting mesh still receives exact full gates.
    b+=np.sqrt(np.finfo(float).eps)
    made=anchor.copy()
    for step in range(iterations):
        violation=b-a@made
        row=int(np.argmax(violation))
        if violation[row]<=1.e-10:return made,step
        made+=violation[row]*a[row]
        blocks=made.reshape(-1,2)
        radius=np.linalg.norm(blocks,axis=1)
        blocks/=np.maximum(radius,1.)[:,None]
    return made,iterations


# Only these array operations change shape in the optional arithmetic path.
_CONSTRAINT_NO_VALUE=np._NoValue

def _constraint_operation_pin(function):
    from types import FunctionType,BuiltinFunctionType
    from ._material_region_binding import _boundary_literal_pin,_BOUNDARY_ARRAY_DISPATCH_TYPE
    def literal(value):
        if value is _CONSTRAINT_NO_VALUE:return ('numpy_no_value',id(_CONSTRAINT_NO_VALUE))
        kind=type(value)
        if not (kind is type(None) or kind is bool or kind is int or kind is str or kind is bytes or kind is float or kind is tuple and not value):return None
        return _boundary_literal_pin(value)
    if (type(function) is BuiltinFunctionType or type(function) is np.ufunc):return ('native',id(function),id(type(function)))
    if type(function) is _BOUNDARY_ARRAY_DISPATCH_TYPE:
        wrapped=object.__getattribute__(function,'__wrapped__')
        if (type(wrapped) is BuiltinFunctionType or type(wrapped) is np.ufunc):return ('dispatch_native',id(function),id(wrapped),id(type(wrapped)))
        if type(wrapped) is not FunctionType:return None
        identity=('dispatch',id(function));function=wrapped
    elif type(function) is FunctionType:identity=('python',)
    else:return None
    defaults=function.__defaults__;keywords=function.__kwdefaults__
    if defaults is not None and type(defaults) is not tuple:return None
    if keywords is not None and type(keywords) is not dict:return None
    positional=tuple(literal(v) for v in defaults) if defaults is not None else ()
    named=tuple((_boundary_literal_pin(k),literal(v)) for k,v in keywords.items()) if keywords is not None else ()
    if any(v is None for v in positional) or any(k is None or v is None for k,v in named):return None
    return identity+(id(function),id(function.__code__),positional,named)


_CONSTRAINT_NUMPY = tuple(_constraint_operation_pin(f) for f in
    (np.asarray,np.frombuffer,np.roll,np.linalg.norm,np.concatenate,np.minimum,np.maximum,np.where,np.full,np.isfinite))


def _constraint_numpy_current():
    return all(pin is not None for pin in _CONSTRAINT_NUMPY) and _CONSTRAINT_NUMPY == tuple(_constraint_operation_pin(f) for f in
        (np.asarray,np.frombuffer,np.roll,np.linalg.norm,np.concatenate,np.minimum,np.maximum,np.where,np.full,np.isfinite))


def _constraint_topology(mesh):
    return tuple((array.dtype.str,array.shape,array.tobytes()) for array in
        (mesh.triangle_connectivity,mesh.triangle_active,mesh.quad_connectivity,mesh.quad_active))


def _prepare_constraint_layout(mesh):
    """Operation-local immutable row/edge data; no geometry or quality authority."""
    if not _constraint_numpy_current():return None
    if (mesh.triangle_connectivity.shape[1]!=3 or mesh.quad_connectivity.shape[1]!=4):return None
    key=_constraint_topology(mesh);groups=[];incidence={};offset=0
    for rows,active in ((mesh.triangle_connectivity,mesh.triangle_active),
                        (mesh.quad_connectivity,mesh.quad_active)):
        if rows.dtype!=np.dtype('int64') or active.dtype!=np.dtype('bool'):return None
        selected=rows[active]
        groups.append((selected.shape,selected.tobytes()))
        for index,cell in enumerate(selected,offset):
            for a,b in zip(cell,np.roll(cell,-1)):
                incidence.setdefault(tuple(sorted((int(a),int(b)))),[]).append(index)
        offset+=len(selected)
    pairs=np.asarray([owners for owners in incidence.values() if len(owners)==2],dtype=np.int64).reshape((-1,2))
    return key,tuple(groups),pairs.tobytes()


def _constraint_growth_vector(mesh, settings, layout):
    # Recheck the actual trial after evaluator and quality-kernel calls. Any
    # topology mutation or custom array kernel keeps the original dynamic path.
    if (layout is None or not _constraint_numpy_current() or layout[0]!=_constraint_topology(mesh)):
        return None
    if mesh.node_coordinates.dtype!=np.dtype("float64") or not np.isfinite(mesh.node_coordinates).all():
        return None
    lengths=[]
    for shape,data in layout[1]:
        cells=np.frombuffer(data,dtype=np.int64).reshape(shape)
        xyz=mesh.node_coordinates[cells]
        # Same per-row edge order, 3-coordinate norm reduction, then mean.
        lengths.append(np.linalg.norm(np.roll(xyz,-1,axis=1)-xyz,axis=2).mean(axis=1))
    lengths=np.concatenate(lengths)
    if not np.isfinite(lengths).all():return None
    pairs=np.frombuffer(layout[2],dtype=np.int64).reshape((-1,2))
    a,b=lengths[pairs[:,0]],lengths[pairs[:,1]]
    small=np.minimum(a,b);large=np.maximum(a,b)
    # Avoid evaluating division for zero lengths, exactly as the scalar route.
    ratios=np.full(len(pairs),float('inf'))
    positive=small>0;ratios[positive]=large[positive]/small[positive]
    return (settings.max_element_growth-ratios)/settings.max_element_growth


def _physical_constraint_values(mesh, quality, settings, *, _layout=None):
    """Actual physical inequalities >=0; no surrogate admission criterion."""
    values=[]
    for group in (quality.triangles,quality.quadrilaterals):
        values.extend(((group.minimum_angle-settings.min_angle)/max(1.,settings.min_angle),
            (settings.max_angle-group.maximum_angle)/max(1.,settings.max_angle),
            (settings.max_aspect_ratio-group.aspect_ratio)/max(1.,settings.max_aspect_ratio),
            group.scaled_jacobian-settings.min_scaled_jacobian,
            settings.max_warpage-group.warpage))
    fast=_constraint_growth_vector(mesh,settings,_layout)
    if fast is not None:
        values.append(fast)
        return np.concatenate(values)
    xyz=mesh.node_coordinates;lengths=[];incidence={}
    for rows,active in ((mesh.triangle_connectivity,mesh.triangle_active),
                        (mesh.quad_connectivity,mesh.quad_active)):
        for cell in rows[active]:
            index=len(lengths)
            lengths.append(float(np.linalg.norm(np.roll(xyz[cell],-1,axis=0)-xyz[cell],axis=1).mean()))
            for a,b in zip(cell,np.roll(cell,-1)):
                incidence.setdefault(tuple(sorted((int(a),int(b)))),[]).append(index)
    growth=[]
    for owners in incidence.values():
        if len(owners)==2:
            a,b=owners;small=min(lengths[a],lengths[b])
            ratio=max(lengths[a],lengths[b])/small if small>0 else float('inf')
            growth.append((settings.max_element_growth-ratio)/settings.max_element_growth)
    values.append(np.asarray(growth))
    return np.concatenate(values)


def repair_mixed_coordinates(points, triangles, report, settings, evaluate,
                             fixed_rows, *, cancellation_check=None, _trial_guard=None, _diagnostics=None):
    from dataclasses import replace
    made,used,truncated=repair_mixed_core_coordinates(points,triangles,report.mesh,settings,
        evaluate,fixed_rows,exchange_work=report.exchange_work,
        cancellation_check=cancellation_check,_trial_guard=_trial_guard,_diagnostics=_diagnostics)
    return made,replace(report,exchange_work=report.exchange_work+used,
                        exchange_truncated=report.exchange_truncated or truncated)


def repair_mixed_core_coordinates(points, triangles, mesh, settings, evaluate,
                                  fixed_rows, *, exchange_work=0,cancellation_check=None,
                                  _trial_guard=None,evaluate_jacobians=None,_diagnostics=None,_owned_context=None):
    """Numerical core worker; a MeshCore replay supplies no pairing authority."""
    from .core import MeshCore
    from .quality_v2 import evaluate_quality
    from .surface_mesh import _quality_threshold_report, _published_growth
    from .errors import MeshError

    # Only the module-owned immutable numerical guard is callable in the new
    # canonical shape phase. No caller closure gains fast-scope admission.
    owned_data=None;owned_guard=None
    def owned_check():
        if _owned_context is None:return
        from types import FunctionType
        from . import _physical_shape_patches as shape
        checker,code,kind,call,callcode=_owned_context[6:11]
        if not (checker is shape._CHECK_REF and shape._check_context is checker and type(checker) is FunctionType and checker.__code__ is code and checker.__defaults__ is None and checker.__kwdefaults__ is None):raise MeshError('owned shape checker changed before invocation')
        checker(_owned_context)
        if not (type(_trial_guard) is kind and kind is shape._OriginalStars and vars(kind).get('__call__') is call and type(call) is FunctionType and call.__code__ is callcode and call.__defaults__ is None and call.__kwdefaults__ is None):raise MeshError('owned original-star validator changed before invocation')
        if object.__getattribute__(_trial_guard,'_data') is not _owned_context[11]:raise MeshError('owned original-star operands changed')
    if _owned_context is not None:
        if type(_owned_context) is not tuple or len(_owned_context)!=12 or cancellation_check is not None:raise MeshError('invalid owned shape calculation context')
        owned_check();owned_data=object.__getattribute__(_trial_guard,'_data');owned_guard=_owned_context[9]
        data=owned_data
        if type(data) is not tuple or len(data)!=4:raise MeshError('original-star data must be an immutable four-tuple')
        dimensions=data[0]
        if (type(dimensions) is not tuple or len(dimensions)!=2 or type(dimensions[0]) is not int or type(dimensions[1]) is not int
                or dimensions[0]<0 or dimensions[1]!=2 or type(data[1]) is not bytes or len(data[1])!=16*dimensions[0]):raise MeshError('invalid original-star shape or bytes')
        if type(data[2]) is not tuple or any(type(node) is not int or node<0 or node>=dimensions[0] for node in data[2]):raise MeshError('invalid original-star frozen rows')
        if type(data[3]) is not tuple:raise MeshError('invalid original-star cells')
        for cell in data[3]:
            if type(cell) is not tuple or len(cell) not in (3,4) or any(type(node) is not int or node<0 or node>=dimensions[0] for node in cell):raise MeshError('invalid original-star cell data')
        # The owned payload is a numerical snapshot of THESE exact operands,
        # not a caller-set permission to inspect a different/fabricated star.
        if type(points) is not np.ndarray or points.dtype!=np.dtype('float64') or points.ndim!=2 or points.shape!=dimensions or data[1]!=points.tobytes():raise MeshError('owned original-star points differ')
        if type(fixed_rows) is not list and type(fixed_rows) is not tuple:raise MeshError('owned original-star fixed rows must be literal indices')
        if any(type(node) is not int or node<0 or node>=dimensions[0] for node in fixed_rows):raise MeshError('invalid owned original-star fixed input')
        if data[2]!=tuple(sorted(set(fixed_rows))):raise MeshError('owned original-star fixed set differs')
        if type(mesh) is not MeshCore:raise MeshError('owned original-star mesh must be canonical')
        for cells,active,width in ((mesh.triangle_connectivity,mesh.triangle_active,3),(mesh.quad_connectivity,mesh.quad_active,4)):
            if (type(cells) is not np.ndarray or cells.dtype!=np.dtype('int64') or cells.ndim!=2 or cells.shape[1]!=width
                    or type(active) is not np.ndarray or active.dtype!=np.dtype('bool') or active.shape!=(len(cells),)):
                raise MeshError('owned original-star mesh operands must be canonical numeric rows')
        movable=set(range(dimensions[0]))-set(fixed_rows)
        expected=tuple(tuple(map(int,cell)) for rows,active in ((mesh.triangle_connectivity,mesh.triangle_active),(mesh.quad_connectivity,mesh.quad_active)) for cell in rows[active] if set(map(int,cell))&movable)
        if data[3]!=expected:raise MeshError('owned original-star cells differ from actual mesh')

    def trial_allowed(proposed):
        if _owned_context is None:return _trial_guard(proposed)
        owned_check()
        return owned_guard(_trial_guard,proposed)

    if _diagnostics is not None and type(_diagnostics) is not dict:
        raise MeshError('mixed repair diagnostics require a plain dictionary')
    diagnostic=_diagnostics
    if diagnostic is not None:
        diagnostic.update(chart_refusals=0,caller_refusals=0,scored_trials=0,commits=0,
            validity_refusals=0,other_quality_violations=0,constraint_probes=0,
            patch_attempts=0,patch_projection_steps=0,patch_line_trials=0,best_growth=None,
            best_feasible_growth=None,best_trial=None,evaluation_packets=[])
    phase='initial';proposal_tag=None
    # Proposal resolution only; the maximum radius and total trial cap stay fixed.
    growth_fractions=tuple(2.**-k for k in range(13))
    original = np.asarray(points, dtype=float)
    current = original.copy()
    fixed = set(map(int, fixed_rows))
    if (original.ndim != 2 or original.shape != (mesh.num_nodes, 2)
            or not np.isfinite(original).all()
            or np.asarray(triangles).shape != (mesh.num_triangles, 3)
            or mesh.triangle_connectivity.shape[1] != 3
            or mesh.quad_connectivity.shape[1] != 4
            or not np.array_equal(triangles, mesh.triangle_connectivity)
            or any(row < 0 or row >= len(original) for row in fixed)):
        raise MeshError('mixed UV repair requires finite linear T3/Q4 rows and valid fixed nodes')
    if type(exchange_work) is not int or exchange_work<0:
        raise MeshError("mixed repair requires nonnegative consumed exchange work")
    used = 0
    limit = min(512, max(0, settings.max_recombination_work-exchange_work))
    incidence = {}
    neighbours = {}
    for row, cell in enumerate(triangles):
        for node in cell:
            incidence.setdefault(int(node), []).append(row)
            neighbours.setdefault(int(node), set()).update(map(int, cell))

    def check():
        owned_check()
        from ._stage_cancellation import check_stage
        check_stage(cancellation_check,'physical mixed-cell UV repair')

    def orientation(p, cell):
        a,b,c = [[Fraction(float(v)) for v in p[node]] for node in cell]
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])

    signs = [orientation(current, cell) > 0 for cell in triangles]
    if any(orientation(current, cell) == 0 for cell in triangles):
        raise MeshError('mixed UV repair requires nondegenerate source triangles')

    constraint_layout=None
    score_details={}
    def score_xyz(xyz, supplied_quality=None):
        owned_check()
        trial = MeshCore(xyz, mesh.triangle_connectivity, mesh.quad_connectivity,
            node_ids=mesh.node_ids, triangle_ids=mesh.triangle_ids, quad_ids=mesh.quad_ids,
            node_active=mesh.node_active, triangle_active=mesh.triangle_active,
            quad_active=mesh.quad_active, owner_table=mesh.owner_table,
            node_owner_handles=mesh.node_owner_handles,
            triangle_owner_handles=mesh.triangle_owner_handles,
            quad_owner_handles=mesh.quad_owner_handles)
        # An invalid detached physical Q4 is a scored refusal. Owner evaluation
        # and cancellation remain outside any exception-catching mechanism.
        quality = evaluate_quality(trial, check_validity=False) if supplied_quality is None else supplied_quality
        owned_check()
        threshold = _quality_threshold_report(quality, settings)
        growth, growth_poor = _published_growth(trial, settings.max_element_growth)
        counts = sum(threshold['violation_counts'].values())
        score_details['other_admitted']=not quality.validity.errors and not counts
        score_details['valid']=not quality.validity.errors
        penalty = 0.
        for group in (quality.triangles, quality.quadrilaterals):
            penalty += float(np.maximum(settings.min_angle-group.minimum_angle, 0.).sum())
            penalty += float(np.maximum(group.maximum_angle-settings.max_angle, 0.).sum())
            penalty += float(np.maximum(group.aspect_ratio-settings.max_aspect_ratio, 0.).sum())
            penalty += float(np.maximum(settings.min_scaled_jacobian-group.scaled_jacobian, 0.).sum())
            penalty += float(np.maximum(group.warpage-settings.max_warpage, 0.).sum())
        growth_excess = max(0., growth-settings.max_element_growth)
        if len(growth_poor) > 2:
            poor_lengths = {}; poor_edges = {}
            for rows, active, ids in ((mesh.triangle_connectivity, mesh.triangle_active, mesh.triangle_ids),
                                      (mesh.quad_connectivity, mesh.quad_active, mesh.quad_ids)):
                for cell, identifier in zip(rows[active], ids[active]):
                    item = int(identifier)
                    if item in growth_poor:
                        poor_lengths[item] = float(np.linalg.norm(np.roll(xyz[cell],-1,axis=0)-xyz[cell],axis=1).mean())
                        for a, b in zip(cell, np.roll(cell, -1)):
                            poor_edges.setdefault(tuple(sorted((int(a), int(b)))), []).append(item)
            growth_excess = sum(
                max(poor_lengths[a], poor_lengths[b])/min(poor_lengths[a], poor_lengths[b])-settings.max_element_growth
                for owners in poor_edges.values() if len(owners)==2
                for a, b in (owners,)
                if min(poor_lengths[a], poor_lengths[b])>0
                and max(poor_lengths[a], poor_lengths[b])/min(poor_lengths[a], poor_lengths[b])>settings.max_element_growth
            )
        key = (len(quality.validity.errors), counts+len(growth_poor),
               penalty+growth_excess)
        if diagnostic is not None:
            sample=dict(growth=float(growth),validity_errors=len(quality.validity.errors),
                violation_counts=dict(threshold['violation_counts']),growth_poor_count=len(growth_poor),
                growth_poor_ids=sorted(growth_poor)[:12],key=key,proposal=proposal_tag)
            if phase=='initial':diagnostic['initial']=sample
            elif phase=='final':diagnostic['final']=sample
            elif phase=='constraint':pass
            else:
                diagnostic['scored_trials']+=1
                diagnostic['validity_refusals']+=bool(quality.validity.errors)
                diagnostic['other_quality_violations']+=bool(counts)
                if diagnostic['best_growth'] is None or growth<diagnostic['best_growth']:
                    diagnostic['best_growth']=float(growth);diagnostic['best_growth_sample']=sample
                if not quality.validity.errors and not counts:
                    previous=diagnostic['best_feasible_growth']
                    if previous is None or growth<previous:diagnostic['best_feasible_growth']=float(growth)
                if diagnostic['best_trial'] is None or key<tuple(diagnostic['best_trial']['key']):
                    diagnostic['best_trial']=sample
        bad = set(threshold['poor_element_ids']) | growth_poor
        roots = set()
        for cells, active, ids in ((mesh.triangle_connectivity, mesh.triangle_active, mesh.triangle_ids),
                                  (mesh.quad_connectivity, mesh.quad_active, mesh.quad_ids)):
            for cell, identifier in zip(cells[active], ids[active]):
                if int(identifier) in bad:
                    roots.update(map(int, cell))
        return key, roots

    def owner_evaluate(uv, kind=None):
        if diagnostic is not None:
            if kind is None:
                kind=(phase if phase in ('initial','final') else
                      proposal_tag[0] if proposal_tag else 'unclassified')
            diagnostic['evaluation_packets'].append(dict(kind=kind,rows=len(uv),used=used,
                proposal=proposal_tag))
        owned_check()
        values=evaluate(uv)
        owned_check()
        return values

    def score(uv):
        check()
        xyz=owner_evaluate(uv)
        key,roots=score_xyz(xyz)
        return key,roots,xyz

    batch_chart = _batch_chart(evaluate, cancellation_check)
    def batch_evaluate(uv, kind=None):
        if batch_chart.check is not None:
            raise MeshError('batched mixed UV repair acquired an owner callback')
        xyz = owner_evaluate(uv,kind)
        if batch_chart.check is not None:
            raise MeshError('batched mixed UV repair acquired an owner callback')
        return xyz
    if evaluate_jacobians is None:
        from ._analytic_metric_chart import AnalyticMetricChart
        chart=getattr(evaluate,'__self__',None)
        if type(chart) is AnalyticMetricChart and getattr(evaluate,'__func__',None) is AnalyticMetricChart.evaluate:
            evaluate_jacobians=chart.jacobians
    if evaluate_jacobians is not None:
        constraint_layout=_prepare_constraint_layout(mesh)
    if batch_chart is not None:
        current_xyz=batch_evaluate(current);key,roots=score_xyz(current_xyz)
    else:key,roots,current_xyz=score(current)
    current_other_admitted=score_details['other_admitted']
    current_valid=score_details['valid']
    changed = False
    phase='trial'
    while roots and used < limit:
        best, best_key = None, key
        remaining=set(roots);components=[]
        while remaining:
            seed=min(remaining);comp={seed};queue=[seed];remaining.remove(seed)
            while queue:
                node=queue.pop()
                reach=neighbours[node]|set().union(*(neighbours[m] for m in neighbours[node]-fixed))
                linked=remaining&reach
                if linked:comp.update(linked);remaining-=linked;queue.extend(linked)
            components.append(comp)
        # Worst physical growth pair first. Jacobians only propose moves;
        # every trial keeps the complete original quality/admission scoring.
        descent={}
        if evaluate_jacobians is not None:
            gradient=_growth_descent(mesh,current_xyz,settings.max_element_growth)
            movable=sorted(set(gradient)-fixed)
            if movable:
                check()
                if batch_chart is not None and batch_chart.check is not None:
                    raise MeshError('batched mixed UV repair acquired an owner callback')
                jac=np.asarray(evaluate_jacobians(current[movable]),dtype=float)
                if batch_chart is not None and batch_chart.check is not None:
                    raise MeshError('batched mixed UV repair acquired an owner callback')
                if jac.shape!=(len(movable),3,2) or not np.isfinite(jac).all():
                    raise MeshError('mixed growth proposal requires finite (n, 3, 2) Jacobians')
                descent={node:-(gradient[node]@matrix) for node,matrix in zip(movable,jac)}
        # Constrained local patch before unconstrained directions. All probes
        # and line trials consume this same remaining budget, including refusals.
        if evaluate_jacobians is not None and current_valid:
            for comp in components:
                root_nodes=sorted(comp-fixed)
                ring=sorted(set().union(*(neighbours[n] for n in comp))-comp-fixed)
                patch=root_nodes[:24] if (not (comp&fixed) and len(root_nodes)>=4 and len(root_nodes+ring)>8) else (root_nodes+ring)[:24]
                if patch and limit-used>=1+4*len(patch)+1:
                    if diagnostic is not None:
                        diagnostic['patch_attempts']+=1;diagnostic['patch_nodes']=patch
                    radii=np.array([.2*float(np.min(np.linalg.norm(current[sorted(neighbours[n]-{n})]-current[n],axis=1)))
                        for n in patch])
                    if np.all(radii>0):
                        origin=current.copy();anchor=np.zeros(2*len(patch));h=2.**-12
                        affected=sorted(set().union(*(set(incidence[n]) for n in patch)))
                        def patch_probe(vector, label):
                            nonlocal used,phase,proposal_tag
                            if used>=limit:return None
                            used+=1;check();phase='constraint' if label=='derivative' else 'trial'
                            proposal_tag=('constrained_patch',label)
                            proposed=origin.copy();proposed[patch]+=vector.reshape(-1,2)*radii[:,None]
                            if any((orientation(proposed,triangles[row])>0)!=signs[row]
                                   or orientation(proposed,triangles[row])==0 for row in affected):
                                if diagnostic is not None:diagnostic['chart_refusals']+=1
                                return None
                            if _trial_guard is not None and not trial_allowed(proposed):
                                if diagnostic is not None:diagnostic['caller_refusals']+=1
                                return None
                            if diagnostic is not None:
                                diagnostic['constraint_probes' if label=='derivative' else 'patch_line_trials']+=1
                            xyz=batch_evaluate(proposed) if batch_chart is not None else owner_evaluate(proposed)
                            trial=MeshCore(xyz,mesh.triangle_connectivity,mesh.quad_connectivity,
                                node_ids=mesh.node_ids,triangle_ids=mesh.triangle_ids,quad_ids=mesh.quad_ids,
                                node_active=mesh.node_active,triangle_active=mesh.triangle_active,quad_active=mesh.quad_active)
                            q=evaluate_quality(trial,check_validity=False)
                            made_key,made_roots=score_xyz(xyz,q)
                            values=_physical_constraint_values(trial,q,settings,_layout=constraint_layout)
                            if q.validity.errors or not np.isfinite(values).all():return None
                            return values,proposed,made_key,made_roots,xyz,score_details['other_admitted'],score_details['valid']
                        for linearization in range(3):
                            if limit-used<1+2*len(anchor)+1:break
                            base=patch_probe(anchor,'derivative')
                            if base is None:break
                            columns=[]
                            for axis in range(len(anchor)):
                                plus=anchor.copy();minus=anchor.copy();plus[axis]+=h;minus[axis]-=h
                                # One-sided differentiation at the unchanged radius
                                # boundary; no out-of-radius actual probe is run.
                                p=patch_probe(plus,'derivative') if np.all(np.linalg.norm(plus.reshape(-1,2),axis=1)<=1.) else None
                                m=patch_probe(minus,'derivative') if np.all(np.linalg.norm(minus.reshape(-1,2),axis=1)<=1.) else None
                                if p is not None and m is not None:columns.append((p[0]-m[0])/(2*h))
                                elif p is not None:columns.append((p[0]-base[0])/h)
                                elif m is not None:columns.append((base[0]-m[0])/h)
                                else:break
                            if len(columns)!=len(anchor):break
                            matrix=np.column_stack(columns)
                            target,steps=_project_growth_halfspaces(base[0],matrix,anchor)
                            if diagnostic is not None:diagnostic['patch_projection_steps']+=steps
                            if target is None:
                                if diagnostic is not None:diagnostic['patch_constant_constraint_blocked']=True
                                break
                            if diagnostic is not None:
                                movable_rows=np.any(matrix!=0.,axis=1)
                                residual=-(base[0]+matrix@(target-anchor))
                                diagnostic['patch_linear_residual']=float(np.max(np.maximum(residual[movable_rows] if np.any(movable_rows) else residual,0.)))
                            next_anchor=None;next_violation=float(np.maximum(-base[0],0.).sum())
                            for fraction in (1.,.5,.25,.125):
                                trial=patch_probe(anchor+fraction*(target-anchor),'line')
                                if trial is None:continue
                                if trial[2]<best_key:
                                    best,best_key,best_roots,best_xyz,best_other,best_valid=trial[1],trial[2],trial[3],trial[4],trial[5],trial[6]
                                violation=float(np.maximum(-trial[0],0.).sum())
                                if violation<next_violation:
                                    next_anchor=anchor+fraction*(target-anchor);next_violation=violation
                                if best_key==(0,0,0.):break
                            if best is not None and best_other:break
                            if next_anchor is None:break
                            # Rejected anchors remain private optimizer operands;
                            # only the unchanged complete merit can commit a mesh.
                            anchor=next_anchor
                        phase='trial'
                        if best is not None:break
            if best is not None:
                current,key,roots,current_xyz=best,best_key,best_roots,best_xyz
                current_other_admitted=best_other;current_valid=best_valid;changed=True
                if diagnostic is not None:diagnostic['commits']+=1
                continue
        # A coupled proposal can reduce equal limiting ratios together. It is
        # charged as one complete trial, uses the same per-node radius and
        # fractions, and receives the same full chart/physical admission.
        for comp in components:
            comp_descent={n:descent[n] for n in comp if n in descent}
            magnitude=max((float(np.linalg.norm(v)) for v in comp_descent.values()),default=0.)
            if len(comp_descent)>1 and 0<magnitude<float('inf'):
                displacement=np.zeros_like(current)
                for node,direction in comp_descent.items():
                    adjacent=sorted(neighbours[node]-{node})
                    if adjacent:
                        radius=float(np.min(np.linalg.norm(current[adjacent]-current[node],axis=1)))*.2
                        displacement[node]=radius*direction/magnitude
                affected=sorted(set().union(*(set(incidence[n]) for n in comp_descent)))
                for fraction in growth_fractions:
                    if used>=limit:break
                    used+=1;check();proposal_tag=('coupled_growth',fraction);proposed=current.copy()
                    moving=sorted(comp_descent)
                    proposed[moving]+=fraction*displacement[moving]
                    if any((orientation(proposed,triangles[row])>0)!=signs[row]
                           or orientation(proposed,triangles[row])==0 for row in affected):
                        if diagnostic is not None:diagnostic['chart_refusals']+=1
                        continue
                    if _trial_guard is not None and not trial_allowed(proposed):
                        if diagnostic is not None:diagnostic['caller_refusals']+=1
                        continue
                    proposed_key,proposed_roots,proposed_xyz=score(proposed)
                    if proposed_key<best_key:
                        best,best_key,best_roots,best_xyz=proposed,proposed_key,proposed_roots,proposed_xyz
                        best_other=score_details['other_admitted'];best_valid=score_details['valid']
                    if best_key==(0,0,0.):break
                if best is not None:break
        if best is not None:
            current,key,roots,current_xyz=best,best_key,best_roots,best_xyz
            current_other_admitted=best_other;current_valid=best_valid
            changed=True
            if diagnostic is not None:diagnostic['commits']+=1
            continue
        # Root cells first; one-ring nodes follow, with stable row ordering.
        root_only = sorted(roots-fixed)
        ring_only = sorted(set().union(*(neighbours[n] for n in roots))-roots-fixed)
        priority=sorted(descent,key=lambda n:(-float(np.linalg.norm(descent[n])),n))
        nodes=[n for n in root_only if n not in descent]+priority+[n for n in ring_only if n not in descent]
        for node in nodes:
            check()
            adjacent = sorted(neighbours[node]-{node})
            if not adjacent:
                continue
            delta = current[adjacent]-current[node]
            radius = float(np.min(np.linalg.norm(delta, axis=1)))*.2
            directions=[];growth_direction=False
            if node in descent:
                norm=float(np.linalg.norm(descent[node]))
                if norm>0 and np.isfinite(norm):
                    directions.append(radius*descent[node]/norm);growth_direction=True
            directions += [delta.mean(axis=0), -delta.mean(axis=0)]
            directions += [radius*np.array((a,b))/max(1.,np.hypot(a,b))
                           for a,b in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))]
            candidates = [] if batch_chart is not None else None
            for direction_index,direction in enumerate(directions):
                fractions=growth_fractions if growth_direction and direction_index==0 else (1.,.5,.25)
                for fraction in fractions:
                    if used >= limit:
                        break
                    used += 1
                    check()
                    proposal_tag=('node',node,direction_index,fraction)
                    proposed = current.copy()
                    proposed[node] += direction*fraction
                    if any((orientation(proposed, triangles[row]) > 0) != signs[row]
                           or orientation(proposed, triangles[row]) == 0 for row in incidence[node]):
                        if diagnostic is not None:diagnostic['chart_refusals']+=1
                        continue
                    # A connected caller may require stricter mixed-cell
                    # orientation than the unchanged T3 carrier. Refused UV
                    # proposals remain charged and never reach owner scoring.
                    if _trial_guard is not None and not trial_allowed(proposed):
                        if diagnostic is not None:diagnostic['caller_refusals']+=1
                        continue
                    if candidates is not None:
                        candidates.append(proposed[node].copy())
                        continue
                    proposed_key, proposed_roots, proposed_xyz = score(proposed)
                    if proposed_key < best_key:
                        best, best_key, best_roots = proposed, proposed_key, proposed_roots
                        best_xyz=proposed_xyz
                        best_other=score_details['other_admitted'];best_valid=score_details['valid']
                    if evaluate_jacobians is not None and best_key==(0,0,0.):break
                if used >= limit or (evaluate_jacobians is not None and (best_key==(0,0,0.) or (growth_direction and direction_index==0 and best is not None))):
                    break
            if candidates:
                # Only this node's candidates are evaluated. Scratch storage
                # is O(nodes + candidates), never candidate full point clouds.
                candidate_xyz = batch_evaluate(np.asarray(candidates, dtype=float),"node_batch")
                for point, xyz in zip(candidates, candidate_xyz):
                    proposal_tag=('node_batch',node)
                    proposed_xyz = current_xyz.copy()
                    proposed_xyz[node] = xyz
                    proposed_key, proposed_roots = score_xyz(proposed_xyz)
                    if proposed_key < best_key:
                        best = current.copy()
                        best[node] = point
                        best_key, best_roots = proposed_key, proposed_roots
                        best_xyz = proposed_xyz
                        best_other=score_details['other_admitted'];best_valid=score_details['valid']
            if used >= limit:
                break
            if best is not None:
                # Commit a complete improvement before searching another node;
                # otherwise the first pass could spend every trial before a
                # second independently defective cell receives its repair.
                break
        if best is None:
            break
        current, key, roots = best, best_key, best_roots
        current_xyz = best_xyz
        current_other_admitted=best_other;current_valid=best_valid
        changed = True
        if diagnostic is not None:diagnostic['commits']+=1
    check()
    phase='final'
    if batch_chart is not None or (changed and evaluate_jacobians is not None):
        # Final complete owner evaluation is required for all committed moves.
        final_key, final_roots = score_xyz(batch_evaluate(current) if batch_chart is not None else owner_evaluate(current))
        if final_key != key or final_roots != roots:
            raise MeshError('mixed UV repair differs from final full owner evaluation')
    # The original report and caller points are immutable throughout trials.
    if diagnostic is not None:diagnostic.update(trials=used,limit=limit,remaining_bad_nodes=len(roots),final_key=key)
    return current if changed else original,used,bool(roots and used>=limit)
