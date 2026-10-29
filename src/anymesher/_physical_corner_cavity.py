"""Detached, owner-evaluated relocation/two-flip proposals at fixed corners.

No node is removed or renumbered. Every relocation preserves the ORIGINAL
oriented mixed stars; native flips are used only after that check. Complete
proposals alone are scored against the unchanged full mixed quality policy.
"""
from dataclasses import replace
import numpy as np

from .errors import MeshError
from .native_v2 import MutableT3Topology, orient2d


def _edges(cell):
    return tuple(tuple(sorted((int(a),int(b)))) for a,b in zip(cell,np.roll(cell,-1)))


def _positive(points,cell):
    # Never canonicalize an inverted relocation into a positive triangle.
    if len(cell) == 3:
        return orient2d(*(points[int(node)] for node in cell)) > 0
    return all(orient2d(points[int(cell[i-1])],points[int(cell[i])],points[int(cell[(i+1)%4])]) > 0
               for i in range(4))


def _valid_star(points,mesh,moved):
    for cells,active in ((mesh.triangle_connectivity,mesh.triangle_active),
                         (mesh.quad_connectivity,mesh.quad_active)):
        for cell in cells[active]:
            if any(int(node) in moved for node in cell) and not _positive(points,cell):
                return False
    return True


def _closed_mixed_link(mesh,node):
    """One oriented disk star, including every incident Q4 interface."""
    links = {}
    incoming = {}
    for cells,active in ((mesh.triangle_connectivity,mesh.triangle_active),
                         (mesh.quad_connectivity,mesh.quad_active)):
        for cell in cells[active]:
            values = list(map(int,cell))
            if node not in values:
                continue
            if values.count(node) != 1:
                return False
            i = values.index(node)
            path = (values+values)[i+1:i+len(values)]
            for a,b in zip(path,path[1:]):
                if a in links or b in incoming:
                    return False
                links[a],incoming[b] = b,a
    if not links or set(links) != set(incoming):
        return False
    start = next(iter(links)); seen = set(); current = start
    while current not in seen:
        seen.add(current); current = links[current]
    return current == start and seen == set(links)


def _fans(mesh,fixed,protected,poor):
    active = tuple(map(int,np.flatnonzero(mesh.triangle_active)))
    for corner in sorted(fixed):
        rows = tuple(row for row in active if corner in mesh.triangle_connectivity[row])
        if (len(rows) != 3 or not any(int(mesh.triangle_ids[row]) in poor for row in rows)
                or any(corner in quad for quad in mesh.quad_connectivity[mesh.quad_active])):
            continue
        if len({int(mesh.triangle_owner_handles[row]) for row in rows}) != 1:
            continue
        link = {}
        counts = {}
        for row in rows:
            cell = mesh.triangle_connectivity[row]
            ends = [int(node) for node in cell if int(node) != corner]
            for a,b in (ends,ends[::-1]): link.setdefault(a,set()).add(b)
            for edge in _edges(cell): counts[edge] = counts.get(edge,0)+1
        endpoints = sorted(node for node,near in link.items() if len(near) == 1)
        if len(link) != 4 or len(endpoints) != 2 or any(len(near) not in (1,2) for near in link.values()):
            continue
        a,b = endpoints
        if not {a,b} <= fixed or not {(min(corner,a),max(corner,a)),(min(corner,b),max(corner,b))} <= protected:
            continue
        p = next(iter(link[a])); q = next(iter(link[b]))
        if p == q or q not in link[p] or p in fixed or q in fixed:
            continue
        if not _closed_mixed_link(mesh,p) or not _closed_mixed_link(mesh,q):
            continue
        perimeter = tuple(edge for edge,count in counts.items() if count == 1)
        if len(perimeter) == 5:
            yield corner,a,p,q,b,rows,perimeter


def _ray_points(points,mesh,node,corner,a,b):
    """Numerical proposals only; exact original-star signs decide eligibility."""
    origin = points[node]; direction = origin-points[corner]
    if not np.isfinite(direction).all() or not np.any(direction): return (origin.copy(),)
    cap = 2.
    for cells,active in ((mesh.triangle_connectivity,mesh.triangle_active),
                         (mesh.quad_connectivity,mesh.quad_active)):
        for cell in cells[active]:
            if node not in cell: continue
            triples = (tuple(cell),) if len(cell) == 3 else tuple((cell[i-1],cell[i],cell[(i+1)%4]) for i in range(4))
            for triple in triples:
                if node not in triple: continue
                before = float(orient2d(*(points[int(n)] for n in triple)))
                after_points = [origin+direction if int(n) == node else points[int(n)] for n in triple]
                slope = float(orient2d(*after_points))-before
                if slope < 0: cap = min(cap,-before/slope)
    fractions = [0.,cap/4,cap/2,3*cap/4]
    before = float(orient2d(points[a],points[b],origin))
    slope = float(orient2d(points[a],points[b],origin+direction))-before
    if slope:
        crossing = -before/slope
        if 0 < crossing < cap:
            fractions = [0.,crossing+(cap-crossing)/4,crossing+(cap-crossing)/2,crossing+3*(cap-crossing)/4]
    return tuple(origin+value*direction for value in fractions if np.isfinite(value) and value >= 0)


def _core(old,xyz,triangles):
    from .core import MeshCore
    return MeshCore(xyz,triangles,old.quad_connectivity,node_ids=old.node_ids,
        triangle_ids=old.triangle_ids,quad_ids=old.quad_ids,node_active=old.node_active,
        triangle_active=old.triangle_active,quad_active=old.quad_active,owner_table=old.owner_table,
        node_owner_handles=old.node_owner_handles,triangle_owner_handles=old.triangle_owner_handles,
        quad_owner_handles=old.quad_owner_handles)


def _score(core,settings):
    from .quality_v2 import evaluate_quality
    from .surface_mesh import _quality_threshold_report,_published_quality_report
    quality = evaluate_quality(core,check_validity=False)
    threshold = _quality_threshold_report(quality,settings)
    report = _published_quality_report(core,settings,quality,threshold)
    report['violation_counts'] = dict(threshold['violation_counts'])
    report['invalid_element_count'] = len(quality.validity.errors)
    return report


def repair_corner_cavities(points,report,settings,evaluate,fixed_rows,protected_edges,*,
                           topology_remaining,cancellation_check=None):
    """Bounded complete proposals; return UV, mixed report, charged flips, facts.

    Rejection and exceptions never mutate input topology, seed registries or
    reports. Successful cores carry matching fresh physical XYZ and all original
    node/element identities. No partial relocation or first flip is published.
    """
    from ._physical_mixed_repair import _batch_chart
    from ._physical_t3_refinement import _alternative_progress
    chart = _batch_chart(evaluate,cancellation_check)
    facts = dict(enabled=chart is not None,relocation_trials=0,flip_attempts=0,
                 complete_proposals=0,commits=0,polish_work=0,polish_accepted=False,corners=[])
    if chart is None: return points,report,0,facts
    if type(topology_remaining) is not int or topology_remaining < 0:
        raise MeshError('corner cavity requires an exact remaining topology allowance')
    original = np.asarray(points,dtype=float)
    current = original.copy(); old = report.mesh
    if current.shape != (old.num_nodes,2) or not np.isfinite(current).all():
        raise MeshError('corner cavity requires finite unchanged UV row identities')
    fixed = set(map(int,fixed_rows))
    protected = {tuple(sorted(map(int,edge))) for edge in protected_edges}
    fixed.update(node for edge in protected for node in edge)
    if any(row < 0 or row >= len(current) for row in fixed):
        raise MeshError('corner cavity fixed row is invalid')
    fixed_order = sorted(fixed)
    fixed_uv = current[fixed_order].copy()
    fixed_xyz = old.node_coordinates[fixed_order].copy()
    relocation_limit = min(128,max(0,settings.max_recombination_work-report.exchange_work))
    flip_limit = min(256,topology_remaining)
    def relocation_exhausted():
        return (facts['relocation_trials']>=relocation_limit or
            report.exchange_work+facts['relocation_trials']+facts['polish_work']>=settings.max_recombination_work)

    def checked(rows):
        if chart.check is not None: raise MeshError('corner cavity acquired an owner callback')
        values = np.asarray(evaluate(rows),dtype=float)
        if chart.check is not None: raise MeshError('corner cavity acquired an owner callback')
        return values

    xyz = checked(current)
    xyz[fixed_order] = fixed_xyz
    baseline = _core(old,xyz,old.triangle_connectivity)
    before = _score(baseline,settings)
    facts['initial_quality']=before
    def polish(uv,mesh,work):
        from ._physical_mixed_repair import repair_mixed_coordinates
        stage=replace(report,mesh=mesh,exchange_work=work)
        link_cache={}
        def valid_trial(proposed):
            moved=set(map(int,np.flatnonzero(np.any(proposed!=uv,axis=1))))
            for node in moved:
                if node not in link_cache: link_cache[node]=_closed_mixed_link(mesh,node)
            return (not moved.intersection(fixed) and all(link_cache[node] for node in moved)
                and _valid_star(proposed,mesh,moved))
        proposed,receipt=repair_mixed_coordinates(uv,mesh.triangle_connectivity,stage,settings,
            evaluate,fixed_order,cancellation_check=cancellation_check,_trial_guard=valid_trial)
        facts['polish_work']+=receipt.exchange_work-work
        if not np.array_equal(proposed[fixed_order],fixed_uv):
            raise MeshError('corner cavity polish changed a fixed source row')
        if not valid_trial(proposed):
            raise MeshError('corner cavity polish lost its strict mixed star')
        fresh=checked(proposed); fresh[fixed_order]=fixed_xyz
        candidate=_core(mesh,fresh,mesh.triangle_connectivity)
        return proposed,candidate,_score(candidate,settings)

    committed = False
    for _pass in range(4):
        if (not before['poor_element_ids'] or relocation_exhausted()
                or facts['flip_attempts'] >= flip_limit): break
        accepted = False
        for corner,a,p,q,b,rows,perimeter in _fans(baseline,fixed,protected,set(before['poor_element_ids'])):
            if not _valid_star(current,baseline,{p,q}): continue
            diagnostic=dict(corner=int(corner),nodes=[int(p),int(q)],
                cell_ids=[int(baseline.triangle_ids[row]) for row in rows],tested=0,refusals={},best=None)
            if len(facts['corners'])<8: facts['corners'].append(diagnostic)
            best_diagnostic_key=None
            polish_seeds=[]
            proposals = []
            for first,second in ((p,q),(q,p)):
                for first_point in _ray_points(current,baseline,first,corner,a,b):
                    if relocation_exhausted(): break
                    facts['relocation_trials'] += 1
                    interim = current.copy(); interim[first] = first_point
                    if not _valid_star(interim,baseline,{first}): continue
                    for second_point in _ray_points(interim,baseline,second,corner,a,b):
                        if relocation_exhausted(): break
                        # Charge each sequential relocation before validation
                        # or owner evaluation, including a refused old star.
                        facts['relocation_trials'] += 1
                        proposed = interim.copy(); proposed[second] = second_point
                        if not _valid_star(proposed,baseline,{p,q}): continue
                        for sequence in (((corner,p),(corner,q)),((corner,q),(corner,p))):
                            trial = MutableT3Topology(proposed,baseline.triangle_connectivity[list(rows)],perimeter)
                            complete = True
                            for edge in sequence:
                                if facts['flip_attempts'] >= flip_limit:
                                    complete = False; break
                                facts['flip_attempts'] += 1
                                if not trial.flip_edge(tuple(sorted(edge))):
                                    complete = False; break
                            if complete:
                                proposals.append((proposed[[p,q]].copy(),trial.triangles.copy()))
                            if facts['flip_attempts'] >= flip_limit: break
                        if facts['flip_attempts'] >= flip_limit: break
                    if relocation_exhausted() or facts['flip_attempts'] >= flip_limit: break
                if relocation_exhausted() or facts['flip_attempts'] >= flip_limit: break
            if not proposals: continue
            # One pointwise owner batch for THIS connected corner only.
            values = checked(np.vstack([positions for positions,_ in proposals]))
            for index,(positions,triangles) in enumerate(proposals):
                proposed = current.copy(); proposed[[p,q]] = positions
                trial_xyz = baseline.node_coordinates.copy(); trial_xyz[[p,q]] = values[2*index:2*index+2]
                cells = baseline.triangle_connectivity.copy(); cells[list(rows)] = triangles
                candidate = _core(baseline,trial_xyz,cells)
                summary = _score(candidate,settings)
                facts['complete_proposals'] += 1
                diagnostic['tested']+=1
                diagnostic_key=(summary['invalid_element_count'],sum(summary['violation_counts'].values()),
                    summary['elements_above_maximum_growth'],summary['max_element_growth'])
                if best_diagnostic_key is None or diagnostic_key<best_diagnostic_key:
                    best_diagnostic_key=diagnostic_key
                    diagnostic['best']=dict(quality=summary,positions=positions.tolist(),triangles=triangles.tolist())
                progress=_alternative_progress(before,summary)
                if not progress:
                    reasons=[]
                    if summary['invalid_element_count']: reasons.append('invalid')
                    reasons.extend(name for name,count in before['violation_counts'].items()
                        if summary['violation_counts'][name]>count)
                    if summary['quality_violation_count']>before['quality_violation_count']: reasons.append('poor_count')
                    if summary['elements_above_maximum_growth']>before['elements_above_maximum_growth']: reasons.append('growth_count')
                    if not reasons: reasons.append('no_strict_progress')
                    for reason in reasons:
                        diagnostic['refusals'][reason]=diagnostic['refusals'].get(reason,0)+1
                    if not summary['invalid_element_count'] and not any(summary['violation_counts'].values()):
                        # Private literal proposals only, never diagnostic facts.
                        # Keep at most TWO shape-cleared complete cavities.
                        seed_key=(summary['elements_above_maximum_growth'],summary['max_element_growth'],index)
                        canonical=[]
                        for cell in triangles:
                            row=tuple(map(int,cell))
                            canonical.append(min(row[i:]+row[:i] for i in range(3)))
                        identity=(positions.tobytes(),tuple(sorted(canonical)),
                            int(baseline.triangle_owner_handles[rows[0]]))
                        if not any(seed[3]==identity for seed in polish_seeds):
                            polish_seeds.append((seed_key,positions.copy(),triangles.copy(),identity))
                            polish_seeds.sort(key=lambda seed:seed[0])
                            del polish_seeds[2:]
                    continue
                # Fresh FULL owner values, fixed literal targets, and the
                # original global gate are the sole adoption frontier.
                fresh = checked(proposed); fresh[fixed_order] = fixed_xyz
                candidate = _core(baseline,fresh,cells)
                summary = _score(candidate,settings)
                if not _alternative_progress(before,summary): continue
                if not np.array_equal(proposed[fixed_order],fixed_uv):
                    raise MeshError('corner cavity changed a fixed source row')
                current,baseline,before = proposed,candidate,summary
                committed = accepted = True; facts['commits'] += 1
                break
            if not accepted:
                selected=None; selected_rank=None
                for seed_index,(_key,positions,triangles,_identity) in enumerate(polish_seeds):
                    diagnostic_seed=dict(corner=int(corner),seed=seed_index,initial=None,final=None,
                        work=0,status='not_started')
                    facts.setdefault('compound_seeds',[]).append(diagnostic_seed)
                    work=report.exchange_work+facts['relocation_trials']+facts['polish_work']
                    if work>=settings.max_recombination_work:
                        diagnostic_seed['status']='budget_exhausted'; continue
                    proposed=current.copy(); proposed[[p,q]]=positions
                    if not _valid_star(proposed,baseline,{p,q}):
                        diagnostic_seed['status']='original_star_refused'; continue
                    cells=baseline.triangle_connectivity.copy(); cells[list(rows)]=triangles
                    fresh=checked(proposed); fresh[fixed_order]=fixed_xyz
                    candidate=_core(baseline,fresh,cells)
                    summary=_score(candidate,settings)
                    diagnostic_seed['initial']=summary
                    if summary['invalid_element_count'] or any(summary['violation_counts'].values()):
                        diagnostic_seed['status']='fresh_shape_refused'; continue
                    facts['compound_polish_attempts']=facts.get('compound_polish_attempts',0)+1
                    work_before=facts['polish_work']
                    proposed,candidate,summary=polish(proposed,candidate,work)
                    diagnostic_seed.update(final=summary,work=facts['polish_work']-work_before,
                        status='progress' if _alternative_progress(before,summary) else 'refused')
                    if not _alternative_progress(before,summary): continue
                    full=(not summary['invalid_element_count'] and not summary['poor_element_ids'])
                    rank=(not full,summary['quality_violation_count'],summary['max_element_growth'],seed_index)
                    if selected_rank is None or rank<selected_rank:
                        selected_rank=rank; selected=(proposed,candidate,summary,seed_index)
                if selected is not None:
                    current,baseline,before,seed_index=selected
                    committed=accepted=True; facts['commits']+=1
                    facts['compound_polish_accepted']=True
                    facts['compound_selected_seed']=seed_index
            if accepted: break
        if not accepted: break
    # Existing coordinate repair ranks proposals only. Its complete result is
    # detached until original mixed orientation, fixed rows, fresh owner XYZ
    # and the SAME post-cavity progress predicate all agree.
    charged_work=report.exchange_work+facts['relocation_trials']+facts['polish_work']
    if committed and before['poor_element_ids'] and charged_work<settings.max_recombination_work:
        polished,candidate,summary=polish(current,baseline,charged_work)
        if _alternative_progress(before,summary):
            current,baseline,before=polished,candidate,summary
            facts['polish_accepted']=True
    charged_work=report.exchange_work+facts['relocation_trials']+facts['polish_work']
    chart.validate_reference()
    if chart.check is not None: raise MeshError('corner cavity acquired an owner callback')
    facts['final_quality'] = before
    facts['relocation_stage_capped'] = bool(before['poor_element_ids']
        and facts['relocation_trials']>=relocation_limit)
    result = replace(report,exchange_work=charged_work,
        exchange_truncated=report.exchange_truncated or bool(before['poor_element_ids']
            and charged_work >= settings.max_recombination_work),
        mesh=baseline if committed else old)
    return current if committed else points,result,facts['flip_attempts'],facts
