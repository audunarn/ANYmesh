"""Local, fixed-interface Delaunay refinement for the hidden prototype."""
from math import fsum, hypot, isfinite

import numpy as np

from .errors import MeshError
from .triangulation import orient2d, triangulate_polygon, _point_on_segment
from .quad.quality_gate import QUAD_FIRST_T3_MIN_ANGLE, corner_metrics
from ._smart_mapped_improve import improve, _metric, _summary


def _edges(body):
    return tuple(zip(body,body[1:]+body[:1]))


def _score(points,triangles):
    deficits = [max(0.,QUAD_FIRST_T3_MIN_ANGLE-corner_metrics([points[n] for n in t])[0])
                for t in triangles]
    return max(deficits,default=0.),sum(v > 0 for v in deficits),fsum(deficits)


def _circumcenter(polygon):
    a,b,c = polygon
    bx,by = b[0]-a[0],b[1]-a[1]
    cx,cy = c[0]-a[0],c[1]-a[1]
    denominator = 2*(bx*cy-by*cx)
    if not isfinite(denominator) or denominator == 0.:
        return None
    bb,cc = bx*bx+by*by,cx*cx+cy*cy
    point = (a[0]+(cy*bb-by*cc)/denominator,a[1]+(bx*cc-cx*bb)/denominator)
    return point if all(isfinite(v) for v in point) else None


def refine(points,quads,triangles,*,protected_edges,enabled,budget,work,cancellation_check):
    from ._smart_mapped_prototype import _area2
    report = dict(enabled=enabled,budget=budget,cdt_calls=0,added_nodes=0,
                  accepted_cavities=0,declined_cavities=0,released_quad_indices=[],
                  restored_quad_indices=[],cavities=[],budget_exhausted=False,
                  expanded_cell_indices=[],
                  before=_summary(points,quads,triangles))
    if not enabled or not budget:
        report['after'] = report['before']
        return points,quads,triangles,report
    if cancellation_check is not None:
        cancellation_check('smart-mapped:local-refinement')
    if report['before']['triangles_below_minimum'] == 0:
        report['after'] = report['before']
        return points,quads,triangles,report
    cells = [*quads,*triangles]
    edge_cells = {}
    for row,body in enumerate(cells):
        work.step('smart-mapped:refinement-cavities')
        for edge in _edges(body):
            edge_cells.setdefault(tuple(sorted(edge)),[]).append(row)
    bad = {len(quads)+i for i,t in enumerate(triangles)
           if corner_metrics([points[n] for n in t])[0] < QUAD_FIRST_T3_MIN_ANGLE}
    regular_rows = {row for row,body in enumerate(quads)
                    if all((points[b][0] == points[a][0]) != (points[b][1] == points[a][1])
                           for a,b in _edges(body))}
    selected = set(bad)
    for row in sorted(bad):
        for edge in _edges(cells[row]):
            if tuple(sorted(edge)) not in protected_edges:
                selected.update(n for n in edge_cells[tuple(sorted(edge))] if n not in regular_rows)
    # Grow through artificial, encroached interfaces only. Geometry constraints
    # and every original axis-aligned rectangle remain impermeable barriers.
    # Merge in the single selected set before connected components are formed.
    centers = [p for row in sorted(bad)
               if (p := _circumcenter([points[n] for n in cells[row]])) is not None]
    initial_selected = set(selected)
    while selected and len(selected) < 128:
        additions = set()
        for edge,owners in sorted(edge_cells.items()):
            work.step('smart-mapped:refinement-expansion')
            if edge in protected_edges or not any(n in selected for n in owners):
                continue
            outside = [n for n in owners if n not in selected and n not in regular_rows]
            if not outside:
                continue
            a,b = (points[n] for n in edge)
            for p in centers:
                work.step('smart-mapped:refinement-expansion')
                if (p[0]-a[0])*(p[0]-b[0])+(p[1]-a[1])*(p[1]-b[1]) < 0.:
                    additions.update(outside)
                    break
        additions = set(sorted(additions)[:128-len(selected)])
        if not additions:
            break
        selected.update(additions)
    report['expanded_cell_indices'] = sorted(selected-initial_selected)
    components = []
    unseen = set(selected)
    for seed in sorted(selected):
        if seed not in unseen:
            continue
        todo = [seed]; component = set()
        while todo:
            work.step('smart-mapped:refinement-cavities')
            row = todo.pop()
            if row not in unseen:
                continue
            unseen.remove(row); component.add(row)
            for edge in _edges(cells[row]):
                todo.extend(n for n in edge_cells[tuple(sorted(edge))] if n in unseen)
        components.append(sorted(component))
    output_points = [tuple(map(float,p)) for p in points]
    output_quads = []; output_triangles = []; removed = set()
    for component in components:
        if report['cdt_calls'] >= budget or report['added_nodes'] >= 128:
            break
        work.step('smart-mapped:local-refinement')
        detail = dict(cells=len(component),attempts=[],reason='no_improvement')
        if len(report['cavities']) < 64:
            report['cavities'].append(detail)
        if len(component) > 128:
            detail['reason'] = 'cell_limit'
            report['declined_cavities'] += 1
            continue
        counts = {}; directed = {}
        for row in component:
            work.step('smart-mapped:refinement-cavities')
            for edge in _edges(cells[row]):
                key = tuple(sorted(edge)); counts[key] = counts.get(key,0)+1
                directed[key] = edge
        boundary = [directed[e] for e,c in counts.items() if c == 1]
        following = dict(boundary)
        if len(following) != len(boundary) or len(set(following.values())) != len(boundary):
            detail['reason'] = 'nonmanifold_boundary'
            report['declined_cavities'] += 1
            continue
        remaining = set(following); loops = []
        while remaining:
            start = min(remaining); loop = []; node = start
            while node in remaining:
                remaining.remove(node); loop.append(node); node = following[node]
            if node != start:
                loops = []; break
            loops.append(tuple(loop))
        areas = [_area2(tuple(output_points[n] for n in loop)) for loop in loops]
        outers = [loop for loop,a in zip(loops,areas) if a > 0]
        if len(outers) != 1 or any(a == 0 for a in areas):
            detail['reason'] = 'unsupported_boundary_loops'
            report['declined_cavities'] += 1
            continue
        outer = tuple(output_points[n] for n in outers[0])
        holes = tuple(tuple(output_points[n] for n in loop) for loop,a in zip(loops,areas) if a < 0)
        fixed_edges = tuple(tuple(output_points[n] for n in edge) for edge in sorted(counts)
                            if edge in protected_edges and counts[edge] == 2)
        original_ids = sorted({n for row in component for n in cells[row]})
        boundary_ids = {n for edge in boundary for n in edge}
        extras = [output_points[n] for n in original_ids if n not in boundary_ids]
        all_points = set(output_points[n] for n in original_ids)
        barriers = [tuple(output_points[n] for n in edge) for edge in boundary]+list(fixed_edges)
        epsilon = max(max(p[a] for p in outer)-min(p[a] for p in outer) for a in (0,1))
        epsilon = max(1.,epsilon)*1.e-12
        current_triangles = [cells[row] for row in component if row >= len(quads)]
        current_points = output_points
        before = _score(points,current_triangles)
        detail['before_score'] = before
        best = None; new_points = []
        # The first attempt reconnects, the remaining bounded attempts resolve
        # progressively finer transition layers under the shared CDT budget.
        for iteration in range(min(8,budget-report['cdt_calls'])):
            if report['cdt_calls'] >= budget:
                break
            proposals = []
            for body in (() if iteration == 0 else current_triangles):
                work.step('smart-mapped:refinement-stations')
                polygon = [current_points[n] for n in body]
                if corner_metrics(polygon)[0] >= QUAD_FIRST_T3_MIN_ANGLE:
                    continue
                p = _circumcenter(polygon)
                if p is None:
                    continue
                separation = min(hypot(v[0]-u[0],v[1]-u[1]) for u,v in _edges(tuple(polygon)))/2
                rejection = None
                if not all(isfinite(v) for v in p) or p in all_points:
                    rejection = dict(reason='nonfinite_or_existing')
                for q in sorted(all_points):
                    work.step('smart-mapped:refinement-stations')
                    if rejection is not None:
                        break
                    if hypot(p[0]-q[0],p[1]-q[1]) < separation:
                        rejection = dict(reason='station_separation',station=q)
                for c,d in barriers:
                    work.step('smart-mapped:refinement-stations')
                    if rejection is not None:
                        break
                    if (_point_on_segment(p,c,d,epsilon)
                            or (p[0]-c[0])*(p[0]-d[0])+(p[1]-c[1])*(p[1]-d[1]) < 0.):
                        rejection = dict(reason='protected_edge_encroachment',edge=(c,d))
                # A point must lie strictly inside some original cavity cell.
                inside = False
                if rejection is None:
                    for row in component:
                        work.step('smart-mapped:refinement-stations')
                        if all(orient2d(output_points[a],output_points[b],p) > 0.
                               for a,b in _edges(cells[row])):
                            inside = True; break
                    if not inside:
                        rejection = dict(reason='outside_cavity')
                if rejection is not None:
                    refusals = detail.setdefault('proposal_refusals',[])
                    if len(refusals) < 8:
                        refusals.append(dict(point=tuple(map(float,p)),
                            triangle=[tuple(map(float,q)) for q in polygon],
                            separation=float(separation),**rejection))
                    continue
                if len(new_points)+len(proposals)+report['added_nodes'] >= 128:
                    break
                proposals.append(p); all_points.add(p)
            if not proposals and iteration:
                detail['reason'] = 'no_admissible_circumcenter'
                break
            new_points.extend(proposals)
            extra = (*extras,*new_points)
            raw = (*outer,*(p for hole in holes for p in hole),*extra,
                   *(p for edge in fixed_edges for p in edge))
            canonical = {}; protected = {}
            for row,p in enumerate(raw):
                protected[row] = canonical.setdefault(p,len(canonical))
            cancelled = []
            def check(stage):
                if cancellation_check is not None:
                    try:
                        cancellation_check(stage)
                    except BaseException as error:
                        cancelled.append(error)
                        raise
            report['cdt_calls'] += 1
            try:
                cdt = triangulate_polygon(np.asarray(outer),tuple(np.asarray(h) for h in holes),
                    tuple(np.asarray(e) for e in fixed_edges),interior_points=np.asarray(extra).reshape((-1,2)),
                    protected_node_ids=protected,backend='python',cancellation_check=check)
            except MeshError as error:
                if cancelled:
                    raise
                detail['reason'] = 'cdt_refusal: '+str(error)[:180]
                break
            coords = {tuple(p):i for i,p in enumerate(cdt.points)}
            if len(coords) != len(cdt.points) or set(coords) != set(raw):
                detail['reason'] = 'changed_station_inventory'
                break
            local_protected = {tuple(sorted((coords[a],coords[b]))) for a,b in barriers}
            pp,qq,tt,pair_report = improve(cdt.points,(),cdt.triangles,
                fixed_nodes=set(range(len(cdt.points))),protected_edges=local_protected,
                enable_pairing=True,enable_smoothing=False,budget=4096,
                work=work,cancellation_check=cancellation_check)
            incidence = {}; total = 0
            valid = True
            for body in (*qq,*tt):
                work.step('smart-mapped:refinement-audit')
                if _metric(pp,body) is None:
                    valid = False; break
                total += _area2(tuple(tuple(pp[n]) for n in body))
                for edge in _edges(body):
                    key = tuple(sorted(edge)); incidence[key] = incidence.get(key,0)+1
            wanted = {tuple(sorted((coords[output_points[a]],coords[output_points[b]]))) for a,b in boundary}
            expected_area = sum((_area2(tuple(output_points[n] for n in cells[row])) for row in component),0)
            # Axis-aligned regular rectangles are retained even inside a patch.
            regular = {frozenset(cells[row]) for row in component if row < len(quads)
                       and all((points[b][0] == points[a][0]) != (points[b][1] == points[a][1])
                               for a,b in _edges(cells[row]))}
            recovered = {frozenset(tuple(pp[n]) for n in body) for body in qq}
            regular_restored = all(frozenset(output_points[n] for n in body) in recovered for body in regular)
            if (not valid or total != expected_area
                    or {n for body in (*qq,*tt) for n in body} != set(range(len(pp)))
                    or not regular_restored
                    or {e for e,c in incidence.items() if c == 1} != wanted
                    or any(c not in (1,2) for c in incidence.values())
                    or any(incidence.get(tuple(sorted((coords[a],coords[b])))) != 2 for a,b in fixed_edges)):
                detail['reason'] = 'exact_cavity_audit_declined'
                break
            score = _score(pp,tt)
            detail['attempts'].append(dict(new_points=len(new_points),score=score))
            if (score[1] <= before[1] and score < before
                    and (best is None or score < best[0])):
                best = score,pp,qq,tt
            current_points,current_triangles = pp,tt
        if best is None:
            report['declined_cavities'] += 1
            continue
        _,pp,qq,tt = best
        detail['reason'] = 'accepted'
        lookup = {p:i for i,p in enumerate(output_points)}
        if report['added_nodes']+len(set(map(tuple,pp))-set(lookup)) > 128:
            detail['reason'] = 'actual_added_node_limit'
            report['declined_cavities'] += 1
            continue
        mapping = {}
        for i,p in enumerate(map(tuple,pp)):
            if p not in lookup:
                lookup[p] = len(output_points); output_points.append(p)
                report['added_nodes'] += 1
            mapping[i] = lookup[p]
        output_quads.extend(tuple(mapping[n] for n in body) for body in qq)
        output_triangles.extend(tuple(mapping[n] for n in body) for body in tt)
        removed.update(component)
        released = [row for row in component if row < len(quads)]
        report['released_quad_indices'].extend(released)
        restored = {frozenset(body) for body in output_quads}
        report['restored_quad_indices'].extend(row for row in released if frozenset(quads[row]) in restored)
        report['accepted_cavities'] += 1
    result_quads = tuple(body for row,body in enumerate(quads) if row not in removed)+tuple(output_quads)
    result_triangles = tuple(body for row,body in enumerate(triangles) if row+len(quads) not in removed)+tuple(output_triangles)
    result_points = np.asarray(output_points)
    report['after'] = _summary(result_points,result_quads,result_triangles)
    report['budget_exhausted'] = report['cdt_calls'] >= budget
    return result_points,result_quads,result_triangles,report
