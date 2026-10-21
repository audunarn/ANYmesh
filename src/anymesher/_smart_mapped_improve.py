"""Bounded optional quality proposals for the hidden planar prototype."""
from __future__ import annotations

from math import fsum, hypot, isfinite

import numpy as np

from .errors import MeshError
from .quad.quality_gate import (QUAD_FIRST_T3_MIN_ANGLE, corner_metrics,
                                quad_violation)
from .triangulation import orient2d


def _edges(body):
    return tuple(zip(body,body[1:]+body[:1]))


def _metric(points,body):
    polygon = [points[n] for n in body]
    if any(orient2d(polygon[i-1],polygon[i],polygon[(i+1)%len(body)]) <= 0.
           for i in range(len(body))):
        return None
    minimum,maximum,jacobian,aspect = corner_metrics(polygon)
    if not all(isfinite(v) for v in (minimum,maximum,jacobian,aspect)):
        return None
    if len(body) == 4 and quad_violation(polygon) is not None:
        return None
    return minimum,max(0.,QUAD_FIRST_T3_MIN_ANGLE-minimum) if len(body) == 3 else 0.


def _summary(points,quads,triangles):
    angles = [corner_metrics([points[n] for n in body])[0] for body in triangles]
    return dict(quads=len(quads),triangles=len(triangles),
                triangles_below_minimum=sum(a < QUAD_FIRST_T3_MIN_ANGLE for a in angles),
                minimum_triangle_angle=min(angles,default=None))


def improve(points,quads,triangles,*,fixed_nodes,protected_edges,
            enable_pairing,enable_smoothing,budget,work,cancellation_check):
    """Return detached candidate data; caller owns the final exact mesh audit.

    Pairing never removes a protected edge. Movement is limited to the initial
    deficient-triangle vertex set and cannot create/worsen a triangle deficit
    or violate an affected quadrilateral's existing shape policy.
    """
    points = np.array(points,dtype=float,copy=True)
    quads = list(map(tuple,quads))
    triangles = list(map(tuple,triangles))
    report = dict(enabled_pairing=enable_pairing,enabled_smoothing=enable_smoothing,
                  budget=budget,evaluations=0,accepted_pairs=0,accepted_moves=0,
                  moved_node_ids=[],before=_summary(points,quads,triangles))
    if cancellation_check is not None:
        cancellation_check('smart-mapped:improvement')
    if enable_pairing and budget:
        edge_owners = {}
        for row,body in enumerate(triangles):
            work.step('smart-mapped:pairing')
            for edge in _edges(body):
                edge_owners.setdefault(tuple(sorted(edge)),[]).append(row)
        proposals = []
        for edge,owners in sorted(edge_owners.items()):
            if len(owners) != 2 or edge in protected_edges:
                continue
            if report['evaluations'] >= budget:
                break
            work.step('smart-mapped:pairing')
            report['evaluations'] += 1
            first,second = owners
            boundary = [pair for row in owners for pair in _edges(triangles[row])
                        if tuple(sorted(pair)) != edge]
            if len({node for pair in boundary for node in pair}) != 4:
                continue
            following = dict(boundary)
            start = min(following)
            body = [start]
            for _ in range(3):
                body.append(following.get(body[-1],start))
            body = tuple(body)
            if len(set(body)) != 4 or following.get(body[-1]) != start:
                continue
            metric = _metric(points,body)
            if metric is None:
                continue
            source_metrics = [_metric(points,triangles[row]) for row in owners]
            if any(value is None for value in source_metrics):
                continue
            bad = sum(value[1] > 0. for value in source_metrics)
            proposals.append((-bad,-metric[0],edge,first,second,body))
        consumed = set()
        for _,_,edge,first,second,body in sorted(proposals):
            work.step('smart-mapped:pairing')
            if first in consumed or second in consumed:
                continue
            consumed.update((first,second))
            quads.append(body)
            report['accepted_pairs'] += 1
        triangles = [body for row,body in enumerate(triangles) if row not in consumed]
    report['after_pairing'] = _summary(points,quads,triangles)
    cells = [*quads,*triangles]
    metrics = [_metric(points,body) for body in cells]
    if any(value is None for value in metrics):
        raise MeshError('invalid input cell in prototype improvement')
    incident = {}
    for row,body in enumerate(cells):
        for node in body:
            incident.setdefault(node,[]).append(row)
    eligible = sorted({node for row,body in enumerate(cells) if metrics[row][1] > 0.
                       for node in body}-set(fixed_nodes))
    report['eligible_move_nodes'] = len(eligible)
    moved = set()
    if enable_smoothing and budget:
        for sweep in range(8):
            progress = False
            for node in eligible:
                if report['evaluations'] >= budget:
                    break
                if cancellation_check is not None:
                    cancellation_check('smart-mapped:node-movement')
                affected = incident[node]
                deficits = [metrics[row][1] for row in affected]
                before = (max(deficits),fsum(deficits))
                if before[0] == 0.:
                    continue
                neighbors = sorted({n for row in affected for n in cells[row]}-{node})
                here = points[node].copy()
                center = tuple(fsum(float(points[n,axis]) for n in neighbors)/len(neighbors)
                               for axis in (0,1))
                scale = min(hypot(*(points[n]-here)) for n in neighbors)
                directions = [(center[0]-here[0],center[1]-here[1])]
                for other in neighbors:
                    delta = points[other]-here
                    length = hypot(*delta)
                    if length:
                        directions.append(tuple(delta*(scale/length)))
                        directions.append(tuple(-delta*(scale/length)))
                best = None
                for divisor in (2.,4.,8.,16.):
                    for dx,dy in directions:
                        if report['evaluations'] >= budget:
                            break
                        work.step('smart-mapped:node-movement')
                        report['evaluations'] += 1
                        proposed = (here[0]+dx/divisor,here[1]+dy/divisor)
                        if not all(isfinite(v) for v in proposed):
                            continue
                        points[node] = proposed
                        values = [_metric(points,cells[row]) for row in affected]
                        points[node] = here
                        if any(value is None for value in values):
                            continue
                        new = [value[1] for value in values]
                        if any(a > b for a,b in zip(new,deficits)):
                            continue
                        score = (max(new),fsum(new))
                        if score < before and (best is None or score < best[0]):
                            best = score,proposed,values
                if best is not None:
                    _,points[node],values = best
                    for row,value in zip(affected,values):
                        metrics[row] = value
                    moved.add(node)
                    report['accepted_moves'] += 1
                    progress = True
            if not progress or report['evaluations'] >= budget:
                break
    report['moved_node_ids'] = sorted(moved)
    report['after'] = _summary(points,quads,triangles)
    report['budget_exhausted'] = report['evaluations'] >= budget
    return points,tuple(quads),tuple(triangles),report
