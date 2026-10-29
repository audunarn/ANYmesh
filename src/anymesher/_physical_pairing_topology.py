"""Detached flips of two-triangle fixed-boundary fans before mixed matching.

This proposes connectivity only. The caller retains owner/source authority and
must evaluate the final physical mesh; no source receipt is made here.
"""
from collections import Counter
from dataclasses import replace
import numpy as np


def repair_two_triangle_fans(points, triangles, xyz, settings, fixed_rows,
                             protected_edges, *, topology_remaining):
    from .core import MeshCore
    from .errors import MeshError
    from .native_v2 import MutableT3Topology
    from .quality_v2 import evaluate_quality
    from .surface_mesh import (_quality_threshold_report, _published_quality_report,
                               _physical_chart_invalid_count)
    from ._physical_t3_refinement import _alternative_progress
    if type(topology_remaining) is not int or topology_remaining < 0:
        raise MeshError('physical fan recovery requires a remaining topology allowance')
    uv=np.asarray(points);original=np.asarray(triangles);world=np.asarray(xyz)
    current=original.copy()
    protected={tuple(sorted(map(int,e))) for e in protected_edges}
    fixed=set(map(int,fixed_rows));facts=dict(attempts=0,commits=0,corners=[])
    def score(cells):
        core=MeshCore(world,cells)
        quality=evaluate_quality(core,check_validity=False)
        threshold=_quality_threshold_report(quality,settings)
        report=_published_quality_report(core,settings,quality,threshold)
        report['violation_counts']=dict(threshold['violation_counts'])
        report['invalid_element_count']=(len(quality.validity.errors)
                                         + _physical_chart_invalid_count(uv,cells))
        return report
    before=score(current)
    if before['invalid_element_count'] or not before['quality_violation_count']:
        facts['final_quality']=before
        return current,0,facts
    for corner in sorted(fixed):
        if not before['quality_violation_count']:break
        if facts['attempts'] >= topology_remaining:break
        rows=np.flatnonzero(np.any(current==corner,axis=1))
        if len(rows)!=2:continue
        local=current[rows]
        counts=Counter(tuple(sorted((int(a),int(b)))) for cell in local
                       for a,b in zip(cell,np.roll(cell,-1)))
        shared=[edge for edge,count in counts.items() if count==2]
        perimeter=[edge for edge,count in counts.items() if count==1]
        if len(shared)!=1 or len(perimeter)!=4 or corner not in shared[0]:continue
        boundary=[edge for edge in perimeter if corner in edge]
        if len(boundary)!=2 or not all(edge in protected for edge in boundary):continue
        if shared[0] in protected:continue
        # Both boundary spokes and all cavity perimeter edges are immutable.
        # The existing primitive checks convexity, incidence and UV orientation.
        facts['attempts']+=1
        topology=MutableT3Topology(uv.copy(),local.copy(),perimeter)
        if not topology.flip_edge(shared[0]):continue
        replacement=topology.triangles
        proposed=current.copy();proposed[rows]=replacement
        # An opposite edge already outside the cavity is not a valid flip.
        # Full structural validation below remains the sole topology gate.
        summary=score(proposed)
        if not _alternative_progress(before,summary):continue
        current=proposed;before=summary;facts['commits']+=1
        facts['corners'].append(int(corner))
    facts['final_quality']=before
    return current,facts['attempts'],facts

def repair_active_triangle_flips(points, report, settings, protected_edges, *,
                                 topology_remaining):
    """Improve only active T3 neighbours, retaining every mixed-cell identity.

    Complete mixed reports, including all remote growth pairs, decide adoption.
    A Q4-covered inactive triangle is never an operand of a flip.
    """
    from .errors import MeshError
    from .native_v2 import MutableT3Topology, orient2d
    from ._physical_corner_cavity import _core, _score
    from .surface_mesh import _physical_chart_invalid_count
    from ._physical_t3_refinement import _alternative_progress
    if type(topology_remaining) is not int or topology_remaining<0:
        raise MeshError('physical mixed flips require a remaining topology allowance')
    uv=np.asarray(points);current=report.mesh
    protected={tuple(sorted(map(int,e))) for e in protected_edges}
    facts=dict(attempts=0,commits=0)
    before=_score(current,settings)
    before['invalid_element_count']+=_physical_chart_invalid_count(uv,current.triangle_connectivity[current.triangle_active])
    if before['invalid_element_count']:return report,0,facts
    # Original signed chart orientation cannot be repaired by canonicalization.
    if any(orient2d(*(uv[int(n)] for n in row))<=0
           for row in current.triangle_connectivity[current.triangle_active]):
        return report,0,facts
    while before['quality_violation_count'] and facts['attempts']<topology_remaining:
        incidence={}
        for kind,cells,active in (('T',current.triangle_connectivity,current.triangle_active),
                                  ('Q',current.quad_connectivity,current.quad_active)):
            for row in np.flatnonzero(active):
                for a,b in zip(cells[row],np.roll(cells[row],-1)):
                    incidence.setdefault(tuple(sorted((int(a),int(b)))),[]).append((kind,int(row)))
        poor=set(before['poor_element_ids']);committed=False
        for edge,attached in sorted(incidence.items()):
            if facts['attempts']>=topology_remaining:break
            if edge in protected or len(attached)!=2 or any(kind!='T' for kind,_ in attached):continue
            rows=[row for _,row in attached]
            if not any(int(current.triangle_ids[row]) in poor for row in rows):continue
            if current.triangle_owner_handles[rows[0]]!=current.triangle_owner_handles[rows[1]]:continue
            local=current.triangle_connectivity[rows]
            counts=Counter(tuple(sorted((int(a),int(b)))) for cell in local for a,b in zip(cell,np.roll(cell,-1)))
            perimeter=[e for e,n in counts.items() if n==1]
            if len(perimeter)!=4:continue
            facts['attempts']+=1
            topology=MutableT3Topology(uv.copy(),local.copy(),perimeter)
            if not topology.flip_edge(edge):continue
            cells=current.triangle_connectivity.copy();cells[rows]=topology.triangles
            candidate=_core(current,current.node_coordinates,cells)
            summary=_score(candidate,settings)
            summary['invalid_element_count']+=_physical_chart_invalid_count(uv,candidate.triangle_connectivity[candidate.triangle_active])
            if not _alternative_progress(before,summary):continue
            current=candidate;before=summary;committed=True;facts['commits']+=1
            break
        if not committed:break
    facts['final_quality']=before
    return replace(report,mesh=current),facts['attempts'],facts
