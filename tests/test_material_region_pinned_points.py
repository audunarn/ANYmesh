"""Explicit vertex inputs survive native frontal candidate optimization."""
import numpy as np

from anymesher.surface_mesh import mesh_planar_surface, SurfaceMeshOptions
from anymesher.native_v2 import NativeMeshingOptions


def test_frontal_smoothing_receives_explicit_pinned_vertices(monkeypatch):
    import anymesher.surface_mesh as surface
    original = surface._optimize_candidate
    seen = []
    point = np.asarray(((.37,.43),))
    def capture(candidate, protected, explicit, *args, **kwargs):
        seen.append(np.asarray(explicit).copy())
        return original(candidate,protected,explicit,*args,**kwargs)
    monkeypatch.setattr(surface,'_optimize_candidate',capture)
    core = mesh_planar_surface(((0,0),(1,0),(1,1),(0,1)), interior_points=point,
        options=SurfaceMeshOptions(target_size=.35,recombine=False,backend='python',min_angle=15.,
            native_options=NativeMeshingOptions(point_placement='frontal_delaunay',
                metric_mode='isotropic_spatial',max_insertions=32,max_topology_operations=256)))
    assert len(seen) >= 2
    assert all(np.array_equal(rows,point) for rows in seen)
    assert sum(np.array_equal(row[:2],point[0]) for row in core.node_coordinates) == 1


def test_transition_repair_cannot_move_an_explicit_vertex_or_erase_spent_work(monkeypatch):
    from dataclasses import replace
    import anymesher._frontal_transition_quality as transition
    import anymesher._physical_t3_refinement as physical
    point = np.asarray(((.37,.43),))
    observed = []
    def invalid_move(candidate, protected, settings, report, *args, **kwargs):
        row = np.flatnonzero(np.all(candidate.points == point[0],axis=1))[0]
        moved = candidate.points.copy()
        moved[row] += .01
        observed.append(int(report['topology_operations']) + 1)
        return replace(candidate,points=moved), dict(report,
            topology_operations=observed[-1], chart_transition_repair={
                'accepted':True,'candidate_moved_nodes':[int(row)],'trials':1})
    monkeypatch.setattr(transition,'repair_frontal_transition',invalid_move)
    # Isolate the move-rejection contract from the subsequent bisection remedy.
    monkeypatch.setattr(physical,'refine_physical_candidate',
                        lambda candidate, triangulation, settings, report, *args:
                        (candidate,triangulation,report))
    diagnostics = {}
    core = mesh_planar_surface(((0,0),(1,0),(1,1),(0,1)),interior_points=point,
        options=SurfaceMeshOptions(target_size=1.,recombine=False,backend='python',
            min_angle=45.,enforce_quality=False,native_options=NativeMeshingOptions(
                point_placement='frontal_delaunay',metric_mode='isotropic_spatial',
                max_insertions=1,max_topology_operations=8)),
        diagnostics=diagnostics, _preserve_spatial_refinement=True,
        _metric_to_physical=lambda rows:np.column_stack((rows,np.zeros(len(rows)))),
        _metric_jacobian=np.asarray(((1.,0.),(0.,1.),(0.,0.))))
    assert observed
    assert sum(np.array_equal(row[:2],point[0]) for row in core.node_coordinates) == 1
    receipt = diagnostics['native_v2']
    assert receipt['topology_operations'] == observed[-1]
    assert not receipt['chart_transition_repair']['accepted']
    assert receipt['chart_transition_repair']['pinned_input_rejection']
