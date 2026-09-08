"""Exercise phase ordering without invoking a full cylindrical refinement."""
from types import SimpleNamespace

import numpy as np

from anymesher import NativeMeshingOptions
from anymesher.surface_mesh import SurfaceMeshOptions
import anymesher._cylindrical_transition_repair as repair


def test_background_restoration_precedes_corner_construction(monkeypatch):
    p = np.array([[0., 0.], [.3927, 0.], [0., 1/9]])
    t = np.array([[0, 1, 2]])
    edges = np.array([[0, 1], [1, 2], [2, 0]])
    xyz = np.column_stack((p, np.zeros(len(p))))
    tris = {1: (1, 2, 3)}
    mesh = SimpleNamespace(nodes={i + 1: row.copy() for i, row in enumerate(xyz)},
                           tris=tris, shells=tris, elements_of_face={1: [1]},
                           elements_of_sheet={}, elements_of_member={})
    events = []
    def restore(points, triangles, protected, **kwargs):
        events.append("restore")
        return SimpleNamespace(triangles=triangles.copy(), flips=0, converged=True)
    def corner(points, triangles, protected, poor, **kwargs):
        events.append("corner")
        return SimpleNamespace(points=points, triangles=triangles, attempts=0,
                               added_points=0, corner_node=None, improved=False)
    monkeypatch.setattr(repair, "restore_constrained_delaunay", restore)
    monkeypatch.setattr(repair, "repair_corner_cavity", corner)
    monkeypatch.setattr(repair, "repair_triangle_angles", lambda points, *args, **kw:
                        SimpleNamespace(points=points.copy(), trials=0))
    monkeypatch.setattr(repair, "repair_joint_triangle_quality", lambda points, *args, **kw:
                        SimpleNamespace(points=points.copy(), trials=0))
    monkeypatch.setattr(repair, "repair_transition_patch", lambda points, triangles, *args, **kw:
                        SimpleNamespace(points=points.copy(), triangles=triangles.copy(),
                                        attempts=0, committed_flips=0, improved=False))
    options = SurfaceMeshOptions(native_options=NativeMeshingOptions(
        max_insertions=1, max_topology_operations=32))
    diagnostics = {"faces": {"1": {"work_totals": {"staged_points": 0}}}}
    repair.repair_component_transitions(
        mesh, [1], {1: SimpleNamespace(evaluate=lambda x: np.column_stack((x, np.zeros(len(x)))))},
        lambda face: ((1, 2, 3), p.copy(), t.copy(), [], edges, {}, 1),
        options, {1: 0}, diagnostics, lambda *args: None,
    )
    assert events == ["restore", "corner"]
    assert all(mesh.nodes[i + 1].tobytes() == row.tobytes() for i, row in enumerate(xyz))
