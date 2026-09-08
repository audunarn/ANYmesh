from types import SimpleNamespace

import numpy as np
import pytest

from anymesher import Mesh

from anymesher._cylindrical_transition_repair import repair_component_transitions
from anymesher.native_v2 import NativeMeshingOptions
from anymesher.surface_mesh import SurfaceMeshOptions


def fixture():
    points = np.array([[0., 0.], [2., 0.], [2., 1.25], [0., 2.]])
    triangles = np.array([[0, 1, 3], [1, 2, 3]])
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    mesh = Mesh(
        nodes={i: np.array([*p, 0.]) for i, p in enumerate(points)},
        tris={10: (0, 1, 3), 11: (1, 2, 3)}, elements_of_face={1: [10, 11]},
    )
    def face_data(face):
        assert face == 1
        return tuple(range(4)), points.copy(), triangles.copy(), [], edges.copy(), {}, 1
    charts = {1: SimpleNamespace(evaluate=lambda xy: np.column_stack((xy, np.zeros(len(xy)))))}
    settings = SurfaceMeshOptions(native_options=NativeMeshingOptions(max_topology_operations=20000))
    diagnostics = {"faces": {"1": {"work_totals": {"topology_operations": 0}}}}
    return mesh, charts, face_data, settings, diagnostics


def test_component_transition_repair_preserves_boundary_and_element_ids():
    mesh, charts, face_data, settings, diagnostics = fixture()
    original = {key: value.tobytes() for key, value in mesh.nodes.items()}
    phases = []
    def checkpoint(phase="transition"):
        phases.append(phase)
    spent = {1: 0}
    repair_component_transitions(mesh, [1], charts, face_data, settings, spent, diagnostics, checkpoint)
    assert phases and all(isinstance(phase, str) for phase in phases)
    assert original == {key: value.tobytes() for key, value in mesh.nodes.items()}
    assert set(mesh.tris) == {10, 11}
    assert mesh.elements_of_face == {1: [10, 11]}
    assert diagnostics["faces"]["1"]["transition_repair"]["target_met"]
    assert diagnostics["faces"]["1"]["transition_repair"]["committed_restoration_flips"] == 1
    assert diagnostics["faces"]["1"]["final_chart_quality"]["quality_violation_count"] == 0
    assert 0 < spent[1] <= 20000


def test_exhausted_component_budget_preserves_connectivity():
    mesh, charts, face_data, settings, diagnostics = fixture()
    original = dict(mesh.tris)
    spent = {1: 20000}
    repair_component_transitions(mesh, [1], charts, face_data, settings, spent, diagnostics, lambda: None)
    assert mesh.tris == original
    report = diagnostics["faces"]["1"]["transition_repair"]
    assert report["attempts"] == 0 and report["work_units"] == 0
    assert report["budget_exhausted"] and not report["target_met"]


def test_component_cancel_before_work_preserves_detached_mesh():
    mesh, charts, face_data, settings, diagnostics = fixture()
    original = dict(mesh.tris)
    def checkpoint():
        raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        repair_component_transitions(mesh, [1], charts, face_data, settings,
                                     {1: 0}, diagnostics, checkpoint)
    assert mesh.tris == original
