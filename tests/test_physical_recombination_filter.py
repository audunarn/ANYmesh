import copy
import pickle

import numpy as np
import pytest

from anymesher import Mesh
from anymesher._physical_recombination_filter import retain_physical_recombination
from anymesher.surface_mesh import SurfaceMeshOptions


def fixture(height=.5):
    nodes = {i + 1: np.array(p, dtype=float) for i, p in enumerate(
        [(0, 0, 0), (1, 0, 0), (1, 1, height), (0, 1, 0)])}
    baseline = Mesh(nodes=copy.deepcopy(nodes), tris={10: (1, 2, 3), 11: (1, 3, 4)},
                    elements_of_face={7: [10, 11]})
    candidate = Mesh(nodes=nodes, quads={20: (1, 2, 3, 4)}, elements_of_face={7: [20]},
                     elements_of_sheet={8: [20]})
    return baseline, candidate


def test_warped_pair_restores_original_triangles_and_ownership():
    baseline, candidate = fixture()
    original = pickle.dumps(baseline)
    points = {key: value.tobytes() for key, value in candidate.nodes.items()}
    report = retain_physical_recombination(candidate, baseline, SurfaceMeshOptions())
    assert report["rejected_quad_ids"] == [20]
    assert report["restored_triangle_count"] == 2
    assert report["physical_quality_after"]["accepted"]
    assert not candidate.quads
    assert set(candidate.tris.values()) == set(baseline.tris.values())
    assert set(candidate.elements_of_face[7]) == set(candidate.tris)
    assert set(candidate.elements_of_sheet[8]) == set(candidate.tris)
    assert points == {key: value.tobytes() for key, value in candidate.nodes.items()}
    assert pickle.dumps(baseline) == original


def test_valid_physical_quad_is_retained_byte_exact():
    baseline, candidate = fixture(0.)
    original = pickle.dumps(candidate)
    report = retain_physical_recombination(candidate, baseline, SurfaceMeshOptions())
    assert report["rejected_quad_ids"] == []
    assert pickle.dumps(candidate) == original


def test_missing_original_pair_fails_without_mutating_candidate():
    baseline, candidate = fixture()
    del baseline.tris[11]
    baseline.elements_of_face[7].remove(11)
    original = pickle.dumps(candidate)
    with pytest.raises(ValueError, match="uniquely recover"):
        retain_physical_recombination(candidate, baseline, SurfaceMeshOptions())
    assert pickle.dumps(candidate) == original
