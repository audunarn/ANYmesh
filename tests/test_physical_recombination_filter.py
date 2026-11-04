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



def mixed_target_miss():
    baseline, candidate = fixture()
    points = {5: np.array((2., 0., 0.)), 6: np.array((3., 0., 0.)),
              7: np.array((2.5, .2, 0.))}
    for value in (baseline, candidate):
        value.nodes.update(copy.deepcopy(points))
        value.tris[30] = (5, 6, 7)
        value.elements_of_face[9] = [30]
    candidate.elements_of_sheet[9] = [30]
    candidate.elements_of_member[12] = [20, 30]
    return baseline, candidate


def test_advisory_mixed_target_miss_restores_only_quads_and_preserves_associations():
    baseline, candidate = mixed_target_miss()
    original = pickle.dumps(baseline)
    points = {key: value.tobytes() for key, value in candidate.nodes.items()}
    report = retain_physical_recombination(candidate, baseline, SurfaceMeshOptions())
    assert report["rejected_quad_ids"] == [20]
    assert report["restored_triangle_count"] == 2
    assert set(report["physical_quality_before"]["poor_element_ids"]) == {30, 20}
    assert report["physical_quality_after"]["poor_element_ids"] == [30]
    assert report["physical_quality_after"]["accepted"] is False
    assert candidate.tris[30] == (5, 6, 7)
    assert candidate.elements_of_face[9] == [30]
    assert candidate.elements_of_sheet[9] == [30]
    restored = set(candidate.elements_of_face[7])
    assert set(candidate.elements_of_sheet[8]) == restored
    assert set(candidate.elements_of_member[12]) == restored | {30}
    assert {candidate.tris[e] for e in restored} == set(baseline.tris[e] for e in (10, 11))
    assert not candidate.quads
    assert points == {key: value.tobytes() for key, value in candidate.nodes.items()}
    assert pickle.dumps(baseline) == original


def triangle_miss():
    nodes = {1: np.array((0., 0., 0.)), 2: np.array((1., 0., 0.)),
             3: np.array((.5, .2, 0.))}
    return Mesh(nodes=nodes, tris={30: (1, 2, 3)}, elements_of_face={9: [30]})


def test_advisory_triangle_only_target_miss_is_retained_byte_exact():
    candidate = triangle_miss()
    baseline = copy.deepcopy(candidate)
    original = pickle.dumps(candidate)
    report = retain_physical_recombination(candidate, baseline, SurfaceMeshOptions())
    assert report["rejected_quad_ids"] == []
    assert report["restored_triangle_count"] == 0
    assert report["physical_quality_before"]["accepted"] is False
    assert report["physical_quality_after"] == report["physical_quality_before"]
    assert report["physical_quality_after"]["violation_counts"]["minimum_angle"] == 1
    assert pickle.dumps(candidate) == original


def test_explicit_mixed_target_miss_keeps_original_refusal_and_atomicity():
    baseline, candidate = mixed_target_miss()
    original = pickle.dumps(candidate)
    source = pickle.dumps(baseline)
    with pytest.raises(ValueError, match="physical recombination rejection includes a non-quad"):
        retain_physical_recombination(candidate, baseline, SurfaceMeshOptions(enforce_quality=True))
    assert pickle.dumps(candidate) == original
    assert pickle.dumps(baseline) == source


@pytest.mark.parametrize("enforce", (False, True))
def test_restored_target_poor_triangles_respect_explicit_enforcement(enforce):
    baseline, candidate = fixture()
    for value in (baseline, candidate):
        value.nodes[2][0] = value.nodes[3][0] = 10.
    original = pickle.dumps(candidate)
    source = pickle.dumps(baseline)
    settings = SurfaceMeshOptions(enforce_quality=enforce)
    if enforce:
        with pytest.raises(ValueError, match="restoring original triangle pairs did not preserve physical quality"):
            retain_physical_recombination(candidate, baseline, settings)
        assert pickle.dumps(candidate) == original
    else:
        report = retain_physical_recombination(candidate, baseline, settings)
        assert report["rejected_quad_ids"] == [20]
        assert report["restored_triangle_count"] == 2
        assert report["physical_quality_after"]["accepted"] is False
        assert not candidate.quads
        assert set(candidate.tris.values()) == set(baseline.tris.values())
    assert pickle.dumps(baseline) == source


@pytest.mark.parametrize("invalid", ("source", "candidate"))
def test_advisory_triangle_miss_cannot_hide_hard_invalidity(invalid):
    from anymesher.errors import MeshError
    candidate = triangle_miss()
    baseline = copy.deepcopy(candidate)
    (baseline if invalid == "source" else candidate).nodes[3][1] = 0.
    original, source = pickle.dumps(candidate), pickle.dumps(baseline)
    with pytest.raises(MeshError):
        retain_physical_recombination(candidate, baseline, SurfaceMeshOptions())
    assert pickle.dumps(candidate) == original
    assert pickle.dumps(baseline) == source
