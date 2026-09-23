from copy import deepcopy
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from anymesher._cylindrical_quadratic import QuadraticComponentStage
from anymesher._shared_triangle_split import propagate_triangle_split
from anymesher.errors import MeshError
from anymesher.mesh import Mesh


def fixture():
    nodes = {1: (0., 0., 0.), 2: (1., 0., 0.), 3: (0., 1., 0.),
             4: (0.3, -0.0, 0.), 5: (0.5, 0.5, 0.), 6: (0., 0.5, 0.)}
    mesh = Mesh(geometry_model_id="owner", geometry_revision=7, order="quadratic",
                nodes={n: np.array(p) for n, p in nodes.items()},
                tris={10: (1, 2, 3, 4, 5, 6)}, elements_of_face={9: [10]},
                nodes_of_edge={7: [1, 4, 2]})
    entries = tuple(SimpleNamespace(key=SimpleNamespace(edge_id=7, parameter=t),
                                    node_id=n, point=mesh.nodes[n])
                    for n, t in ((1, 0.), (4, 0.3), (2, 1.)))
    return mesh, entries


def stage(eligible=True):
    mesh, entries = fixture()
    return mesh, QuadraticComponentStage(mesh, (9,), entries, eligible_edges=(7,) if eligible else ())


def promote(state, candidate=None, **changes):
    return state.promote(candidate or state.mesh,
                         evaluate_edge=changes.pop("evaluate_edge", lambda edge, t: (t, 0., 0.)),
                         evaluate_interior=lambda face, a, b: 0.5 * (a + b), **changes)


def test_round_trip_reuses_every_original_midside_without_evaluation():
    source, state = stage()
    def forbidden(*args):
        raise AssertionError("unchanged edge was re-evaluated")
    result = state.promote(state.mesh, evaluate_edge=forbidden, evaluate_interior=forbidden)
    assert result.mesh.tris == source.tris
    assert result.mesh.nodes_of_edge == source.nodes_of_edge
    assert result.boundary_stations == ()
    assert all(result.mesh.nodes[n].tobytes() == p.tobytes() for n, p in source.nodes.items())
    assert state.mesh.tris == {10: (1, 2, 3)} and source.order == "quadratic"


def test_existing_native_station_is_reused_in_both_orientations():
    _, state = stage()
    proposal = state.split_proposal(lambda edge, node, p: p[:2])
    for lower, upper in ((Fraction(0), Fraction(1)), (Fraction(1), Fraction(0))):
        station, point, node = proposal(7, lower, upper)
        assert station == Fraction.from_float(0.3) and node == 4
        assert point.tobytes() == np.array((0.3, -0.0)).tobytes()
    assert proposal(7, Fraction(0), Fraction.from_float(0.3)) is None


def test_shared_split_is_promoted_late_with_owner_evaluated_child_midsides():
    source, state = stage()
    propagate_triangle_split(state.mesh, (9,), (1, 2), 4, cache={})
    state.mesh.nodes_of_edge[7] = [1, 4, 2]
    calls = []
    def evaluate(edge, t):
        calls.append((edge, t))
        return (t, 0., 0.)
    result = promote(state, evaluate_edge=evaluate)
    assert len(result.mesh.tris) == 2
    assert all(len(nodes) == 6 for nodes in result.mesh.tris.values())
    assert {t for _, t in calls} == {float(Fraction.from_float(0.3) / 2), float((Fraction.from_float(0.3) + 1) / 2)}
    assert len(result.boundary_stations) == 2
    assert result.mesh.nodes[4].tobytes() == source.nodes[4].tobytes()
    assert result.mesh.nodes_of_edge[7][::2] == [1, 4, 2]
    assert source.tris == {10: (1, 2, 3, 4, 5, 6)}


@pytest.mark.parametrize("mutation", ("coordinate", "order", "binding", "incidence"))
def test_invalid_staging_never_mutates_source_or_candidate(mutation):
    source, state = stage()
    candidate = deepcopy(state.mesh)
    if mutation == "coordinate":
        candidate.nodes[4][0] += 0.01
    elif mutation == "order":
        candidate.nodes_of_edge[7].reverse()
    elif mutation == "binding":
        candidate.geometry_revision += 1
    else:
        candidate.tris[10] = (1, 2, 6)
    before = {n: p.tobytes() for n, p in candidate.nodes.items()}
    with pytest.raises(MeshError):
        promote(state, candidate)
    assert before == {n: p.tobytes() for n, p in candidate.nodes.items()}
    assert source.nodes[4][0] == 0.3


def test_unsplittable_interval_cannot_acquire_a_new_corner_role():
    _, state = stage(False)
    assert state.split_proposal(lambda *args: None)(7, Fraction(0), Fraction(1)) is None
    propagate_triangle_split(state.mesh, (9,), (1, 2), 4, cache={})
    state.mesh.nodes_of_edge[7] = [1, 4, 2]
    with pytest.raises(MeshError, match="ineligible"):
        promote(state)


def test_operational_evaluation_failure_preserves_the_linear_candidate():
    source, state = stage()
    propagate_triangle_split(state.mesh, (9,), (1, 2), 4, cache={})
    state.mesh.nodes_of_edge[7] = [1, 4, 2]
    error = RuntimeError("owner evaluation failed")
    def fail(*args):
        raise error
    with pytest.raises(RuntimeError) as caught:
        promote(state, evaluate_edge=fail)
    assert caught.value is error
    assert all(len(row) == 3 for row in state.mesh.tris.values())
    assert source.order == "quadratic"


def test_cancellation_after_promotion_has_no_partial_publication():
    _, state = stage()
    def cancel(phase):
        if phase.endswith("complete"):
            raise RuntimeError("cancel")
    with pytest.raises(RuntimeError, match="cancel"):
        promote(state, cancellation_check=cancel)
    assert state.mesh.order == "linear" and len(state.mesh.tris[10]) == 3


def test_foreign_boundary_coordinates_and_external_roles_fail_closed():
    source, entries = fixture()
    source.beams[22] = (1, 4, 2)
    with pytest.raises(MeshError, match="protected midside role"):
        QuadraticComponentStage(source, (9,), entries, eligible_edges=(7,))
    source.beams.clear()
    entries[1].point = entries[1].point.copy()
    entries[1].point[0] += 0.01
    with pytest.raises(MeshError, match="byte-inconsistent"):
        QuadraticComponentStage(source, (9,), entries)
