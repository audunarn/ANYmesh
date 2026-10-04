"""Opt-in row lineage tests; no compiled native meshing or mixed assembly."""

from types import SimpleNamespace

import numpy as np
import pytest

from anymesher._row_provenance import (bind_output_rows, material_input_rows,
                                       retry_input_rows)
from anymesher.core import MeshCore
from anymesher.errors import MeshError
from anymesher.surface_mesh import mesh_planar_surface


def test_receipt_input_rows_include_repeated_source_occurrences():
    payload = {
        "emitted_chart_loops": (((0, 0), (1, 0), (1, 1), (0, 1)),),
        "boundary_inputs": tuple({"global_node": node} for node in (11, 12, 13, 14)),
        "pinned_inputs": ({"global_node": 15},),
        "interior_paths": ({"global_nodes": (11, 15, 13)},),
        "emitted_chart_constraints": (((0, 0), (.5, .5)), ((.5, .5), (1, 1))),
    }
    receipt = SimpleNamespace(to_dict=lambda: payload)
    assert material_input_rows(receipt) == {
        0: 11, 1: 12, 2: 13, 3: 14, 4: 15,
        5: 11, 6: 15, 7: 15, 8: 13,
    }
    payload["emitted_chart_constraints"] = ()
    with pytest.raises(MeshError, match="constraint token count"):
        material_input_rows(receipt)


def test_retry_row_lineage_uses_prior_indices():
    original = {0: 11, 1: 12, 2: 13, 3: 14, 4: 15,
                5: 11, 6: 15, 7: 15, 8: 13}
    previous = ((11, 0), (12, 1), (13, 2), (14, 3), (15, 4))
    assert retry_input_rows(original, previous, 4, [4, 5, 6], 4, 2) == {
        0: 11, 1: 12, 2: 13, 3: 14, 4: 15,
        8: 11, 9: 15, 10: 15, 11: 13,
    }
    with pytest.raises(MeshError, match="omits added rows"):
        retry_input_rows(original, previous, 4, [4, 5, 6], 2, 2)


def test_selected_retry_keeps_constraint_ids_after_added_point(monkeypatch):
    import anymesher.surface_mesh as surface
    from anymesher.surface_mesh import SurfaceMeshOptions

    make = surface._make_candidate
    key = surface._candidate_selection_key

    def candidate(*args, **kwargs):
        value = make(*args, **kwargs)
        value.report["poor_element_ids"] = [1] if value.rounds == 0 else []
        value.report["repair_element_ids"] = value.report["poor_element_ids"]
        return value

    monkeypatch.setattr(surface, "_make_candidate", candidate)
    monkeypatch.setattr(surface, "_candidate_selection_key",
                        lambda value, *, prefer_growth: (-value.rounds,
                            *key(value, prefer_growth=prefer_growth)))
    monkeypatch.setattr(surface, "_refinement_midpoints",
                        lambda *_args, **_kwargs: np.asarray(((.25, .25),)))
    outer = np.asarray(((0., 0.), (1., 0.), (1., 1.), (0., 1.)))
    pin = np.asarray(((.5, .5),))
    constraints = (np.asarray(((0., 0.), (.5, .5))),
                   np.asarray(((.5, .5), (1., 1.))))
    path = surface._run_quality_path(
        "staggered_chart", outer, (), constraints, pin,
        np.asarray(((.25, .75), (.75, .25))),
        SurfaceMeshOptions(target_size=.5, recombine=False, backend="python"),
        None, protected_node_ids={0: 11, 1: 12, 2: 13, 3: 14, 4: 15,
                                  5: 11, 6: 15, 7: 15, 8: 13})
    assert path["best"].rounds >= 1
    assert path["attempted_added_points"] >= 1
    points = path["best"].points
    core = MeshCore(points, path["best"].triangles)
    expected = np.column_stack((points, np.zeros(len(points))))
    origin = bind_output_rows(core, points, path["triangulation"],
                              {11, 12, 13, 14, 15}, expected)
    assert set(origin.validate(core).values()) == {11, 12, 13, 14, 15}


@pytest.mark.parametrize("recombine", (False, True))
@pytest.mark.parametrize("reversed_loop", (False, True))
def test_small_public_surface_preserves_source_rows_and_marks_generated(recombine,
                                                                      reversed_loop):
    outer = np.asarray(((0., 0.), (1., 0.), (1., 1.), (0., 1.)))
    if reversed_loop:
        outer = outer[::-1].copy()
    core = mesh_planar_surface(
        outer, interior_points=np.asarray(((.5, .5),)), target_size=.5,
        recombine=recombine, backend="python", _boundary_is_seeded=True,
        _protected_node_ids={0: 11, 1: 12, 2: 13, 3: 14, 4: 15})
    origin = core._output_row_provenance
    assert set(origin.validate(core).values()) == {11, 12, 13, 14, 15}
    assert origin.source_by_row.count(None) == core.num_nodes - 5
    assert core.num_nodes > 5
    if not recombine:
        plain = mesh_planar_surface(
            outer, interior_points=np.asarray(((.5, .5),)), target_size=.5,
            recombine=False, backend="python", _boundary_is_seeded=True)
        np.testing.assert_array_equal(core.node_coordinates, plain.node_coordinates)
        np.testing.assert_array_equal(core.triangle_connectivity, plain.triangle_connectivity)
        assert not hasattr(plain, "_output_row_provenance")


def test_shared_constraint_occurrences_bind_one_source_row():
    outer = np.asarray(((0., 0.), (1., 0.), (1., 1.), (0., 1.)))
    pin = np.asarray(((.5, .5),))
    constraints = (np.asarray(((0., 0.), (.5, .5))),
                   np.asarray(((.5, .5), (1., 1.))))
    core = mesh_planar_surface(
        outer, constraints=constraints, interior_points=pin,
        target_size=.5, recombine=False, backend="python",
        _boundary_is_seeded=True,
        _protected_node_ids={0: 11, 1: 12, 2: 13, 3: 14, 4: 15,
                             5: 11, 6: 15, 7: 15, 8: 13})
    origin = core._output_row_provenance
    assert set(origin.validate(core).values()) == {11, 12, 13, 14, 15}
    assert len(origin.source_to_row) == 5


def test_callback_cannot_rebase_selected_output_origin(monkeypatch):
    import anymesher._row_provenance as provenance
    original = provenance.bind_output_rows
    selected = []

    def capture(*args):
        record = original(*args)
        selected.append(record)
        return record

    monkeypatch.setattr(provenance, "bind_output_rows", capture)

    def substitute(stage):
        if stage == "native surface validation complete":
            object.__setattr__(selected[0], "source_by_row", (99, 12, 13, 14))

    outer = np.asarray(((0., 0.), (1., 0.), (1., 1.), (0., 1.)))
    with pytest.raises(MeshError, match="origin changed before return"):
        mesh_planar_surface(
            outer, recombine=False, backend="python", _boundary_is_seeded=True,
            _protected_node_ids={0: 11, 1: 12, 2: 13, 3: 14},
            cancellation_check=substitute)


def test_permutation_merge_and_generated_identity_leak_refuse():
    points = np.asarray(((0., 0.), (1., 0.), (0., 1.), (.3, .3)))
    core = MeshCore(points, ((0, 1, 2), (0, 2, 3)))
    triangulation = SimpleNamespace(points=points.copy(),
                                    protected_node_rows=((11, 0), (12, 1), (13, 2)))
    expected = core.node_coordinates.copy()
    origin = bind_output_rows(core, points, triangulation, {11, 12, 13}, expected)
    assert origin.source_by_row == (11, 12, 13, None)
    moved = points.copy()
    moved[[0, 1]] = moved[[1, 0]]
    with pytest.raises(MeshError, match="moved"):
        bind_output_rows(core, moved, triangulation, {11, 12, 13}, expected)
    permuted_core = MeshCore(expected[[1, 0, 2, 3]], ((0, 1, 2), (0, 2, 3)))
    with pytest.raises(MeshError, match="reordered"):
        bind_output_rows(permuted_core, points, triangulation, {11, 12, 13}, expected)
    triangulation.protected_node_rows = ((11, 0), (12, 0), (13, 2))
    with pytest.raises(MeshError, match="conflicting"):
        bind_output_rows(core, points, triangulation, {11, 12, 13}, expected)
    triangulation.protected_node_rows = ((11, 0), (12, 1), (13, 2))
    object.__setattr__(origin, "source_by_row", (11, 12, 13, 99))
    with pytest.raises(MeshError, match="generated row"):
        origin.validate(core)
