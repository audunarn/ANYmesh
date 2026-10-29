"""Acceptance tests for the opt-in parametric tensor route (written before the code)."""
import numpy as np
import pytest

from anymesher._parametric_tensor import ParametricTensorUnsupported, parametric_tensor_core


def _square(nx=7, ny=5, skew=0.3):
    a = np.linspace(0, 1, nx + 1) ** 1.3
    b = np.linspace(0, 1, ny + 1) ** 0.8
    chart = np.array([[2.0, 0.0], [skew, 1.5]])
    inverse = np.linalg.inv(chart)
    uv = ([(x, 0.) for x in a[:-1]] + [(1., y) for y in b[:-1]]
          + [(x, 1.) for x in a[::-1][:-1]] + [(0., y) for y in b[::-1][:-1]])
    outer = np.asarray(uv) @ chart
    # chart rows use the row-vector convention uv = chart @ inverse
    return outer, inverse, {row: 100 + row for row in range(len(outer))}


def test_core_keeps_boundary_rows_and_covers_the_box():
    outer, inverse, rows = _square()
    core, record = parametric_tensor_core(outer, (), (), np.empty((0, 2)), inverse, rows)
    assert np.array_equal(core.node_coordinates[:len(outer), :2], outer)
    assert int(core.quad_active.sum()) == 7 * 5 and not core.triangle_active.any()
    q = core.quad_connectivity
    xy = core.node_coordinates[:, :2][q]
    area = 0.5 * (np.cross(xy[:, 2] - xy[:, 0], xy[:, 3] - xy[:, 1]))
    assert np.all(area > 0.0)
    assert np.isclose(area.sum(), 2.0 * 1.5)
    assert record.validate(core) == {row: node for row, node in rows.items()}


@pytest.mark.parametrize('mutate', ['holes', 'constraints', 'pinned', 'unequal', 'skewed_top', 'trimmed', 'pole'])
def test_refusals_are_typed(mutate):
    outer, inverse, rows = _square()
    holes, constraints, pinned = (), (), np.empty((0, 2))
    if mutate == 'holes':
        holes = (np.zeros((4, 2)),)
    elif mutate == 'constraints':
        constraints = (np.zeros((2, 2)),)
    elif mutate == 'pinned':
        pinned = np.zeros((1, 2))
    elif mutate == 'unequal':
        outer, inverse, rows = _square(7, 5)
        outer = np.delete(outer, 3 + 7 + 5 + 2, axis=0)  # drop one top station
        rows = {row: 100 + row for row in range(len(outer))}
    elif mutate == 'skewed_top':
        outer = outer.copy()
        outer[7 + 5 + 2, 1] += 1e-3
    elif mutate == 'trimmed':
        outer = outer.copy()
        outer[2] = outer[2] * 0.5 + outer[1] * 0.5 + np.array([0., 0.05])
    elif mutate == 'pole':
        outer = outer.copy()
        outer[7:] = outer[7]  # collapse everything but the bottom to one point
    with pytest.raises(ParametricTensorUnsupported):
        parametric_tensor_core(outer, holes, constraints, pinned, inverse, rows)


def test_unequal_station_fractions_are_refused():
    outer, inverse, rows = _square()
    outer = outer.copy()
    # move one top station along the top edge: counts equal, fractions differ
    chart = np.array([[2.0, 0.0], [0.3, 1.5]])
    outer[7 + 5 + 3] += np.array([0.01, 0.0]) @ chart
    with pytest.raises(ParametricTensorUnsupported):
        parametric_tensor_core(outer, (), (), np.empty((0, 2)), inverse, rows)


def test_tensor_topology_identities_and_no_inserted_nodes():
    """Disc topology and exact row accounting: nodes = boundary + interior, no row is invented or dropped."""
    nx, ny = 7, 5
    outer, inverse, rows = _square(nx, ny)
    core, record = parametric_tensor_core(outer, (), (), np.empty((0, 2)), inverse, rows)
    q = core.quad_connectivity[core.quad_active]
    assert core.num_nodes == (nx + 1) * (ny + 1) == len(outer) + (nx - 1) * (ny - 1)
    assert len(q) == nx * ny
    edges = {tuple(sorted((int(a), int(b)))) for cell in q for a, b in zip(cell, np.roll(cell, -1))}
    assert len(edges) == (nx + 1) * ny + nx * (ny + 1)
    assert core.num_nodes - len(edges) + len(q) == 1  # Euler characteristic of a disc
    assert sorted(set(map(int, q.ravel()))) == list(range(core.num_nodes))  # every row is used
    protected = [node for node in record.source_by_row if node is not None]
    assert sorted(protected) == sorted(rows.values()) and len(protected) == len(outer)
    assert all(node is None for node in record.source_by_row[len(outer):])  # interior rows carry no source identity


def test_opt_in_plumbing_is_off_by_default_and_refusal_is_recorded():
    import inspect
    import anymesher.hybrid as hybrid
    # S2/S4: the prototype boolean `_parametric_tensor` is retired; the replacement is the tri-state `_nurbs_tensor`.
    parameter = inspect.signature(hybrid._mesh_native_face_once).parameters['_nurbs_tensor']
    assert parameter.default is None
    assert '_parametric_tensor' not in inspect.signature(hybrid._mesh_native_face_once).parameters
    assert inspect.signature(hybrid.generate_hybrid_mesh_result.__wrapped__ if hasattr(
        hybrid.generate_hybrid_mesh_result, '__wrapped__') else hybrid.generate_hybrid_mesh_result
        ).parameters['_nurbs_tensor'].default is None
    diagnostics = {}
    outer, inverse, rows = _square()
    result = hybrid._try_parametric_tensor(None, 1, outer, (), (), np.empty((0, 2)), inverse, rows,
                                           None, None, None, 1., diagnostics)
    assert result is None and 'owner-bound NURBS material region' in diagnostics['parametric_tensor_refusal']
