"""Locating a point in the plating, and the coupling record that results."""

from __future__ import annotations

import numpy as np
import pytest

from anymesher import (
    Coupling,
    MeshError,
    build_structured_shell_grid,
    locate_shell_element_at_xy,
    shape_functions_4node,
    shape_functions_8node,
)


def _unit_grid(nx: int = 2, ny: int = 2):
    """A structured nx-by-ny grid of unit cells, as coordinate and element maps."""

    nodes = {}
    node_id = 1
    index = {}
    for i in range(nx + 1):
        for j in range(ny + 1):
            index[(i, j)] = node_id
            nodes[node_id] = (float(i), float(j), 0.0)
            node_id += 1
    elements = {}
    element_id = 1
    for i in range(nx):
        for j in range(ny):
            elements[element_id] = (
                [index[(i, j)], index[(i + 1, j)], index[(i + 1, j + 1)], index[(i, j + 1)]],
                0.01,
            )
            element_id += 1
    return nodes, elements


@pytest.mark.parametrize("shape_functions,count", [(shape_functions_4node, 4), (shape_functions_8node, 8)])
def test_shape_functions_are_a_partition_of_unity(shape_functions, count) -> None:
    for xi in (-1.0, -0.5, 0.0, 0.3, 1.0):
        for eta in (-1.0, 0.0, 0.7, 1.0):
            weights = shape_functions(xi, eta)
            assert weights.shape == (count,)
            # Summing to one is what makes an interpolated coupling reproduce a
            # rigid translation instead of scaling it.
            assert float(np.sum(weights)) == pytest.approx(1.0)


def test_shape_functions_are_one_at_their_own_node() -> None:
    corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    for index, (xi, eta) in enumerate(corners):
        assert shape_functions_4node(xi, eta)[index] == pytest.approx(1.0)

    mid_sides = [(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]
    for index, (xi, eta) in enumerate(corners + mid_sides):
        weights = shape_functions_8node(xi, eta)
        assert weights[index] == pytest.approx(1.0)
        assert np.allclose(np.delete(weights, index), 0.0)


def test_a_structured_grid_is_indexed_and_an_irregular_one_is_not() -> None:
    nodes, elements = _unit_grid(3, 2)
    grid = build_structured_shell_grid(nodes, elements, 1.0e-6)

    assert grid is not None
    assert len(grid.cells) == 6
    assert grid.x_edges.tolist() == [0.0, 1.0, 2.0, 3.0]

    # A rotated cell is not axis aligned, so the fast index declines rather than
    # indexing it wrongly.  Declining is not an error: the caller falls back to a
    # sequential search and still gets the right answer.
    skewed_nodes = dict(nodes)
    skewed_nodes[1] = (-0.3, -0.3, 0.0)
    assert build_structured_shell_grid(skewed_nodes, elements, 1.0e-6) is None
    assert build_structured_shell_grid({}, {}, 1.0e-6) is None


def test_locating_a_point_returns_weights_that_reproduce_it() -> None:
    nodes, elements = _unit_grid(2, 2)
    grid = build_structured_shell_grid(nodes, elements, 1.0e-6)

    for x, y in ((0.25, 0.25), (1.5, 0.5), (0.5, 1.75), (1.0, 1.0)):
        located = locate_shell_element_at_xy(x, y, nodes, elements, 1.0e-6, grid)
        assert located is not None
        node_ids, weights, point = located
        assert len(node_ids) == len(weights) == 4
        assert float(np.sum(weights)) == pytest.approx(1.0)
        assert point[0] == pytest.approx(x)
        assert point[1] == pytest.approx(y)


def test_the_fast_index_and_the_sequential_fallback_agree() -> None:
    nodes, elements = _unit_grid(3, 3)
    grid = build_structured_shell_grid(nodes, elements, 1.0e-6)

    for x, y in ((0.1, 2.9), (2.5, 0.5), (1.5, 1.5)):
        with_index = locate_shell_element_at_xy(x, y, nodes, elements, 1.0e-6, grid)
        without = locate_shell_element_at_xy(x, y, nodes, elements, 1.0e-6, None)
        assert with_index is not None and without is not None
        assert with_index[0] == without[0]
        assert np.allclose(with_index[1], without[1])


def test_a_point_outside_the_plating_is_not_located() -> None:
    nodes, elements = _unit_grid(2, 2)

    assert locate_shell_element_at_xy(5.0, 0.5, nodes, elements, 1.0e-6) is None
    assert locate_shell_element_at_xy(0.5, -3.0, nodes, elements, 1.0e-6) is None


def test_eight_node_cells_give_eight_weights() -> None:
    nodes = {
        1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0), 3: (1.0, 1.0, 0.0), 4: (0.0, 1.0, 0.0),
        5: (0.5, 0.0, 0.0), 6: (1.0, 0.5, 0.0), 7: (0.5, 1.0, 0.0), 8: (0.0, 0.5, 0.0),
    }
    elements = {1: ([1, 2, 3, 4, 5, 6, 7, 8], 0.01)}

    located = locate_shell_element_at_xy(0.3, 0.7, nodes, elements, 1.0e-6)
    assert located is not None
    node_ids, weights, point = located
    assert len(node_ids) == 8 and len(weights) == 8
    assert point[0] == pytest.approx(0.3)
    assert point[1] == pytest.approx(0.7)


def test_a_coupling_record_validates_its_own_weights() -> None:
    ok = Coupling(beam_node=7, plate_nodes=(1, 2), weights=(0.25, 0.75), eccentricity=(0.0, 0.0, 0.3))
    assert not ok.is_node_to_node

    with pytest.raises(MeshError, match="one weight per plate node"):
        Coupling(beam_node=7, plate_nodes=(1, 2), weights=(1.0,), eccentricity=(0.0, 0.0, 0.0))
    with pytest.raises(MeshError, match="at least one plate node"):
        Coupling(beam_node=7, plate_nodes=(), weights=(), eccentricity=(0.0, 0.0, 0.0))
    with pytest.raises(MeshError, match="must sum to 1.0"):
        Coupling(beam_node=7, plate_nodes=(1, 2), weights=(0.5, 0.7), eccentricity=(0.0, 0.0, 0.0))
    with pytest.raises(MeshError, match="3-vector"):
        Coupling(beam_node=7, plate_nodes=(1,), weights=(1.0,), eccentricity=(0.0, 0.0))


def test_the_node_to_node_case_exposes_a_single_master() -> None:
    coupling = Coupling.node_to_node(7, 3, np.array([0.0, 0.0, 0.25]))

    assert coupling.is_node_to_node
    assert coupling.plate_node == 3
    assert coupling.weights == (1.0,)
    assert coupling.eccentricity == (0.0, 0.0, 0.25)

    interpolated = Coupling(
        beam_node=7, plate_nodes=(1, 2), weights=(0.5, 0.5), eccentricity=(0.0, 0.0, 0.1)
    )
    with pytest.raises(MeshError, match="no single master"):
        interpolated.plate_node
