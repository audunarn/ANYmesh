"""The edge-seeding constraint solver.

Under mapped meshing this has no fallback, so it is tested on its own rather
than only through the mesher.
"""

from __future__ import annotations

import pytest

from anymesher.geometry import GeometryModel
from anymesher.seeding import SeedingConflict, solve_seeding


def rectangle(model: GeometryModel, width: float, height: float):
    points = model.add_points(
        [(0, 0, 0), (width, 0, 0), (width, height, 0), (0, height, 0)]
    )
    edges = model.add_polyline(points, close=True)
    face = model.add_face(edges)
    return edges, face


def test_opposite_sides_agree_on_a_plain_rectangle():
    model = GeometryModel()
    edges, _ = rectangle(model, 2.0, 1.0)
    seeding = solve_seeding(model, target_size=0.25)

    bottom, right, top, left = edges
    assert seeding[bottom] == seeding[top] == 8
    assert seeding[right] == seeding[left] == 4


def test_unequal_opposite_sides_are_refined_to_the_finer():
    """A trapezoid's opposite sides differ in length but share a count."""

    model = GeometryModel()
    points = model.add_points([(0, 0, 0), (4, 0, 0), (3, 1, 0), (0, 1, 0)])
    edges = model.add_polyline(points, close=True)
    model.add_face(edges)

    seeding = solve_seeding(model, target_size=1.0)
    bottom, _, top, _ = edges
    # Errs finer: the 4 m side wants 4, the 3 m side wants 3, both get 4.
    assert seeding[bottom] == seeding[top] == 4


def test_constraints_propagate_across_shared_edges():
    """Seeding one face forces its neighbours through the shared edge."""

    model = GeometryModel()
    points = model.add_points(
        [(0, 0, 0), (1, 0, 0), (5, 0, 0), (5, 1, 0), (1, 1, 0), (0, 1, 0)]
    )
    left_bottom = model.add_line(points[0], points[1])
    right_bottom = model.add_line(points[1], points[2])
    right = model.add_line(points[2], points[3])
    right_top = model.add_line(points[3], points[4])
    left_top = model.add_line(points[4], points[5])
    left = model.add_line(points[5], points[0])
    shared = model.add_line(points[1], points[4])

    model.add_face([left_bottom, shared, left_top, left])
    model.add_face([right_bottom, right, right_top, shared])

    seeding = solve_seeding(model, target_size=0.5)
    # left, shared and right are all forced equal through the two faces.
    assert seeding[left] == seeding[shared] == seeding[right]
    assert seeding[left_bottom] == seeding[left_top]
    assert seeding[right_bottom] == seeding[right_top]


def test_multi_edge_chain_side_matches_its_opposite():
    model = GeometryModel()
    points = model.add_points(
        [(0, 0, 0), (0.4, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    )
    edges = model.add_polyline(points, close=True)
    face = model.add_face(edges)

    seeding = solve_seeding(model, target_size=0.2)
    sides = model.faces[face].sides()
    assert seeding.side_divisions(sides[0]) == seeding.side_divisions(sides[2])
    assert seeding.side_divisions(sides[1]) == seeding.side_divisions(sides[3])


def test_explicit_override_is_honoured():
    model = GeometryModel()
    edges, _ = rectangle(model, 2.0, 1.0)
    bottom, _, top, _ = edges

    seeding = solve_seeding(model, target_size=0.5, overrides={bottom: 7})
    assert seeding[bottom] == seeding[top] == 7


def test_planner_minimum_is_nonlocking_and_propagates_across_a_seed_class():
    model = GeometryModel()
    edges, _ = rectangle(model, 2.0, 1.0)
    bottom, _, top, _ = edges

    bounded = solve_seeding(
        model,
        target_size=0.5,
        minimums={bottom: 6},
    )
    refined = solve_seeding(
        model,
        target_size=0.2,
        minimums={bottom: 6},
    )

    assert bounded[bottom] == bounded[top] == 6
    assert refined[bottom] == refined[top] == 10


def test_exact_override_below_a_planner_minimum_fails_closed():
    model = GeometryModel()
    edges, _ = rectangle(model, 2.0, 1.0)
    bottom, _, _, _ = edges

    with pytest.raises(SeedingConflict, match="below the non-locking planner minimum"):
        solve_seeding(
            model,
            target_size=0.5,
            overrides={bottom: 4},
            minimums={bottom: 5},
        )


def test_conflicting_overrides_are_reported_not_resolved():
    model = GeometryModel()
    edges, _ = rectangle(model, 2.0, 1.0)
    bottom, _, top, _ = edges

    with pytest.raises(SeedingConflict, match="conflicting division overrides"):
        solve_seeding(model, target_size=0.5, overrides={bottom: 3, top: 5})


def test_over_pinned_chain_is_reported():
    """A pinned chain that cannot reach its opposite side fails closed."""

    model = GeometryModel()
    points = model.add_points(
        [(0, 0, 0), (0.4, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    )
    edges = model.add_polyline(points, close=True)
    model.add_face(edges)

    with pytest.raises(SeedingConflict, match="pinned"):
        solve_seeding(
            model,
            target_size=0.2,
            overrides={edges[0]: 1, edges[1]: 1, edges[3]: 5},
        )


def test_free_edges_are_seeded_without_constraints():
    """A line carrying a beam but bounding no plate is seeded on its own."""

    model = GeometryModel()
    start = model.add_point(0, 0, 0)
    end = model.add_point(3, 0, 0)
    edge = model.add_line(start, end)

    seeding = solve_seeding(model, target_size=0.5)
    assert seeding[edge] == 6


def test_every_edge_gets_at_least_one_division():
    model = GeometryModel()
    edges, _ = rectangle(model, 0.01, 0.01)
    seeding = solve_seeding(model, target_size=1.0)
    assert all(seeding[edge] >= 1 for edge in edges)


def test_target_size_must_be_positive():
    model = GeometryModel()
    rectangle(model, 1.0, 1.0)
    with pytest.raises(ValueError, match="positive"):
        solve_seeding(model, target_size=0.0)
