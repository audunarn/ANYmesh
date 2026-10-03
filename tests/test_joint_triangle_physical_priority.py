"""Owner-only root priority changes selection, not merit or physical admission."""
from dataclasses import fields
import numpy as np
import pytest
from anymesher._joint_triangle_repair import repair_joint_triangle_quality, _physical_priority_key


def grid(*, size=6, height=.25, defect=True):
    points = np.asarray([(x/(size-1), height*y/(size-1)) for y in range(size) for x in range(size)])
    if defect:
        points[-size-2] = (.99, height-.001)
    cells = np.asarray([row for y in range(size-1) for x in range(size-1)
        for row in ((size*y+x, size*y+x+1, size*y+x+size+1),
                    (size*y+x, size*y+x+size+1, size*y+x+size))])
    incidence = {}
    for row, cell in enumerate(cells):
        for a, b in zip(cell, np.roll(cell, -1)):
            incidence.setdefault(tuple(sorted((int(a), int(b)))), []).append(row)
    protected = np.asarray([edge for edge, rows in incidence.items() if len(rows) == 1])
    pairs = np.asarray([rows for rows in incidence.values() if len(rows) == 2])
    return points, cells, protected, pairs


def owner(rows):
    return np.column_stack((rows, np.zeros(len(rows))))


def reference_key(points, cells, xyz, pairs, angle=30., growth=1.5):
    # Deliberately independent slow reference, used only for the small fixtures.
    rows = xyz[cells]
    forward, backward = np.roll(rows, -1, axis=1)-rows, np.roll(rows, 1, axis=1)-rows
    angles = np.degrees(np.arctan2(np.linalg.norm(np.cross(forward, backward), axis=2),
                                   np.sum(forward*backward, axis=2)))
    lengths = np.linalg.norm(forward, axis=2).mean(axis=1)
    ratios = np.ones(len(cells))
    for a, b in pairs:
        value = max(lengths[a]/lengths[b], lengths[b]/lengths[a])
        ratios[a] = max(ratios[a], value)
        ratios[b] = max(ratios[b], value)
    def key(node):
        parents = [i for i, row in enumerate(cells) if node in row]
        deficit = max(0., (angle - min(float(np.min(angles[i])) for i in parents))/angle)
        excess = max(0., max(ratios[i] for i in parents)/growth-1.)
        return (-max(deficit, excess), -deficit, -excess, *points[node], node)
    return key, angles, lengths


def run(points, cells, protected, *, bad=None, priority=True, budget=0, **kwargs):
    return repair_joint_triangle_quality(points, cells, protected,
        range(len(cells)) if bad is None else bad,
        min_angle=30., max_growth=1.5, max_trials=budget,
        evaluate_coordinates=owner, physical_priority=priority, neighbourhood_rings=1, **kwargs)


def test_high_id_worst_defect_selected_and_geometry_invariant_under_relabelling_and_cell_order():
    points, cells, protected, pairs = grid()
    interior = sorted(set(range(len(points))) - set(map(int, protected.ravel())))
    assert len(interior) == 16
    expected, _, _ = reference_key(points, cells, owner(points), pairs)
    first = run(points, cells, protected)
    assert first.selected_nodes == first.root_nodes == tuple(sorted(interior, key=expected)[:12])
    assert first.selected_nodes[0] == 28 and not first.neighbour_nodes
    assert first.priority_mode == 'physical_severity' and first.trials == 0
    permutation = np.arange(len(points))
    permutation[interior] = np.asarray(interior)[::-1]
    inverse = np.argsort(permutation)
    relabelled = run(points[permutation], inverse[cells[::-1]], inverse[protected])
    assert points[list(first.selected_nodes)].tobytes() == points[permutation][list(relabelled.selected_nodes)].tobytes()
    default = run(points, cells, protected, priority=False)
    again = run(points[permutation], inverse[cells], inverse[protected], priority=False)
    assert 28 not in default.selected_nodes
    assert sorted(map(tuple, points[list(default.selected_nodes)])) != sorted(map(tuple, points[permutation][list(again.selected_nodes)]))


def test_highest_severity_pin_excluded_before_cap():
    points, cells, protected, pairs = grid()
    expected, _, _ = reference_key(points, cells, owner(points), pairs)
    eligible = set(range(len(points))) - set(map(int, protected.ravel())) - {28}
    result = run(points, cells, protected, pinned_nodes=(28,))
    assert result.selected_nodes == tuple(sorted(eligible, key=expected)[:12])
    assert 28 not in result.selected_nodes and len(result.selected_nodes) == 12
    assert result.points[28].tobytes() == points[28].tobytes()


def test_non_isometric_owner_ranks_physical_geometry_instead_of_chart():
    points, cells, protected, pairs = grid()
    def stretched(rows):
        return np.column_stack((.02 * rows[:, 0], rows[:, 1], np.zeros(len(rows))))
    eligible = set(range(len(points))) - set(map(int, protected.ravel()))
    physical_key, _, _ = reference_key(points, cells, stretched(points), pairs)
    chart_key, _, _ = reference_key(points, cells, owner(points), pairs)
    expected = tuple(sorted(eligible, key=physical_key)[:12])
    assert expected != tuple(sorted(eligible, key=chart_key)[:12])
    result = repair_joint_triangle_quality(points, cells, protected, range(len(cells)),
        min_angle=30., max_growth=1.5, max_trials=0, evaluate_coordinates=stretched,
        physical_priority=True)
    assert result.selected_nodes == expected
    assert result.points.tobytes() == points.tobytes() and result.trials == 0


def test_exact_severity_ties_use_chart_coordinates_before_ids():
    points, cells, protected, pairs = grid(size=5, height=1., defect=False)
    eligible = set(range(len(points))) - set(map(int, protected.ravel()))
    result = run(points, cells, protected)
    assert result.selected_nodes == tuple(sorted(eligible, key=lambda node: (*points[node], node)))
    key, angles, lengths = reference_key(points, cells, owner(points), pairs)
    actual = _physical_priority_key(points, cells, angles, lengths, pairs, 30., 1.5)
    assert all(actual(node) == key(node) for node in eligible)


def test_coincident_chart_coordinate_tie_uses_id_last_without_welding():
    points = np.asarray(((0., 0.), (.5, .5), (.5, .5), (1., 0.)))
    cells = np.asarray(((0, 1, 3), (0, 2, 3)))
    key = _physical_priority_key(points, cells, np.asarray(((45., 90., 45.),)*2),
                                np.ones(2), np.asarray(((0, 1),)), 30., 1.5)
    assert key(1)[:-1] == key(2)[:-1] and sorted((2, 1), key=key) == [1, 2]


def test_direct_roots_precede_severe_neighbours_that_only_fill_unused_slots():
    points, cells, protected, pairs = grid()
    bad = [24]
    fixed = set(map(int, protected.ravel()))
    roots = set(map(int, cells[bad].ravel())) - fixed
    neighbours = {int(n) for cell in cells if any(int(n) in roots for n in cell) for n in cell} - roots - fixed
    key, _, _ = reference_key(points, cells, owner(points), pairs)
    result = run(points, cells, protected, bad=bad)
    assert result.root_nodes == tuple(sorted(roots, key=key))
    assert result.neighbour_nodes == tuple(sorted(neighbours, key=key)[:12-len(roots)])
    assert result.selected_nodes == result.root_nodes + result.neighbour_nodes


@pytest.mark.parametrize('priority', (False, True))
def test_no_extra_owner_calls_and_zero_trials_still_record_selection(priority):
    points, cells, protected, _ = grid()
    calls = []
    def evaluate(rows):
        calls.append(rows.copy())
        return owner(rows)
    result = repair_joint_triangle_quality(points, cells, protected, range(len(cells)),
        min_angle=30., max_growth=1.5, max_trials=0, evaluate_coordinates=evaluate,
        physical_priority=priority)
    assert len(calls) == 2 and all(row.tobytes() == points.tobytes() for row in calls)
    assert result.trials == 0 and result.budget_exhausted and len(result.selected_nodes) == 12
    assert result.points.tobytes() == points.tobytes()


@pytest.mark.parametrize('budget', (1, 8))
@pytest.mark.parametrize('priority', (False, True))
def test_scalar_and_batch_preserve_selected_order_merit_and_budget(budget, priority):
    points, cells, protected, _ = grid()
    one = run(points, cells, protected, priority=priority, budget=budget)
    eight = run(points, cells, protected, priority=priority, budget=budget, coordinate_batch_size=8)
    assert one.points.tobytes() == eight.points.tobytes()
    for field in fields(one):
        if field.name != 'points':
            assert getattr(one, field.name) == getattr(eight, field.name)
    assert one.trials == budget and one.budget_exhausted


@pytest.mark.parametrize('value', (1, 'physical', None, np.bool_(True)))
def test_priority_requires_bool(value):
    points, cells, protected, _ = grid()
    with pytest.raises(ValueError, match='invalid joint'):
        run(points, cells, protected, priority=value)


def test_priority_requires_owner_evaluator():
    points, cells, protected, _ = grid()
    with pytest.raises(ValueError, match='invalid joint'):
        repair_joint_triangle_quality(points, cells, protected, range(len(cells)),
            min_angle=30., max_growth=1.5, physical_priority=True)


@pytest.mark.parametrize('cancel_at', (1, 2))
def test_priority_preparation_cancellation_propagates_without_trial_or_mutation(cancel_at):
    points, cells, protected, _ = grid()
    before = points.tobytes(), cells.tobytes()
    error = RuntimeError('cancel physical root priority')
    checks = []
    def cancel():
        checks.append(None)
        if len(checks) == cancel_at:
            raise error
    with pytest.raises(RuntimeError) as caught:
        run(points, cells, protected, budget=1, cancellation_check=cancel)
    assert caught.value is error and len(checks) == cancel_at
    assert (points.tobytes(), cells.tobytes()) == before


def test_default_zero_trial_callback_sequence_is_unchanged():
    points, cells, protected, _ = grid()
    checks = []
    result = run(points, cells, protected, priority=False, cancellation_check=lambda: checks.append(None))
    assert not checks and result.trials == 0


def test_stale_owner_error_is_not_converted_to_priority_or_local_refusal():
    from anygeometry import GeometryError
    points, cells, protected, _ = grid()
    error = GeometryError('stale initial owner state')
    def stale(rows):
        raise error
    with pytest.raises(GeometryError) as caught:
        repair_joint_triangle_quality(points, cells, protected, range(len(cells)),
            min_angle=30., max_growth=1.5, max_trials=1, physical_priority=True, evaluate_coordinates=stale)
    assert caught.value is error


@pytest.mark.parametrize('invalid', ('angle', 'length', 'zero_ratio'))
def test_nonfinite_sortable_priority_is_rejected(invalid):
    points, cells, _, pairs = grid()
    _, angles, lengths = reference_key(points, cells, owner(points), pairs)
    if invalid == 'angle':
        angles[0, 0] = np.nan
    elif invalid == 'length':
        lengths[0] = np.inf
    else:
        lengths[pairs[0, 0]] = 0.
    with pytest.raises(ValueError, match='nonfinite physical'):
        _physical_priority_key(points, cells, angles, lengths, pairs, 30., 1.5)


def test_degenerate_finite_owner_geometry_cannot_publish_priority_result():
    points, cells, protected, _ = grid()
    before = points.tobytes()
    with pytest.raises(ValueError, match='nonfinite physical'):
        repair_joint_triangle_quality(points, cells, protected, range(len(cells)),
            min_angle=30., max_growth=1.5, max_trials=1, physical_priority=True,
            evaluate_coordinates=lambda rows: np.zeros((len(rows), 3)))
    assert points.tobytes() == before
