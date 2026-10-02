import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from anymesher.native_cpp import compiled_native_boundary
from anymesher.triangulation import (
    _bowyer_watson, _complete_hull_seed, _finite_hull_seed,
    _prepare_pslg, constrained_planar_triangulation,
)


def twice_area(points, ring):
    # Independent high-precision shoelace check on the stored floating inputs.
    with localcontext() as ctx:
        ctx.prec = 80
        p = [[Decimal.from_float(float(x)) for x in row] for row in points]
        return sum(p[a][0]*p[b][1]-p[b][0]*p[a][1]
                   for a, b in zip(ring, np.roll(ring, -1)))


@pytest.mark.parametrize('backend', ['python', 'compiled'])
@pytest.mark.parametrize('reverse,offset', [(False, 0.), (True, 0.), (False, 10.)])
def test_thin_wall_boundary_is_complete_and_conserves_material(backend, reverse, offset):
    raw = json.loads(Path(__file__).with_name('thin_analytic_boundary.json').read_text())
    points = np.array(raw['points']) + offset
    outer = list(raw['outer'])
    if reverse:
        outer.reverse()
    chosen = 'python' if backend == 'python' else compiled_native_boundary()
    if chosen is None:
        pytest.skip('compiled triangulation unavailable')
    original = points.copy()
    result = constrained_planar_triangulation(points, outer, backend=chosen)
    np.testing.assert_array_equal(result.points, original)
    assert result.actual_backend == ('python' if backend == 'python' else 'anymesher-cpp17')
    assert result.fallback_reason is None
    assert len(result.triangles) == 10
    cells_area = sum(twice_area(result.points, cell) for cell in result.triangles)
    assert abs(cells_area - abs(twice_area(points, outer))) < Decimal('1e-65')
    incidence = {}
    for cell in result.triangles:
        assert twice_area(result.points, cell) > 0
        for a, b in zip(cell, np.roll(cell, -1)):
            edge = tuple(sorted((int(a), int(b))))
            incidence[edge] = incidence.get(edge, 0) + 1
    assert {edge for edge, count in incidence.items() if count == 1} == {
        tuple(sorted((a, b))) for a, b in zip(outer, outer[1:] + outer[:1])}
    assert max(incidence.values()) == 2


def test_incomplete_supertriangle_is_detected_before_recovery():
    raw = json.loads(Path(__file__).with_name('thin_analytic_boundary.json').read_text())
    points = np.array(raw['points'])
    assert not _complete_hull_seed(points, _bowyer_watson(points))
    assert _complete_hull_seed(points, _finite_hull_seed(points))


def test_finite_seed_preserves_collinear_hull_and_exact_diagonal_stations():
    points = np.array([(0., 0.), (1., 0.), (2., 0.), (2., 2.),
                       (0., 2.), (1., 1.), (.5, .5)])
    original = points.copy()
    cells = _finite_hull_seed(points)
    assert _complete_hull_seed(points, cells)
    np.testing.assert_array_equal(points, original)
    assert sum(twice_area(points, cell) for cell in cells) == Decimal(8)


@pytest.mark.parametrize('phase', ['convex hull', 'finite hull insertion', 'finite hull legality'])
def test_finite_seed_cancellation_is_preserved(phase):
    points = np.array([(0., 0.), (2., 0.), (2., 2.), (0., 2.), (1., 1.)])
    original = points.copy()
    class Cancelled(Exception):
        pass
    def cancel(where):
        if phase in where:
            raise Cancelled(where)
    with pytest.raises(Cancelled, match=phase):
        _finite_hull_seed(points, cancel)
    np.testing.assert_array_equal(points, original)


@pytest.mark.parametrize('backend', ['python', 'compiled'])
def test_boundary_subdivision_and_hole_are_filtered_by_topology(backend):
    points = np.array([(0., 0.), (4., 0.), (4., 4.), (0., 4.),
                       (1., 1.), (1., 3.), (3., 3.), (3., 1.), (2., 0.)])
    chosen = 'python' if backend == 'python' else compiled_native_boundary()
    if chosen is None:
        pytest.skip('compiled triangulation unavailable')
    result = constrained_planar_triangulation(points, (0, 1, 2, 3), holes=((4, 5, 6, 7),), backend=chosen)
    assert 8 in result.outer_loop
    assert sum(twice_area(result.points, cell) for cell in result.triangles) == Decimal(24)


def test_endpoint_adjacent_membership_does_not_duplicate_ring_corner():
    epsilon = 1e-6
    points = np.array([(0., 0.), (1., 0.), (1., 1.), (0., 1.),
                       (1.+.8*epsilon, .8*epsilon)])
    prepared = _prepare_pslg(points, (0, 1, 2, 3), (), (), epsilon)
    assert len(set(prepared.outer)) == len(prepared.outer)
    assert all(a != b for a, b in zip(prepared.outer, np.roll(prepared.outer, -1)))
    assert all(a != b for a, b in prepared.segments)


def test_domain_filter_cancellation_is_preserved():
    def cancel(where):
        if where == 'python triangulation domain boundary':
            raise RuntimeError('cancel domain filter')
    with pytest.raises(RuntimeError, match='cancel domain filter'):
        constrained_planar_triangulation(np.array([(0.,0.),(1.,0.),(1.,1.),(0.,1.)]),
                                         backend='python', cancellation_check=cancel)


def test_compiled_finite_seed_preserves_original_cancellation_error():
    boundary = compiled_native_boundary()
    if boundary is None:
        pytest.skip('compiled triangulation unavailable')
    raw = json.loads(Path(__file__).with_name('thin_analytic_boundary.json').read_text())
    points = np.array(raw['points'])
    original = points.copy()
    def cancel(where):
        if where == 'native triangulation finite hull seed':
            raise RuntimeError('cancel finite native hull')
    with pytest.raises(RuntimeError,match='cancel finite native hull'):
        constrained_planar_triangulation(points,raw['outer'],backend=boundary,cancellation_check=cancel)
    np.testing.assert_array_equal(points,original)
