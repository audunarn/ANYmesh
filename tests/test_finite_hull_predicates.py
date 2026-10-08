"""Exact sign regressions for near-collinear finite-hull flip decisions."""
from fractions import Fraction
from itertools import permutations

import pytest

from anymesher.triangulation import incircle


# Four original stations from a plate rotated by 0.37 radians. The previous
# floating-point filter accepted inconsistent signs and cycled between two
# finite-hull triangulations. Preserve the actual binary input coordinates.
QUAD = (
    (0.23308183640150862, 0.0904038579912405),
    (1.1654091820075432, 0.4520192899562025),
    (0.46616367280301724, 0.180807715982481),
    (0.6992455092045259, 0.2712115739737215),
)

# Binary-coordinate rectangle captured from the schema6 849-point candidate.
# Decimal arithmetic at fixed precision produced nonzero signs for this exact
# cocircular input, allowing two opposite diagonal flips indefinitely.
RECTANGLE = (
    (0.7127416614545834, 0.0),
    (0.6924154054183753, 0.020156360163291708),
    (0.6924154054183753, 0.0),
    (0.7127416614545834, 0.020156360163291708),
)


def exact_sign(a, b, c, d):
    (ax, ay), (bx, by), (cx, cy), (dx, dy) = (
        tuple(Fraction(value) for value in row) for row in (a, b, c, d)
    )
    orientation = (ax-cx)*(by-cy) - (ay-cy)*(bx-cx)
    ax, ay, bx, by, cx, cy = ax-dx, ay-dy, bx-dx, by-dy, cx-dx, cy-dy
    value = ((ax*ax+ay*ay)*(bx*cy-by*cx)
             - (bx*bx+by*by)*(ax*cy-ay*cx)
             + (cx*cx+cy*cy)*(ax*by-ay*bx))
    if orientation < 0:
        value = -value
    return (value > 0) - (value < 0)


@pytest.mark.parametrize('indices', tuple(permutations(range(4))))
def test_near_collinear_incircle_sign_matches_exact_binary_input(indices):
    points = tuple(QUAD[index] for index in indices)
    result = incircle(*points)
    assert int(result > 0) - int(result < 0) == exact_sign(*points)


@pytest.mark.parametrize('indices', tuple(permutations(range(4))))
def test_cocircular_incircle_is_exactly_zero_for_binary_rectangle(indices):
    points = tuple(RECTANGLE[index] for index in indices)
    assert exact_sign(*points) == 0
    assert incircle(*points) == 0.0


def test_cocircular_fallback_reads_only_the_two_planar_coordinates():
    assert incircle(*(point + ("unused",) for point in RECTANGLE)) == 0.0
